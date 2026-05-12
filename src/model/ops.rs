use crate::{Error, Result, Tensor};

use super::RMS_NORM_EPS;
use super::weights::{CgmlpWeights, GluFfnWeights};

const BLOCKED_MASK_VALUE: f32 = -10_000.0;

pub fn glu_ffn<T: Tensor>(x: &T, weights: &GluFfnWeights<T>) -> Result<T> {
    let hidden = x
        .linear(&weights.ln1.weight, Some(&weights.ln1.bias))
        .map_err(|err| Error::message(format!("glu_ffn ln1 failed: {err}")))?;
    let (x1, x2) = split_last_dim_two(&hidden)?;
    let gated = x1
        .gelu()
        .and_then(|lhs| lhs.mul(&x2))
        .map_err(|err| Error::message(format!("glu_ffn gate failed: {err}")))?;
    gated
        .linear(&weights.ln2.weight, Some(&weights.ln2.bias))
        .map_err(|err| Error::message(format!("glu_ffn ln2 failed: {err}")))
}

pub fn cgmlp<T: Tensor>(x: &T, weights: &CgmlpWeights<T>) -> Result<T> {
    let hidden = x
        .linear(&weights.pw1.weight, Some(&weights.pw1.bias))
        .and_then(Tensor::gelu)
        .map_err(|err| Error::message(format!("cgmlp pw1 failed: {err}")))?;
    let (x1, x2) = split_last_dim_two(&hidden)?;
    let x2 = x2
        .rms_norm(&weights.norm, RMS_NORM_EPS)
        .and_then(|tensor| {
            tensor.conv1d_dw(
                &weights.dw.weight,
                weights.dw.bias.as_ref(),
                1,
                (weights.dw.kernel_size.saturating_sub(1)) / 2,
            )
        })
        .and_then(Tensor::gelu)
        .map_err(|err| Error::message(format!("cgmlp depthwise branch failed: {err}")))?;
    let merged = x1
        .mul(&x2)
        .map_err(|err| Error::message(format!("cgmlp gate failed: {err}")))?;
    merged
        .linear(&weights.pw2.weight, Some(&weights.pw2.bias))
        .map_err(|err| Error::message(format!("cgmlp pw2 failed: {err}")))
}

pub fn build_joint_attn_mask(regions: &[i32], n_regions: usize) -> Vec<f32> {
    let seq_len = regions.len();
    let total = n_regions + seq_len;
    let mut mask = vec![BLOCKED_MASK_VALUE; total * total];

    let region = |index: usize| -> i32 {
        if index < n_regions {
            i32::try_from(index + 1).unwrap_or(i32::MAX)
        } else {
            regions[index - n_regions]
        }
    };
    let is_pool = |index: usize| index < n_regions;
    let valid = |index: usize| -> bool {
        if index < n_regions {
            true
        } else {
            regions[index - n_regions] != 0
        }
    };

    for query in 0..total {
        for key in 0..total {
            let allowed = if valid(query) && valid(key) {
                let same_stream = is_pool(query) == is_pool(key);
                let query_region = region(query);
                let key_region = region(key);
                let same_region =
                    query_region != 0 && key_region != 0 && query_region == key_region;
                same_stream || same_region
            } else {
                false
            };
            mask[query * total + key] = if allowed { 0.0 } else { BLOCKED_MASK_VALUE };
        }
    }

    mask
}

pub(crate) fn split_last_dim_two<T: Tensor>(tensor: &T) -> Result<(T, T)> {
    split_last_dim(tensor, 2).and_then(|parts| {
        let mut iter = parts.into_iter();
        let first = iter
            .next()
            .ok_or_else(|| Error::message("split_last_dim_two produced no first half"))?;
        let second = iter
            .next()
            .ok_or_else(|| Error::message("split_last_dim_two produced no second half"))?;
        Ok((first, second))
    })
}

pub(crate) fn split_last_dim_three<T: Tensor>(tensor: &T) -> Result<(T, T, T)> {
    split_last_dim(tensor, 3).and_then(|parts| {
        let mut iter = parts.into_iter();
        let first = iter
            .next()
            .ok_or_else(|| Error::message("split_last_dim_three produced no first chunk"))?;
        let second = iter
            .next()
            .ok_or_else(|| Error::message("split_last_dim_three produced no second chunk"))?;
        let third = iter
            .next()
            .ok_or_else(|| Error::message("split_last_dim_three produced no third chunk"))?;
        Ok((first, second, third))
    })
}

fn split_last_dim<T: Tensor>(tensor: &T, parts: usize) -> Result<Vec<T>> {
    if parts == 0 {
        return Err(Error::message("split_last_dim requires at least one part"));
    }

    let axis = tensor
        .shape()
        .len()
        .checked_sub(1)
        .ok_or_else(|| Error::message("split_last_dim requires a tensor with rank >= 1"))?;
    let dim = tensor.shape()[axis];
    if dim % parts != 0 {
        return Err(Error::message(format!(
            "cannot split tensor shape {:?} into {parts} equal chunks along the last axis",
            tensor.shape()
        )));
    }

    let chunk = dim / parts;
    let mut out = Vec::with_capacity(parts);
    for index in 0..parts {
        let start = index * chunk;
        let end = start + chunk;
        out.push(
            tensor
                .clone()
                .slice(axis, start, end)
                .map_err(|err| Error::message(format!("split_last_dim slice failed: {err}")))?,
        );
    }
    Ok(out)
}
