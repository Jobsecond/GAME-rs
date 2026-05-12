use crate::{Error, Result, Tensor};

use super::RMS_NORM_EPS;
use super::ops::{cgmlp, glu_ffn, split_last_dim_three, split_last_dim_two};
use super::weights::{
    AttentionWeights, EbfBlockWeights, JebfBlockWeights, JointAttentionWeights, PacWeights,
    PjacWeights, ResidualGluWeights,
};

// Chunk attention score rows so temporary score tensors stay bounded on both
// CPU and GPU backends.
const MAX_ATTENTION_SCORE_ELEMENTS: usize = 4_000_000;

#[derive(Clone, Debug)]
pub struct JointAttentionOutput<T> {
    pub pool: T,
    pub x: T,
}

pub fn attention<T: Tensor>(
    x: &T,
    weights: &AttentionWeights<T>,
    positions: &[i32],
    num_heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<T> {
    validate_head_config(num_heads, head_dim)?;

    let q = x
        .linear(&weights.q.weight, Some(&weights.q.bias))
        .map_err(|err| Error::message(format!("attention q projection failed: {err}")))?;
    let kv = x
        .linear(&weights.kv.weight, Some(&weights.kv.bias))
        .map_err(|err| Error::message(format!("attention kv projection failed: {err}")))?;
    let (k, v) = split_last_dim_two(&kv)?;

    let q = reshape_for_heads(q, num_heads, head_dim)?
        .rope(positions, head_dim, num_heads, head_dim, theta)
        .map_err(|err| Error::message(format!("attention q rope failed: {err}")))?;
    let k = reshape_for_heads(k, num_heads, head_dim)?
        .rope(positions, head_dim, num_heads, head_dim, theta)
        .map_err(|err| Error::message(format!("attention k rope failed: {err}")))?;
    let v = reshape_for_heads(v, num_heads, head_dim)?;

    let attended = scaled_dot_product_attention(&q, &k, &v, None, head_dim)?;
    merge_heads(attended)?
        .linear(&weights.out.weight, Some(&weights.out.bias))
        .map_err(|err| Error::message(format!("attention output projection failed: {err}")))
}

pub fn joint_attention<T: Tensor>(
    pool: &T,
    x: &T,
    weights: &JointAttentionWeights<T>,
    global_positions: &[i32],
    region_ids: &[i32],
    mask: &T,
    num_heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<JointAttentionOutput<T>> {
    validate_head_config(num_heads, head_dim)?;
    validate_sequence_2d("joint_attention pool", pool)?;
    validate_sequence_2d("joint_attention x", x)?;

    let pool_len = pool.shape()[0];
    let x_len = x.shape()[0];
    let total_len = pool_len + x_len;
    if global_positions.len() != total_len {
        return Err(Error::message(format!(
            "joint_attention expected {} global positions, got {}",
            total_len,
            global_positions.len()
        )));
    }
    if region_ids.len() != total_len {
        return Err(Error::message(format!(
            "joint_attention expected {} region ids, got {}",
            total_len,
            region_ids.len()
        )));
    }

    let pool_norm = pool
        .clone()
        .rms_norm(&weights.pool.norm, RMS_NORM_EPS)
        .map_err(|err| Error::message(format!("joint_attention pool norm failed: {err}")))?;
    let x_norm = x
        .clone()
        .rms_norm(&weights.x.norm, RMS_NORM_EPS)
        .map_err(|err| Error::message(format!("joint_attention x norm failed: {err}")))?;

    let pool_qkv = pool_norm
        .linear(&weights.pool.qkv.weight, Some(&weights.pool.qkv.bias))
        .map_err(|err| Error::message(format!("joint_attention pool qkv failed: {err}")))?;
    let x_qkv = x_norm
        .linear(&weights.x.qkv.weight, Some(&weights.x.qkv.bias))
        .map_err(|err| Error::message(format!("joint_attention x qkv failed: {err}")))?;

    let (pool_q, pool_k, pool_v) = split_last_dim_three(&pool_qkv)?;
    let (x_q, x_k, x_v) = split_last_dim_three(&x_qkv)?;

    let pool_q = apply_optional_qk_norm(
        reshape_for_heads(pool_q, num_heads, head_dim)?,
        weights.pool.qk_norm.as_ref().map(|norm| &norm.q),
        "joint_attention pool q_norm",
    )?;
    let pool_k = apply_optional_qk_norm(
        reshape_for_heads(pool_k, num_heads, head_dim)?,
        weights.pool.qk_norm.as_ref().map(|norm| &norm.k),
        "joint_attention pool k_norm",
    )?;
    let pool_v = reshape_for_heads(pool_v, num_heads, head_dim)?;

    let x_q = apply_optional_qk_norm(
        reshape_for_heads(x_q, num_heads, head_dim)?,
        weights.x.qk_norm.as_ref().map(|norm| &norm.q),
        "joint_attention x q_norm",
    )?;
    let x_k = apply_optional_qk_norm(
        reshape_for_heads(x_k, num_heads, head_dim)?,
        weights.x.qk_norm.as_ref().map(|norm| &norm.k),
        "joint_attention x k_norm",
    )?;
    let x_v = reshape_for_heads(x_v, num_heads, head_dim)?;

    let q = T::concat(&[&pool_q, &x_q], 1)
        .map_err(|err| Error::message(format!("joint_attention q concat failed: {err}")))?
        .region_rope(
            global_positions,
            region_ids,
            head_dim,
            num_heads,
            head_dim,
            theta,
        )
        .map_err(|err| Error::message(format!("joint_attention q mixed rope failed: {err}")))?;
    let k = T::concat(&[&pool_k, &x_k], 1)
        .map_err(|err| Error::message(format!("joint_attention k concat failed: {err}")))?
        .region_rope(
            global_positions,
            region_ids,
            head_dim,
            num_heads,
            head_dim,
            theta,
        )
        .map_err(|err| Error::message(format!("joint_attention k mixed rope failed: {err}")))?;
    let v = T::concat(&[&pool_v, &x_v], 1)
        .map_err(|err| Error::message(format!("joint_attention v concat failed: {err}")))?;

    let merged = merge_heads(scaled_dot_product_attention(
        &q,
        &k,
        &v,
        Some(mask),
        head_dim,
    )?)?;

    let pool_proj = merged
        .clone()
        .slice(0, 0, pool_len)
        .map_err(|err| Error::message(format!("joint_attention pool slice failed: {err}")))?
        .linear(&weights.pool.out.weight, Some(&weights.pool.out.bias))
        .map_err(|err| Error::message(format!("joint_attention pool out failed: {err}")))?;
    let x_proj = merged
        .slice(0, pool_len, total_len)
        .map_err(|err| Error::message(format!("joint_attention x slice failed: {err}")))?
        .linear(&weights.x.out.weight, Some(&weights.x.out.bias))
        .map_err(|err| Error::message(format!("joint_attention x out failed: {err}")))?;

    Ok(JointAttentionOutput {
        pool: pool_proj,
        x: x_proj,
    })
}

pub fn pac<T: Tensor>(
    x: &T,
    weights: &PacWeights<T>,
    positions: &[i32],
    num_heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<T> {
    let ax = x
        .clone()
        .rms_norm(&weights.a_norm, RMS_NORM_EPS)
        .map_err(|err| Error::message(format!("pac attention norm failed: {err}")))?;
    let a = attention(&ax, &weights.attn, positions, num_heads, head_dim, theta)?;

    let cx = x
        .clone()
        .rms_norm(&weights.c_norm, RMS_NORM_EPS)
        .map_err(|err| Error::message(format!("pac cgmlp norm failed: {err}")))?;
    let c = cgmlp(&cx, &weights.cgmlp)?;

    merge_stream(&a, &c, &weights.merge)
}

pub fn pjac<T: Tensor>(
    pool: &T,
    x: &T,
    weights: &PjacWeights<T>,
    global_positions: &[i32],
    region_ids: &[i32],
    mask: &T,
    num_heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<JointAttentionOutput<T>> {
    let attn = joint_attention(
        pool,
        x,
        &weights.jattn,
        global_positions,
        region_ids,
        mask,
        num_heads,
        head_dim,
        theta,
    )?;

    let pool_cn = pool
        .clone()
        .rms_norm(&weights.c_norm_pool, RMS_NORM_EPS)
        .map_err(|err| Error::message(format!("pjac pool cgmlp norm failed: {err}")))?;
    let x_cn = x
        .clone()
        .rms_norm(&weights.c_norm_x, RMS_NORM_EPS)
        .map_err(|err| Error::message(format!("pjac x cgmlp norm failed: {err}")))?;
    let c_pool = cgmlp(&pool_cn, &weights.cgmlp_pool)?;
    let c_x = cgmlp(&x_cn, &weights.cgmlp_x)?;

    Ok(JointAttentionOutput {
        pool: merge_stream(&attn.pool, &c_pool, &weights.merge_pool)?,
        x: merge_stream(&attn.x, &c_x, &weights.merge_x)?,
    })
}

pub fn ebf_block<T: Tensor>(
    x: &T,
    weights: &EbfBlockWeights<T>,
    positions: &[i32],
    num_heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<T> {
    let mut x = x.clone();
    if let Some(ffn1) = weights.ffn1.as_ref() {
        x = apply_residual_glu(&x, ffn1, 0.5)?;
    }

    let mut attn = pac(&x, &weights.pac, positions, num_heads, head_dim, theta)?;
    if let Some(layer_scale) = weights.pac_layer_scale.as_ref() {
        attn = attn
            .mul(layer_scale)
            .map_err(|err| Error::message(format!("ebf_block pac layer scale failed: {err}")))?;
    }
    x = x
        .add(&attn)
        .map_err(|err| Error::message(format!("ebf_block pac residual failed: {err}")))?;

    if let Some(ffn2) = weights.ffn2.as_ref() {
        x = apply_residual_glu(&x, ffn2, 0.5)?;
    }

    Ok(x)
}

pub fn jebf_block<T: Tensor>(
    pool: &T,
    x: &T,
    weights: &JebfBlockWeights<T>,
    global_positions: &[i32],
    region_ids: &[i32],
    mask: &T,
    num_heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<JointAttentionOutput<T>> {
    let mut pool = pool.clone();
    let mut x = x.clone();

    if let Some(ffn1_x) = weights.ffn1_x.as_ref() {
        x = apply_residual_glu(&x, ffn1_x, 1.0)?;
    }
    if let Some(ffn1_pool) = weights.ffn1_pool.as_ref() {
        pool = apply_residual_glu(&pool, ffn1_pool, 1.0)?;
    }

    let mut attn = pjac(
        &pool,
        &x,
        &weights.pjac,
        global_positions,
        region_ids,
        mask,
        num_heads,
        head_dim,
        theta,
    )?;
    if let Some(layer_scale) = weights.pjac_layer_scale_x.as_ref() {
        attn.x = attn.x.mul(layer_scale).map_err(|err| {
            Error::message(format!(
                "jebf_block x joint-attention layer scale failed: {err}"
            ))
        })?;
    }
    if let Some(layer_scale) = weights.pjac_layer_scale_pool.as_ref() {
        attn.pool = attn.pool.mul(layer_scale).map_err(|err| {
            Error::message(format!(
                "jebf_block pool joint-attention layer scale failed: {err}"
            ))
        })?;
    }
    x = x
        .add(&attn.x)
        .map_err(|err| Error::message(format!("jebf_block x joint residual failed: {err}")))?;
    pool = pool
        .add(&attn.pool)
        .map_err(|err| Error::message(format!("jebf_block pool joint residual failed: {err}")))?;

    if let Some(ffn2_x) = weights.ffn2_x.as_ref() {
        x = apply_residual_glu(&x, ffn2_x, 1.0)?;
    }
    if let Some(ffn2_pool) = weights.ffn2_pool.as_ref() {
        pool = apply_residual_glu(&pool, ffn2_pool, 1.0)?;
    }

    Ok(JointAttentionOutput { pool, x })
}

fn apply_optional_qk_norm<T: Tensor>(tensor: T, norm: Option<&T>, label: &str) -> Result<T> {
    match norm {
        Some(norm) => tensor
            .rms_norm(norm, RMS_NORM_EPS)
            .map_err(|err| Error::message(format!("{label} failed: {err}"))),
        None => Ok(tensor),
    }
}

fn apply_residual_glu<T: Tensor>(
    residual: &T,
    weights: &ResidualGluWeights<T>,
    branch_scale: f32,
) -> Result<T> {
    let normed = residual
        .clone()
        .rms_norm(&weights.norm, RMS_NORM_EPS)
        .map_err(|err| Error::message(format!("residual glu norm failed: {err}")))?;
    let mut branch = glu_ffn(&normed, &weights.ffn)?;
    if let Some(layer_scale) = weights.layer_scale.as_ref() {
        branch = branch
            .mul(layer_scale)
            .map_err(|err| Error::message(format!("residual glu layer scale failed: {err}")))?;
    }
    if branch_scale != 1.0 {
        branch = branch
            .scale(branch_scale)
            .map_err(|err| Error::message(format!("residual glu scaling failed: {err}")))?;
    }
    residual
        .clone()
        .add(&branch)
        .map_err(|err| Error::message(format!("residual glu add failed: {err}")))
}

fn merge_stream<T: Tensor>(
    attn: &T,
    cgmlp_branch: &T,
    weights: &super::weights::MergeWeights<T>,
) -> Result<T> {
    validate_sequence_2d("merge_stream attention", attn)?;
    let feature_axis = attn
        .shape()
        .len()
        .checked_sub(1)
        .ok_or_else(|| Error::message("merge_stream requires rank >= 1"))?;
    let merged = T::concat(&[attn, cgmlp_branch], feature_axis)
        .map_err(|err| Error::message(format!("merge_stream concat failed: {err}")))?;
    let merged = if let Some(dw) = weights.dw.as_ref() {
        let convolved = merged
            .clone()
            .conv1d_dw(
                &dw.weight,
                dw.bias.as_ref(),
                1,
                (dw.kernel_size.saturating_sub(1)) / 2,
            )
            .map_err(|err| Error::message(format!("merge_stream depthwise conv failed: {err}")))?;
        convolved.add(&merged).map_err(|err| {
            Error::message(format!("merge_stream residual conv add failed: {err}"))
        })?
    } else {
        merged
    };
    merged
        .linear(&weights.linear.weight, Some(&weights.linear.bias))
        .map_err(|err| Error::message(format!("merge_stream linear failed: {err}")))
}

fn reshape_for_heads<T: Tensor>(tensor: T, num_heads: usize, head_dim: usize) -> Result<T> {
    validate_sequence_2d("reshape_for_heads", &tensor)?;
    let seq_len = tensor.shape()[0];
    let expected = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::message("attention projection dimension overflow"))?;
    if tensor.shape()[1] != expected {
        return Err(Error::message(format!(
            "reshape_for_heads expected last dim {}, got shape {:?}",
            expected,
            tensor.shape()
        )));
    }

    tensor
        .reshape(&[seq_len, num_heads, head_dim])
        .and_then(|tensor| tensor.transpose(0, 1))
        .map_err(|err| Error::message(format!("reshape_for_heads failed: {err}")))
}

fn merge_heads<T: Tensor>(tensor: T) -> Result<T> {
    if tensor.shape().len() != 3 {
        return Err(Error::message(format!(
            "merge_heads expects [num_heads, seq_len, head_dim], got {:?}",
            tensor.shape()
        )));
    }

    let num_heads = tensor.shape()[0];
    let seq_len = tensor.shape()[1];
    let head_dim = tensor.shape()[2];
    let merged_dim = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::message("merge_heads dimension overflow"))?;

    tensor
        .transpose(0, 1)
        .and_then(|tensor| tensor.reshape(&[seq_len, merged_dim]))
        .map_err(|err| Error::message(format!("merge_heads failed: {err}")))
}

fn scaled_dot_product_attention<T: Tensor>(
    q: &T,
    k: &T,
    v: &T,
    mask: Option<&T>,
    head_dim: usize,
) -> Result<T> {
    if q.shape().len() != 3 || k.shape().len() != 3 || v.shape().len() != 3 {
        return Err(Error::message(format!(
            "scaled_dot_product_attention expects rank-3 tensors, got q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if q.shape()[0] != k.shape()[0] || q.shape()[0] != v.shape()[0] {
        return Err(Error::message(format!(
            "scaled_dot_product_attention head count mismatch: q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if q.shape()[2] != k.shape()[2] || q.shape()[2] != v.shape()[2] {
        return Err(Error::message(format!(
            "scaled_dot_product_attention head dimension mismatch: q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if k.shape()[1] != v.shape()[1] {
        return Err(Error::message(format!(
            "scaled_dot_product_attention key/value sequence mismatch: k={:?}, v={:?}",
            k.shape(),
            v.shape()
        )));
    }

    let query_len = q.shape()[1];
    let key_len = k.shape()[1];
    let query_chunk_len = choose_attention_query_chunk_len(q.shape()[0], query_len, key_len);
    scaled_dot_product_attention_chunked(q, k, v, mask, head_dim, query_chunk_len)
}

fn scaled_dot_product_attention_chunked<T: Tensor>(
    q: &T,
    k: &T,
    v: &T,
    mask: Option<&T>,
    head_dim: usize,
    query_chunk_len: usize,
) -> Result<T> {
    let query_len = q.shape()[1];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let k_t = k
        .clone()
        .transpose(1, 2)
        .map_err(|err| Error::message(format!("attention key transpose failed: {err}")))?;

    if query_chunk_len >= query_len {
        let mut scores = q
            .matmul(&k_t)
            .and_then(|scores| scores.scale(scale))
            .map_err(|err| Error::message(format!("attention score matmul failed: {err}")))?;
        if let Some(mask) = mask {
            scores = scores
                .add(mask)
                .map_err(|err| Error::message(format!("attention mask add failed: {err}")))?;
        }
        return scores
            .softmax(-1)
            .and_then(|probs| probs.matmul(v))
            .map_err(|err| Error::message(format!("attention value matmul failed: {err}")));
    }

    let mut outputs = Vec::new();
    for start in (0..query_len).step_by(query_chunk_len) {
        let end = (start + query_chunk_len).min(query_len);
        let q_chunk = q
            .clone()
            .slice(1, start, end)
            .map_err(|err| Error::message(format!("attention query chunk slice failed: {err}")))?;
        let mask_chunk = match mask {
            Some(mask) => Some(mask.clone().slice(0, start, end).map_err(|err| {
                Error::message(format!("attention mask chunk slice failed: {err}"))
            })?),
            None => None,
        };

        let mut scores = q_chunk
            .matmul(&k_t)
            .and_then(|scores| scores.scale(scale))
            .map_err(|err| Error::message(format!("attention score matmul failed: {err}")))?;
        if let Some(mask_chunk) = mask_chunk.as_ref() {
            scores = scores
                .add(mask_chunk)
                .map_err(|err| Error::message(format!("attention mask add failed: {err}")))?;
        }
        let chunk_output = scores
            .softmax(-1)
            .and_then(|probs| probs.matmul(v))
            .map_err(|err| Error::message(format!("attention value matmul failed: {err}")))?;
        outputs.push(chunk_output);
    }

    let refs = outputs.iter().collect::<Vec<_>>();
    T::concat(&refs, 1)
        .map_err(|err| Error::message(format!("attention output concat failed: {err}")))
}

fn choose_attention_query_chunk_len(num_heads: usize, query_len: usize, key_len: usize) -> usize {
    if num_heads == 0 || query_len == 0 || key_len == 0 {
        return query_len;
    }

    let score_elements = num_heads.saturating_mul(query_len).saturating_mul(key_len);
    if score_elements <= MAX_ATTENTION_SCORE_ELEMENTS {
        return query_len;
    }

    let rows = MAX_ATTENTION_SCORE_ELEMENTS / num_heads.saturating_mul(key_len).max(1);
    rows.max(1).min(query_len)
}

fn validate_head_config(num_heads: usize, head_dim: usize) -> Result<()> {
    if num_heads == 0 {
        return Err(Error::message("attention requires num_heads > 0"));
    }
    if head_dim == 0 {
        return Err(Error::message("attention requires head_dim > 0"));
    }
    Ok(())
}

fn validate_sequence_2d<T: Tensor>(label: &str, tensor: &T) -> Result<()> {
    if tensor.shape().len() != 2 {
        return Err(Error::message(format!(
            "{label} expects a rank-2 tensor shaped [seq_len, dim], got {:?}",
            tensor.shape()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::scaled_dot_product_attention_chunked;
    use crate::{CpuDevice, CpuTensor, Tensor};

    #[test]
    fn chunked_attention_matches_full_attention_with_mask() {
        let device = CpuDevice;
        let q = CpuTensor::from_data(
            &[
                1.0, 0.0, 0.5, 0.5, 0.2, 0.8, 0.0, 1.0, //
                0.3, 0.7, 0.6, 0.4, 0.9, 0.1, 0.4, 0.6,
            ],
            &[2, 4, 2],
            &device,
        )
        .unwrap();
        let k = CpuTensor::from_data(
            &[
                0.9, 0.1, 0.1, 0.9, 0.8, 0.2, 0.3, 0.7, //
                0.2, 0.8, 0.7, 0.3, 0.4, 0.6, 0.6, 0.4,
            ],
            &[2, 4, 2],
            &device,
        )
        .unwrap();
        let v = CpuTensor::from_data(
            &[
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, //
                0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
            ],
            &[2, 4, 2],
            &device,
        )
        .unwrap();
        let mask = CpuTensor::from_data(
            &[
                0.0, 0.0, -10_000.0, -10_000.0, //
                0.0, 0.0, 0.0, -10_000.0, //
                0.0, 0.0, 0.0, 0.0, //
                -10_000.0, 0.0, 0.0, 0.0,
            ],
            &[4, 4],
            &device,
        )
        .unwrap();

        let full = scaled_dot_product_attention_chunked(&q, &k, &v, Some(&mask), 2, 4).unwrap();
        let chunked = scaled_dot_product_attention_chunked(&q, &k, &v, Some(&mask), 2, 2).unwrap();

        assert_eq!(full.shape(), chunked.shape());
        let full_data = full.to_vec().unwrap();
        let chunked_data = chunked.to_vec().unwrap();
        for (lhs, rhs) in full_data.iter().zip(chunked_data.iter()) {
            assert!((lhs - rhs).abs() <= 1e-6, "lhs={lhs} rhs={rhs}");
        }
    }
}
