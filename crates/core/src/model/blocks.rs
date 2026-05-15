use std::sync::OnceLock;

use crate::profiler::{scope, scope_with};
use crate::{Error, Result, Tensor};

use super::RMS_NORM_EPS;
use super::ops::{cgmlp, glu_ffn};
use super::weights::{
    AttentionWeights, EbfBlockWeights, JebfBlockWeights, JointAttentionWeights, PacWeights,
    PjacWeights, ResidualGluWeights,
};

// Chunk attention score rows so temporary score tensors stay bounded on both
// CPU and GPU backends.
const DEFAULT_MAX_ATTENTION_SCORE_ELEMENTS: usize = 32_000_000;
static MAX_ATTENTION_SCORE_ELEMENTS: OnceLock<usize> = OnceLock::new();

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
    let _scope = scope_with("attention", || {
        format!(
            "x={:?} heads={} head_dim={}",
            x.shape(),
            num_heads,
            head_dim
        )
    });
    validate_head_config(num_heads, head_dim)?;

    let q = x
        .linear(&weights.q.weight, Some(&weights.q.bias))
        .map_err(|err| Error::message(format!("attention q projection failed: {err}")))?;
    let kv = x
        .linear(&weights.kv.weight, Some(&weights.kv.bias))
        .map_err(|err| Error::message(format!("attention kv projection failed: {err}")))?;
    let (k, v) = kv
        .split_last_dim_two_for_attention_heads(num_heads, head_dim)
        .map_err(|err| Error::message(format!("attention kv split/layout failed: {err}")))?;

    let q = reshape_for_heads(q, num_heads, head_dim)?
        .rope(positions, head_dim, num_heads, head_dim, theta)
        .map_err(|err| Error::message(format!("attention q rope failed: {err}")))?;
    let k = k
        .rope(positions, head_dim, num_heads, head_dim, theta)
        .map_err(|err| Error::message(format!("attention k rope failed: {err}")))?;

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
    let _scope = scope_with("joint_attention", || {
        format!(
            "pool={:?} x={:?} total_len={} heads={} head_dim={}",
            pool.shape(),
            x.shape(),
            pool.shape()[0] + x.shape()[0],
            num_heads,
            head_dim
        )
    });
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

    let _pool_norm_scope = scope(
        "joint_attention.pool_norm",
        format!("pool={:?}", pool.shape()),
    );
    let pool_norm = pool
        .clone()
        .rms_norm(&weights.pool.norm, RMS_NORM_EPS)
        .map_err(|err| Error::message(format!("joint_attention pool norm failed: {err}")))?;
    let _x_norm_scope = scope("joint_attention.x_norm", format!("x={:?}", x.shape()));
    let x_norm = x
        .clone()
        .rms_norm(&weights.x.norm, RMS_NORM_EPS)
        .map_err(|err| Error::message(format!("joint_attention x norm failed: {err}")))?;

    let _pool_qkv_scope = scope(
        "joint_attention.pool_qkv",
        format!("pool_norm={:?}", pool_norm.shape()),
    );
    let pool_qkv = pool_norm
        .linear(&weights.pool.qkv.weight, Some(&weights.pool.qkv.bias))
        .map_err(|err| Error::message(format!("joint_attention pool qkv failed: {err}")))?;
    let _x_qkv_scope = scope(
        "joint_attention.x_qkv",
        format!("x_norm={:?}", x_norm.shape()),
    );
    let x_qkv = x_norm
        .linear(&weights.x.qkv.weight, Some(&weights.x.qkv.bias))
        .map_err(|err| Error::message(format!("joint_attention x qkv failed: {err}")))?;

    let (pool_q, pool_k, pool_v) = pool_qkv
        .split_last_dim_three_for_attention_heads(num_heads, head_dim)
        .map_err(|err| {
            Error::message(format!(
                "joint_attention pool qkv split/layout failed: {err}"
            ))
        })?;
    let (x_q, x_k, x_v) = x_qkv
        .split_last_dim_three_for_attention_heads(num_heads, head_dim)
        .map_err(|err| {
            Error::message(format!("joint_attention x qkv split/layout failed: {err}"))
        })?;

    let pool_q = apply_optional_qk_norm(
        pool_q,
        weights.pool.qk_norm.as_ref().map(|norm| &norm.q),
        "joint_attention pool q_norm",
    )?;
    let pool_k = apply_optional_qk_norm(
        pool_k,
        weights.pool.qk_norm.as_ref().map(|norm| &norm.k),
        "joint_attention pool k_norm",
    )?;

    let x_q = apply_optional_qk_norm(
        x_q,
        weights.x.qk_norm.as_ref().map(|norm| &norm.q),
        "joint_attention x q_norm",
    )?;
    let x_k = apply_optional_qk_norm(
        x_k,
        weights.x.qk_norm.as_ref().map(|norm| &norm.k),
        "joint_attention x k_norm",
    )?;

    let _q_rope_scope = scope(
        "joint_attention.q_rope",
        format!("pool_q={:?} x_q={:?}", pool_q.shape(), x_q.shape()),
    );
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
    let _k_rope_scope = scope(
        "joint_attention.k_rope",
        format!("pool_k={:?} x_k={:?}", pool_k.shape(), x_k.shape()),
    );
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
    let _v_concat_scope = scope(
        "joint_attention.v_concat",
        format!("pool_v={:?} x_v={:?}", pool_v.shape(), x_v.shape()),
    );
    let v = T::concat(&[&pool_v, &x_v], 1)
        .map_err(|err| Error::message(format!("joint_attention v concat failed: {err}")))?;

    let _attn_scope = scope(
        "joint_attention.attention",
        format!("q={:?} k={:?} v={:?}", q.shape(), k.shape(), v.shape()),
    );
    let merged = merge_heads(scaled_dot_product_attention(
        &q,
        &k,
        &v,
        Some(mask),
        head_dim,
    )?)?;

    let _pool_out_scope = scope(
        "joint_attention.pool_out",
        format!("merged={:?} pool_len={}", merged.shape(), pool_len),
    );
    let pool_proj = merged
        .clone()
        .slice(0, 0, pool_len)
        .map_err(|err| Error::message(format!("joint_attention pool slice failed: {err}")))?
        .linear(&weights.pool.out.weight, Some(&weights.pool.out.bias))
        .map_err(|err| Error::message(format!("joint_attention pool out failed: {err}")))?;
    let _x_out_scope = scope(
        "joint_attention.x_out",
        format!("merged={:?} x_len={}", merged.shape(), x_len),
    );
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
    let _scope = scope_with("pac", || format!("x={:?}", x.shape()));
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
    let _scope = scope_with("pjac", || {
        format!("pool={:?} x={:?}", pool.shape(), x.shape())
    });
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
    let _scope = scope_with("ebf_block", || format!("x={:?}", x.shape()));
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
    let _scope = scope_with("jebf_block", || {
        format!("pool={:?} x={:?}", pool.shape(), x.shape())
    });
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
    let _scope = scope_with("merge_stream", || {
        format!("attn={:?} cgmlp={:?}", attn.shape(), cgmlp_branch.shape())
    });
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
        .layout_for_attention_heads(num_heads, head_dim)
        .map_err(|err| Error::message(format!("reshape_for_heads failed: {err}")))
}

fn merge_heads<T: Tensor>(tensor: T) -> Result<T> {
    if tensor.shape().len() != 3 {
        return Err(Error::message(format!(
            "merge_heads expects [num_heads, seq_len, head_dim], got {:?}",
            tensor.shape()
        )));
    }

    tensor
        .merge_attention_heads()
        .map_err(|err| Error::message(format!("merge_heads failed: {err}")))
}

fn scaled_dot_product_attention<T: Tensor>(
    q: &T,
    k: &T,
    v: &T,
    mask: Option<&T>,
    head_dim: usize,
) -> Result<T> {
    let _scope = scope_with("scaled_dot_product_attention", || {
        format!(
            "q={:?} k={:?} v={:?} mask={} head_dim={}",
            q.shape(),
            k.shape(),
            v.shape(),
            mask.is_some(),
            head_dim
        )
    });
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
    let _scope = scope_with("scaled_dot_product_attention_chunked", || {
        format!(
            "q={:?} k={:?} v={:?} mask={} head_dim={} query_chunk_len={}",
            q.shape(),
            k.shape(),
            v.shape(),
            mask.is_some(),
            head_dim,
            query_chunk_len
        )
    });
    let query_len = q.shape()[1];
    let scale = 1.0 / (head_dim as f32).sqrt();

    if query_chunk_len >= query_len {
        return T::fused_attention(q, k, v, mask, scale)
            .map_err(|err| Error::message(format!("fused attention failed: {err}")));
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

        let chunk_output = T::fused_attention(&q_chunk, k, v, mask_chunk.as_ref(), scale)
            .map_err(|err| Error::message(format!("fused attention failed: {err}")))?;
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
    let max_score_elements = max_attention_score_elements();
    if score_elements <= max_score_elements {
        return query_len;
    }

    let rows = max_score_elements / num_heads.saturating_mul(key_len).max(1);
    rows.max(1).min(query_len)
}

fn max_attention_score_elements() -> usize {
    *MAX_ATTENTION_SCORE_ELEMENTS.get_or_init(|| {
        std::env::var("GAME_MAX_ATTENTION_SCORE_ELEMENTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_MAX_ATTENTION_SCORE_ELEMENTS)
    })
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
