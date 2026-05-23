use rayon::prelude::*;

use crate::Result;
use crate::profiler::op_scope_with;

use super::CpuTensor;
use super::elementwise::apply_softmax_inplace;
use super::matmul::attention_gemm_f32;
use super::util::*;

impl CpuTensor {
    pub(super) fn attention_score_softmax(
        q: &Self,
        k_t: &Self,
        mask: Option<&Self>,
        scale: f32,
    ) -> Result<Self> {
        let _profile = op_scope_with("cpu.attention_score_softmax", || {
            format!(
                "q={:?} k_t={:?} mask={} scale={}",
                q.shape(),
                k_t.shape(),
                mask.is_some(),
                scale
            )
        });
        if q.shape().len() != 3 || k_t.shape().len() != 3 {
            return Err(invalid_arg(format!(
                "attention_score_softmax expects q/k_t rank-3, got {:?} and {:?}",
                q.shape(),
                k_t.shape()
            )));
        }

        let (heads, query_len, head_dim) = (q.shape()[0], q.shape()[1], q.shape()[2]);
        let (k_heads, k_head_dim, key_len) = (k_t.shape()[0], k_t.shape()[1], k_t.shape()[2]);
        if heads != k_heads || head_dim != k_head_dim {
            return Err(invalid_arg(format!(
                "attention_score_softmax shape mismatch: q={:?} k_t={:?}",
                q.shape(),
                k_t.shape()
            )));
        }
        let q_ptr = q.data_ptr() as usize;
        let q_batch_stride = q.strides[0];
        let q_rs = q.strides[1] as isize;
        let q_cs = q.strides[2] as isize;

        let k_ptr = k_t.data_ptr() as usize;
        let k_batch_stride = k_t.strides[0];
        let k_rs = k_t.strides[1] as isize;
        let k_cs = k_t.strides[2] as isize;

        let mask_shape = mask.map(|mask| mask.shape().to_vec());
        if let Some(mask_shape) = mask_shape.as_deref()
            && mask_shape != [query_len, key_len]
            && mask_shape != [heads, query_len, key_len]
        {
            return Err(invalid_arg(format!(
                "attention_score_softmax mask shape must be [{query_len}, {key_len}] or [{heads}, {query_len}, {key_len}], got {:?}",
                mask_shape
            )));
        }

        let (mask_data_owned, mask_data_borrowed): (Option<Vec<f32>>, Option<&[f32]>) =
            if let Some(mask) = mask {
                if mask.is_contiguous() {
                    let n = mask.num_elements();
                    (None, Some(&mask.data[mask.offset..mask.offset + n]))
                } else {
                    (Some(mask.to_vec()?), None)
                }
            } else {
                (None, None)
            };
        let mask_data: Option<&[f32]> = mask_data_borrowed.or(mask_data_owned.as_deref());
        let mask_is_contiguous_copy = mask_data_owned.is_some();
        let mask_outer_stride = mask.map(|mask| {
            if mask_is_contiguous_copy {
                key_len
            } else {
                match mask.shape().len() {
                    2 => mask.strides[0],
                    3 => mask.strides[1],
                    _ => 0,
                }
            }
        });
        let mask_head_stride = mask.and_then(|mask| {
            if mask.shape().len() == 3 {
                if mask_is_contiguous_copy {
                    Some(query_len * key_len)
                } else {
                    Some(mask.strides[0])
                }
            } else {
                None
            }
        });

        let mut out = vec![0.0; heads * query_len * key_len];

        if should_parallelize(out.len()) && query_len * key_len > 0 {
            out.par_chunks_mut(query_len * key_len)
                .enumerate()
                .for_each(|(head, out_head)| {
                    let lhs_ptr =
                        (q_ptr + head * q_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                    let rhs_ptr =
                        (k_ptr + head * k_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                    unsafe {
                        attention_gemm_f32(
                            query_len,
                            key_len,
                            head_dim,
                            scale,
                            lhs_ptr,
                            q_cs,
                            q_rs,
                            rhs_ptr,
                            k_cs,
                            k_rs,
                            out_head.as_mut_ptr(),
                            1,
                            key_len as isize,
                        );
                    }
                });
        } else {
            for head in 0..heads {
                let out_head =
                    &mut out[head * query_len * key_len..(head + 1) * query_len * key_len];
                let lhs_ptr =
                    (q_ptr + head * q_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                let rhs_ptr =
                    (k_ptr + head * k_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                unsafe {
                    attention_gemm_f32(
                        query_len,
                        key_len,
                        head_dim,
                        scale,
                        lhs_ptr,
                        q_cs,
                        q_rs,
                        rhs_ptr,
                        k_cs,
                        k_rs,
                        out_head.as_mut_ptr(),
                        1,
                        key_len as isize,
                    );
                }
            }
        }

        if should_parallelize(out.len()) && key_len > 0 {
            out.par_chunks_mut(key_len)
                .enumerate()
                .for_each(|(flat_row, row_scores)| {
                    let head = flat_row / query_len;
                    let row = flat_row % query_len;
                    apply_mask_and_softmax_row(
                        row_scores,
                        mask_data,
                        mask_shape.as_deref(),
                        mask_outer_stride,
                        mask_head_stride,
                        head,
                        row,
                    );
                });
        } else {
            for flat_row in 0..heads * query_len {
                let head = flat_row / query_len;
                let row = flat_row % query_len;
                let row_start = flat_row * key_len;
                let row_end = row_start + key_len;
                apply_mask_and_softmax_row(
                    &mut out[row_start..row_end],
                    mask_data,
                    mask_shape.as_deref(),
                    mask_outer_stride,
                    mask_head_stride,
                    head,
                    row,
                );
            }
        }

        Self::from_owned(out, &[heads, query_len, key_len])
    }

    pub(super) fn attention_value_matmul(probs: &Self, v: &Self) -> Result<Self> {
        let _profile = op_scope_with("cpu.attention_value_matmul", || {
            format!("probs={:?} v={:?}", probs.shape(), v.shape())
        });
        if probs.shape().len() != 3 || v.shape().len() != 3 {
            return probs.matmul(v);
        }

        let (heads, query_len, key_len) = (probs.shape()[0], probs.shape()[1], probs.shape()[2]);
        let (v_heads, v_key_len, head_dim) = (v.shape()[0], v.shape()[1], v.shape()[2]);
        if heads != v_heads || key_len != v_key_len {
            return probs.matmul(v);
        }
        let probs_ptr = probs.data_ptr() as usize;
        let probs_batch_stride = probs.strides[0];
        let probs_rs = probs.strides[1] as isize;
        let probs_cs = probs.strides[2] as isize;
        let probs_row_stride = probs.strides[1];

        let v_ptr = v.data_ptr() as usize;
        let v_batch_stride = v.strides[0];
        let v_rs = v.strides[1] as isize;
        let v_cs = v.strides[2] as isize;

        let mut out = vec![0.0; heads * query_len * head_dim];
        let head_block = query_len * head_dim;
        let row_chunk_len = choose_parallel_attention_row_chunk_len(query_len, key_len, head_dim);

        if should_parallelize_attention_matmul(heads, query_len, key_len, head_dim)
            && row_chunk_len < query_len
            && head_block > 0
        {
            out.par_chunks_mut(head_block)
                .enumerate()
                .for_each(|(head, out_head)| {
                    let lhs_base =
                        probs_ptr + head * probs_batch_stride * std::mem::size_of::<f32>();
                    let rhs_base = v_ptr + head * v_batch_stride * std::mem::size_of::<f32>();
                    out_head
                        .par_chunks_mut(row_chunk_len * head_dim)
                        .enumerate()
                        .for_each(|(chunk_index, out_chunk)| {
                            let row_start = chunk_index * row_chunk_len;
                            let chunk_rows = out_chunk.len() / head_dim;
                            let lhs_ptr = (lhs_base
                                + row_start * probs_row_stride * std::mem::size_of::<f32>())
                                as *const f32;
                            let rhs_ptr = rhs_base as *const f32;
                            unsafe {
                                attention_gemm_f32(
                                    chunk_rows,
                                    head_dim,
                                    key_len,
                                    1.0,
                                    lhs_ptr,
                                    probs_cs,
                                    probs_rs,
                                    rhs_ptr,
                                    v_cs,
                                    v_rs,
                                    out_chunk.as_mut_ptr(),
                                    1,
                                    head_dim as isize,
                                );
                            }
                        });
                });
        } else if should_parallelize(out.len()) && query_len * head_dim > 0 {
            out.par_chunks_mut(query_len * head_dim)
                .enumerate()
                .for_each(|(head, out_head)| {
                    let lhs_ptr = (probs_ptr
                        + head * probs_batch_stride * std::mem::size_of::<f32>())
                        as *const f32;
                    let rhs_ptr =
                        (v_ptr + head * v_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                    unsafe {
                        attention_gemm_f32(
                            query_len,
                            head_dim,
                            key_len,
                            1.0,
                            lhs_ptr,
                            probs_cs,
                            probs_rs,
                            rhs_ptr,
                            v_cs,
                            v_rs,
                            out_head.as_mut_ptr(),
                            1,
                            head_dim as isize,
                        );
                    }
                });
        } else {
            for head in 0..heads {
                let out_head =
                    &mut out[head * query_len * head_dim..(head + 1) * query_len * head_dim];
                let lhs_ptr = (probs_ptr + head * probs_batch_stride * std::mem::size_of::<f32>())
                    as *const f32;
                let rhs_ptr =
                    (v_ptr + head * v_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                unsafe {
                    attention_gemm_f32(
                        query_len,
                        head_dim,
                        key_len,
                        1.0,
                        lhs_ptr,
                        probs_cs,
                        probs_rs,
                        rhs_ptr,
                        v_cs,
                        v_rs,
                        out_head.as_mut_ptr(),
                        1,
                        head_dim as isize,
                    );
                }
            }
        }

        Self::from_owned(out, &[heads, query_len, head_dim])
    }

    pub(super) fn fused_attention(
        q: &Self,
        k: &Self,
        v: &Self,
        mask: Option<&Self>,
        scale: f32,
    ) -> Result<Self> {
        let _profile = op_scope_with("cpu.fused_attention", || {
            format!(
                "q={:?} k={:?} v={:?} mask={} scale={}",
                q.shape(),
                k.shape(),
                v.shape(),
                mask.is_some(),
                scale
            )
        });

        if q.shape().len() != 3 || k.shape().len() != 3 || v.shape().len() != 3 {
            return Err(invalid_arg(format!(
                "fused_attention expects rank-3 q/k/v, got {:?}, {:?}, {:?}",
                q.shape(),
                k.shape(),
                v.shape()
            )));
        }
        let (heads, q_len, head_dim) = (q.shape()[0], q.shape()[1], q.shape()[2]);
        let key_len = k.shape()[1];

        if k.shape()[2] != head_dim || v.shape()[2] != head_dim {
            return Err(invalid_arg(format!(
                "fused_attention head_dim mismatch: q={:?} k={:?} v={:?}",
                q.shape(),
                k.shape(),
                v.shape()
            )));
        }
        if q.shape()[0] != k.shape()[0] || q.shape()[0] != v.shape()[0] {
            return Err(invalid_arg(format!(
                "fused_attention head count mismatch: q={:?} k={:?} v={:?}",
                q.shape(),
                k.shape(),
                v.shape()
            )));
        }
        if k.shape()[1] != v.shape()[1] {
            return Err(invalid_arg(format!(
                "fused_attention key/value seq mismatch: k={:?} v={:?}",
                k.shape(),
                v.shape()
            )));
        }
        let mask_shape = mask.map(|m| m.shape().to_vec());
        let mask_2d = mask_shape.as_deref().map(|s| s.len() == 2).unwrap_or(false);

        let k_t = k.clone().transpose(1, 2)?;
        let k_ptr = k_t.data_ptr() as usize;
        let k_batch_stride = k_t.strides[0];
        let k_rs = k_t.strides[1] as isize;
        let k_cs = k_t.strides[2] as isize;

        let q_ptr = q.data_ptr() as usize;
        let q_batch_stride = q.strides[0];
        let q_rs = q.strides[1] as isize;
        let q_cs = q.strides[2] as isize;

        let v_ptr = v.data_ptr() as usize;
        let v_batch_stride = v.strides[0];
        let v_rs = v.strides[1] as isize;
        let v_cs = v.strides[2] as isize;

        let mask_owned: Option<Vec<f32>> = mask.map(|m| m.to_vec()).transpose()?;

        let mut scores = vec![0.0; heads * q_len * key_len];
        let score_head_stride = q_len * key_len;
        let score_row_stride = key_len;

        let score_rs = key_len as isize;
        let score_cs = 1isize;
        if should_parallelize(heads * q_len * key_len) && key_len > 0 && head_dim > 0 {
            scores
                .par_chunks_mut(score_head_stride)
                .enumerate()
                .for_each(|(head, out_head)| {
                    let lhs_ptr =
                        (q_ptr + head * q_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                    let rhs_ptr =
                        (k_ptr + head * k_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                    unsafe {
                        attention_gemm_f32(
                            q_len,
                            key_len,
                            head_dim,
                            scale,
                            lhs_ptr,
                            q_cs,
                            q_rs,
                            rhs_ptr,
                            k_cs,
                            k_rs,
                            out_head.as_mut_ptr(),
                            score_cs,
                            score_rs,
                        );
                    }
                });
        } else {
            for head in 0..heads {
                let lhs_ptr =
                    (q_ptr + head * q_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                let rhs_ptr =
                    (k_ptr + head * k_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                let out_head =
                    &mut scores[head * score_head_stride..(head + 1) * score_head_stride];
                unsafe {
                    attention_gemm_f32(
                        q_len,
                        key_len,
                        head_dim,
                        scale,
                        lhs_ptr,
                        q_cs,
                        q_rs,
                        rhs_ptr,
                        k_cs,
                        k_rs,
                        out_head.as_mut_ptr(),
                        score_cs,
                        score_rs,
                    );
                }
            }
        }

        let total_score_rows = heads * q_len;
        let mask_ref: Option<&[f32]> = mask_owned.as_deref();
        if should_parallelize(total_score_rows * key_len) && key_len > 0 {
            scores
                .par_chunks_mut(score_row_stride)
                .enumerate()
                .for_each(|(flat_row, row_scores)| {
                    let head = flat_row / q_len;
                    let row = flat_row % q_len;
                    if let Some(mask_data) = mask_ref {
                        let mask_base = if mask_2d {
                            row * key_len
                        } else {
                            head * q_len * key_len + row * key_len
                        };
                        for j in 0..key_len {
                            row_scores[j] += mask_data[mask_base + j];
                        }
                    }
                    apply_softmax_inplace(row_scores);
                });
        } else {
            for flat_row in 0..total_score_rows {
                let head = flat_row / q_len;
                let row = flat_row % q_len;
                let row_start = flat_row * key_len;
                let row_scores = &mut scores[row_start..row_start + key_len];
                if let Some(mask_data) = mask_ref {
                    let mask_base = if mask_2d {
                        row * key_len
                    } else {
                        head * q_len * key_len + row * key_len
                    };
                    for j in 0..key_len {
                        row_scores[j] += mask_data[mask_base + j];
                    }
                }
                apply_softmax_inplace(row_scores);
            }
        }

        let mut out = vec![0.0; heads * q_len * head_dim];
        let out_head_stride = q_len * head_dim;
        if should_parallelize(heads * q_len * head_dim) && key_len > 0 && head_dim > 0 {
            out.par_chunks_mut(out_head_stride)
                .enumerate()
                .for_each(|(head, out_head)| {
                    let score_base = head * score_head_stride;
                    let rhs_ptr =
                        (v_ptr + head * v_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                    unsafe {
                        attention_gemm_f32(
                            q_len,
                            head_dim,
                            key_len,
                            1.0,
                            scores[score_base..].as_ptr(),
                            score_cs,
                            score_rs,
                            rhs_ptr,
                            v_cs,
                            v_rs,
                            out_head.as_mut_ptr(),
                            1,
                            head_dim as isize,
                        );
                    }
                });
        } else {
            for head in 0..heads {
                let score_base = head * score_head_stride;
                let rhs_ptr =
                    (v_ptr + head * v_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                let out_head = &mut out[head * out_head_stride..(head + 1) * out_head_stride];
                unsafe {
                    attention_gemm_f32(
                        q_len,
                        head_dim,
                        key_len,
                        1.0,
                        scores[score_base..].as_ptr(),
                        score_cs,
                        score_rs,
                        rhs_ptr,
                        v_cs,
                        v_rs,
                        out_head.as_mut_ptr(),
                        1,
                        head_dim as isize,
                    );
                }
            }
        }

        Self::from_owned(out, &[heads, q_len, head_dim])
    }
}

pub(super) fn should_parallelize_attention_matmul(
    batch: usize,
    rows: usize,
    shared_dim: usize,
    out_dim: usize,
) -> bool {
    if rayon::current_num_threads() <= 1
        || batch == 0
        || rows == 0
        || shared_dim == 0
        || out_dim == 0
    {
        return false;
    }

    let work = batch
        .saturating_mul(rows)
        .saturating_mul(shared_dim)
        .saturating_mul(out_dim);
    work >= 4_000_000 && batch.saturating_mul(rows) >= 32
}

pub(super) fn choose_parallel_attention_row_chunk_len(
    rows: usize,
    shared_dim: usize,
    out_dim: usize,
) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let target_tasks = threads.saturating_mul(4);
    let by_tasks = rows.div_ceil(target_tasks);
    let per_row_work = shared_dim.saturating_mul(out_dim).max(1);
    let min_chunk_rows = 1_000_000usize.div_ceil(per_row_work);
    by_tasks.max(min_chunk_rows).max(1).min(rows)
}

pub(super) fn attention_mask_row_slice<'a>(
    mask_data: Option<&'a [f32]>,
    mask_shape: Option<&[usize]>,
    mask_outer_stride: Option<usize>,
    mask_head_stride: Option<usize>,
    head: usize,
    row: usize,
    col_start: usize,
    len: usize,
) -> Option<&'a [f32]> {
    let mask = mask_data?;
    let outer_stride = mask_outer_stride?;
    let mask_base = match mask_shape {
        Some([_, _]) => row * outer_stride,
        Some([_, _, _]) => head * mask_head_stride.unwrap_or(0) + row * outer_stride,
        _ => return None,
    };
    let start = mask_base + col_start;
    let end = start + len;
    if end > mask.len() {
        return None;
    }
    Some(&mask[start..end])
}

pub(super) fn apply_mask_and_softmax_row(
    row_scores: &mut [f32],
    mask_data: Option<&[f32]>,
    mask_shape: Option<&[usize]>,
    mask_outer_stride: Option<usize>,
    mask_head_stride: Option<usize>,
    head: usize,
    row: usize,
) {
    if row_scores.is_empty() {
        return;
    }

    let key_len = row_scores.len();
    if let Some(mask_row) = attention_mask_row_slice(
        mask_data,
        mask_shape,
        mask_outer_stride,
        mask_head_stride,
        head,
        row,
        0,
        key_len,
    ) {
        for col in 0..key_len {
            row_scores[col] += mask_row[col];
        }
    }

    let mut max_value = f32::NEG_INFINITY;
    for &value in row_scores.iter() {
        if value > max_value {
            max_value = value;
        }
    }

    let mut sum = 0.0;
    for value in row_scores.iter_mut() {
        *value = (*value - max_value).exp();
        sum += *value;
    }

    if sum > 0.0 {
        for value in row_scores.iter_mut() {
            *value /= sum;
        }
    }
}
