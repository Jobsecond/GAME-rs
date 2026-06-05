use std::sync::OnceLock;

use gemm::Parallelism;
use rayon::prelude::*;

use crate::Result;
use crate::profiler::op_scope_with;

use super::CpuTensor;
use super::util::*;

impl CpuTensor {
    pub(super) fn matmul(&self, rhs: &Self) -> Result<Self> {
        let lhs_shape = self.shape();
        let rhs_shape = rhs.shape();
        let lhs_rank = lhs_shape.len();
        let rhs_rank = rhs_shape.len();

        match (lhs_rank, rhs_rank) {
            (2, 2) => {
                let (m, k) = (lhs_shape[0], lhs_shape[1]);
                let (rhs_k, n) = (rhs_shape[0], rhs_shape[1]);
                if k != rhs_k {
                    return Err(invalid_arg(format!(
                        "matmul shape mismatch: {:?} @ {:?}",
                        lhs_shape, rhs_shape
                    )));
                }

                self.with_contiguous_data(|lhs_data| {
                    rhs.with_contiguous_data(|rhs_data| {
                        let mut out = vec![0.0; m * n];
                        if m > 0 && n > 0 && k > 0 {
                            unsafe {
                                gemm::gemm(
                                    m,
                                    n,
                                    k,
                                    out.as_mut_ptr(),
                                    1,
                                    n as isize,
                                    false,
                                    lhs_data.as_ptr(),
                                    1,
                                    k as isize,
                                    rhs_data.as_ptr(),
                                    1,
                                    n as isize,
                                    0.0,
                                    1.0,
                                    false,
                                    false,
                                    false,
                                    Parallelism::None,
                                );
                            }
                        }
                        Self::from_owned(out, &[m, n])
                    })
                })
            }
            (3, 3) => {
                let (batch, m, k) = (lhs_shape[0], lhs_shape[1], lhs_shape[2]);
                let (rhs_batch, rhs_k, n) = (rhs_shape[0], rhs_shape[1], rhs_shape[2]);
                if batch != rhs_batch || k != rhs_k {
                    return Err(invalid_arg(format!(
                        "batched matmul shape mismatch: {:?} @ {:?}",
                        lhs_shape, rhs_shape
                    )));
                }

                self.with_contiguous_data(|lhs_data| {
                    rhs.with_contiguous_data(|rhs_data| {
                        let lhs_batch_stride = m * k;
                        let rhs_batch_stride = k * n;
                        let out_batch_stride = m * n;
                        let mut out = vec![0.0; batch * out_batch_stride];
                        if m > 0 && n > 0 && k > 0 {
                            for b in 0..batch {
                                let lhs_ptr =
                                    unsafe { lhs_data.as_ptr().add(b * lhs_batch_stride) };
                                let rhs_ptr =
                                    unsafe { rhs_data.as_ptr().add(b * rhs_batch_stride) };
                                let out_ptr = unsafe { out.as_mut_ptr().add(b * out_batch_stride) };
                                unsafe {
                                    gemm::gemm(
                                        m,
                                        n,
                                        k,
                                        out_ptr,
                                        1,
                                        n as isize,
                                        false,
                                        lhs_ptr,
                                        1,
                                        k as isize,
                                        rhs_ptr,
                                        1,
                                        n as isize,
                                        0.0,
                                        1.0,
                                        false,
                                        false,
                                        false,
                                        Parallelism::None,
                                    );
                                }
                            }
                        }
                        Self::from_owned(out, &[batch, m, n])
                    })
                })
            }
            _ => Err(invalid_arg(format!(
                "matmul expects rank-2 or rank-3 tensors, got {:?} and {:?}",
                lhs_shape, rhs_shape
            ))),
        }
    }

    pub(super) fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self> {
        let _profile = op_scope_with("cpu.linear", || {
            format!(
                "input={:?} weight={:?} bias={} contiguous_input={} contiguous_weight={}",
                self.shape(),
                weight.shape(),
                bias.is_some(),
                self.is_contiguous(),
                weight.is_contiguous()
            )
        });
        let input_shape = self.shape();
        if input_shape.is_empty() {
            return Err(invalid_arg(
                "linear expects an input tensor with at least one dimension",
            ));
        }
        if weight.shape().len() != 2 {
            return Err(invalid_arg(format!(
                "linear weight must be rank-2 [out_dim, in_dim], got {:?}",
                weight.shape()
            )));
        }

        let in_dim = *input_shape.last().unwrap_or(&0);
        let out_dim = weight.shape()[0];
        if weight.shape()[1] != in_dim {
            return Err(invalid_arg(format!(
                "linear shape mismatch: input {:?}, weight {:?}",
                input_shape,
                weight.shape()
            )));
        }
        if let Some(bias) = bias
            && bias.shape() != [out_dim]
        {
            return Err(invalid_arg(format!(
                "linear bias must have shape [{out_dim}], got {:?}",
                bias.shape()
            )));
        }

        let rows = plain_num_elements(&input_shape[..input_shape.len() - 1]);
        let bias_data = bias.map(CpuTensor::to_vec).transpose()?;

        self.with_contiguous_data(|input| {
            weight.with_contiguous_data(|weight_data| {
                let mut out = vec![0.0; rows * out_dim];

                if should_parallelize_linear(rows, in_dim, out_dim) {
                    let row_chunk_len = choose_parallel_row_chunk_len(rows, out_dim);
                    out.par_chunks_mut(row_chunk_len * out_dim)
                        .enumerate()
                        .for_each(|(chunk_index, out_chunk)| {
                            let row_start = chunk_index * row_chunk_len;
                            let chunk_rows = out_chunk.len() / out_dim;
                            unsafe {
                                gemm::gemm(
                                    chunk_rows,
                                    out_dim,
                                    in_dim,
                                    out_chunk.as_mut_ptr(),
                                    1,
                                    out_dim as isize,
                                    false,
                                    input.as_ptr().add(row_start * in_dim),
                                    1,
                                    in_dim as isize,
                                    weight_data.as_ptr(),
                                    in_dim as isize,
                                    1,
                                    0.0,
                                    1.0,
                                    false,
                                    false,
                                    false,
                                    Parallelism::None,
                                );
                            }

                            if let Some(bias_values) = &bias_data {
                                for out_row in out_chunk.chunks_mut(out_dim) {
                                    for (out_idx, value) in out_row.iter_mut().enumerate() {
                                        *value += bias_values[out_idx];
                                    }
                                }
                            }
                        });
                } else {
                    unsafe {
                        gemm::gemm(
                            rows,
                            out_dim,
                            in_dim,
                            out.as_mut_ptr(),
                            1,
                            out_dim as isize,
                            false,
                            input.as_ptr(),
                            1,
                            in_dim as isize,
                            weight_data.as_ptr(),
                            in_dim as isize,
                            1,
                            0.0,
                            1.0,
                            false,
                            false,
                            false,
                            Parallelism::None,
                        );
                    }

                    if let Some(bias_values) = &bias_data {
                        for row in 0..rows {
                            for out_idx in 0..out_dim {
                                out[row * out_dim + out_idx] += bias_values[out_idx];
                            }
                        }
                    }
                }

                let mut out_shape = input_shape[..input_shape.len() - 1].to_vec();
                out_shape.push(out_dim);
                Self::from_owned(out, &out_shape)
            })
        })
    }
}

#[cfg(feature = "cpu-attention-gemm-matrixmultiply")]
#[inline]
/// # Safety
///
/// The caller must ensure:
/// - `lhs`, `rhs`, and `dst` are valid, properly-aligned pointers for their respective sizes.
/// - The pointed-to memory is correctly bounded: `lhs` must cover `rows * inner` elements
///   (via stride `lhs_rs`), `rhs` must cover `inner * cols` elements (via stride `rhs_rs`),
///   and `dst` must cover `rows * cols` elements (via stride `dst_rs`).
/// - All memory regions are properly scoped and don't alias or overlap during the call.
/// - Strides are signed and in units of elements (f32), not bytes.
/// - Shape validation (rows, cols, inner > 0 and shape consistency) happens upstream.
pub(super) unsafe fn attention_gemm_f32(
    rows: usize,
    cols: usize,
    inner: usize,
    product_scale: f32,
    lhs: *const f32,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const f32,
    rhs_cs: isize,
    rhs_rs: isize,
    dst: *mut f32,
    dst_cs: isize,
    dst_rs: isize,
) {
    unsafe {
        matrixmultiply::sgemm(
            rows,
            inner,
            cols,
            product_scale,
            lhs,
            lhs_rs,
            lhs_cs,
            rhs,
            rhs_rs,
            rhs_cs,
            0.0,
            dst,
            dst_rs,
            dst_cs,
        );
    }
}

#[cfg(feature = "cpu-attention-gemm-matrixmultiply")]
#[inline]
/// # Safety
///
/// The caller must ensure:
/// - `lhs`, `rhs`, and `dst` are valid, properly-aligned pointers for their respective sizes.
/// - The pointed-to memory is correctly bounded: `lhs` must cover `rows * inner` elements
///   (via stride `lhs_rs`), `rhs` must cover `inner * cols` elements (via stride `rhs_rs`),
///   and `dst` must cover `rows * cols` elements (via stride `dst_rs`).
/// - All memory regions are properly scoped and don't alias or overlap during the call.
/// - Strides are signed and in units of elements (f32), not bytes.
/// - Shape validation (rows, cols, inner > 0 and shape consistency) happens upstream.
pub(super) unsafe fn attention_gemm_f32_accum(
    rows: usize,
    cols: usize,
    inner: usize,
    product_scale: f32,
    lhs: *const f32,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const f32,
    rhs_cs: isize,
    rhs_rs: isize,
    dst: *mut f32,
    dst_cs: isize,
    dst_rs: isize,
) {
    unsafe {
        matrixmultiply::sgemm(
            rows,
            inner,
            cols,
            product_scale,
            lhs,
            lhs_rs,
            lhs_cs,
            rhs,
            rhs_rs,
            rhs_cs,
            1.0,
            dst,
            dst_rs,
            dst_cs,
        );
    }
}

#[cfg(not(feature = "cpu-attention-gemm-matrixmultiply"))]
#[inline]
/// # Safety
///
/// The caller must ensure:
/// - `lhs`, `rhs`, and `dst` are valid, properly-aligned pointers for their respective sizes.
/// - The pointed-to memory is correctly bounded: `lhs` must cover `rows * inner` elements
///   (via stride `lhs_rs`), `rhs` must cover `inner * cols` elements (via stride `rhs_rs`),
///   and `dst` must cover `rows * cols` elements (via stride `dst_rs`).
/// - All memory regions are properly scoped and don't alias or overlap during the call.
/// - Strides are signed and in units of elements (f32), not bytes.
/// - Shape validation (rows, cols, inner > 0 and shape consistency) happens upstream.
pub(super) unsafe fn attention_gemm_f32(
    rows: usize,
    cols: usize,
    inner: usize,
    product_scale: f32,
    lhs: *const f32,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const f32,
    rhs_cs: isize,
    rhs_rs: isize,
    dst: *mut f32,
    dst_cs: isize,
    dst_rs: isize,
) {
    unsafe {
        gemm::gemm(
            rows,
            cols,
            inner,
            dst,
            dst_cs,
            dst_rs,
            false,
            lhs,
            lhs_cs,
            lhs_rs,
            rhs,
            rhs_cs,
            rhs_rs,
            0.0,
            product_scale,
            false,
            false,
            false,
            Parallelism::None,
        );
    }
}

#[cfg(not(feature = "cpu-attention-gemm-matrixmultiply"))]
#[inline]
/// # Safety
///
/// The caller must ensure:
/// - `lhs`, `rhs`, and `dst` are valid, properly-aligned pointers for their respective sizes.
/// - The pointed-to memory is correctly bounded: `lhs` must cover `rows * inner` elements
///   (via stride `lhs_rs`), `rhs` must cover `inner * cols` elements (via stride `rhs_rs`),
///   and `dst` must cover `rows * cols` elements (via stride `dst_rs`).
/// - All memory regions are properly scoped and don't alias or overlap during the call.
/// - Strides are signed and in units of elements (f32), not bytes.
/// - Shape validation (rows, cols, inner > 0 and shape consistency) happens upstream.
pub(super) unsafe fn attention_gemm_f32_accum(
    rows: usize,
    cols: usize,
    inner: usize,
    product_scale: f32,
    lhs: *const f32,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const f32,
    rhs_cs: isize,
    rhs_rs: isize,
    dst: *mut f32,
    dst_cs: isize,
    dst_rs: isize,
) {
    unsafe {
        gemm::gemm(
            rows,
            cols,
            inner,
            dst,
            dst_cs,
            dst_rs,
            true,
            lhs,
            lhs_cs,
            lhs_rs,
            rhs,
            rhs_cs,
            rhs_rs,
            1.0,
            product_scale,
            false,
            false,
            false,
            Parallelism::None,
        );
    }
}

pub(super) fn should_parallelize_linear(rows: usize, in_dim: usize, out_dim: usize) -> bool {
    if rayon::current_num_threads() <= 1 || rows < 2 || in_dim == 0 || out_dim == 0 {
        return false;
    }

    let work = rows.saturating_mul(in_dim).saturating_mul(out_dim);
    work >= 1_000_000 && rows.saturating_mul(out_dim) >= 8_192
}

pub(super) fn choose_parallel_row_chunk_len(rows: usize, out_dim: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let target_tasks = linear_target_tasks(threads);
    let by_tasks = rows.div_ceil(target_tasks);
    let min_chunk_rows = linear_min_outputs_per_chunk().div_ceil(out_dim.max(1));
    by_tasks.max(min_chunk_rows).max(1).min(rows)
}

pub(super) fn linear_target_tasks(threads: usize) -> usize {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    static DEFAULT: OnceLock<usize> = OnceLock::new();
    OVERRIDE
        .get_or_init(|| {
            std::env::var("GAME_LINEAR_TARGET_TASKS")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|&value| value > 0)
        })
        .unwrap_or(*DEFAULT.get_or_init(|| num_cpus::get_physical().max(1)))
        .min(threads)
        .max(1)
}

pub(super) fn linear_min_outputs_per_chunk() -> usize {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    OVERRIDE
        .get_or_init(|| {
            std::env::var("GAME_LINEAR_MIN_OUTPUTS_PER_CHUNK")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|&value| value > 0)
        })
        .unwrap_or(16_384)
}
