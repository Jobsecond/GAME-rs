use std::sync::OnceLock;

use gemm::Parallelism;
use rayon::prelude::*;

use crate::Result;
use crate::profiler::op_scope_with;

use super::CpuTensor;
use super::util::*;

impl CpuTensor {
    pub(super) fn matmul(&self, rhs: &Self) -> Result<Self> {
        Ok(Self::from_tensor(self.tensor.matmul(&rhs.tensor)?))
    }

    pub(super) fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self> {
        let _profile = op_scope_with("cpu.linear", || {
            format!(
                "input={:?} weight={:?} bias={} contiguous_input={} contiguous_weight={}",
                self.shape(),
                weight.shape(),
                bias.is_some(),
                self.tensor.is_contiguous(),
                weight.tensor.is_contiguous()
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

#[cfg(not(feature = "cpu-attention-gemm-matrixmultiply"))]
#[inline]
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
