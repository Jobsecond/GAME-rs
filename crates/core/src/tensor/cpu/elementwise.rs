use std::f32::consts::SQRT_2;

use rayon::prelude::*;

use crate::profiler::op_scope_with;
use crate::{Error, Result};

use super::CpuTensor;
use super::util::*;

impl CpuTensor {
    pub(super) fn unary_op(
        self,
        op_name: &str,
        f: impl Fn(f32) -> f32 + Send + Sync,
    ) -> Result<Self> {
        let shape = self.shape().to_vec();
        let mut data = self.to_vec()?;
        if should_parallelize(data.len()) {
            data.par_iter_mut().for_each(|value| *value = f(*value));
        } else {
            for value in &mut data {
                *value = f(*value);
            }
        }
        Self::from_owned(data, &shape)
            .map_err(|err| Error::message(format!("failed to build result for {op_name}: {err}")))
    }

    pub(super) fn binary_op(
        self,
        rhs: &Self,
        op_name: &str,
        f: impl Fn(f32, f32) -> f32 + Send + Sync,
    ) -> Result<Self> {
        let _profile = op_scope_with("cpu.binary_op", || {
            format!(
                "op={} lhs={:?} rhs={:?}",
                op_name,
                self.shape(),
                rhs.shape()
            )
        });
        let lhs_shape = self.shape().to_vec();
        let rhs_shape = rhs.shape().to_vec();
        let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
        let out_rank = out_shape.len();

        if lhs_shape == out_shape && rhs_shape == out_shape {
            return self.with_contiguous_data(|lhs_data| {
                rhs.with_contiguous_data(|rhs_data| {
                    let mut out = vec![0.0; checked_num_elements(&out_shape)?];
                    if should_parallelize(out.len()) {
                        out.par_iter_mut()
                            .enumerate()
                            .for_each(|(flat, value)| *value = f(lhs_data[flat], rhs_data[flat]));
                    } else {
                        for flat in 0..out.len() {
                            out[flat] = f(lhs_data[flat], rhs_data[flat]);
                        }
                    }
                    Self::from_owned(out, &out_shape).map_err(|err| {
                        Error::message(format!("failed to build result for {op_name}: {err}"))
                    })
                })
            });
        }

        if let Some(block_len) = suffix_broadcast_block_len(&lhs_shape, &rhs_shape, &out_shape) {
            return self.with_contiguous_data(|lhs_data| {
                rhs.with_contiguous_data(|rhs_data| {
                    let mut out = vec![0.0; checked_num_elements(&out_shape)?];
                    let blocks = out.len() / block_len.max(1);
                    for block_index in 0..blocks {
                        let base = block_index * block_len;
                        for index in 0..block_len {
                            out[base + index] = f(lhs_data[base + index], rhs_data[index]);
                        }
                    }
                    Self::from_owned(out, &out_shape).map_err(|err| {
                        Error::message(format!("failed to build result for {op_name}: {err}"))
                    })
                })
            });
        }

        if let Some(block_len) = suffix_broadcast_block_len(&rhs_shape, &lhs_shape, &out_shape) {
            return self.with_contiguous_data(|lhs_data| {
                rhs.with_contiguous_data(|rhs_data| {
                    let mut out = vec![0.0; checked_num_elements(&out_shape)?];
                    let blocks = out.len() / block_len.max(1);
                    for block_index in 0..blocks {
                        let base = block_index * block_len;
                        for index in 0..block_len {
                            out[base + index] = f(lhs_data[index], rhs_data[base + index]);
                        }
                    }
                    Self::from_owned(out, &out_shape).map_err(|err| {
                        Error::message(format!("failed to build result for {op_name}: {err}"))
                    })
                })
            });
        }

        if let Some(last_dim) = trailing_feature_broadcast_dim(&lhs_shape, &rhs_shape, &out_shape) {
            return self.with_contiguous_data(|lhs_data| {
                rhs.with_contiguous_data(|rhs_data| {
                    let mut out = vec![0.0; checked_num_elements(&out_shape)?];
                    let rows = out.len() / last_dim.max(1);
                    if lhs_shape == out_shape {
                        if should_parallelize(out.len()) {
                            out.par_chunks_mut(last_dim)
                                .enumerate()
                                .for_each(|(row, out_row)| {
                                    let lhs_row = &lhs_data[row * last_dim..(row + 1) * last_dim];
                                    for col in 0..last_dim {
                                        out_row[col] = f(lhs_row[col], rhs_data[col]);
                                    }
                                });
                        } else {
                            for row in 0..rows {
                                let lhs_row = &lhs_data[row * last_dim..(row + 1) * last_dim];
                                let out_row = &mut out[row * last_dim..(row + 1) * last_dim];
                                for col in 0..last_dim {
                                    out_row[col] = f(lhs_row[col], rhs_data[col]);
                                }
                            }
                        }
                    } else if rhs_shape == out_shape {
                        if should_parallelize(out.len()) {
                            out.par_chunks_mut(last_dim)
                                .enumerate()
                                .for_each(|(row, out_row)| {
                                    let rhs_row = &rhs_data[row * last_dim..(row + 1) * last_dim];
                                    for col in 0..last_dim {
                                        out_row[col] = f(lhs_data[col], rhs_row[col]);
                                    }
                                });
                        } else {
                            for row in 0..rows {
                                let rhs_row = &rhs_data[row * last_dim..(row + 1) * last_dim];
                                let out_row = &mut out[row * last_dim..(row + 1) * last_dim];
                                for col in 0..last_dim {
                                    out_row[col] = f(lhs_data[col], rhs_row[col]);
                                }
                            }
                        }
                    } else {
                        let lhs_strides = contiguous_strides(&lhs_shape);
                        let rhs_strides = contiguous_strides(&rhs_shape);
                        for_each_index(&out_shape, |coords, flat| {
                            let lhs_index =
                                broadcast_offset(coords, &lhs_shape, &lhs_strides, out_rank);
                            let rhs_index =
                                broadcast_offset(coords, &rhs_shape, &rhs_strides, out_rank);
                            out[flat] = f(lhs_data[lhs_index], rhs_data[rhs_index]);
                        });
                    }
                    Self::from_owned(out, &out_shape).map_err(|err| {
                        Error::message(format!("failed to build result for {op_name}: {err}"))
                    })
                })
            });
        }

        let lhs_data = self.to_vec()?;
        let rhs_data = rhs.to_vec()?;
        let mut out = vec![0.0; checked_num_elements(&out_shape)?];
        let lhs_strides = contiguous_strides(&lhs_shape);
        let rhs_strides = contiguous_strides(&rhs_shape);
        for_each_index(&out_shape, |coords, flat| {
            let lhs_index = broadcast_offset(coords, &lhs_shape, &lhs_strides, out_rank);
            let rhs_index = broadcast_offset(coords, &rhs_shape, &rhs_strides, out_rank);
            out[flat] = f(lhs_data[lhs_index], rhs_data[rhs_index]);
        });

        Self::from_owned(out, &out_shape)
            .map_err(|err| Error::message(format!("failed to build result for {op_name}: {err}")))
    }

    pub(super) fn add(self, rhs: &Self) -> Result<Self> {
        self.binary_op(rhs, "add", |lhs, rhs| lhs + rhs)
    }

    pub(super) fn mul(self, rhs: &Self) -> Result<Self> {
        self.binary_op(rhs, "mul", |lhs, rhs| lhs * rhs)
    }

    pub(super) fn scale(self, s: f32) -> Result<Self> {
        self.unary_op("scale", |value| value * s)
    }

    pub(super) fn sigmoid(self) -> Result<Self> {
        self.unary_op("sigmoid", |value| 1.0 / (1.0 + (-value).exp()))
    }

    pub(super) fn split_last_dim_two_gelu_mul(self) -> Result<Self> {
        let _profile = op_scope_with("cpu.split_last_dim_two_gelu_mul", || {
            format!(
                "shape={:?} contiguous={}",
                self.shape(),
                self.is_contiguous()
            )
        });
        let shape = self.shape().to_vec();
        let axis = shape
            .len()
            .checked_sub(1)
            .ok_or_else(|| invalid_arg("split_last_dim_two_gelu_mul requires rank >= 1"))?;
        let dim = shape[axis];
        if dim % 2 != 0 {
            return Err(invalid_arg(format!(
                "split_last_dim_two_gelu_mul requires an even last dimension, got shape {:?}",
                shape
            )));
        }
        let half = dim / 2;
        let rows = plain_num_elements(&shape[..axis]);

        self.with_contiguous_data(|input| {
            let mut out = vec![0.0; rows * half];
            if should_parallelize(out.len()) && half > 0 {
                out.par_chunks_mut(half)
                    .enumerate()
                    .for_each(|(row, out_row)| {
                        let row_start = row * dim;
                        let lhs = &input[row_start..row_start + half];
                        let rhs = &input[row_start + half..row_start + dim];
                        for col in 0..half {
                            let value = lhs[col];
                            out_row[col] =
                                0.5 * value * (1.0 + erf_approx(value / SQRT_2)) * rhs[col];
                        }
                    });
            } else {
                for row in 0..rows {
                    let row_start = row * dim;
                    let lhs = &input[row_start..row_start + half];
                    let rhs = &input[row_start + half..row_start + dim];
                    let out_row = &mut out[row * half..(row + 1) * half];
                    for col in 0..half {
                        let value = lhs[col];
                        out_row[col] = 0.5 * value * (1.0 + erf_approx(value / SQRT_2)) * rhs[col];
                    }
                }
            }

            let mut out_shape = shape;
            out_shape[axis] = half;
            Self::from_owned(out, &out_shape)
        })
    }

    pub(super) fn gelu(self) -> Result<Self> {
        self.unary_op("gelu", |value| {
            0.5 * value * (1.0 + erf_approx(value / SQRT_2))
        })
    }

    pub(super) fn softmax(self, axis: isize) -> Result<Self> {
        let _profile = op_scope_with("cpu.softmax", || {
            format!(
                "shape={:?} axis={} contiguous={}",
                self.shape(),
                axis,
                self.is_contiguous()
            )
        });
        if self.shape().is_empty() {
            return Err(invalid_arg(
                "softmax expects a tensor with at least one dimension",
            ));
        }

        let shape = self.shape().to_vec();
        let axis = normalize_axis(axis, shape.len(), "softmax")?;
        let mut data = self.to_vec()?;
        let axis_len = shape[axis];
        if axis_len == 0 {
            return Self::from_owned(data, &shape);
        }

        let outer = plain_num_elements(&shape[..axis]);
        let inner = plain_num_elements(&shape[axis + 1..]);
        if outer == 0 || inner == 0 {
            return Self::from_owned(data, &shape);
        }

        let outer_block = axis_len * inner;
        if should_parallelize(data.len()) {
            data.par_chunks_mut(outer_block).for_each(|outer_chunk| {
                for inner_index in 0..inner {
                    let mut max_value = f32::NEG_INFINITY;
                    for axis_index in 0..axis_len {
                        let value = outer_chunk[axis_index * inner + inner_index];
                        if value > max_value {
                            max_value = value;
                        }
                    }

                    let mut sum = 0.0;
                    for axis_index in 0..axis_len {
                        let index = axis_index * inner + inner_index;
                        let value = (outer_chunk[index] - max_value).exp();
                        outer_chunk[index] = value;
                        sum += value;
                    }

                    for axis_index in 0..axis_len {
                        let index = axis_index * inner + inner_index;
                        outer_chunk[index] /= sum;
                    }
                }
            });
        } else {
            for outer_index in 0..outer {
                for inner_index in 0..inner {
                    let base = outer_index * outer_block + inner_index;
                    let mut max_value = f32::NEG_INFINITY;
                    for axis_index in 0..axis_len {
                        let value = data[base + axis_index * inner];
                        if value > max_value {
                            max_value = value;
                        }
                    }

                    let mut sum = 0.0;
                    for axis_index in 0..axis_len {
                        let index = base + axis_index * inner;
                        let value = (data[index] - max_value).exp();
                        data[index] = value;
                        sum += value;
                    }

                    for axis_index in 0..axis_len {
                        data[base + axis_index * inner] /= sum;
                    }
                }
            }
        }

        Self::from_owned(data, &shape)
    }
}

pub(super) fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t)
            * (-x * x).exp();
    sign * y
}

pub(super) fn apply_softmax_inplace(row_scores: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &s in row_scores.iter() {
        if s > max_val {
            max_val = s;
        }
    }
    let mut sum = 0.0;
    for s in row_scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    if sum > 0.0 {
        for s in row_scores.iter_mut() {
            *s /= sum;
        }
    }
}
