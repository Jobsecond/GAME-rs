use std::f32::consts::SQRT_2;
use std::sync::OnceLock;

use candle_core::{DType, Device, Storage, Tensor as CandleTensor};
use rayon::prelude::*;

use crate::profiler::op_scope_with;
use crate::{Error, Result};

use super::Tensor;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuDevice;

#[derive(Debug, Clone)]
pub struct CpuTensor {
    tensor: CandleTensor,
    device: CpuDevice,
}

impl CpuTensor {
    pub fn num_elements(&self) -> usize {
        self.tensor.elem_count()
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        Ok(self.tensor.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
    }

    fn from_tensor(tensor: CandleTensor) -> Self {
        Self {
            tensor,
            device: CpuDevice,
        }
    }

    fn from_owned(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        if shape.is_empty() {
            if data.len() != 1 {
                return Err(invalid_arg(format!(
                    "scalar tensor requires exactly one element, got {}",
                    data.len()
                )));
            }
            return Ok(Self::from_tensor(CandleTensor::new(data[0], &Device::Cpu)?));
        }
        Ok(Self::from_tensor(CandleTensor::from_vec(
            data,
            shape.to_vec(),
            &Device::Cpu,
        )?))
    }

    fn with_contiguous_data<R>(&self, f: impl FnOnce(&[f32]) -> Result<R>) -> Result<R> {
        let (storage, layout) = self.tensor.storage_and_layout();
        if let Storage::Cpu(storage) = &*storage
            && let Some((start, end)) = layout.contiguous_offsets()
        {
            let data = storage.as_slice::<f32>()?;
            return f(&data[start..end]);
        }

        let owned = self.to_vec()?;
        f(&owned)
    }

    fn validate_attention_layout_input(
        &self,
        op_name: &str,
        parts: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(usize, usize)> {
        if self.shape().len() != 2 {
            return Err(invalid_arg(format!(
                "{op_name} expects [seq_len, dim], got {:?}",
                self.shape()
            )));
        }

        let seq_len = self.shape()[0];
        let part_dim = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| invalid_arg("attention projection dimension overflow"))?;
        let expected = part_dim
            .checked_mul(parts)
            .ok_or_else(|| invalid_arg(format!("{op_name} dimension overflow")))?;
        if self.shape()[1] != expected {
            return Err(invalid_arg(format!(
                "{op_name} expected last dim {}, got {:?}",
                expected,
                self.shape()
            )));
        }

        Ok((seq_len, part_dim))
    }

    fn split_last_dim_parts_for_attention_heads(
        self,
        parts: usize,
        num_heads: usize,
        head_dim: usize,
        op_name: &'static str,
    ) -> Result<Vec<Self>> {
        let _profile = op_scope_with(op_name, || {
            format!(
                "shape={:?} parts={} num_heads={} head_dim={} contiguous={}",
                self.shape(),
                parts,
                num_heads,
                head_dim,
                self.tensor.is_contiguous()
            )
        });
        let (seq_len, part_dim) =
            self.validate_attention_layout_input(op_name, parts, num_heads, head_dim)?;
        let full_dim = part_dim * parts;
        self.with_contiguous_data(|input| {
            let mut outputs = (0..parts)
                .map(|_| vec![0.0; num_heads * seq_len * head_dim])
                .collect::<Vec<_>>();
            let head_block = seq_len * head_dim;

            if should_parallelize(input.len()) && head_block > 0 {
                outputs
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(part, out_part)| {
                        for head in 0..num_heads {
                            let dst_head =
                                &mut out_part[head * head_block..(head + 1) * head_block];
                            let src_head_offset = part * part_dim + head * head_dim;
                            for token in 0..seq_len {
                                let src_start = token * full_dim + src_head_offset;
                                let src_end = src_start + head_dim;
                                let dst_start = token * head_dim;
                                let dst_end = dst_start + head_dim;
                                dst_head[dst_start..dst_end]
                                    .copy_from_slice(&input[src_start..src_end]);
                            }
                        }
                    });
            } else {
                for part in 0..parts {
                    for head in 0..num_heads {
                        for token in 0..seq_len {
                            let src_start = token * full_dim + part * part_dim + head * head_dim;
                            let src_end = src_start + head_dim;
                            let dst_start = (head * seq_len + token) * head_dim;
                            let dst_end = dst_start + head_dim;
                            outputs[part][dst_start..dst_end]
                                .copy_from_slice(&input[src_start..src_end]);
                        }
                    }
                }
            }

            outputs
                .into_iter()
                .map(|data| Self::from_owned(data, &[num_heads, seq_len, head_dim]))
                .collect()
        })
    }

    fn unary_op(self, op_name: &str, f: impl Fn(f32) -> f32 + Send + Sync) -> Result<Self> {
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

    fn binary_op(
        self,
        rhs: &Self,
        op_name: &str,
        f: impl Fn(f32, f32) -> f32 + Send + Sync,
    ) -> Result<Self> {
        let _profile = op_scope_with("cpu.binary_op", || {
            format!("op={} lhs={:?} rhs={:?}", op_name, self.shape(), rhs.shape())
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
                                    let lhs_row =
                                        &lhs_data[row * last_dim..(row + 1) * last_dim];
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
                                    let rhs_row =
                                        &rhs_data[row * last_dim..(row + 1) * last_dim];
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
}

impl Tensor for CpuTensor {
    type Device = CpuDevice;

    fn from_data(data: &[f32], shape: &[usize], _device: &Self::Device) -> Result<Self> {
        Self::from_owned(data.to_vec(), shape)
    }

    fn zeros(shape: &[usize], _device: &Self::Device) -> Result<Self> {
        Ok(Self::from_tensor(CandleTensor::zeros(
            shape.to_vec(),
            DType::F32,
            &Device::Cpu,
        )?))
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn shape(&self) -> &[usize] {
        self.tensor.dims()
    }

    fn export(&self, buf: &mut [f32]) -> Result<()> {
        let values = self.to_vec()?;
        if buf.len() != values.len() {
            return Err(invalid_arg(format!(
                "export buffer length {} does not match tensor shape {:?} ({} elements)",
                buf.len(),
                self.shape(),
                values.len()
            )));
        }
        buf.copy_from_slice(&values);
        Ok(())
    }

    fn reshape(self, shape: &[usize]) -> Result<Self> {
        Ok(Self::from_tensor(self.tensor.reshape(shape.to_vec())?))
    }

    fn transpose(self, dim0: usize, dim1: usize) -> Result<Self> {
        Ok(Self::from_tensor(self.tensor.transpose(dim0, dim1)?))
    }

    fn contiguous(self) -> Result<Self> {
        Ok(Self::from_tensor(self.tensor.contiguous()?))
    }

    fn slice(self, axis: usize, start: usize, end: usize) -> Result<Self> {
        validate_axis(axis, self.shape().len(), "slice")?;
        if end < start {
            return Err(invalid_arg(format!(
                "slice end {} is less than start {}",
                end, start
            )));
        }
        Ok(Self::from_tensor(
            self.tensor.narrow(axis, start, end - start)?,
        ))
    }

    fn layout_for_attention_heads(self, num_heads: usize, head_dim: usize) -> Result<Self> {
        let _profile = op_scope_with("cpu.layout_for_attention_heads", || {
            format!(
                "shape={:?} num_heads={} head_dim={} contiguous={}",
                self.shape(),
                num_heads,
                head_dim,
                self.tensor.is_contiguous()
            )
        });
        if self.shape().len() != 2 {
            return Err(invalid_arg(format!(
                "layout_for_attention_heads expects [seq_len, dim], got {:?}",
                self.shape()
            )));
        }

        let seq_len = self.shape()[0];
        let dim = self.shape()[1];
        let expected = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| invalid_arg("attention projection dimension overflow"))?;
        if dim != expected {
            return Err(invalid_arg(format!(
                "layout_for_attention_heads expected last dim {}, got {:?}",
                expected,
                self.shape()
            )));
        }

        self.with_contiguous_data(|input| {
            let mut out = vec![0.0; input.len()];
            let head_block = seq_len * head_dim;
            if should_parallelize(out.len()) && head_block > 0 {
                out.par_chunks_mut(head_block)
                    .enumerate()
                    .for_each(|(head, out_head)| {
                        for token in 0..seq_len {
                            let src_start = token * dim + head * head_dim;
                            let src_end = src_start + head_dim;
                            let dst_start = token * head_dim;
                            let dst_end = dst_start + head_dim;
                            out_head[dst_start..dst_end]
                                .copy_from_slice(&input[src_start..src_end]);
                        }
                    });
            } else {
                for head in 0..num_heads {
                    for token in 0..seq_len {
                        let src_start = token * dim + head * head_dim;
                        let src_end = src_start + head_dim;
                        let dst_start = (head * seq_len + token) * head_dim;
                        let dst_end = dst_start + head_dim;
                        out[dst_start..dst_end].copy_from_slice(&input[src_start..src_end]);
                    }
                }
            }

            Self::from_owned(out, &[num_heads, seq_len, head_dim])
        })
    }

    fn split_last_dim_two_for_attention_heads(
        self,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(Self, Self)> {
        let mut parts = self.split_last_dim_parts_for_attention_heads(
            2,
            num_heads,
            head_dim,
            "cpu.split_last_dim_two_for_attention_heads",
        )?;
        let second = parts
            .pop()
            .ok_or_else(|| invalid_arg("split_last_dim_two_for_attention_heads missing second part"))?;
        let first = parts
            .pop()
            .ok_or_else(|| invalid_arg("split_last_dim_two_for_attention_heads missing first part"))?;
        Ok((first, second))
    }

    fn split_last_dim_three_for_attention_heads(
        self,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(Self, Self, Self)> {
        let mut parts = self.split_last_dim_parts_for_attention_heads(
            3,
            num_heads,
            head_dim,
            "cpu.split_last_dim_three_for_attention_heads",
        )?;
        let third = parts
            .pop()
            .ok_or_else(|| invalid_arg("split_last_dim_three_for_attention_heads missing third part"))?;
        let second = parts
            .pop()
            .ok_or_else(|| invalid_arg("split_last_dim_three_for_attention_heads missing second part"))?;
        let first = parts
            .pop()
            .ok_or_else(|| invalid_arg("split_last_dim_three_for_attention_heads missing first part"))?;
        Ok((first, second, third))
    }

    fn merge_attention_heads(self) -> Result<Self> {
        let _profile = op_scope_with("cpu.merge_attention_heads", || {
            format!(
                "shape={:?} contiguous={}",
                self.shape(),
                self.tensor.is_contiguous()
            )
        });
        if self.shape().len() != 3 {
            return Err(invalid_arg(format!(
                "merge_attention_heads expects [num_heads, seq_len, head_dim], got {:?}",
                self.shape()
            )));
        }

        let num_heads = self.shape()[0];
        let seq_len = self.shape()[1];
        let head_dim = self.shape()[2];
        let merged_dim = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| invalid_arg("merge_attention_heads dimension overflow"))?;

        self.with_contiguous_data(|input| {
            let mut out = vec![0.0; seq_len * merged_dim];
            if should_parallelize(out.len()) && merged_dim > 0 {
                out.par_chunks_mut(merged_dim)
                    .enumerate()
                    .for_each(|(token, out_row)| {
                        for head in 0..num_heads {
                            let src_start = (head * seq_len + token) * head_dim;
                            let src_end = src_start + head_dim;
                            let dst_start = head * head_dim;
                            let dst_end = dst_start + head_dim;
                            out_row[dst_start..dst_end]
                                .copy_from_slice(&input[src_start..src_end]);
                        }
                    });
            } else {
                for token in 0..seq_len {
                    for head in 0..num_heads {
                        let src_start = (head * seq_len + token) * head_dim;
                        let src_end = src_start + head_dim;
                        let dst_start = token * merged_dim + head * head_dim;
                        let dst_end = dst_start + head_dim;
                        out[dst_start..dst_end].copy_from_slice(&input[src_start..src_end]);
                    }
                }
            }

            Self::from_owned(out, &[seq_len, merged_dim])
        })
    }

    fn concat(parts: &[&Self], axis: usize) -> Result<Self> {
        let _profile = op_scope_with("cpu.concat", || {
            format!(
                "parts={} axis={} first_shape={:?}",
                parts.len(),
                axis,
                parts.first().map(|part| part.shape().to_vec()).unwrap_or_default()
            )
        });
        let first = parts
            .first()
            .ok_or_else(|| invalid_arg("concat requires at least one tensor"))?;
        let rank = first.shape().len();
        validate_axis(axis, rank, "concat")?;
        let expected_shape = first.shape().to_vec();
        let mut out_shape = expected_shape.clone();
        out_shape[axis] = 0;

        for part in parts {
            if part.shape().len() != rank {
                return Err(invalid_arg(format!(
                    "concat rank mismatch: expected rank {}, got shape {:?}",
                    rank,
                    part.shape()
                )));
            }
            for dim in 0..rank {
                if dim != axis && part.shape()[dim] != out_shape[dim] {
                    return Err(invalid_arg(format!(
                        "concat shape mismatch on axis {}: expected non-concat dims {:?}, got {:?}",
                        axis,
                        expected_shape,
                        part.shape()
                    )));
                }
            }
            out_shape[axis] += part.shape()[axis];
        }

        let out_len = checked_num_elements(&out_shape)?;
        let mut out = vec![0.0; out_len];
        let outer = plain_num_elements(&out_shape[..axis]);
        let inner = plain_num_elements(&out_shape[axis + 1..]);
        let out_axis_span = out_shape[axis] * inner;
        let mut axis_offset = 0usize;

        for part in parts {
            let part_block = part.shape()[axis] * inner;
            part.with_contiguous_data(|data| {
                for outer_index in 0..outer {
                    let dst_start = outer_index * out_axis_span + axis_offset * inner;
                    let dst_end = dst_start + part_block;
                    let src_start = outer_index * part_block;
                    let src_end = src_start + part_block;
                    out[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
                }
                Ok(())
            })?;
            axis_offset += part.shape()[axis];
        }

        Self::from_owned(out, &out_shape)
    }

    fn add(self, rhs: &Self) -> Result<Self> {
        self.binary_op(rhs, "add", |lhs, rhs| lhs + rhs)
    }

    fn mul(self, rhs: &Self) -> Result<Self> {
        self.binary_op(rhs, "mul", |lhs, rhs| lhs * rhs)
    }

    fn scale(self, s: f32) -> Result<Self> {
        self.unary_op("scale", |value| value * s)
    }

    fn sigmoid(self) -> Result<Self> {
        self.unary_op("sigmoid", |value| 1.0 / (1.0 + (-value).exp()))
    }

    fn split_last_dim_two_gelu_mul(self) -> Result<Self> {
        let _profile = op_scope_with("cpu.split_last_dim_two_gelu_mul", || {
            format!("shape={:?} contiguous={}", self.shape(), self.tensor.is_contiguous())
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
                        out_row[col] =
                            0.5 * value * (1.0 + erf_approx(value / SQRT_2)) * rhs[col];
                    }
                }
            }

            let mut out_shape = shape;
            out_shape[axis] = half;
            Self::from_owned(out, &out_shape)
        })
    }

    fn matmul(&self, rhs: &Self) -> Result<Self> {
        Ok(Self::from_tensor(self.tensor.matmul(&rhs.tensor)?))
    }

    fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self> {
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
                                matrixmultiply::sgemm(
                                    chunk_rows,
                                    in_dim,
                                    out_dim,
                                    1.0,
                                    input.as_ptr().add(row_start * in_dim),
                                    in_dim as isize,
                                    1,
                                    weight_data.as_ptr(),
                                    1,
                                    in_dim as isize,
                                    0.0,
                                    out_chunk.as_mut_ptr(),
                                    out_dim as isize,
                                    1,
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
                        matrixmultiply::sgemm(
                            rows,
                            in_dim,
                            out_dim,
                            1.0,
                            input.as_ptr(),
                            in_dim as isize,
                            1,
                            weight_data.as_ptr(),
                            1,
                            in_dim as isize,
                            0.0,
                            out.as_mut_ptr(),
                            out_dim as isize,
                            1,
                        );
                    }

                    if let Some(bias_values) = &bias_data {
                        if should_parallelize(out.len()) && out_dim > 0 {
                            out.par_chunks_mut(out_dim).for_each(|out_row| {
                                for (out_idx, value) in out_row.iter_mut().enumerate() {
                                    *value += bias_values[out_idx];
                                }
                            });
                        } else {
                            for row in 0..rows {
                                for out_idx in 0..out_dim {
                                    out[row * out_dim + out_idx] += bias_values[out_idx];
                                }
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

    fn attention_score_softmax(
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
        let (q_storage, q_layout) = q.tensor.storage_and_layout();
        let q_data = match &*q_storage {
            Storage::Cpu(storage) => storage.as_slice::<f32>()?,
            _ => return Err(invalid_arg("CpuTensor expected CPU storage for q")),
        };
        let q_ptr = q_data[q_layout.start_offset()..].as_ptr() as usize;
        let q_batch_stride = q_layout.stride()[0];
        let q_rs = q_layout.stride()[1] as isize;
        let q_cs = q_layout.stride()[2] as isize;

        let (k_storage, k_layout) = k_t.tensor.storage_and_layout();
        let k_data = match &*k_storage {
            Storage::Cpu(storage) => storage.as_slice::<f32>()?,
            _ => return Err(invalid_arg("CpuTensor expected CPU storage for k_t")),
        };
        let k_ptr = k_data[k_layout.start_offset()..].as_ptr() as usize;
        let k_batch_stride = k_layout.stride()[0];
        let k_rs = k_layout.stride()[1] as isize;
        let k_cs = k_layout.stride()[2] as isize;

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

        let (mask_guard, mask_owned, mask_layout_owned) = if let Some(mask) = mask {
            let (storage, layout) = mask.tensor.storage_and_layout();
            let owned = if layout.contiguous_offsets().is_none() {
                Some(mask.to_vec()?)
            } else {
                None
            };
            (Some(storage), owned, Some(layout.clone()))
        } else {
            (None, None, None)
        };
        let mask_data: Option<&[f32]> = if let (Some(storage), Some(layout)) =
            (mask_guard.as_ref(), mask_layout_owned.as_ref())
        {
            if let Some((start, end)) = layout.contiguous_offsets() {
                let data = match &**storage {
                    Storage::Cpu(storage) => storage.as_slice::<f32>()?,
                    _ => return Err(invalid_arg("CpuTensor expected CPU storage for mask")),
                };
                Some(&data[start..end])
            } else {
                mask_owned.as_deref()
            }
        } else {
            None
        };
        let mask_outer_stride = mask_layout_owned.as_ref().map(|layout| match layout.dims().len() {
            2 => layout.stride()[0],
            3 => layout.stride()[1],
            _ => 0,
        });
        let mask_head_stride = mask_layout_owned.as_ref().and_then(|layout| {
            if layout.dims().len() == 3 {
                Some(layout.stride()[0])
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
                        matrixmultiply::sgemm(
                            query_len,
                            head_dim,
                            key_len,
                            scale,
                            lhs_ptr,
                            q_rs,
                            q_cs,
                            rhs_ptr,
                            k_rs,
                            k_cs,
                            0.0,
                            out_head.as_mut_ptr(),
                            key_len as isize,
                            1,
                        );
                    }
                });
        } else {
            for head in 0..heads {
                let out_head = &mut out[head * query_len * key_len..(head + 1) * query_len * key_len];
                let lhs_ptr =
                    (q_ptr + head * q_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                let rhs_ptr =
                    (k_ptr + head * k_batch_stride * std::mem::size_of::<f32>()) as *const f32;
                unsafe {
                    matrixmultiply::sgemm(
                        query_len,
                        head_dim,
                        key_len,
                        scale,
                        lhs_ptr,
                        q_rs,
                        q_cs,
                        rhs_ptr,
                        k_rs,
                        k_cs,
                        0.0,
                        out_head.as_mut_ptr(),
                        key_len as isize,
                        1,
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
                        mask_data.as_deref(),
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

    fn attention_value_matmul(probs: &Self, v: &Self) -> Result<Self> {
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
        let (probs_storage, probs_layout) = probs.tensor.storage_and_layout();
        let probs_data = match &*probs_storage {
            Storage::Cpu(storage) => storage.as_slice::<f32>()?,
            _ => return Err(invalid_arg("CpuTensor expected CPU storage for probs")),
        };
        let probs_ptr = probs_data[probs_layout.start_offset()..].as_ptr() as usize;
        let probs_batch_stride = probs_layout.stride()[0];
        let probs_rs = probs_layout.stride()[1] as isize;
        let probs_cs = probs_layout.stride()[2] as isize;
        let probs_row_stride = probs_layout.stride()[1];

        let (v_storage, v_layout) = v.tensor.storage_and_layout();
        let v_data = match &*v_storage {
            Storage::Cpu(storage) => storage.as_slice::<f32>()?,
            _ => return Err(invalid_arg("CpuTensor expected CPU storage for value tensor")),
        };
        let v_ptr = v_data[v_layout.start_offset()..].as_ptr() as usize;
        let v_batch_stride = v_layout.stride()[0];
        let v_rs = v_layout.stride()[1] as isize;
        let v_cs = v_layout.stride()[2] as isize;

        let mut out = vec![0.0; heads * query_len * head_dim];
        let head_block = query_len * head_dim;
        let row_chunk_len =
            choose_parallel_attention_row_chunk_len(query_len, key_len, head_dim);

        if should_parallelize_attention_matmul(heads, query_len, key_len, head_dim)
            && row_chunk_len < query_len
            && head_block > 0
        {
            out.par_chunks_mut(head_block)
                .enumerate()
                .for_each(|(head, out_head)| {
                    let lhs_base =
                        probs_ptr + head * probs_batch_stride * std::mem::size_of::<f32>();
                    let rhs_base =
                        v_ptr + head * v_batch_stride * std::mem::size_of::<f32>();
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
                                matrixmultiply::sgemm(
                                    chunk_rows,
                                    key_len,
                                    head_dim,
                                    1.0,
                                    lhs_ptr,
                                    probs_rs,
                                    probs_cs,
                                    rhs_ptr,
                                    v_rs,
                                    v_cs,
                                    0.0,
                                    out_chunk.as_mut_ptr(),
                                    head_dim as isize,
                                    1,
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
                    let rhs_ptr = (v_ptr
                        + head * v_batch_stride * std::mem::size_of::<f32>())
                        as *const f32;
                    unsafe {
                        matrixmultiply::sgemm(
                            query_len,
                            key_len,
                            head_dim,
                            1.0,
                            lhs_ptr,
                            probs_rs,
                            probs_cs,
                            rhs_ptr,
                            v_rs,
                            v_cs,
                            0.0,
                            out_head.as_mut_ptr(),
                            head_dim as isize,
                            1,
                        );
                    }
                });
        } else {
            for head in 0..heads {
                let out_head = &mut out[head * query_len * head_dim..(head + 1) * query_len * head_dim];
                let lhs_ptr = (probs_ptr
                    + head * probs_batch_stride * std::mem::size_of::<f32>())
                    as *const f32;
                let rhs_ptr = (v_ptr
                    + head * v_batch_stride * std::mem::size_of::<f32>())
                    as *const f32;
                unsafe {
                    matrixmultiply::sgemm(
                        query_len,
                        key_len,
                        head_dim,
                        1.0,
                        lhs_ptr,
                        probs_rs,
                        probs_cs,
                        rhs_ptr,
                        v_rs,
                        v_cs,
                        0.0,
                        out_head.as_mut_ptr(),
                        head_dim as isize,
                        1,
                    );
                }
            }
        }

        Self::from_owned(out, &[heads, query_len, head_dim])
    }

    fn rms_norm(self, weight: &Self, eps: f32) -> Result<Self> {
        let _profile = op_scope_with("cpu.rms_norm", || {
            format!(
                "input={:?} weight={:?} contiguous_input={} contiguous_weight={}",
                self.shape(),
                weight.shape(),
                self.tensor.is_contiguous(),
                weight.tensor.is_contiguous()
            )
        });
        if self.shape().is_empty() {
            return Err(invalid_arg(
                "rms_norm expects an input tensor with at least one dimension",
            ));
        }

        let feature_dim = *self.shape().last().unwrap_or(&0);
        if weight.shape() != [feature_dim] {
            return Err(invalid_arg(format!(
                "rms_norm weight must have shape [{feature_dim}], got {:?}",
                weight.shape()
            )));
        }

        let shape = self.shape().to_vec();
        self.with_contiguous_data(|input| {
            weight.with_contiguous_data(|weight_data| {
                let mut data = input.to_vec();
                if feature_dim == 0 {
                    return Self::from_owned(data, &shape);
                }

                if should_parallelize(data.len()) {
                    data.par_chunks_mut(feature_dim).for_each(|row_slice| {
                        let mean_square = row_slice.iter().map(|value| value * value).sum::<f32>()
                            / feature_dim as f32;
                        let inv_rms = 1.0 / (mean_square + eps).sqrt();
                        for (value, scale) in row_slice.iter_mut().zip(weight_data.iter()) {
                            *value *= inv_rms * scale;
                        }
                    });
                } else {
                    let rows = data.len() / feature_dim;
                    for row in 0..rows {
                        let row_start = row * feature_dim;
                        let row_end = row_start + feature_dim;
                        let row_slice = &mut data[row_start..row_end];
                        let mean_square = row_slice.iter().map(|value| value * value).sum::<f32>()
                            / feature_dim as f32;
                        let inv_rms = 1.0 / (mean_square + eps).sqrt();
                        for (value, scale) in row_slice.iter_mut().zip(weight_data.iter()) {
                            *value *= inv_rms * scale;
                        }
                    }
                }

                Self::from_owned(data, &shape)
            })
        })
    }

    fn gelu(self) -> Result<Self> {
        self.unary_op("gelu", |value| 0.5 * value * (1.0 + erf_approx(value / SQRT_2)))
    }

    fn softmax(self, axis: isize) -> Result<Self> {
        let _profile = op_scope_with("cpu.softmax", || {
            format!(
                "shape={:?} axis={} contiguous={}",
                self.shape(),
                axis,
                self.tensor.is_contiguous()
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

    fn rope(
        self,
        positions: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self> {
        let _profile = op_scope_with("cpu.rope", || {
            format!(
                "shape={:?} positions={} head_dim={} num_heads={} rope_dims={} contiguous={}",
                self.shape(),
                positions.len(),
                head_dim,
                num_heads,
                rope_dims,
                self.tensor.is_contiguous()
            )
        });
        let shape = self.shape().to_vec();
        validate_rope_shape(&shape, positions.len(), head_dim, num_heads, "rope")?;
        let rope_dims = normalize_rope_dims(head_dim, rope_dims, "rope", false)?;
        let mut data = self.to_vec()?;
        let seq_len = shape[1];

        let head_block = seq_len * head_dim;
        if should_parallelize(data.len()) && head_block > 0 {
            data.par_chunks_mut(head_block).for_each(|head_slice| {
                for (token, &position) in positions.iter().enumerate() {
                    let base = token * head_dim;
                    apply_rope_chunk(
                        &mut head_slice[base..base + head_dim],
                        0,
                        rope_dims,
                        position as f32,
                        theta,
                    );
                }
            });
        } else {
            for head in 0..num_heads {
                for (token, &position) in positions.iter().enumerate() {
                    let base = (head * seq_len + token) * head_dim;
                    apply_rope_chunk(
                        &mut data[base..base + head_dim],
                        0,
                        rope_dims,
                        position as f32,
                        theta,
                    );
                }
            }
        }

        Self::from_owned(data, &shape)
    }

    fn region_rope(
        self,
        global_pos: &[i32],
        region_ids: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self> {
        let _profile = op_scope_with("cpu.region_rope", || {
            format!(
                "shape={:?} tokens={} head_dim={} num_heads={} rope_dims={} contiguous={}",
                self.shape(),
                global_pos.len(),
                head_dim,
                num_heads,
                rope_dims,
                self.tensor.is_contiguous()
            )
        });
        let shape = self.shape().to_vec();
        validate_rope_shape(&shape, global_pos.len(), head_dim, num_heads, "region_rope")?;
        if region_ids.len() != global_pos.len() {
            return Err(invalid_arg(format!(
                "region_rope expected {} region ids, got {}",
                global_pos.len(),
                region_ids.len()
            )));
        }
        let mixed_dims = normalize_rope_dims(head_dim, rope_dims, "region_rope", true)?;
        let half = mixed_dims / 2;
        let seq_len = shape[1];
        let mut data = self.to_vec()?;

        let head_block = seq_len * head_dim;
        if should_parallelize(data.len()) && head_block > 0 {
            data.par_chunks_mut(head_block).for_each(|head_slice| {
                for token in 0..seq_len {
                    let base = token * head_dim;
                    let values = &mut head_slice[base..base + head_dim];
                    apply_rope_chunk(values, 0, half, global_pos[token] as f32, theta);
                    apply_rope_chunk(values, half, half, region_ids[token] as f32, theta);
                }
            });
        } else {
            for head in 0..num_heads {
                for token in 0..seq_len {
                    let base = (head * seq_len + token) * head_dim;
                    let values = &mut data[base..base + head_dim];
                    apply_rope_chunk(values, 0, half, global_pos[token] as f32, theta);
                    apply_rope_chunk(values, half, half, region_ids[token] as f32, theta);
                }
            }
        }

        Self::from_owned(data, &shape)
    }

    fn conv1d_dw(
        self,
        kernel: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let _profile = op_scope_with("cpu.conv1d_dw", || {
            format!(
                "input={:?} kernel={:?} bias={} stride={} padding={} contiguous_input={} contiguous_kernel={}",
                self.shape(),
                kernel.shape(),
                bias.is_some(),
                stride,
                padding,
                self.tensor.is_contiguous(),
                kernel.tensor.is_contiguous()
            )
        });
        if stride == 0 {
            return Err(invalid_arg("conv1d_dw requires stride > 0"));
        }
        if self.shape().len() != 2 {
            return Err(invalid_arg(format!(
                "conv1d_dw expects input shape [time, channels], got {:?}",
                self.shape()
            )));
        }
        if kernel.shape().len() != 2 {
            return Err(invalid_arg(format!(
                "conv1d_dw kernel must have shape [channels, kernel_size], got {:?}",
                kernel.shape()
            )));
        }

        let (time, channels) = (self.shape()[0], self.shape()[1]);
        let (kernel_channels, kernel_size) = (kernel.shape()[0], kernel.shape()[1]);
        if channels != kernel_channels {
            return Err(invalid_arg(format!(
                "conv1d_dw channel mismatch: input {:?}, kernel {:?}",
                self.shape(),
                kernel.shape()
            )));
        }
        if kernel_size == 0 {
            return Err(invalid_arg(
                "conv1d_dw kernel size must be greater than zero",
            ));
        }
        if let Some(bias) = bias
            && bias.shape() != [channels]
        {
            return Err(invalid_arg(format!(
                "conv1d_dw bias must have shape [{channels}], got {:?}",
                bias.shape()
            )));
        }

        let padded = time.saturating_add(padding.saturating_mul(2));
        let out_time = if padded < kernel_size {
            0
        } else {
            (padded - kernel_size) / stride + 1
        };

        let bias_data = bias.map(CpuTensor::to_vec).transpose()?;
        self.with_contiguous_data(|input| {
            kernel.with_contiguous_data(|kernel_data| {
                let mut out = vec![0.0; out_time * channels];

                if should_parallelize(out.len()) && channels > 0 {
                    out.par_chunks_mut(channels)
                        .enumerate()
                        .for_each(|(out_t, out_row)| {
                            for (channel, value) in out_row.iter_mut().enumerate() {
                                let mut sum = bias_data
                                    .as_ref()
                                    .map_or(0.0, |bias_values| bias_values[channel]);
                                for kernel_index in 0..kernel_size {
                                    let input_index = out_t * stride + kernel_index;
                                    if input_index < padding {
                                        continue;
                                    }
                                    let input_t = input_index - padding;
                                    if input_t >= time {
                                        continue;
                                    }
                                    sum += input[input_t * channels + channel]
                                        * kernel_data[channel * kernel_size + kernel_index];
                                }
                                *value = sum;
                            }
                        });
                } else {
                    for out_t in 0..out_time {
                        for channel in 0..channels {
                            let mut sum = bias_data
                                .as_ref()
                                .map_or(0.0, |bias_values| bias_values[channel]);
                            for kernel_index in 0..kernel_size {
                                let input_index = out_t * stride + kernel_index;
                                if input_index < padding {
                                    continue;
                                }
                                let input_t = input_index - padding;
                                if input_t >= time {
                                    continue;
                                }
                                sum += input[input_t * channels + channel]
                                    * kernel_data[channel * kernel_size + kernel_index];
                            }
                            out[out_t * channels + channel] = sum;
                        }
                    }
                }

                Self::from_owned(out, &[out_time, channels])
            })
        })
    }

    fn embedding(table: &Self, indices: &[i32]) -> Result<Self> {
        if table.shape().len() != 2 {
            return Err(invalid_arg(format!(
                "embedding table must have shape [rows, dim], got {:?}",
                table.shape()
            )));
        }

        let rows = table.shape()[0];
        let dim = table.shape()[1];
        table.with_contiguous_data(|table_data| {
            let mut out = vec![0.0; indices.len() * dim];

            for (row_index, index) in indices.iter().copied().enumerate() {
                let source_row = usize::try_from(index)
                    .map_err(|_| invalid_arg(format!("embedding index {index} is negative")))?;
                if source_row >= rows {
                    return Err(invalid_arg(format!(
                        "embedding index {} is out of bounds for {} rows",
                        source_row, rows
                    )));
                }

                let src_start = source_row * dim;
                let src_end = src_start + dim;
                let dst_start = row_index * dim;
                let dst_end = dst_start + dim;
                out[dst_start..dst_end].copy_from_slice(&table_data[src_start..src_end]);
            }

            Self::from_owned(out, &[indices.len(), dim])
        })
    }

    fn repeat(self, axis: usize, n: usize) -> Result<Self> {
        let shape = self.shape().to_vec();
        validate_axis(axis, shape.len(), "repeat")?;
        let data = self.to_vec()?;

        let mut out_shape = shape.clone();
        out_shape[axis] = out_shape[axis]
            .checked_mul(n)
            .ok_or_else(|| invalid_arg("repeat axis size overflow"))?;
        let mut out = vec![0.0; checked_num_elements(&out_shape)?];

        let outer = plain_num_elements(&shape[..axis]);
        let inner = plain_num_elements(&shape[axis + 1..]);
        let axis_block = shape[axis] * inner;
        let out_axis_block = out_shape[axis] * inner;

        for outer_index in 0..outer {
            let src_start = outer_index * axis_block;
            let src_end = src_start + axis_block;
            let src = &data[src_start..src_end];
            for repeat_index in 0..n {
                let dst_start = outer_index * out_axis_block + repeat_index * axis_block;
                let dst_end = dst_start + axis_block;
                out[dst_start..dst_end].copy_from_slice(src);
            }
        }

        Self::from_owned(out, &out_shape)
    }
}

fn invalid_arg(message: impl Into<String>) -> Error {
    Error::message(message.into())
}

fn checked_num_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| invalid_arg(format!("tensor shape {:?} is too large", shape)))
    })
}

fn plain_num_elements(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride.saturating_mul(shape[axis]);
    }
    strides
}

fn validate_axis(axis: usize, rank: usize, op_name: &str) -> Result<()> {
    if axis >= rank {
        return Err(invalid_arg(format!(
            "{op_name} axis {} is out of bounds for rank {}",
            axis, rank
        )));
    }
    Ok(())
}

fn normalize_axis(axis: isize, rank: usize, op_name: &str) -> Result<usize> {
    if rank == 0 {
        return Err(invalid_arg(format!(
            "{op_name} requires a tensor with at least one dimension"
        )));
    }

    let rank_isize = isize::try_from(rank).map_err(|_| invalid_arg("rank overflow"))?;
    let normalized = if axis < 0 { rank_isize + axis } else { axis };
    if normalized < 0 || normalized >= rank_isize {
        return Err(invalid_arg(format!(
            "{op_name} axis {} is out of bounds for rank {}",
            axis, rank
        )));
    }

    usize::try_from(normalized).map_err(|_| invalid_arg("axis overflow"))
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut out = vec![1usize; rank];

    for axis in 0..rank {
        let lhs_dim = lhs
            .len()
            .checked_sub(rank - axis)
            .and_then(|index| lhs.get(index))
            .copied()
            .unwrap_or(1);
        let rhs_dim = rhs
            .len()
            .checked_sub(rank - axis)
            .and_then(|index| rhs.get(index))
            .copied()
            .unwrap_or(1);

        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Err(invalid_arg(format!(
                "cannot broadcast shapes {:?} and {:?}",
                lhs, rhs
            )));
        }
        out[axis] = lhs_dim.max(rhs_dim);
    }

    Ok(out)
}

fn broadcast_offset(coords: &[usize], shape: &[usize], strides: &[usize], out_rank: usize) -> usize {
    if shape.is_empty() {
        return 0;
    }

    let rank_diff = out_rank - shape.len();
    let mut offset = 0usize;
    for (out_axis, &coord) in coords.iter().enumerate() {
        if out_axis < rank_diff {
            continue;
        }
        let axis = out_axis - rank_diff;
        if shape[axis] != 1 {
            offset += coord * strides[axis];
        }
    }
    offset
}

fn trailing_feature_broadcast_dim(lhs: &[usize], rhs: &[usize], out: &[usize]) -> Option<usize> {
    if out.is_empty() {
        return None;
    }

    let feature_dim = *out.last()?;
    if feature_dim == 0 {
        return Some(0);
    }

    let lhs_feature = *lhs.last().unwrap_or(&1);
    let rhs_feature = *rhs.last().unwrap_or(&1);
    if lhs_feature != feature_dim || rhs_feature != feature_dim {
        return None;
    }

    let lhs_matches = lhs.len() == 1 || lhs == out;
    let rhs_matches = rhs.len() == 1 || rhs == out;
    if lhs_matches && rhs_matches && (lhs.len() == 1 || rhs.len() == 1) {
        Some(feature_dim)
    } else {
        None
    }
}

fn suffix_broadcast_block_len(lhs: &[usize], rhs: &[usize], out: &[usize]) -> Option<usize> {
    if lhs != out || rhs.len() >= out.len() || rhs.is_empty() {
        return None;
    }

    let rank_diff = out.len() - rhs.len();
    if out[..rank_diff].iter().any(|&dim| dim == 0) {
        return None;
    }
    if out[rank_diff..] != rhs[..] {
        return None;
    }

    Some(rhs.iter().copied().product())
}

fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize], usize)) {
    let len = plain_num_elements(shape);
    if len == 0 {
        return;
    }
    if shape.is_empty() {
        f(&[], 0);
        return;
    }

    let mut coords = vec![0usize; shape.len()];
    for flat in 0..len {
        f(&coords, flat);
        for axis in (0..coords.len()).rev() {
            coords[axis] += 1;
            if coords[axis] < shape[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
}

fn validate_rope_shape(
    shape: &[usize],
    positions_len: usize,
    head_dim: usize,
    num_heads: usize,
    op_name: &str,
) -> Result<()> {
    if shape.len() != 3 {
        return Err(invalid_arg(format!(
            "{op_name} expects a rank-3 tensor shaped [num_heads, seq_len, head_dim], got {:?}",
            shape
        )));
    }
    if shape[0] != num_heads {
        return Err(invalid_arg(format!(
            "{op_name} expected num_heads={}, got shape {:?}",
            num_heads, shape
        )));
    }
    if shape[1] != positions_len {
        return Err(invalid_arg(format!(
            "{op_name} expected seq_len={}, got shape {:?}",
            positions_len, shape
        )));
    }
    if shape[2] != head_dim {
        return Err(invalid_arg(format!(
            "{op_name} expected head_dim={}, got shape {:?}",
            head_dim, shape
        )));
    }
    Ok(())
}

fn normalize_rope_dims(
    head_dim: usize,
    rope_dims: usize,
    op_name: &str,
    mixed: bool,
) -> Result<usize> {
    if head_dim == 0 {
        return Err(invalid_arg(format!("{op_name} requires head_dim > 0")));
    }

    let dims = if rope_dims == 0 { head_dim } else { rope_dims };
    if dims > head_dim {
        return Err(invalid_arg(format!(
            "{op_name} rope_dims {} exceeds head_dim {}",
            dims, head_dim
        )));
    }
    if mixed {
        if dims % 4 != 0 {
            return Err(invalid_arg(format!(
                "{op_name} requires rope_dims divisible by 4 for mixed RoPE, got {}",
                dims
            )));
        }
    } else if dims % 2 != 0 {
        return Err(invalid_arg(format!(
            "{op_name} requires an even rope_dims, got {}",
            dims
        )));
    }

    Ok(dims)
}

fn apply_rope_chunk(values: &mut [f32], start: usize, dims: usize, position: f32, theta: f32) {
    for local_offset in (0..dims).step_by(2) {
        let angle = position / theta.powf(local_offset as f32 / dims as f32);
        let (sin, cos) = angle.sin_cos();
        let i0 = start + local_offset;
        let i1 = i0 + 1;
        let x0 = values[i0];
        let x1 = values[i1];
        values[i0] = x0 * cos - x1 * sin;
        values[i1] = x0 * sin + x1 * cos;
    }
}

fn erf_approx(x: f32) -> f32 {
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

fn should_parallelize(len: usize) -> bool {
    len >= 16_384 && rayon::current_num_threads() > 1
}

fn should_parallelize_linear(rows: usize, in_dim: usize, out_dim: usize) -> bool {
    if rayon::current_num_threads() <= 1 || rows < 2 || in_dim == 0 || out_dim == 0 {
        return false;
    }

    let work = rows.saturating_mul(in_dim).saturating_mul(out_dim);
    work >= 1_000_000 && rows.saturating_mul(out_dim) >= 8_192
}

fn choose_parallel_row_chunk_len(rows: usize, out_dim: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let target_tasks = linear_target_tasks(threads);
    let by_tasks = rows.div_ceil(target_tasks);
    let min_chunk_rows = linear_min_outputs_per_chunk().div_ceil(out_dim.max(1));
    by_tasks.max(min_chunk_rows).max(1).min(rows)
}

fn linear_target_tasks(threads: usize) -> usize {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    static DEFAULT: OnceLock<usize> = OnceLock::new();
    OVERRIDE
        .get_or_init(|| {
            std::env::var("CRABML_LINEAR_TARGET_TASKS")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|&value| value > 0)
        })
        .unwrap_or(*DEFAULT.get_or_init(|| num_cpus::get_physical().max(1)))
        .min(threads)
        .max(1)
}

fn linear_min_outputs_per_chunk() -> usize {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    OVERRIDE
        .get_or_init(|| {
            std::env::var("CRABML_LINEAR_MIN_OUTPUTS_PER_CHUNK")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|&value| value > 0)
        })
        .unwrap_or(16_384)
}

fn should_parallelize_attention_matmul(
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

fn choose_parallel_attention_row_chunk_len(
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

fn apply_mask_and_softmax_row(
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
    if let Some(mask) = mask_data {
        let mask_base = match mask_shape {
            Some([_, _]) => row * mask_outer_stride.unwrap_or(key_len),
            Some([_, _, _]) => {
                head * mask_head_stride.unwrap_or(0) + row * mask_outer_stride.unwrap_or(key_len)
            }
            _ => 0,
        };
        let mask_row = &mask[mask_base..mask_base + key_len];
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

    for value in row_scores.iter_mut() {
        *value /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::{CpuDevice, CpuTensor};
    use crate::tensor::tests;

    #[test]
    fn layout_ops_preserve_view_semantics() {
        tests::run_layout_ops_preserve_view_semantics::<CpuTensor>(&CpuDevice);
    }

    #[test]
    fn broadcast_add_and_mul_match_expected_values() {
        tests::run_broadcast_add_and_mul_match_expected_values::<CpuTensor>(&CpuDevice);
    }

    #[test]
    fn matmul_supports_2d_and_batched_3d_inputs() {
        tests::run_matmul_supports_2d_and_batched_3d_inputs::<CpuTensor>(&CpuDevice);
    }

    #[test]
    fn linear_applies_weight_rows_and_optional_bias() {
        tests::run_linear_applies_weight_rows_and_optional_bias::<CpuTensor>(&CpuDevice);
    }

    #[test]
    fn normalization_and_activation_ops_match_reference_values() {
        tests::run_normalization_and_activation_ops_match_reference_values::<CpuTensor>(&CpuDevice);
    }

    #[test]
    fn rope_rotates_each_head_using_global_positions() {
        tests::run_rope_rotates_each_head_using_global_positions::<CpuTensor>(&CpuDevice);
    }

    #[test]
    fn region_rope_splits_global_and_region_rotation_halves() {
        tests::run_region_rope_splits_global_and_region_rotation_halves::<CpuTensor>(&CpuDevice);
    }

    #[test]
    fn depthwise_conv_applies_per_channel_kernels() {
        tests::run_depthwise_conv_applies_per_channel_kernels::<CpuTensor>(&CpuDevice);
    }

    #[test]
    fn embedding_and_repeat_return_expected_rows() {
        tests::run_embedding_and_repeat_return_expected_rows::<CpuTensor>(&CpuDevice);
    }

    #[test]
    fn roundtrip_matches_uploaded_values() {
        tests::run_roundtrip::<CpuTensor>(&CpuDevice);
    }
}
