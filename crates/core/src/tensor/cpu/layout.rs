use rayon::prelude::*;

use crate::Result;
use crate::profiler::op_scope_with;

use super::CpuTensor;
use super::util::*;

impl CpuTensor {
    pub(super) fn validate_attention_layout_input(
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

    pub(super) fn split_last_dim_parts_for_attention_heads(
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
                self.is_contiguous()
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

    pub(super) fn reshape(self, shape: &[usize]) -> Result<Self> {
        let new_n = checked_num_elements(shape)?;
        let old_n = self.num_elements();
        if new_n != old_n {
            return Err(invalid_arg(format!(
                "reshape: cannot reshape {:?} ({} elements) to {:?} ({} elements)",
                self.shape, old_n, shape, new_n
            )));
        }
        if self.is_contiguous() {
            Ok(Self {
                data: self.data,
                shape: shape.to_vec(),
                strides: contiguous_strides(shape),
                offset: self.offset,
                device: self.device,
            })
        } else {
            let data = self.to_vec()?;
            Self::from_owned(data, shape)
        }
    }

    pub(super) fn transpose(self, dim0: usize, dim1: usize) -> Result<Self> {
        let rank = self.shape.len();
        if dim0 >= rank || dim1 >= rank {
            return Err(invalid_arg(format!(
                "transpose: dimensions ({}, {}) out of range for rank {}",
                dim0, dim1, rank
            )));
        }
        let mut shape = self.shape;
        let mut strides = self.strides;
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Ok(Self {
            data: self.data,
            shape,
            strides,
            offset: self.offset,
            device: self.device,
        })
    }

    pub(super) fn contiguous(self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self);
        }
        let shape = self.shape.clone();
        let data = self.to_vec()?;
        Self::from_owned(data, &shape)
    }

    pub(super) fn slice(self, axis: usize, start: usize, end: usize) -> Result<Self> {
        validate_axis(axis, self.shape.len(), "slice")?;
        if end < start {
            return Err(invalid_arg(format!(
                "slice end {} is less than start {}",
                end, start
            )));
        }
        if end > self.shape[axis] {
            return Err(invalid_arg(format!(
                "slice end {} exceeds dimension size {} on axis {}",
                end, self.shape[axis], axis
            )));
        }
        let new_offset = self.offset + start * self.strides[axis];
        let mut new_shape = self.shape;
        new_shape[axis] = end - start;
        Ok(Self {
            data: self.data,
            shape: new_shape,
            strides: self.strides,
            offset: new_offset,
            device: self.device,
        })
    }

    pub(super) fn layout_for_attention_heads(
        self,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let _profile = op_scope_with("cpu.layout_for_attention_heads", || {
            format!(
                "shape={:?} num_heads={} head_dim={} contiguous={}",
                self.shape(),
                num_heads,
                head_dim,
                self.is_contiguous()
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

    pub(super) fn split_last_dim_two_for_attention_heads(
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
        let second = parts.pop().ok_or_else(|| {
            invalid_arg("split_last_dim_two_for_attention_heads missing second part")
        })?;
        let first = parts.pop().ok_or_else(|| {
            invalid_arg("split_last_dim_two_for_attention_heads missing first part")
        })?;
        Ok((first, second))
    }

    pub(super) fn split_last_dim_three_for_attention_heads(
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
        let third = parts.pop().ok_or_else(|| {
            invalid_arg("split_last_dim_three_for_attention_heads missing third part")
        })?;
        let second = parts.pop().ok_or_else(|| {
            invalid_arg("split_last_dim_three_for_attention_heads missing second part")
        })?;
        let first = parts.pop().ok_or_else(|| {
            invalid_arg("split_last_dim_three_for_attention_heads missing first part")
        })?;
        Ok((first, second, third))
    }

    pub(super) fn merge_attention_heads(self) -> Result<Self> {
        let _profile = op_scope_with("cpu.merge_attention_heads", || {
            format!(
                "shape={:?} contiguous={}",
                self.shape(),
                self.is_contiguous()
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
                            out_row[dst_start..dst_end].copy_from_slice(&input[src_start..src_end]);
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

    pub(super) fn concat(parts: &[&Self], axis: usize) -> Result<Self> {
        let _profile = op_scope_with("cpu.concat", || {
            format!(
                "parts={} axis={} first_shape={:?}",
                parts.len(),
                axis,
                parts
                    .first()
                    .map(|part| part.shape().to_vec())
                    .unwrap_or_default()
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
}
