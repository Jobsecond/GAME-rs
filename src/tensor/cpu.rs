use std::f32::consts::SQRT_2;
use std::sync::Arc;

use rayon::prelude::*;

use crate::{Error, Result};

use super::Tensor;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CpuDevice;

#[derive(Debug, Clone)]
pub struct CpuTensor {
    data: Arc<Vec<f32>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    device: CpuDevice,
}

impl CpuTensor {
    pub fn num_elements(&self) -> usize {
        plain_num_elements(&self.shape)
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let mut out = vec![0.0; self.num_elements()];
        self.export(&mut out)?;
        Ok(out)
    }

    fn from_owned(data: Vec<f32>, shape: Vec<usize>, device: CpuDevice) -> Result<Self> {
        let expected = checked_num_elements(&shape)?;
        if data.len() != expected {
            return Err(invalid_arg(format!(
                "tensor data length {} does not match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            )));
        }

        Ok(Self {
            data: Arc::new(data),
            strides: contiguous_strides(&shape),
            shape,
            offset: 0,
            device,
        })
    }

    fn is_dense_contiguous_view(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    /// Returns a borrowed slice of the underlying data when the tensor is
    /// dense and contiguous, avoiding a heap allocation.  Falls back to `None`
    /// for transposed / sliced views, where callers should use `to_vec()`.
    fn as_contiguous_slice(&self) -> Option<&[f32]> {
        if self.is_dense_contiguous_view() {
            Some(&self.data[self.offset..self.offset + self.num_elements()])
        } else {
            None
        }
    }

    fn has_compact_storage(&self) -> bool {
        self.offset == 0
            && self.is_dense_contiguous_view()
            && self.data.len() == self.num_elements()
    }

    fn into_contiguous_vec(self) -> Result<Vec<f32>> {
        if self.has_compact_storage() {
            return Ok(match Arc::try_unwrap(self.data) {
                Ok(data) => data,
                Err(data) => (*data).clone(),
            });
        }

        // Fast path for dense contiguous views with an offset (e.g., slices).
        // The `export` fast path handles this too, but going through `export`
        // requires allocating a zeroed buffer first; this avoids that.
        if self.is_dense_contiguous_view() {
            let offset = self.offset;
            let len = self.num_elements();
            return Ok(self.data[offset..offset + len].to_vec());
        }

        let mut out = vec![0.0; self.num_elements()];
        self.export(&mut out)?;
        Ok(out)
    }

    fn raw_offset(&self, coords: &[usize]) -> usize {
        self.offset
            + coords
                .iter()
                .zip(&self.strides)
                .map(|(coord, stride)| coord * stride)
                .sum::<usize>()
    }

    fn unary_op(self, op_name: &str, f: impl Fn(f32) -> f32 + Send + Sync) -> Result<Self> {
        let shape = self.shape.clone();
        let device = self.device.clone();
        let mut data = self.into_contiguous_vec()?;
        if should_parallelize(data.len()) {
            data.par_iter_mut().for_each(|value| *value = f(*value));
        } else {
            for value in &mut data {
                *value = f(*value);
            }
        }
        Self::from_owned(data, shape, device)
            .map_err(|err| Error::message(format!("failed to build result for {op_name}: {err}")))
    }

    fn binary_op(
        self,
        rhs: &Self,
        op_name: &str,
        f: impl Fn(f32, f32) -> f32 + Send + Sync,
    ) -> Result<Self> {
        let lhs_shape = self.shape.clone();
        let lhs_strides = contiguous_strides(&lhs_shape);
        let device = self.device.clone();
        let lhs_data = self.into_contiguous_vec()?;

        let rhs_shape = rhs.shape.clone();
        let rhs_strides = contiguous_strides(&rhs_shape);
        let rhs_owned;
        let rhs_data: &[f32] = match rhs.as_contiguous_slice() {
            Some(s) => s,
            None => {
                rhs_owned = rhs.to_vec()?;
                &rhs_owned
            }
        };

        let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
        let mut out = vec![0.0; checked_num_elements(&out_shape)?];
        let out_rank = out_shape.len();

        if lhs_shape == out_shape && rhs_shape == out_shape && should_parallelize(out.len()) {
            out.par_iter_mut().enumerate().for_each(|(flat, value)| {
                *value = f(lhs_data[flat], rhs_data[flat]);
            });
        } else {
            for_each_index(&out_shape, |coords, flat| {
                let lhs_index = broadcast_offset(coords, &lhs_shape, &lhs_strides, out_rank);
                let rhs_index = broadcast_offset(coords, &rhs_shape, &rhs_strides, out_rank);
                out[flat] = f(lhs_data[lhs_index], rhs_data[rhs_index]);
            });
        }

        Self::from_owned(out, out_shape, device)
            .map_err(|err| Error::message(format!("failed to build result for {op_name}: {err}")))
    }

    fn validate_rope_shape(
        &self,
        positions_len: usize,
        head_dim: usize,
        num_heads: usize,
        op_name: &str,
    ) -> Result<()> {
        if self.shape.len() != 3 {
            return Err(invalid_arg(format!(
                "{op_name} expects a rank-3 tensor shaped [num_heads, seq_len, head_dim], got {:?}",
                self.shape
            )));
        }
        if self.shape[0] != num_heads {
            return Err(invalid_arg(format!(
                "{op_name} expected num_heads={}, got shape {:?}",
                num_heads, self.shape
            )));
        }
        if self.shape[1] != positions_len {
            return Err(invalid_arg(format!(
                "{op_name} expected seq_len={}, got shape {:?}",
                positions_len, self.shape
            )));
        }
        if self.shape[2] != head_dim {
            return Err(invalid_arg(format!(
                "{op_name} expected head_dim={}, got shape {:?}",
                head_dim, self.shape
            )));
        }

        Ok(())
    }
}

impl Tensor for CpuTensor {
    type Device = CpuDevice;

    fn from_data(data: &[f32], shape: &[usize], device: &Self::Device) -> Result<Self> {
        let expected = checked_num_elements(shape)?;
        if data.len() != expected {
            return Err(invalid_arg(format!(
                "tensor data length {} does not match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            )));
        }

        Self::from_owned(data.to_vec(), shape.to_vec(), device.clone())
    }

    fn zeros(shape: &[usize], device: &Self::Device) -> Result<Self> {
        let len = checked_num_elements(shape)?;
        Self::from_owned(vec![0.0; len], shape.to_vec(), device.clone())
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn export(&self, buf: &mut [f32]) -> Result<()> {
        let expected = self.num_elements();
        if buf.len() != expected {
            return Err(invalid_arg(format!(
                "export buffer length {} does not match tensor shape {:?} ({} elements)",
                buf.len(),
                self.shape,
                expected
            )));
        }

        if self.is_dense_contiguous_view() {
            buf.copy_from_slice(&self.data[self.offset..self.offset + buf.len()]);
        } else {
            for_each_index(&self.shape, |coords, flat| {
                buf[flat] = self.data[self.raw_offset(coords)];
            });
        }

        Ok(())
    }

    fn reshape(mut self, shape: &[usize]) -> Result<Self> {
        let expected = checked_num_elements(shape)?;
        if expected != self.num_elements() {
            return Err(invalid_arg(format!(
                "cannot reshape tensor {:?} into {:?}: element count mismatch",
                self.shape, shape
            )));
        }

        if self.is_dense_contiguous_view() {
            self.shape = shape.to_vec();
            self.strides = contiguous_strides(shape);
            return Ok(self);
        }

        let device = self.device.clone();
        let data = self.into_contiguous_vec()?;
        Self::from_owned(data, shape.to_vec(), device)
    }

    fn transpose(mut self, dim0: usize, dim1: usize) -> Result<Self> {
        let rank = self.shape.len();
        validate_axis(dim0, rank, "transpose")?;
        validate_axis(dim1, rank, "transpose")?;
        self.shape.swap(dim0, dim1);
        self.strides.swap(dim0, dim1);
        Ok(self)
    }

    fn contiguous(self) -> Result<Self> {
        if self.has_compact_storage() {
            return Ok(self);
        }

        let shape = self.shape.clone();
        let device = self.device.clone();
        let data = self.into_contiguous_vec()?;
        Self::from_owned(data, shape, device)
    }

    fn slice(mut self, axis: usize, start: usize, end: usize) -> Result<Self> {
        let rank = self.shape.len();
        validate_axis(axis, rank, "slice")?;
        let dim = self.shape[axis];
        if start > end || end > dim {
            return Err(invalid_arg(format!(
                "slice axis {} with range {}..{} is out of bounds for shape {:?}",
                axis, start, end, self.shape
            )));
        }

        self.offset += start * self.strides[axis];
        self.shape[axis] = end - start;
        Ok(self)
    }

    fn concat(parts: &[&Self], axis: usize) -> Result<Self> {
        if parts.is_empty() {
            return Err(invalid_arg("concat requires at least one tensor"));
        }

        let rank = parts[0].shape.len();
        validate_axis(axis, rank, "concat")?;
        let device = parts[0].device.clone();
        let expected_shape = parts[0].shape.clone();
        let mut out_shape = expected_shape.clone();
        out_shape[axis] = 0;

        for part in parts {
            if part.shape.len() != rank {
                return Err(invalid_arg(format!(
                    "concat rank mismatch: expected rank {}, got shape {:?}",
                    rank, part.shape
                )));
            }
            for dim in 0..rank {
                if dim != axis && part.shape[dim] != out_shape[dim] {
                    return Err(invalid_arg(format!(
                        "concat shape mismatch on axis {}: expected non-concat dims {:?}, got {:?}",
                        axis, expected_shape, part.shape
                    )));
                }
            }
            out_shape[axis] += part.shape[axis];
        }

        let out_len = checked_num_elements(&out_shape)?;
        let mut out = vec![0.0; out_len];
        let outer = plain_num_elements(&out_shape[..axis]);
        let inner = plain_num_elements(&out_shape[axis + 1..]);
        let out_axis_span = out_shape[axis] * inner;
        let mut axis_offset = 0usize;

        for part in parts {
            let part_block = part.shape[axis] * inner;
            let part_owned;
            let part_data: &[f32] = match part.as_contiguous_slice() {
                Some(slice) => slice,
                None => {
                    part_owned = part.to_vec()?;
                    &part_owned
                }
            };
            for outer_index in 0..outer {
                let src_start = outer_index * part_block;
                let src_end = src_start + part_block;
                let dst_start = outer_index * out_axis_span + axis_offset * inner;
                let dst_end = dst_start + part_block;
                out[dst_start..dst_end].copy_from_slice(&part_data[src_start..src_end]);
            }
            axis_offset += part.shape[axis];
        }

        Self::from_owned(out, out_shape, device)
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

    fn matmul(&self, rhs: &Self) -> Result<Self> {
        let lhs_shape = self.shape.clone();
        let rhs_shape = rhs.shape.clone();
        let device = self.device.clone();

        // Stride-aware data access: for 2D tensors we can pass the actual strides
        // directly to sgemm, avoiding the expensive to_vec() coordinate walk on
        // transposed views (e.g. k_t in attention).
        //
        // For the pointer to be valid the underlying Arc<Vec<f32>> storage must be
        // a single contiguous allocation that the strides index into.  This is
        // always true for tensors produced by `from_owned` / `transpose` / `slice`
        // — the data is never scattered.

        match (lhs_shape.len(), rhs_shape.len()) {
            (2, 2) => {
                let (m, k) = (lhs_shape[0], lhs_shape[1]);
                let (rhs_k, n) = (rhs_shape[0], rhs_shape[1]);
                if k != rhs_k {
                    return Err(invalid_arg(format!(
                        "matmul shape mismatch: {:?} @ {:?}",
                        lhs_shape, rhs_shape
                    )));
                }

                let lhs_ptr = self.data[self.offset..].as_ptr();
                let lhs_rs = self.strides[0] as isize;
                let lhs_cs = self.strides[1] as isize;

                let rhs_ptr = rhs.data[rhs.offset..].as_ptr();
                let rhs_rs = rhs.strides[0] as isize;
                let rhs_cs = rhs.strides[1] as isize;

                let mut out = vec![0.0; m * n];
                unsafe {
                    matrixmultiply::sgemm(
                        m,
                        k,
                        n,
                        1.0,
                        lhs_ptr,
                        lhs_rs,
                        lhs_cs,
                        rhs_ptr,
                        rhs_rs,
                        rhs_cs,
                        0.0,
                        out.as_mut_ptr(),
                        n as isize,
                        1,
                    );
                }
                Self::from_owned(out, vec![m, n], device)
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

                let mut out = vec![0.0; batch * m * n];
                let batch_out_size = m * n;

                // For 3D batched matmul, use stride-aware access.
                // Batch stride is strides[0], inner matrix strides are strides[1..2].
                let lhs_batch_stride = self.strides[0];
                let lhs_rs = self.strides[1] as isize;
                let lhs_cs = self.strides[2] as isize;

                let rhs_batch_stride = rhs.strides[0];
                let rhs_rs = rhs.strides[1] as isize;
                let rhs_cs = rhs.strides[2] as isize;

                if should_parallelize(out.len()) {
                    // Parallelize across batches
                    out.par_chunks_mut(batch_out_size)
                        .enumerate()
                        .for_each(|(b, out_batch)| {
                            let lhs_ptr =
                                self.data[self.offset + b * lhs_batch_stride..].as_ptr();
                            let rhs_ptr =
                                rhs.data[rhs.offset + b * rhs_batch_stride..].as_ptr();
                            unsafe {
                                matrixmultiply::sgemm(
                                    m,
                                    k,
                                    n,
                                    1.0,
                                    lhs_ptr,
                                    lhs_rs,
                                    lhs_cs,
                                    rhs_ptr,
                                    rhs_rs,
                                    rhs_cs,
                                    0.0,
                                    out_batch.as_mut_ptr(),
                                    n as isize,
                                    1,
                                );
                            }
                        });
                } else {
                    for b in 0..batch {
                        let lhs_ptr =
                            self.data[self.offset + b * lhs_batch_stride..].as_ptr();
                        let rhs_ptr =
                            rhs.data[rhs.offset + b * rhs_batch_stride..].as_ptr();
                        let out_offset = b * batch_out_size;
                        unsafe {
                            matrixmultiply::sgemm(
                                m,
                                k,
                                n,
                                1.0,
                                lhs_ptr,
                                lhs_rs,
                                lhs_cs,
                                rhs_ptr,
                                rhs_rs,
                                rhs_cs,
                                0.0,
                                out[out_offset..].as_mut_ptr(),
                                n as isize,
                                1,
                            );
                        }
                    }
                }
                Self::from_owned(out, vec![batch, m, n], device)
            }
            _ => Err(invalid_arg(format!(
                "matmul expects rank-2 or rank-3 tensors, got {:?} and {:?}",
                lhs_shape, rhs_shape
            ))),
        }
    }

    fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self> {
        if self.shape.is_empty() {
            return Err(invalid_arg(
                "linear expects an input tensor with at least one dimension",
            ));
        }
        if weight.shape.len() != 2 {
            return Err(invalid_arg(format!(
                "linear weight must be rank-2 [out_dim, in_dim], got {:?}",
                weight.shape
            )));
        }

        let input_shape = self.shape.clone();
        let in_dim = *input_shape.last().unwrap_or(&0);
        let out_dim = weight.shape[0];
        if weight.shape[1] != in_dim {
            return Err(invalid_arg(format!(
                "linear shape mismatch: input {:?}, weight {:?}",
                input_shape, weight.shape
            )));
        }
        if let Some(bias) = bias
            && bias.shape != [out_dim]
        {
            return Err(invalid_arg(format!(
                "linear bias must have shape [{out_dim}], got {:?}",
                bias.shape
            )));
        }

        let rows = plain_num_elements(&input_shape[..input_shape.len() - 1]);

        let input_owned;
        let input: &[f32] = match self.as_contiguous_slice() {
            Some(s) => s,
            None => {
                input_owned = self.to_vec()?;
                &input_owned
            }
        };
        let weight_owned;
        let weight_data: &[f32] = match weight.as_contiguous_slice() {
            Some(s) => s,
            None => {
                weight_owned = weight.to_vec()?;
                &weight_owned
            }
        };
        let bias_data = bias.map(CpuTensor::to_vec).transpose()?;
        let mut out = vec![0.0; rows * out_dim];

        // Large sequence-major linears dominate encoder / segmenter / estimator CPU time.
        // Split independent row blocks across Rayon workers so multiple sgemm calls can
        // run concurrently while sharing the same read-only weight matrix.
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
            // Use sgemm for y = x @ W^T
            // x is [rows, in_dim], W is [out_dim, in_dim], W^T is [in_dim, out_dim]
            // Result is [rows, out_dim]
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
                    in_dim as isize, // transposed stride
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
        Self::from_owned(out, out_shape, self.device.clone())
    }

    fn rms_norm(self, weight: &Self, eps: f32) -> Result<Self> {
        if self.shape.is_empty() {
            return Err(invalid_arg(
                "rms_norm expects an input tensor with at least one dimension",
            ));
        }

        let feature_dim = *self.shape.last().unwrap_or(&0);
        if weight.shape != [feature_dim] {
            return Err(invalid_arg(format!(
                "rms_norm weight must have shape [{feature_dim}], got {:?}",
                weight.shape
            )));
        }

        let shape = self.shape.clone();
        let device = self.device.clone();
        let mut data = self.into_contiguous_vec()?;
        let weight_owned;
        let weight_data: &[f32] = match weight.as_contiguous_slice() {
            Some(s) => s,
            None => {
                weight_owned = weight.to_vec()?;
                &weight_owned
            }
        };
        if feature_dim == 0 {
            return Self::from_owned(data, shape, device);
        }

        if should_parallelize(data.len()) {
            data.par_chunks_mut(feature_dim).for_each(|row_slice| {
                let mean_square =
                    row_slice.iter().map(|value| value * value).sum::<f32>() / feature_dim as f32;
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
                let mean_square =
                    row_slice.iter().map(|value| value * value).sum::<f32>() / feature_dim as f32;
                let inv_rms = 1.0 / (mean_square + eps).sqrt();
                for (value, scale) in row_slice.iter_mut().zip(weight_data.iter()) {
                    *value *= inv_rms * scale;
                }
            }
        }

        Self::from_owned(data, shape, device)
    }

    fn gelu(self) -> Result<Self> {
        self.unary_op("gelu", |value| {
            0.5 * value * (1.0 + erf_approx(value / SQRT_2))
        })
    }

    fn softmax(self, axis: isize) -> Result<Self> {
        if self.shape.is_empty() {
            return Err(invalid_arg(
                "softmax expects a tensor with at least one dimension",
            ));
        }

        let shape = self.shape.clone();
        let device = self.device.clone();
        let axis = normalize_axis(axis, shape.len(), "softmax")?;
        let mut data = self.into_contiguous_vec()?;
        let axis_len = shape[axis];
        if axis_len == 0 {
            return Self::from_owned(data, shape, device);
        }

        let outer = plain_num_elements(&shape[..axis]);
        let inner = plain_num_elements(&shape[axis + 1..]);

        if outer == 0 || inner == 0 {
            return Self::from_owned(data, shape, device);
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
                        let index = base + axis_index * inner;
                        data[index] /= sum;
                    }
                }
            }
        }

        Self::from_owned(data, shape, device)
    }

    fn rope(
        self,
        positions: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self> {
        self.validate_rope_shape(positions.len(), head_dim, num_heads, "rope")?;
        let rope_dims = normalize_rope_dims(head_dim, rope_dims, "rope", false)?;

        let shape = self.shape.clone();
        let device = self.device.clone();
        let mut data = self.into_contiguous_vec()?;
        let seq_len = shape[1];

        let head_block = seq_len * head_dim;
        if should_parallelize(data.len()) && head_block > 0 {
            data.par_chunks_mut(head_block).for_each(|head_slice| {
                for (token, position) in positions.iter().copied().enumerate() {
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
                for (token, position) in positions.iter().copied().enumerate() {
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

        Self::from_owned(data, shape, device)
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
        if global_pos.len() != region_ids.len() {
            return Err(invalid_arg(format!(
                "region_rope expects matching global_pos and region_ids lengths, got {} and {}",
                global_pos.len(),
                region_ids.len()
            )));
        }
        self.validate_rope_shape(global_pos.len(), head_dim, num_heads, "region_rope")?;
        let mixed_dims = normalize_rope_dims(head_dim, rope_dims, "region_rope", true)?;
        let half = mixed_dims / 2;

        let shape = self.shape.clone();
        let device = self.device.clone();
        let mut data = self.into_contiguous_vec()?;
        let seq_len = shape[1];

        let head_block = seq_len * head_dim;
        if should_parallelize(data.len()) && head_block > 0 {
            data.par_chunks_mut(head_block).for_each(|head_slice| {
                for token in 0..seq_len {
                    let base = token * head_dim;
                    let token_slice = &mut head_slice[base..base + head_dim];
                    apply_rope_chunk(token_slice, 0, half, global_pos[token] as f32, theta);
                    apply_rope_chunk(token_slice, half, half, region_ids[token] as f32, theta);
                }
            });
        } else {
            for head in 0..num_heads {
                for token in 0..seq_len {
                    let base = (head * seq_len + token) * head_dim;
                    let head_slice = &mut data[base..base + head_dim];
                    apply_rope_chunk(head_slice, 0, half, global_pos[token] as f32, theta);
                    apply_rope_chunk(head_slice, half, half, region_ids[token] as f32, theta);
                }
            }
        }

        Self::from_owned(data, shape, device)
    }

    fn conv1d_dw(
        self,
        kernel: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        if stride == 0 {
            return Err(invalid_arg("conv1d_dw requires stride > 0"));
        }
        if self.shape.len() != 2 {
            return Err(invalid_arg(format!(
                "conv1d_dw expects input shape [time, channels], got {:?}",
                self.shape
            )));
        }
        if kernel.shape.len() != 2 {
            return Err(invalid_arg(format!(
                "conv1d_dw kernel must have shape [channels, kernel_size], got {:?}",
                kernel.shape
            )));
        }

        let (time, channels) = (self.shape[0], self.shape[1]);
        let (kernel_channels, kernel_size) = (kernel.shape[0], kernel.shape[1]);
        if channels != kernel_channels {
            return Err(invalid_arg(format!(
                "conv1d_dw channel mismatch: input {:?}, kernel {:?}",
                self.shape, kernel.shape
            )));
        }
        if kernel_size == 0 {
            return Err(invalid_arg(
                "conv1d_dw kernel size must be greater than zero",
            ));
        }
        if let Some(bias) = bias
            && bias.shape != [channels]
        {
            return Err(invalid_arg(format!(
                "conv1d_dw bias must have shape [{channels}], got {:?}",
                bias.shape
            )));
        }

        let padded = time.saturating_add(padding.saturating_mul(2));
        let out_time = if padded < kernel_size {
            0
        } else {
            (padded - kernel_size) / stride + 1
        };

        let device = self.device.clone();
        let input = self.into_contiguous_vec()?;
        let kernel_data = kernel.to_vec()?;
        let bias_data = bias.map(CpuTensor::to_vec).transpose()?;
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

        Self::from_owned(out, vec![out_time, channels], device)
    }

    fn embedding(table: &Self, indices: &[i32]) -> Result<Self> {
        if table.shape.len() != 2 {
            return Err(invalid_arg(format!(
                "embedding table must have shape [rows, dim], got {:?}",
                table.shape
            )));
        }

        let rows = table.shape[0];
        let dim = table.shape[1];
        let table_data = table.to_vec()?;
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

        Self::from_owned(out, vec![indices.len(), dim], table.device.clone())
    }

    fn repeat(self, axis: usize, n: usize) -> Result<Self> {
        let rank = self.shape.len();
        validate_axis(axis, rank, "repeat")?;
        let shape = self.shape.clone();
        let device = self.device.clone();
        let data = self.into_contiguous_vec()?;

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

        Self::from_owned(out, out_shape, device)
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

fn should_parallelize(len: usize) -> bool {
    len >= 16_384 && rayon::current_num_threads() > 1
}

fn should_parallelize_linear(rows: usize, in_dim: usize, out_dim: usize) -> bool {
    if rayon::current_num_threads() <= 1 || rows < 2 || in_dim == 0 || out_dim == 0 {
        return false;
    }

    let work = rows.saturating_mul(in_dim).saturating_mul(out_dim);
    work >= 2_000_000 && rows.saturating_mul(out_dim) >= 16_384
}

fn choose_parallel_row_chunk_len(rows: usize, out_dim: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let target_tasks = threads.saturating_mul(2);
    let by_tasks = rows.div_ceil(target_tasks);
    let min_chunk_rows = 16_384usize.div_ceil(out_dim.max(1));
    by_tasks.max(min_chunk_rows).max(1).min(rows)
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

fn broadcast_offset(
    coords: &[usize],
    shape: &[usize],
    strides: &[usize],
    out_rank: usize,
) -> usize {
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

fn normalize_rope_dims(
    head_dim: usize,
    rope_dims: usize,
    op_name: &str,
    mixed: bool,
) -> Result<usize> {
    if head_dim == 0 {
        return Err(invalid_arg(format!("{op_name} requires head_dim > 0")));
    }
    if theta_is_invalid(head_dim) {
        return Err(invalid_arg("invalid head_dim"));
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

fn theta_is_invalid(head_dim: usize) -> bool {
    head_dim == 0
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
