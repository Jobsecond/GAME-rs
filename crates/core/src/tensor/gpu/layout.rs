use std::sync::Arc;

use crate::Result;

use super::base::GpuTensor;
use super::params::*;
use super::pipelines::*;
use super::util::*;

impl GpuTensor {
    pub(super) fn reshape(mut self, shape: &[usize]) -> Result<Self> {
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

        let mut contiguous = self.contiguous()?;
        contiguous.shape = shape.to_vec();
        contiguous.strides = contiguous_strides(shape);
        Ok(contiguous)
    }

    pub(super) fn transpose(mut self, dim0: usize, dim1: usize) -> Result<Self> {
        let rank = self.shape.len();
        validate_axis(dim0, rank, "transpose")?;
        validate_axis(dim1, rank, "transpose")?;
        self.shape.swap(dim0, dim1);
        self.strides.swap(dim0, dim1);
        Ok(self)
    }

    pub(super) fn contiguous(self) -> Result<Self> {
        if self.has_compact_storage() {
            return Ok(self);
        }

        if self.shape.len() > MAX_DIMS {
            let data = self.materialize_view_on_cpu()?;
            return Self::from_owned(data, self.shape.clone(), self.device.clone());
        }

        let out_len = self.num_elements();
        if out_len == 0 {
            return Self::from_owned(Vec::new(), self.shape.clone(), self.device.clone());
        }

        let out_shape = self.shape.clone();
        let params = LayoutParams {
            out_len: usize_to_u32(out_len, "output element count")?,
            rank: usize_to_u32(self.shape.len(), "tensor rank")?,
            offset: usize_to_u32(self.offset, "tensor offset")?,
            _reserved: 0,
            shape: Self::pack_dims(&self.shape, &self.strides)?.0,
            out_strides: Self::pack_contiguous_strides(&self.shape)?,
            src_strides: Self::pack_dims(&self.shape, &self.strides)?.1,
        };
        let params_buffer = self
            .device
            .create_storage_buffer_from_pod(&params, "gpu-layout-params");
        let out_buffer = self
            .device
            .create_empty_storage_buffer(out_len, "gpu-contiguous-out")?;
        self.device.dispatch_compute(
            &self.device.inner.pipelines.contiguous,
            &[&self.buffer, &out_buffer, &params_buffer],
            elementwise_workgroups(params.out_len),
            "contiguous",
            None,
        )?;

        Ok(Self {
            buffer: out_buffer,
            storage_elements: out_len,
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
            device: self.device.clone(),
        })
    }

    pub(super) fn slice(mut self, axis: usize, start: usize, end: usize) -> Result<Self> {
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

    pub(super) fn concat(parts: &[&Self], axis: usize) -> Result<Self> {
        if parts.is_empty() {
            return Err(invalid_arg("concat requires at least one tensor"));
        }
        let device = parts[0].device.clone();
        for part in parts.iter().skip(1) {
            if !Arc::ptr_eq(&device.inner, &part.device.inner) {
                return Err(invalid_arg(
                    "concat requires tensors on the same GPU device",
                ));
            }
        }
        let rank = parts[0].shape.len();
        validate_axis(axis, rank, "concat")?;
        let mut contiguous_parts = Vec::with_capacity(parts.len());
        let mut out_shape = parts[0].shape.clone();
        out_shape[axis] = 0;

        for part in parts {
            let contiguous = (*part).clone().contiguous()?;
            if contiguous.shape.len() != rank {
                return Err(invalid_arg(format!(
                    "concat rank mismatch: expected rank {}, got shape {:?}",
                    rank, contiguous.shape
                )));
            }
            for dim in 0..rank {
                if dim != axis && contiguous.shape[dim] != parts[0].shape[dim] {
                    return Err(invalid_arg(format!(
                        "concat shape mismatch on axis {}: expected non-concat dims {:?}, got {:?}",
                        axis, parts[0].shape, contiguous.shape
                    )));
                }
            }
            out_shape[axis] += contiguous.shape[axis];
            contiguous_parts.push(contiguous);
        }

        let out_len = checked_num_elements(&out_shape)?;
        if out_len == 0 {
            return Self::zeros(&out_shape, &device);
        }

        let inner = plain_num_elements(&out_shape[axis + 1..]);
        let out_buffer = device.create_empty_storage_buffer(out_len, "gpu-concat-out")?;
        let mut axis_offset = 0usize;
        for part in &contiguous_parts {
            let params = ConcatParams {
                part_len: usize_to_u32(part.num_elements(), "concat part length")?,
                inner: usize_to_u32(inner, "concat inner span")?,
                part_axis_len: usize_to_u32(part.shape[axis], "concat part axis length")?,
                out_axis_len: usize_to_u32(out_shape[axis], "concat output axis length")?,
                axis_offset: usize_to_u32(axis_offset, "concat axis offset")?,
                _reserved0: 0,
                _reserved1: 0,
                _reserved2: 0,
            };
            let params_buffer = device.create_storage_buffer_from_pod(&params, "gpu-concat-params");
            device.dispatch_compute(
                &device.inner.pipelines.concat,
                &[&part.buffer, &out_buffer, &params_buffer],
                elementwise_workgroups(params.part_len),
                "concat",
                None,
            )?;
            axis_offset += part.shape[axis];
        }

        Ok(Self {
            buffer: out_buffer,
            storage_elements: out_len,
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
            device,
        })
    }
}
