use crate::Result;

use super::base::GpuTensor;
use super::params::*;
use super::pipelines::*;
use super::util::*;

impl GpuTensor {
    pub(super) fn embedding(table: &Self, indices: &[i32]) -> Result<Self> {
        let table = table.clone().contiguous()?;
        if table.shape.len() != 2 {
            return Err(invalid_arg(format!(
                "embedding table must have shape [rows, dim], got {:?}",
                table.shape
            )));
        }

        let rows = table.shape[0];
        let dim = table.shape[1];
        for &index in indices {
            let source_row = usize::try_from(index)
                .map_err(|_| invalid_arg(format!("embedding index {index} is negative")))?;
            if source_row >= rows {
                return Err(invalid_arg(format!(
                    "embedding index {} is out of bounds for {} rows",
                    source_row, rows
                )));
            }
        }

        let out_shape = vec![indices.len(), dim];
        let out_len = checked_num_elements(&out_shape)?;
        if out_len == 0 {
            return Self::zeros(&out_shape, &table.device);
        }

        let params = EmbeddingParams {
            out_len: usize_to_u32(out_len, "embedding output length")?,
            dim: usize_to_u32(dim, "embedding dim")?,
            _reserved0: 0,
            _reserved1: 0,
        };
        let params_buffer = table
            .device
            .create_storage_buffer_from_pod(&params, "gpu-embedding-params");
        let indices_buffer = table
            .device
            .create_storage_buffer_from_i32(indices, "gpu-embedding-indices");
        let out_buffer = table
            .device
            .create_empty_storage_buffer(out_len, "gpu-embedding-out")?;
        table.device.dispatch_compute(
            &table.device.inner.pipelines.embedding,
            &[&table.buffer, &indices_buffer, &out_buffer, &params_buffer],
            elementwise_workgroups(params.out_len),
            "embedding",
            None,
        )?;

        Ok(Self {
            buffer: out_buffer,
            storage_elements: out_len,
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
            device: table.device.clone(),
        })
    }

    pub(super) fn repeat(self, axis: usize, n: usize) -> Result<Self> {
        let input = self.contiguous()?;
        let rank = input.shape.len();
        validate_axis(axis, rank, "repeat")?;

        let mut out_shape = input.shape.clone();
        out_shape[axis] = out_shape[axis]
            .checked_mul(n)
            .ok_or_else(|| invalid_arg("repeat axis size overflow"))?;
        let out_len = checked_num_elements(&out_shape)?;
        if out_len == 0 {
            return Self::zeros(&out_shape, &input.device);
        }

        let params = RepeatParams {
            out_len: usize_to_u32(out_len, "repeat output length")?,
            outer: usize_to_u32(plain_num_elements(&input.shape[..axis]), "repeat outer")?,
            axis_len: usize_to_u32(input.shape[axis], "repeat axis length")?,
            inner: usize_to_u32(plain_num_elements(&input.shape[axis + 1..]), "repeat inner")?,
            repeat_n: usize_to_u32(n, "repeat count")?,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let params_buffer = input
            .device
            .create_storage_buffer_from_pod(&params, "gpu-repeat-params");
        let out_buffer = input
            .device
            .create_empty_storage_buffer(out_len, "gpu-repeat-out")?;
        input.device.dispatch_compute(
            &input.device.inner.pipelines.repeat,
            &[&input.buffer, &out_buffer, &params_buffer],
            elementwise_workgroups(params.out_len),
            "repeat",
            None,
        )?;

        Ok(Self {
            buffer: out_buffer,
            storage_elements: out_len,
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
            device: input.device.clone(),
        })
    }
}
