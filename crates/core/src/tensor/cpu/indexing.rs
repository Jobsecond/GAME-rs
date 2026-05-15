use crate::Result;

use super::CpuTensor;
use super::util::*;

impl CpuTensor {
    pub(super) fn embedding(table: &Self, indices: &[i32]) -> Result<Self> {
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

    pub(super) fn repeat(self, axis: usize, n: usize) -> Result<Self> {
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
