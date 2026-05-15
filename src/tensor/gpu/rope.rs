use crate::Result;

use super::base::GpuTensor;
use super::params::*;
use super::pipelines::*;
use super::util::*;

impl GpuTensor {
    pub(super) fn rope(
        self,
        positions: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self> {
        let input = self.contiguous()?;
        validate_rope_shape(&input.shape, positions.len(), head_dim, num_heads, "rope")?;
        let rope_dims = normalize_rope_dims(head_dim, rope_dims, "rope", false)?;
        if input.num_elements() == 0 {
            return Ok(input);
        }

        let params = RopeParams {
            num_heads: usize_to_u32(num_heads, "rope num_heads")?,
            seq_len: usize_to_u32(positions.len(), "rope sequence length")?,
            head_dim: usize_to_u32(head_dim, "rope head dimension")?,
            rope_dims: usize_to_u32(rope_dims, "rope_dims")?,
            theta,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let params_buffer = input
            .device
            .create_storage_buffer_from_pod(&params, "gpu-rope-params");
        let positions_buffer = input
            .device
            .create_storage_buffer_from_i32(positions, "gpu-rope-positions");
        let out_buffer = input
            .device
            .create_empty_storage_buffer(input.num_elements(), "gpu-rope-out")?;
        input.device.dispatch_compute(
            &input.device.inner.pipelines.rope,
            &[
                &input.buffer,
                &out_buffer,
                &positions_buffer,
                &params_buffer,
            ],
            (
                div_ceil_u32(usize_to_u32(rope_dims / 2, "rope pairs")?, ROPE_WORKGROUP_X),
                div_ceil_u32(
                    usize_to_u32(positions.len(), "rope sequence length")?,
                    ROPE_WORKGROUP_Y,
                ),
                usize_to_u32(num_heads, "rope num_heads")?,
            ),
            "rope",
            Some((&input.buffer, &out_buffer, input.num_elements())),
        )?;

        Ok(Self {
            buffer: out_buffer,
            storage_elements: input.num_elements(),
            shape: input.shape.clone(),
            strides: contiguous_strides(&input.shape),
            offset: 0,
            device: input.device.clone(),
        })
    }

    pub(super) fn region_rope(
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

        let input = self.contiguous()?;
        validate_rope_shape(
            &input.shape,
            global_pos.len(),
            head_dim,
            num_heads,
            "region_rope",
        )?;
        let rope_dims = normalize_rope_dims(head_dim, rope_dims, "region_rope", true)?;
        if input.num_elements() == 0 {
            return Ok(input);
        }

        let params = RopeParams {
            num_heads: usize_to_u32(num_heads, "region_rope num_heads")?,
            seq_len: usize_to_u32(global_pos.len(), "region_rope sequence length")?,
            head_dim: usize_to_u32(head_dim, "region_rope head dimension")?,
            rope_dims: usize_to_u32(rope_dims, "region_rope_dims")?,
            theta,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let params_buffer = input
            .device
            .create_storage_buffer_from_pod(&params, "gpu-region-rope-params");
        let global_buffer = input
            .device
            .create_storage_buffer_from_i32(global_pos, "gpu-region-rope-global");
        let region_buffer = input
            .device
            .create_storage_buffer_from_i32(region_ids, "gpu-region-rope-region");
        let out_buffer = input
            .device
            .create_empty_storage_buffer(input.num_elements(), "gpu-region-rope-out")?;
        input.device.dispatch_compute(
            &input.device.inner.pipelines.region_rope,
            &[
                &input.buffer,
                &out_buffer,
                &global_buffer,
                &region_buffer,
                &params_buffer,
            ],
            (
                div_ceil_u32(
                    usize_to_u32(rope_dims / 2, "region_rope pairs")?,
                    ROPE_WORKGROUP_X,
                ),
                div_ceil_u32(
                    usize_to_u32(global_pos.len(), "region_rope sequence length")?,
                    ROPE_WORKGROUP_Y,
                ),
                usize_to_u32(num_heads, "region_rope num_heads")?,
            ),
            "region_rope",
            Some((&input.buffer, &out_buffer, input.num_elements())),
        )?;

        Ok(Self {
            buffer: out_buffer,
            storage_elements: input.num_elements(),
            shape: input.shape.clone(),
            strides: contiguous_strides(&input.shape),
            offset: 0,
            device: input.device.clone(),
        })
    }
}
