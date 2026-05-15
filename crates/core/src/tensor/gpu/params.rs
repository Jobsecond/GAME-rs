use bytemuck::{Pod, Zeroable};

use crate::Result;

use super::base::GpuTensor;
use super::util::*;

pub(super) const MAX_DIMS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct LayoutParams {
    pub(super) out_len: u32,
    pub(super) rank: u32,
    pub(super) offset: u32,
    pub(super) _reserved: u32,
    pub(super) shape: [u32; MAX_DIMS],
    pub(super) out_strides: [u32; MAX_DIMS],
    pub(super) src_strides: [u32; MAX_DIMS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ConcatParams {
    pub(super) part_len: u32,
    pub(super) inner: u32,
    pub(super) part_axis_len: u32,
    pub(super) out_axis_len: u32,
    pub(super) axis_offset: u32,
    pub(super) _reserved0: u32,
    pub(super) _reserved1: u32,
    pub(super) _reserved2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ArithmeticParams {
    pub(super) out_len: u32,
    pub(super) out_rank: u32,
    pub(super) lhs_rank: u32,
    pub(super) rhs_rank: u32,
    pub(super) scalar: f32,
    pub(super) _reserved0: u32,
    pub(super) _reserved1: u32,
    pub(super) _reserved2: u32,
    pub(super) out_shape: [u32; MAX_DIMS],
    pub(super) out_strides: [u32; MAX_DIMS],
    pub(super) lhs_shape: [u32; MAX_DIMS],
    pub(super) lhs_strides: [u32; MAX_DIMS],
    pub(super) rhs_shape: [u32; MAX_DIMS],
    pub(super) rhs_strides: [u32; MAX_DIMS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct MatmulParams {
    pub(super) batch: u32,
    pub(super) m: u32,
    pub(super) k: u32,
    pub(super) n: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Conv1dDwParams {
    pub(super) time: u32,
    pub(super) channels: u32,
    pub(super) kernel_size: u32,
    pub(super) stride: u32,
    pub(super) padding: u32,
    pub(super) out_time: u32,
    pub(super) has_bias: u32,
    pub(super) _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct EmbeddingParams {
    pub(super) out_len: u32,
    pub(super) dim: u32,
    pub(super) _reserved0: u32,
    pub(super) _reserved1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct LinearParams {
    pub(super) rows: u32,
    pub(super) in_dim: u32,
    pub(super) out_dim: u32,
    pub(super) has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct RepeatParams {
    pub(super) out_len: u32,
    pub(super) outer: u32,
    pub(super) axis_len: u32,
    pub(super) inner: u32,
    pub(super) repeat_n: u32,
    pub(super) _reserved0: u32,
    pub(super) _reserved1: u32,
    pub(super) _reserved2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct SoftmaxParams {
    pub(super) outer: u32,
    pub(super) axis_len: u32,
    pub(super) inner: u32,
    pub(super) _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct RopeParams {
    pub(super) num_heads: u32,
    pub(super) seq_len: u32,
    pub(super) head_dim: u32,
    pub(super) rope_dims: u32,
    pub(super) theta: f32,
    pub(super) _reserved0: u32,
    pub(super) _reserved1: u32,
    pub(super) _reserved2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct RmsNormParams {
    pub(super) rows: u32,
    pub(super) feature_dim: u32,
    pub(super) eps: f32,
    pub(super) _reserved: u32,
}

impl GpuTensor {
    pub(super) fn pack_dims(
        shape: &[usize],
        strides: &[usize],
    ) -> Result<([u32; MAX_DIMS], [u32; MAX_DIMS])> {
        if shape.len() > MAX_DIMS {
            return Err(invalid_arg(format!(
                "GPU backend currently supports tensors up to rank {MAX_DIMS}, got shape {:?}",
                shape
            )));
        }

        let mut packed_shape = [0u32; MAX_DIMS];
        let mut packed_strides = [0u32; MAX_DIMS];
        for (index, &dim) in shape.iter().enumerate() {
            packed_shape[index] = usize_to_u32(dim, "tensor dimension")?;
        }
        for (index, &stride) in strides.iter().enumerate() {
            packed_strides[index] = usize_to_u32(stride, "tensor stride")?;
        }
        Ok((packed_shape, packed_strides))
    }

    pub(super) fn pack_contiguous_strides(shape: &[usize]) -> Result<[u32; MAX_DIMS]> {
        let strides = contiguous_strides(shape);
        let (_, packed_strides) = Self::pack_dims(shape, &strides)?;
        Ok(packed_strides)
    }
}
