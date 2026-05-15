mod attention;
mod base;
mod conv;
mod elementwise;
mod indexing;
mod layout;
mod matmul;
mod norm;
mod rope;
mod util;

use crate::Result;

use super::Tensor;

pub use base::{CpuDevice, CpuTensor};

#[cfg(all(
    feature = "cpu-attention-gemm-gemm",
    feature = "cpu-attention-gemm-matrixmultiply"
))]
compile_error!(
    "cpu-attention-gemm-gemm and cpu-attention-gemm-matrixmultiply are mutually exclusive"
);

impl Tensor for CpuTensor {
    type Device = CpuDevice;

    fn from_data(data: &[f32], shape: &[usize], _device: &Self::Device) -> Result<Self> {
        CpuTensor::from_data(data, shape, _device)
    }

    fn zeros(shape: &[usize], _device: &Self::Device) -> Result<Self> {
        CpuTensor::zeros(shape, _device)
    }

    fn device(&self) -> &Self::Device {
        CpuTensor::device(self)
    }

    fn shape(&self) -> &[usize] {
        CpuTensor::shape(self)
    }

    fn export(&self, buf: &mut [f32]) -> Result<()> {
        CpuTensor::export(self, buf)
    }

    fn reshape(self, shape: &[usize]) -> Result<Self> {
        CpuTensor::reshape(self, shape)
    }

    fn transpose(self, dim0: usize, dim1: usize) -> Result<Self> {
        CpuTensor::transpose(self, dim0, dim1)
    }

    fn contiguous(self) -> Result<Self> {
        CpuTensor::contiguous(self)
    }

    fn slice(self, axis: usize, start: usize, end: usize) -> Result<Self> {
        CpuTensor::slice(self, axis, start, end)
    }

    fn layout_for_attention_heads(self, num_heads: usize, head_dim: usize) -> Result<Self> {
        CpuTensor::layout_for_attention_heads(self, num_heads, head_dim)
    }

    fn split_last_dim_two_for_attention_heads(
        self,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(Self, Self)> {
        CpuTensor::split_last_dim_two_for_attention_heads(self, num_heads, head_dim)
    }

    fn split_last_dim_three_for_attention_heads(
        self,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(Self, Self, Self)> {
        CpuTensor::split_last_dim_three_for_attention_heads(self, num_heads, head_dim)
    }

    fn merge_attention_heads(self) -> Result<Self> {
        CpuTensor::merge_attention_heads(self)
    }

    fn concat(parts: &[&Self], axis: usize) -> Result<Self> {
        CpuTensor::concat(parts, axis)
    }

    fn add(self, rhs: &Self) -> Result<Self> {
        CpuTensor::add(self, rhs)
    }

    fn mul(self, rhs: &Self) -> Result<Self> {
        CpuTensor::mul(self, rhs)
    }

    fn scale(self, s: f32) -> Result<Self> {
        CpuTensor::scale(self, s)
    }

    fn sigmoid(self) -> Result<Self> {
        CpuTensor::sigmoid(self)
    }

    fn split_last_dim_two_gelu_mul(self) -> Result<Self> {
        CpuTensor::split_last_dim_two_gelu_mul(self)
    }

    fn matmul(&self, rhs: &Self) -> Result<Self> {
        CpuTensor::matmul(self, rhs)
    }

    fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self> {
        CpuTensor::linear(self, weight, bias)
    }

    fn attention_score_softmax(
        q: &Self,
        k_t: &Self,
        mask: Option<&Self>,
        scale: f32,
    ) -> Result<Self> {
        CpuTensor::attention_score_softmax(q, k_t, mask, scale)
    }

    fn attention_value_matmul(probs: &Self, v: &Self) -> Result<Self> {
        CpuTensor::attention_value_matmul(probs, v)
    }

    fn fused_attention(
        q: &Self,
        k: &Self,
        v: &Self,
        mask: Option<&Self>,
        scale: f32,
    ) -> Result<Self> {
        CpuTensor::fused_attention(q, k, v, mask, scale)
    }

    fn rms_norm(self, weight: &Self, eps: f32) -> Result<Self> {
        CpuTensor::rms_norm(self, weight, eps)
    }

    fn gelu(self) -> Result<Self> {
        CpuTensor::gelu(self)
    }

    fn softmax(self, axis: isize) -> Result<Self> {
        CpuTensor::softmax(self, axis)
    }

    fn rope(
        self,
        positions: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self> {
        CpuTensor::rope(self, positions, head_dim, num_heads, rope_dims, theta)
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
        CpuTensor::region_rope(
            self, global_pos, region_ids, head_dim, num_heads, rope_dims, theta,
        )
    }

    fn conv1d_dw(
        self,
        kernel: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        CpuTensor::conv1d_dw(self, kernel, bias, stride, padding)
    }

    fn embedding(table: &Self, indices: &[i32]) -> Result<Self> {
        CpuTensor::embedding(table, indices)
    }

    fn repeat(self, axis: usize, n: usize) -> Result<Self> {
        CpuTensor::repeat(self, axis, n)
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
    fn fused_attention_matches_reference() {
        tests::run_fused_attention_matches_reference::<CpuTensor>(&CpuDevice);
    }

    #[test]
    fn roundtrip_matches_uploaded_values() {
        tests::run_roundtrip::<CpuTensor>(&CpuDevice);
    }
}
