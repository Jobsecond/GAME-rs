mod base;
mod conv;
mod elementwise;
mod indexing;
mod layout;
mod matmul;
mod norm;
mod params;
mod pipelines;
mod rope;
mod util;

use crate::Result;

use super::Tensor;

pub use base::{GpuAdapterSelector, GpuDevice, GpuTensor};
#[cfg(test)]
use base::{adapter_backend_priority, adapter_device_type_priority};

impl Tensor for GpuTensor {
    type Device = GpuDevice;

    fn from_data(data: &[f32], shape: &[usize], device: &Self::Device) -> Result<Self> {
        GpuTensor::from_data(data, shape, device)
    }

    fn zeros(shape: &[usize], device: &Self::Device) -> Result<Self> {
        GpuTensor::zeros(shape, device)
    }

    fn device(&self) -> &Self::Device {
        GpuTensor::device(self)
    }

    fn shape(&self) -> &[usize] {
        GpuTensor::shape(self)
    }

    fn export(&self, buf: &mut [f32]) -> Result<()> {
        GpuTensor::export(self, buf)
    }

    fn reshape(self, shape: &[usize]) -> Result<Self> {
        GpuTensor::reshape(self, shape)
    }

    fn transpose(self, dim0: usize, dim1: usize) -> Result<Self> {
        GpuTensor::transpose(self, dim0, dim1)
    }

    fn contiguous(self) -> Result<Self> {
        GpuTensor::contiguous(self)
    }

    fn slice(self, axis: usize, start: usize, end: usize) -> Result<Self> {
        GpuTensor::slice(self, axis, start, end)
    }

    fn concat(parts: &[&Self], axis: usize) -> Result<Self> {
        GpuTensor::concat(parts, axis)
    }

    fn add(self, rhs: &Self) -> Result<Self> {
        GpuTensor::add(self, rhs)
    }

    fn mul(self, rhs: &Self) -> Result<Self> {
        GpuTensor::mul(self, rhs)
    }

    fn scale(self, s: f32) -> Result<Self> {
        GpuTensor::scale(self, s)
    }

    fn sigmoid(self) -> Result<Self> {
        GpuTensor::sigmoid(self)
    }

    fn matmul(&self, rhs: &Self) -> Result<Self> {
        GpuTensor::matmul(self, rhs)
    }

    fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self> {
        GpuTensor::linear(self, weight, bias)
    }

    fn rms_norm(self, weight: &Self, eps: f32) -> Result<Self> {
        GpuTensor::rms_norm(self, weight, eps)
    }

    fn gelu(self) -> Result<Self> {
        GpuTensor::gelu(self)
    }

    fn softmax(self, axis: isize) -> Result<Self> {
        GpuTensor::softmax(self, axis)
    }

    fn rope(
        self,
        positions: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self> {
        GpuTensor::rope(self, positions, head_dim, num_heads, rope_dims, theta)
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
        GpuTensor::region_rope(
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
        GpuTensor::conv1d_dw(self, kernel, bias, stride, padding)
    }

    fn embedding(table: &Self, indices: &[i32]) -> Result<Self> {
        GpuTensor::embedding(table, indices)
    }

    fn repeat(self, axis: usize, n: usize) -> Result<Self> {
        GpuTensor::repeat(self, axis, n)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GpuAdapterSelector, GpuDevice, GpuTensor, adapter_backend_priority,
        adapter_device_type_priority,
    };
    use crate::tensor::tests;

    fn with_gpu_device(f: impl FnOnce(&GpuDevice)) {
        match GpuDevice::new() {
            Ok(device) => f(&device),
            Err(err) => eprintln!("skipping GPU tensor test: {err}"),
        }
    }

    #[test]
    fn roundtrip_matches_uploaded_values() {
        with_gpu_device(|device| tests::run_roundtrip::<GpuTensor>(device));
    }

    #[test]
    fn layout_ops_preserve_view_semantics() {
        with_gpu_device(|device| {
            tests::run_layout_ops_preserve_view_semantics::<GpuTensor>(device)
        });
    }

    #[test]
    fn broadcast_add_and_mul_match_expected_values() {
        with_gpu_device(|device| {
            tests::run_broadcast_add_and_mul_match_expected_values::<GpuTensor>(device)
        });
    }

    #[test]
    fn matmul_supports_2d_and_batched_3d_inputs() {
        with_gpu_device(|device| {
            tests::run_matmul_supports_2d_and_batched_3d_inputs::<GpuTensor>(device)
        });
    }

    #[test]
    fn linear_applies_weight_rows_and_optional_bias() {
        with_gpu_device(|device| {
            tests::run_linear_applies_weight_rows_and_optional_bias::<GpuTensor>(device)
        });
    }

    #[test]
    fn normalization_and_activation_ops_match_reference_values() {
        with_gpu_device(|device| {
            tests::run_normalization_and_activation_ops_match_reference_values::<GpuTensor>(device)
        });
    }

    #[test]
    fn rope_rotates_each_head_using_global_positions() {
        with_gpu_device(|device| {
            tests::run_rope_rotates_each_head_using_global_positions::<GpuTensor>(device)
        });
    }

    #[test]
    fn region_rope_splits_global_and_region_rotation_halves() {
        with_gpu_device(|device| {
            tests::run_region_rope_splits_global_and_region_rotation_halves::<GpuTensor>(device)
        });
    }

    #[test]
    fn depthwise_conv_applies_per_channel_kernels() {
        with_gpu_device(|device| {
            tests::run_depthwise_conv_applies_per_channel_kernels::<GpuTensor>(device)
        });
    }

    #[test]
    fn embedding_and_repeat_return_expected_rows() {
        with_gpu_device(|device| {
            tests::run_embedding_and_repeat_return_expected_rows::<GpuTensor>(device)
        });
    }

    #[test]
    fn gpu_matches_cpu_reference_for_add_then_softmax() {
        with_gpu_device(|device| {
            tests::run_gpu_against_cpu_reference::<GpuTensor>(device).unwrap()
        });
    }

    #[test]
    fn selector_matches_expected_adapter_fields() {
        let selector = GpuAdapterSelector {
            name_substring: Some("nvidia".to_string()),
            vendor_id: Some(0x10de),
            device_id: Some(0x2484),
            backend: Some(wgpu::Backend::Vulkan),
            device_type: Some(wgpu::DeviceType::DiscreteGpu),
        };
        let matching = wgpu::AdapterInfo {
            name: "NVIDIA GeForce RTX".to_string(),
            vendor: 0x10de,
            device: 0x2484,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };
        let wrong_vendor = wgpu::AdapterInfo {
            vendor: 0x1002,
            ..matching.clone()
        };
        let wrong_name = wgpu::AdapterInfo {
            name: "Intel Arc".to_string(),
            ..matching.clone()
        };

        assert!(selector.matches(&matching));
        assert!(!selector.matches(&wrong_vendor));
        assert!(!selector.matches(&wrong_name));
    }

    #[test]
    fn selector_describe_includes_all_explicit_fields() {
        let selector = GpuAdapterSelector {
            name_substring: Some("RTX".to_string()),
            vendor_id: Some(0x10de),
            device_id: Some(0x2484),
            backend: Some(wgpu::Backend::Vulkan),
            device_type: Some(wgpu::DeviceType::DiscreteGpu),
        };

        let description = selector.describe();
        assert!(description.contains("RTX"));
        assert!(description.contains("0x10de"));
        assert!(description.contains("0x2484"));
        assert!(description.contains("vulkan"));
        assert!(description.contains("DiscreteGpu"));
    }

    #[test]
    fn selector_empty_only_when_no_constraints_are_set() {
        assert!(GpuAdapterSelector::default().is_empty());
        assert!(
            !GpuAdapterSelector {
                name_substring: Some("AMD".to_string()),
                ..GpuAdapterSelector::default()
            }
            .is_empty()
        );
    }

    #[test]
    fn adapter_priority_prefers_discrete_then_vulkan() {
        assert!(
            adapter_device_type_priority(wgpu::DeviceType::DiscreteGpu)
                > adapter_device_type_priority(wgpu::DeviceType::IntegratedGpu)
        );
        assert!(
            adapter_backend_priority(wgpu::Backend::Vulkan)
                > adapter_backend_priority(wgpu::Backend::Dx12)
        );
    }
}
