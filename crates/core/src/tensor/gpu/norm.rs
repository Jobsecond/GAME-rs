use crate::Result;

use super::base::GpuTensor;
use super::params::*;
use super::util::*;

impl GpuTensor {
    pub(super) fn rms_norm(self, weight: &Self, eps: f32) -> Result<Self> {
        self.ensure_same_device(weight, "rms_norm")?;
        if self.shape.is_empty() {
            return Err(invalid_arg(
                "rms_norm expects an input tensor with at least one dimension",
            ));
        }

        let input = self.contiguous()?;
        let weight = weight.clone().contiguous()?;
        let feature_dim = *input.shape.last().unwrap_or(&0);
        if weight.shape != [feature_dim] {
            return Err(invalid_arg(format!(
                "rms_norm weight must have shape [{feature_dim}], got {:?}",
                weight.shape
            )));
        }
        if feature_dim == 0 {
            return Ok(input);
        }

        let rows = input.num_elements() / feature_dim;
        let params = RmsNormParams {
            rows: usize_to_u32(rows, "rms_norm rows")?,
            feature_dim: usize_to_u32(feature_dim, "rms_norm feature dimension")?,
            eps,
            _reserved: 0,
        };
        let params_buffer = input
            .device
            .create_storage_buffer_from_pod(&params, "gpu-rms-norm-params");
        let out_buffer = input
            .device
            .create_empty_storage_buffer(input.num_elements(), "gpu-rms-norm-out")?;
        input.device.dispatch_compute(
            &input.device.inner.pipelines.rms_norm,
            &[&input.buffer, &weight.buffer, &out_buffer, &params_buffer],
            (params.rows, 1, 1),
            "rms_norm",
            None,
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
