use crate::Result;

use super::base::GpuTensor;
use super::params::*;
use super::pipelines::*;
use super::util::*;

impl GpuTensor {
    pub(super) fn conv1d_dw(
        self,
        kernel: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        self.ensure_same_device(kernel, "conv1d_dw")?;
        if stride == 0 {
            return Err(invalid_arg("conv1d_dw requires stride > 0"));
        }

        let input = self.contiguous()?;
        let kernel = kernel.clone().contiguous()?;
        if input.shape.len() != 2 {
            return Err(invalid_arg(format!(
                "conv1d_dw expects input shape [time, channels], got {:?}",
                input.shape
            )));
        }
        if kernel.shape.len() != 2 {
            return Err(invalid_arg(format!(
                "conv1d_dw kernel must have shape [channels, kernel_size], got {:?}",
                kernel.shape
            )));
        }

        let (time, channels) = (input.shape[0], input.shape[1]);
        let (kernel_channels, kernel_size) = (kernel.shape[0], kernel.shape[1]);
        if channels != kernel_channels {
            return Err(invalid_arg(format!(
                "conv1d_dw channel mismatch: input {:?}, kernel {:?}",
                input.shape, kernel.shape
            )));
        }
        if kernel_size == 0 {
            return Err(invalid_arg(
                "conv1d_dw kernel size must be greater than zero",
            ));
        }

        let bias_tensor = if let Some(bias) = bias {
            input.ensure_same_device(bias, "conv1d_dw")?;
            let bias = bias.clone().contiguous()?;
            if bias.shape != [channels] {
                return Err(invalid_arg(format!(
                    "conv1d_dw bias must have shape [{channels}], got {:?}",
                    bias.shape
                )));
            }
            Some(bias)
        } else {
            None
        };

        let padded = time.saturating_add(padding.saturating_mul(2));
        let out_time = if padded < kernel_size {
            0
        } else {
            (padded - kernel_size) / stride + 1
        };
        let out_shape = vec![out_time, channels];
        let out_len = checked_num_elements(&out_shape)?;
        if out_len == 0 {
            return Self::zeros(&out_shape, &input.device);
        }

        let params = Conv1dDwParams {
            time: usize_to_u32(time, "conv1d_dw time")?,
            channels: usize_to_u32(channels, "conv1d_dw channels")?,
            kernel_size: usize_to_u32(kernel_size, "conv1d_dw kernel size")?,
            stride: usize_to_u32(stride, "conv1d_dw stride")?,
            padding: usize_to_u32(padding, "conv1d_dw padding")?,
            out_time: usize_to_u32(out_time, "conv1d_dw output time")?,
            has_bias: if bias_tensor.is_some() { 1 } else { 0 },
            _reserved: 0,
        };
        let params_buffer = input
            .device
            .create_storage_buffer_from_pod(&params, "gpu-conv1d-dw-params");
        let out_buffer = input
            .device
            .create_empty_storage_buffer(out_len, "gpu-conv1d-dw-out")?;
        let bias_buffer = bias_tensor
            .as_ref()
            .map(|tensor| &tensor.buffer)
            .unwrap_or(&input.device.inner.dummy_buffer);
        input.device.dispatch_compute(
            &input.device.inner.pipelines.conv1d_dw,
            &[
                &input.buffer,
                &kernel.buffer,
                bias_buffer,
                &out_buffer,
                &params_buffer,
            ],
            (
                div_ceil_u32(params.channels, ROW_WORKGROUP_X),
                div_ceil_u32(params.out_time, ROW_WORKGROUP_Y),
                1,
            ),
            "conv1d_dw",
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
