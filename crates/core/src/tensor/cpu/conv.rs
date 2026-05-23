use rayon::prelude::*;

use crate::Result;
use crate::profiler::op_scope_with;

use super::CpuTensor;
use super::util::*;

impl CpuTensor {
    pub(super) fn conv1d_dw(
        self,
        kernel: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let _profile = op_scope_with("cpu.conv1d_dw", || {
            format!(
                "input={:?} kernel={:?} bias={} stride={} padding={} contiguous_input={} contiguous_kernel={}",
                self.shape(),
                kernel.shape(),
                bias.is_some(),
                stride,
                padding,
                self.is_contiguous(),
                kernel.is_contiguous()
            )
        });
        if stride == 0 {
            return Err(invalid_arg("conv1d_dw requires stride > 0"));
        }
        if self.shape().len() != 2 {
            return Err(invalid_arg(format!(
                "conv1d_dw expects input shape [time, channels], got {:?}",
                self.shape()
            )));
        }
        if kernel.shape().len() != 2 {
            return Err(invalid_arg(format!(
                "conv1d_dw kernel must have shape [channels, kernel_size], got {:?}",
                kernel.shape()
            )));
        }

        let (time, channels) = (self.shape()[0], self.shape()[1]);
        let (kernel_channels, kernel_size) = (kernel.shape()[0], kernel.shape()[1]);
        if channels != kernel_channels {
            return Err(invalid_arg(format!(
                "conv1d_dw channel mismatch: input {:?}, kernel {:?}",
                self.shape(),
                kernel.shape()
            )));
        }
        if kernel_size == 0 {
            return Err(invalid_arg(
                "conv1d_dw kernel size must be greater than zero",
            ));
        }
        if let Some(bias) = bias
            && bias.shape() != [channels]
        {
            return Err(invalid_arg(format!(
                "conv1d_dw bias must have shape [{channels}], got {:?}",
                bias.shape()
            )));
        }

        let padded = time
            .checked_add(padding.checked_mul(2).ok_or_else(|| {
                invalid_arg("conv1d_dw padding overflow")
            })?)
            .ok_or_else(|| invalid_arg("conv1d_dw padded size overflow"))?;
        let out_time = if padded < kernel_size {
            0
        } else {
            (padded - kernel_size) / stride + 1
        };

        let bias_data = bias.map(CpuTensor::to_vec).transpose()?;
        self.with_contiguous_data(|input| {
            kernel.with_contiguous_data(|kernel_data| {
                let mut out = vec![0.0; out_time * channels];

                let valid_start = padding;
                let valid_end = time + padding;
                let mid_start = valid_start;
                let mid_end = if valid_end >= kernel_size {
                    let last_valid = (valid_end - kernel_size) / stride + 1;
                    last_valid.min(out_time)
                } else {
                    0
                };

                let process_channel = |out_t: usize,
                                       channel: usize,
                                       input: &[f32],
                                       kernel_data: &[f32],
                                       bias_data: &Option<Vec<f32>>|
                 -> f32 {
                    let mut sum = bias_data.as_ref().map_or(0.0, |b| b[channel]);
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
                    sum
                };

                if should_parallelize(out.len()) && channels > 0 {
                    out.par_chunks_mut(channels)
                        .enumerate()
                        .for_each(|(out_t, out_row)| {
                            if out_t < mid_start || out_t >= mid_end {
                                for (channel, value) in out_row.iter_mut().enumerate() {
                                    *value = process_channel(
                                        out_t,
                                        channel,
                                        input,
                                        kernel_data,
                                        &bias_data,
                                    );
                                }
                            } else {
                                for (channel, value) in out_row.iter_mut().enumerate() {
                                    let mut sum = bias_data.as_ref().map_or(0.0, |b| b[channel]);
                                    for kernel_index in 0..kernel_size {
                                        let input_t = out_t * stride + kernel_index - padding;
                                        sum += input[input_t * channels + channel]
                                            * kernel_data[channel * kernel_size + kernel_index];
                                    }
                                    *value = sum;
                                }
                            }
                        });
                } else {
                    for out_t in 0..out_time {
                        if out_t < mid_start || out_t >= mid_end {
                            for channel in 0..channels {
                                out[out_t * channels + channel] =
                                    process_channel(out_t, channel, input, kernel_data, &bias_data);
                            }
                        } else {
                            for channel in 0..channels {
                                let mut sum = bias_data.as_ref().map_or(0.0, |b| b[channel]);
                                for kernel_index in 0..kernel_size {
                                    let input_t = out_t * stride + kernel_index - padding;
                                    sum += input[input_t * channels + channel]
                                        * kernel_data[channel * kernel_size + kernel_index];
                                }
                                out[out_t * channels + channel] = sum;
                            }
                        }
                    }
                }

                Self::from_owned(out, &[out_time, channels])
            })
        })
    }
}
