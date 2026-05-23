use rayon::prelude::*;

use crate::Result;
use crate::profiler::op_scope_with;

use super::CpuTensor;
use super::util::*;

impl CpuTensor {
    pub(super) fn rms_norm(self, weight: &Self, eps: f32) -> Result<Self> {
        let _profile = op_scope_with("cpu.rms_norm", || {
            format!(
                "input={:?} weight={:?} contiguous_input={} contiguous_weight={}",
                self.shape(),
                weight.shape(),
                self.is_contiguous(),
                weight.is_contiguous()
            )
        });
        if self.shape().is_empty() {
            return Err(invalid_arg(
                "rms_norm expects an input tensor with at least one dimension",
            ));
        }

        let feature_dim = *self.shape().last().unwrap_or(&0);
        if weight.shape() != [feature_dim] {
            return Err(invalid_arg(format!(
                "rms_norm weight must have shape [{feature_dim}], got {:?}",
                weight.shape()
            )));
        }

        let shape = self.shape().to_vec();
        self.with_contiguous_data(|input| {
            weight.with_contiguous_data(|weight_data| {
                let mut data = input.to_vec();
                if feature_dim == 0 {
                    return Self::from_owned(data, &shape);
                }

                if should_parallelize(data.len()) {
                    data.par_chunks_mut(feature_dim).for_each(|row_slice| {
                        let mean_square = row_slice.iter().map(|value| value * value).sum::<f32>()
                            / feature_dim as f32;
                        let inv_rms = 1.0 / (mean_square + eps).sqrt();
                        for (value, scale) in row_slice.iter_mut().zip(weight_data.iter()) {
                            *value *= inv_rms * scale;
                        }
                    });
                } else {
                    let rows = data.len() / feature_dim;
                    for row in 0..rows {
                        let row_start = row * feature_dim;
                        let row_end = row_start + feature_dim;
                        let row_slice = &mut data[row_start..row_end];
                        let mean_square = row_slice.iter().map(|value| value * value).sum::<f32>()
                            / feature_dim as f32;
                        let inv_rms = 1.0 / (mean_square + eps).sqrt();
                        for (value, scale) in row_slice.iter_mut().zip(weight_data.iter()) {
                            *value *= inv_rms * scale;
                        }
                    }
                }

                Self::from_owned(data, &shape)
            })
        })
    }
}
