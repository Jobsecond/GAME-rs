use crate::Result;

use super::base::GpuTensor;
use super::params::*;
use super::pipelines::*;
use super::util::*;

impl GpuTensor {
    pub(super) fn matmul(&self, rhs: &Self) -> Result<Self> {
        self.ensure_same_device(rhs, "matmul")?;
        let lhs = self.clone().contiguous()?;
        let rhs = rhs.clone().contiguous()?;
        let device = lhs.device.clone();

        let (batch, m, k, n, out_shape) = match (lhs.shape.len(), rhs.shape.len()) {
            (2, 2) => {
                let (m, k) = (lhs.shape[0], lhs.shape[1]);
                let (rhs_k, n) = (rhs.shape[0], rhs.shape[1]);
                if k != rhs_k {
                    return Err(invalid_arg(format!(
                        "matmul shape mismatch: {:?} @ {:?}",
                        lhs.shape, rhs.shape
                    )));
                }
                (1usize, m, k, n, vec![m, n])
            }
            (3, 3) => {
                let (batch, m, k) = (lhs.shape[0], lhs.shape[1], lhs.shape[2]);
                let (rhs_batch, rhs_k, n) = (rhs.shape[0], rhs.shape[1], rhs.shape[2]);
                if batch != rhs_batch || k != rhs_k {
                    return Err(invalid_arg(format!(
                        "batched matmul shape mismatch: {:?} @ {:?}",
                        lhs.shape, rhs.shape
                    )));
                }
                (batch, m, k, n, vec![batch, m, n])
            }
            _ => {
                return Err(invalid_arg(format!(
                    "matmul expects rank-2 or rank-3 tensors, got {:?} and {:?}",
                    lhs.shape, rhs.shape
                )));
            }
        };

        let out_len = checked_num_elements(&out_shape)?;
        if out_len == 0 {
            return Self::zeros(&out_shape, &device);
        }

        let params = MatmulParams {
            batch: usize_to_u32(batch, "matmul batch")?,
            m: usize_to_u32(m, "matmul rows")?,
            k: usize_to_u32(k, "matmul shared dimension")?,
            n: usize_to_u32(n, "matmul columns")?,
        };
        let params_buffer = device.create_storage_buffer_from_pod(&params, "gpu-matmul-params");
        let out_buffer = device.create_empty_storage_buffer(out_len, "gpu-matmul-out")?;
        device.dispatch_compute(
            &device.inner.pipelines.matmul,
            &[&lhs.buffer, &rhs.buffer, &out_buffer, &params_buffer],
            (
                div_ceil_u32(params.n, ROW_WORKGROUP_X),
                div_ceil_u32(params.m, ROW_WORKGROUP_Y),
                params.batch,
            ),
            "matmul",
            None,
        )?;

        Ok(Self {
            buffer: out_buffer,
            storage_elements: out_len,
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
            device,
        })
    }

    pub(super) fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self> {
        self.ensure_same_device(weight, "linear")?;
        if self.shape.is_empty() {
            return Err(invalid_arg(
                "linear expects an input tensor with at least one dimension",
            ));
        }

        let input = self.clone().contiguous()?;
        let weight = weight.clone().contiguous()?;
        if weight.shape.len() != 2 {
            return Err(invalid_arg(format!(
                "linear weight must be rank-2 [out_dim, in_dim], got {:?}",
                weight.shape
            )));
        }

        let input_shape = input.shape.clone();
        let in_dim = *input_shape.last().unwrap_or(&0);
        let out_dim = weight.shape[0];
        if weight.shape[1] != in_dim {
            return Err(invalid_arg(format!(
                "linear shape mismatch: input {:?}, weight {:?}",
                input_shape, weight.shape
            )));
        }

        let bias_tensor = if let Some(bias) = bias {
            input.ensure_same_device(bias, "linear")?;
            let bias = bias.clone().contiguous()?;
            if bias.shape != [out_dim] {
                return Err(invalid_arg(format!(
                    "linear bias must have shape [{out_dim}], got {:?}",
                    bias.shape
                )));
            }
            Some(bias)
        } else {
            None
        };

        let rows = plain_num_elements(&input_shape[..input_shape.len() - 1]);
        let mut out_shape = input_shape[..input_shape.len() - 1].to_vec();
        out_shape.push(out_dim);
        let out_len = checked_num_elements(&out_shape)?;
        if out_len == 0 {
            return Self::zeros(&out_shape, &input.device);
        }

        let params = LinearParams {
            rows: usize_to_u32(rows, "linear rows")?,
            in_dim: usize_to_u32(in_dim, "linear input dimension")?,
            out_dim: usize_to_u32(out_dim, "linear output dimension")?,
            has_bias: if bias_tensor.is_some() { 1 } else { 0 },
        };
        let params_buffer = input
            .device
            .create_storage_buffer_from_pod(&params, "gpu-linear-params");
        let out_buffer = input
            .device
            .create_empty_storage_buffer(out_len, "gpu-linear-out")?;
        let bias_buffer = bias_tensor
            .as_ref()
            .map(|tensor| &tensor.buffer)
            .unwrap_or(&input.device.inner.dummy_buffer);
        input.device.dispatch_compute(
            &input.device.inner.pipelines.linear,
            &[
                &input.buffer,
                &weight.buffer,
                bias_buffer,
                &out_buffer,
                &params_buffer,
            ],
            (
                div_ceil_u32(params.out_dim, ROW_WORKGROUP_X),
                div_ceil_u32(params.rows, ROW_WORKGROUP_Y),
                1,
            ),
            "linear",
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
