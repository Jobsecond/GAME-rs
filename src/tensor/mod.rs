mod cpu;
#[cfg(feature = "gpu")]
mod gpu;

#[cfg(test)]
pub(crate) mod tests;

use crate::Result;

pub use cpu::{CpuDevice, CpuTensor};
#[cfg(feature = "gpu")]
pub use gpu::{GpuAdapterSelector, GpuDevice, GpuTensor};

pub trait Tensor: Sized + Clone {
    type Device: Clone;

    fn from_data(data: &[f32], shape: &[usize], device: &Self::Device) -> Result<Self>;
    fn zeros(shape: &[usize], device: &Self::Device) -> Result<Self>;
    fn device(&self) -> &Self::Device;
    fn shape(&self) -> &[usize];
    fn export(&self, buf: &mut [f32]) -> Result<()>;

    fn reshape(self, shape: &[usize]) -> Result<Self>;
    fn transpose(self, dim0: usize, dim1: usize) -> Result<Self>;
    fn contiguous(self) -> Result<Self>;
    fn slice(self, axis: usize, start: usize, end: usize) -> Result<Self>;
    fn concat(parts: &[&Self], axis: usize) -> Result<Self>;

    fn layout_for_attention_heads(self, num_heads: usize, head_dim: usize) -> Result<Self> {
        let shape = self.shape().to_vec();
        if shape.len() != 2 {
            return Err(crate::Error::message(format!(
                "layout_for_attention_heads expects [seq_len, dim], got {:?}",
                shape
            )));
        }
        let seq_len = shape[0];
        let expected = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| crate::Error::message("attention projection dimension overflow"))?;
        if shape[1] != expected {
            return Err(crate::Error::message(format!(
                "layout_for_attention_heads expected last dim {}, got {:?}",
                expected, shape
            )));
        }

        self.reshape(&[seq_len, num_heads, head_dim])?.transpose(0, 1)
    }

    fn merge_attention_heads(self) -> Result<Self> {
        let shape = self.shape().to_vec();
        if shape.len() != 3 {
            return Err(crate::Error::message(format!(
                "merge_attention_heads expects [num_heads, seq_len, head_dim], got {:?}",
                shape
            )));
        }

        let num_heads = shape[0];
        let seq_len = shape[1];
        let head_dim = shape[2];
        let merged_dim = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| crate::Error::message("merge_attention_heads dimension overflow"))?;

        self.transpose(0, 1)?.reshape(&[seq_len, merged_dim])
    }

    fn add(self, rhs: &Self) -> Result<Self>;
    fn mul(self, rhs: &Self) -> Result<Self>;
    fn scale(self, s: f32) -> Result<Self>;
    fn sigmoid(self) -> Result<Self>;

    fn matmul(&self, rhs: &Self) -> Result<Self>;
    fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self>;

    fn rms_norm(self, weight: &Self, eps: f32) -> Result<Self>;
    fn gelu(self) -> Result<Self>;
    fn softmax(self, axis: isize) -> Result<Self>;

    fn rope(
        self,
        positions: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self>;

    fn region_rope(
        self,
        global_pos: &[i32],
        region_ids: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self>;

    fn conv1d_dw(
        self,
        kernel: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
    ) -> Result<Self>;

    fn embedding(table: &Self, indices: &[i32]) -> Result<Self>;
    fn repeat(self, axis: usize, n: usize) -> Result<Self>;
}
