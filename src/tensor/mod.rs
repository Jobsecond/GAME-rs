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
    fn shape(&self) -> &[usize];
    fn export(&self, buf: &mut [f32]) -> Result<()>;

    fn reshape(self, shape: &[usize]) -> Result<Self>;
    fn transpose(self, dim0: usize, dim1: usize) -> Result<Self>;
    fn contiguous(self) -> Result<Self>;
    fn slice(self, axis: usize, start: usize, end: usize) -> Result<Self>;
    fn concat(parts: &[&Self], axis: usize) -> Result<Self>;

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
