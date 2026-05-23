use std::sync::Arc;

use crate::Result;

use super::util::{checked_num_elements, contiguous_strides, invalid_arg};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuDevice;

#[derive(Debug, Clone)]
pub struct CpuTensor {
    pub(super) data: Arc<Vec<f32>>,
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
    pub(super) offset: usize,
    pub(super) device: CpuDevice,
}

impl CpuTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected = 1usize;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected {
                return false;
            }
            expected *= self.shape[i];
        }
        true
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let n = self.num_elements();
        if n == 0 {
            return Ok(Vec::new());
        }
        if self.is_contiguous() {
            return Ok(self.data[self.offset..self.offset + n].to_vec());
        }
        let mut result = Vec::with_capacity(n);
        let ndims = self.shape.len();
        let mut indices = vec![0usize; ndims];
        for _ in 0..n {
            let mut idx = self.offset;
            for d in 0..ndims {
                idx += indices[d] * self.strides[d];
            }
            result.push(self.data[idx]);
            for d in (0..ndims).rev() {
                indices[d] += 1;
                if indices[d] < self.shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        Ok(result)
    }

    pub(super) fn data_ptr(&self) -> *const f32 {
        self.data[self.offset..].as_ptr()
    }

    pub(super) fn from_owned(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let n = checked_num_elements(shape)?;
        if data.len() != n {
            return Err(invalid_arg(format!(
                "from_owned: data length {} does not match shape {:?} ({} elements)",
                data.len(),
                shape,
                n
            )));
        }
        Ok(Self {
            data: Arc::new(data),
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: 0,
            device: CpuDevice,
        })
    }

    pub(super) fn from_data(data: &[f32], shape: &[usize], _device: &CpuDevice) -> Result<Self> {
        Self::from_owned(data.to_vec(), shape)
    }

    pub(super) fn zeros(shape: &[usize], _device: &CpuDevice) -> Result<Self> {
        let n = checked_num_elements(shape)?;
        Self::from_owned(vec![0.0; n], shape)
    }

    pub(super) fn device(&self) -> &CpuDevice {
        &self.device
    }

    pub(super) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(super) fn with_contiguous_data<R>(&self, f: impl FnOnce(&[f32]) -> Result<R>) -> Result<R> {
        let n = self.num_elements();
        if self.is_contiguous() {
            return f(&self.data[self.offset..self.offset + n]);
        }
        let owned = self.to_vec()?;
        f(&owned)
    }

    pub(super) fn export(&self, buf: &mut [f32]) -> Result<()> {
        let n = self.num_elements();
        if buf.len() != n {
            return Err(invalid_arg(format!(
                "export buffer length {} does not match tensor shape {:?} ({} elements)",
                buf.len(),
                self.shape(),
                n
            )));
        }
        if self.is_contiguous() {
            buf.copy_from_slice(&self.data[self.offset..self.offset + n]);
        } else {
            let values = self.to_vec()?;
            buf.copy_from_slice(&values);
        }
        Ok(())
    }
}
