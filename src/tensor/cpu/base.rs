use candle_core::{DType, Device, Storage, Tensor as CandleTensor};

use crate::Result;

use super::util::invalid_arg;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuDevice;

#[derive(Debug, Clone)]
pub struct CpuTensor {
    pub(super) tensor: CandleTensor,
    pub(super) device: CpuDevice,
}

impl CpuTensor {
    pub fn num_elements(&self) -> usize {
        self.tensor.elem_count()
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        Ok(self.tensor.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
    }

    pub(super) fn from_tensor(tensor: CandleTensor) -> Self {
        Self {
            tensor,
            device: CpuDevice,
        }
    }

    pub(super) fn from_owned(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        if shape.is_empty() {
            if data.len() != 1 {
                return Err(invalid_arg(format!(
                    "scalar tensor requires exactly one element, got {}",
                    data.len()
                )));
            }
            return Ok(Self::from_tensor(CandleTensor::new(data[0], &Device::Cpu)?));
        }
        Ok(Self::from_tensor(CandleTensor::from_vec(
            data,
            shape.to_vec(),
            &Device::Cpu,
        )?))
    }

    pub(super) fn with_contiguous_data<R>(&self, f: impl FnOnce(&[f32]) -> Result<R>) -> Result<R> {
        let (storage, layout) = self.tensor.storage_and_layout();
        if let Storage::Cpu(storage) = &*storage
            && let Some((start, end)) = layout.contiguous_offsets()
        {
            let data = storage.as_slice::<f32>()?;
            return f(&data[start..end]);
        }

        let owned = self.to_vec()?;
        f(&owned)
    }

    pub(super) fn from_data(data: &[f32], shape: &[usize], _device: &CpuDevice) -> Result<Self> {
        Self::from_owned(data.to_vec(), shape)
    }

    pub(super) fn zeros(shape: &[usize], _device: &CpuDevice) -> Result<Self> {
        Ok(Self::from_tensor(CandleTensor::zeros(
            shape.to_vec(),
            DType::F32,
            &Device::Cpu,
        )?))
    }

    pub(super) fn device(&self) -> &CpuDevice {
        &self.device
    }

    pub(super) fn shape(&self) -> &[usize] {
        self.tensor.dims()
    }

    pub(super) fn export(&self, buf: &mut [f32]) -> Result<()> {
        let values = self.to_vec()?;
        if buf.len() != values.len() {
            return Err(invalid_arg(format!(
                "export buffer length {} does not match tensor shape {:?} ({} elements)",
                buf.len(),
                self.shape(),
                values.len()
            )));
        }
        buf.copy_from_slice(&values);
        Ok(())
    }
}
