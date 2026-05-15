use std::sync::Arc;
use std::sync::mpsc;

use bytemuck::Pod;

use crate::{Error, Result};

use super::pipelines::{Pipelines, create_buffer_with_bytes};
use super::util::*;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuAdapterSelector {
    pub name_substring: Option<String>,
    pub vendor_id: Option<u32>,
    pub device_id: Option<u32>,
    pub backend: Option<wgpu::Backend>,
    pub device_type: Option<wgpu::DeviceType>,
}

#[derive(Clone)]
pub struct GpuDevice {
    pub(super) inner: Arc<GpuContext>,
}

pub(super) struct GpuContext {
    pub(super) adapter_info: wgpu::AdapterInfo,
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) dummy_buffer: wgpu::Buffer,
    pub(super) pipelines: Pipelines,
}

#[derive(Clone)]
pub struct GpuTensor {
    pub(super) buffer: wgpu::Buffer,
    pub(super) storage_elements: usize,
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
    pub(super) offset: usize,
    pub(super) device: GpuDevice,
}

impl GpuDevice {
    pub fn new() -> Result<Self> {
        Self::new_with_selector(None)
    }

    pub fn new_with_selector(selector: Option<&GpuAdapterSelector>) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = select_adapter(&instance, selector)?;
        let adapter_info = adapter.get_info();

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                .map_err(|err| Error::message(format!("failed to request GPU device: {err}")))?;

        let dummy_buffer = create_buffer_with_bytes(
            &device,
            bytemuck::bytes_of(&0.0f32),
            Some("gpu-dummy-buffer"),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let pipelines = Pipelines::new(&device);

        Ok(Self {
            inner: Arc::new(GpuContext {
                adapter_info,
                device,
                queue,
                dummy_buffer,
                pipelines,
            }),
        })
    }

    pub fn available_adapters() -> Vec<wgpu::AdapterInfo> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        instance
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .map(|adapter| adapter.get_info())
            .collect()
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.inner.adapter_info
    }

    pub(super) fn create_storage_buffer_from_f32(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        let contents = if data.is_empty() {
            bytemuck::bytes_of(&0.0f32)
        } else {
            bytemuck::cast_slice(data)
        };
        create_buffer_with_bytes(
            &self.inner.device,
            contents,
            Some(label),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    }

    pub(super) fn create_storage_buffer_from_i32(&self, data: &[i32], label: &str) -> wgpu::Buffer {
        let contents = if data.is_empty() {
            bytemuck::bytes_of(&0i32)
        } else {
            bytemuck::cast_slice(data)
        };
        create_buffer_with_bytes(
            &self.inner.device,
            contents,
            Some(label),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    }

    pub(super) fn create_storage_buffer_from_pod<T: Pod>(
        &self,
        value: &T,
        label: &str,
    ) -> wgpu::Buffer {
        create_buffer_with_bytes(
            &self.inner.device,
            bytemuck::bytes_of(value),
            Some(label),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    }

    pub(super) fn create_empty_storage_buffer(
        &self,
        elements: usize,
        label: &str,
    ) -> Result<wgpu::Buffer> {
        let size = bytes_for_elements(elements)?;
        Ok(self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    pub(super) fn readback_f32(
        &self,
        source: &wgpu::Buffer,
        source_offset_elements: usize,
        elements: usize,
    ) -> Result<Vec<f32>> {
        if elements == 0 {
            return Ok(Vec::new());
        }

        let byte_len = bytes_for_elements(elements)?;
        let source_offset = bytes_for_offset(source_offset_elements)?;
        let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpu-readback-encoder"),
                });
        encoder.copy_buffer_to_buffer(source, source_offset, &staging, 0, byte_len);
        self.inner.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.inner
            .device
            .poll(wgpu::Maintain::wait())
            .panic_on_timeout();
        rx.recv()
            .map_err(|err| Error::message(format!("failed to receive GPU map status: {err}")))?
            .map_err(|err| Error::message(format!("failed to map GPU readback buffer: {err}")))?;

        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        Ok(values)
    }
}

impl GpuTensor {
    pub fn num_elements(&self) -> usize {
        plain_num_elements(&self.shape)
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let mut out = vec![0.0; self.num_elements()];
        self.export(&mut out)?;
        Ok(out)
    }

    pub(super) fn from_owned(data: Vec<f32>, shape: Vec<usize>, device: GpuDevice) -> Result<Self> {
        let expected = checked_num_elements(&shape)?;
        if data.len() != expected {
            return Err(invalid_arg(format!(
                "tensor data length {} does not match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            )));
        }

        Ok(Self {
            buffer: device.create_storage_buffer_from_f32(&data, "gpu-tensor"),
            storage_elements: expected,
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            offset: 0,
            device,
        })
    }

    pub(super) fn ensure_same_device(&self, other: &Self, op_name: &str) -> Result<()> {
        if !Arc::ptr_eq(&self.device.inner, &other.device.inner) {
            return Err(invalid_arg(format!(
                "{op_name} requires tensors on the same GPU device"
            )));
        }
        Ok(())
    }

    pub(super) fn is_dense_contiguous_view(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    pub(super) fn has_compact_storage(&self) -> bool {
        self.offset == 0
            && self.is_dense_contiguous_view()
            && self.storage_elements == self.num_elements()
    }

    pub(super) fn download_storage(&self) -> Result<Vec<f32>> {
        self.device
            .readback_f32(&self.buffer, 0, self.storage_elements)
    }

    pub(super) fn materialize_view_on_cpu(&self) -> Result<Vec<f32>> {
        let backing = self.download_storage()?;
        let mut out = vec![0.0; self.num_elements()];
        for_each_index(&self.shape, |coords, flat| {
            let index = self.offset
                + coords
                    .iter()
                    .zip(&self.strides)
                    .map(|(coord, stride)| coord * stride)
                    .sum::<usize>();
            out[flat] = backing[index];
        });
        Ok(out)
    }

    pub(super) fn from_data(data: &[f32], shape: &[usize], device: &GpuDevice) -> Result<Self> {
        Self::from_owned(data.to_vec(), shape.to_vec(), device.clone())
    }

    pub(super) fn zeros(shape: &[usize], device: &GpuDevice) -> Result<Self> {
        let len = checked_num_elements(shape)?;
        let zeroes = vec![0.0; len];
        Self::from_owned(zeroes, shape.to_vec(), device.clone())
    }

    pub(super) fn device(&self) -> &GpuDevice {
        &self.device
    }

    pub(super) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(super) fn export(&self, buf: &mut [f32]) -> Result<()> {
        let expected = self.num_elements();
        if buf.len() != expected {
            return Err(invalid_arg(format!(
                "export buffer length {} does not match tensor shape {:?} ({} elements)",
                buf.len(),
                self.shape,
                expected
            )));
        }

        if self.has_compact_storage() {
            let values = self.device.readback_f32(&self.buffer, 0, expected)?;
            buf.copy_from_slice(&values);
            return Ok(());
        }

        if self.is_dense_contiguous_view() {
            let values = self
                .device
                .readback_f32(&self.buffer, self.offset, expected)?;
            buf.copy_from_slice(&values);
            return Ok(());
        }

        let contiguous = self.clone().contiguous()?;
        contiguous.export(buf)
    }
}

impl GpuAdapterSelector {
    pub(super) fn is_empty(&self) -> bool {
        self.name_substring.is_none()
            && self.vendor_id.is_none()
            && self.device_id.is_none()
            && self.backend.is_none()
            && self.device_type.is_none()
    }

    pub(super) fn matches(&self, info: &wgpu::AdapterInfo) -> bool {
        if let Some(name_substring) = self.name_substring.as_deref() {
            let needle = name_substring.to_ascii_lowercase();
            if !info.name.to_ascii_lowercase().contains(&needle) {
                return false;
            }
        }
        if let Some(vendor_id) = self.vendor_id
            && info.vendor != vendor_id
        {
            return false;
        }
        if let Some(device_id) = self.device_id
            && info.device != device_id
        {
            return false;
        }
        if let Some(backend) = self.backend
            && info.backend != backend
        {
            return false;
        }
        if let Some(device_type) = self.device_type
            && info.device_type != device_type
        {
            return false;
        }

        true
    }

    pub(super) fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(name_substring) = self.name_substring.as_deref() {
            parts.push(format!("name contains `{name_substring}`"));
        }
        if let Some(vendor_id) = self.vendor_id {
            parts.push(format!("vendor_id=0x{vendor_id:04x}"));
        }
        if let Some(device_id) = self.device_id {
            parts.push(format!("device_id=0x{device_id:04x}"));
        }
        if let Some(backend) = self.backend {
            parts.push(format!("backend={backend}"));
        }
        if let Some(device_type) = self.device_type {
            parts.push(format!("device_type={device_type:?}"));
        }

        if parts.is_empty() {
            "any adapter".to_string()
        } else {
            parts.join(", ")
        }
    }
}

pub(super) fn select_adapter(
    instance: &wgpu::Instance,
    selector: Option<&GpuAdapterSelector>,
) -> Result<wgpu::Adapter> {
    if let Some(selector) = selector
        && !selector.is_empty()
    {
        return select_adapter_explicit(instance, selector);
    }

    if let Some(adapter) = wgpu::util::initialize_adapter_from_env(instance, None) {
        return Ok(adapter);
    }

    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::from_env().unwrap_or_default(),
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok_or_else(|| Error::message("failed to find a suitable GPU adapter"))
}

pub(super) fn select_adapter_explicit(
    instance: &wgpu::Instance,
    selector: &GpuAdapterSelector,
) -> Result<wgpu::Adapter> {
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let mut matches = adapters
        .into_iter()
        .filter_map(|adapter| {
            let info = adapter.get_info();
            selector.matches(&info).then_some((adapter, info))
        })
        .collect::<Vec<_>>();

    matches.sort_by_key(|(_, info)| {
        std::cmp::Reverse((
            adapter_device_type_priority(info.device_type),
            adapter_backend_priority(info.backend),
        ))
    });

    matches
        .into_iter()
        .map(|(adapter, _)| adapter)
        .next()
        .ok_or_else(|| {
            let available = GpuDevice::available_adapters()
                .into_iter()
                .map(|info| format_adapter_info(&info))
                .collect::<Vec<_>>()
                .join("; ");
            Error::message(format!(
                "no GPU adapter matched selector ({}){}",
                selector.describe(),
                if available.is_empty() {
                    String::new()
                } else {
                    format!("; available adapters: {available}")
                }
            ))
        })
}

pub(super) fn format_adapter_info(info: &wgpu::AdapterInfo) -> String {
    format!(
        "{} [vendor=0x{:04x}, device=0x{:04x}, backend={}, type={:?}]",
        info.name, info.vendor, info.device, info.backend, info.device_type
    )
}

pub(super) fn adapter_device_type_priority(device_type: wgpu::DeviceType) -> u8 {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => 5,
        wgpu::DeviceType::IntegratedGpu => 4,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 1,
    }
}

pub(super) fn adapter_backend_priority(backend: wgpu::Backend) -> u8 {
    match backend {
        wgpu::Backend::Vulkan => 5,
        wgpu::Backend::Metal => 4,
        wgpu::Backend::Dx12 => 3,
        wgpu::Backend::Gl => 2,
        wgpu::Backend::BrowserWebGpu => 1,
        wgpu::Backend::Empty => 0,
    }
}
