use std::sync::Arc;
use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{Error, Result};

use super::{CpuDevice, CpuTensor, Tensor};

const MAX_DIMS: usize = 8;
const ELEMENT_WORKGROUP_SIZE: u32 = 64;
const ROW_WORKGROUP_X: u32 = 8;
const ROW_WORKGROUP_Y: u32 = 8;
const ROPE_WORKGROUP_X: u32 = 32;
const ROPE_WORKGROUP_Y: u32 = 4;

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
    inner: Arc<GpuContext>,
}

struct GpuContext {
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    dummy_buffer: wgpu::Buffer,
    pipelines: Pipelines,
}

struct Pipelines {
    contiguous: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    mul: wgpu::ComputePipeline,
    scale: wgpu::ComputePipeline,
    sigmoid: wgpu::ComputePipeline,
    gelu: wgpu::ComputePipeline,
    rms_norm: wgpu::ComputePipeline,
    matmul: wgpu::ComputePipeline,
    linear: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    region_rope: wgpu::ComputePipeline,
}

#[derive(Clone)]
pub struct GpuTensor {
    buffer: wgpu::Buffer,
    storage_elements: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    device: GpuDevice,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayoutParams {
    out_len: u32,
    rank: u32,
    offset: u32,
    _reserved: u32,
    shape: [u32; MAX_DIMS],
    out_strides: [u32; MAX_DIMS],
    src_strides: [u32; MAX_DIMS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ArithmeticParams {
    out_len: u32,
    out_rank: u32,
    lhs_rank: u32,
    rhs_rank: u32,
    scalar: f32,
    _reserved0: u32,
    _reserved1: u32,
    _reserved2: u32,
    out_shape: [u32; MAX_DIMS],
    out_strides: [u32; MAX_DIMS],
    lhs_shape: [u32; MAX_DIMS],
    lhs_strides: [u32; MAX_DIMS],
    rhs_shape: [u32; MAX_DIMS],
    rhs_strides: [u32; MAX_DIMS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatmulParams {
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearParams {
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SoftmaxParams {
    outer: u32,
    axis_len: u32,
    inner: u32,
    _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RopeParams {
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    rope_dims: u32,
    theta: f32,
    _reserved0: u32,
    _reserved1: u32,
    _reserved2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RmsNormParams {
    rows: u32,
    feature_dim: u32,
    eps: f32,
    _reserved: u32,
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

    fn create_storage_buffer_from_f32(&self, data: &[f32], label: &str) -> wgpu::Buffer {
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

    fn create_storage_buffer_from_i32(&self, data: &[i32], label: &str) -> wgpu::Buffer {
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

    fn create_storage_buffer_from_pod<T: Pod>(&self, value: &T, label: &str) -> wgpu::Buffer {
        create_buffer_with_bytes(
            &self.inner.device,
            bytemuck::bytes_of(value),
            Some(label),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    }

    fn create_empty_storage_buffer(&self, elements: usize, label: &str) -> Result<wgpu::Buffer> {
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

    fn readback_f32(
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

    fn dispatch_compute(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        workgroups: (u32, u32, u32),
        label: &str,
        pre_copy: Option<(&wgpu::Buffer, &wgpu::Buffer, usize)>,
    ) -> Result<()> {
        let needs_dispatch = workgroups.0 > 0 && workgroups.1 > 0 && workgroups.2 > 0;
        let needs_copy = pre_copy.is_some();
        if !needs_dispatch && !needs_copy {
            return Ok(());
        }

        let layout = pipeline.get_bind_group_layout(0);
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = self
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &entries,
            });

        let mut encoder = self
            .inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });

        if let Some((src, dst, elements)) = pre_copy
            && elements > 0
        {
            encoder.copy_buffer_to_buffer(src, 0, dst, 0, bytes_for_elements(elements)?);
        }

        if needs_dispatch {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        self.inner.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
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

    fn from_owned(data: Vec<f32>, shape: Vec<usize>, device: GpuDevice) -> Result<Self> {
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

    fn from_cpu_tensor(tensor: CpuTensor, device: &GpuDevice) -> Result<Self> {
        let shape = tensor.shape().to_vec();
        let data = tensor.to_vec()?;
        Self::from_owned(data, shape, device.clone())
    }

    fn to_cpu_tensor(&self) -> Result<CpuTensor> {
        let data = self.to_vec()?;
        CpuTensor::from_data(&data, &self.shape, &CpuDevice)
    }

    fn ensure_same_device(&self, other: &Self, op_name: &str) -> Result<()> {
        if !Arc::ptr_eq(&self.device.inner, &other.device.inner) {
            return Err(invalid_arg(format!(
                "{op_name} requires tensors on the same GPU device"
            )));
        }
        Ok(())
    }

    fn is_dense_contiguous_view(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    fn has_compact_storage(&self) -> bool {
        self.offset == 0
            && self.is_dense_contiguous_view()
            && self.storage_elements == self.num_elements()
    }

    fn download_storage(&self) -> Result<Vec<f32>> {
        self.device
            .readback_f32(&self.buffer, 0, self.storage_elements)
    }

    fn materialize_view_on_cpu(&self) -> Result<Vec<f32>> {
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

    fn pack_dims(shape: &[usize], strides: &[usize]) -> Result<([u32; MAX_DIMS], [u32; MAX_DIMS])> {
        if shape.len() > MAX_DIMS {
            return Err(invalid_arg(format!(
                "GPU backend currently supports tensors up to rank {MAX_DIMS}, got shape {:?}",
                shape
            )));
        }

        let mut packed_shape = [0u32; MAX_DIMS];
        let mut packed_strides = [0u32; MAX_DIMS];
        for (index, &dim) in shape.iter().enumerate() {
            packed_shape[index] = usize_to_u32(dim, "tensor dimension")?;
        }
        for (index, &stride) in strides.iter().enumerate() {
            packed_strides[index] = usize_to_u32(stride, "tensor stride")?;
        }
        Ok((packed_shape, packed_strides))
    }

    fn pack_contiguous_strides(shape: &[usize]) -> Result<[u32; MAX_DIMS]> {
        let strides = contiguous_strides(shape);
        let (_, packed_strides) = Self::pack_dims(shape, &strides)?;
        Ok(packed_strides)
    }

    fn cpu_fallback_unary(
        self,
        op_name: &str,
        f: impl FnOnce(CpuTensor) -> Result<CpuTensor>,
    ) -> Result<Self> {
        let device = self.device.clone();
        let tensor = self.to_cpu_tensor()?;
        let out = f(tensor)
            .map_err(|err| Error::message(format!("CPU fallback for {op_name} failed: {err}")))?;
        Self::from_cpu_tensor(out, &device)
    }

    fn cpu_fallback_binary(
        self,
        rhs: &Self,
        op_name: &str,
        f: impl FnOnce(CpuTensor, &CpuTensor) -> Result<CpuTensor>,
    ) -> Result<Self> {
        self.ensure_same_device(rhs, op_name)?;
        let device = self.device.clone();
        let lhs = self.to_cpu_tensor()?;
        let rhs_cpu = rhs.to_cpu_tensor()?;
        let out = f(lhs, &rhs_cpu)
            .map_err(|err| Error::message(format!("CPU fallback for {op_name} failed: {err}")))?;
        Self::from_cpu_tensor(out, &device)
    }

    fn arithmetic(
        self,
        rhs: &Self,
        pipeline: &wgpu::ComputePipeline,
        op_name: &str,
    ) -> Result<Self> {
        self.ensure_same_device(rhs, op_name)?;
        let lhs = self.contiguous()?;
        let rhs = rhs.clone().contiguous()?;
        let out_shape = broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out_len = checked_num_elements(&out_shape)?;
        if out_len == 0 {
            return Self::zeros(&out_shape, &lhs.device);
        }

        let (out_shape_packed, out_strides_packed) =
            Self::pack_dims(&out_shape, &contiguous_strides(&out_shape))?;
        let (lhs_shape_packed, lhs_strides_packed) = Self::pack_dims(&lhs.shape, &lhs.strides)?;
        let (rhs_shape_packed, rhs_strides_packed) = Self::pack_dims(&rhs.shape, &rhs.strides)?;
        let params = ArithmeticParams {
            out_len: usize_to_u32(out_len, "output element count")?,
            out_rank: usize_to_u32(out_shape.len(), "output rank")?,
            lhs_rank: usize_to_u32(lhs.shape.len(), "lhs rank")?,
            rhs_rank: usize_to_u32(rhs.shape.len(), "rhs rank")?,
            scalar: 0.0,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
            out_shape: out_shape_packed,
            out_strides: out_strides_packed,
            lhs_shape: lhs_shape_packed,
            lhs_strides: lhs_strides_packed,
            rhs_shape: rhs_shape_packed,
            rhs_strides: rhs_strides_packed,
        };
        let params_buffer = lhs
            .device
            .create_storage_buffer_from_pod(&params, "gpu-arithmetic-params");
        let out_buffer = lhs
            .device
            .create_empty_storage_buffer(out_len, "gpu-arithmetic-out")?;
        lhs.device.dispatch_compute(
            pipeline,
            &[&lhs.buffer, &rhs.buffer, &out_buffer, &params_buffer],
            (div_ceil_u32(params.out_len, ELEMENT_WORKGROUP_SIZE), 1, 1),
            op_name,
            None,
        )?;

        Ok(Self {
            buffer: out_buffer,
            storage_elements: out_len,
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
            device: lhs.device.clone(),
        })
    }

    fn unary_elementwise(
        self,
        pipeline: &wgpu::ComputePipeline,
        op_name: &str,
        scalar: f32,
    ) -> Result<Self> {
        let input = self.contiguous()?;
        let out_len = input.num_elements();
        if out_len == 0 {
            return Self::zeros(&input.shape, &input.device);
        }

        let params = ArithmeticParams {
            out_len: usize_to_u32(out_len, "output element count")?,
            out_rank: usize_to_u32(input.shape.len(), "output rank")?,
            lhs_rank: usize_to_u32(input.shape.len(), "input rank")?,
            rhs_rank: 0,
            scalar,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
            out_shape: Self::pack_dims(&input.shape, &contiguous_strides(&input.shape))?.0,
            out_strides: Self::pack_contiguous_strides(&input.shape)?,
            lhs_shape: Self::pack_dims(&input.shape, &input.strides)?.0,
            lhs_strides: Self::pack_dims(&input.shape, &input.strides)?.1,
            rhs_shape: [0; MAX_DIMS],
            rhs_strides: [0; MAX_DIMS],
        };
        let params_buffer = input
            .device
            .create_storage_buffer_from_pod(&params, "gpu-unary-params");
        let out_buffer = input
            .device
            .create_empty_storage_buffer(out_len, "gpu-unary-out")?;
        input.device.dispatch_compute(
            pipeline,
            &[
                &input.buffer,
                &input.device.inner.dummy_buffer,
                &out_buffer,
                &params_buffer,
            ],
            (div_ceil_u32(params.out_len, ELEMENT_WORKGROUP_SIZE), 1, 1),
            op_name,
            None,
        )?;

        Ok(Self {
            buffer: out_buffer,
            storage_elements: out_len,
            shape: input.shape.clone(),
            strides: contiguous_strides(&input.shape),
            offset: 0,
            device: input.device.clone(),
        })
    }
}

impl Tensor for GpuTensor {
    type Device = GpuDevice;

    fn from_data(data: &[f32], shape: &[usize], device: &Self::Device) -> Result<Self> {
        Self::from_owned(data.to_vec(), shape.to_vec(), device.clone())
    }

    fn zeros(shape: &[usize], device: &Self::Device) -> Result<Self> {
        let len = checked_num_elements(shape)?;
        let zeroes = vec![0.0; len];
        Self::from_owned(zeroes, shape.to_vec(), device.clone())
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn export(&self, buf: &mut [f32]) -> Result<()> {
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

    fn reshape(mut self, shape: &[usize]) -> Result<Self> {
        let expected = checked_num_elements(shape)?;
        if expected != self.num_elements() {
            return Err(invalid_arg(format!(
                "cannot reshape tensor {:?} into {:?}: element count mismatch",
                self.shape, shape
            )));
        }

        if self.is_dense_contiguous_view() {
            self.shape = shape.to_vec();
            self.strides = contiguous_strides(shape);
            return Ok(self);
        }

        let mut contiguous = self.contiguous()?;
        contiguous.shape = shape.to_vec();
        contiguous.strides = contiguous_strides(shape);
        Ok(contiguous)
    }

    fn transpose(mut self, dim0: usize, dim1: usize) -> Result<Self> {
        let rank = self.shape.len();
        validate_axis(dim0, rank, "transpose")?;
        validate_axis(dim1, rank, "transpose")?;
        self.shape.swap(dim0, dim1);
        self.strides.swap(dim0, dim1);
        Ok(self)
    }

    fn contiguous(self) -> Result<Self> {
        if self.has_compact_storage() {
            return Ok(self);
        }

        if self.shape.len() > MAX_DIMS {
            let data = self.materialize_view_on_cpu()?;
            return Self::from_owned(data, self.shape.clone(), self.device.clone());
        }

        let out_len = self.num_elements();
        if out_len == 0 {
            return Self::from_owned(Vec::new(), self.shape.clone(), self.device.clone());
        }

        let out_shape = self.shape.clone();
        let params = LayoutParams {
            out_len: usize_to_u32(out_len, "output element count")?,
            rank: usize_to_u32(self.shape.len(), "tensor rank")?,
            offset: usize_to_u32(self.offset, "tensor offset")?,
            _reserved: 0,
            shape: Self::pack_dims(&self.shape, &self.strides)?.0,
            out_strides: Self::pack_contiguous_strides(&self.shape)?,
            src_strides: Self::pack_dims(&self.shape, &self.strides)?.1,
        };
        let params_buffer = self
            .device
            .create_storage_buffer_from_pod(&params, "gpu-layout-params");
        let out_buffer = self
            .device
            .create_empty_storage_buffer(out_len, "gpu-contiguous-out")?;
        self.device.dispatch_compute(
            &self.device.inner.pipelines.contiguous,
            &[&self.buffer, &out_buffer, &params_buffer],
            (div_ceil_u32(params.out_len, ELEMENT_WORKGROUP_SIZE), 1, 1),
            "contiguous",
            None,
        )?;

        Ok(Self {
            buffer: out_buffer,
            storage_elements: out_len,
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
            device: self.device.clone(),
        })
    }

    fn slice(mut self, axis: usize, start: usize, end: usize) -> Result<Self> {
        let rank = self.shape.len();
        validate_axis(axis, rank, "slice")?;
        let dim = self.shape[axis];
        if start > end || end > dim {
            return Err(invalid_arg(format!(
                "slice axis {} with range {}..{} is out of bounds for shape {:?}",
                axis, start, end, self.shape
            )));
        }

        self.offset += start * self.strides[axis];
        self.shape[axis] = end - start;
        Ok(self)
    }

    fn concat(parts: &[&Self], axis: usize) -> Result<Self> {
        if parts.is_empty() {
            return Err(invalid_arg("concat requires at least one tensor"));
        }
        let device = parts[0].device.clone();
        for part in parts.iter().skip(1) {
            if !Arc::ptr_eq(&device.inner, &part.device.inner) {
                return Err(invalid_arg(
                    "concat requires tensors on the same GPU device",
                ));
            }
        }

        let cpu_parts = parts
            .iter()
            .map(|part| part.to_cpu_tensor())
            .collect::<Result<Vec<_>>>()?;
        let cpu_refs = cpu_parts.iter().collect::<Vec<_>>();
        let out = CpuTensor::concat(&cpu_refs, axis)?;
        Self::from_cpu_tensor(out, &device)
    }

    fn add(self, rhs: &Self) -> Result<Self> {
        let pipeline = self.device.inner.pipelines.add.clone();
        self.arithmetic(rhs, &pipeline, "add")
    }

    fn mul(self, rhs: &Self) -> Result<Self> {
        let pipeline = self.device.inner.pipelines.mul.clone();
        self.arithmetic(rhs, &pipeline, "mul")
    }

    fn scale(self, s: f32) -> Result<Self> {
        let pipeline = self.device.inner.pipelines.scale.clone();
        self.unary_elementwise(&pipeline, "scale", s)
    }

    fn sigmoid(self) -> Result<Self> {
        let pipeline = self.device.inner.pipelines.sigmoid.clone();
        self.unary_elementwise(&pipeline, "sigmoid", 0.0)
    }

    fn matmul(&self, rhs: &Self) -> Result<Self> {
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

    fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self> {
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

    fn rms_norm(self, weight: &Self, eps: f32) -> Result<Self> {
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
            (div_ceil_u32(params.rows, ELEMENT_WORKGROUP_SIZE), 1, 1),
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

    fn gelu(self) -> Result<Self> {
        let pipeline = self.device.inner.pipelines.gelu.clone();
        self.unary_elementwise(&pipeline, "gelu", 0.0)
    }

    fn softmax(self, axis: isize) -> Result<Self> {
        if self.shape.is_empty() {
            return Err(invalid_arg(
                "softmax expects a tensor with at least one dimension",
            ));
        }

        let input = self.contiguous()?;
        let axis = normalize_axis(axis, input.shape.len(), "softmax")?;
        let axis_len = input.shape[axis];
        if axis_len == 0 || input.num_elements() == 0 {
            return Self::from_owned(Vec::new(), input.shape.clone(), input.device.clone());
        }

        let outer = plain_num_elements(&input.shape[..axis]);
        let inner = plain_num_elements(&input.shape[axis + 1..]);
        let params = SoftmaxParams {
            outer: usize_to_u32(outer, "softmax outer dimension")?,
            axis_len: usize_to_u32(axis_len, "softmax axis length")?,
            inner: usize_to_u32(inner, "softmax inner dimension")?,
            _reserved: 0,
        };
        let params_buffer = input
            .device
            .create_storage_buffer_from_pod(&params, "gpu-softmax-params");
        let out_buffer = input
            .device
            .create_empty_storage_buffer(input.num_elements(), "gpu-softmax-out")?;
        input.device.dispatch_compute(
            &input.device.inner.pipelines.softmax,
            &[&input.buffer, &out_buffer, &params_buffer],
            (
                div_ceil_u32(params.inner.max(1), ROW_WORKGROUP_X),
                div_ceil_u32(params.outer.max(1), ROW_WORKGROUP_Y),
                1,
            ),
            "softmax",
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

    fn rope(
        self,
        positions: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self> {
        let input = self.contiguous()?;
        validate_rope_shape(&input.shape, positions.len(), head_dim, num_heads, "rope")?;
        let rope_dims = normalize_rope_dims(head_dim, rope_dims, "rope", false)?;
        if input.num_elements() == 0 {
            return Ok(input);
        }

        let params = RopeParams {
            num_heads: usize_to_u32(num_heads, "rope num_heads")?,
            seq_len: usize_to_u32(positions.len(), "rope sequence length")?,
            head_dim: usize_to_u32(head_dim, "rope head dimension")?,
            rope_dims: usize_to_u32(rope_dims, "rope_dims")?,
            theta,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let params_buffer = input
            .device
            .create_storage_buffer_from_pod(&params, "gpu-rope-params");
        let positions_buffer = input
            .device
            .create_storage_buffer_from_i32(positions, "gpu-rope-positions");
        let out_buffer = input
            .device
            .create_empty_storage_buffer(input.num_elements(), "gpu-rope-out")?;
        input.device.dispatch_compute(
            &input.device.inner.pipelines.rope,
            &[
                &input.buffer,
                &out_buffer,
                &positions_buffer,
                &params_buffer,
            ],
            (
                div_ceil_u32(usize_to_u32(rope_dims / 2, "rope pairs")?, ROPE_WORKGROUP_X),
                div_ceil_u32(
                    usize_to_u32(positions.len(), "rope sequence length")?,
                    ROPE_WORKGROUP_Y,
                ),
                usize_to_u32(num_heads, "rope num_heads")?,
            ),
            "rope",
            Some((&input.buffer, &out_buffer, input.num_elements())),
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

    fn region_rope(
        self,
        global_pos: &[i32],
        region_ids: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self> {
        if global_pos.len() != region_ids.len() {
            return Err(invalid_arg(format!(
                "region_rope expects matching global_pos and region_ids lengths, got {} and {}",
                global_pos.len(),
                region_ids.len()
            )));
        }

        let input = self.contiguous()?;
        validate_rope_shape(
            &input.shape,
            global_pos.len(),
            head_dim,
            num_heads,
            "region_rope",
        )?;
        let rope_dims = normalize_rope_dims(head_dim, rope_dims, "region_rope", true)?;
        if input.num_elements() == 0 {
            return Ok(input);
        }

        let params = RopeParams {
            num_heads: usize_to_u32(num_heads, "region_rope num_heads")?,
            seq_len: usize_to_u32(global_pos.len(), "region_rope sequence length")?,
            head_dim: usize_to_u32(head_dim, "region_rope head dimension")?,
            rope_dims: usize_to_u32(rope_dims, "region_rope_dims")?,
            theta,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let params_buffer = input
            .device
            .create_storage_buffer_from_pod(&params, "gpu-region-rope-params");
        let global_buffer = input
            .device
            .create_storage_buffer_from_i32(global_pos, "gpu-region-rope-global");
        let region_buffer = input
            .device
            .create_storage_buffer_from_i32(region_ids, "gpu-region-rope-region");
        let out_buffer = input
            .device
            .create_empty_storage_buffer(input.num_elements(), "gpu-region-rope-out")?;
        input.device.dispatch_compute(
            &input.device.inner.pipelines.region_rope,
            &[
                &input.buffer,
                &out_buffer,
                &global_buffer,
                &region_buffer,
                &params_buffer,
            ],
            (
                div_ceil_u32(
                    usize_to_u32(rope_dims / 2, "region_rope pairs")?,
                    ROPE_WORKGROUP_X,
                ),
                div_ceil_u32(
                    usize_to_u32(global_pos.len(), "region_rope sequence length")?,
                    ROPE_WORKGROUP_Y,
                ),
                usize_to_u32(num_heads, "region_rope num_heads")?,
            ),
            "region_rope",
            Some((&input.buffer, &out_buffer, input.num_elements())),
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

    fn conv1d_dw(
        self,
        kernel: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let device = self.device.clone();
        self.cpu_fallback_binary(kernel, "conv1d_dw", |lhs, rhs| {
            let bias_cpu = bias.map(GpuTensor::to_cpu_tensor).transpose()?;
            lhs.conv1d_dw(rhs, bias_cpu.as_ref(), stride, padding)
        })
        .map(|mut tensor| {
            tensor.device = device;
            tensor
        })
    }

    fn embedding(table: &Self, indices: &[i32]) -> Result<Self> {
        let table_cpu = table.to_cpu_tensor()?;
        let out = CpuTensor::embedding(&table_cpu, indices)?;
        Self::from_cpu_tensor(out, &table.device)
    }

    fn repeat(self, axis: usize, n: usize) -> Result<Self> {
        self.cpu_fallback_unary("repeat", |tensor| tensor.repeat(axis, n))
    }
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        let contiguous_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-contiguous-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/contiguous.wgsl").into()),
        });
        let arithmetic_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-arithmetic-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/arithmetic.wgsl").into()),
        });
        let rms_norm_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-rms-norm-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rms_norm.wgsl").into()),
        });
        let matmul_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-matmul-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
        });
        let linear_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-linear-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/linear.wgsl").into()),
        });
        let softmax_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-softmax-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/softmax.wgsl").into()),
        });
        let rope_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-rope-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rope.wgsl").into()),
        });

        Self {
            contiguous: create_pipeline(device, &contiguous_module, "main", "tensor-contiguous"),
            add: create_pipeline(device, &arithmetic_module, "add_main", "tensor-add"),
            mul: create_pipeline(device, &arithmetic_module, "mul_main", "tensor-mul"),
            scale: create_pipeline(device, &arithmetic_module, "scale_main", "tensor-scale"),
            sigmoid: create_pipeline(device, &arithmetic_module, "sigmoid_main", "tensor-sigmoid"),
            gelu: create_pipeline(device, &arithmetic_module, "gelu_main", "tensor-gelu"),
            rms_norm: create_pipeline(device, &rms_norm_module, "main", "tensor-rms-norm"),
            matmul: create_pipeline(device, &matmul_module, "main", "tensor-matmul"),
            linear: create_pipeline(device, &linear_module, "main", "tensor-linear"),
            softmax: create_pipeline(device, &softmax_module, "main", "tensor-softmax"),
            rope: create_pipeline(device, &rope_module, "rope_main", "tensor-rope"),
            region_rope: create_pipeline(
                device,
                &rope_module,
                "region_rope_main",
                "tensor-region-rope",
            ),
        }
    }
}

impl GpuAdapterSelector {
    fn is_empty(&self) -> bool {
        self.name_substring.is_none()
            && self.vendor_id.is_none()
            && self.device_id.is_none()
            && self.backend.is_none()
            && self.device_type.is_none()
    }

    fn matches(&self, info: &wgpu::AdapterInfo) -> bool {
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

    fn describe(&self) -> String {
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

fn create_buffer_with_bytes(
    device: &wgpu::Device,
    contents: &[u8],
    label: Option<&str>,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label,
        contents,
        usage,
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn select_adapter(
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

fn select_adapter_explicit(
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

fn invalid_arg(message: impl Into<String>) -> Error {
    Error::message(message.into())
}

fn format_adapter_info(info: &wgpu::AdapterInfo) -> String {
    format!(
        "{} [vendor=0x{:04x}, device=0x{:04x}, backend={}, type={:?}]",
        info.name, info.vendor, info.device, info.backend, info.device_type
    )
}

fn adapter_device_type_priority(device_type: wgpu::DeviceType) -> u8 {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => 5,
        wgpu::DeviceType::IntegratedGpu => 4,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 1,
    }
}

fn adapter_backend_priority(backend: wgpu::Backend) -> u8 {
    match backend {
        wgpu::Backend::Vulkan => 5,
        wgpu::Backend::Metal => 4,
        wgpu::Backend::Dx12 => 3,
        wgpu::Backend::Gl => 2,
        wgpu::Backend::BrowserWebGpu => 1,
        wgpu::Backend::Empty => 0,
    }
}

fn checked_num_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| invalid_arg(format!("tensor shape {:?} is too large", shape)))
    })
}

fn plain_num_elements(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride.saturating_mul(shape[axis]);
    }
    strides
}

fn validate_axis(axis: usize, rank: usize, op_name: &str) -> Result<()> {
    if axis >= rank {
        return Err(invalid_arg(format!(
            "{op_name} axis {} is out of bounds for rank {}",
            axis, rank
        )));
    }
    Ok(())
}

fn normalize_axis(axis: isize, rank: usize, op_name: &str) -> Result<usize> {
    if rank == 0 {
        return Err(invalid_arg(format!(
            "{op_name} requires a tensor with at least one dimension"
        )));
    }

    let rank_isize = isize::try_from(rank).map_err(|_| invalid_arg("rank overflow"))?;
    let normalized = if axis < 0 { rank_isize + axis } else { axis };
    if normalized < 0 || normalized >= rank_isize {
        return Err(invalid_arg(format!(
            "{op_name} axis {} is out of bounds for rank {}",
            axis, rank
        )));
    }

    usize::try_from(normalized).map_err(|_| invalid_arg("axis overflow"))
}

fn validate_rope_shape(
    shape: &[usize],
    positions_len: usize,
    head_dim: usize,
    num_heads: usize,
    op_name: &str,
) -> Result<()> {
    if shape.len() != 3 {
        return Err(invalid_arg(format!(
            "{op_name} expects a rank-3 tensor shaped [num_heads, seq_len, head_dim], got {:?}",
            shape
        )));
    }
    if shape[0] != num_heads {
        return Err(invalid_arg(format!(
            "{op_name} expected num_heads={}, got shape {:?}",
            num_heads, shape
        )));
    }
    if shape[1] != positions_len {
        return Err(invalid_arg(format!(
            "{op_name} expected seq_len={}, got shape {:?}",
            positions_len, shape
        )));
    }
    if shape[2] != head_dim {
        return Err(invalid_arg(format!(
            "{op_name} expected head_dim={}, got shape {:?}",
            head_dim, shape
        )));
    }
    Ok(())
}

fn normalize_rope_dims(
    head_dim: usize,
    rope_dims: usize,
    op_name: &str,
    mixed: bool,
) -> Result<usize> {
    if head_dim == 0 {
        return Err(invalid_arg(format!("{op_name} requires head_dim > 0")));
    }

    let dims = if rope_dims == 0 { head_dim } else { rope_dims };
    if dims > head_dim {
        return Err(invalid_arg(format!(
            "{op_name} rope_dims {} exceeds head_dim {}",
            dims, head_dim
        )));
    }
    if mixed {
        if dims % 4 != 0 {
            return Err(invalid_arg(format!(
                "{op_name} requires rope_dims divisible by 4 for mixed RoPE, got {}",
                dims
            )));
        }
    } else if dims % 2 != 0 {
        return Err(invalid_arg(format!(
            "{op_name} requires an even rope_dims, got {}",
            dims
        )));
    }

    Ok(dims)
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut out = vec![1usize; rank];

    for axis in 0..rank {
        let lhs_dim = lhs
            .len()
            .checked_sub(rank - axis)
            .and_then(|index| lhs.get(index))
            .copied()
            .unwrap_or(1);
        let rhs_dim = rhs
            .len()
            .checked_sub(rank - axis)
            .and_then(|index| rhs.get(index))
            .copied()
            .unwrap_or(1);

        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Err(invalid_arg(format!(
                "cannot broadcast shapes {:?} and {:?}",
                lhs, rhs
            )));
        }
        out[axis] = lhs_dim.max(rhs_dim);
    }

    Ok(out)
}

fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize], usize)) {
    let len = plain_num_elements(shape);
    if len == 0 {
        return;
    }
    if shape.is_empty() {
        f(&[], 0);
        return;
    }

    let mut coords = vec![0usize; shape.len()];
    for flat in 0..len {
        f(&coords, flat);
        for axis in (0..coords.len()).rev() {
            coords[axis] += 1;
            if coords[axis] < shape[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
}

fn bytes_for_elements(elements: usize) -> Result<u64> {
    let elements = elements.max(1);
    u64::try_from(
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid_arg("buffer size overflow"))?,
    )
    .map_err(|_| invalid_arg("buffer size overflow"))
}

fn bytes_for_offset(elements: usize) -> Result<u64> {
    u64::try_from(
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid_arg("buffer offset overflow"))?,
    )
    .map_err(|_| invalid_arg("buffer offset overflow"))
}

fn usize_to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        invalid_arg(format!(
            "{label} {value} exceeds u32::MAX for the GPU backend"
        ))
    })
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

#[cfg(test)]
mod tests {
    use super::{
        GpuAdapterSelector, GpuDevice, GpuTensor, adapter_backend_priority,
        adapter_device_type_priority,
    };
    use crate::tensor::tests;

    fn with_gpu_device(f: impl FnOnce(&GpuDevice)) {
        match GpuDevice::new() {
            Ok(device) => f(&device),
            Err(err) => eprintln!("skipping GPU tensor test: {err}"),
        }
    }

    #[test]
    fn roundtrip_matches_uploaded_values() {
        with_gpu_device(|device| tests::run_roundtrip::<GpuTensor>(device));
    }

    #[test]
    fn layout_ops_preserve_view_semantics() {
        with_gpu_device(|device| {
            tests::run_layout_ops_preserve_view_semantics::<GpuTensor>(device)
        });
    }

    #[test]
    fn broadcast_add_and_mul_match_expected_values() {
        with_gpu_device(|device| {
            tests::run_broadcast_add_and_mul_match_expected_values::<GpuTensor>(device)
        });
    }

    #[test]
    fn matmul_supports_2d_and_batched_3d_inputs() {
        with_gpu_device(|device| {
            tests::run_matmul_supports_2d_and_batched_3d_inputs::<GpuTensor>(device)
        });
    }

    #[test]
    fn linear_applies_weight_rows_and_optional_bias() {
        with_gpu_device(|device| {
            tests::run_linear_applies_weight_rows_and_optional_bias::<GpuTensor>(device)
        });
    }

    #[test]
    fn normalization_and_activation_ops_match_reference_values() {
        with_gpu_device(|device| {
            tests::run_normalization_and_activation_ops_match_reference_values::<GpuTensor>(device)
        });
    }

    #[test]
    fn rope_rotates_each_head_using_global_positions() {
        with_gpu_device(|device| {
            tests::run_rope_rotates_each_head_using_global_positions::<GpuTensor>(device)
        });
    }

    #[test]
    fn region_rope_splits_global_and_region_rotation_halves() {
        with_gpu_device(|device| {
            tests::run_region_rope_splits_global_and_region_rotation_halves::<GpuTensor>(device)
        });
    }

    #[test]
    fn depthwise_conv_applies_per_channel_kernels() {
        with_gpu_device(|device| {
            tests::run_depthwise_conv_applies_per_channel_kernels::<GpuTensor>(device)
        });
    }

    #[test]
    fn embedding_and_repeat_return_expected_rows() {
        with_gpu_device(|device| {
            tests::run_embedding_and_repeat_return_expected_rows::<GpuTensor>(device)
        });
    }

    #[test]
    fn gpu_matches_cpu_reference_for_add_then_softmax() {
        with_gpu_device(|device| {
            tests::run_gpu_against_cpu_reference::<GpuTensor>(device).unwrap()
        });
    }

    #[test]
    fn selector_matches_expected_adapter_fields() {
        let selector = GpuAdapterSelector {
            name_substring: Some("nvidia".to_string()),
            vendor_id: Some(0x10de),
            device_id: Some(0x2484),
            backend: Some(wgpu::Backend::Vulkan),
            device_type: Some(wgpu::DeviceType::DiscreteGpu),
        };
        let matching = wgpu::AdapterInfo {
            name: "NVIDIA GeForce RTX".to_string(),
            vendor: 0x10de,
            device: 0x2484,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };
        let wrong_vendor = wgpu::AdapterInfo {
            vendor: 0x1002,
            ..matching.clone()
        };
        let wrong_name = wgpu::AdapterInfo {
            name: "Intel Arc".to_string(),
            ..matching.clone()
        };

        assert!(selector.matches(&matching));
        assert!(!selector.matches(&wrong_vendor));
        assert!(!selector.matches(&wrong_name));
    }

    #[test]
    fn selector_describe_includes_all_explicit_fields() {
        let selector = GpuAdapterSelector {
            name_substring: Some("RTX".to_string()),
            vendor_id: Some(0x10de),
            device_id: Some(0x2484),
            backend: Some(wgpu::Backend::Vulkan),
            device_type: Some(wgpu::DeviceType::DiscreteGpu),
        };

        let description = selector.describe();
        assert!(description.contains("RTX"));
        assert!(description.contains("0x10de"));
        assert!(description.contains("0x2484"));
        assert!(description.contains("vulkan"));
        assert!(description.contains("DiscreteGpu"));
    }

    #[test]
    fn selector_empty_only_when_no_constraints_are_set() {
        assert!(GpuAdapterSelector::default().is_empty());
        assert!(
            !GpuAdapterSelector {
                name_substring: Some("AMD".to_string()),
                ..GpuAdapterSelector::default()
            }
            .is_empty()
        );
    }

    #[test]
    fn adapter_priority_prefers_discrete_then_vulkan() {
        assert!(
            adapter_device_type_priority(wgpu::DeviceType::DiscreteGpu)
                > adapter_device_type_priority(wgpu::DeviceType::IntegratedGpu)
        );
        assert!(
            adapter_backend_priority(wgpu::Backend::Vulkan)
                > adapter_backend_priority(wgpu::Backend::Dx12)
        );
    }
}
