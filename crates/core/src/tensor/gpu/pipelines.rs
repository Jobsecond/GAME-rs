use wgpu::util::DeviceExt;

use crate::Result;

use super::base::GpuDevice;
use super::util::*;

pub(super) const ELEMENT_WORKGROUP_SIZE: u32 = 64;
pub(super) const MAX_DISPATCH_X: u32 = 65_535;
pub(super) const ROW_WORKGROUP_X: u32 = 8;
pub(super) const ROW_WORKGROUP_Y: u32 = 8;
pub(super) const MATMUL_TILE: u32 = 16;
pub(super) const ROPE_WORKGROUP_X: u32 = 32;
pub(super) const ROPE_WORKGROUP_Y: u32 = 4;

pub(super) struct Pipelines {
    pub(super) contiguous: wgpu::ComputePipeline,
    pub(super) concat: wgpu::ComputePipeline,
    pub(super) add: wgpu::ComputePipeline,
    pub(super) mul: wgpu::ComputePipeline,
    pub(super) scale: wgpu::ComputePipeline,
    pub(super) sigmoid: wgpu::ComputePipeline,
    pub(super) gelu: wgpu::ComputePipeline,
    pub(super) rms_norm: wgpu::ComputePipeline,
    pub(super) matmul: wgpu::ComputePipeline,
    pub(super) linear: wgpu::ComputePipeline,
    pub(super) softmax: wgpu::ComputePipeline,
    pub(super) rope: wgpu::ComputePipeline,
    pub(super) region_rope: wgpu::ComputePipeline,
    pub(super) conv1d_dw: wgpu::ComputePipeline,
    pub(super) embedding: wgpu::ComputePipeline,
    pub(super) repeat: wgpu::ComputePipeline,
}

impl GpuDevice {
    pub(super) fn dispatch_compute(
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

impl Pipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let contiguous_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-contiguous-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/contiguous.wgsl").into()),
        });
        let arithmetic_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-arithmetic-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/arithmetic.wgsl").into()),
        });
        let layout_ops_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-layout-ops-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/layout_ops.wgsl").into()),
        });
        let rms_norm_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-rms-norm-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rms_norm.wgsl").into()),
        });
        let matmul_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-matmul-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/matmul.wgsl").into()),
        });
        let linear_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-linear-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/linear.wgsl").into()),
        });
        let softmax_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-softmax-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/softmax.wgsl").into()),
        });
        let rope_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor-rope-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rope.wgsl").into()),
        });

        Self {
            contiguous: create_pipeline(device, &contiguous_module, "main", "tensor-contiguous"),
            concat: create_pipeline(device, &layout_ops_module, "concat_main", "tensor-concat"),
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
            conv1d_dw: create_pipeline(
                device,
                &layout_ops_module,
                "conv1d_dw_main",
                "tensor-conv1d-dw",
            ),
            embedding: create_pipeline(
                device,
                &layout_ops_module,
                "embedding_main",
                "tensor-embedding",
            ),
            repeat: create_pipeline(device, &layout_ops_module, "repeat_main", "tensor-repeat"),
        }
    }
}

pub(super) fn create_buffer_with_bytes(
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

pub(super) fn create_pipeline(
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

pub(super) fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

pub(super) fn elementwise_workgroups(out_len: u32) -> (u32, u32, u32) {
    let total_x = div_ceil_u32(out_len, ELEMENT_WORKGROUP_SIZE);
    let x = total_x.min(MAX_DISPATCH_X);
    let y = div_ceil_u32(total_x, MAX_DISPATCH_X).max(1);
    (x, y, 1)
}
