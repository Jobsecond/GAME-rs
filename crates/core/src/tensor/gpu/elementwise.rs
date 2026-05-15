use crate::Result;

use super::base::GpuTensor;
use super::params::*;
use super::pipelines::*;
use super::util::*;

impl GpuTensor {
    pub(super) fn arithmetic(
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
            elementwise_workgroups(params.out_len),
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

    pub(super) fn unary_elementwise(
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
            elementwise_workgroups(params.out_len),
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

    pub(super) fn add(self, rhs: &Self) -> Result<Self> {
        let pipeline = self.device.inner.pipelines.add.clone();
        self.arithmetic(rhs, &pipeline, "add")
    }

    pub(super) fn mul(self, rhs: &Self) -> Result<Self> {
        let pipeline = self.device.inner.pipelines.mul.clone();
        self.arithmetic(rhs, &pipeline, "mul")
    }

    pub(super) fn scale(self, s: f32) -> Result<Self> {
        let pipeline = self.device.inner.pipelines.scale.clone();
        self.unary_elementwise(&pipeline, "scale", s)
    }

    pub(super) fn sigmoid(self) -> Result<Self> {
        let pipeline = self.device.inner.pipelines.sigmoid.clone();
        self.unary_elementwise(&pipeline, "sigmoid", 0.0)
    }

    pub(super) fn gelu(self) -> Result<Self> {
        let pipeline = self.device.inner.pipelines.gelu.clone();
        self.unary_elementwise(&pipeline, "gelu", 0.0)
    }

    pub(super) fn softmax(self, axis: isize) -> Result<Self> {
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
}
