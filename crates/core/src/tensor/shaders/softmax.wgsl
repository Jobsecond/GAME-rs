struct SoftmaxParams {
    outer: u32,
    axis_len: u32,
    inner: u32,
    _reserved: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> params: SoftmaxParams;

const WG: u32 = 256u;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let row_id = wid.x;
    let total_rows = params.outer * params.inner;
    if row_id >= total_rows {
        return;
    }

    let outer_index = row_id / params.inner;
    let inner_index = row_id % params.inner;
    let base = outer_index * params.axis_len * params.inner + inner_index;
    let stride = params.inner;

    // Phase 1: Each thread finds local max over its elements.
    var local_max = -3.402823e+38;
    for (var i = tid; i < params.axis_len; i += WG) {
        local_max = max(local_max, x[base + i * stride]);
    }
    shared_val[tid] = local_max;
    workgroupBarrier();

    // Tree reduction for global max.
    if tid < 128u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 128u]); }
    workgroupBarrier();
    if tid < 64u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 64u]); }
    workgroupBarrier();
    if tid < 32u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 32u]); }
    workgroupBarrier();
    if tid < 16u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 16u]); }
    workgroupBarrier();
    if tid < 8u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 8u]); }
    workgroupBarrier();
    if tid < 4u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 4u]); }
    workgroupBarrier();
    if tid < 2u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 2u]); }
    workgroupBarrier();
    if tid < 1u { shared_val[tid] = max(shared_val[tid], shared_val[tid + 1u]); }
    workgroupBarrier();

    let global_max = shared_val[0];

    // Phase 2: Each thread computes partial exp sum.
    var local_sum = 0.0;
    for (var i = tid; i < params.axis_len; i += WG) {
        local_sum += exp(x[base + i * stride] - global_max);
    }
    shared_val[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for global sum.
    if tid < 128u { shared_val[tid] += shared_val[tid + 128u]; }
    workgroupBarrier();
    if tid < 64u { shared_val[tid] += shared_val[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { shared_val[tid] += shared_val[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { shared_val[tid] += shared_val[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { shared_val[tid] += shared_val[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { shared_val[tid] += shared_val[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { shared_val[tid] += shared_val[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { shared_val[tid] += shared_val[tid + 1u]; }
    workgroupBarrier();

    let inv_sum = 1.0 / shared_val[0];

    // Phase 3: Normalize and write output.
    for (var i = tid; i < params.axis_len; i += WG) {
        let idx = base + i * stride;
        dst[idx] = exp(x[idx] - global_max) * inv_sum;
    }
}
