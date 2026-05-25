struct RmsNormParams {
    rows: u32,
    feature_dim: u32,
    eps: f32,
    _reserved: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> params: RmsNormParams;

const WG: u32 = 256u;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let row = wid.x;
    if row >= params.rows {
        return;
    }

    let base = row * params.feature_dim;

    // Phase 1: Each thread accumulates squared values for its elements.
    var local_sq = 0.0;
    for (var i = tid; i < params.feature_dim; i += WG) {
        let val = x[base + i];
        local_sq += val * val;
    }
    shared_val[tid] = local_sq;
    workgroupBarrier();

    // Tree reduction for sum of squares.
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

    let inv_rms = inverseSqrt(shared_val[0] / f32(params.feature_dim) + params.eps);

    // Phase 2: Normalize and apply weight.
    for (var i = tid; i < params.feature_dim; i += WG) {
        dst[base + i] = x[base + i] * inv_rms * weight[i];
    }
}
