struct SoftmaxParams {
    outer: u32,
    axis_len: u32,
    inner: u32,
    _reserved: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> params: SoftmaxParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let inner_index = gid.x;
    let outer_index = gid.y;
    if (inner_index >= params.inner || outer_index >= params.outer) {
        return;
    }

    let base = outer_index * params.axis_len * params.inner + inner_index;
    let stride = params.inner;

    // Online max + sum in a single pass (no intermediate writes).
    var max_val = x[base];
    var sum_val = 1.0;
    for (var i = 1u; i < params.axis_len; i = i + 1u) {
        let val = x[base + i * stride];
        let new_max = max(max_val, val);
        sum_val = sum_val * exp(max_val - new_max) + exp(val - new_max);
        max_val = new_max;
    }

    let inv_sum = 1.0 / sum_val;
    for (var i = 0u; i < params.axis_len; i = i + 1u) {
        let idx = base + i * stride;
        dst[idx] = exp(x[idx] - max_val) * inv_sum;
    }
}
