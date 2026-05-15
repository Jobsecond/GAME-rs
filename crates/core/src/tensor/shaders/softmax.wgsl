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
    var max_value = -3.4028235e38;
    var axis_index: u32 = 0u;
    loop {
        if (axis_index >= params.axis_len) {
            break;
        }
        let value = x[base + axis_index * params.inner];
        max_value = max(max_value, value);
        axis_index = axis_index + 1u;
    }

    var sum = 0.0;
    axis_index = 0u;
    loop {
        if (axis_index >= params.axis_len) {
            break;
        }
        let index = base + axis_index * params.inner;
        let value = exp(x[index] - max_value);
        dst[index] = value;
        sum = sum + value;
        axis_index = axis_index + 1u;
    }

    axis_index = 0u;
    loop {
        if (axis_index >= params.axis_len) {
            break;
        }
        let index = base + axis_index * params.inner;
        dst[index] = dst[index] / sum;
        axis_index = axis_index + 1u;
    }
}
