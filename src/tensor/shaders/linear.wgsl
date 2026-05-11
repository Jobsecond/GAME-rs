struct LinearParams {
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    has_bias: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<storage, read> params: LinearParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_index = gid.x;
    let row = gid.y;
    if (row >= params.rows || out_index >= params.out_dim) {
        return;
    }

    var sum = 0.0;
    if (params.has_bias != 0u) {
        sum = bias[out_index];
    }

    var in_index: u32 = 0u;
    loop {
        if (in_index >= params.in_dim) {
            break;
        }
        let x_index = row * params.in_dim + in_index;
        let weight_index = out_index * params.in_dim + in_index;
        sum = sum + x[x_index] * weight[weight_index];
        in_index = in_index + 1u;
    }

    dst[row * params.out_dim + out_index] = sum;
}
