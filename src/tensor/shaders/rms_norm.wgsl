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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }

    var mean_square = 0.0;
    var feature_index: u32 = 0u;
    loop {
        if (feature_index >= params.feature_dim) {
            break;
        }
        let value = x[row * params.feature_dim + feature_index];
        mean_square = mean_square + value * value;
        feature_index = feature_index + 1u;
    }
    let inv_rms = inverseSqrt(mean_square / f32(params.feature_dim) + params.eps);

    feature_index = 0u;
    loop {
        if (feature_index >= params.feature_dim) {
            break;
        }
        let index = row * params.feature_dim + feature_index;
        dst[index] = x[index] * inv_rms * weight[feature_index];
        feature_index = feature_index + 1u;
    }
}
