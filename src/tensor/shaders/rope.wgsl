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

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> positions: array<i32>;
@group(0) @binding(3) var<storage, read> params: RopeParams;

@group(0) @binding(0) var<storage, read> region_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> region_dst: array<f32>;
@group(0) @binding(2) var<storage, read> global_pos: array<i32>;
@group(0) @binding(3) var<storage, read> region_ids: array<i32>;
@group(0) @binding(4) var<storage, read> region_params: RopeParams;

@compute @workgroup_size(32, 4, 1)
fn rope_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair = gid.x;
    let token = gid.y;
    let head = gid.z;
    let dim = pair * 2u;
    if (head >= params.num_heads || token >= params.seq_len || dim + 1u >= params.rope_dims) {
        return;
    }

    let angle = f32(positions[token]) / pow(params.theta, f32(dim) / f32(params.rope_dims));
    let sin_angle = sin(angle);
    let cos_angle = cos(angle);
    let base = (head * params.seq_len + token) * params.head_dim + dim;
    let x0 = src[base];
    let x1 = src[base + 1u];
    dst[base] = x0 * cos_angle - x1 * sin_angle;
    dst[base + 1u] = x0 * sin_angle + x1 * cos_angle;
}

@compute @workgroup_size(32, 4, 1)
fn region_rope_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair = gid.x;
    let token = gid.y;
    let head = gid.z;
    let dim = pair * 2u;
    if (
        head >= region_params.num_heads ||
        token >= region_params.seq_len ||
        dim + 1u >= region_params.rope_dims
    ) {
        return;
    }

    let half = region_params.rope_dims / 2u;
    let local_dim = select(dim - half, dim, dim < half);
    let local_dims = half;
    let position = select(
        f32(region_ids[token]),
        f32(global_pos[token]),
        dim < half,
    );
    let angle = position / pow(region_params.theta, f32(local_dim) / f32(local_dims));
    let sin_angle = sin(angle);
    let cos_angle = cos(angle);
    let base = (head * region_params.seq_len + token) * region_params.head_dim + dim;
    let x0 = region_src[base];
    let x1 = region_src[base + 1u];
    region_dst[base] = x0 * cos_angle - x1 * sin_angle;
    region_dst[base + 1u] = x0 * sin_angle + x1 * cos_angle;
}
