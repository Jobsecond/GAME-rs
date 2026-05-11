const MAX_DIMS: u32 = 8u;

struct ArithmeticParams {
    out_len: u32,
    out_rank: u32,
    lhs_rank: u32,
    rhs_rank: u32,
    scalar: f32,
    _reserved0: u32,
    _reserved1: u32,
    _reserved2: u32,
    out_shape: array<u32, 8>,
    out_strides: array<u32, 8>,
    lhs_shape: array<u32, 8>,
    lhs_strides: array<u32, 8>,
    rhs_shape: array<u32, 8>,
    rhs_strides: array<u32, 8>,
}

@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> params: ArithmeticParams;

fn coords_for(flat: u32) -> array<u32, 8> {
    var coords: array<u32, 8>;
    var remainder = flat;
    var axis: u32 = 0u;
    loop {
        if (axis >= params.out_rank) {
            break;
        }
        let stride = params.out_strides[axis];
        if (stride == 0u) {
            coords[axis] = 0u;
        } else {
            coords[axis] = remainder / stride;
            remainder = remainder % stride;
        }
        axis = axis + 1u;
    }
    return coords;
}

fn broadcast_offset(
    coords: array<u32, 8>,
    rank: u32,
    shape: array<u32, 8>,
    strides: array<u32, 8>,
) -> u32 {
    if (rank == 0u) {
        return 0u;
    }

    let rank_diff = params.out_rank - rank;
    var offset = 0u;
    var axis: u32 = 0u;
    loop {
        if (axis >= params.out_rank) {
            break;
        }
        if (axis >= rank_diff) {
            let src_axis = axis - rank_diff;
            if (shape[src_axis] != 1u) {
                offset = offset + coords[axis] * strides[src_axis];
            }
        }
        axis = axis + 1u;
    }
    return offset;
}

fn erf_approx(x: f32) -> f32 {
    let sign = select(1.0, -1.0, x < 0.0);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449672) * t
        + 0.2548296) * t);
    let y = 1.0 - poly * exp(-ax * ax);
    return sign * y;
}

@compute @workgroup_size(64)
fn add_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.out_len) {
        return;
    }
    let coords = coords_for(flat);
    let lhs_index = broadcast_offset(coords, params.lhs_rank, params.lhs_shape, params.lhs_strides);
    let rhs_index = broadcast_offset(coords, params.rhs_rank, params.rhs_shape, params.rhs_strides);
    dst[flat] = lhs[lhs_index] + rhs[rhs_index];
}

@compute @workgroup_size(64)
fn mul_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.out_len) {
        return;
    }
    let coords = coords_for(flat);
    let lhs_index = broadcast_offset(coords, params.lhs_rank, params.lhs_shape, params.lhs_strides);
    let rhs_index = broadcast_offset(coords, params.rhs_rank, params.rhs_shape, params.rhs_strides);
    dst[flat] = lhs[lhs_index] * rhs[rhs_index];
}

@compute @workgroup_size(64)
fn scale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.out_len) {
        return;
    }
    let _keep_layout = rhs[0];
    dst[flat] = lhs[flat] * params.scalar;
}

@compute @workgroup_size(64)
fn sigmoid_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.out_len) {
        return;
    }
    let _keep_layout = rhs[0];
    let value = lhs[flat];
    dst[flat] = 1.0 / (1.0 + exp(-value));
}

@compute @workgroup_size(64)
fn gelu_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.out_len) {
        return;
    }
    let _keep_layout = rhs[0];
    let value = lhs[flat];
    dst[flat] = 0.5 * value * (1.0 + erf_approx(value / sqrt(2.0)));
}
