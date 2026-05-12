const MAX_DIMS: u32 = 8u;

struct LayoutParams {
    out_len: u32,
    rank: u32,
    offset: u32,
    _reserved: u32,
    shape: array<u32, 8>,
    out_strides: array<u32, 8>,
    src_strides: array<u32, 8>,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> params: LayoutParams;

fn flat_index(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * 65535u * 64u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = flat_index(gid);
    if (flat >= params.out_len) {
        return;
    }

    var coords: array<u32, 8>;
    var remainder = flat;
    var axis: u32 = 0u;
    loop {
        if (axis >= params.rank) {
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

    var src_index = params.offset;
    axis = 0u;
    loop {
        if (axis >= params.rank) {
            break;
        }
        src_index = src_index + coords[axis] * params.src_strides[axis];
        axis = axis + 1u;
    }

    dst[flat] = src[src_index];
}
