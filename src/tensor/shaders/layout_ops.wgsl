struct ConcatParams {
    part_len: u32,
    inner: u32,
    part_axis_len: u32,
    out_axis_len: u32,
    axis_offset: u32,
    _reserved0: u32,
    _reserved1: u32,
    _reserved2: u32,
}

struct Conv1dDwParams {
    time: u32,
    channels: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    out_time: u32,
    has_bias: u32,
    _reserved: u32,
}

struct EmbeddingParams {
    out_len: u32,
    dim: u32,
    _reserved0: u32,
    _reserved1: u32,
}

struct RepeatParams {
    out_len: u32,
    outer: u32,
    axis_len: u32,
    inner: u32,
    repeat_n: u32,
    _reserved0: u32,
    _reserved1: u32,
    _reserved2: u32,
}

@group(0) @binding(0) var<storage, read> src0: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst1: array<f32>;
@group(0) @binding(2) var<storage, read> params_concat: ConcatParams;

fn flat_index(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * 65535u * 64u;
}

@compute @workgroup_size(64)
fn concat_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = flat_index(gid);
    if (flat >= params_concat.part_len) {
        return;
    }

    let inner = max(params_concat.inner, 1u);
    let part_axis_block = params_concat.part_axis_len * inner;
    let out_axis_block = params_concat.out_axis_len * inner;
    let outer_index = flat / part_axis_block;
    let part_remainder = flat % part_axis_block;
    let dst_index =
        outer_index * out_axis_block + params_concat.axis_offset * inner + part_remainder;
    dst1[dst_index] = src0[flat];
}

@group(0) @binding(0) var<storage, read> input_conv: array<f32>;
@group(0) @binding(1) var<storage, read> kernel_conv: array<f32>;
@group(0) @binding(2) var<storage, read> bias_conv: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_conv: array<f32>;
@group(0) @binding(4) var<storage, read> params_conv: Conv1dDwParams;

@compute @workgroup_size(8, 8, 1)
fn conv1d_dw_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let channel = gid.x;
    let out_t = gid.y;
    if (channel >= params_conv.channels || out_t >= params_conv.out_time) {
        return;
    }

    var sum = 0.0;
    if (params_conv.has_bias != 0u) {
        sum = bias_conv[channel];
    }

    var kernel_index: u32 = 0u;
    loop {
        if (kernel_index >= params_conv.kernel_size) {
            break;
        }

        let input_index = out_t * params_conv.stride + kernel_index;
        if (input_index >= params_conv.padding) {
            let input_t = input_index - params_conv.padding;
            if (input_t < params_conv.time) {
                let input_offset = input_t * params_conv.channels + channel;
                let kernel_offset = channel * params_conv.kernel_size + kernel_index;
                sum = sum + input_conv[input_offset] * kernel_conv[kernel_offset];
            }
        }

        kernel_index = kernel_index + 1u;
    }

    output_conv[out_t * params_conv.channels + channel] = sum;
}

@group(0) @binding(0) var<storage, read> table_emb: array<f32>;
@group(0) @binding(1) var<storage, read> indices_emb: array<i32>;
@group(0) @binding(2) var<storage, read_write> output_emb: array<f32>;
@group(0) @binding(3) var<storage, read> params_emb: EmbeddingParams;

@compute @workgroup_size(64)
fn embedding_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = flat_index(gid);
    if (flat >= params_emb.out_len) {
        return;
    }

    let row_index = flat / params_emb.dim;
    let dim_index = flat % params_emb.dim;
    let table_row = u32(indices_emb[row_index]);
    output_emb[flat] = table_emb[table_row * params_emb.dim + dim_index];
}

@group(0) @binding(0) var<storage, read> input_repeat: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_repeat: array<f32>;
@group(0) @binding(2) var<storage, read> params_repeat: RepeatParams;

@compute @workgroup_size(64)
fn repeat_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = flat_index(gid);
    if (flat >= params_repeat.out_len) {
        return;
    }

    let inner = max(params_repeat.inner, 1u);
    let out_axis_len = params_repeat.axis_len * params_repeat.repeat_n;
    let out_axis_block = out_axis_len * inner;
    let axis_block = params_repeat.axis_len * inner;

    let outer_index = flat / out_axis_block;
    let out_remainder = flat % out_axis_block;
    let axis_offset = out_remainder % axis_block;
    let src_index = outer_index * axis_block + axis_offset;
    output_repeat[flat] = input_repeat[src_index];
}
