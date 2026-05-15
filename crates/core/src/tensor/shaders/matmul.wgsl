struct MatmulParams {
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
}

@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> params: MatmulParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    let batch = gid.z;
    if (batch >= params.batch || row >= params.m || col >= params.n) {
        return;
    }

    let lhs_batch_offset = batch * params.m * params.k;
    let rhs_batch_offset = batch * params.k * params.n;

    var sum = 0.0;
    var kk: u32 = 0u;
    loop {
        if (kk >= params.k) {
            break;
        }
        let lhs_index = lhs_batch_offset + row * params.k + kk;
        let rhs_index = rhs_batch_offset + kk * params.n + col;
        sum = sum + lhs[lhs_index] * rhs[rhs_index];
        kk = kk + 1u;
    }

    let out_index = batch * params.m * params.n + row * params.n + col;
    dst[out_index] = sum;
}
