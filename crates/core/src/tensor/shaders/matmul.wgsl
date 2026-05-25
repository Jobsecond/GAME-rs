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

const TILE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>;  // TILE * TILE
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = gid.x;
    let row = gid.y;
    let batch = gid.z;
    let tx = lid.x;
    let ty = lid.y;

    if (batch >= params.batch) {
        return;
    }

    let lhs_base = batch * params.m * params.k;
    let rhs_base = batch * params.k * params.n;
    let num_tiles = (params.k + TILE - 1u) / TILE;

    var sum = 0.0;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let k_off = t * TILE;

        if (row < params.m && k_off + tx < params.k) {
            tile_a[ty * TILE + tx] = lhs[lhs_base + row * params.k + k_off + tx];
        } else {
            tile_a[ty * TILE + tx] = 0.0;
        }

        if (k_off + ty < params.k && col < params.n) {
            tile_b[ty * TILE + tx] = rhs[rhs_base + (k_off + ty) * params.n + col];
        } else {
            tile_b[ty * TILE + tx] = 0.0;
        }

        workgroupBarrier();

        for (var i = 0u; i < TILE; i = i + 1u) {
            sum = sum + tile_a[ty * TILE + i] * tile_b[i * TILE + tx];
        }

        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        dst[batch * params.m * params.n + row * params.n + col] = sum;
    }
}
