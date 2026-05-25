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

const TILE: u32 = 16u;

var<workgroup> tile_x: array<f32, 256>;  // TILE * TILE
var<workgroup> tile_w: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let out_index = gid.x;
    let row = gid.y;
    let tx = lid.x;
    let ty = lid.y;

    let num_tiles = (params.in_dim + TILE - 1u) / TILE;

    var sum = 0.0;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let k_off = t * TILE;

        if (row < params.rows && k_off + tx < params.in_dim) {
            tile_x[ty * TILE + tx] = x[row * params.in_dim + k_off + tx];
        } else {
            tile_x[ty * TILE + tx] = 0.0;
        }

        // weight is [out_dim, in_dim]; we compute x @ weight^T
        if (out_index < params.out_dim && k_off + ty < params.in_dim) {
            tile_w[ty * TILE + tx] = weight[out_index * params.in_dim + k_off + ty];
        } else {
            tile_w[ty * TILE + tx] = 0.0;
        }

        workgroupBarrier();

        for (var i = 0u; i < TILE; i = i + 1u) {
            sum = sum + tile_x[ty * TILE + i] * tile_w[i * TILE + tx];
        }

        workgroupBarrier();
    }

    if (row < params.rows && out_index < params.out_dim) {
        if (params.has_bias != 0u) {
            sum = sum + bias[out_index];
        }
        dst[row * params.out_dim + out_index] = sum;
    }
}
