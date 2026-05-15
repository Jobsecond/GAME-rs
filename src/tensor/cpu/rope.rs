use rayon::prelude::*;

use crate::Result;
use crate::profiler::op_scope_with;

use super::CpuTensor;
use super::util::*;

impl CpuTensor {
    pub(super) fn rope(
        self,
        positions: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self> {
        let _profile = op_scope_with("cpu.rope", || {
            format!(
                "shape={:?} positions={} head_dim={} num_heads={} rope_dims={} contiguous={}",
                self.shape(),
                positions.len(),
                head_dim,
                num_heads,
                rope_dims,
                self.tensor.is_contiguous()
            )
        });
        let shape = self.shape().to_vec();
        validate_rope_shape(&shape, positions.len(), head_dim, num_heads, "rope")?;
        let rope_dims = normalize_rope_dims(head_dim, rope_dims, "rope", false)?;
        let mut data = self.to_vec()?;
        let seq_len = shape[1];

        let inv_freqs = precompute_inv_freqs(rope_dims, theta);

        let head_block = seq_len * head_dim;
        if should_parallelize(data.len()) && head_block > 0 {
            data.par_chunks_mut(head_block).for_each(|head_slice| {
                for (token, &position) in positions.iter().enumerate() {
                    let base = token * head_dim;
                    apply_rope_chunk(
                        &mut head_slice[base..base + head_dim],
                        0,
                        rope_dims,
                        position as f32,
                        &inv_freqs,
                    );
                }
            });
        } else {
            for head in 0..num_heads {
                for (token, &position) in positions.iter().enumerate() {
                    let base = (head * seq_len + token) * head_dim;
                    apply_rope_chunk(
                        &mut data[base..base + head_dim],
                        0,
                        rope_dims,
                        position as f32,
                        &inv_freqs,
                    );
                }
            }
        }

        Self::from_owned(data, &shape)
    }

    pub(super) fn region_rope(
        self,
        global_pos: &[i32],
        region_ids: &[i32],
        head_dim: usize,
        num_heads: usize,
        rope_dims: usize,
        theta: f32,
    ) -> Result<Self> {
        let _profile = op_scope_with("cpu.region_rope", || {
            format!(
                "shape={:?} tokens={} head_dim={} num_heads={} rope_dims={} contiguous={}",
                self.shape(),
                global_pos.len(),
                head_dim,
                num_heads,
                rope_dims,
                self.tensor.is_contiguous()
            )
        });
        let shape = self.shape().to_vec();
        validate_rope_shape(&shape, global_pos.len(), head_dim, num_heads, "region_rope")?;
        if region_ids.len() != global_pos.len() {
            return Err(invalid_arg(format!(
                "region_rope expected {} region ids, got {}",
                global_pos.len(),
                region_ids.len()
            )));
        }
        let mixed_dims = normalize_rope_dims(head_dim, rope_dims, "region_rope", true)?;
        let half = mixed_dims / 2;
        let seq_len = shape[1];
        let mut data = self.to_vec()?;

        let inv_freqs = precompute_inv_freqs(half, theta);

        let head_block = seq_len * head_dim;
        if should_parallelize(data.len()) && head_block > 0 {
            data.par_chunks_mut(head_block).for_each(|head_slice| {
                for token in 0..seq_len {
                    let base = token * head_dim;
                    let values = &mut head_slice[base..base + head_dim];
                    apply_rope_chunk(values, 0, half, global_pos[token] as f32, &inv_freqs);
                    apply_rope_chunk(values, half, half, region_ids[token] as f32, &inv_freqs);
                }
            });
        } else {
            for head in 0..num_heads {
                for token in 0..seq_len {
                    let base = (head * seq_len + token) * head_dim;
                    let values = &mut data[base..base + head_dim];
                    apply_rope_chunk(values, 0, half, global_pos[token] as f32, &inv_freqs);
                    apply_rope_chunk(values, half, half, region_ids[token] as f32, &inv_freqs);
                }
            }
        }

        Self::from_owned(data, &shape)
    }
}

pub(super) fn precompute_inv_freqs(dims: usize, theta: f32) -> Vec<f32> {
    (0..dims)
        .step_by(2)
        .map(|local_offset| 1.0 / theta.powf(local_offset as f32 / dims as f32))
        .collect()
}

pub(super) fn apply_rope_chunk(
    values: &mut [f32],
    start: usize,
    dims: usize,
    position: f32,
    inv_freqs: &[f32],
) {
    for (idx, local_offset) in (0..dims).step_by(2).enumerate() {
        let angle = position * inv_freqs[idx];
        let (sin, cos) = angle.sin_cos();
        let i0 = start + local_offset;
        let i1 = i0 + 1;
        let x0 = values[i0];
        let x1 = values[i1];
        values[i0] = x0 * cos - x1 * sin;
        values[i1] = x0 * sin + x1 * cos;
    }
}
