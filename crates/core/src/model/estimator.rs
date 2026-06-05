use crate::config::GameModelConfig;
use crate::profiler::{scope, scope_with};
use crate::{Error, Result, Tensor};

use super::RMS_NORM_EPS;
use super::blocks::jebf_block;
use super::ops::build_joint_attn_mask;
use super::weights::EstimatorWeights;
use super::{positive_usize, usize_to_i32};

#[derive(Clone, Debug)]
pub struct EstimatorOutputs<T> {
    pub pool_logits: T,
}

pub fn run_estimator<T: Tensor>(
    x_est: &T,
    regions: &[i32],
    weights: &EstimatorWeights<T>,
    cfg: &GameModelConfig,
) -> Result<EstimatorOutputs<T>> {
    let _scope = scope_with("run_estimator", || {
        format!(
            "x_est={:?} regions={} layers={}",
            x_est.shape(),
            regions.len(),
            weights.layers.len()
        )
    });
    validate_estimator_input(x_est, regions, cfg)?;
    if !cfg.estimator.use_rope {
        return Err(Error::message(
            "estimator configuration with use_rope=false is not implemented",
        ));
    }
    if cfg.estimator.use_pool_offset {
        return Err(Error::message(
            "estimator configuration with use_pool_offset=true is not implemented",
        ));
    }

    let out_dim = positive_usize("game.model.estimator_out_dim", cfg.estimator_out_dim)?;
    let n_regions = max_region(regions)?;
    if n_regions == 0 {
        return Ok(EstimatorOutputs {
            pool_logits: T::zeros(&[0, out_dim], x_est.device()).map_err(|err| {
                Error::message(format!("estimator empty output allocation failed: {err}"))
            })?,
        });
    }

    let seq_len = x_est.shape()[0];
    let num_heads = positive_usize("game.estimator.num_heads", cfg.estimator.num_heads)?;
    let head_dim = positive_usize("game.estimator.head_dim", cfg.estimator.head_dim)?;
    let region_cycle_len = positive_usize("game.model.region_cycle_len", cfg.region_cycle_len)?;

    let regions_mod = regions_mod_cycle(regions, region_cycle_len)?;
    let region_embedding = T::embedding(&weights.region_embedding, &regions_mod)
        .map_err(|err| Error::message(format!("estimator region embedding failed: {err}")))?;
    let mut x = x_est
        .clone()
        .add(&region_embedding)
        .map_err(|err| Error::message(format!("estimator region injection failed: {err}")))?;
    x = x
        .linear(&weights.input_proj.weight, Some(&weights.input_proj.bias))
        .map_err(|err| Error::message(format!("estimator input projection failed: {err}")))?;

    let mut pool = weights
        .pool_token_gen
        .clone()
        .repeat(0, n_regions)
        .map_err(|err| Error::message(format!("estimator pool token repeat failed: {err}")))?;

    let global_positions = build_global_positions(n_regions, seq_len)?;
    let region_ids = build_region_rope_ids(regions, n_regions)?;
    let total_len = n_regions + seq_len;
    let mask_data = build_joint_attn_mask(regions, n_regions)
        .map_err(|err| Error::message(format!("estimator mask failed: {err}")))?;
    let mask = T::from_data(&mask_data, &[total_len, total_len], x_est.device())
        .map_err(|err| Error::message(format!("estimator mask allocation failed: {err}")))?;

    for layer in &weights.layers {
        let _layer_scope = scope(
            "run_estimator.layer",
            format!("pool={:?} x={:?}", pool.shape(), x.shape()),
        );
        let out = jebf_block(
            &pool,
            &x,
            layer,
            &global_positions,
            &region_ids,
            &mask,
            num_heads,
            head_dim,
            cfg.estimator.theta,
        )?;
        pool = out.pool;
        x = out.x;
    }

    if let Some(output_norm) = weights.output_norm_pool.as_ref() {
        let _norm_scope = scope(
            "run_estimator.output_norm_pool",
            format!("pool={:?}", pool.shape()),
        );
        pool = pool
            .rms_norm(output_norm, RMS_NORM_EPS)
            .map_err(|err| Error::message(format!("estimator pool output norm failed: {err}")))?;
    }

    let _proj_scope = scope(
        "run_estimator.output_proj_pool",
        format!("pool={:?}", pool.shape()),
    );
    let pool_logits = pool
        .linear(
            &weights.output_proj_pool.weight,
            Some(&weights.output_proj_pool.bias),
        )
        .map_err(|err| Error::message(format!("estimator pool output projection failed: {err}")))?;
    if pool_logits.shape() != [n_regions, out_dim] {
        return Err(Error::message(format!(
            "estimator pool_logits has shape {:?}, expected [{n_regions}, {out_dim}]",
            pool_logits.shape()
        )));
    }

    Ok(EstimatorOutputs { pool_logits })
}

fn validate_estimator_input<T: Tensor>(
    x_est: &T,
    regions: &[i32],
    cfg: &GameModelConfig,
) -> Result<()> {
    if x_est.shape().len() != 2 {
        return Err(Error::message(format!(
            "estimator expects x_est shaped [num_frames, embedding_dim], got {:?}",
            x_est.shape()
        )));
    }

    let seq_len = x_est.shape()[0];
    if regions.len() != seq_len {
        return Err(Error::message(format!(
            "estimator regions length {} does not match x_est sequence length {}",
            regions.len(),
            seq_len
        )));
    }

    let embedding_dim = positive_usize("game.model.embedding_dim", cfg.embedding_dim)?;
    if x_est.shape()[1] != embedding_dim {
        return Err(Error::message(format!(
            "estimator expected embedding dim {}, got shape {:?}",
            embedding_dim,
            x_est.shape()
        )));
    }

    Ok(())
}

fn max_region(regions: &[i32]) -> Result<usize> {
    let mut max_region = 0usize;
    for &region in regions {
        let region = usize::try_from(region).map_err(|_| {
            Error::message(format!(
                "estimator regions must be >= 0, got invalid region id {region}"
            ))
        })?;
        max_region = max_region.max(region);
    }
    Ok(max_region)
}

fn regions_mod_cycle(regions: &[i32], cycle_len: usize) -> Result<Vec<i32>> {
    regions
        .iter()
        .copied()
        .map(|region| {
            let region_usize = usize::try_from(region).map_err(|_| {
                Error::message(format!(
                    "estimator regions must be >= 0, got invalid region id {region}"
                ))
            })?;
            usize_to_i32("region cycle remainder", region_usize % cycle_len)
        })
        .collect()
}

fn build_global_positions(n_regions: usize, seq_len: usize) -> Result<Vec<i32>> {
    let mut positions = Vec::with_capacity(n_regions + seq_len);
    for index in 0..n_regions {
        positions.push(usize_to_i32("pool global position", index)?);
    }
    for index in 0..seq_len {
        positions.push(usize_to_i32("frame global position", index)?);
    }
    Ok(positions)
}

fn build_region_rope_ids(regions: &[i32], n_regions: usize) -> Result<Vec<i32>> {
    let mut ids = vec![0; n_regions + regions.len()];
    let mut current_region = 0i32;
    let mut current_local = 0usize;
    for (index, &region) in regions.iter().enumerate() {
        if region < 0 {
            return Err(Error::message(format!(
                "estimator regions must be >= 0, got invalid region id {region}"
            )));
        }
        if region != current_region {
            current_region = region;
            current_local = 0;
        }
        ids[n_regions + index] = if region > 0 {
            usize_to_i32("region-local position", current_local + 1)?
        } else {
            0
        };
        current_local += 1;
    }
    Ok(ids)
}
