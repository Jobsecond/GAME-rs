use crate::config::GameModelConfig;
use crate::{Error, Result, Tensor};

use super::RMS_NORM_EPS;
use super::blocks::ebf_block;
use super::sequence_positions;
use super::weights::SegmenterWeights;
use super::{positive_usize, usize_to_i32};

#[derive(Clone, Debug)]
pub struct SegmenterOutputs<T> {
    pub logits: T,
    pub latent: Option<T>,
}

pub fn run_segmenter_step<T: Tensor>(
    x_seg: &T,
    noise_mod: &[i32],
    t_scalar: Option<f32>,
    language: Option<i32>,
    weights: &SegmenterWeights<T>,
    cfg: &GameModelConfig,
) -> Result<SegmenterOutputs<T>> {
    validate_segmenter_input(x_seg, noise_mod, cfg)?;
    if !cfg.segmenter.use_rope {
        return Err(Error::message(
            "segmenter configuration with use_rope=false is not implemented",
        ));
    }

    let seq_len = x_seg.shape()[0];
    let num_heads = positive_usize("game.segmenter.num_heads", cfg.segmenter.num_heads)?;
    let head_dim = positive_usize("game.segmenter.head_dim", cfg.segmenter.head_dim)?;
    let positions = sequence_positions(seq_len)?;

    let noise_embedding = T::embedding(&weights.noise_embedding, noise_mod)
        .map_err(|err| Error::message(format!("segmenter noise embedding failed: {err}")))?;
    let mut x = x_seg
        .clone()
        .add(&noise_embedding)
        .map_err(|err| Error::message(format!("segmenter noise injection failed: {err}")))?;

    if let Some(time_embedding) = weights.time_embedding.as_ref() {
        let t_scalar = t_scalar
            .ok_or_else(|| Error::message("segmenter time embedding requires a timestep scalar"))?;
        let t = T::from_data(&[t_scalar], &[1], x.device()).map_err(|err| {
            Error::message(format!("segmenter time tensor allocation failed: {err}"))
        })?;
        let t = t
            .linear(
                &time_embedding.layer0.weight,
                Some(&time_embedding.layer0.bias),
            )
            .and_then(Tensor::gelu)
            .and_then(|tensor| {
                tensor.linear(
                    &time_embedding.layer2.weight,
                    Some(&time_embedding.layer2.bias),
                )
            })
            .map_err(|err| Error::message(format!("segmenter time embedding failed: {err}")))?;
        x = x
            .add(&t)
            .map_err(|err| Error::message(format!("segmenter time injection failed: {err}")))?;
    }

    if let Some(language_embedding) = weights.language_embedding.as_ref() {
        let language = language
            .ok_or_else(|| Error::message("segmenter language embedding requires a language id"))?;
        let lang = T::embedding(language_embedding, &[language])
            .map_err(|err| Error::message(format!("segmenter language embedding failed: {err}")))?;
        x = x
            .add(&lang)
            .map_err(|err| Error::message(format!("segmenter language injection failed: {err}")))?;
    }

    x = x
        .linear(&weights.input_proj.weight, Some(&weights.input_proj.bias))
        .map_err(|err| Error::message(format!("segmenter input projection failed: {err}")))?;

    let latent_layer_idx = if cfg.segmenter.return_latent {
        Some(
            positive_usize(
                "game.segmenter.latent_layer_idx",
                cfg.segmenter.latent_layer_idx,
            )? - 1,
        )
    } else {
        None
    };

    let mut latent_tap = None;
    for (index, layer) in weights.layers.iter().enumerate() {
        x = ebf_block(
            &x,
            layer,
            &positions,
            num_heads,
            head_dim,
            cfg.segmenter.theta,
        )?;
        if latent_layer_idx == Some(index) {
            latent_tap = Some(x.clone());
        }
    }

    let latent = match (latent_tap, weights.latent.as_ref()) {
        (Some(latent), Some(latent_weights)) => {
            let latent = if let Some(norm) = latent_weights.norm.as_ref() {
                latent
                    .rms_norm(norm, RMS_NORM_EPS)
                    .map_err(|err| Error::message(format!("segmenter latent norm failed: {err}")))?
            } else {
                latent
            };
            Some(
                latent
                    .linear(&latent_weights.proj.weight, Some(&latent_weights.proj.bias))
                    .map_err(|err| {
                        Error::message(format!("segmenter latent projection failed: {err}"))
                    })?,
            )
        }
        _ => None,
    };

    if let Some(output_norm) = weights.output_norm.as_ref() {
        x = x
            .rms_norm(output_norm, RMS_NORM_EPS)
            .map_err(|err| Error::message(format!("segmenter output norm failed: {err}")))?;
    }

    let logits = x
        .linear(&weights.output_proj.weight, Some(&weights.output_proj.bias))
        .and_then(|tensor| tensor.reshape(&[seq_len]))
        .map_err(|err| Error::message(format!("segmenter logits projection failed: {err}")))?;

    Ok(SegmenterOutputs { logits, latent })
}

fn validate_segmenter_input<T: Tensor>(
    x_seg: &T,
    noise_mod: &[i32],
    cfg: &GameModelConfig,
) -> Result<()> {
    if x_seg.shape().len() != 2 {
        return Err(Error::message(format!(
            "segmenter expects x_seg shaped [num_frames, embedding_dim], got {:?}",
            x_seg.shape()
        )));
    }

    let seq_len = x_seg.shape()[0];
    if noise_mod.len() != seq_len {
        return Err(Error::message(format!(
            "segmenter noise_mod length {} does not match x_seg sequence length {}",
            noise_mod.len(),
            seq_len
        )));
    }

    let embedding_dim = positive_usize("game.model.embedding_dim", cfg.embedding_dim)?;
    if x_seg.shape()[1] != embedding_dim {
        return Err(Error::message(format!(
            "segmenter expected embedding dim {}, got shape {:?}",
            embedding_dim,
            x_seg.shape()
        )));
    }
    if usize_to_i32("segmenter sequence length", seq_len).is_err() {
        return Err(Error::message(format!(
            "segmenter sequence length {} exceeds supported i32 positions",
            seq_len
        )));
    }

    Ok(())
}
