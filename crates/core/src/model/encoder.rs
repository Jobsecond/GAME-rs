use crate::config::GameModelConfig;
use crate::{Error, Result, Tensor};

use super::blocks::ebf_block;
use super::ops::split_last_dim_two;
use super::sequence_positions;
use super::weights::{EncoderWeights, LinearWeights};
use super::{positive_usize, usize_to_i32};

#[derive(Clone, Debug)]
pub struct EncoderOutputs<T> {
    pub x_seg: T,
    pub x_est: T,
}

pub fn run_encoder<T: Tensor>(
    mel: &T,
    spectrogram_projection: &LinearWeights<T>,
    weights: &EncoderWeights<T>,
    cfg: &GameModelConfig,
) -> Result<EncoderOutputs<T>> {
    validate_encoder_input(mel, cfg)?;
    if !cfg.encoder.use_rope {
        return Err(Error::message(
            "encoder configuration with use_rope=false is not implemented",
        ));
    }

    let num_heads = positive_usize("game.encoder.num_heads", cfg.encoder.num_heads)?;
    let head_dim = positive_usize("game.encoder.head_dim", cfg.encoder.head_dim)?;
    let seq_len = mel.shape()[0];
    let positions = sequence_positions(seq_len)?;

    let mut x = mel
        .linear(
            &spectrogram_projection.weight,
            Some(&spectrogram_projection.bias),
        )
        .map_err(|err| Error::message(format!("encoder spectrogram projection failed: {err}")))?;
    x = x
        .linear(&weights.input_proj.weight, Some(&weights.input_proj.bias))
        .map_err(|err| Error::message(format!("encoder input projection failed: {err}")))?;

    for layer in &weights.layers {
        x = ebf_block(
            &x,
            layer,
            &positions,
            num_heads,
            head_dim,
            cfg.encoder.theta,
        )?;
    }

    if let Some(output_norm) = weights.output_norm.as_ref() {
        x = x
            .rms_norm(output_norm, super::RMS_NORM_EPS)
            .map_err(|err| Error::message(format!("encoder output norm failed: {err}")))?;
    }

    let full = x
        .linear(&weights.output_proj.weight, Some(&weights.output_proj.bias))
        .map_err(|err| Error::message(format!("encoder output projection failed: {err}")))?;
    let (x_seg, x_est) = split_last_dim_two(&full)?;

    let expected_dim = positive_usize("game.model.embedding_dim", cfg.embedding_dim)?;
    for (label, tensor) in [("x_seg", &x_seg), ("x_est", &x_est)] {
        if tensor.shape().len() != 2
            || tensor.shape()[0] != seq_len
            || tensor.shape()[1] != expected_dim
        {
            return Err(Error::message(format!(
                "encoder {label} has shape {:?}, expected [{seq_len}, {expected_dim}]",
                tensor.shape()
            )));
        }
    }

    Ok(EncoderOutputs { x_seg, x_est })
}

fn validate_encoder_input<T: Tensor>(mel: &T, cfg: &GameModelConfig) -> Result<()> {
    if mel.shape().len() != 2 {
        return Err(Error::message(format!(
            "encoder expects mel input shaped [num_frames, n_mels], got {:?}",
            mel.shape()
        )));
    }

    let expected_mels = positive_usize("game.model.in_dim", cfg.in_dim)?;
    if mel.shape()[1] != expected_mels {
        return Err(Error::message(format!(
            "encoder expected {} mel bins, got shape {:?}",
            expected_mels,
            mel.shape()
        )));
    }
    if usize_to_i32("encoder sequence length", mel.shape()[0]).is_err() {
        return Err(Error::message(format!(
            "encoder sequence length {} exceeds supported i32 positions",
            mel.shape()[0]
        )));
    }

    Ok(())
}
