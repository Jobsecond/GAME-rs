pub mod blocks;
pub mod encoder;
pub mod estimator;
pub mod ops;
pub mod segmenter;
pub mod weights;

use crate::{Error, Result};

pub use encoder::{EncoderOutputs, run_encoder};
pub use estimator::{EstimatorOutputs, run_estimator};
pub use ops::build_joint_attn_mask;
pub use segmenter::{SegmenterOutputs, run_segmenter_step};
pub use weights::{GameModelWeights, bind_model_weights};

#[cfg(test)]
mod tests;

pub(crate) const RMS_NORM_EPS: f32 = 1e-6;

pub(crate) fn positive_usize(field: &str, value: i32) -> Result<usize> {
    let value = non_negative_usize(field, value)?;
    if value == 0 {
        return Err(Error::message(format!(
            "configuration field `{field}` must be > 0"
        )));
    }
    Ok(value)
}

pub(crate) fn non_negative_usize(field: &str, value: i32) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        Error::message(format!(
            "configuration field `{field}` must be >= 0, got {value}"
        ))
    })
}

pub(crate) fn usize_to_i32(label: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::message(format!(
            "{label} {value} exceeds i32::MAX and cannot be represented in model positions"
        ))
    })
}

pub(crate) fn sequence_positions(len: usize) -> Result<Vec<i32>> {
    (0..len)
        .map(|index| usize_to_i32("sequence position", index))
        .collect()
}
