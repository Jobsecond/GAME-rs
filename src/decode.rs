use std::cmp::{max, min};

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct GaussianBlurredResult {
    pub values: Vec<f32>,
    pub presence: Vec<u8>,
}

pub fn boundaries_to_regions(boundaries: &[u8], mask: Option<&[u8]>) -> Result<Vec<i32>> {
    validate_optional_len("mask", mask, boundaries.len())?;

    let mut regions = vec![0; boundaries.len()];
    let mut running = 1_i32;
    for index in 0..boundaries.len() {
        if boundaries[index] != 0 {
            running = running
                .checked_add(1)
                .ok_or_else(|| Error::message("region id overflowed i32"))?;
        }
        let valid = mask.is_none_or(|mask| mask[index] != 0);
        regions[index] = if valid { running } else { 0 };
    }
    Ok(regions)
}

pub fn decode_soft_boundaries(
    probs: &[f32],
    barriers: Option<&[u8]>,
    mask: Option<&[u8]>,
    threshold: f32,
    radius: i32,
) -> Result<Vec<u8>> {
    validate_optional_len("barriers", barriers, probs.len())?;
    validate_optional_len("mask", mask, probs.len())?;
    if !threshold.is_finite() {
        return Err(Error::message(format!(
            "boundary threshold must be finite, got {threshold}"
        )));
    }
    let radius = usize::try_from(radius).map_err(|_| {
        Error::message(format!(
            "boundary radius must be non-negative, got {radius}"
        ))
    })?;

    if probs.is_empty() {
        return Ok(Vec::new());
    }

    let mut values = Vec::with_capacity(probs.len());
    for index in 0..probs.len() {
        let value = if mask.is_some_and(|mask| mask[index] == 0)
            || barriers.is_some_and(|barriers| barriers[index] != 0)
        {
            f32::INFINITY
        } else {
            probs[index]
        };
        values.push(value);
    }

    let mut out = vec![0; probs.len()];
    for index in 0..probs.len() {
        let lo = index.saturating_sub(radius);
        let hi = min(probs.len() - 1, index.saturating_add(radius));
        let mut best = values[lo];
        let mut arg = lo;
        for k in (lo + 1)..=hi {
            if values[k] > best {
                best = values[k];
                arg = k;
            }
        }
        if arg != index {
            continue;
        }

        let is_barrier = barriers.is_some_and(|barriers| barriers[index] != 0);
        let meets_threshold = probs[index] >= threshold;
        if is_barrier || meets_threshold {
            out[index] = 1;
        }
        if mask.is_some_and(|mask| mask[index] == 0) {
            out[index] = 0;
        }
    }

    Ok(out)
}

pub fn decode_gaussian_blurred_probs(
    probs: &[f32],
    n: usize,
    bins: usize,
    min_val: f32,
    max_val: f32,
    deviation: f32,
    threshold: f32,
) -> Result<GaussianBlurredResult> {
    let expected_len = n
        .checked_mul(bins)
        .ok_or_else(|| Error::message("probability shape overflow"))?;
    if probs.len() != expected_len {
        return Err(Error::message(format!(
            "probability length mismatch: expected {expected_len} values for shape [{n}, {bins}], got {}",
            probs.len()
        )));
    }
    if bins == 0 {
        return Err(Error::message("gaussian decode requires bins > 0"));
    }
    if !min_val.is_finite()
        || !max_val.is_finite()
        || !deviation.is_finite()
        || !threshold.is_finite()
    {
        return Err(Error::message(
            "gaussian decode scalar parameters must all be finite",
        ));
    }
    if deviation < 0.0 {
        return Err(Error::message(format!(
            "gaussian deviation must be non-negative, got {deviation}"
        )));
    }
    if max_val <= min_val {
        return Err(Error::message(format!(
            "gaussian decode requires max_val > min_val, got min={min_val}, max={max_val}"
        )));
    }

    if n == 0 {
        return Ok(GaussianBlurredResult {
            values: Vec::new(),
            presence: Vec::new(),
        });
    }
    if bins < 2 {
        return Err(Error::message(format!(
            "gaussian decode requires at least 2 bins when n > 0, got {bins}"
        )));
    }

    let step = (max_val - min_val) / ((bins - 1) as f32);
    let width = max(0, (deviation / step).ceil() as i32) as usize;
    let centers = (0..bins)
        .map(|k| min_val + step * (k as f32))
        .collect::<Vec<_>>();

    let mut values = vec![0.0; n];
    let mut presence = vec![0; n];
    for row_index in 0..n {
        let row = &probs[(row_index * bins)..((row_index + 1) * bins)];
        let mut argmax = 0usize;
        let mut max_prob = row[0];
        for (index, &value) in row.iter().enumerate().skip(1) {
            if value > max_prob {
                max_prob = value;
                argmax = index;
            }
        }

        let lo = argmax.saturating_sub(width);
        let hi = min(bins, argmax.saturating_add(width).saturating_add(1));
        let mut weight_sum = 0.0f32;
        let mut value_sum = 0.0f32;
        for index in lo..hi {
            weight_sum += row[index];
            value_sum += row[index] * centers[index];
        }
        values[row_index] = value_sum / (weight_sum + 1e-8);
        presence[row_index] = if max_prob >= threshold { 1 } else { 0 };
    }

    Ok(GaussianBlurredResult { values, presence })
}

fn validate_optional_len(label: &str, values: Option<&[u8]>, expected: usize) -> Result<()> {
    if let Some(values) = values {
        if values.len() != expected {
            return Err(Error::message(format!(
                "{label} length mismatch: expected {expected}, got {}",
                values.len()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{boundaries_to_regions, decode_gaussian_blurred_probs, decode_soft_boundaries};

    #[test]
    fn boundaries_to_regions_keeps_counting_across_masked_frames() {
        let regions = boundaries_to_regions(&[0, 1, 1, 0], Some(&[1, 1, 0, 1])).unwrap();
        assert_eq!(regions, vec![1, 2, 0, 3]);
    }

    #[test]
    fn decode_soft_boundaries_matches_barrier_mask_and_leftmost_tie_rules() {
        let decoded = decode_soft_boundaries(
            &[0.4, 0.4, 0.1, 0.7, 0.6],
            Some(&[0, 0, 1, 0, 0]),
            Some(&[1, 1, 0, 1, 1]),
            0.3,
            1,
        )
        .unwrap();

        assert_eq!(decoded, vec![1, 0, 0, 0, 0]);
    }

    #[test]
    fn decode_soft_boundaries_forces_valid_barriers_even_below_threshold() {
        let decoded =
            decode_soft_boundaries(&[0.1, 0.2, 0.1], Some(&[0, 1, 0]), None, 0.9, 1).unwrap();
        assert_eq!(decoded, vec![0, 1, 0]);
    }

    #[test]
    fn decode_gaussian_blurred_probs_returns_weighted_pitch_and_presence() {
        let decoded = decode_gaussian_blurred_probs(
            &[
                0.0, 0.2, 0.6, 0.3, 0.0, //
                0.4, 0.3, 0.2, 0.1, 0.0,
            ],
            2,
            5,
            60.0,
            64.0,
            1.0,
            0.5,
        )
        .unwrap();

        assert!((decoded.values[0] - 62.09091).abs() < 1e-4);
        assert!((decoded.values[1] - 60.42857).abs() < 1e-4);
        assert_eq!(decoded.presence, vec![1, 0]);
    }
}
