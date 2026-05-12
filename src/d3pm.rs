use std::f32::consts::PI;

use crate::rng::RandomSource;
use crate::{Error, Result};

pub fn d3pm_time_schedule(t: f32) -> f32 {
    0.5 * (1.0 + (t * PI).cos())
}

pub fn remove_mutable_boundaries<R: RandomSource>(
    boundaries: &[u8],
    immutable: &[u8],
    p: f32,
    rng: &mut R,
) -> Result<Vec<u8>> {
    let mut out = vec![0; boundaries.len()];
    remove_mutable_boundaries_into(boundaries, immutable, p, rng, &mut out)?;
    Ok(out)
}

pub fn remove_mutable_boundaries_into<R: RandomSource>(
    boundaries: &[u8],
    immutable: &[u8],
    p: f32,
    rng: &mut R,
    out: &mut [u8],
) -> Result<()> {
    if boundaries.len() != immutable.len() {
        return Err(Error::message(format!(
            "boundary length mismatch: boundaries has len {}, immutable has len {}",
            boundaries.len(),
            immutable.len()
        )));
    }
    if out.len() != boundaries.len() {
        return Err(Error::message(format!(
            "output length mismatch: out has len {}, boundaries has len {}",
            out.len(),
            boundaries.len()
        )));
    }
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(Error::message(format!(
            "boundary removal probability must be finite and within [0, 1], got {p}"
        )));
    }

    let mut total_boundaries = 0usize;
    let mut mutable_boundaries = 0usize;
    for index in 0..boundaries.len() {
        let boundary = boundaries[index] != 0;
        let is_immutable = immutable[index] != 0;
        if boundary {
            total_boundaries += 1;
        }
        if boundary && !is_immutable {
            mutable_boundaries += 1;
        }
    }

    let mutable_drop_probability = if mutable_boundaries == 0 {
        1.0
    } else {
        ((total_boundaries as f32) * p / (mutable_boundaries as f32)).min(1.0)
    };
    let keep_probability = 1.0 - mutable_drop_probability;

    let mut samples = vec![0.0; boundaries.len()];
    rng.fill_uniform(&mut samples)?;

    for index in 0..boundaries.len() {
        let boundary = boundaries[index] != 0;
        let is_immutable = immutable[index] != 0;
        let is_mutable_boundary = boundary && !is_immutable;
        let keep_mutable = is_mutable_boundary && samples[index] <= keep_probability;
        out[index] = if boundary && is_immutable {
            1
        } else if keep_mutable {
            1
        } else {
            0
        };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{d3pm_time_schedule, remove_mutable_boundaries, remove_mutable_boundaries_into};
    use crate::InjectedRng;

    #[test]
    fn d3pm_time_schedule_matches_expected_points() {
        assert!((d3pm_time_schedule(0.0) - 1.0).abs() < 1e-6);
        assert!((d3pm_time_schedule(0.5) - 0.5).abs() < 1e-6);
        assert!((d3pm_time_schedule(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn mutable_boundaries_are_rescaled_and_rng_consumes_full_length() {
        let boundaries = [1, 0, 1, 1];
        let immutable = [1, 0, 0, 0];
        let mut rng = InjectedRng::new(vec![0.9, 0.8, 0.2, 0.3]);

        let out = remove_mutable_boundaries(&boundaries, &immutable, 0.5, &mut rng).unwrap();
        assert_eq!(out, vec![1, 0, 1, 0]);
        assert_eq!(rng.remaining(), 0);
    }

    #[test]
    fn mutable_boundary_output_buffer_is_validated() {
        let boundaries = [1, 0];
        let immutable = [1, 0];
        let mut out = [0u8; 1];
        let mut rng = InjectedRng::new(vec![0.1, 0.2]);

        let err = remove_mutable_boundaries_into(&boundaries, &immutable, 0.5, &mut rng, &mut out)
            .unwrap_err();
        assert!(err.to_string().contains("output length mismatch"));
    }
}
