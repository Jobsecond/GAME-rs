use rand_mt::Mt64;

use crate::{Error, Result};

const FLOAT_BITS: u32 = 24;
const FLOAT_SCALE: f32 = 1.0 / ((1u32 << FLOAT_BITS) as f32);

pub fn random_u64() -> u64 {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf).expect("OS RNG unavailable");
    u64::from_le_bytes(buf)
}

pub trait RandomSource {
    fn uniform_f32(&mut self) -> Result<f32>;

    fn fill_uniform(&mut self, dst: &mut [f32]) -> Result<()> {
        for value in dst {
            *value = self.uniform_f32()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mt19937Rng {
    inner: Mt64,
}

impl Mt19937Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: Mt64::new(seed),
        }
    }
}

impl RandomSource for Mt19937Rng {
    fn uniform_f32(&mut self) -> Result<f32> {
        // MSVC's std::uniform_real_distribution<float> over std::mt19937_64
        // takes the high 24 random bits and scales them by 2^-24.
        let sample = (self.inner.next_u64() >> (64 - FLOAT_BITS)) as u32;
        Ok(sample as f32 * FLOAT_SCALE)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InjectedRng {
    values: Vec<f32>,
    cursor: usize,
}

impl InjectedRng {
    pub fn new(values: Vec<f32>) -> Self {
        Self { values, cursor: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.values.len().saturating_sub(self.cursor)
    }
}

impl RandomSource for InjectedRng {
    fn uniform_f32(&mut self) -> Result<f32> {
        let value = self.values.get(self.cursor).copied().ok_or_else(|| {
            Error::message(format!(
                "InjectedRng exhausted: needed 1 more value, have {} remaining",
                self.remaining()
            ))
        })?;
        self.cursor += 1;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{InjectedRng, Mt19937Rng, RandomSource};

    #[test]
    fn mt19937_matches_cpp_uniform_real_distribution_seed_42() {
        let expected = [
            0x3f4151df, 0x3f23978f, 0x3f408c96, 0x3e0b8b10, 0x3f673ca2, 0x3dc0a6e0, 0x3f13170a,
            0x3ebeeb22, 0x3e8c3938, 0x3ec7d194, 0x3c4ae100, 0x3f061191, 0x3f2f6df0, 0x3f232897,
            0x3f5398c3, 0x3f72194b, 0x3f40d2ed, 0x3ee5d852, 0x3d3fb480, 0x3d844a20, 0x3f3f62e2,
            0x3e18e6f0, 0x3ed9bb02, 0x3dcce568, 0x3e135e34, 0x3dc0dd70, 0x3f07b8a6, 0x3ee473ec,
            0x3f45e48f, 0x3dfaeac0, 0x3f3ec32a, 0x3ea5c3a4, 0x3f403f87, 0x3f4de8ba, 0x3d323f60,
            0x3f6d150d, 0x3f3e5fc1, 0x3ed50452, 0x3e9862f4, 0x3c9e1b80, 0x3f332a05, 0x3eeb3bba,
            0x3f097443, 0x3efa4c28, 0x3e58c5b0, 0x3f1b703d, 0x3efc7470, 0x3ec4948c, 0x3f2ac752,
            0x3ed4fb3c, 0x3ede791c, 0x3dbb9e58, 0x3d980b30, 0x3f520d5c, 0x3ef47c18, 0x3ee30a0a,
            0x3f7ed5bb, 0x3f184e01, 0x3f1e795e, 0x3f13a970, 0x3f5c7bfd, 0x3ef3917a, 0x3f52e2ae,
            0x3efbcb72, 0x3ee3a3c6, 0x3e9ebc04, 0x3e981706, 0x3f05d5b8, 0x3ebad05e, 0x3f438841,
            0x3e9c687c, 0x3f2fa13b, 0x3f3aa602, 0x3f26955e, 0x3f50f64c, 0x3f498139, 0x3f26ef78,
            0x3f4b399e, 0x3ece1c2a, 0x3f6fc5f7, 0x3eb20e0a, 0x3e995308, 0x3f3cf643, 0x3f18088e,
            0x3eb48d88, 0x3e0e0378, 0x3f1fea20, 0x3f17bca1, 0x3f05885a, 0x3f1ce7e5, 0x3d7260c0,
            0x3ebbe1a0, 0x3f3c3f22, 0x3f1c9c25, 0x3eca8050, 0x3f6eaf31, 0x3f3d9af3, 0x3e6ba594,
            0x3f077464, 0x3dfda5b8,
        ];

        let mut rng = Mt19937Rng::new(42);
        for bits in expected {
            assert_eq!(rng.uniform_f32().unwrap().to_bits(), bits);
        }
    }

    #[test]
    fn mt19937_fill_uniform_consumes_values_in_order() {
        let mut rng = Mt19937Rng::new(42);
        let mut values = [0.0; 4];
        rng.fill_uniform(&mut values).unwrap();

        assert_eq!(values[0].to_bits(), 0x3f4151df);
        assert_eq!(values[1].to_bits(), 0x3f23978f);
        assert_eq!(values[2].to_bits(), 0x3f408c96);
        assert_eq!(values[3].to_bits(), 0x3e0b8b10);
    }

    #[test]
    fn injected_rng_tracks_remaining_and_errors_on_exhaustion() {
        let mut rng = InjectedRng::new(vec![0.25, 0.5]);
        assert_eq!(rng.remaining(), 2);
        assert_eq!(rng.uniform_f32().unwrap(), 0.25);
        assert_eq!(rng.remaining(), 1);
        assert_eq!(rng.uniform_f32().unwrap(), 0.5);
        assert_eq!(rng.remaining(), 0);
        assert!(
            rng.uniform_f32()
                .unwrap_err()
                .to_string()
                .contains("exhausted")
        );
    }
}
