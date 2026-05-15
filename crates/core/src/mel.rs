use std::f32::consts::PI;
use std::sync::{Arc, Mutex};

use realfft::{RealFftPlanner, RealToComplex};

use crate::config::InferenceConfig;
use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub clip_val: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            n_fft: 2_048,
            win_length: 2_048,
            hop_length: 441,
            n_mels: 80,
            fmin: 0.0,
            fmax: 8_000.0,
            clip_val: 1e-5,
        }
    }
}

impl TryFrom<&InferenceConfig> for MelConfig {
    type Error = Error;

    fn try_from(value: &InferenceConfig) -> Result<Self> {
        if value.spectrogram_type != "mel" {
            return Err(Error::message(format!(
                "unsupported spectrogram type `{}` (expected `mel`)",
                value.spectrogram_type
            )));
        }

        Ok(Self {
            sample_rate: positive_usize(value.audio_sample_rate, "audio_sample_rate")?,
            n_fft: positive_usize(value.fft_size, "fft_size")?,
            win_length: positive_usize(value.win_size, "win_size")?,
            hop_length: positive_usize(value.hop_size, "hop_size")?,
            n_mels: positive_usize(value.n_mels, "n_mels")?,
            fmin: value.fmin,
            fmax: value.fmax,
            clip_val: Self::default().clip_val,
        })
    }
}

impl TryFrom<InferenceConfig> for MelConfig {
    type Error = Error;

    fn try_from(value: InferenceConfig) -> Result<Self> {
        Self::try_from(&value)
    }
}

#[derive(Clone)]
pub struct MelExtractor {
    cfg: MelConfig,
    window: Vec<f32>,
    mel_fb: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    fft_lock: Arc<Mutex<()>>,
}

impl MelExtractor {
    pub fn new(cfg: MelConfig) -> Result<Self> {
        validate_config(&cfg)?;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);

        Ok(Self {
            window: make_hann_window(cfg.win_length),
            mel_fb: make_mel_filterbank(cfg.n_fft, cfg.n_mels, cfg.sample_rate, cfg.fmin, cfg.fmax),
            fft,
            fft_lock: Arc::new(Mutex::new(())),
            cfg,
        })
    }

    pub fn from_inference_config(cfg: &InferenceConfig) -> Result<Self> {
        Self::new(MelConfig::try_from(cfg)?)
    }

    pub fn config(&self) -> &MelConfig {
        &self.cfg
    }

    pub fn num_frames(&self, n_samples: usize) -> usize {
        let (pad_l, pad_r) = pad_sizes(self.cfg.win_length, self.cfg.hop_length);
        let padded = n_samples as i64 + pad_l as i64 + pad_r as i64;
        let frames = (padded - self.cfg.win_length as i64) / self.cfg.hop_length as i64 + 1;
        frames.max(0) as usize
    }

    pub fn forward(&self, waveform: &[f32]) -> Result<Vec<f32>> {
        let n_frames = self.num_frames(waveform.len());
        if n_frames == 0 {
            return Ok(Vec::new());
        }

        let (pad_l, pad_r) = pad_sizes(self.cfg.win_length, self.cfg.hop_length);
        let padded = reflect_pad(waveform, pad_l, pad_r)?;
        let n_bins = self.cfg.n_fft / 2 + 1;
        let _fft_guard = self
            .fft_lock
            .lock()
            .map_err(|_| Error::message("mel FFT lock poisoned"))?;

        let mut frame = self.fft.make_input_vec();
        let mut spec = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut mag = vec![0.0; n_bins];
        let mut out = vec![0.0; n_frames * self.cfg.n_mels];

        for t in 0..n_frames {
            let offset = t * self.cfg.hop_length;
            frame.fill(0.0);
            for k in 0..self.cfg.win_length {
                frame[k] = padded[offset + k] * self.window[k];
            }

            self.fft
                .process_with_scratch(&mut frame, &mut spec, &mut scratch)
                .map_err(|err| Error::message(format!("mel FFT failed: {err}")))?;

            for (dst, value) in mag.iter_mut().zip(spec.iter()) {
                *dst = value.norm();
            }

            for m in 0..self.cfg.n_mels {
                let row = &self.mel_fb[m * n_bins..(m + 1) * n_bins];
                let acc: f32 = row.iter().zip(&mag).map(|(w, x)| w * x).sum();
                out[t * self.cfg.n_mels + m] = acc.max(self.cfg.clip_val).ln();
            }
        }

        Ok(out)
    }
}

fn positive_usize(value: i32, field: &str) -> Result<usize> {
    if value <= 0 {
        return Err(Error::message(format!(
            "mel config `{field}` must be positive, got {value}"
        )));
    }
    Ok(value as usize)
}

fn validate_config(cfg: &MelConfig) -> Result<()> {
    if cfg.sample_rate == 0 {
        return Err(Error::message("mel config `sample_rate` must be positive"));
    }
    if cfg.n_fft == 0 {
        return Err(Error::message("mel config `n_fft` must be positive"));
    }
    if cfg.win_length == 0 {
        return Err(Error::message("mel config `win_length` must be positive"));
    }
    if cfg.hop_length == 0 {
        return Err(Error::message("mel config `hop_length` must be positive"));
    }
    if cfg.n_mels == 0 {
        return Err(Error::message("mel config `n_mels` must be positive"));
    }
    if cfg.win_length > cfg.n_fft {
        return Err(Error::message(format!(
            "MelExtractor: win_length ({}) must be <= n_fft ({})",
            cfg.win_length, cfg.n_fft
        )));
    }
    if cfg.hop_length > cfg.win_length {
        return Err(Error::message(format!(
            "MelExtractor: hop_length ({}) must be <= win_length ({})",
            cfg.hop_length, cfg.win_length
        )));
    }
    if !cfg.fmin.is_finite() || !cfg.fmax.is_finite() {
        return Err(Error::message("mel config frequencies must be finite"));
    }
    if cfg.fmin < 0.0 {
        return Err(Error::message(format!(
            "mel config `fmin` must be >= 0, got {}",
            cfg.fmin
        )));
    }
    if cfg.fmax <= cfg.fmin {
        return Err(Error::message(format!(
            "mel config `fmax` must be greater than `fmin`, got fmin={} fmax={}",
            cfg.fmin, cfg.fmax
        )));
    }
    if !cfg.clip_val.is_finite() || cfg.clip_val <= 0.0 {
        return Err(Error::message(format!(
            "mel config `clip_val` must be positive and finite, got {}",
            cfg.clip_val
        )));
    }

    Ok(())
}

fn slaney_freq_sp() -> f32 {
    200.0 / 3.0
}

fn slaney_min_log_hz() -> f32 {
    1_000.0
}

fn slaney_min_log_mel() -> f32 {
    slaney_min_log_hz() / slaney_freq_sp()
}

fn slaney_logstep() -> f32 {
    6.4_f32.ln() / 27.0
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    if hz >= slaney_min_log_hz() {
        slaney_min_log_mel() + (hz / slaney_min_log_hz()).ln() / slaney_logstep()
    } else {
        hz / slaney_freq_sp()
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    if mel >= slaney_min_log_mel() {
        slaney_min_log_hz() * (slaney_logstep() * (mel - slaney_min_log_mel())).exp()
    } else {
        mel * slaney_freq_sp()
    }
}

fn make_mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    sample_rate: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mut fb = vec![0.0; n_mels * n_bins];

    let fft_freqs = (0..n_bins)
        .map(|k| k as f32 * sample_rate as f32 / n_fft as f32)
        .collect::<Vec<_>>();

    let mel_min = hz_to_mel_slaney(fmin);
    let mel_max = hz_to_mel_slaney(fmax);
    let hz_points = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
            mel_to_hz_slaney(mel)
        })
        .collect::<Vec<_>>();

    for m in 0..n_mels {
        let lower = hz_points[m];
        let center = hz_points[m + 1];
        let upper = hz_points[m + 2];
        let enorm = 2.0 / (upper - lower);

        for (k, freq) in fft_freqs.iter().copied().enumerate() {
            let weight = if freq >= lower && freq <= center {
                (freq - lower) / (center - lower)
            } else if freq > center && freq <= upper {
                (upper - freq) / (upper - center)
            } else {
                0.0
            };
            fb[m * n_bins + k] = weight * enorm;
        }
    }

    fb
}

fn make_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / size as f32).cos())
        .collect()
}

fn reflect_pad(signal: &[f32], pad_l: usize, pad_r: usize) -> Result<Vec<f32>> {
    if signal.is_empty() {
        if pad_l == 0 && pad_r == 0 {
            return Ok(Vec::new());
        }
        return Err(Error::message(
            "reflect padding requires a non-empty waveform when padding is non-zero",
        ));
    }
    if pad_l >= signal.len() || pad_r >= signal.len() {
        return Err(Error::message(format!(
            "reflect padding requires pad sizes smaller than the waveform length: left={pad_l} right={pad_r} len={}",
            signal.len()
        )));
    }

    let mut out = vec![0.0; signal.len() + pad_l + pad_r];
    for i in 0..pad_l {
        out[i] = signal[pad_l - i];
    }
    out[pad_l..pad_l + signal.len()].copy_from_slice(signal);
    for i in 0..pad_r {
        out[pad_l + signal.len() + i] = signal[signal.len() - 2 - i];
    }
    Ok(out)
}

fn pad_sizes(win_length: usize, hop_length: usize) -> (usize, usize) {
    let diff = win_length - hop_length;
    (diff / 2, (diff + 1) / 2)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::{
        MelConfig, MelExtractor, hz_to_mel_slaney, make_hann_window, make_mel_filterbank,
        mel_to_hz_slaney, reflect_pad,
    };
    use crate::config::InferenceConfig;

    #[test]
    fn default_config_matches_cpp_defaults() {
        assert_eq!(
            MelConfig::default(),
            MelConfig {
                sample_rate: 44_100,
                n_fft: 2_048,
                win_length: 2_048,
                hop_length: 441,
                n_mels: 80,
                fmin: 0.0,
                fmax: 8_000.0,
                clip_val: 1e-5,
            }
        );
    }

    #[test]
    fn slaney_scale_roundtrips_across_linear_and_log_regions() {
        for hz in [0.0, 440.0, 1_000.0, 8_000.0] {
            let recovered = mel_to_hz_slaney(hz_to_mel_slaney(hz));
            let tol = 1e-4 * hz.max(1.0);
            assert!(
                (recovered - hz).abs() <= tol,
                "hz={hz} recovered={recovered} tol={tol}"
            );
        }
    }

    #[test]
    fn hann_window_matches_torch_periodic_definition() {
        let got = make_hann_window(4);
        assert_close(&got, &[0.0, 0.5, 1.0, 0.5], 1e-6);
    }

    #[test]
    fn reflect_pad_matches_torch_style_edge_exclusion() {
        let got = reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(got, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn mel_filterbank_matches_simple_triangle_case() {
        let got = make_mel_filterbank(4, 1, 8, 0.0, 4.0);
        assert_close(&got, &[0.0, 0.5, 0.0], 1e-6);
    }

    #[test]
    fn num_frames_matches_cpp_behavior_for_default_config() {
        let extractor = MelExtractor::new(MelConfig::default()).unwrap();
        assert_eq!(extractor.num_frames(0), 0);
        assert_eq!(extractor.num_frames(441), 1);
        assert_eq!(extractor.num_frames(44_100), 100);
    }

    #[test]
    fn forward_matches_small_hand_computed_example() {
        let cfg = MelConfig {
            sample_rate: 8,
            n_fft: 4,
            win_length: 4,
            hop_length: 2,
            n_mels: 1,
            fmin: 0.0,
            fmax: 4.0,
            clip_val: 1e-5,
        };
        let extractor = MelExtractor::new(cfg).unwrap();
        let got = extractor.forward(&[1.0, 0.0, 0.0, 0.0]).unwrap();

        assert_close(&got, &[0.25_f32.ln(), 1e-5_f32.ln()], 1e-5);
    }

    #[test]
    fn mel_config_can_be_derived_from_inference_config() {
        let inference = InferenceConfig {
            audio_sample_rate: 16_000,
            hop_size: 320,
            fft_size: 1_024,
            win_size: 1_024,
            n_mels: 80,
            fmin: 0.0,
            fmax: 8_000.0,
            spectrogram_type: "mel".to_owned(),
            ..Default::default()
        };

        let cfg = MelConfig::try_from(&inference).unwrap();
        assert_eq!(
            cfg,
            MelConfig {
                sample_rate: 16_000,
                n_fft: 1_024,
                win_length: 1_024,
                hop_length: 320,
                n_mels: 80,
                fmin: 0.0,
                fmax: 8_000.0,
                clip_val: 1e-5,
            }
        );
    }

    #[test]
    fn forward_matches_reference_dump_when_available() {
        let Some(root) = reference_root() else {
            return;
        };

        let wav_path = root.join("mel").join("mel_wav.bin");
        let out_path = root.join("mel").join("mel_output.bin");
        if !wav_path.exists() || !out_path.exists() {
            return;
        }

        let wav = load_ref_f32(&wav_path).unwrap();
        let expected = load_ref_f32(&out_path).unwrap();

        assert_eq!(wav.shape.len(), 1);
        assert_eq!(expected.shape.len(), 2);

        let extractor = MelExtractor::new(MelConfig::default()).unwrap();
        let got = extractor.forward(&wav.data).unwrap();
        assert_eq!(got.len(), expected.data.len());
        assert_close_with_rtol(&got, &expected.data, 1e-3, 1e-3);
    }

    fn assert_close(actual: &[f32], expected: &[f32], atol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= atol,
                "index={index} actual={actual} expected={expected} diff={diff} atol={atol}"
            );
        }
    }

    fn assert_close_with_rtol(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual - expected).abs();
            let tol = atol + rtol * expected.abs();
            assert!(
                diff <= tol,
                "index={index} actual={actual} expected={expected} diff={diff} tol={tol}"
            );
        }
    }

    fn reference_root() -> Option<PathBuf> {
        if let Some(path) = std::env::var_os("GAME_GGML_TEST_DATA") {
            let path = PathBuf::from(path);
            if path.exists() {
                return Some(path);
            }
        }

        let sibling = Path::new(env!("CARGO_MANIFEST_DIR")).join("../GAME-ggml/tests/data");
        sibling.exists().then_some(sibling)
    }

    fn load_ref_f32(path: &Path) -> Option<RefTensor> {
        let bytes = fs::read(path).ok()?;
        if bytes.len() < 80 || &bytes[0..4] != b"GREF" {
            return None;
        }

        let version = i32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let dtype = i32::from_le_bytes(bytes[8..12].try_into().ok()?);
        let ndim = i32::from_le_bytes(bytes[12..16].try_into().ok()?);
        if version != 1 || dtype != 0 || !(0..=8).contains(&ndim) {
            return None;
        }

        let mut shape = Vec::with_capacity(ndim as usize);
        for dim_index in 0..ndim as usize {
            let start = 16 + dim_index * 8;
            let end = start + 8;
            let dim = i64::from_le_bytes(bytes[start..end].try_into().ok()?);
            shape.push(usize::try_from(dim).ok()?);
        }

        let numel = shape.iter().copied().product::<usize>();
        let payload = &bytes[80..];
        if payload.len() != numel * std::mem::size_of::<f32>() {
            return None;
        }

        let mut data = Vec::with_capacity(numel);
        for chunk in payload.chunks_exact(4) {
            data.push(f32::from_le_bytes(chunk.try_into().ok()?));
        }

        Some(RefTensor { shape, data })
    }

    struct RefTensor {
        shape: Vec<usize>,
        data: Vec<f32>,
    }
}
