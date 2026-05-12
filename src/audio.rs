use std::path::Path;

use hound::{SampleFormat, WavReader};

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedWaveform {
    pub samples: Vec<f32>,
    pub sample_rate: usize,
    pub source_sample_rate: usize,
    pub source_channels: usize,
}

impl PreparedWaveform {
    pub fn was_resampled(&self) -> bool {
        self.sample_rate != self.source_sample_rate
    }

    pub fn was_downmixed(&self) -> bool {
        self.source_channels != 1
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SliceChunk {
    pub offset_seconds: f64,
    pub waveform: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlicerConfig {
    pub sample_rate: usize,
    pub threshold_db: f32,
    pub min_length_ms: usize,
    pub min_interval_ms: usize,
    pub hop_ms: usize,
    pub max_sil_kept_ms: usize,
}

impl Default for SlicerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            threshold_db: -40.0,
            min_length_ms: 5_000,
            min_interval_ms: 300,
            hop_ms: 20,
            max_sil_kept_ms: 5_000,
        }
    }
}

pub fn prepare_wav_for_inference(path: &Path, target_sample_rate: i32) -> Result<PreparedWaveform> {
    let source = load_wav_mono_f32(path)?;
    let target_sample_rate = resolve_target_sample_rate(target_sample_rate, source.sample_rate)?;
    let samples = if source.sample_rate == target_sample_rate {
        source.samples.clone()
    } else {
        resample_linear(&source.samples, source.sample_rate, target_sample_rate)?
    };

    Ok(PreparedWaveform {
        samples,
        sample_rate: target_sample_rate,
        source_sample_rate: source.sample_rate,
        source_channels: source.source_channels,
    })
}

pub fn load_wav_mono_f32(path: &Path) -> Result<PreparedWaveform> {
    let mut reader = WavReader::open(path)
        .map_err(|err| Error::message(format!("failed to open WAV {}: {err}", path.display())))?;
    let spec = reader.spec();
    let sample_rate = usize::try_from(spec.sample_rate).map_err(|_| {
        Error::message(format!(
            "WAV sample rate {} cannot fit in usize ({})",
            spec.sample_rate,
            path.display()
        ))
    })?;
    let channels = usize::from(spec.channels);
    if channels == 0 {
        return Err(Error::message(format!(
            "WAV {} reports zero channels",
            path.display()
        )));
    }

    let interleaved = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|err| {
                Error::message(format!("failed to decode WAV {}: {err}", path.display()))
            })?,
        SampleFormat::Int => {
            let bits = u32::from(spec.bits_per_sample);
            if bits == 0 || bits > 32 {
                return Err(Error::message(format!(
                    "unsupported WAV bit depth {} in {}",
                    spec.bits_per_sample,
                    path.display()
                )));
            }

            let scale = 1u64.checked_shl(bits - 1).ok_or_else(|| {
                Error::message(format!(
                    "unsupported WAV bit depth {} in {}",
                    spec.bits_per_sample,
                    path.display()
                ))
            })? as f32;

            reader
                .samples::<i32>()
                .map(|sample| {
                    sample.map(|value| value as f32 / scale).map_err(|err| {
                        Error::message(format!("failed to decode WAV {}: {err}", path.display()))
                    })
                })
                .collect::<Result<Vec<_>>>()?
        }
    };

    let samples = interleaved_to_mono(&interleaved, channels)?;
    Ok(PreparedWaveform {
        samples,
        sample_rate,
        source_sample_rate: sample_rate,
        source_channels: channels,
    })
}

pub fn resample_linear(
    samples: &[f32],
    src_sample_rate: usize,
    dst_sample_rate: usize,
) -> Result<Vec<f32>> {
    if src_sample_rate == 0 || dst_sample_rate == 0 {
        return Err(Error::message(format!(
            "resample requires positive sample rates, got src={} dst={}",
            src_sample_rate, dst_sample_rate
        )));
    }
    if src_sample_rate == dst_sample_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    if samples.len() == 1 {
        return Ok(vec![
            samples[0];
            resampled_length(
                samples.len(),
                src_sample_rate,
                dst_sample_rate
            )?
        ]);
    }

    let dst_len = resampled_length(samples.len(), src_sample_rate, dst_sample_rate)?;
    let step = src_sample_rate as f64 / dst_sample_rate as f64;
    let mut out = Vec::with_capacity(dst_len);
    for index in 0..dst_len {
        let position = index as f64 * step;
        let left = position.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let frac = (position - left as f64) as f32;
        let value = samples[left] + (samples[right] - samples[left]) * frac;
        out.push(value);
    }
    Ok(out)
}

pub fn slice_waveform(samples: &[f32], cfg: &SlicerConfig) -> Result<Vec<SliceChunk>> {
    let sr = cfg.sample_rate;
    if sr == 0 {
        return Err(Error::message(
            "slicer config `sample_rate` must be positive",
        ));
    }
    if cfg.hop_ms == 0 {
        return Err(Error::message("slicer config `hop_ms` must be positive"));
    }
    if cfg.min_interval_ms == 0 {
        return Err(Error::message(
            "slicer config `min_interval_ms` must be positive",
        ));
    }

    let hop_size = ((sr * cfg.hop_ms) / 1_000).max(1);
    let min_interval_samples = sr as f64 * cfg.min_interval_ms as f64 / 1_000.0;
    let win_size = min_interval_samples.round() as usize;
    let win_size = win_size.min(4 * hop_size).max(1);
    let min_length =
        ((sr as f64 * cfg.min_length_ms as f64 / 1_000.0) / hop_size as f64).round() as usize;
    let min_interval = (min_interval_samples / hop_size as f64).round() as usize;
    let max_sil_kept =
        ((sr as f64 * cfg.max_sil_kept_ms as f64 / 1_000.0) / hop_size as f64).round() as usize;
    let threshold = 10.0f32.powf(cfg.threshold_db / 20.0);

    if samples.len().div_ceil(hop_size) <= min_length {
        return Ok(vec![SliceChunk {
            offset_seconds: 0.0,
            waveform: samples.to_vec(),
        }]);
    }

    let rms = frame_rms(samples, win_size, hop_size);
    let n_frames = rms.len();
    let mut sil_tags = Vec::<(usize, usize)>::new();
    let mut silence_start = None::<usize>;
    let mut clip_start = 0usize;

    for index in 0..n_frames {
        if rms[index] < threshold {
            silence_start.get_or_insert(index);
            continue;
        }

        let Some(start) = silence_start else {
            continue;
        };

        let is_leading_silence = start == 0 && index > max_sil_kept;
        let need_slice_middle = index.saturating_sub(start) >= min_interval
            && index.saturating_sub(clip_start) >= min_length;
        if !is_leading_silence && !need_slice_middle {
            silence_start = None;
            continue;
        }

        if index - start <= max_sil_kept {
            let pos = argmin(&rms[start..=index]) + start;
            if start == 0 {
                sil_tags.push((0, pos));
            } else {
                sil_tags.push((pos, pos));
            }
            clip_start = pos;
        } else if index - start <= max_sil_kept.saturating_mul(2) {
            let lo = index - max_sil_kept;
            let hi = (start + max_sil_kept).min(n_frames - 1);
            let pos = argmin(&rms[lo..=hi]) + lo;
            let pos_l = argmin(&rms[start..=start + max_sil_kept]) + start;
            let pos_r = argmin(&rms[lo..=index]) + lo;
            if start == 0 {
                sil_tags.push((0, pos_r));
                clip_start = pos_r;
            } else {
                sil_tags.push((pos_l.min(pos), pos_r.max(pos)));
                clip_start = pos_r.max(pos);
            }
        } else {
            let pos_l = argmin(&rms[start..=start + max_sil_kept]) + start;
            let lo = index - max_sil_kept;
            let pos_r = argmin(&rms[lo..=index]) + lo;
            if start == 0 {
                sil_tags.push((0, pos_r));
            } else {
                sil_tags.push((pos_l, pos_r));
            }
            clip_start = pos_r;
        }

        silence_start = None;
    }

    if let Some(start) = silence_start {
        if n_frames.saturating_sub(start) >= min_interval {
            let silence_end = start
                .saturating_add(max_sil_kept)
                .min(n_frames.saturating_sub(1));
            let pos = argmin(&rms[start..=silence_end]) + start;
            sil_tags.push((pos, n_frames.saturating_add(1)));
        }
    }

    let apply_slice = |begin: usize, end: usize| {
        let start = begin.saturating_mul(hop_size);
        let end = samples.len().min(end.saturating_mul(hop_size));
        SliceChunk {
            offset_seconds: start as f64 / sr as f64,
            waveform: samples[start..end].to_vec(),
        }
    };

    let mut chunks = Vec::new();
    if sil_tags.is_empty() {
        chunks.push(SliceChunk {
            offset_seconds: 0.0,
            waveform: samples.to_vec(),
        });
        return Ok(chunks);
    }

    if sil_tags[0].0 > 0 {
        chunks.push(apply_slice(0, sil_tags[0].0));
    }
    for index in 0..sil_tags.len().saturating_sub(1) {
        chunks.push(apply_slice(sil_tags[index].1, sil_tags[index + 1].0));
    }
    if sil_tags.last().is_some_and(|(_, end)| *end < n_frames) {
        let (_, end) = sil_tags[sil_tags.len() - 1];
        chunks.push(apply_slice(end, n_frames));
    }
    if chunks.is_empty() {
        chunks.push(SliceChunk {
            offset_seconds: 0.0,
            waveform: samples.to_vec(),
        });
    }

    Ok(chunks)
}

pub fn split_long_chunks(
    chunks: &[SliceChunk],
    sample_rate: usize,
    max_chunk_samples: usize,
) -> Result<Vec<SliceChunk>> {
    if sample_rate == 0 {
        return Err(Error::message(
            "chunk splitting requires a positive sample rate",
        ));
    }
    if max_chunk_samples == 0 {
        return Err(Error::message(
            "chunk splitting requires a positive max_chunk_samples",
        ));
    }

    let mut out = Vec::new();
    for chunk in chunks {
        if chunk.waveform.len() <= max_chunk_samples {
            out.push(chunk.clone());
            continue;
        }

        for start in (0..chunk.waveform.len()).step_by(max_chunk_samples) {
            let end = (start + max_chunk_samples).min(chunk.waveform.len());
            out.push(SliceChunk {
                offset_seconds: chunk.offset_seconds + start as f64 / sample_rate as f64,
                waveform: chunk.waveform[start..end].to_vec(),
            });
        }
    }

    Ok(out)
}

fn resolve_target_sample_rate(
    target_sample_rate: i32,
    fallback_sample_rate: usize,
) -> Result<usize> {
    match target_sample_rate {
        value if value < 0 => Err(Error::message(format!(
            "target sample rate must be >= 0, got {value}"
        ))),
        0 => Ok(fallback_sample_rate),
        value => Ok(value as usize),
    }
}

fn interleaved_to_mono(interleaved: &[f32], channels: usize) -> Result<Vec<f32>> {
    if channels == 0 {
        return Err(Error::message("cannot mix down zero-channel audio"));
    }
    if interleaved.len() % channels != 0 {
        return Err(Error::message(format!(
            "interleaved sample count {} is not divisible by channel count {}",
            interleaved.len(),
            channels
        )));
    }
    if channels == 1 {
        return Ok(interleaved.to_vec());
    }

    let frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for frame in interleaved.chunks_exact(channels) {
        mono.push(frame.iter().copied().sum::<f32>() / channels as f32);
    }
    Ok(mono)
}

fn resampled_length(
    input_len: usize,
    src_sample_rate: usize,
    dst_sample_rate: usize,
) -> Result<usize> {
    let numerator = (input_len as u128)
        .checked_mul(dst_sample_rate as u128)
        .ok_or_else(|| Error::message("resampled output length overflow"))?;
    let length = ((numerator + (src_sample_rate as u128 / 2)) / src_sample_rate as u128).max(1);
    usize::try_from(length).map_err(|_| Error::message("resampled output length overflow"))
}

fn frame_rms(samples: &[f32], frame_len: usize, hop_len: usize) -> Vec<f32> {
    let pad_l = frame_len / 2;
    let pad_r = frame_len / 2;
    let mut padded = vec![0.0; samples.len() + pad_l + pad_r];
    padded[pad_l..pad_l + samples.len()].copy_from_slice(samples);

    let n_frames = (padded.len() - frame_len) / hop_len + 1;
    let mut rms = Vec::with_capacity(n_frames);
    for frame_index in 0..n_frames {
        let start = frame_index * hop_len;
        let acc: f64 = padded[start..start + frame_len]
            .iter()
            .map(|&value| {
                let value = f64::from(value);
                value * value
            })
            .sum();
        rms.push((acc / frame_len as f64).sqrt() as f32);
    }
    rms
}

fn argmin(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, &value) in values.iter().enumerate().skip(1) {
        if value < values[best] {
            best = index;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::{
        PreparedWaveform, SliceChunk, SlicerConfig, interleaved_to_mono, resample_linear,
        slice_waveform, split_long_chunks,
    };

    #[test]
    fn prepared_waveform_flags_track_resample_and_mixdown() {
        let waveform = PreparedWaveform {
            samples: vec![0.0; 4],
            sample_rate: 44_100,
            source_sample_rate: 48_000,
            source_channels: 2,
        };
        assert!(waveform.was_resampled());
        assert!(waveform.was_downmixed());
    }

    #[test]
    fn mixdown_averages_each_frame() {
        let mono = interleaved_to_mono(&[1.0, 3.0, 5.0, 7.0], 2).unwrap();
        assert_eq!(mono, vec![2.0, 6.0]);
    }

    #[test]
    fn linear_resampler_preserves_constant_signal_and_target_length() {
        let signal = vec![0.25; 48_000];
        let resampled = resample_linear(&signal, 48_000, 44_100).unwrap();
        assert_eq!(resampled.len(), 44_100);
        assert!(resampled.iter().all(|&value| (value - 0.25).abs() <= 1e-6));
    }

    #[test]
    fn linear_resampler_interpolates_between_adjacent_samples() {
        let resampled = resample_linear(&[0.0, 10.0], 2, 4).unwrap();
        assert_eq!(resampled, vec![0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn slicer_returns_single_chunk_for_short_clip() {
        let samples = vec![0.1f32; 22_050];
        let cfg = SlicerConfig {
            sample_rate: 44_100,
            min_length_ms: 1_000,
            ..SlicerConfig::default()
        };
        let chunks = slice_waveform(&samples, &cfg).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].offset_seconds, 0.0);
        assert_eq!(chunks[0].waveform.len(), samples.len());
    }

    #[test]
    fn slicer_splits_on_silence() {
        let sr = 44_100usize;
        let seg = sr;
        let mut samples = vec![0.0f32; seg * 3];
        for index in 0..seg {
            let value = 0.5 * (2.0 * PI * 440.0 * index as f32 / sr as f32).sin();
            samples[index] = value;
            samples[2 * seg + index] = value;
        }

        let cfg = SlicerConfig {
            sample_rate: sr,
            min_length_ms: 500,
            min_interval_ms: 300,
            max_sil_kept_ms: 200,
            ..SlicerConfig::default()
        };
        let chunks = slice_waveform(&samples, &cfg).unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks[0].offset_seconds <= 0.1);
    }

    #[test]
    fn long_chunks_are_split_without_losing_offsets_or_samples() {
        let chunks = vec![SliceChunk {
            offset_seconds: 1.5,
            waveform: (0..10).map(|value| value as f32).collect(),
        }];

        let split = split_long_chunks(&chunks, 2, 4).unwrap();
        assert_eq!(split.len(), 3);
        assert_eq!(split[0].offset_seconds, 1.5);
        assert_eq!(split[1].offset_seconds, 3.5);
        assert_eq!(split[2].offset_seconds, 5.5);
        assert_eq!(split[0].waveform, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(split[1].waveform, vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(split[2].waveform, vec![8.0, 9.0]);
    }
}
