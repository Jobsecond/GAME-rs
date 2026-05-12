pub mod blocks;
pub mod encoder;
pub mod estimator;
pub mod ops;
pub mod segmenter;
pub mod weights;

use std::path::Path;

use rand::random;

use crate::tensor::{CpuDevice, CpuTensor};
#[cfg(feature = "gpu")]
use crate::tensor::{GpuAdapterSelector, GpuDevice, GpuTensor};
use crate::{
    Error, GameModelConfig, InferParams, InferResult, LoadedGgufModel, MelExtractor, Mt19937Rng,
    Note, RandomSource, Result, Tensor, boundaries_to_regions, d3pm_time_schedule,
    decode_gaussian_blurred_probs, decode_soft_boundaries, load_gguf, remove_mutable_boundaries,
};

pub use encoder::{EncoderOutputs, run_encoder};
pub use estimator::{EstimatorOutputs, run_estimator};
pub use ops::build_joint_attn_mask;
pub use segmenter::{SegmenterOutputs, run_segmenter_step};
pub use weights::{GameModelWeights, bind_model_weights};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Gpu,
}

pub struct Model {
    inner: ModelDispatch,
}

enum ModelDispatch {
    Cpu(ModelInner<CpuTensor>),
    #[cfg(feature = "gpu")]
    Gpu(ModelInner<GpuTensor>),
}

struct ModelInner<T: Tensor> {
    config: GameModelConfig,
    device: T::Device,
    mel_extractor: MelExtractor,
    weights: GameModelWeights<T>,
}

pub(crate) const RMS_NORM_EPS: f32 = 1e-6;

impl Model {
    pub fn load(path: impl AsRef<Path>, backend: Backend) -> Result<Self> {
        let loaded = load_gguf(path)?;
        Self::from_loaded_model(loaded, backend)
    }

    #[cfg(feature = "gpu")]
    pub fn load_with_gpu_selector(
        path: impl AsRef<Path>,
        selector: Option<&GpuAdapterSelector>,
    ) -> Result<Self> {
        let loaded = load_gguf(path)?;
        Self::from_loaded_model_with_gpu_selector(loaded, selector)
    }

    pub fn backend(&self) -> Backend {
        self.inner.backend()
    }

    pub fn config(&self) -> &GameModelConfig {
        self.inner.config()
    }

    #[cfg(feature = "gpu")]
    pub fn gpu_adapter_info(&self) -> Option<wgpu::AdapterInfo> {
        self.inner.gpu_adapter_info()
    }

    pub fn infer(&self, waveform: &[f32], params: &InferParams) -> Result<InferResult> {
        let seed = if params.seed == 0 {
            random::<u64>()
        } else {
            params.seed
        };
        let mut rng = Mt19937Rng::new(seed);
        self.infer_with_rng(waveform, params, &mut rng)
    }

    fn from_loaded_model(model: LoadedGgufModel, backend: Backend) -> Result<Self> {
        match backend {
            Backend::Cpu => Self::from_loaded_cpu_model(model),
            Backend::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    Self::from_loaded_model_with_gpu_selector(model, None)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    let _ = model;
                    Err(Error::message(
                        "GPU backend requested but the `gpu` cargo feature is disabled",
                    ))
                }
            }
        }
    }

    fn from_loaded_cpu_model(model: LoadedGgufModel) -> Result<Self> {
        Ok(Self {
            inner: ModelDispatch::Cpu(ModelInner::from_loaded(model, CpuDevice)?),
        })
    }

    #[cfg(feature = "gpu")]
    fn from_loaded_model_with_gpu_selector(
        model: LoadedGgufModel,
        selector: Option<&GpuAdapterSelector>,
    ) -> Result<Self> {
        let device = GpuDevice::new_with_selector(selector)?;
        Ok(Self {
            inner: ModelDispatch::Gpu(ModelInner::from_loaded(model, device)?),
        })
    }

    fn infer_with_rng<R: RandomSource>(
        &self,
        waveform: &[f32],
        params: &InferParams,
        rng: &mut R,
    ) -> Result<InferResult> {
        self.inner.infer_with_rng(waveform, params, rng)
    }
}

impl ModelDispatch {
    fn backend(&self) -> Backend {
        match self {
            Self::Cpu(_) => Backend::Cpu,
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => Backend::Gpu,
        }
    }

    fn config(&self) -> &GameModelConfig {
        match self {
            Self::Cpu(inner) => inner.config(),
            #[cfg(feature = "gpu")]
            Self::Gpu(inner) => inner.config(),
        }
    }

    #[cfg(feature = "gpu")]
    fn gpu_adapter_info(&self) -> Option<wgpu::AdapterInfo> {
        match self {
            Self::Cpu(_) => None,
            Self::Gpu(inner) => Some(inner.device.adapter_info().clone()),
        }
    }

    fn infer_with_rng<R: RandomSource>(
        &self,
        waveform: &[f32],
        params: &InferParams,
        rng: &mut R,
    ) -> Result<InferResult> {
        match self {
            Self::Cpu(inner) => inner.infer_with_rng(waveform, params, rng),
            #[cfg(feature = "gpu")]
            Self::Gpu(inner) => inner.infer_with_rng(waveform, params, rng),
        }
    }
}

impl<T: Tensor> ModelInner<T> {
    fn from_loaded(model: LoadedGgufModel, device: T::Device) -> Result<Self> {
        let mel_extractor = MelExtractor::from_inference_config(&model.config.inference)?;
        let weights = bind_model_weights(&model, &device)?;
        Ok(Self {
            config: model.config,
            device,
            mel_extractor,
            weights,
        })
    }

    fn config(&self) -> &GameModelConfig {
        &self.config
    }

    fn infer_with_rng<R: RandomSource>(
        &self,
        waveform: &[f32],
        params: &InferParams,
        rng: &mut R,
    ) -> Result<InferResult> {
        let seq_len = self.mel_extractor.num_frames(waveform.len());
        if seq_len == 0 {
            return Err(Error::message("waveform too short for one mel frame"));
        }

        let mel = self.mel_extractor.forward(waveform)?;
        let mel_tensor = T::from_data(
            &mel,
            &[seq_len, self.mel_extractor.config().n_mels],
            &self.device,
        )
        .map_err(|err| Error::message(format!("failed to upload mel spectrogram: {err}")))?;

        let encoder = run_encoder(
            &mel_tensor,
            &self.weights.spectrogram_projection,
            &self.weights.encoder,
            &self.config,
        )?;

        let region_cycle_len = i32::try_from(positive_usize(
            "game.model.region_cycle_len",
            self.config.region_cycle_len,
        )?)
        .map_err(|_| Error::message("game.model.region_cycle_len exceeds i32::MAX"))?;
        let schedule = d3pm_schedule(params)?;
        let known = vec![0u8; seq_len];
        let mask = vec![1u8; seq_len];
        let mut boundaries = known.clone();
        let mut noise_mod = vec![0i32; seq_len];

        for t in schedule {
            let removal_probability = d3pm_time_schedule(t);
            boundaries = remove_mutable_boundaries(&boundaries, &known, removal_probability, rng)?;

            let regions = boundaries_to_regions(&boundaries, Some(&mask))?;
            for (dst, region) in noise_mod.iter_mut().zip(regions.iter().copied()) {
                *dst = region % region_cycle_len;
            }

            let segmenter = run_segmenter_step(
                &encoder.x_seg,
                &noise_mod,
                (self.config.mode == "d3pm").then_some(t),
                Some(params.language),
                &self.weights.segmenter,
                &self.config,
            )?;
            let logits = export_tensor(&segmenter.logits)?;
            let probs = sigmoid_all(&logits);
            boundaries = decode_soft_boundaries(
                &probs,
                Some(&known),
                Some(&mask),
                params.boundary_threshold,
                params.boundary_radius,
            )?;
        }

        let regions = boundaries_to_regions(&boundaries, Some(&mask))?;
        let n_regions = max_region_id(&regions)?;
        let mut result = InferResult {
            notes: Vec::new(),
            num_frames: usize_to_i32("inference frame count", seq_len)?,
        };
        if n_regions == 0 {
            return Ok(result);
        }

        let pool_logits = run_estimator(
            &encoder.x_est,
            &regions,
            &self.weights.estimator,
            &self.config,
        )?
        .pool_logits;
        let pool_probs = sigmoid_all(&export_tensor(&pool_logits)?);
        let bins = positive_usize(
            "game.model.estimator_out_dim",
            self.config.estimator_out_dim,
        )?;
        let decoded = decode_gaussian_blurred_probs(
            &pool_probs,
            n_regions,
            bins,
            self.config.inference.midi_min,
            self.config.inference.midi_max,
            self.config.inference.midi_std * 3.0,
            params.note_threshold,
        )?;

        let region_durations = count_region_durations(&regions, n_regions)?;
        let timestep = self.config.inference.timestep();
        let mut offset_seconds = 0.0f32;
        for note_index in 0..n_regions {
            let duration_seconds = region_durations[note_index + 1] as f32 * timestep;
            result.notes.push(Note {
                offset_seconds,
                duration_seconds,
                pitch_midi: decoded.values[note_index],
                voiced: decoded.presence[note_index] != 0,
            });
            offset_seconds += duration_seconds;
        }

        Ok(result)
    }
}

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

fn d3pm_schedule(params: &InferParams) -> Result<Vec<f32>> {
    if params.d3pm_ts.is_empty() {
        return default_d3pm_schedule(params.d3pm_t0, params.d3pm_nsteps);
    }

    let mut out = Vec::with_capacity(params.d3pm_ts.len());
    for (index, &value) in params.d3pm_ts.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::message(format!(
                "InferParams.d3pm_ts[{index}] must be finite, got {value}"
            )));
        }
        out.push(value);
    }
    Ok(out)
}

fn default_d3pm_schedule(t0: f32, n_steps: i32) -> Result<Vec<f32>> {
    if !t0.is_finite() {
        return Err(Error::message(format!(
            "InferParams.d3pm_t0 must be finite, got {t0}"
        )));
    }

    let n_steps = positive_usize("InferParams.d3pm_nsteps", n_steps)?;
    let step = (1.0 - t0) / n_steps as f32;
    let mut ts = Vec::with_capacity(n_steps);
    for index in 0..n_steps {
        ts.push(t0 + step * index as f32);
    }
    Ok(ts)
}

fn sigmoid_all(values: &[f32]) -> Vec<f32> {
    values.iter().copied().map(sigmoid_scalar).collect()
}

fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn export_tensor<T: Tensor>(tensor: &T) -> Result<Vec<f32>> {
    let len = checked_num_elements(tensor.shape())?;
    let mut out = vec![0.0; len];
    tensor.export(&mut out)?;
    Ok(out)
}

fn max_region_id(regions: &[i32]) -> Result<usize> {
    let mut max_region = 0usize;
    for &region in regions {
        let region = usize::try_from(region).map_err(|_| {
            Error::message(format!(
                "region ids must be >= 0, got invalid region id {region}"
            ))
        })?;
        max_region = max_region.max(region);
    }
    Ok(max_region)
}

fn count_region_durations(regions: &[i32], n_regions: usize) -> Result<Vec<usize>> {
    let mut durations = vec![0usize; n_regions + 1];
    for &region in regions {
        let region = usize::try_from(region).map_err(|_| {
            Error::message(format!(
                "region ids must be >= 0, got invalid region id {region}"
            ))
        })?;
        if region > n_regions {
            return Err(Error::message(format!(
                "region id {region} exceeds computed max region count {n_regions}"
            )));
        }
        durations[region] += 1;
    }
    Ok(durations)
}

fn checked_num_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::message(format!("shape {:?} overflows usize", shape)))
    })
}
