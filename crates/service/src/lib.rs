use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Arc, Condvar, OnceLock};
use std::time::{Duration, Instant};

use game_audio::{
    PreparedWaveform, SliceChunk, SlicerConfig, prepare_wav_for_inference, slice_waveform,
    split_long_chunks,
};
#[cfg(feature = "gpu")]
use game_core::GpuAdapterSelector;
use game_core::random_u64;
use game_core::{MelExtractor, Model};
use game_output::{write_midi_file, write_text_file};
use rayon::prelude::*;

pub use game_core::{
    Backend, CoreEvent, Error, InferParams, Note, NotificationLevel, Notifier, NullNotifier, Result,
};
pub use game_output::{MidiWriteOptions, TextOutputFormat, TextWriteOptions};

pub const DEFAULT_MAX_CHUNK_SECONDS: usize = 60;

/// Semaphore for limiting concurrent chunk inference to prevent memory exhaustion.
/// Initialized once per process with capacity = number of Rayon threads.
struct ChunkSemaphore {
    inner: Arc<(Mutex<usize>, Condvar)>,
}

impl Clone for ChunkSemaphore {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl ChunkSemaphore {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new((Mutex::new(capacity), Condvar::new())),
        }
    }

    fn acquire(&self) {
        let (lock, cond) = &*self.inner;
        let mut count = lock.lock().unwrap();
        while *count == 0 {
            count = cond.wait(count).unwrap();
        }
        *count -= 1;
    }

    fn release(&self) {
        let (lock, cond) = &*self.inner;
        let mut count = lock.lock().unwrap();
        *count += 1;
        cond.notify_one();
    }
}

struct SemaphoreGuard(ChunkSemaphore);
impl Drop for SemaphoreGuard {
    fn drop(&mut self) {
        self.0.release();
    }
}

static CHUNK_SEMAPHORE: OnceLock<ChunkSemaphore> = OnceLock::new();

fn get_chunk_semaphore() -> ChunkSemaphore {
    CHUNK_SEMAPHORE
        .get_or_init(|| {
            let max_concurrent = std::env::var("GAME_MAX_CONCURRENT_CHUNKS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or_else(|| rayon::current_num_threads());
            ChunkSemaphore::new(max_concurrent)
        })
        .clone()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtractDevice {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkParallelism {
    #[default]
    Auto,
    On,
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractFormat {
    Midi,
    Txt,
    Csv,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuSelector {
    pub name_substring: Option<String>,
    pub vendor_id: Option<u32>,
    pub device_id: Option<u32>,
}

impl GpuSelector {
    pub fn has_any(&self) -> bool {
        self.name_substring.is_some() || self.vendor_id.is_some() || self.device_id.is_some()
    }

    #[cfg(feature = "gpu")]
    fn to_core_selector(&self) -> Option<GpuAdapterSelector> {
        self.has_any().then(|| GpuAdapterSelector {
            name_substring: self.name_substring.clone(),
            vendor_id: self.vendor_id,
            device_id: self.device_id,
            backend: None,
            device_type: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractOutputRequest {
    pub path: PathBuf,
    pub format: Option<ExtractFormat>,
    pub midi_options: MidiWriteOptions,
    pub text_options: TextWriteOptions,
}

impl ExtractOutputRequest {
    pub fn resolved_format(&self) -> ExtractFormat {
        resolve_extract_format(self.format, &self.path)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractRequest {
    pub model_path: PathBuf,
    pub input_path: PathBuf,
    pub output: Option<ExtractOutputRequest>,
    pub device: ExtractDevice,
    pub gpu: GpuSelector,
    pub infer_params: InferParams,
    pub chunk_parallelism: ChunkParallelism,
    pub max_chunk_seconds: usize,
}

impl Default for ExtractRequest {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            input_path: PathBuf::new(),
            output: None,
            device: ExtractDevice::Auto,
            gpu: GpuSelector::default(),
            infer_params: InferParams::default(),
            chunk_parallelism: ChunkParallelism::Auto,
            max_chunk_seconds: DEFAULT_MAX_CHUNK_SECONDS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractOutputResult {
    pub path: PathBuf,
    pub format: ExtractFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedAudioInfo {
    pub sample_rate: usize,
    pub source_sample_rate: usize,
    pub source_channels: usize,
    pub sample_count: usize,
}

impl PreparedAudioInfo {
    pub fn was_resampled(&self) -> bool {
        self.sample_rate != self.source_sample_rate
    }

    pub fn was_downmixed(&self) -> bool {
        self.source_channels != 1
    }

    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.sample_count as f64 / self.sample_rate as f64
        }
    }
}

impl From<&PreparedWaveform> for PreparedAudioInfo {
    fn from(value: &PreparedWaveform) -> Self {
        Self {
            sample_rate: value.sample_rate,
            source_sample_rate: value.source_sample_rate,
            source_channels: value.source_channels,
            sample_count: value.samples.len(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExtractTimings {
    pub total: Duration,
    pub model_load: Duration,
    pub audio_prepare: Duration,
    pub silence_slice: Duration,
    pub long_chunk_split: Duration,
    pub mel_setup: Duration,
    pub inference: Duration,
    pub output_write: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractResult {
    pub notes: Vec<Note>,
    pub backend: Backend,
    pub gpu_adapter: Option<GpuAdapterInfo>,
    pub audio: PreparedAudioInfo,
    pub total_frames: usize,
    pub timestep_seconds: f32,
    pub chunk_count: usize,
    pub chunks_before_long_split: usize,
    pub output: Option<ExtractOutputResult>,
    pub timings: ExtractTimings,
}

#[derive(Debug)]
struct ChunkedExtractResult {
    notes: Vec<Note>,
    chunk_count: usize,
}

#[derive(Debug)]
struct ChunkInferenceResult {
    index: usize,
    notes: Vec<Note>,
    elapsed: Duration,
}

#[derive(Clone, Copy)]
struct PrefixedNotifier<'a> {
    inner: &'a dyn Notifier,
    chunk_index: usize,
    chunk_count: usize,
}

impl Notifier for PrefixedNotifier<'_> {
    fn notify(&self, event: CoreEvent) {
        self.inner.notify(match event {
            CoreEvent::Status { stage, message } => CoreEvent::Status {
                stage,
                message: self.prefix_text(&message),
            },
            CoreEvent::Progress {
                stage,
                current,
                total,
                detail,
            } => CoreEvent::Progress {
                stage,
                current,
                total,
                detail: Some(match detail {
                    Some(detail) => self.prefix_text(&detail),
                    None => self.chunk_label(),
                }),
            },
            CoreEvent::Timing {
                stage,
                elapsed,
                detail,
            } => CoreEvent::Timing {
                stage,
                elapsed,
                detail: Some(match detail {
                    Some(detail) => self.prefix_text(&detail),
                    None => self.chunk_label(),
                }),
            },
            CoreEvent::Message { level, message } => CoreEvent::Message {
                level,
                message: self.prefix_text(&message),
            },
            CoreEvent::ModelLoaded { backend, elapsed } => {
                CoreEvent::ModelLoaded { backend, elapsed }
            }
        });
    }
}

impl PrefixedNotifier<'_> {
    fn chunk_label(&self) -> String {
        format!("chunk {}/{}", self.chunk_index + 1, self.chunk_count)
    }

    fn prefix_text(&self, text: &str) -> String {
        format!("{}: {text}", self.chunk_label())
    }
}

pub fn extract(request: &ExtractRequest) -> Result<ExtractResult> {
    extract_with_notifier(request, &NullNotifier)
}

pub fn extract_with_notifier(
    request: &ExtractRequest,
    notifier: &dyn Notifier,
) -> Result<ExtractResult> {
    validate_request(request)?;

    let total_started_at = Instant::now();
    let (model, model_load) = timed_result(|| {
        load_model_for_extract(&request.model_path, request.device, &request.gpu, notifier)
    })?;
    let backend = model.backend();
    let gpu_adapter = gpu_adapter_info(&model);
    if let Some(adapter) = &gpu_adapter {
        emit_message(
            notifier,
            NotificationLevel::Info,
            format!(
                "gpu adapter: {} (backend={}, type={}, vendor=0x{:04x}, device=0x{:04x})",
                adapter.name,
                adapter.backend,
                adapter.device_type,
                adapter.vendor_id,
                adapter.device_id
            ),
        );
    }

    emit_status(notifier, "audio_prepare", "preparing audio");
    let (waveform, audio_prepare) = timed_result(|| {
        prepare_wav_for_inference(
            &request.input_path,
            model.config().inference.audio_sample_rate,
        )
    })?;
    let audio = PreparedAudioInfo::from(&waveform);
    emit_timing(
        notifier,
        "audio_prepare",
        audio_prepare,
        Some(audio_prepare_detail(&audio)),
    );

    let slicer_config = SlicerConfig {
        sample_rate: waveform.sample_rate,
        ..SlicerConfig::default()
    };
    emit_status(notifier, "silence_slice", "slicing audio on silence");
    let (sliced_chunks, silence_slice) =
        timed_result(|| slice_waveform(&waveform.samples, &slicer_config))?;
    emit_timing(
        notifier,
        "silence_slice",
        silence_slice,
        Some(format!("chunks={}", sliced_chunks.len())),
    );

    emit_status(notifier, "long_chunk_split", "splitting long chunks");
    let max_chunk_samples = waveform
        .sample_rate
        .saturating_mul(request.max_chunk_seconds);
    let (chunks, long_chunk_split) = timed_result(|| {
        split_long_chunks(&sliced_chunks, waveform.sample_rate, max_chunk_samples)
    })?;
    emit_timing(
        notifier,
        "long_chunk_split",
        long_chunk_split,
        Some(format!(
            "chunks={} max_chunk_seconds={} max_chunk_samples={}",
            chunks.len(),
            request.max_chunk_seconds,
            max_chunk_samples
        )),
    );

    emit_status(notifier, "mel_setup", "initializing mel extractor");
    let (mel_extractor, mel_setup) =
        timed_result(|| MelExtractor::from_inference_config(&model.config().inference))?;
    let total_frames = mel_extractor.num_frames(waveform.samples.len());
    emit_timing(
        notifier,
        "mel_setup",
        mel_setup,
        Some(format!("frames={total_frames}")),
    );

    emit_status(
        notifier,
        "extract_infer",
        format!("running inference across {} chunk(s)", chunks.len()),
    );
    let (chunked_result, inference) = timed_result(|| {
        run_chunked_extract_with_notifier(
            &model,
            &chunks,
            &request.infer_params,
            request.chunk_parallelism,
            notifier,
        )
    })?;
    emit_timing(
        notifier,
        "extract_infer",
        inference,
        Some(format!(
            "chunks={} notes={}",
            chunked_result.chunk_count,
            chunked_result.notes.len()
        )),
    );

    let (output, output_write) = match &request.output {
        Some(output) => {
            let format = output.resolved_format();
            emit_status(notifier, "output_write", "writing output");
            let (_, elapsed) = timed_result(|| {
                ensure_output_parent_dir(&output.path)?;
                write_extract_output(
                    &output.path,
                    format,
                    &chunked_result.notes,
                    &output.midi_options,
                    &output.text_options,
                )
            })?;
            emit_timing(
                notifier,
                "output_write",
                elapsed,
                Some(format!(
                    "path={} format={}",
                    output.path.display(),
                    extract_format_name(format)
                )),
            );
            (
                Some(ExtractOutputResult {
                    path: output.path.clone(),
                    format,
                }),
                elapsed,
            )
        }
        None => (None, Duration::ZERO),
    };

    Ok(ExtractResult {
        notes: chunked_result.notes,
        backend,
        gpu_adapter,
        audio,
        total_frames,
        timestep_seconds: model.config().inference.timestep(),
        chunk_count: chunked_result.chunk_count,
        chunks_before_long_split: sliced_chunks.len(),
        output,
        timings: ExtractTimings {
            total: total_started_at.elapsed(),
            model_load,
            audio_prepare,
            silence_slice,
            long_chunk_split,
            mel_setup,
            inference,
            output_write,
        },
    })
}

pub fn resolve_extract_format(format: Option<ExtractFormat>, output: &Path) -> ExtractFormat {
    format
        .or_else(|| infer_extract_format(output))
        .unwrap_or(ExtractFormat::Midi)
}

pub fn infer_extract_format(output: &Path) -> Option<ExtractFormat> {
    let extension = output.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "mid" | "midi" => Some(ExtractFormat::Midi),
        "txt" => Some(ExtractFormat::Txt),
        "csv" => Some(ExtractFormat::Csv),
        _ => None,
    }
}

fn validate_request(request: &ExtractRequest) -> Result<()> {
    if request.device == ExtractDevice::Cpu && request.gpu.has_any() {
        return Err(Error::message(
            "GPU selector fields cannot be used with CPU extraction",
        ));
    }
    if request.max_chunk_seconds == 0 {
        return Err(Error::message(
            "max_chunk_seconds must be greater than zero",
        ));
    }
    Ok(())
}

fn run_chunked_extract_with_notifier(
    model: &Model,
    chunks: &[SliceChunk],
    params: &InferParams,
    chunk_parallelism: ChunkParallelism,
    notifier: &dyn Notifier,
) -> Result<ChunkedExtractResult> {
    let chunk_count = chunks.len();
    let parallel_chunks = chunk_parallelism_enabled(model, chunk_count, chunk_parallelism);
    if chunk_parallelism == ChunkParallelism::On
        && !parallel_chunks
        && model.backend() != Backend::Cpu
    {
        emit_message(
            notifier,
            NotificationLevel::Warn,
            "chunk parallelism forced on but GPU backend does not support it; falling back to serial",
        );
    }
    let random_chunk_seed_base = (parallel_chunks && params.seed == 0).then(random_u64);
    for (index, chunk) in chunks.iter().enumerate() {
        let chunk_duration_seconds =
            chunk.waveform.len() as f64 / model.config().inference.audio_sample_rate as f64;
        emit_status(
            notifier,
            "chunk_infer",
            format!(
                "chunk {}/{}: infer start offset={:.2}s duration={:.2}s",
                index + 1,
                chunk_count,
                chunk.offset_seconds,
                chunk_duration_seconds
            ),
        );
    }

    let results = if parallel_chunks {
        chunks
            .par_iter()
            .enumerate()
            .map(|(index, chunk)| {
                infer_chunk_caught(
                    model,
                    chunk,
                    params,
                    index,
                    chunk_count,
                    random_chunk_seed_base,
                    notifier,
                )
            })
            .collect::<Vec<_>>()
    } else {
        chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                infer_chunk_caught(
                    model,
                    chunk,
                    params,
                    index,
                    chunk_count,
                    random_chunk_seed_base,
                    notifier,
                )
            })
            .collect::<Vec<_>>()
    };

    let mut results = results
        .into_iter()
        .enumerate()
        .map(|(index, r)| {
            r.map_err(|err| {
                let msg = err.to_string();
                if msg.contains("chunk") {
                    err
                } else {
                    Error::message(format!("chunk {}/{}: {err}", index + 1, chunk_count))
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;
    results.sort_unstable_by_key(|result| result.index);

    let mut notes = Vec::new();
    for result in results {
        emit_timing(
            notifier,
            "chunk_infer",
            result.elapsed,
            Some(format!(
                "chunk {}/{} notes={}",
                result.index + 1,
                chunk_count,
                result.notes.len()
            )),
        );
        notes.extend(result.notes);
    }

    Ok(ChunkedExtractResult { notes, chunk_count })
}

fn chunk_parallelism_enabled(
    model: &Model,
    chunk_count: usize,
    chunk_parallelism: ChunkParallelism,
) -> bool {
    let cli_enabled = match chunk_parallelism {
        ChunkParallelism::Auto => true,
        ChunkParallelism::On => true,
        ChunkParallelism::Off => false,
    };
    cli_enabled
        && model.backend() == Backend::Cpu
        && chunk_count > 1
        && rayon::current_num_threads() > 1
        && std::env::var_os("GAME_DISABLE_CHUNK_PARALLELISM").is_none()
}

/// Runs [`infer_chunk`] but converts a panic into a typed [`Error`] instead of
/// letting it unwind. In the parallel path Rayon re-raises a worker panic on the
/// collecting thread, which would bypass the CLI's clean error/exit-code handling
/// and abort with a raw backtrace. Catching it here keeps a single bad chunk from
/// taking down the whole process: the error flows through the normal `Result`
/// aggregation and the operator gets an attributable message.
///
/// Also acquires a semaphore permit before inference to limit concurrent chunks
/// and prevent memory exhaustion when many chunks are queued on Rayon workers.
fn infer_chunk_caught(
    model: &Model,
    chunk: &SliceChunk,
    params: &InferParams,
    index: usize,
    chunk_count: usize,
    random_chunk_seed_base: Option<u64>,
    notifier: &dyn Notifier,
) -> Result<ChunkInferenceResult> {
    catch_chunk_panic(index, chunk_count, || {
        let sem = get_chunk_semaphore();
        sem.acquire();
        let _guard = SemaphoreGuard(sem);
        infer_chunk(
            model,
            chunk,
            params,
            index,
            chunk_count,
            random_chunk_seed_base,
            notifier,
        )
    })
}

/// Runs `f`, converting a panic into a typed [`Error`] tagged with the chunk
/// position. `AssertUnwindSafe` is sound here because each chunk's inference is
/// independent: on panic we discard that chunk's partial work and surface an
/// error rather than continuing to read possibly-broken state.
fn catch_chunk_panic<T>(
    index: usize,
    chunk_count: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_owned());
            Err(Error::message(format!(
                "chunk {}/{} panicked during inference: {detail}",
                index + 1,
                chunk_count
            )))
        }
    }
}

fn infer_chunk(
    model: &Model,
    chunk: &SliceChunk,
    params: &InferParams,
    index: usize,
    chunk_count: usize,
    random_chunk_seed_base: Option<u64>,
    notifier: &dyn Notifier,
) -> Result<ChunkInferenceResult> {
    let started_at = Instant::now();
    let chunk_seed = random_chunk_seed_base
        .map(|base_seed| derive_chunk_seed(base_seed, index))
        .unwrap_or(params.seed);
    let chunk_notifier = PrefixedNotifier {
        inner: notifier,
        chunk_index: index,
        chunk_count,
    };
    let result = if chunk_seed == params.seed {
        model.infer_with_notifier(&chunk.waveform, params, &chunk_notifier)?
    } else {
        let mut chunk_params = params.clone();
        chunk_params.seed = chunk_seed;
        model.infer_with_notifier(&chunk.waveform, &chunk_params, &chunk_notifier)?
    };
    let mut notes = result.notes;
    let offset_seconds = chunk.offset_seconds as f32;
    for note in &mut notes {
        note.offset_seconds += offset_seconds;
    }

    Ok(ChunkInferenceResult {
        index,
        notes,
        elapsed: started_at.elapsed(),
    })
}

fn derive_chunk_seed(base_seed: u64, index: usize) -> u64 {
    let mix = (index as u64)
        .wrapping_add(1)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut value = base_seed.wrapping_add(mix);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    value.max(1)
}

fn write_extract_output(
    path: &Path,
    format: ExtractFormat,
    notes: &[Note],
    midi_options: &MidiWriteOptions,
    text_options: &TextWriteOptions,
) -> Result<()> {
    match format {
        ExtractFormat::Midi => write_midi_file(path, notes, midi_options),
        ExtractFormat::Txt => write_text_file(path, notes, TextOutputFormat::Txt, text_options),
        ExtractFormat::Csv => write_text_file(path, notes, TextOutputFormat::Csv, text_options),
    }
}

fn ensure_output_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn load_model_for_extract(
    model_path: &Path,
    device: ExtractDevice,
    gpu: &GpuSelector,
    notifier: &dyn Notifier,
) -> Result<Model> {
    match device {
        ExtractDevice::Cpu => Model::load_with_notifier(model_path, Backend::Cpu, notifier),
        ExtractDevice::Gpu => load_gpu_model(model_path, gpu, notifier),
        ExtractDevice::Auto if gpu.has_any() => load_gpu_model(model_path, gpu, notifier),
        ExtractDevice::Auto => load_auto_model(model_path, notifier),
    }
}

#[cfg(feature = "gpu")]
fn load_gpu_model(model_path: &Path, gpu: &GpuSelector, notifier: &dyn Notifier) -> Result<Model> {
    let selector = gpu.to_core_selector();
    Model::load_with_gpu_selector_and_notifier(model_path, selector.as_ref(), notifier)
}

#[cfg(not(feature = "gpu"))]
fn load_gpu_model(
    _model_path: &Path,
    _gpu: &GpuSelector,
    _notifier: &dyn Notifier,
) -> Result<Model> {
    Err(Error::message(
        "GPU extraction requested but the `gpu` cargo feature is disabled",
    ))
}

#[cfg(feature = "gpu")]
fn load_auto_model(model_path: &Path, notifier: &dyn Notifier) -> Result<Model> {
    match Model::load_with_gpu_selector_and_notifier(model_path, None, notifier) {
        Ok(model) => Ok(model),
        Err(gpu_err) => match Model::load_with_notifier(model_path, Backend::Cpu, notifier) {
            Ok(model) => {
                emit_message(
                    notifier,
                    NotificationLevel::Warn,
                    format!("GPU backend unavailable ({gpu_err}); falling back to CPU"),
                );
                Ok(model)
            }
            Err(cpu_err) => Err(Error::message(format!(
                "failed to load model on GPU ({gpu_err}) and CPU fallback also failed ({cpu_err})"
            ))),
        },
    }
}

#[cfg(not(feature = "gpu"))]
fn load_auto_model(model_path: &Path, notifier: &dyn Notifier) -> Result<Model> {
    Model::load_with_notifier(model_path, Backend::Cpu, notifier)
}

#[cfg(feature = "gpu")]
fn gpu_adapter_info(model: &Model) -> Option<GpuAdapterInfo> {
    model.gpu_adapter_info().map(|adapter| GpuAdapterInfo {
        name: adapter.name,
        backend: format!("{:?}", adapter.backend).to_ascii_lowercase(),
        device_type: format!("{:?}", adapter.device_type).to_ascii_lowercase(),
        vendor_id: adapter.vendor,
        device_id: adapter.device,
    })
}

#[cfg(not(feature = "gpu"))]
fn gpu_adapter_info(_model: &Model) -> Option<GpuAdapterInfo> {
    None
}

fn extract_format_name(format: ExtractFormat) -> &'static str {
    match format {
        ExtractFormat::Midi => "midi",
        ExtractFormat::Txt => "txt",
        ExtractFormat::Csv => "csv",
    }
}

fn audio_prepare_detail(audio: &PreparedAudioInfo) -> String {
    if audio.was_resampled() || audio.was_downmixed() {
        format!(
            "{} Hz/{} ch -> {} Hz mono, samples={}, duration={:.2}s",
            audio.source_sample_rate,
            audio.source_channels,
            audio.sample_rate,
            audio.sample_count,
            audio.duration_seconds()
        )
    } else {
        format!(
            "{} Hz mono, samples={}, duration={:.2}s",
            audio.sample_rate,
            audio.sample_count,
            audio.duration_seconds()
        )
    }
}

fn emit_status(notifier: &dyn Notifier, stage: &'static str, message: impl Into<String>) {
    notifier.notify(CoreEvent::Status {
        stage,
        message: message.into(),
    });
}

fn emit_timing(
    notifier: &dyn Notifier,
    stage: &'static str,
    elapsed: Duration,
    detail: Option<String>,
) {
    notifier.notify(CoreEvent::Timing {
        stage,
        elapsed,
        detail,
    });
}

fn emit_message(notifier: &dyn Notifier, level: NotificationLevel, message: impl Into<String>) {
    notifier.notify(CoreEvent::Message {
        level,
        message: message.into(),
    });
}

fn timed_result<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, Duration)> {
    let started_at = Instant::now();
    let value = f()?;
    Ok((value, started_at.elapsed()))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        DEFAULT_MAX_CHUNK_SECONDS, ChunkSemaphore, ExtractDevice, ExtractFormat, ExtractRequest,
        GpuSelector, SemaphoreGuard, catch_chunk_panic, derive_chunk_seed, extract,
        infer_extract_format, resolve_extract_format,
    };

    #[test]
    fn infer_extract_format_from_output_extension() {
        assert_eq!(
            infer_extract_format(Path::new("notes.mid")),
            Some(ExtractFormat::Midi)
        );
        assert_eq!(
            infer_extract_format(Path::new("notes.txt")),
            Some(ExtractFormat::Txt)
        );
        assert_eq!(
            infer_extract_format(Path::new("notes.csv")),
            Some(ExtractFormat::Csv)
        );
        assert_eq!(infer_extract_format(Path::new("notes.unknown")), None);
    }

    #[test]
    fn resolve_extract_format_defaults_to_midi() {
        assert_eq!(
            resolve_extract_format(None, Path::new("notes.unknown")),
            ExtractFormat::Midi
        );
    }

    #[test]
    fn extract_rejects_zero_max_chunk_seconds_before_touching_files() {
        let request = ExtractRequest {
            model_path: "model.gguf".into(),
            input_path: "input.wav".into(),
            max_chunk_seconds: 0,
            ..ExtractRequest::default()
        };

        let err = extract(&request).unwrap_err();
        assert!(err.to_string().contains("max_chunk_seconds"));
    }

    #[test]
    fn extract_rejects_gpu_selector_with_cpu_device() {
        let request = ExtractRequest {
            model_path: "model.gguf".into(),
            input_path: "input.wav".into(),
            device: ExtractDevice::Cpu,
            gpu: GpuSelector {
                vendor_id: Some(0x10de),
                ..GpuSelector::default()
            },
            ..ExtractRequest::default()
        };

        let err = extract(&request).unwrap_err();
        assert!(err.to_string().contains("GPU selector"));
    }

    #[test]
    fn derive_chunk_seed_wraps_without_overflowing() {
        assert_ne!(derive_chunk_seed(0, 0), 0);
        assert_ne!(derive_chunk_seed(u64::MAX, usize::MAX), 0);
    }

    #[test]
    fn catch_chunk_panic_converts_panic_to_tagged_error() {
        let result = catch_chunk_panic::<()>(2, 5, || panic!("kernel exploded"));
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("chunk 3/5"), "missing chunk tag: {msg}");
        assert!(msg.contains("kernel exploded"), "missing payload: {msg}");
    }

    #[test]
    fn catch_chunk_panic_passes_ok_through() {
        let result = catch_chunk_panic(0, 1, || Ok(42usize));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn default_request_uses_expected_chunk_limit() {
        assert_eq!(
            ExtractRequest::default().max_chunk_seconds,
            DEFAULT_MAX_CHUNK_SECONDS
        );
    }

    #[test]
    fn semaphore_allows_concurrent_acquires_up_to_capacity() {
        let sem = ChunkSemaphore::new(2);
        sem.acquire();
        sem.acquire();
        // Both acquired, no hang; would deadlock if capacity was wrong.
        // Release in reverse order to clean up.
        drop(SemaphoreGuard(sem.clone()));
        drop(SemaphoreGuard(sem));
    }

    #[test]
    fn derive_chunk_seed_is_deterministic() {
        let seed1 = derive_chunk_seed(12345, 0);
        let seed2 = derive_chunk_seed(12345, 0);
        assert_eq!(seed1, seed2, "derive_chunk_seed must be deterministic");
    }

    #[test]
    fn derive_chunk_seed_varies_by_index() {
        let seed_0 = derive_chunk_seed(12345, 0);
        let seed_1 = derive_chunk_seed(12345, 1);
        let seed_2 = derive_chunk_seed(12345, 2);
        assert_ne!(seed_0, seed_1);
        assert_ne!(seed_1, seed_2);
        assert_ne!(seed_0, seed_2);
    }
}
