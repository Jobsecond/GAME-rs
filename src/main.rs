use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
#[cfg(feature = "gpu")]
use game_crabml::GpuAdapterSelector;
use game_crabml::{
    BackboneConfig, Backend, Error, InferParams, LoadedGgufModel, LoadedTensor, MelExtractor,
    MidiWriteOptions, Model, Result, SliceChunk, SlicerConfig, TextOutputFormat, TextWriteOptions,
    load_gguf, prepare_wav_for_inference, slice_waveform, split_long_chunks, write_midi_file,
    write_text_file,
};
use log::{Level, info};
use serde_json::{Map, Value, json};

#[derive(Debug, Parser)]
#[command(author, version, about = "Rust port of GAME GGUF inference")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Inspect(InspectArgs),
    Extract(ExtractArgs),
}

#[derive(Debug, Args)]
struct InspectArgs {
    #[arg(short = 'm', long = "model")]
    model: PathBuf,

    #[arg(long, default_value_t = 8)]
    show_tensors: usize,

    #[arg(long)]
    tensor_prefix: Option<String>,

    #[arg(long, value_enum, default_value_t = InspectFormat::Text)]
    format: InspectFormat,
}

#[derive(Debug, Args)]
struct ExtractArgs {
    #[arg(short = 'm', long = "model")]
    model: PathBuf,

    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    #[arg(long, value_enum)]
    format: Option<ExtractFormat>,

    #[arg(long, value_enum, help = "default: gpu if available, otherwise cpu")]
    device: Option<ExtractDevice>,

    #[arg(long, default_value_t = 0)]
    language: i32,

    #[arg(long = "d3pm-nsteps", default_value_t = 1)]
    d3pm_nsteps: i32,

    #[arg(long = "d3pm-t0", default_value_t = 0.0)]
    d3pm_t0: f32,

    #[arg(long = "boundary-threshold", default_value_t = 0.2)]
    boundary_threshold: f32,

    #[arg(long = "boundary-radius", default_value_t = 2)]
    boundary_radius: i32,

    #[arg(long = "note-threshold", default_value_t = 0.2)]
    note_threshold: f32,

    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(
        long = "max-chunk-seconds",
        default_value_t = DEFAULT_MAX_CHUNK_SECONDS,
        value_parser = parse_positive_usize,
        help = "hard-split sliced audio chunks longer than this many seconds"
    )]
    max_chunk_seconds: usize,

    #[command(flatten)]
    gpu: GpuSelectorArgs,

    input: PathBuf,
}

#[derive(Debug, Args, Default, Clone)]
struct GpuSelectorArgs {
    #[arg(long = "gpu-name")]
    gpu_name: Option<String>,

    #[arg(long = "gpu-vendor-id", value_parser = parse_u32_auto)]
    gpu_vendor_id: Option<u32>,

    #[arg(long = "gpu-device-id", value_parser = parse_u32_auto)]
    gpu_device_id: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum InspectFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ExtractFormat {
    Midi,
    Txt,
    Csv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ExtractDevice {
    Cpu,
    Gpu,
}

const DEFAULT_MAX_CHUNK_SECONDS: usize = 60;
static LOGGER_START: OnceLock<Instant> = OnceLock::new();

fn main() -> ExitCode {
    init_logging();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Inspect(args) => inspect(
            args.model,
            args.show_tensors,
            args.tensor_prefix,
            args.format,
        ),
        Command::Extract(args) => extract(args),
    }
}

fn extract(args: ExtractArgs) -> Result<()> {
    if matches!(args.device, Some(ExtractDevice::Cpu)) && args.gpu.has_any() {
        return Err(Error::message(
            "GPU selector flags cannot be used with `--device cpu`",
        ));
    }

    let mut progress = ExtractProgress::new(&args);
    let format = resolve_extract_format(args.format, &args.output);
    progress.log_start(format);

    progress.log_step_start("loading model");
    let (model, load_elapsed) =
        timed_result(|| load_model_for_extract(&args.model, args.device, &args.gpu))?;
    progress.record_model_loaded(&model, load_elapsed);

    progress.log_step_start("preparing audio");
    let (waveform, audio_elapsed) = timed_result(|| {
        prepare_wav_for_inference(&args.input, model.config().inference.audio_sample_rate)
    })?;
    progress.record_audio_prepared(&waveform, audio_elapsed);

    let slicer_config = SlicerConfig {
        sample_rate: waveform.sample_rate,
        ..SlicerConfig::default()
    };
    progress.log_step_start("slicing audio on silence");
    let (sliced_chunks, slice_elapsed) =
        timed_result(|| slice_waveform(&waveform.samples, &slicer_config))?;
    progress.record_slice_complete(&sliced_chunks, slice_elapsed);
    progress.log_step_start("splitting long chunks");
    let (chunks, split_elapsed) = timed_result(|| {
        split_long_chunks(
            &sliced_chunks,
            waveform.sample_rate,
            waveform.sample_rate.saturating_mul(args.max_chunk_seconds),
        )
    })?;
    progress.record_split_complete(&chunks, split_elapsed, &waveform);

    progress.log_step_start("initializing mel extractor");
    let (mel_extractor, mel_setup_elapsed) =
        timed_result(|| MelExtractor::from_inference_config(&model.config().inference))?;
    let total_frames = mel_extractor.num_frames(waveform.samples.len());
    progress.record_mel_setup(total_frames, mel_setup_elapsed);

    let params = InferParams {
        language: args.language,
        d3pm_t0: args.d3pm_t0,
        d3pm_nsteps: args.d3pm_nsteps,
        boundary_threshold: args.boundary_threshold,
        boundary_radius: args.boundary_radius,
        note_threshold: args.note_threshold,
        seed: args.seed,
        ..InferParams::default()
    };
    progress.log_inference_start(chunks.len());
    let (result, inference_elapsed) = timed_result(|| {
        run_chunked_extract_with_progress(&model, &chunks, &params, &mut progress)
    })?;
    progress.record_inference_complete(&result, inference_elapsed);

    progress.log_step_start("writing output");
    let (_, write_elapsed) = timed_result(|| {
        ensure_output_parent_dir(&args.output)?;
        write_extract_output(&args.output, format, &result.notes)
    })?;
    progress.record_output_written(&args.output, write_elapsed);
    progress.print_summary(&waveform, &model, total_frames, &result);
    Ok(())
}

struct ChunkedExtractResult {
    notes: Vec<game_crabml::Note>,
    chunk_count: usize,
}

#[cfg(test)]
fn run_chunked_extract(
    model: &Model,
    chunks: &[SliceChunk],
    params: &InferParams,
) -> Result<ChunkedExtractResult> {
    run_chunked_extract_with_progress(model, chunks, params, &mut NoopExtractProgress)
}

fn run_chunked_extract_with_progress(
    model: &Model,
    chunks: &[SliceChunk],
    params: &InferParams,
    progress: &mut impl ExtractProgressLogger,
) -> Result<ChunkedExtractResult> {
    let mut notes = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let chunk_start = Instant::now();
        let chunk_duration_seconds =
            chunk.waveform.len() as f64 / model.config().inference.audio_sample_rate as f64;
        progress.log_chunk_start(
            index,
            chunks.len(),
            chunk.offset_seconds,
            chunk_duration_seconds,
        );
        let result = model.infer(&chunk.waveform, params)?;
        let chunk_notes = result.notes.len();
        for mut note in result.notes {
            note.offset_seconds += chunk.offset_seconds as f32;
            notes.push(note);
        }
        progress.log_chunk_complete(index, chunks.len(), chunk_notes, chunk_start.elapsed());
    }

    Ok(ChunkedExtractResult {
        notes,
        chunk_count: chunks.len(),
    })
}

fn write_extract_output(
    path: &Path,
    format: ExtractFormat,
    notes: &[game_crabml::Note],
) -> Result<()> {
    match format {
        ExtractFormat::Midi => write_midi_file(path, notes, &MidiWriteOptions::default()),
        ExtractFormat::Txt => write_text_file(
            path,
            notes,
            TextOutputFormat::Txt,
            &TextWriteOptions::default(),
        ),
        ExtractFormat::Csv => write_text_file(
            path,
            notes,
            TextOutputFormat::Csv,
            &TextWriteOptions::default(),
        ),
    }
}

fn resolve_extract_format(format: Option<ExtractFormat>, output: &Path) -> ExtractFormat {
    format
        .or_else(|| infer_extract_format(output))
        .unwrap_or(ExtractFormat::Midi)
}

fn infer_extract_format(output: &Path) -> Option<ExtractFormat> {
    let extension = output.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "mid" | "midi" => Some(ExtractFormat::Midi),
        "txt" => Some(ExtractFormat::Txt),
        "csv" => Some(ExtractFormat::Csv),
        _ => None,
    }
}

fn load_model_for_extract(
    model_path: &Path,
    device: Option<ExtractDevice>,
    gpu: &GpuSelectorArgs,
) -> Result<Model> {
    match device {
        Some(ExtractDevice::Cpu) => Model::load(model_path, Backend::Cpu),
        Some(ExtractDevice::Gpu) => load_gpu_model(model_path, gpu),
        None if gpu.has_any() => load_gpu_model(model_path, gpu),
        None => load_auto_model(model_path),
    }
}

#[cfg(feature = "gpu")]
fn load_gpu_model(model_path: &Path, gpu: &GpuSelectorArgs) -> Result<Model> {
    let selector = gpu.to_selector();
    Model::load_with_gpu_selector(model_path, selector.as_ref())
}

#[cfg(not(feature = "gpu"))]
fn load_gpu_model(_model_path: &Path, _gpu: &GpuSelectorArgs) -> Result<Model> {
    Err(Error::message(
        "GPU extraction requested but the `gpu` cargo feature is disabled",
    ))
}

#[cfg(feature = "gpu")]
fn load_auto_model(model_path: &Path) -> Result<Model> {
    match Model::load_with_gpu_selector(model_path, None) {
        Ok(model) => Ok(model),
        Err(gpu_err) => match Model::load(model_path, Backend::Cpu) {
            Ok(model) => {
                warn!("GPU backend unavailable ({gpu_err}); falling back to CPU");
                Ok(model)
            }
            Err(cpu_err) => Err(Error::message(format!(
                "failed to load model on GPU ({gpu_err}) and CPU fallback also failed ({cpu_err})"
            ))),
        },
    }
}

#[cfg(not(feature = "gpu"))]
fn load_auto_model(model_path: &Path) -> Result<Model> {
    Model::load(model_path, Backend::Cpu)
}

fn ensure_output_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        Backend::Gpu => "gpu",
    }
}

#[cfg(feature = "gpu")]
fn wgpu_backend_name(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Empty => "empty",
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gl",
        wgpu::Backend::BrowserWebGpu => "webgpu",
    }
}

#[cfg(feature = "gpu")]
fn wgpu_device_type_name(device_type: wgpu::DeviceType) -> &'static str {
    match device_type {
        wgpu::DeviceType::Other => "other",
        wgpu::DeviceType::IntegratedGpu => "integrated",
        wgpu::DeviceType::DiscreteGpu => "discrete",
        wgpu::DeviceType::VirtualGpu => "virtual",
        wgpu::DeviceType::Cpu => "cpu",
    }
}

fn init_logging() {
    LOGGER_START.get_or_init(Instant::now);
    let env = env_logger::Env::default().default_filter_or("info");
    let mut builder = env_logger::Builder::from_env(env);
    builder.format_timestamp(None);
    builder.format_module_path(false);
    builder.format_target(false);
    builder.format(|buf, record| {
        let elapsed = LOGGER_START
            .get()
            .map(|start| format_duration(start.elapsed()))
            .unwrap_or_else(|| "0.000s".to_owned());
        writeln!(
            buf,
            "[{} +{}] {}",
            log_level_name(record.level()),
            elapsed,
            record.args()
        )
    });
    let _ = builder.try_init();
}

fn timed_result<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, Duration)> {
    let start = Instant::now();
    let value = f()?;
    Ok((value, start.elapsed()))
}

trait ExtractProgressLogger {
    fn log_chunk_start(
        &mut self,
        chunk_index: usize,
        chunk_count: usize,
        offset_seconds: f64,
        duration_seconds: f64,
    );

    fn log_chunk_complete(
        &mut self,
        chunk_index: usize,
        chunk_count: usize,
        chunk_notes: usize,
        elapsed: Duration,
    );
}

#[cfg(test)]
struct NoopExtractProgress;

#[cfg(test)]
impl ExtractProgressLogger for NoopExtractProgress {
    fn log_chunk_start(
        &mut self,
        _chunk_index: usize,
        _chunk_count: usize,
        _offset_seconds: f64,
        _duration_seconds: f64,
    ) {
    }

    fn log_chunk_complete(
        &mut self,
        _chunk_index: usize,
        _chunk_count: usize,
        _chunk_notes: usize,
        _elapsed: Duration,
    ) {
    }
}

struct ExtractProgress<'a> {
    args: &'a ExtractArgs,
    started_at: Instant,
    model_load_elapsed: Duration,
    audio_prepare_elapsed: Duration,
    slice_elapsed: Duration,
    split_elapsed: Duration,
    mel_setup_elapsed: Duration,
    inference_elapsed: Duration,
    write_elapsed: Duration,
    original_chunk_count: usize,
}

impl<'a> ExtractProgress<'a> {
    fn new(args: &'a ExtractArgs) -> Self {
        Self {
            args,
            started_at: Instant::now(),
            model_load_elapsed: Duration::ZERO,
            audio_prepare_elapsed: Duration::ZERO,
            slice_elapsed: Duration::ZERO,
            split_elapsed: Duration::ZERO,
            mel_setup_elapsed: Duration::ZERO,
            inference_elapsed: Duration::ZERO,
            write_elapsed: Duration::ZERO,
            original_chunk_count: 0,
        }
    }

    fn log_start(&self, format: ExtractFormat) {
        info!(
            "starting extract: input={} model={} output={} format={} requested_device={}",
            self.args.input.display(),
            self.args.model.display(),
            self.args.output.display(),
            extract_format_name(format),
            self.args.device.map(extract_device_name).unwrap_or("auto")
        );
    }

    fn log_step_start(&self, step_name: &str) {
        info!("{step_name}...");
    }

    fn record_model_loaded(&mut self, model: &Model, elapsed: Duration) {
        self.model_load_elapsed = elapsed;
        info!(
            "model loaded: backend={} elapsed={}",
            backend_name(model.backend()),
            format_duration(elapsed)
        );
        #[cfg(feature = "gpu")]
        if let Some(adapter) = model.gpu_adapter_info() {
            info!(
                "gpu adapter: {} (backend={}, type={}, vendor=0x{:04x}, device=0x{:04x})",
                adapter.name,
                wgpu_backend_name(adapter.backend),
                wgpu_device_type_name(adapter.device_type),
                adapter.vendor,
                adapter.device
            );
        }
    }

    fn record_audio_prepared(
        &mut self,
        waveform: &game_crabml::PreparedWaveform,
        elapsed: Duration,
    ) {
        self.audio_prepare_elapsed = elapsed;
        let input_seconds = if waveform.source_sample_rate == 0 {
            0.0
        } else {
            waveform.samples.len() as f64 / waveform.sample_rate as f64
        };
        if waveform.was_resampled() || waveform.was_downmixed() {
            info!(
                "audio prepared: {} Hz/{} ch -> {} Hz mono, samples={}, duration={:.2}s, elapsed={}",
                waveform.source_sample_rate,
                waveform.source_channels,
                waveform.sample_rate,
                waveform.samples.len(),
                input_seconds,
                format_duration(elapsed)
            );
        } else {
            info!(
                "audio prepared: {} Hz mono, samples={}, duration={:.2}s, elapsed={}",
                waveform.sample_rate,
                waveform.samples.len(),
                input_seconds,
                format_duration(elapsed)
            );
        }
    }

    fn record_slice_complete(&mut self, sliced_chunks: &[SliceChunk], slice_elapsed: Duration) {
        self.slice_elapsed = slice_elapsed;
        self.original_chunk_count = sliced_chunks.len();
        info!(
            "silence slicing complete: chunks={} elapsed={}",
            sliced_chunks.len(),
            format_duration(slice_elapsed)
        );
    }

    fn record_split_complete(
        &mut self,
        chunks: &[SliceChunk],
        split_elapsed: Duration,
        waveform: &game_crabml::PreparedWaveform,
    ) {
        self.split_elapsed = split_elapsed;
        let max_samples = waveform
            .sample_rate
            .saturating_mul(self.args.max_chunk_seconds);
        info!(
            "long-chunk split complete: chunks={} max_chunk_seconds={} max_chunk_samples={} elapsed={}",
            chunks.len(),
            self.args.max_chunk_seconds,
            max_samples,
            format_duration(split_elapsed)
        );
    }

    fn record_mel_setup(&mut self, total_frames: usize, elapsed: Duration) {
        self.mel_setup_elapsed = elapsed;
        info!(
            "mel extractor ready: frames={} elapsed={}",
            total_frames,
            format_duration(elapsed)
        );
    }

    fn log_inference_start(&self, chunk_count: usize) {
        info!("running inference across {chunk_count} chunk(s)");
    }

    fn record_inference_complete(&mut self, result: &ChunkedExtractResult, elapsed: Duration) {
        self.inference_elapsed = elapsed;
        info!(
            "inference complete: chunks={} notes={} elapsed={}",
            result.chunk_count,
            result.notes.len(),
            format_duration(elapsed)
        );
    }

    fn record_output_written(&mut self, output: &Path, elapsed: Duration) {
        self.write_elapsed = elapsed;
        info!(
            "output written: path={} elapsed={}",
            output.display(),
            format_duration(elapsed)
        );
    }

    fn print_summary(
        &self,
        waveform: &game_crabml::PreparedWaveform,
        model: &Model,
        total_frames: usize,
        result: &ChunkedExtractResult,
    ) {
        if waveform.was_resampled() || waveform.was_downmixed() {
            eprintln!(
                "audio: {} Hz/{} ch -> {} Hz mono",
                waveform.source_sample_rate, waveform.source_channels, waveform.sample_rate
            );
        } else {
            eprintln!("audio: {} Hz mono", waveform.sample_rate);
        }
        eprintln!("backend: {}", backend_name(model.backend()));
        #[cfg(feature = "gpu")]
        if let Some(adapter) = model.gpu_adapter_info() {
            eprintln!(
                "gpu: {} (backend={}, type={}, vendor=0x{:04x}, device=0x{:04x})",
                adapter.name,
                wgpu_backend_name(adapter.backend),
                wgpu_device_type_name(adapter.device_type),
                adapter.vendor,
                adapter.device
            );
        }
        eprintln!("max_chunk_seconds: {}", self.args.max_chunk_seconds);
        eprintln!("chunks: {}", result.chunk_count);
        eprintln!("frames: {}", total_frames);
        eprintln!("notes: {}", result.notes.len());
        eprintln!("wrote {}", self.args.output.display());
        eprintln!(
            "elapsed_total: {}",
            format_duration(self.started_at.elapsed())
        );
        eprintln!(
            "elapsed_model_load: {}",
            format_duration(self.model_load_elapsed)
        );
        eprintln!(
            "elapsed_audio_prepare: {}",
            format_duration(self.audio_prepare_elapsed)
        );
        eprintln!("elapsed_slice: {}", format_duration(self.slice_elapsed));
        eprintln!(
            "elapsed_long_chunk_split: {}",
            format_duration(self.split_elapsed)
        );
        eprintln!(
            "elapsed_mel_setup: {}",
            format_duration(self.mel_setup_elapsed)
        );
        eprintln!(
            "elapsed_inference: {}",
            format_duration(self.inference_elapsed)
        );
        eprintln!(
            "elapsed_output_write: {}",
            format_duration(self.write_elapsed)
        );
        if self.original_chunk_count != 0 && self.original_chunk_count != result.chunk_count {
            eprintln!("chunks_before_long_split: {}", self.original_chunk_count);
        }
    }
}

impl ExtractProgressLogger for ExtractProgress<'_> {
    fn log_chunk_start(
        &mut self,
        chunk_index: usize,
        chunk_count: usize,
        offset_seconds: f64,
        duration_seconds: f64,
    ) {
        info!(
            "chunk {}/{}: infer start offset={:.2}s duration={:.2}s",
            chunk_index + 1,
            chunk_count,
            offset_seconds,
            duration_seconds
        );
    }

    fn log_chunk_complete(
        &mut self,
        chunk_index: usize,
        chunk_count: usize,
        chunk_notes: usize,
        elapsed: Duration,
    ) {
        info!(
            "chunk {}/{}: infer done notes={} elapsed={}",
            chunk_index + 1,
            chunk_count,
            chunk_notes,
            format_duration(elapsed)
        );
    }
}

fn format_duration(duration: Duration) -> String {
    format!("{:.3}s", duration.as_secs_f64())
}

fn log_level_name(level: Level) -> &'static str {
    match level {
        Level::Error => "error",
        Level::Warn => "warn",
        Level::Info => "info",
        Level::Debug => "debug",
        Level::Trace => "trace",
    }
}

fn extract_device_name(device: ExtractDevice) -> &'static str {
    match device {
        ExtractDevice::Cpu => "cpu",
        ExtractDevice::Gpu => "gpu",
    }
}

fn extract_format_name(format: ExtractFormat) -> &'static str {
    match format {
        ExtractFormat::Midi => "midi",
        ExtractFormat::Txt => "txt",
        ExtractFormat::Csv => "csv",
    }
}

fn inspect(
    model_path: PathBuf,
    show_tensors: usize,
    tensor_prefix: Option<String>,
    format: InspectFormat,
) -> Result<()> {
    let model = load_gguf(&model_path)?;
    let file_size_bytes = std::fs::metadata(&model_path)?.len();
    let total_parameters = model.total_parameters() as u64;
    let total_loaded_bytes = model.total_loaded_bytes() as u64;
    let filtered_tensors = filtered_tensors(&model, tensor_prefix.as_deref());
    let filtered_stats = tensor_stats(&filtered_tensors);

    if format == InspectFormat::Json {
        let summary = inspect_json_summary(
            &model,
            file_size_bytes,
            total_parameters,
            total_loaded_bytes,
            show_tensors,
            tensor_prefix.as_deref(),
            &filtered_tensors,
            filtered_stats,
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&summary).map_err(|err| Error::message(format!(
                "failed to serialize inspect JSON: {err}"
            )))?
        );
        return Ok(());
    }

    println!("model: {}", model.path.display());
    println!(
        "file_size: {} bytes ({})",
        format_count(file_size_bytes),
        format_bytes(file_size_bytes)
    );
    println!();

    println!("gguf:");
    println!("  version: {}", model.gguf_version);
    println!("  architecture: {}", model.config.architecture);
    println!(
        "  quantization_version: {}",
        model
            .quantization_version
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_owned())
    );
    println!(
        "  metadata_keys: {}",
        format_count(model.metadata_count as u64)
    );
    println!(
        "  tensor_count: {}",
        format_count(model.tensor_count() as u64)
    );
    println!("  parameter_count: {}", format_count(total_parameters));
    println!(
        "  loaded_weights: {} bytes ({})",
        format_count(total_loaded_bytes),
        format_bytes(total_loaded_bytes)
    );
    println!();

    if let Some(prefix) = tensor_prefix.as_deref() {
        println!("tensor_filter:");
        println!("  prefix: {prefix}");
        println!(
            "  matched_tensors: {}",
            format_count(filtered_stats.tensor_count as u64)
        );
        println!(
            "  matched_parameters: {}",
            format_count(filtered_stats.parameter_count)
        );
        println!(
            "  matched_bytes: {}",
            format_bytes(filtered_stats.byte_count)
        );
        println!();
    }

    println!("model_config:");
    println!("  name: {}", display_or_dash(&model.config.name));
    println!("  version: {}", display_or_dash(&model.config.version));
    println!("  mode: {}", model.config.mode);
    println!("  embedding_dim: {}", model.config.embedding_dim);
    println!("  input_dim: {}", model.config.in_dim);
    println!("  estimator_out_dim: {}", model.config.estimator_out_dim);
    println!("  region_cycle_len: {}", model.config.region_cycle_len);
    println!("  use_languages: {}", model.config.use_languages);
    println!("  num_languages: {}", model.config.num_languages);
    println!();

    let inference = &model.config.inference;
    println!("inference:");
    println!("  sample_rate: {}", inference.audio_sample_rate);
    println!("  hop_size: {}", inference.hop_size);
    println!("  timestep_seconds: {:.6}", inference.timestep());
    println!("  fft_size: {}", inference.fft_size);
    println!("  win_size: {}", inference.win_size);
    println!(
        "  spectrogram: type={} bins={} fmin={} fmax={}",
        inference.spectrogram_type, inference.n_mels, inference.fmin, inference.fmax
    );
    println!(
        "  midi: min={} max={} bins={} std={}",
        inference.midi_min, inference.midi_max, inference.midi_num_bins, inference.midi_std
    );
    println!(
        "  lang_map: {}",
        if inference.lang_map.is_empty() {
            "none".to_owned()
        } else {
            inference
                .lang_map
                .iter()
                .map(|(lang, id)| format!("{lang}={id}"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!();

    println!("backbones:");
    print_backbone("encoder", &model.config.encoder);
    print_backbone("segmenter", &model.config.segmenter);
    print_backbone("estimator", &model.config.estimator);
    println!();

    println!("tensor_types:");
    for (tensor_type, count) in tensor_type_counts(&model) {
        println!("  {tensor_type}: {}", format_count(count as u64));
    }
    println!();

    println!("tensor_prefixes:");
    for (prefix, stats) in tensor_prefix_stats(&model) {
        println!(
            "  {prefix}: tensors={} params={} bytes={}",
            format_count(stats.tensor_count as u64),
            format_count(stats.parameter_count),
            format_bytes(stats.byte_count)
        );
    }

    if show_tensors > 0 {
        println!();
        println!(
            "{}:",
            if tensor_prefix.is_some() {
                "largest_matching_tensors"
            } else {
                "largest_tensors"
            }
        );
        for (index, (name, tensor)) in largest_tensors(&filtered_tensors, show_tensors)
            .into_iter()
            .enumerate()
        {
            print_tensor_summary(index + 1, name, tensor);
        }

        println!();
        println!(
            "{}:",
            if tensor_prefix.is_some() {
                "sample_matching_tensors"
            } else {
                "sample_tensors"
            }
        );
        for (name, tensor) in filtered_tensors.iter().take(show_tensors) {
            print_tensor_summary(0, name, tensor);
        }
    }

    Ok(())
}

#[derive(Debug, Default, Clone, Copy)]
struct PrefixStats {
    tensor_count: usize,
    parameter_count: u64,
    byte_count: u64,
}

fn tensor_type_counts(model: &LoadedGgufModel) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for tensor in model.tensors.values() {
        *counts.entry(tensor.tensor_type.to_string()).or_default() += 1;
    }
    counts
}

fn tensor_prefix_stats(model: &LoadedGgufModel) -> BTreeMap<String, PrefixStats> {
    let mut stats: BTreeMap<String, PrefixStats> = BTreeMap::new();
    for (name, tensor) in &model.tensors {
        let prefix = name.split('.').next().unwrap_or("unknown").to_owned();
        let entry = stats.entry(prefix).or_default();
        entry.tensor_count += 1;
        entry.parameter_count += tensor.num_elements() as u64;
        entry.byte_count += tensor.byte_len() as u64;
    }
    stats
}

fn largest_tensors<'a>(
    tensors: &[(&'a str, &'a LoadedTensor)],
    limit: usize,
) -> Vec<(&'a str, &'a LoadedTensor)> {
    let mut tensors = tensors.to_vec();
    tensors.sort_by_key(|(name, tensor)| (Reverse(tensor.num_elements()), *name));
    tensors.truncate(limit);
    tensors
}

fn print_backbone(name: &str, backbone: &BackboneConfig) {
    println!(
        "  {name}: cls={} dim={} layers={} heads={} head_dim={} ffn_type={}",
        display_or_dash(&backbone.cls),
        backbone.dim,
        backbone.num_layers,
        backbone.num_heads,
        backbone.head_dim,
        backbone.ffn_type
    );
    println!(
        "    conv: c_kernel={} m_kernel={} use_ls={} use_out_norm={} skip_first_ffn={} skip_out_ffn={}",
        backbone.c_kernel_size,
        backbone.m_kernel_size,
        backbone.use_ls,
        backbone.use_out_norm,
        backbone.skip_first_ffn,
        backbone.skip_out_ffn
    );

    if backbone.return_latent {
        println!(
            "    latent: enabled=true layer_idx={} out_dim={}",
            backbone.latent_layer_idx, backbone.latent_out_dim
        );
    }

    if backbone.region_token_num != 1
        || backbone.c_kernel_size_pool != 0
        || backbone.m_kernel_size_pool != 0
        || backbone.c_kernel_size_x != 0
        || backbone.m_kernel_size_x != 0
    {
        println!(
            "    joint: region_tokens={} merge={} attn_type={} rope_mode={} qk_norm={} region_bias={} use_rope={} use_pool_offset={} theta={}",
            backbone.region_token_num,
            backbone.pool_merge_mode,
            backbone.attn_type,
            backbone.rope_mode,
            backbone.qk_norm,
            backbone.use_region_bias,
            backbone.use_rope,
            backbone.use_pool_offset,
            backbone.theta
        );
        println!(
            "    joint_conv: pool(c={}, m={}) x(c={}, m={})",
            backbone.c_kernel_size_pool,
            backbone.m_kernel_size_pool,
            backbone.c_kernel_size_x,
            backbone.m_kernel_size_x
        );
    }
}

fn print_tensor_summary(index: usize, name: &str, tensor: &LoadedTensor) {
    let prefix = if index == 0 {
        "  ".to_owned()
    } else {
        format!("  {index}.")
    };
    println!(
        "{prefix} {name}: shape={:?} type={} numel={} bytes={}",
        tensor.shape,
        tensor.tensor_type,
        format_count(tensor.num_elements() as u64),
        format_bytes(tensor.byte_len() as u64)
    );
}

fn inspect_json_summary(
    model: &LoadedGgufModel,
    file_size_bytes: u64,
    total_parameters: u64,
    total_loaded_bytes: u64,
    show_tensors: usize,
    tensor_prefix: Option<&str>,
    filtered_tensors: &[(&str, &LoadedTensor)],
    filtered_stats: PrefixStats,
) -> Value {
    let tensor_types = tensor_type_counts(model)
        .into_iter()
        .map(|(typ, count)| (typ, json!(count)))
        .collect::<Map<String, Value>>();

    let tensor_prefixes = tensor_prefix_stats(model)
        .into_iter()
        .map(|(prefix, stats)| {
            (
                prefix,
                json!({
                    "tensor_count": stats.tensor_count,
                    "parameter_count": stats.parameter_count,
                    "byte_count": stats.byte_count,
                    "byte_count_human": format_bytes(stats.byte_count),
                }),
            )
        })
        .collect::<Map<String, Value>>();

    let largest_tensors = largest_tensors(filtered_tensors, show_tensors)
        .into_iter()
        .map(|(name, tensor)| tensor_json_summary(name, tensor))
        .collect::<Vec<_>>();

    let sample_tensors = filtered_tensors
        .iter()
        .take(show_tensors)
        .map(|(name, tensor)| tensor_json_summary(name, tensor))
        .collect::<Vec<_>>();

    json!({
        "model_path": model.path,
        "file_size_bytes": file_size_bytes,
        "file_size_human": format_bytes(file_size_bytes),
        "gguf": {
            "version": model.gguf_version.to_string(),
            "architecture": model.config.architecture,
            "quantization_version": model.quantization_version,
            "metadata_keys": model.metadata_count,
            "tensor_count": model.tensor_count(),
            "parameter_count": total_parameters,
            "loaded_weights_bytes": total_loaded_bytes,
            "loaded_weights_human": format_bytes(total_loaded_bytes),
        },
        "tensor_filter": {
            "prefix": tensor_prefix,
            "matched_tensor_count": filtered_stats.tensor_count,
            "matched_parameter_count": filtered_stats.parameter_count,
            "matched_byte_count": filtered_stats.byte_count,
            "matched_byte_count_human": format_bytes(filtered_stats.byte_count),
        },
        "model_config": {
            "name": optional_string_json(&model.config.name),
            "version": optional_string_json(&model.config.version),
            "mode": model.config.mode,
            "embedding_dim": model.config.embedding_dim,
            "input_dim": model.config.in_dim,
            "estimator_out_dim": model.config.estimator_out_dim,
            "region_cycle_len": model.config.region_cycle_len,
            "use_languages": model.config.use_languages,
            "num_languages": model.config.num_languages,
        },
        "inference": {
            "audio_sample_rate": model.config.inference.audio_sample_rate,
            "hop_size": model.config.inference.hop_size,
            "timestep_seconds": model.config.inference.timestep(),
            "fft_size": model.config.inference.fft_size,
            "win_size": model.config.inference.win_size,
            "spectrogram_type": model.config.inference.spectrogram_type,
            "n_mels": model.config.inference.n_mels,
            "fmin": model.config.inference.fmin,
            "fmax": model.config.inference.fmax,
            "midi_min": model.config.inference.midi_min,
            "midi_max": model.config.inference.midi_max,
            "midi_num_bins": model.config.inference.midi_num_bins,
            "midi_std": model.config.inference.midi_std,
            "lang_map": model.config.inference.lang_map,
        },
        "backbones": {
            "encoder": backbone_json_summary(&model.config.encoder),
            "segmenter": backbone_json_summary(&model.config.segmenter),
            "estimator": backbone_json_summary(&model.config.estimator),
        },
        "tensor_types": tensor_types,
        "tensor_prefixes": tensor_prefixes,
        "largest_tensors": largest_tensors,
        "sample_tensors": sample_tensors,
    })
}

fn backbone_json_summary(backbone: &BackboneConfig) -> Value {
    json!({
        "cls": optional_string_json(&backbone.cls),
        "dim": backbone.dim,
        "num_layers": backbone.num_layers,
        "num_heads": backbone.num_heads,
        "head_dim": backbone.head_dim,
        "c_kernel_size": backbone.c_kernel_size,
        "m_kernel_size": backbone.m_kernel_size,
        "ffn_type": backbone.ffn_type,
        "use_ls": backbone.use_ls,
        "use_out_norm": backbone.use_out_norm,
        "skip_first_ffn": backbone.skip_first_ffn,
        "skip_out_ffn": backbone.skip_out_ffn,
        "return_latent": backbone.return_latent,
        "latent_layer_idx": backbone.latent_layer_idx,
        "latent_out_dim": backbone.latent_out_dim,
        "region_token_num": backbone.region_token_num,
        "pool_merge_mode": backbone.pool_merge_mode,
        "attn_type": backbone.attn_type,
        "rope_mode": backbone.rope_mode,
        "qk_norm": backbone.qk_norm,
        "use_region_bias": backbone.use_region_bias,
        "c_kernel_size_pool": backbone.c_kernel_size_pool,
        "m_kernel_size_pool": backbone.m_kernel_size_pool,
        "c_kernel_size_x": backbone.c_kernel_size_x,
        "m_kernel_size_x": backbone.m_kernel_size_x,
        "use_rope": backbone.use_rope,
        "use_pool_offset": backbone.use_pool_offset,
        "theta": backbone.theta,
    })
}

fn tensor_json_summary(name: &str, tensor: &LoadedTensor) -> Value {
    json!({
        "name": name,
        "shape": tensor.shape,
        "tensor_type": tensor.tensor_type.to_string(),
        "num_elements": tensor.num_elements(),
        "byte_count": tensor.byte_len(),
        "byte_count_human": format_bytes(tensor.byte_len() as u64),
    })
}

fn filtered_tensors<'a>(
    model: &'a LoadedGgufModel,
    tensor_prefix: Option<&str>,
) -> Vec<(&'a str, &'a LoadedTensor)> {
    model
        .tensors
        .iter()
        .filter(|(name, _)| tensor_prefix.is_none_or(|prefix| matches_tensor_prefix(name, prefix)))
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect()
}

fn tensor_stats(tensors: &[(&str, &LoadedTensor)]) -> PrefixStats {
    let mut stats = PrefixStats::default();
    for (_, tensor) in tensors {
        stats.tensor_count += 1;
        stats.parameter_count += tensor.num_elements() as u64;
        stats.byte_count += tensor.byte_len() as u64;
    }
    stats
}

fn matches_tensor_prefix(name: &str, prefix: &str) -> bool {
    name == prefix
        || name
            .strip_prefix(prefix)
            .is_some_and(|suffix| suffix.starts_with('.'))
}

fn optional_string_json(value: &str) -> Value {
    if value.is_empty() {
        Value::Null
    } else {
        Value::String(value.to_owned())
    }
}

fn format_count(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

fn display_or_dash(value: &str) -> &str {
    if value.is_empty() { "-" } else { value }
}

fn parse_u32_auto(value: &str) -> std::result::Result<u32, String> {
    let value = value.trim();
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u32::from_str_radix(hex, 16).map_err(|err| format!("invalid hex value `{value}`: {err}"))
    } else {
        value
            .parse::<u32>()
            .map_err(|err| format!("invalid integer value `{value}`: {err}"))
    }
}

fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    let value = value.trim();
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("invalid integer value `{value}`: {err}"))?;
    if parsed == 0 {
        return Err("value must be greater than 0".to_owned());
    }
    Ok(parsed)
}

impl GpuSelectorArgs {
    fn has_any(&self) -> bool {
        self.gpu_name.is_some() || self.gpu_vendor_id.is_some() || self.gpu_device_id.is_some()
    }

    #[cfg(feature = "gpu")]
    fn to_selector(&self) -> Option<GpuAdapterSelector> {
        self.has_any().then(|| GpuAdapterSelector {
            name_substring: self.gpu_name.clone(),
            vendor_id: self.gpu_vendor_id,
            device_id: self.gpu_device_id,
            backend: None,
            device_type: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use clap::Parser;
    use game_crabml::{
        Backend, Error, InferParams, Model, Note, SlicerConfig, prepare_wav_for_inference,
        slice_waveform, split_long_chunks,
    };

    use super::{
        Cli, Command, DEFAULT_MAX_CHUNK_SECONDS, ExtractFormat, infer_extract_format,
        parse_u32_auto, run_chunked_extract,
    };

    #[test]
    fn parse_u32_auto_accepts_decimal_and_hex() {
        assert_eq!(parse_u32_auto("1234").unwrap(), 1234);
        assert_eq!(parse_u32_auto("0x10de").unwrap(), 0x10de);
        assert_eq!(parse_u32_auto("0X2484").unwrap(), 0x2484);
    }

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
    fn extract_cli_uses_default_max_chunk_seconds() {
        let cli = Cli::try_parse_from([
            "game-crabml",
            "extract",
            "-m",
            "model.gguf",
            "-o",
            "notes.txt",
            "input.wav",
        ])
        .unwrap();

        let Command::Extract(args) = cli.command else {
            panic!("expected extract command");
        };
        assert_eq!(args.max_chunk_seconds, DEFAULT_MAX_CHUNK_SECONDS);
    }

    #[test]
    fn extract_cli_accepts_custom_max_chunk_seconds() {
        let cli = Cli::try_parse_from([
            "game-crabml",
            "extract",
            "-m",
            "model.gguf",
            "-o",
            "notes.txt",
            "--max-chunk-seconds",
            "5",
            "input.wav",
        ])
        .unwrap();

        let Command::Extract(args) = cli.command else {
            panic!("expected extract command");
        };
        assert_eq!(args.max_chunk_seconds, 5);
    }

    #[test]
    fn extract_cli_rejects_zero_max_chunk_seconds() {
        let err = Cli::try_parse_from([
            "game-crabml",
            "extract",
            "-m",
            "model.gguf",
            "-o",
            "notes.txt",
            "--max-chunk-seconds",
            "0",
            "input.wav",
        ])
        .unwrap_err();

        assert!(
            err.to_string().contains("--max-chunk-seconds"),
            "unexpected clap error: {err}"
        );
    }

    #[test]
    #[ignore = "real-model CPU regression with local assets; run with `cargo test --release -- --ignored --nocapture`"]
    fn vocal2_cpu_extract_matches_expected_output_fixture() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let model_path = root.join("assets").join("models").join("large.gguf");
        let audio_path = root
            .join("assets")
            .join("audio")
            .join("vocal2-44100-16bit-1ch.wav");
        let expected_path = root
            .join("assets")
            .join("expected_output")
            .join("vocal2-44100-16bit-1ch.txt");

        if let Some(missing) = [&model_path, &audio_path, &expected_path]
            .into_iter()
            .find(|path| !path.exists())
        {
            eprintln!(
                "skipping vocal2 real-model regression: missing {}",
                missing.display()
            );
            return;
        }

        let model = Model::load(&model_path, Backend::Cpu).unwrap();
        let waveform =
            prepare_wav_for_inference(&audio_path, model.config().inference.audio_sample_rate)
                .unwrap();
        let chunks = slice_waveform(
            &waveform.samples,
            &SlicerConfig {
                sample_rate: waveform.sample_rate,
                ..SlicerConfig::default()
            },
        )
        .unwrap();
        let chunks = split_long_chunks(
            &chunks,
            waveform.sample_rate,
            waveform.sample_rate * DEFAULT_MAX_CHUNK_SECONDS,
        )
        .unwrap();
        let params = InferParams {
            seed: 1,
            ..InferParams::default()
        };
        let actual = run_chunked_extract(&model, &chunks, &params).unwrap();
        let expected = parse_expected_notes(&expected_path).unwrap();
        let metrics = compare_notes_by_frame(
            &expected,
            &actual.notes,
            model.config().inference.timestep(),
        );

        eprintln!(
            "vocal2 CPU regression: expected_notes={} actual_notes={} expected_frames={} actual_frames={} voicedness_match={:.4} pitch_mae={:.4} pitch_le_0_5={:.4}",
            expected.len(),
            actual.notes.len(),
            metrics.expected_frames,
            metrics.actual_frames,
            metrics.voicedness_match_rate,
            metrics.voiced_pitch_mae,
            metrics.voiced_pitch_within_half_semitone_rate
        );

        // The checked-in expected-output fixture is a current extract snapshot,
        // but keep frame-level tolerances so minor formatting or note grouping
        // changes do not make this real-model regression unnecessarily brittle.
        assert!(
            metrics.frame_count_delta <= 4,
            "frame count drift too large: expected {} frames, got {}",
            metrics.expected_frames,
            metrics.actual_frames
        );
        assert!(
            metrics.voicedness_match_rate >= 0.97,
            "frame voiced/unvoiced agreement too low: {:.4}",
            metrics.voicedness_match_rate
        );
        assert!(
            metrics.voiced_pitch_mae <= 0.25,
            "voiced-frame pitch MAE too high: {:.4}",
            metrics.voiced_pitch_mae
        );
        assert!(
            metrics.voiced_pitch_within_half_semitone_rate >= 0.92,
            "too few voiced frames are within 0.5 semitone: {:.4}",
            metrics.voiced_pitch_within_half_semitone_rate
        );
    }

    #[test]
    #[cfg(feature = "gpu")]
    #[ignore = "real-model CPU-vs-GPU regression on current audios with shared default chunking; run with `cargo test --release current_audio_cpu_gpu_shared_chunking_regression -- --ignored --nocapture`"]
    fn current_audio_cpu_gpu_shared_chunking_regression() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        for audio_name in ["vocal1-48000-16bit-2ch.wav", "vocal2-44100-16bit-1ch.wav"] {
            assert_real_model_cpu_gpu_regression_case(&root, audio_name, DEFAULT_MAX_CHUNK_SECONDS);
        }
    }

    #[cfg(feature = "gpu")]
    fn assert_real_model_cpu_gpu_regression_case(
        root: &Path,
        audio_name: &str,
        max_chunk_seconds: usize,
    ) {
        let audio_path = root.join("assets").join("audio").join(audio_name);
        if !audio_path.exists() {
            eprintln!(
                "skipping CPU-vs-GPU regression: missing {}",
                audio_path.display()
            );
            return;
        }

        let cpu = run_real_model_extract_with_shared_chunking(
            root,
            &audio_path,
            Backend::Cpu,
            max_chunk_seconds,
        )
        .unwrap();
        let gpu = run_real_model_extract_with_shared_chunking(
            root,
            &audio_path,
            Backend::Gpu,
            max_chunk_seconds,
        )
        .unwrap();
        let metrics = compare_notes_by_frame(&cpu.notes, &gpu.notes, cpu.timestep);

        eprintln!(
            "{audio_name} CPU-vs-GPU shared chunking: cpu_notes={} gpu_notes={} cpu_frames={} gpu_frames={} voicedness_match={:.4} pitch_mae={:.4} pitch_le_0_5={:.4}",
            cpu.notes.len(),
            gpu.notes.len(),
            metrics.expected_frames,
            metrics.actual_frames,
            metrics.voicedness_match_rate,
            metrics.voiced_pitch_mae,
            metrics.voiced_pitch_within_half_semitone_rate
        );

        assert!(
            metrics.frame_count_delta <= 1,
            "{audio_name}: frame count drift too large: cpu {} gpu {}",
            metrics.expected_frames,
            metrics.actual_frames
        );
        assert!(
            metrics.voicedness_match_rate >= 0.985,
            "{audio_name}: frame voiced/unvoiced agreement too low: {:.4}",
            metrics.voicedness_match_rate
        );
        assert!(
            metrics.voiced_pitch_mae <= 0.15,
            "{audio_name}: voiced-frame pitch MAE too high: {:.4}",
            metrics.voiced_pitch_mae
        );
        assert!(
            metrics.voiced_pitch_within_half_semitone_rate >= 0.94,
            "{audio_name}: too few voiced frames are within 0.5 semitone: {:.4}",
            metrics.voiced_pitch_within_half_semitone_rate
        );
    }

    #[derive(Debug, Clone, Copy)]
    struct FrameComparisonMetrics {
        expected_frames: usize,
        actual_frames: usize,
        frame_count_delta: usize,
        voicedness_match_rate: f32,
        voiced_pitch_mae: f32,
        voiced_pitch_within_half_semitone_rate: f32,
    }

    fn parse_expected_notes(path: &Path) -> std::io::Result<Vec<Note>> {
        let text = fs::read_to_string(path)?;
        let mut notes = Vec::new();
        for (line_index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let mut parts = line.split('\t');
            let offset = parts
                .next()
                .ok_or_else(|| invalid_expected_output(line_index, "missing offset"))?
                .parse::<f32>()
                .map_err(|_| invalid_expected_output(line_index, "invalid offset"))?;
            let duration = parts
                .next()
                .ok_or_else(|| invalid_expected_output(line_index, "missing duration"))?
                .parse::<f32>()
                .map_err(|_| invalid_expected_output(line_index, "invalid duration"))?;
            let pitch = parts
                .next()
                .ok_or_else(|| invalid_expected_output(line_index, "missing pitch"))?;
            if parts.next().is_some() {
                return Err(invalid_expected_output(line_index, "too many columns"));
            }

            let (pitch_midi, voiced) = if pitch == "rest" {
                (0.0, false)
            } else {
                (
                    pitch
                        .parse::<f32>()
                        .map_err(|_| invalid_expected_output(line_index, "invalid pitch"))?,
                    true,
                )
            };

            notes.push(Note {
                offset_seconds: offset,
                duration_seconds: duration,
                pitch_midi,
                voiced,
            });
        }
        Ok(notes)
    }

    struct RealModelExtractResult {
        notes: Vec<Note>,
        timestep: f32,
    }

    fn run_real_model_extract_with_shared_chunking(
        root: &Path,
        audio_path: &Path,
        backend: Backend,
        max_chunk_seconds: usize,
    ) -> Result<RealModelExtractResult, Error> {
        let model_path = root.join("assets").join("models").join("large.gguf");
        let model = Model::load(&model_path, backend)?;
        let waveform =
            prepare_wav_for_inference(audio_path, model.config().inference.audio_sample_rate)?;
        let chunks = slice_waveform(
            &waveform.samples,
            &SlicerConfig {
                sample_rate: waveform.sample_rate,
                ..SlicerConfig::default()
            },
        )?;
        let chunks = split_long_chunks(
            &chunks,
            waveform.sample_rate,
            waveform.sample_rate.saturating_mul(max_chunk_seconds),
        )?;
        let params = InferParams {
            seed: 1,
            ..InferParams::default()
        };
        let result = run_chunked_extract(&model, &chunks, &params)?;
        Ok(RealModelExtractResult {
            notes: result.notes,
            timestep: model.config().inference.timestep(),
        })
    }

    fn invalid_expected_output(line_index: usize, reason: &str) -> std::io::Error {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected output line {}: {reason}", line_index + 1),
        )
    }

    fn compare_notes_by_frame(
        expected: &[Note],
        actual: &[Note],
        timestep: f32,
    ) -> FrameComparisonMetrics {
        let expected_frames = expand_notes_to_frames(expected, timestep);
        let actual_frames = expand_notes_to_frames(actual, timestep);
        let common = expected_frames.len().min(actual_frames.len());
        let total = expected_frames.len().max(actual_frames.len());
        let mut voicedness_matches = 0usize;
        let mut voiced_frame_count = 0usize;
        let mut voiced_pitch_abs_sum = 0.0f32;
        let mut voiced_pitch_within_half = 0usize;

        for index in 0..common {
            match (expected_frames[index], actual_frames[index]) {
                (None, None) => {
                    voicedness_matches += 1;
                }
                (Some(expected_pitch), Some(actual_pitch)) => {
                    voicedness_matches += 1;
                    voiced_frame_count += 1;
                    let diff = (expected_pitch - actual_pitch).abs();
                    voiced_pitch_abs_sum += diff;
                    if diff <= 0.5 {
                        voiced_pitch_within_half += 1;
                    }
                }
                _ => {}
            }
        }

        FrameComparisonMetrics {
            expected_frames: expected_frames.len(),
            actual_frames: actual_frames.len(),
            frame_count_delta: expected_frames.len().abs_diff(actual_frames.len()),
            voicedness_match_rate: ratio(voicedness_matches, total),
            voiced_pitch_mae: if voiced_frame_count == 0 {
                0.0
            } else {
                voiced_pitch_abs_sum / voiced_frame_count as f32
            },
            voiced_pitch_within_half_semitone_rate: ratio(
                voiced_pitch_within_half,
                voiced_frame_count,
            ),
        }
    }

    fn expand_notes_to_frames(notes: &[Note], timestep: f32) -> Vec<Option<f32>> {
        let mut frames = Vec::new();
        for note in notes {
            let frame_count = ((note.duration_seconds / timestep).round() as i32).max(0) as usize;
            let value = note.voiced.then_some(note.pitch_midi);
            frames.extend(std::iter::repeat_n(value, frame_count));
        }
        frames
    }

    fn ratio(numerator: usize, denominator: usize) -> f32 {
        if denominator == 0 {
            0.0
        } else {
            numerator as f32 / denominator as f32
        }
    }
}
