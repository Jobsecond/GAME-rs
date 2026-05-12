use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
#[cfg(feature = "gpu")]
use game_crabml::GpuAdapterSelector;
use game_crabml::{
    BackboneConfig, Backend, Error, InferParams, LoadedGgufModel, LoadedTensor, MidiWriteOptions,
    Model, Result, TextOutputFormat, TextWriteOptions, load_gguf, write_midi_file, write_text_file,
};
use hound::{SampleFormat, WavReader};
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

fn main() -> ExitCode {
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

    let format = resolve_extract_format(args.format, &args.output);
    let model = load_model_for_extract(&args.model, args.device, &args.gpu)?;
    let waveform = load_wav_mono_f32(&args.input, model.config().inference.audio_sample_rate)?;

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
    let result = model.infer(&waveform, &params)?;

    ensure_output_parent_dir(&args.output)?;
    write_extract_output(&args.output, format, &result.notes)?;

    eprintln!("backend: {}", backend_name(model.backend()));
    eprintln!("frames: {}", result.num_frames);
    eprintln!("notes: {}", result.notes.len());
    eprintln!("wrote {}", args.output.display());
    Ok(())
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
                eprintln!("GPU backend unavailable ({gpu_err}); falling back to CPU.");
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

fn load_wav_mono_f32(path: &Path, expected_sample_rate: i32) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)
        .map_err(|err| Error::message(format!("failed to open WAV {}: {err}", path.display())))?;
    let spec = reader.spec();

    if expected_sample_rate > 0 && spec.sample_rate != expected_sample_rate as u32 {
        return Err(Error::message(format!(
            "WAV sample rate {} != expected {} ({})",
            spec.sample_rate,
            expected_sample_rate,
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
            let scale = 1u64.checked_shl(bits.saturating_sub(1)).ok_or_else(|| {
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

    let channels = usize::from(spec.channels);
    if channels == 1 {
        return Ok(interleaved);
    }

    let frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for frame in interleaved.chunks_exact(channels) {
        mono.push(frame.iter().copied().sum::<f32>() / channels as f32);
    }
    Ok(mono)
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
    use std::path::Path;

    use super::{ExtractFormat, infer_extract_format, parse_u32_auto};

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
}
