use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use game_crabml::{BackboneConfig, Error, LoadedGgufModel, LoadedTensor, Result, load_gguf};
use serde_json::{Map, Value, json};

#[derive(Debug, Parser)]
#[command(author, version, about = "Rust port scaffold for GAME GGUF inference")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Inspect {
        #[arg(short = 'm', long = "model")]
        model: PathBuf,

        #[arg(long, default_value_t = 8)]
        show_tensors: usize,

        #[arg(long)]
        tensor_prefix: Option<String>,

        #[arg(long, value_enum, default_value_t = InspectFormat::Text)]
        format: InspectFormat,
    },
    Extract {
        #[arg(short = 'm', long = "model")]
        model: PathBuf,

        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        input: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum InspectFormat {
    Text,
    Json,
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
        Command::Inspect {
            model,
            show_tensors,
            tensor_prefix,
            format,
        } => inspect(model, show_tensors, tensor_prefix, format),
        Command::Extract {
            model,
            output,
            input,
        } => Err(Error::message(format!(
            "`extract` is not implemented yet. Phase 1 currently supports GGUF inspection only. model={}, output={}, input={}",
            model.display(),
            output.display(),
            input.display()
        ))),
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
