use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::panic;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use console::Style;
use game_core::{
    BackboneConfig, Backend, CoreEvent, Error, InferParams, LoadedGgufModel, LoadedTensor,
    NotificationLevel, Notifier, Result, load_gguf,
};
use game_service::{
    ChunkParallelism as ServiceChunkParallelism, DEFAULT_MAX_CHUNK_SECONDS,
    ExtractDevice as ServiceExtractDevice, ExtractFormat as ServiceExtractFormat,
    ExtractOutputRequest, ExtractRequest as ServiceExtractRequest,
    ExtractResult as ServiceExtractResult, GpuSelector as ServiceGpuSelector,
    extract_with_notifier as run_extract_with_notifier,
};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use log::Level;
use serde_json::{Map, Value, json};

thread_local! {
    static PANIC_CONTEXT: std::cell::RefCell<PanicContext> = std::cell::RefCell::new(PanicContext::default());
}

#[derive(Debug, Default, Clone)]
struct PanicContext {
    input_file: String,
    stage: String,
}

impl PanicContext {
    fn set(input_file: impl Into<String>, stage: impl Into<String>) {
        PANIC_CONTEXT.with(|ctx| {
            let mut c = ctx.borrow_mut();
            c.input_file = input_file.into();
            c.stage = stage.into();
        });
    }

    fn display(&self) -> String {
        if self.input_file.is_empty() && self.stage.is_empty() {
            String::new()
        } else if self.input_file.is_empty() {
            format!("stage={}", self.stage)
        } else if self.stage.is_empty() {
            format!("file={}", self.input_file)
        } else {
            format!("file={} stage={}", self.input_file, self.stage)
        }
    }
}

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

    #[arg(long = "chunk-parallelism", value_enum, default_value_t = ChunkParallelism::Auto)]
    chunk_parallelism: ChunkParallelism,

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ChunkParallelism {
    Auto,
    On,
    Off,
}

static LOGGER_START: OnceLock<Instant> = OnceLock::new();

impl From<ExtractFormat> for ServiceExtractFormat {
    fn from(value: ExtractFormat) -> Self {
        match value {
            ExtractFormat::Midi => ServiceExtractFormat::Midi,
            ExtractFormat::Txt => ServiceExtractFormat::Txt,
            ExtractFormat::Csv => ServiceExtractFormat::Csv,
        }
    }
}

impl From<ExtractDevice> for ServiceExtractDevice {
    fn from(value: ExtractDevice) -> Self {
        match value {
            ExtractDevice::Cpu => ServiceExtractDevice::Cpu,
            ExtractDevice::Gpu => ServiceExtractDevice::Gpu,
        }
    }
}

impl From<ChunkParallelism> for ServiceChunkParallelism {
    fn from(value: ChunkParallelism) -> Self {
        match value {
            ChunkParallelism::Auto => ServiceChunkParallelism::Auto,
            ChunkParallelism::On => ServiceChunkParallelism::On,
            ChunkParallelism::Off => ServiceChunkParallelism::Off,
        }
    }
}

const STAGE_LINE_WIDTH: usize = 50;
const INSPECT_LABEL_WIDTH: usize = 18;
const INSPECT_VALUE_WIDTH: usize = 52;
const SPINNER_TICKS: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const CHUNK_LABEL_WIDTH: usize = 28;
const CHUNK_PROGRESS_BAR_WIDTH: usize = 8;

struct RichState {
    chunk_labels: HashMap<usize, String>,
    chunk_status_bars: HashMap<usize, ProgressBar>,
    total_chunks: Option<usize>,
    completed_chunks: usize,
    silence_chunk_count: Option<usize>,
    chunk_parallelism_on: bool,
}

struct RichNotifier {
    multi: MultiProgress,
    root: ProgressBar,
    state: Mutex<RichState>,
    use_progress: bool,
    style_label: Style,
    style_timing: Style,
    style_warn: Style,
    style_error: Style,
    style_dim: Style,
    style_completed: Style,
    log_events: bool,
}

impl RichNotifier {
    fn new() -> Self {
        let term = console::Term::stderr();
        let is_tty = term.is_term();
        let use_color = is_tty && std::env::var_os("NO_COLOR").is_none();
        let use_progress = is_tty;
        let log_events = !use_progress && rust_log_requested();

        let multi = if use_progress {
            MultiProgress::with_draw_target(ProgressDrawTarget::stderr())
        } else {
            MultiProgress::with_draw_target(ProgressDrawTarget::hidden())
        };
        let root = multi.add(ProgressBar::new_spinner());
        root.set_style(root_progress_style(false));

        let (style_label, style_timing, style_warn, style_error, style_dim, style_completed) =
            if use_color {
                (
                    Style::new().cyan(),
                    Style::new().green(),
                    Style::new().yellow(),
                    Style::new().red(),
                    Style::new().dim(),
                    Style::new().green().bold(),
                )
            } else {
                let plain = Style::new();
                (
                    plain.clone(),
                    plain.clone(),
                    plain.clone(),
                    plain.clone(),
                    plain.clone(),
                    plain,
                )
            };

        Self {
            multi,
            root,
            state: Mutex::new(RichState {
                chunk_labels: HashMap::new(),
                chunk_status_bars: HashMap::new(),
                total_chunks: None,
                completed_chunks: 0,
                silence_chunk_count: None,
                chunk_parallelism_on: false,
            }),
            use_progress,
            style_label,
            style_timing,
            style_warn,
            style_error,
            style_dim,
            style_completed,
            log_events,
        }
    }

    /// Locks the shared state, recovering from a poisoned mutex instead of
    /// panicking. A worker thread can panic mid-inference while holding this
    /// lock; the protected `RichState` is only progress-bar bookkeeping and stays
    /// structurally valid, so we take the inner guard and carry on rather than
    /// cascading into a second panic during shutdown (`finish`/`print_summary`).
    fn state(&self) -> std::sync::MutexGuard<'_, RichState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn start(&self, args: &ExtractArgs) {
        self.state().chunk_parallelism_on = args.chunk_parallelism == ChunkParallelism::On;
        if !self.use_progress {
            self.println(&format!(
                "Extract {} -> {}",
                args.input.display(),
                args.output.display()
            ));
        }
    }

    fn term_width(&self) -> usize {
        let (_, cols) = console::Term::stderr().size();
        if cols == 0 { usize::MAX } else { cols as usize }
    }

    fn truncate(&self, text: &str) -> String {
        console::truncate_str(text, self.term_width(), "").to_string()
    }

    fn println(&self, line: &str) {
        let out = self.truncate(line);
        if self.use_progress {
            self.multi.suspend(|| eprintln!("{out}"));
        } else {
            eprintln!("{out}");
        }
    }

    fn log_core_event(&self, event: &CoreEvent) {
        if !self.log_events {
            return;
        }

        if self.use_progress {
            self.multi.suspend(|| log_event(event));
        } else {
            log_event(event);
        }
    }

    fn set_root_style(&self, with_counts: bool) {
        self.root.set_style(root_progress_style(with_counts));
    }

    fn set_root_message(&self, message: &str) {
        self.root.set_message(self.truncate(message));
    }

    fn chunk_processing_message(&self) -> &'static str {
        if self.state().chunk_parallelism_on {
            "Processing chunks in parallel..."
        } else {
            "Processing chunks..."
        }
    }

    fn ensure_chunk_bar(&self, chunk_idx: usize) -> ProgressBar {
        let mut state = self.state();
        if let Some(bar) = state.chunk_status_bars.get(&chunk_idx) {
            return bar.clone();
        }

        let before_bar = state
            .chunk_status_bars
            .iter()
            .filter_map(|(&idx, bar)| (idx > chunk_idx).then_some((idx, bar.clone())))
            .min_by_key(|(idx, _)| *idx)
            .map(|(_, bar)| bar);
        let after_bar = state
            .chunk_status_bars
            .iter()
            .filter_map(|(&idx, bar)| (idx < chunk_idx).then_some((idx, bar.clone())))
            .max_by_key(|(idx, _)| *idx)
            .map(|(_, bar)| bar);

        let bar = if let Some(before_bar) = before_bar {
            self.multi
                .insert_before(&before_bar, ProgressBar::new_spinner())
        } else if let Some(after_bar) = after_bar {
            self.multi
                .insert_after(&after_bar, ProgressBar::new_spinner())
        } else {
            self.multi
                .insert_after(&self.root, ProgressBar::new_spinner())
        };
        bar.set_style(child_pending_style());
        state.chunk_status_bars.insert(chunk_idx, bar.clone());
        bar
    }

    fn format_stage_line(&self, label: &str, bracket: &str, elapsed: Duration) -> String {
        let time_str = format_duration(elapsed);
        let plain_left = if bracket.is_empty() {
            format!("  {label}")
        } else {
            format!("  {label} [{bracket}]")
        };
        let padding = STAGE_LINE_WIDTH.saturating_sub(plain_left.len() + time_str.len());

        let colored_left = if bracket.is_empty() {
            format!("  {}", self.style_label.apply_to(label))
        } else {
            format!("  {} [{}]", self.style_label.apply_to(label), bracket)
        };
        format!(
            "{colored_left}{:pad$}{}",
            "",
            self.style_timing.apply_to(&time_str),
            pad = padding
        )
    }

    fn handle_d3pm_progress(&self, current: usize, total: usize, detail: Option<&str>) {
        if !self.use_progress {
            return;
        }

        let chunk_idx = detail.and_then(parse_chunk_index).unwrap_or(0);
        let state = self.state();
        let chunk_info = state
            .chunk_labels
            .get(&chunk_idx)
            .cloned()
            .unwrap_or_else(|| format!("chunk {}", chunk_idx + 1));
        drop(state);

        let bar = self.ensure_chunk_bar(chunk_idx);
        bar.set_style(child_numeric_progress_style());
        bar.set_length(total as u64);
        bar.set_position(current as u64);
        bar.set_message(self.truncate(&chunk_info));
        self.set_root_message(self.chunk_processing_message());
    }

    fn finish(&self) {
        let mut state = self.state();
        self.root.set_message("");
        self.root.finish_and_clear();
        for (_, bar) in state.chunk_status_bars.drain() {
            bar.finish_and_clear();
        }
        state.chunk_labels.clear();
    }

    fn print_summary(&self, result: &ServiceExtractResult) {
        let audio_val = if result.audio.was_resampled() || result.audio.was_downmixed() {
            format!(
                "{} Hz/{} ch → {} Hz mono",
                result.audio.source_sample_rate,
                result.audio.source_channels,
                result.audio.sample_rate
            )
        } else {
            format!("{} Hz mono", result.audio.sample_rate)
        };

        let realtime = if result.timings.inference.is_zero() {
            0.0
        } else {
            result.audio.duration_seconds() / result.timings.inference.as_secs_f64()
        };
        let chunks_val = if result.chunks_before_long_split != result.chunk_count {
            format!(
                "{} -> {}",
                result.chunks_before_long_split, result.chunk_count
            )
        } else {
            result.chunk_count.to_string()
        };

        if !self.use_progress {
            self.println("");
        }

        self.println(&format!(
            "{} {} notes in {} ({realtime:.2}x realtime)",
            self.style_completed.apply_to("Extracted"),
            result.notes.len(),
            self.style_timing
                .apply_to(format_duration(result.timings.total))
        ));
        self.println(&format!(
            "  {} {} | {} {} | {} {}",
            self.style_label.apply_to("Audio"),
            audio_val,
            self.style_label.apply_to("Frames"),
            format_count(result.total_frames as u64),
            self.style_label.apply_to("Chunks"),
            chunks_val
        ));

        if let Some(adapter) = &result.gpu_adapter {
            self.println(&format!(
                "  {} {} ({})",
                self.style_label.apply_to("Backend"),
                backend_name(result.backend),
                adapter.name
            ));
        } else {
            self.println(&format!(
                "  {} {}",
                self.style_label.apply_to("Backend"),
                backend_name(result.backend)
            ));
        }

        if let Some(output) = &result.output {
            self.println(&format!(
                "  {} {} ({})",
                self.style_label.apply_to("Output"),
                output.path.display(),
                format!("{:?}", output.format).to_lowercase()
            ));
        }
        self.println(&format!(
            "  {} load={} audio={} infer={} write={}",
            self.style_label.apply_to("Timings"),
            format_duration(result.timings.model_load),
            format_duration(result.timings.audio_prepare),
            format_duration(result.timings.inference),
            format_duration(result.timings.output_write)
        ));
    }
}

impl Notifier for RichNotifier {
    fn notify(&self, event: CoreEvent) {
        self.log_core_event(&event);
        match event {
            CoreEvent::ModelLoaded { backend, elapsed } => {
                self.println(&self.format_stage_line("model loaded", backend, elapsed));
            }

            CoreEvent::Status { stage, message } => match stage {
                "extract_infer" => {
                    if let Some(count) = parse_chunk_count_from_message(&message) {
                        let mut state = self.state();
                        state.total_chunks = Some(count);
                        state.completed_chunks = 0;
                        state.chunk_labels.clear();
                        for (_, bar) in state.chunk_status_bars.drain() {
                            bar.finish_and_clear();
                        }
                        drop(state);
                        self.root.set_position(0);
                        self.root.set_length(count as u64);
                        self.root.reset_elapsed();
                        self.set_root_style(true);
                        self.root.enable_steady_tick(Duration::from_millis(200));
                        self.set_root_message(self.chunk_processing_message());
                    }
                }
                "chunk_infer" => {
                    let chunk_info = format_chunk_status(&message);
                    let idx = parse_chunk_index(&message).unwrap_or(0);
                    {
                        let mut state = self.state();
                        state.chunk_labels.insert(idx, chunk_info.clone());
                    }
                    if self.use_progress {
                        let bar = self.ensure_chunk_bar(idx);
                        bar.set_style(child_pending_style());
                        bar.set_message(self.truncate(&chunk_info));
                        self.set_root_message(self.chunk_processing_message());
                    } else {
                        self.println(&format!(
                            "  {} {chunk_info}",
                            self.style_label.apply_to("Processing")
                        ));
                    }
                }
                _ => {}
            },

            CoreEvent::Progress {
                stage,
                current,
                total,
                detail,
            } => {
                if stage == "d3pm_step" {
                    self.handle_d3pm_progress(current, total, detail.as_deref());
                }
            }

            CoreEvent::Timing {
                stage,
                elapsed,
                detail,
            } => match stage {
                "audio_prepare" => {
                    let bracket = detail
                        .as_deref()
                        .map(parse_audio_bracket)
                        .unwrap_or_default();
                    self.println(&self.format_stage_line("audio prepared", &bracket, elapsed));
                }
                "silence_slice" => {
                    let bracket = detail
                        .as_deref()
                        .and_then(|d| parse_kv_usize(d, "chunks="))
                        .map(|n| {
                            self.state().silence_chunk_count = Some(n);
                            format!("{n} chunk{}", if n != 1 { "s" } else { "" })
                        })
                        .unwrap_or_default();
                    self.println(&self.format_stage_line("sliced on silence", &bracket, elapsed));
                }
                "long_chunk_split" => {
                    let after = detail.as_deref().and_then(|d| parse_kv_usize(d, "chunks="));
                    let before = self.state().silence_chunk_count;
                    if let (Some(before), Some(after)) = (before, after) {
                        if before != after {
                            let bracket = format!("{before} -> {after}");
                            self.println(&self.format_stage_line(
                                "split long chunks",
                                &bracket,
                                elapsed,
                            ));
                        }
                    }
                }
                "mel_setup" => {
                    let bracket = detail
                        .as_deref()
                        .and_then(|d| parse_kv_usize(d, "frames="))
                        .map(|n| format!("{n} frames"))
                        .unwrap_or_default();
                    self.println(&self.format_stage_line("mel setup", &bracket, elapsed));
                }
                "infer_total" => {
                    let idx = detail.as_deref().and_then(parse_chunk_index).unwrap_or(0);
                    let mut state = self.state();
                    state.chunk_labels.remove(&idx);
                    if let Some(bar) = state.chunk_status_bars.remove(&idx) {
                        bar.finish_and_clear();
                    }
                    state.completed_chunks += 1;
                    let completed_chunks = state.completed_chunks;
                    let total_chunks = state.total_chunks.unwrap_or(completed_chunks);
                    drop(state);
                    if self.use_progress {
                        self.root.set_position(completed_chunks as u64);
                        let _ = total_chunks;
                        self.set_root_message(self.chunk_processing_message());
                    }
                }
                "chunk_infer" => {}
                "extract_infer" => {
                    let mut state = self.state();
                    for (_, bar) in state.chunk_status_bars.drain() {
                        bar.finish_and_clear();
                    }
                    state.chunk_labels.clear();
                    drop(state);
                    if self.use_progress {
                        self.root.disable_steady_tick();
                        self.root.set_message("");
                        self.root.finish_and_clear();
                    } else {
                        self.println("");
                    }
                }
                "output_write" => {
                    let path = detail
                        .as_deref()
                        .and_then(|d| d.strip_prefix("path="))
                        .map(|d| d.split_whitespace().next().unwrap_or(d))
                        .unwrap_or("output");
                    let display = display_path_name(Path::new(path));
                    if self.use_progress {
                        self.set_root_message(&format!("Wrote {display}"));
                    } else {
                        self.println(&self.format_stage_line(
                            &format!("wrote {display}"),
                            "",
                            elapsed,
                        ));
                    }
                }
                _ => {}
            },

            CoreEvent::Message { level, message } => match level {
                NotificationLevel::Warn => {
                    self.println(&format!(
                        "  {} {}",
                        self.style_warn.apply_to("warning:"),
                        message
                    ));
                }
                NotificationLevel::Error => {
                    self.println(&format!(
                        "  {} {}",
                        self.style_error.apply_to("error:"),
                        message
                    ));
                }
                NotificationLevel::Info => {
                    self.println(&format!("  {}", self.style_dim.apply_to(&message)));
                }
                _ => {}
            },
        }
    }
}

fn root_progress_style(with_counts: bool) -> ProgressStyle {
    if with_counts {
        ProgressStyle::with_template("{spinner:.white} {msg:.dim} ({pos}/{len})  {elapsed:.dim}")
            .unwrap()
            .tick_strings(&SPINNER_TICKS)
    } else {
        ProgressStyle::with_template("{spinner:.white} {msg:.dim}")
            .unwrap()
            .tick_strings(&SPINNER_TICKS)
    }
}

fn child_pending_style() -> ProgressStyle {
    ProgressStyle::with_template(&format!("{{msg:{CHUNK_LABEL_WIDTH}.dim}}")).unwrap()
}

fn child_numeric_progress_style() -> ProgressStyle {
    ProgressStyle::with_template(&format!(
        "{{msg:{CHUNK_LABEL_WIDTH}.dim}} [{{bar:{CHUNK_PROGRESS_BAR_WIDTH}.green/black.dim}}] D3PM Step {{pos}}/{{len}}"
    ))
    .unwrap()
    .progress_chars("=> ")
}

fn display_path_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}

fn parse_chunk_count_from_message(message: &str) -> Option<usize> {
    let parts: Vec<&str> = message.split_whitespace().collect();
    for (i, part) in parts.iter().enumerate() {
        if part.starts_with("chunk") && i > 0 {
            return parts[i - 1].parse().ok();
        }
    }
    None
}

fn parse_chunk_index(detail: &str) -> Option<usize> {
    let rest = detail.strip_prefix("chunk ")?;
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    rest[..end]
        .parse::<usize>()
        .ok()
        .map(|n| n.saturating_sub(1))
}

fn parse_audio_bracket(detail: &str) -> String {
    if let Some(arrow_pos) = detail.find("-> ") {
        let rest = &detail[arrow_pos + 3..];
        rest.split(',').next().unwrap_or(rest).trim().to_owned()
    } else {
        detail.split(',').next().unwrap_or(detail).trim().to_owned()
    }
}

fn parse_kv_usize(detail: &str, key: &str) -> Option<usize> {
    let pos = detail.find(key)?;
    let rest = &detail[pos + key.len()..];
    rest.split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

fn rust_log_requested() -> bool {
    std::env::var_os("RUST_LOG")
        .map(|value| !value.as_os_str().is_empty())
        .unwrap_or(false)
}

fn format_chunk_status(message: &str) -> String {
    if let Some(pos) = message.find(": infer start ") {
        let chunk_id = &message[..pos];
        let chunk_label = chunk_id
            .split_once('/')
            .map(|(chunk_number, _)| chunk_number)
            .unwrap_or(chunk_id);
        let rest = &message[pos + ": infer start ".len()..];
        let duration = rest
            .split_whitespace()
            .find_map(|part| part.strip_prefix("duration="))
            .unwrap_or_default();
        if duration.is_empty() {
            chunk_label.to_owned()
        } else {
            format!("{chunk_label} (dur: {duration})")
        }
    } else {
        message.to_owned()
    }
}

fn main() -> ExitCode {
    install_panic_hook();
    init_logging();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn log_event(event: &CoreEvent) {
    use log::{error, info, warn};
    match event {
        CoreEvent::Status { stage, message } => {
            info!("[{}] {}", stage, message);
        }
        CoreEvent::Progress {
            stage,
            current,
            total,
            detail,
        } => {
            let detail_str = detail
                .as_ref()
                .map(|d| format!(" ({})", d))
                .unwrap_or_default();
            info!("[{}] progress {}/{}{}", stage, current, total, detail_str);
        }
        CoreEvent::Timing {
            stage,
            elapsed,
            detail,
        } => {
            let detail_str = detail
                .as_ref()
                .map(|d| format!(" {}", d))
                .unwrap_or_default();
            info!(
                "[{}] timing {:.3}s{}",
                stage,
                elapsed.as_secs_f64(),
                detail_str
            );
        }
        CoreEvent::ModelLoaded { backend, elapsed } => {
            info!(
                "model loaded on {} backend in {:.3}s",
                backend,
                elapsed.as_secs_f64()
            );
        }
        CoreEvent::Message { level, message } => match level {
            NotificationLevel::Trace => log::trace!("{}", message),
            NotificationLevel::Debug => log::debug!("{}", message),
            NotificationLevel::Info => info!("{}", message),
            NotificationLevel::Warn => warn!("{}", message),
            NotificationLevel::Error => error!("{}", message),
        },
    }
}

fn install_panic_hook() {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(move |panic_info| {
        PANIC_CONTEXT.with(|ctx| {
            let context = ctx.borrow();
            let context_msg = context.display();
            if !context_msg.is_empty() {
                eprintln!("panic breadcrumb: {context_msg}");
            }
        });
        default_hook(panic_info);
    }));
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
    PanicContext::set(args.input.display().to_string(), "extract");
    let request = build_extract_request(&args);
    let notifier = RichNotifier::new();
    notifier.start(&args);
    let result = run_extract_with_notifier(&request, &notifier);
    notifier.finish();
    let result = result?;
    notifier.print_summary(&result);
    Ok(())
}

fn build_extract_request(args: &ExtractArgs) -> ServiceExtractRequest {
    ServiceExtractRequest {
        model_path: args.model.clone(),
        input_path: args.input.clone(),
        output: Some(ExtractOutputRequest {
            path: args.output.clone(),
            format: args.format.map(Into::into),
            midi_options: Default::default(),
            text_options: Default::default(),
        }),
        device: args.device.map_or(ServiceExtractDevice::Auto, Into::into),
        gpu: args.gpu.to_service_selector(),
        infer_params: InferParams {
            language: args.language,
            d3pm_t0: args.d3pm_t0,
            d3pm_nsteps: args.d3pm_nsteps,
            boundary_threshold: args.boundary_threshold,
            boundary_radius: args.boundary_radius,
            note_threshold: args.note_threshold,
            seed: args.seed,
            ..InferParams::default()
        },
        chunk_parallelism: args.chunk_parallelism.into(),
        max_chunk_seconds: args.max_chunk_seconds,
    }
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        Backend::Gpu => "gpu",
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

    print_inspect_section("gguf");
    print_inspect_kv("version", model.gguf_version.to_string());
    print_inspect_kv("architecture", model.config.architecture.as_str());
    print_inspect_kv(
        "quantization",
        model
            .quantization_version
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_owned()),
    );
    print_inspect_kv("metadata keys", format_count(model.metadata_count as u64));
    print_inspect_kv("tensor count", format_count(model.tensor_count() as u64));
    print_inspect_kv("parameter count", format_count(total_parameters));
    print_inspect_kv(
        "loaded weights",
        format!(
            "{} bytes ({})",
            format_count(total_loaded_bytes),
            format_bytes(total_loaded_bytes)
        ),
    );

    if let Some(prefix) = tensor_prefix.as_deref() {
        print_inspect_section("tensor_filter");
        print_inspect_kv("prefix", prefix);
        print_inspect_kv(
            "matched tensors",
            format_count(filtered_stats.tensor_count as u64),
        );
        print_inspect_kv(
            "matched parameters",
            format_count(filtered_stats.parameter_count),
        );
        print_inspect_kv("matched bytes", format_bytes(filtered_stats.byte_count));
    }

    print_inspect_section("model_config");
    print_inspect_kv("name", display_or_dash(&model.config.name));
    print_inspect_kv("version", display_or_dash(&model.config.version));
    print_inspect_kv("mode", &model.config.mode);
    print_inspect_kv("embedding dim", model.config.embedding_dim.to_string());
    print_inspect_kv("input dim", model.config.in_dim.to_string());
    print_inspect_kv("estimator out", model.config.estimator_out_dim.to_string());
    print_inspect_kv("region cycle", model.config.region_cycle_len.to_string());
    print_inspect_kv("use languages", model.config.use_languages.to_string());
    print_inspect_kv("num languages", model.config.num_languages.to_string());

    let inference = &model.config.inference;
    print_inspect_section("inference");
    print_inspect_kv("sample rate", inference.audio_sample_rate.to_string());
    print_inspect_kv("hop size", inference.hop_size.to_string());
    print_inspect_kv("timestep", format!("{:.6}s", inference.timestep()));
    print_inspect_kv("fft size", inference.fft_size.to_string());
    print_inspect_kv("win size", inference.win_size.to_string());
    print_inspect_kv(
        "spectrogram",
        format!(
            "type={} bins={} fmin={} fmax={}",
            inference.spectrogram_type, inference.n_mels, inference.fmin, inference.fmax
        ),
    );
    print_inspect_kv(
        "midi",
        format!(
            "min={} max={} bins={} std={}",
            inference.midi_min, inference.midi_max, inference.midi_num_bins, inference.midi_std
        ),
    );
    print_inspect_kv(
        "lang map",
        if inference.lang_map.is_empty() {
            "none".to_owned()
        } else {
            inference
                .lang_map
                .iter()
                .map(|(lang, id)| format!("{lang}={id}"))
                .collect::<Vec<_>>()
                .join(", ")
        },
    );

    print_inspect_section("backbones");
    print_backbone("encoder", &model.config.encoder);
    print_backbone("segmenter", &model.config.segmenter);
    print_backbone("estimator", &model.config.estimator);

    print_inspect_section("tensor_types");
    for (tensor_type, count) in tensor_type_counts(&model) {
        print_inspect_kv(&tensor_type, format_count(count as u64));
    }

    print_inspect_section("tensor_prefixes");
    for (prefix, stats) in tensor_prefix_stats(&model) {
        print_inspect_kv(
            &prefix,
            format!(
                "tensors={} params={} bytes={}",
                format_count(stats.tensor_count as u64),
                format_count(stats.parameter_count),
                format_bytes(stats.byte_count)
            ),
        );
    }

    if show_tensors > 0 {
        print_inspect_section(if tensor_prefix.is_some() {
            "largest_matching_tensors"
        } else {
            "largest_tensors"
        });
        for (index, (name, tensor)) in largest_tensors(&filtered_tensors, show_tensors)
            .into_iter()
            .enumerate()
        {
            print_tensor_summary(Some(index + 1), name, tensor);
        }

        print_inspect_section(if tensor_prefix.is_some() {
            "sample_matching_tensors"
        } else {
            "sample_tensors"
        });
        for (name, tensor) in filtered_tensors.iter().take(show_tensors) {
            print_tensor_summary(None, name, tensor);
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

fn print_inspect_section(title: &str) {
    println!("{title}:");
    println!("{}", "─".repeat(title.len().max(INSPECT_LABEL_WIDTH)));
}

fn print_inspect_kv(label: &str, value: impl AsRef<str>) {
    let value = value.as_ref();
    let pad = " ".repeat(INSPECT_LABEL_WIDTH.saturating_sub(label.len()));
    for (index, line) in wrap_text(value, INSPECT_VALUE_WIDTH)
        .into_iter()
        .enumerate()
    {
        if index == 0 {
            println!("  {label}{pad} {line}");
        } else {
            println!("  {:width$} {line}", "", width = INSPECT_LABEL_WIDTH);
        }
    }
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_owned()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let next_len = if current.is_empty() {
            word.len()
        } else {
            current.len() + 1 + word.len()
        };
        if !current.is_empty() && next_len > width {
            lines.push(current);
            current = word.to_owned();
        } else if current.is_empty() {
            current.push_str(word);
        } else {
            current.push(' ');
            current.push_str(word);
        }
    }

    if current.is_empty() {
        lines.push(text.to_owned());
    } else {
        lines.push(current);
    }

    lines
}

fn print_backbone(name: &str, backbone: &BackboneConfig) {
    print_inspect_kv(
        name,
        format!(
            "cls={} dim={} layers={} heads={} head_dim={} ffn_type={}",
            display_or_dash(&backbone.cls),
            backbone.dim,
            backbone.num_layers,
            backbone.num_heads,
            backbone.head_dim,
            backbone.ffn_type
        ),
    );
    print_inspect_kv(
        "conv",
        format!(
            "c_kernel={} m_kernel={} use_ls={} use_out_norm={} skip_first_ffn={} skip_out_ffn={}",
            backbone.c_kernel_size,
            backbone.m_kernel_size,
            backbone.use_ls,
            backbone.use_out_norm,
            backbone.skip_first_ffn,
            backbone.skip_out_ffn
        ),
    );

    if backbone.return_latent {
        print_inspect_kv(
            "latent",
            format!(
                "enabled=true layer_idx={} out_dim={}",
                backbone.latent_layer_idx, backbone.latent_out_dim
            ),
        );
    }

    if backbone.region_token_num != 1
        || backbone.c_kernel_size_pool != 0
        || backbone.m_kernel_size_pool != 0
        || backbone.c_kernel_size_x != 0
        || backbone.m_kernel_size_x != 0
    {
        print_inspect_kv(
            "joint",
            format!(
                "region_tokens={} merge={} attn_type={} rope_mode={} qk_norm={} region_bias={} use_rope={} use_pool_offset={} theta={}",
                backbone.region_token_num,
                backbone.pool_merge_mode,
                backbone.attn_type,
                backbone.rope_mode,
                backbone.qk_norm,
                backbone.use_region_bias,
                backbone.use_rope,
                backbone.use_pool_offset,
                backbone.theta
            ),
        );
        print_inspect_kv(
            "joint_conv",
            format!(
                "pool(c={}, m={}) x(c={}, m={})",
                backbone.c_kernel_size_pool,
                backbone.m_kernel_size_pool,
                backbone.c_kernel_size_x,
                backbone.m_kernel_size_x
            ),
        );
    }
}

fn print_tensor_summary(index: Option<usize>, name: &str, tensor: &LoadedTensor) {
    let label = index.map_or_else(|| name.to_owned(), |index| format!("{index}. {name}"));
    print_inspect_kv(
        &label,
        format!(
            "shape={:?} type={} numel={} bytes={}",
            tensor.shape,
            tensor.tensor_type,
            format_count(tensor.num_elements() as u64),
            format_bytes(tensor.byte_len() as u64)
        ),
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
    fn to_service_selector(&self) -> ServiceGpuSelector {
        ServiceGpuSelector {
            name_substring: self.gpu_name.clone(),
            vendor_id: self.gpu_vendor_id,
            device_id: self.gpu_device_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use clap::Parser;
    use game_core::Note;
    use game_service::{
        ExtractDevice as ServiceExtractDevice, ExtractFormat as ServiceExtractFormat,
        ExtractRequest, extract as run_extract_pipeline,
        infer_extract_format as service_infer_extract_format,
    };

    use super::{
        ChunkParallelism, Cli, Command, DEFAULT_MAX_CHUNK_SECONDS, format_chunk_status,
        parse_audio_bracket, parse_chunk_count_from_message, parse_chunk_index, parse_kv_usize,
        parse_u32_auto,
    };

    #[test]
    fn parse_chunk_count_from_inference_message() {
        assert_eq!(
            parse_chunk_count_from_message("running inference across 5 chunk(s)"),
            Some(5)
        );
        assert_eq!(
            parse_chunk_count_from_message("running inference across 1 chunk(s)"),
            Some(1)
        );
        assert_eq!(parse_chunk_count_from_message("unrelated message"), None);
    }

    #[test]
    fn parse_chunk_index_from_detail() {
        assert_eq!(parse_chunk_index("chunk 2/5 notes=42"), Some(1));
        assert_eq!(parse_chunk_index("chunk 1/1: t=0.000"), Some(0));
        assert_eq!(parse_chunk_index("chunk 10/10: t=0.500"), Some(9));
        assert_eq!(parse_chunk_index("no chunk here"), None);
    }

    #[test]
    fn parse_audio_bracket_with_resample() {
        assert_eq!(
            parse_audio_bracket("44100 Hz/2 ch -> 16000 Hz mono, samples=123, duration=1.23s"),
            "16000 Hz mono"
        );
    }

    #[test]
    fn parse_audio_bracket_no_resample() {
        assert_eq!(
            parse_audio_bracket("16000 Hz mono, samples=123, duration=1.23s"),
            "16000 Hz mono"
        );
    }

    #[test]
    fn parse_kv_usize_extracts_values() {
        assert_eq!(parse_kv_usize("chunks=5", "chunks="), Some(5));
        assert_eq!(parse_kv_usize("frames=1024", "frames="), Some(1024));
        assert_eq!(
            parse_kv_usize("chunks=5 max_chunk_seconds=30", "chunks="),
            Some(5)
        );
        assert_eq!(parse_kv_usize("no match", "chunks="), None);
    }

    #[test]
    fn format_chunk_status_strips_infer_start() {
        assert_eq!(
            format_chunk_status("chunk 1/5: infer start offset=0.00s duration=4.56s"),
            "chunk 1 (dur: 4.56s)"
        );
    }

    #[test]
    fn format_chunk_status_passthrough_unknown() {
        assert_eq!(format_chunk_status("something else"), "something else");
    }

    #[test]
    fn parse_u32_auto_accepts_decimal_and_hex() {
        assert_eq!(parse_u32_auto("1234").unwrap(), 1234);
        assert_eq!(parse_u32_auto("0x10de").unwrap(), 0x10de);
        assert_eq!(parse_u32_auto("0X2484").unwrap(), 0x2484);
    }

    #[test]
    fn infer_extract_format_from_output_extension() {
        assert_eq!(
            service_infer_extract_format(Path::new("notes.mid")),
            Some(ServiceExtractFormat::Midi)
        );
        assert_eq!(
            service_infer_extract_format(Path::new("notes.txt")),
            Some(ServiceExtractFormat::Txt)
        );
        assert_eq!(
            service_infer_extract_format(Path::new("notes.csv")),
            Some(ServiceExtractFormat::Csv)
        );
        assert_eq!(
            service_infer_extract_format(Path::new("notes.unknown")),
            None
        );
    }

    #[test]
    fn extract_cli_uses_default_max_chunk_seconds() {
        let cli = Cli::try_parse_from([
            "game-cli",
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
            "game-cli",
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
            "game-cli",
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
    fn extract_cli_uses_default_chunk_parallelism() {
        let cli = Cli::try_parse_from([
            "game-cli",
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
        assert_eq!(args.chunk_parallelism, ChunkParallelism::Auto);
    }

    #[test]
    fn extract_cli_accepts_chunk_parallelism_flag() {
        let cli = Cli::try_parse_from([
            "game-cli",
            "extract",
            "-m",
            "model.gguf",
            "-o",
            "notes.txt",
            "--chunk-parallelism",
            "off",
            "input.wav",
        ])
        .unwrap();

        let Command::Extract(args) = cli.command else {
            panic!("expected extract command");
        };
        assert_eq!(args.chunk_parallelism, ChunkParallelism::Off);
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

        let actual = run_extract_pipeline(&ExtractRequest {
            model_path,
            input_path: audio_path,
            output: None,
            device: ServiceExtractDevice::Cpu,
            infer_params: game_core::InferParams {
                seed: 1,
                ..game_core::InferParams::default()
            },
            max_chunk_seconds: DEFAULT_MAX_CHUNK_SECONDS,
            ..ExtractRequest::default()
        })
        .unwrap();
        let expected = parse_expected_notes(&expected_path).unwrap();
        let metrics = compare_notes_by_frame(&expected, &actual.notes, actual.timestep_seconds);

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
        let model_path = root.join("assets").join("models").join("large.gguf");
        let audio_path = root.join("assets").join("audio").join(audio_name);
        if let Some(missing) = [&model_path, &audio_path]
            .into_iter()
            .find(|path| !path.exists())
        {
            eprintln!(
                "skipping CPU-vs-GPU regression: missing {}",
                missing.display()
            );
            return;
        }

        let cpu = run_real_model_extract_with_shared_chunking(
            root,
            &audio_path,
            ServiceExtractDevice::Cpu,
            max_chunk_seconds,
        )
        .unwrap();
        let gpu = run_real_model_extract_with_shared_chunking(
            root,
            &audio_path,
            ServiceExtractDevice::Gpu,
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

    #[cfg(feature = "gpu")]
    struct RealModelExtractResult {
        notes: Vec<Note>,
        timestep: f32,
    }

    #[cfg(feature = "gpu")]
    fn run_real_model_extract_with_shared_chunking(
        root: &Path,
        audio_path: &Path,
        device: ServiceExtractDevice,
        max_chunk_seconds: usize,
    ) -> game_core::Result<RealModelExtractResult> {
        let model_path = root.join("assets").join("models").join("large.gguf");
        let result = run_extract_pipeline(&ExtractRequest {
            model_path,
            input_path: audio_path.to_path_buf(),
            output: None,
            device,
            infer_params: game_core::InferParams {
                seed: 1,
                ..game_core::InferParams::default()
            },
            chunk_parallelism: game_service::ChunkParallelism::Auto,
            max_chunk_seconds,
            ..ExtractRequest::default()
        })?;
        Ok(RealModelExtractResult {
            notes: result.notes,
            timestep: result.timestep_seconds,
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
