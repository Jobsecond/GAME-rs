use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::Receiver;
use game_core::{CoreEvent, NotificationLevel};
use game_service::{
    ChunkParallelism, DEFAULT_MAX_CHUNK_SECONDS, Error, ExtractDevice, ExtractFormat,
    ExtractOutputRequest, ExtractRequest, ExtractResult, GpuSelector, InferParams,
    MidiWriteOptions, TextWriteOptions, extract_with_notifier,
};

use crate::notifier::GuiNotifier;
use crate::pages::AppPage;

pub struct AppState {
    pub current_page: AppPage,
    pub config: ExtractConfig,
    pub event_rx: Option<Receiver<CoreEvent>>,
    pub extraction: Option<ExtractionHandle>,
    pub result: Option<ExtractResult>,
    pub status_text: String,
    pub overall_status: String,
    pub is_running: bool,
    pub cancel_requested: bool,
    pub error_message: Option<String>,
    pub overall_progress: Option<(usize, usize)>,
    pub chunk_progress: Vec<ChunkProgress>,
    pub stage_timings: HashMap<&'static str, StageTiming>,
    pub event_log: Vec<GuiLogEntry>,
    pub max_log_entries: usize,
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub gpu_adapters: Vec<GpuAdapterChoice>,
    #[cfg(feature = "gpu")]
    pub gpu_adapters_loading: bool,
    #[cfg(feature = "gpu")]
    gpu_adapter_rx: Option<Receiver<Vec<GpuAdapterChoice>>>,
    run_started_at: Option<Instant>,
}

pub struct ExtractionHandle {
    pub join_handle: thread::JoinHandle<game_service::Result<ExtractResult>>,
    pub cancel_flag: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct ExtractConfig {
    pub model_path: String,
    pub audio_path: String,
    pub output_path: String,
    pub output_format: ExtractFormat,
    pub device: ExtractDevice,
    pub gpu_selector: GpuSelector,
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub selected_gpu_index: Option<usize>,
    pub d3pm_nsteps: i32,
    pub seed: u64,
    pub chunk_parallelism: ChunkParallelism,
    pub max_chunk_seconds: usize,
    pub language: i32,
    pub boundary_threshold: f32,
    pub note_threshold: f32,
    pub boundary_radius: i32,
}

#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub struct GpuAdapterChoice {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
}

#[derive(Debug, Clone)]
pub struct ChunkProgress {
    pub label: String,
    pub d3pm_current: usize,
    pub d3pm_total: usize,
    pub status: ChunkStatus,
}

#[derive(Debug, Clone)]
pub enum ChunkStatus {
    Pending,
    Running,
    Completed,
    #[allow(dead_code)]
    Failed(String),
}

#[derive(Debug, Clone, Copy)]
pub struct StageTiming {
    pub elapsed: Duration,
    pub completed: bool,
}

#[derive(Debug, Clone)]
pub struct GuiLogEntry {
    pub elapsed: Duration,
    pub level: GuiLogLevel,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiLogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            current_page: AppPage::Config,
            config: ExtractConfig::default(),
            event_rx: None,
            extraction: None,
            result: None,
            status_text: String::new(),
            overall_status: "Ready".to_owned(),
            is_running: false,
            cancel_requested: false,
            error_message: None,
            overall_progress: None,
            chunk_progress: Vec::new(),
            stage_timings: HashMap::new(),
            event_log: Vec::new(),
            max_log_entries: 500,
            gpu_adapters: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu_adapters_loading: false,
            #[cfg(feature = "gpu")]
            gpu_adapter_rx: None,
            run_started_at: None,
        }
    }

    pub fn poll_background_work(&mut self) {
        #[cfg(feature = "gpu")]
        self.poll_gpu_adapter_refresh();
    }

    #[cfg(feature = "gpu")]
    pub fn ensure_gpu_adapter_refresh_started(&mut self, ctx: &egui::Context) {
        if self.gpu_adapters_loading || !self.gpu_adapters.is_empty() {
            return;
        }
        self.refresh_gpu_adapters(ctx);
    }

    #[cfg(feature = "gpu")]
    pub fn refresh_gpu_adapters(&mut self, ctx: &egui::Context) {
        if self.gpu_adapters_loading {
            return;
        }

        let (tx, rx) = crossbeam::channel::bounded(1);
        let repaint_ctx = ctx.clone();
        self.gpu_adapters_loading = true;
        self.gpu_adapter_rx = Some(rx);
        thread::spawn(move || {
            let adapters = list_gpu_adapters();
            let _ = tx.send(adapters);
            repaint_ctx.request_repaint();
        });
    }

    #[cfg(feature = "gpu")]
    fn poll_gpu_adapter_refresh(&mut self) {
        let Some(rx) = self.gpu_adapter_rx.take() else {
            return;
        };

        match rx.try_recv() {
            Ok(adapters) => {
                self.gpu_adapters = adapters;
                self.gpu_adapters_loading = false;
                if self
                    .config
                    .selected_gpu_index
                    .is_some_and(|index| index >= self.gpu_adapters.len())
                {
                    self.config.selected_gpu_index = None;
                    self.config.gpu_selector = GpuSelector::default();
                }
            }
            Err(crossbeam::channel::TryRecvError::Empty) => {
                self.gpu_adapter_rx = Some(rx);
            }
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                self.gpu_adapters_loading = false;
            }
        }
    }

    pub fn start_extraction(&mut self, ctx: &egui::Context) {
        if self.is_running {
            return;
        }

        if let Err(message) = self.validate_config() {
            self.error_message = Some(message);
            return;
        }

        self.clear_run_state();
        self.current_page = AppPage::Progress;
        self.is_running = true;
        self.cancel_requested = false;
        self.overall_status = "Starting extraction...".to_owned();
        self.status_text.clear();
        self.run_started_at = Some(Instant::now());

        let (notifier, rx) = GuiNotifier::channel(ctx.clone());
        self.event_rx = Some(rx);
        let config = self.config.clone();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel_flag);
        let join_handle = thread::spawn(move || run_extraction(config, notifier, worker_cancel));
        self.extraction = Some(ExtractionHandle {
            join_handle,
            cancel_flag,
        });
    }

    pub fn cancel_extraction(&mut self) {
        if let Some(handle) = &self.extraction {
            handle.cancel_flag.store(true, Ordering::Relaxed);
            self.cancel_requested = true;
            self.status_text =
                "Cancellation requested. Waiting for the active chunk to finish.".to_owned();
            self.overall_status = "Cancelling...".to_owned();
        }
    }

    pub fn drain_events(&mut self) {
        let Some(rx) = self.event_rx.clone() else {
            return;
        };

        while let Ok(event) = rx.try_recv() {
            self.handle_event(event);
        }
    }

    pub fn check_completion(&mut self) {
        let Some(handle) = self.extraction.as_ref() else {
            return;
        };
        if !handle.join_handle.is_finished() {
            return;
        }

        let handle = self.extraction.take().expect("checked extraction exists");
        let cancelled = handle.cancel_flag.load(Ordering::Relaxed);
        let joined = handle.join_handle.join();
        self.drain_events();
        self.event_rx = None;
        self.is_running = false;
        self.cancel_requested = false;

        match joined {
            Ok(Ok(result)) => {
                self.overall_status = "Extraction complete".to_owned();
                self.status_text.clear();
                self.result = Some(result);
                self.current_page = AppPage::Results;
            }
            Ok(Err(_err)) if cancelled => {
                self.status_text = "Extraction cancelled.".to_owned();
                self.error_message = None;
                self.current_page = AppPage::Config;
            }
            Ok(Err(err)) => {
                self.status_text.clear();
                self.error_message = Some(err.to_string());
                self.current_page = AppPage::Config;
            }
            Err(_) => {
                self.status_text.clear();
                self.error_message = Some("Extraction thread panicked".to_owned());
                self.current_page = AppPage::Config;
            }
        }
    }

    pub fn reset_to_config(&mut self) {
        if self.is_running {
            return;
        }
        self.clear_run_state();
        self.error_message = None;
        self.status_text.clear();
        self.overall_status = "Ready".to_owned();
        self.current_page = AppPage::Config;
    }

    pub fn set_audio_path(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        self.config.audio_path = path.display().to_string();
        if self.config.output_path.trim().is_empty() {
            self.config.output_path = default_output_path(path).display().to_string();
            self.config.output_format = ExtractFormat::Midi;
        }
    }

    pub fn set_output_path(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        self.config.output_path = path.display().to_string();
        if let Some(format) = infer_format_from_path(path) {
            self.config.output_format = format;
        }
    }

    pub fn apply_dropped_path(&mut self, path: &Path) {
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase());
        match extension.as_deref() {
            Some("gguf") => self.config.model_path = path.display().to_string(),
            Some("wav") => self.set_audio_path(path),
            Some("mid") | Some("midi") | Some("txt") | Some("csv") => self.set_output_path(path),
            _ => {}
        }
    }

    pub fn clear_error(&mut self) {
        self.error_message = None;
        self.status_text.clear();
    }

    fn validate_config(&self) -> Result<(), String> {
        let model_path = PathBuf::from(self.config.model_path.trim());
        if self.config.model_path.trim().is_empty() {
            return Err("Choose a GGUF model file.".to_owned());
        }
        if !model_path.is_file() {
            return Err(format!(
                "Model file does not exist: {}",
                model_path.display()
            ));
        }

        let audio_path = PathBuf::from(self.config.audio_path.trim());
        if self.config.audio_path.trim().is_empty() {
            return Err("Choose a WAV audio file.".to_owned());
        }
        if !audio_path.is_file() {
            return Err(format!(
                "Audio file does not exist: {}",
                audio_path.display()
            ));
        }

        if self.config.d3pm_nsteps <= 0 {
            return Err("D3PM steps must be greater than zero.".to_owned());
        }
        if self.config.max_chunk_seconds == 0 {
            return Err("Max chunk seconds must be greater than zero.".to_owned());
        }
        if !(0.0..=1.0).contains(&self.config.boundary_threshold) {
            return Err("Boundary threshold must be between 0.0 and 1.0.".to_owned());
        }
        if !(0.0..=1.0).contains(&self.config.note_threshold) {
            return Err("Note threshold must be between 0.0 and 1.0.".to_owned());
        }
        if self.config.boundary_radius < 0 {
            return Err("Boundary radius must be zero or greater.".to_owned());
        }

        Ok(())
    }

    fn clear_run_state(&mut self) {
        self.event_rx = None;
        self.extraction = None;
        self.result = None;
        self.overall_progress = None;
        self.chunk_progress.clear();
        self.stage_timings.clear();
        self.event_log.clear();
        self.run_started_at = None;
    }

    fn handle_event(&mut self, event: CoreEvent) {
        self.append_log(&event);

        match &event {
            CoreEvent::Status { stage, message } => {
                self.overall_status = friendly_status(stage, message);
                match *stage {
                    "extract_infer" => {
                        if let Some(total) = parse_chunk_count_from_message(message) {
                            self.init_chunks(total);
                        }
                    }
                    "chunk_infer" => {
                        let (index, total) = parse_chunk_position(message)
                            .unwrap_or_else(|| (self.chunk_progress.len(), 0));
                        if total > 0 {
                            self.init_chunks(total);
                        } else {
                            self.ensure_chunk(index, None);
                        }
                        if let Some(chunk) = self.chunk_progress.get_mut(index) {
                            chunk.label = format_chunk_status(message);
                        }
                    }
                    _ => {}
                }
            }
            CoreEvent::Progress {
                stage,
                current,
                total,
                detail,
            } if *stage == "d3pm_step" => {
                let index = detail.as_deref().and_then(parse_chunk_index).unwrap_or(0);
                self.ensure_chunk(index, None);
                if let Some(chunk) = self.chunk_progress.get_mut(index) {
                    chunk.d3pm_current = *current;
                    chunk.d3pm_total = *total;
                    chunk.status = ChunkStatus::Running;
                    self.overall_status = format!(
                        "{}: D3PM step {}/{}",
                        chunk.label, chunk.d3pm_current, chunk.d3pm_total
                    );
                }
                self.update_overall_completed();
            }
            CoreEvent::Timing {
                stage,
                elapsed,
                detail,
            } => {
                self.stage_timings.insert(
                    stage,
                    StageTiming {
                        elapsed: *elapsed,
                        completed: true,
                    },
                );

                if *stage == "chunk_infer" {
                    let index = detail.as_deref().and_then(parse_chunk_index).unwrap_or(0);
                    self.ensure_chunk(index, None);
                    if let Some(chunk) = self.chunk_progress.get_mut(index) {
                        if chunk.d3pm_total > 0 {
                            chunk.d3pm_current = chunk.d3pm_total;
                        }
                        chunk.status = ChunkStatus::Completed;
                    }
                    self.update_overall_completed();
                }
            }
            CoreEvent::ModelLoaded { elapsed, .. } => {
                self.stage_timings.insert(
                    "model_load",
                    StageTiming {
                        elapsed: *elapsed,
                        completed: true,
                    },
                );
                self.overall_status = "Model loaded".to_owned();
            }
            CoreEvent::Message { level, message } => {
                if matches!(level, NotificationLevel::Error) {
                    self.overall_status = message.clone();
                }
            }
            _ => {}
        }
    }

    fn init_chunks(&mut self, total: usize) {
        if self.chunk_progress.len() == total && self.overall_progress.is_some() {
            return;
        }

        let d3pm_total = usize::try_from(self.config.d3pm_nsteps.max(1)).unwrap_or(1);
        self.chunk_progress = (0..total)
            .map(|index| ChunkProgress {
                label: format!("chunk {}/{}", index + 1, total),
                d3pm_current: 0,
                d3pm_total,
                status: ChunkStatus::Pending,
            })
            .collect();
        self.overall_progress = Some((0, total));
    }

    fn ensure_chunk(&mut self, index: usize, total: Option<usize>) {
        if let Some(total) = total {
            if total > self.chunk_progress.len() {
                self.init_chunks(total);
                return;
            }
        }
        if index < self.chunk_progress.len() {
            return;
        }

        let target_len = index + 1;
        let d3pm_total = usize::try_from(self.config.d3pm_nsteps.max(1)).unwrap_or(1);
        let total = total.unwrap_or(target_len);
        while self.chunk_progress.len() < target_len {
            let current = self.chunk_progress.len();
            self.chunk_progress.push(ChunkProgress {
                label: format!("chunk {}/{}", current + 1, total),
                d3pm_current: 0,
                d3pm_total,
                status: ChunkStatus::Pending,
            });
        }
        self.update_overall_completed();
    }

    fn update_overall_completed(&mut self) {
        if self.chunk_progress.is_empty() {
            return;
        }
        let completed = self
            .chunk_progress
            .iter()
            .filter(|chunk| matches!(chunk.status, ChunkStatus::Completed))
            .count();
        self.overall_progress = Some((completed, self.chunk_progress.len()));
    }

    fn append_log(&mut self, event: &CoreEvent) {
        let elapsed = if let Some(started_at) = self.run_started_at {
            started_at.elapsed()
        } else {
            Duration::ZERO
        };

        self.event_log.push(GuiLogEntry {
            elapsed,
            level: GuiLogLevel::from_event(event),
            text: GuiNotifier::format_event(event),
        });

        let overflow = self.event_log.len().saturating_sub(self.max_log_entries);
        if overflow > 0 {
            self.event_log.drain(0..overflow);
        }
    }
}

impl Default for ExtractConfig {
    fn default() -> Self {
        let params = InferParams::default();
        Self {
            model_path: String::new(),
            audio_path: String::new(),
            output_path: String::new(),
            output_format: ExtractFormat::Midi,
            device: ExtractDevice::Auto,
            gpu_selector: GpuSelector::default(),
            selected_gpu_index: None,
            d3pm_nsteps: params.d3pm_nsteps,
            seed: params.seed,
            chunk_parallelism: ChunkParallelism::Auto,
            max_chunk_seconds: DEFAULT_MAX_CHUNK_SECONDS,
            language: params.language,
            boundary_threshold: params.boundary_threshold,
            note_threshold: params.note_threshold,
            boundary_radius: params.boundary_radius,
        }
    }
}

impl ExtractConfig {
    fn to_request(&self) -> ExtractRequest {
        ExtractRequest {
            model_path: PathBuf::from(self.model_path.trim()),
            input_path: PathBuf::from(self.audio_path.trim()),
            output: self.output_request(),
            device: self.device,
            gpu: if self.device == ExtractDevice::Cpu {
                GpuSelector::default()
            } else {
                self.gpu_selector.clone()
            },
            infer_params: InferParams {
                language: self.language,
                d3pm_nsteps: self.d3pm_nsteps,
                boundary_threshold: self.boundary_threshold,
                note_threshold: self.note_threshold,
                boundary_radius: self.boundary_radius,
                seed: self.seed,
                ..InferParams::default()
            },
            chunk_parallelism: self.chunk_parallelism,
            max_chunk_seconds: self.max_chunk_seconds,
        }
    }

    fn output_request(&self) -> Option<ExtractOutputRequest> {
        let path = self.output_path.trim();
        (!path.is_empty()).then(|| ExtractOutputRequest {
            path: PathBuf::from(path),
            format: Some(self.output_format),
            midi_options: MidiWriteOptions::default(),
            text_options: TextWriteOptions::default(),
        })
    }
}

impl GuiLogLevel {
    fn from_event(event: &CoreEvent) -> Self {
        match event {
            CoreEvent::Message { level, .. } => match level {
                NotificationLevel::Trace => Self::Trace,
                NotificationLevel::Debug => Self::Debug,
                NotificationLevel::Info => Self::Info,
                NotificationLevel::Warn => Self::Warn,
                NotificationLevel::Error => Self::Error,
            },
            CoreEvent::Progress { .. } | CoreEvent::Timing { .. } => Self::Debug,
            _ => Self::Info,
        }
    }
}

fn run_extraction(
    config: ExtractConfig,
    notifier: GuiNotifier,
    cancel: Arc<AtomicBool>,
) -> game_service::Result<ExtractResult> {
    if cancel.load(Ordering::Relaxed) {
        return Err(Error::message("Extraction cancelled"));
    }

    let request = config.to_request();
    let result = extract_with_notifier(&request, &notifier)?;

    if cancel.load(Ordering::Relaxed) {
        return Err(Error::message("Extraction cancelled"));
    }

    Ok(result)
}

#[cfg(feature = "gpu")]
fn list_gpu_adapters() -> Vec<GpuAdapterChoice> {
    game_core::GpuDevice::available_adapters()
        .into_iter()
        .map(|info| GpuAdapterChoice {
            name: info.name,
            backend: format!("{:?}", info.backend).to_ascii_lowercase(),
            device_type: format!("{:?}", info.device_type).to_ascii_lowercase(),
            vendor_id: info.vendor,
            device_id: info.device,
        })
        .collect()
}

fn friendly_status(stage: &str, message: &str) -> String {
    match stage {
        "model_load" => "Loading model...".to_owned(),
        "audio_prepare" => "Preparing audio...".to_owned(),
        "silence_slice" => "Slicing audio on silence...".to_owned(),
        "long_chunk_split" => "Splitting long chunks...".to_owned(),
        "mel_setup" => "Initializing mel extractor...".to_owned(),
        "extract_infer" => message.to_owned(),
        "output_write" => "Writing output...".to_owned(),
        "chunk_infer" => format_chunk_status(message),
        _ => message.to_owned(),
    }
}

fn default_output_path(audio_path: &Path) -> PathBuf {
    let mut output = audio_path.to_path_buf();
    output.set_extension("mid");
    output
}

fn infer_format_from_path(path: &Path) -> Option<ExtractFormat> {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("mid" | "midi") => Some(ExtractFormat::Midi),
        Some("txt") => Some(ExtractFormat::Txt),
        Some("csv") => Some(ExtractFormat::Csv),
        _ => None,
    }
}

fn parse_chunk_count_from_message(message: &str) -> Option<usize> {
    let parts: Vec<&str> = message.split_whitespace().collect();
    for (index, part) in parts.iter().enumerate() {
        if part.starts_with("chunk") && index > 0 {
            return parts[index - 1].parse().ok();
        }
    }
    None
}

fn parse_chunk_index(detail: &str) -> Option<usize> {
    parse_chunk_position(detail).map(|(index, _)| index)
}

fn parse_chunk_position(detail: &str) -> Option<(usize, usize)> {
    let rest = detail.strip_prefix("chunk ")?;
    let digit_end = rest.find(|c: char| !c.is_ascii_digit())?;
    let index = rest[..digit_end].parse::<usize>().ok()?.saturating_sub(1);
    let total = rest[digit_end..]
        .strip_prefix('/')
        .and_then(|rest| {
            let total_end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest[..total_end].parse::<usize>().ok()
        })
        .unwrap_or(0);
    Some((index, total))
}

fn format_chunk_status(message: &str) -> String {
    if let Some(pos) = message.find(": infer start ") {
        let chunk_id = &message[..pos];
        let rest = &message[pos + ": infer start ".len()..];
        let duration = rest
            .split_whitespace()
            .find_map(|part| part.strip_prefix("duration="))
            .unwrap_or_default();
        if duration.is_empty() {
            chunk_id.to_owned()
        } else {
            format!("{chunk_id} ({duration})")
        }
    } else {
        message.to_owned()
    }
}

pub fn format_duration(duration: Duration) -> String {
    format!("{:.3}s", duration.as_secs_f64())
}

pub fn format_count(value: usize) -> String {
    let input = value.to_string();
    let mut output = String::with_capacity(input.len() + input.len() / 3);
    for (index, ch) in input.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            output.push(',');
        }
        output.push(ch);
    }
    output.chars().rev().collect()
}

pub fn backend_name(backend: game_service::Backend) -> &'static str {
    match backend {
        game_service::Backend::Cpu => "CPU",
        game_service::Backend::Gpu => "GPU",
    }
}

pub fn output_format_name(format: ExtractFormat) -> &'static str {
    match format {
        ExtractFormat::Midi => "MIDI",
        ExtractFormat::Txt => "TXT",
        ExtractFormat::Csv => "CSV",
    }
}

pub fn device_name(device: ExtractDevice) -> &'static str {
    match device {
        ExtractDevice::Auto => "Auto",
        ExtractDevice::Cpu => "CPU",
        ExtractDevice::Gpu => "GPU",
    }
}

pub fn chunk_parallelism_name(value: ChunkParallelism) -> &'static str {
    match value {
        ChunkParallelism::Auto => "Auto",
        ChunkParallelism::On => "On",
        ChunkParallelism::Off => "Off",
    }
}
