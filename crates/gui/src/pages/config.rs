use std::path::{Path, PathBuf};

use game_service::{ChunkParallelism, ExtractDevice, ExtractFormat};

use crate::state::{AppState, chunk_parallelism_name, device_name, output_format_name};

pub fn render(ui: &mut egui::Ui, state: &mut AppState, ctx: &egui::Context) {
    apply_dropped_files(state, ctx);

    ui.heading("Configuration");
    ui.add_space(8.0);

    if let Some(message) = state.error_message.clone() {
        egui::Frame::group(ui.style())
            .fill(ui.visuals().error_fg_color.linear_multiply(0.08))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(ui.visuals().error_fg_color, message);
                    if ui.button("Dismiss").clicked() {
                        state.clear_error();
                    }
                });
            });
        ui.add_space(8.0);
    } else if !state.status_text.is_empty() {
        ui.label(&state.status_text);
        ui.add_space(8.0);
    }

    let path_width = (ui.available_width() - LABEL_WIDTH - BROWSE_WIDTH - 36.0).clamp(420.0, 760.0);
    let _ = path_row(
        ui,
        "Model (.gguf)",
        &mut state.config.model_path,
        "Path to model.gguf",
        path_width,
        || {
            rfd::FileDialog::new()
                .add_filter("GGUF model", &["gguf"])
                .pick_file()
        },
    );
    if let Some(path) = path_row(
        ui,
        "Audio (.wav)",
        &mut state.config.audio_path,
        "Path to input.wav",
        path_width,
        || {
            rfd::FileDialog::new()
                .add_filter("WAV audio", &["wav"])
                .pick_file()
        },
    ) {
        state.set_audio_path(path);
    }

    let output_path_default = state.config.output_path.clone();
    let audio_path_default = state.config.audio_path.clone();
    let output_format_default = state.config.output_format;
    if let Some(path) = path_row(
        ui,
        "Output",
        &mut state.config.output_path,
        "Path to output.mid",
        path_width,
        || {
            output_dialog_path(
                &output_path_default,
                &audio_path_default,
                output_format_default,
            )
        },
    ) {
        state.set_output_path(path);
    }

    ui.horizontal(|ui| {
        row_label(ui, "Format");
        egui::ComboBox::from_id_salt("output_format")
            .selected_text(output_format_name(state.config.output_format))
            .width(160.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut state.config.output_format, ExtractFormat::Midi, "MIDI");
                ui.selectable_value(&mut state.config.output_format, ExtractFormat::Txt, "TXT");
                ui.selectable_value(&mut state.config.output_format, ExtractFormat::Csv, "CSV");
            });
    });

    ui.add_space(14.0);
    render_device_section(ui, state, ctx);
    ui.add_space(14.0);
    render_inference_section(ui, state);
    ui.add_space(18.0);

    ui.horizontal(|ui| {
        let start = ui.add_enabled(
            !state.is_running,
            egui::Button::new("Start Extraction").min_size(egui::vec2(170.0, 34.0)),
        );
        if start.clicked() {
            state.start_extraction(ctx);
        }
        ui.label("Drop .gguf, .wav, .mid, .txt, or .csv files onto this window to fill paths.");
    });
}

const LABEL_WIDTH: f32 = 112.0;
const BROWSE_WIDTH: f32 = 78.0;
const ROW_HEIGHT: f32 = 30.0;

fn path_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    hint: &str,
    path_width: f32,
    picker: impl FnOnce() -> Option<PathBuf>,
) -> Option<PathBuf> {
    let mut picked = None;
    ui.horizontal(|ui| {
        row_label(ui, label);
        ui.add_sized(
            [path_width, ROW_HEIGHT],
            egui::TextEdit::singleline(value).hint_text(hint),
        );
        if ui
            .add_sized([BROWSE_WIDTH, ROW_HEIGHT], egui::Button::new("Browse"))
            .clicked()
        {
            picked = picker();
            if let Some(path) = &picked {
                *value = path.display().to_string();
            }
        }
    });
    picked
}

fn row_label(ui: &mut egui::Ui, text: &str) {
    ui.add_sized(
        [LABEL_WIDTH, ROW_HEIGHT],
        egui::Label::new(text).selectable(false),
    );
}

fn render_device_section(ui: &mut egui::Ui, state: &mut AppState, _ctx: &egui::Context) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.heading("Device");
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.radio_value(
                &mut state.config.device,
                ExtractDevice::Auto,
                device_name(ExtractDevice::Auto),
            );
            ui.radio_value(
                &mut state.config.device,
                ExtractDevice::Cpu,
                device_name(ExtractDevice::Cpu),
            );
            #[cfg(feature = "gpu")]
            ui.radio_value(
                &mut state.config.device,
                ExtractDevice::Gpu,
                device_name(ExtractDevice::Gpu),
            );
            #[cfg(not(feature = "gpu"))]
            ui.label("GPU option requires --features gpu,gui");
        });

        #[cfg(feature = "gpu")]
        {
            if state.config.device != ExtractDevice::Cpu {
                state.ensure_gpu_adapter_refresh_started(_ctx);
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("GPU");
                    let selected = state
                        .config
                        .selected_gpu_index
                        .and_then(|index| state.gpu_adapters.get(index))
                        .map(adapter_label)
                        .unwrap_or_else(|| "Any available GPU".to_owned());

                    egui::ComboBox::from_id_salt("gpu_adapter")
                        .selected_text(selected)
                        .width(360.0)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    state.config.selected_gpu_index.is_none(),
                                    "Any available GPU",
                                )
                                .clicked()
                            {
                                state.config.selected_gpu_index = None;
                                state.config.gpu_selector = game_service::GpuSelector::default();
                            }

                            if state.gpu_adapters_loading {
                                ui.label("Loading GPU adapters...");
                            }

                            for (index, adapter) in state.gpu_adapters.iter().enumerate() {
                                if ui
                                    .selectable_label(
                                        state.config.selected_gpu_index == Some(index),
                                        adapter_label(adapter),
                                    )
                                    .clicked()
                                {
                                    state.config.selected_gpu_index = Some(index);
                                    state.config.gpu_selector = game_service::GpuSelector {
                                        name_substring: Some(adapter.name.clone()),
                                        vendor_id: Some(adapter.vendor_id),
                                        device_id: Some(adapter.device_id),
                                    };
                                }
                            }
                        });

                    if ui
                        .add_enabled(!state.gpu_adapters_loading, egui::Button::new("Refresh"))
                        .clicked()
                    {
                        state.refresh_gpu_adapters(_ctx);
                    }
                    if state.gpu_adapters_loading {
                        ui.add(egui::Spinner::new().size(16.0));
                    }
                });

                if !state.gpu_adapters_loading && state.gpu_adapters.is_empty() {
                    ui.label("No GPU adapters were reported by wgpu.");
                }
            }
        }
    });
}

fn render_inference_section(ui: &mut egui::Ui, state: &mut AppState) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.heading("Inference Parameters");
        ui.add_space(4.0);

        egui::Grid::new("inference_grid")
            .num_columns(2)
            .spacing([24.0, 10.0])
            .min_col_width(168.0)
            .show(ui, |ui| {
                ui.label("D3PM steps");
                ui.add_sized(
                    [110.0, ROW_HEIGHT],
                    egui::DragValue::new(&mut state.config.d3pm_nsteps).range(1..=256),
                );
                ui.end_row();

                ui.label("Seed");
                ui.add_sized(
                    [110.0, ROW_HEIGHT],
                    egui::DragValue::new(&mut state.config.seed),
                );
                ui.end_row();

                ui.label("Language ID");
                ui.add_sized(
                    [110.0, ROW_HEIGHT],
                    egui::DragValue::new(&mut state.config.language),
                );
                ui.end_row();

                ui.label("Chunk parallelism");
                egui::ComboBox::from_id_salt("chunk_parallelism")
                    .selected_text(chunk_parallelism_name(state.config.chunk_parallelism))
                    .width(160.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.config.chunk_parallelism,
                            ChunkParallelism::Auto,
                            "Auto",
                        );
                        ui.selectable_value(
                            &mut state.config.chunk_parallelism,
                            ChunkParallelism::On,
                            "On",
                        );
                        ui.selectable_value(
                            &mut state.config.chunk_parallelism,
                            ChunkParallelism::Off,
                            "Off",
                        );
                    });
                ui.end_row();

                ui.label("Max chunk seconds");
                ui.add_sized(
                    [110.0, ROW_HEIGHT],
                    egui::DragValue::new(&mut state.config.max_chunk_seconds).range(1..=7200),
                );
                ui.end_row();

                ui.label("Boundary threshold");
                ui.add_sized(
                    [360.0, ROW_HEIGHT],
                    egui::Slider::new(&mut state.config.boundary_threshold, 0.0..=1.0),
                );
                ui.end_row();

                ui.label("Note threshold");
                ui.add_sized(
                    [360.0, ROW_HEIGHT],
                    egui::Slider::new(&mut state.config.note_threshold, 0.0..=1.0),
                );
                ui.end_row();

                ui.label("Boundary radius");
                ui.add_sized(
                    [110.0, ROW_HEIGHT],
                    egui::DragValue::new(&mut state.config.boundary_radius).range(0..=64),
                );
                ui.end_row();
            });
    });
}

fn apply_dropped_files(state: &mut AppState, ctx: &egui::Context) {
    let dropped_files = ctx.input(|input| input.raw.dropped_files.clone());
    for file in dropped_files {
        if let Some(path) = file.path {
            state.apply_dropped_path(&path);
        }
    }
}

fn output_dialog_path(
    output_path: &str,
    audio_path: &str,
    output_format: ExtractFormat,
) -> Option<PathBuf> {
    let mut dialog = rfd::FileDialog::new()
        .add_filter("MIDI", &["mid", "midi"])
        .add_filter("Text", &["txt"])
        .add_filter("CSV", &["csv"]);

    if let Some(parent) = default_output_parent(output_path, audio_path) {
        dialog = dialog.set_directory(parent);
    }
    if let Some(file_name) = default_output_file_name(output_path, audio_path, output_format) {
        dialog = dialog.set_file_name(file_name);
    }

    dialog.save_file()
}

fn default_output_parent(output_path: &str, audio_path: &str) -> Option<PathBuf> {
    let output = output_path.trim();
    if !output.is_empty() {
        return Path::new(output).parent().map(Path::to_path_buf);
    }

    let audio = audio_path.trim();
    (!audio.is_empty()).then(|| Path::new(audio).parent().map(Path::to_path_buf))?
}

fn default_output_file_name(
    output_path: &str,
    audio_path: &str,
    output_format: ExtractFormat,
) -> Option<String> {
    let output = output_path.trim();
    if !output.is_empty() {
        return Path::new(output)
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned);
    }

    let audio = audio_path.trim();
    if audio.is_empty() {
        return Some("output.mid".to_owned());
    }

    let mut path = PathBuf::from(audio);
    path.set_extension(match output_format {
        ExtractFormat::Midi => "mid",
        ExtractFormat::Txt => "txt",
        ExtractFormat::Csv => "csv",
    });
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
}

#[cfg(feature = "gpu")]
fn adapter_label(adapter: &crate::state::GpuAdapterChoice) -> String {
    format!(
        "{} [{} {}, vendor=0x{:04x}, device=0x{:04x}]",
        adapter.name, adapter.backend, adapter.device_type, adapter.vendor_id, adapter.device_id
    )
}
