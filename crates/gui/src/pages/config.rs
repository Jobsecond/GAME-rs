use std::path::{Path, PathBuf};

use game_service::{ChunkParallelism, ExtractDevice, ExtractFormat};

use crate::state::{
    AppState, ThemeChoice, chunk_parallelism_name, device_name, output_format_name,
};

use super::{control_frame, error_frame, page_title, primary_button, section_frame, section_title};

pub fn render(ui: &mut egui::Ui, state: &mut AppState, ctx: &egui::Context) {
    egui::Panel::bottom("config_action_bar")
        .exact_size(58.0)
        .frame(
            egui::Frame::NONE
                .fill(ui.visuals().panel_fill)
                .inner_margin(egui::Margin::symmetric(0, 10)),
        )
        .show_inside(ui, |ui| {
            render_action_bar(ui, state, ctx);
        });

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("config_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    render_config_body(ui, state, ctx);
                    ui.add_space(8.0);
                });
        });
}

fn render_config_body(ui: &mut egui::Ui, state: &mut AppState, ctx: &egui::Context) {
    ui.horizontal(|ui| {
        page_title(ui, "Configuration");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            theme_selector(ui, state, ctx);
        });
    });
    ui.add_space(14.0);

    if state.cjk_font_missing && !state.font_notice_dismissed {
        error_frame(ui).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    "No CJK font was found — non-Latin characters in file paths may not render.",
                );
                if ui.button("Dismiss").clicked() {
                    state.font_notice_dismissed = true;
                }
            });
        });
        ui.add_space(12.0);
    }

    if let Some(message) = state.error_message.clone() {
        error_frame(ui).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.colored_label(ui.visuals().error_fg_color, message);
                if ui.button("Dismiss").clicked() {
                    state.clear_error();
                }
            });
        });
        ui.add_space(12.0);
    } else if !state.status_text.is_empty() {
        ui.label(&state.status_text);
        ui.add_space(12.0);
    }

    section_frame(ui).show(ui, |ui| {
        ui.set_width(ui.available_width());
        let path_width =
            (ui.available_width() - LABEL_WIDTH - BROWSE_WIDTH - 36.0).clamp(420.0, 820.0);
        path_row(
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
        )
        .picked
        {
            state.set_audio_path(path);
        }

        let output_path_default = state.config.output_path.clone();
        let audio_path_default = state.config.audio_path.clone();
        let output_format_default = state.config.output_format;
        let output_row = path_row(
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
        );
        if let Some(path) = output_row.picked {
            state.set_output_path(path);
        } else if output_row.edited {
            // Keep the Format combo in sync as the user types/pastes a path, the
            // same way Browse and drag-drop already do. Unknown or partial
            // extensions return None and leave the combo untouched.
            if let Some(format) =
                game_service::infer_extract_format(Path::new(state.config.output_path.trim()))
            {
                state.config.output_format = format;
            }
        }

        ui.horizontal(|ui| {
            row_label(ui, "Format");
            egui::ComboBox::from_id_salt("output_format")
                .selected_text(output_format_name(state.config.output_format))
                .width(160.0)
                .height(ROW_HEIGHT)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.config.output_format,
                        ExtractFormat::Midi,
                        "MIDI",
                    );
                    ui.selectable_value(&mut state.config.output_format, ExtractFormat::Txt, "TXT");
                    ui.selectable_value(&mut state.config.output_format, ExtractFormat::Csv, "CSV");
                });
        });

        if let Some(warning) = output_format_mismatch(state) {
            ui.add_space(6.0);
            ui.colored_label(egui::Color32::from_rgb(196, 143, 0), warning);
        }
    });

    ui.add_space(16.0);
    render_device_section(ui, state, ctx);
    ui.add_space(16.0);
    render_inference_section(ui, state);
}

fn render_action_bar(ui: &mut egui::Ui, state: &mut AppState, ctx: &egui::Context) {
    let blocking = start_blocking_reason(state);
    ui.horizontal(|ui| {
        let start = ui.add_enabled(
            !state.is_running && blocking.is_none(),
            primary_button(ui, "Start Extraction").min_size(egui::vec2(178.0, 34.0)),
        );
        if start.clicked() {
            state.start_extraction(ctx);
        }
        match blocking {
            Some(reason) => {
                ui.label(egui::RichText::new(reason).color(egui::Color32::from_rgb(196, 143, 0)));
            }
            None => {
                ui.label(
                    egui::RichText::new(
                        "Drop .gguf, .wav, .mid, .txt, or .csv files onto this window to fill paths.",
                    )
                    .color(ui.visuals().weak_text_color()),
                );
            }
        }
    });
}

/// Lightweight, allocation-free reason the run can't start yet, used to disable
/// the Start button and show inline guidance before the user clicks. Full
/// validation (including file existence) still runs in `start_extraction`.
fn start_blocking_reason(state: &AppState) -> Option<&'static str> {
    if state.config.model_path.trim().is_empty() {
        Some("Choose a GGUF model file to begin.")
    } else if state.config.audio_path.trim().is_empty() {
        Some("Choose a WAV audio file to begin.")
    } else {
        None
    }
}

/// Light/Dark/System theme picker. Applying it immediately via `set_theme`
/// re-resolves egui's active visuals; the choice is persisted with the rest of
/// the settings.
fn theme_selector(ui: &mut egui::Ui, state: &mut AppState, ctx: &egui::Context) {
    let mut theme = state.theme;
    egui::ComboBox::from_id_salt("theme_choice")
        .selected_text(theme.label())
        .show_ui(ui, |ui| {
            for choice in [ThemeChoice::System, ThemeChoice::Light, ThemeChoice::Dark] {
                ui.selectable_value(&mut theme, choice, choice.label());
            }
        });
    ui.label(egui::RichText::new("Theme").color(ui.visuals().weak_text_color()));
    if theme != state.theme {
        state.theme = theme;
        ctx.set_theme(theme.to_preference());
    }
}

const LABEL_WIDTH: f32 = 112.0;
const BROWSE_WIDTH: f32 = 78.0;
const ROW_HEIGHT: f32 = 30.0;

/// Result of rendering a file-path row: a path chosen via Browse, and whether
/// the text field was edited this frame (typed/pasted).
struct PathRowOutcome {
    picked: Option<PathBuf>,
    edited: bool,
}

fn path_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    hint: &str,
    path_width: f32,
    picker: impl FnOnce() -> Option<PathBuf>,
) -> PathRowOutcome {
    let mut picked = None;
    let mut edited = false;
    ui.horizontal(|ui| {
        row_label(ui, label);
        let response = ui.add_sized(
            [path_width, ROW_HEIGHT],
            egui::TextEdit::singleline(value)
                .hint_text(hint)
                .background_color(ui.visuals().extreme_bg_color)
                .frame(control_frame(ui))
                .margin(egui::Margin::symmetric(4, 4)),
        );
        edited = response.changed();
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
    PathRowOutcome { picked, edited }
}

/// Warns when the selected output format disagrees with the output file's
/// extension — the format combo is authoritative, so without this the user
/// could silently write (e.g.) MIDI bytes into a `.csv` file.
fn output_format_mismatch(state: &AppState) -> Option<String> {
    let path = state.config.output_path.trim();
    if path.is_empty() {
        return None;
    }
    let ext_format = game_service::infer_extract_format(Path::new(path))?;
    (ext_format != state.config.output_format).then(|| {
        format!(
            "Output file extension looks like {} but the selected format is {}. \
             {} data will be written to this file.",
            output_format_name(ext_format),
            output_format_name(state.config.output_format),
            output_format_name(state.config.output_format),
        )
    })
}

fn row_label(ui: &mut egui::Ui, text: &str) {
    ui.add_sized(
        [LABEL_WIDTH, ROW_HEIGHT],
        egui::Label::new(egui::RichText::new(text).color(ui.visuals().weak_text_color()))
            .selectable(false),
    );
}

fn render_device_section(ui: &mut egui::Ui, state: &mut AppState, _ctx: &egui::Context) {
    section_frame(ui).show(ui, |ui| {
        ui.set_width(ui.available_width());
        section_title(ui, "Device");
        ui.add_space(8.0);
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
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("GPU").color(ui.visuals().weak_text_color()));
                    let selected = state
                        .config
                        .selected_gpu_index
                        .and_then(|index| state.gpu_adapters.get(index))
                        .map(adapter_label)
                        .unwrap_or_else(|| "Any available GPU".to_owned());

                    egui::ComboBox::from_id_salt("gpu_adapter")
                        .selected_text(selected)
                        .width(360.0)
                        .height(ROW_HEIGHT)
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

                if let Some(error) = &state.gpu_adapter_error {
                    ui.colored_label(
                        egui::Color32::from_rgb(196, 43, 28),
                        format!("Failed to enumerate GPU adapters: {error}"),
                    );
                } else if !state.gpu_adapters_loading && state.gpu_adapters.is_empty() {
                    ui.label("No GPU adapters were reported by wgpu.");
                }
            }
        }
    });
}

fn render_inference_section(ui: &mut egui::Ui, state: &mut AppState) {
    section_frame(ui).show(ui, |ui| {
        ui.set_width(ui.available_width());
        section_title(ui, "Inference Parameters");
        ui.add_space(10.0);

        egui::Grid::new("inference_grid")
            .num_columns(2)
            .spacing([24.0, 10.0])
            .min_col_width(168.0)
            .show(ui, |ui| {
                grid_label(ui, "D3PM steps");
                ui.add_sized(
                    [110.0, ROW_HEIGHT],
                    egui::DragValue::new(&mut state.config.d3pm_nsteps).range(1..=256),
                );
                ui.end_row();

                grid_label(ui, "Seed");
                ui.add_sized(
                    [110.0, ROW_HEIGHT],
                    egui::DragValue::new(&mut state.config.seed),
                );
                ui.end_row();

                grid_label(ui, "Language ID");
                ui.add_sized(
                    [110.0, ROW_HEIGHT],
                    egui::DragValue::new(&mut state.config.language),
                );
                ui.end_row();

                grid_label(ui, "Chunk parallelism");
                egui::ComboBox::from_id_salt("chunk_parallelism")
                    .selected_text(chunk_parallelism_name(state.config.chunk_parallelism))
                    .width(160.0)
                    .height(ROW_HEIGHT)
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

                grid_label(ui, "Max chunk seconds");
                ui.add_sized(
                    [110.0, ROW_HEIGHT],
                    egui::DragValue::new(&mut state.config.max_chunk_seconds).range(1..=7200),
                );
                ui.end_row();

                grid_label(ui, "Boundary threshold");
                ui.add_sized(
                    [360.0, ROW_HEIGHT],
                    egui::Slider::new(&mut state.config.boundary_threshold, 0.0..=1.0),
                );
                ui.end_row();

                grid_label(ui, "Note threshold");
                ui.add_sized(
                    [360.0, ROW_HEIGHT],
                    egui::Slider::new(&mut state.config.note_threshold, 0.0..=1.0),
                );
                ui.end_row();

                grid_label(ui, "Boundary radius");
                ui.add_sized(
                    [110.0, ROW_HEIGHT],
                    egui::DragValue::new(&mut state.config.boundary_radius).range(0..=64),
                );
                ui.end_row();
            });
    });
}

fn grid_label(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).color(ui.visuals().weak_text_color()));
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
