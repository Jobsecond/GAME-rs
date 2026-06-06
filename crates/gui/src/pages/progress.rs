use std::time::Duration;

use crate::state::{AppState, ChunkStatus, GuiLogLevel, StageTiming, format_duration};

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    ui.ctx().request_repaint_after(Duration::from_millis(100));

    ui.heading(if state.cancel_requested {
        "Cancelling extraction"
    } else {
        "Extraction in progress"
    });
    ui.add_space(8.0);

    render_overall(ui, state);
    ui.add_space(12.0);
    render_chunks(ui, state);
    ui.add_space(12.0);
    render_timings(ui, state);
    ui.add_space(12.0);
    render_log(ui, state);
    ui.add_space(12.0);

    let cancel = ui.add_enabled(
        state.is_running && !state.cancel_requested,
        egui::Button::new("Cancel").min_size(egui::vec2(110.0, 32.0)),
    );
    if cancel.clicked() {
        state.cancel_extraction();
    }
}

fn render_overall(ui: &mut egui::Ui, state: &AppState) {
    if let Some((current, total)) = state.overall_progress {
        let fraction = if total == 0 {
            0.0
        } else {
            current as f32 / total as f32
        };
        ui.add(
            egui::ProgressBar::new(fraction)
                .text(format!("{current}/{total} chunks"))
                .desired_width(f32::INFINITY)
                .animate(state.is_running && !state.cancel_requested),
        );
    } else {
        ui.add(
            egui::ProgressBar::new(0.0)
                .text("Starting")
                .desired_width(f32::INFINITY)
                .animate(state.is_running && !state.cancel_requested),
        );
    }

    ui.add_space(6.0);
    ui.label(&state.overall_status);
    if !state.status_text.is_empty() {
        ui.label(&state.status_text);
    }
}

fn render_chunks(ui: &mut egui::Ui, state: &AppState) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.heading("Chunks");
        ui.add_space(4.0);

        if state.chunk_progress.is_empty() {
            ui.label("Waiting for chunk discovery...");
            return;
        }

        for chunk in &state.chunk_progress {
            let fraction = if chunk.d3pm_total == 0 {
                0.0
            } else {
                chunk.d3pm_current as f32 / chunk.d3pm_total as f32
            };
            let text = match &chunk.status {
                ChunkStatus::Pending => format!("{}: pending", chunk.label),
                ChunkStatus::Running => format!(
                    "{}: D3PM step {}/{}",
                    chunk.label, chunk.d3pm_current, chunk.d3pm_total
                ),
                ChunkStatus::Completed => format!("{}: complete", chunk.label),
                ChunkStatus::Failed(message) => format!("{}: {message}", chunk.label),
            };

            let mut bar = egui::ProgressBar::new(fraction)
                .text(text)
                .desired_width(f32::INFINITY)
                .animate(matches!(chunk.status, ChunkStatus::Running));

            match &chunk.status {
                ChunkStatus::Completed => {
                    bar = bar.fill(egui::Color32::from_rgb(66, 150, 92));
                }
                ChunkStatus::Failed(_) => {
                    bar = bar.fill(ui.visuals().error_fg_color);
                }
                _ => {}
            }

            ui.add(bar);
        }
    });
}

fn render_timings(ui: &mut egui::Ui, state: &AppState) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.heading("Stage Timings");
        ui.add_space(4.0);

        egui::Grid::new("stage_timings")
            .num_columns(3)
            .spacing([18.0, 6.0])
            .show(ui, |ui| {
                for (stage, label) in STAGE_ORDER {
                    ui.label(*label);
                    match state.stage_timings.get(stage) {
                        Some(StageTiming { elapsed, completed }) => {
                            ui.label(format_duration(*elapsed));
                            ui.label(if *completed { "done" } else { "running" });
                        }
                        None => {
                            ui.label("pending");
                            ui.label("");
                        }
                    }
                    ui.end_row();
                }
            });
    });
}

fn render_log(ui: &mut egui::Ui, state: &AppState) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.heading("Event Log");
        ui.add_space(4.0);
        egui::ScrollArea::vertical()
            .id_salt("event_log")
            .stick_to_bottom(true)
            .max_height(210.0)
            .show(ui, |ui| {
                if state.event_log.is_empty() {
                    ui.label("Waiting for events...");
                }
                for entry in &state.event_log {
                    let color = log_color(ui, entry.level);
                    ui.colored_label(
                        color,
                        format!("[+{}] {}", format_duration(entry.elapsed), entry.text),
                    );
                }
            });
    });
}

fn log_color(ui: &egui::Ui, level: GuiLogLevel) -> egui::Color32 {
    match level {
        GuiLogLevel::Trace | GuiLogLevel::Debug => ui.visuals().weak_text_color(),
        GuiLogLevel::Info => ui.visuals().text_color(),
        GuiLogLevel::Warn => egui::Color32::from_rgb(196, 143, 0),
        GuiLogLevel::Error => ui.visuals().error_fg_color,
    }
}

const STAGE_ORDER: &[(&str, &str)] = &[
    ("model_load", "Model load"),
    ("audio_prepare", "Audio prep"),
    ("silence_slice", "Silence slice"),
    ("long_chunk_split", "Long chunk split"),
    ("mel_setup", "Mel setup"),
    ("extract_infer", "Inference"),
    ("output_write", "Output write"),
];
