use std::time::Duration;

use crate::state::{
    AppState, ChunkProgress, ChunkStatus, GuiLogLevel, StageTiming, format_duration,
};

use super::{
    ACCENT, STROKE, TEXT_PRIMARY, TEXT_SECONDARY, page_title, section_frame, section_title,
};

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    ui.ctx().request_repaint_after(Duration::from_millis(100));

    page_title(
        ui,
        if state.cancel_requested {
            "Cancelling extraction"
        } else {
            "Extraction in progress"
        },
    );
    ui.add_space(14.0);

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
    let fraction = overall_fraction(state);
    let label = match state.overall_progress {
        Some((current, total)) => {
            let percent = (fraction * 100.0).round() as usize;
            format!("{percent}% - {current}/{total} chunks complete")
        }
        None => "Starting".to_owned(),
    };

    progress_header(ui, "Overall", &label, false);
    ui.add_space(4.0);
    progress_track(ui, fraction, ACCENT, 8.0);

    ui.add_space(8.0);
    ui.label(egui::RichText::new(&state.overall_status).color(TEXT_SECONDARY));
    if !state.status_text.is_empty() {
        ui.label(egui::RichText::new(&state.status_text).color(TEXT_SECONDARY));
    }
}

fn render_chunks(ui: &mut egui::Ui, state: &AppState) {
    section_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        section_title(ui, "Chunks");
        ui.add_space(8.0);

        if state.chunk_progress.is_empty() {
            ui.label("Waiting for chunk discovery...");
            return;
        }

        egui::ScrollArea::vertical()
            .id_salt("chunk_progress")
            .max_height(320.0)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                for (index, chunk) in state.chunk_progress.iter().enumerate() {
                    let fraction = chunk_fraction(chunk);
                    let (detail, fill) = chunk_detail(ui, chunk, fraction);
                    let active = matches!(chunk.status, ChunkStatus::Running);

                    progress_header(ui, &chunk.label, &detail, active);
                    ui.add_space(4.0);
                    progress_track(ui, fraction, fill, 6.0);
                    if index + 1 < state.chunk_progress.len() {
                        ui.add_space(10.0);
                    }
                }
            });
    });
}

fn progress_header(ui: &mut egui::Ui, label: &str, detail: &str, active: bool) {
    ui.horizontal(|ui| {
        ui.set_width(ui.available_width());
        if active {
            ui.add(egui::Spinner::new().size(14.0));
            ui.add_space(2.0);
        }
        ui.label(egui::RichText::new(label).color(TEXT_PRIMARY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(detail).color(TEXT_SECONDARY));
        });
    });
}

fn progress_track(
    ui: &mut egui::Ui,
    fraction: f32,
    fill: egui::Color32,
    height: f32,
) -> egui::Response {
    let desired_size = egui::vec2(ui.available_width(), height);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let fraction = fraction.clamp(0.0, 1.0);
    let radius = egui::CornerRadius::same((height / 2.0).round() as u8);

    ui.painter()
        .rect_filled(rect, radius, egui::Color32::from_rgb(230, 230, 230));

    let fill_width = rect.width() * fraction;
    if fill_width >= 0.5 {
        let fill_rect = egui::Rect::from_min_size(rect.min, egui::vec2(fill_width, rect.height()));
        ui.painter().rect_filled(fill_rect, radius, fill);
    }

    ui.painter().rect_stroke(
        rect,
        radius,
        egui::Stroke::new(1.0, STROKE),
        egui::StrokeKind::Inside,
    );

    response
}

fn chunk_detail(ui: &egui::Ui, chunk: &ChunkProgress, fraction: f32) -> (String, egui::Color32) {
    match &chunk.status {
        ChunkStatus::Pending => ("Pending".to_owned(), egui::Color32::TRANSPARENT),
        ChunkStatus::Running => (
            format!(
                "D3PM {}/{} - {}%",
                chunk.d3pm_current,
                chunk.d3pm_total,
                (fraction * 100.0).round() as usize
            ),
            ACCENT,
        ),
        ChunkStatus::Completed => ("Complete".to_owned(), egui::Color32::from_rgb(16, 124, 16)),
        ChunkStatus::Failed(message) => (message.clone(), ui.visuals().error_fg_color),
    }
}

fn overall_fraction(state: &AppState) -> f32 {
    if !state.chunk_progress.is_empty() {
        let total_progress = state.chunk_progress.iter().map(chunk_fraction).sum::<f32>();
        return total_progress / state.chunk_progress.len() as f32;
    }

    state
        .overall_progress
        .map(|(current, total)| {
            if total == 0 {
                0.0
            } else {
                current as f32 / total as f32
            }
        })
        .unwrap_or(0.0)
        .clamp(0.0, 1.0)
}

fn chunk_fraction(chunk: &ChunkProgress) -> f32 {
    match &chunk.status {
        ChunkStatus::Pending => 0.0,
        ChunkStatus::Completed => 1.0,
        ChunkStatus::Running | ChunkStatus::Failed(_) => {
            if chunk.d3pm_total == 0 {
                0.0
            } else {
                chunk.d3pm_current as f32 / chunk.d3pm_total as f32
            }
        }
    }
    .clamp(0.0, 1.0)
}

fn render_timings(ui: &mut egui::Ui, state: &AppState) {
    section_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        section_title(ui, "Stage Timings");
        ui.add_space(8.0);

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
    section_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        section_title(ui, "Event Log");
        ui.add_space(8.0);
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
