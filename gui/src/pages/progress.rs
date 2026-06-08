use crate::state::{
    AppState, ChunkProgress, ChunkStatus, GuiLogLevel, StageTiming, format_duration,
};

use super::{accent, page_title, section_frame, section_title, track_bg};

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    // Animation/refresh while running is driven by `GuiApp::logic` (gated on
    // `is_running`) plus the notifier's per-event repaints, so this page does
    // not need its own unconditional repaint timer.
    egui::Panel::bottom("progress_action_bar")
        .exact_size(58.0)
        .frame(
            egui::Frame::NONE
                .fill(ui.visuals().panel_fill)
                .inner_margin(egui::Margin::symmetric(0, 10)),
        )
        .show_inside(ui, |ui| {
            render_action_bar(ui, state);
        });

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("progress_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    render_progress_body(ui, state);
                    ui.add_space(8.0);
                });
        });
}

fn render_progress_body(ui: &mut egui::Ui, state: &AppState) {
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
}

fn render_action_bar(ui: &mut egui::Ui, state: &mut AppState) {
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

    progress_header(ui, "Overall", &label, HeaderIndicator::None);
    ui.add_space(4.0);
    progress_track(ui, fraction, accent(ui), 8.0);

    ui.add_space(8.0);
    ui.label(egui::RichText::new(&state.overall_status).color(ui.visuals().weak_text_color()));
    if !state.status_text.is_empty() {
        ui.label(egui::RichText::new(&state.status_text).color(ui.visuals().weak_text_color()));
    }
}

fn render_chunks(ui: &mut egui::Ui, state: &AppState) {
    section_frame(ui).show(ui, |ui| {
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
                    let indicator = chunk_indicator(ui, &chunk.status);

                    progress_header(ui, &chunk.label, &detail, indicator);
                    ui.add_space(4.0);
                    progress_track(ui, fraction, fill, 6.0);
                    if index + 1 < state.chunk_progress.len() {
                        ui.add_space(10.0);
                    }
                }
            });
    });
}

fn progress_header(ui: &mut egui::Ui, label: &str, detail: &str, indicator: HeaderIndicator) {
    ui.horizontal(|ui| {
        ui.set_width(ui.available_width());
        match indicator {
            HeaderIndicator::None => {}
            HeaderIndicator::Running => {
                ui.add(egui::Spinner::new().size(14.0));
                ui.add_space(2.0);
            }
            HeaderIndicator::Badge(badge) => {
                render_badge(ui, badge);
                ui.add_space(4.0);
            }
        }
        ui.label(egui::RichText::new(label).color(ui.visuals().text_color()));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(detail).color(ui.visuals().weak_text_color()));
        });
    });
}

#[derive(Clone, Copy)]
enum HeaderIndicator {
    None,
    Running,
    Badge(BadgeStyle),
}

#[derive(Clone, Copy)]
struct BadgeStyle {
    text: &'static str,
    fill: egui::Color32,
    stroke: egui::Color32,
    text_color: egui::Color32,
}

fn chunk_indicator(ui: &egui::Ui, status: &ChunkStatus) -> HeaderIndicator {
    match status {
        ChunkStatus::Pending => HeaderIndicator::Badge(BadgeStyle {
            text: "Pending",
            fill: ui.visuals().faint_bg_color,
            stroke: ui.visuals().window_stroke.color,
            text_color: ui.visuals().weak_text_color(),
        }),
        ChunkStatus::Running => HeaderIndicator::Running,
        ChunkStatus::Completed => {
            let badge = if ui.visuals().dark_mode {
                BadgeStyle {
                    text: "Finished",
                    fill: egui::Color32::from_rgb(22, 64, 42),
                    stroke: egui::Color32::from_rgb(45, 164, 78),
                    text_color: egui::Color32::from_rgb(180, 236, 196),
                }
            } else {
                BadgeStyle {
                    text: "Finished",
                    fill: egui::Color32::from_rgb(223, 246, 221),
                    stroke: egui::Color32::from_rgb(16, 124, 16),
                    text_color: egui::Color32::from_rgb(16, 124, 16),
                }
            };
            HeaderIndicator::Badge(badge)
        }
        ChunkStatus::Failed(_) => {
            let badge = if ui.visuals().dark_mode {
                BadgeStyle {
                    text: "Failed",
                    fill: egui::Color32::from_rgb(64, 26, 28),
                    stroke: egui::Color32::from_rgb(232, 86, 76),
                    text_color: egui::Color32::from_rgb(255, 220, 220),
                }
            } else {
                BadgeStyle {
                    text: "Failed",
                    fill: egui::Color32::from_rgb(253, 231, 233),
                    stroke: egui::Color32::from_rgb(196, 43, 28),
                    text_color: egui::Color32::from_rgb(196, 43, 28),
                }
            };
            HeaderIndicator::Badge(badge)
        }
    }
}

fn render_badge(ui: &mut egui::Ui, badge: BadgeStyle) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(BADGE_WIDTH, BADGE_HEIGHT),
        egui::Sense::hover(),
    );
    let radius = egui::CornerRadius::same(4);
    ui.painter().rect_filled(rect, radius, badge.fill);
    ui.painter().rect_stroke(
        rect,
        radius,
        egui::Stroke::new(1.0, badge.stroke),
        egui::StrokeKind::Inside,
    );
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        badge.text,
        egui::FontId::proportional(12.0),
        badge.text_color,
    );
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

    ui.painter().rect_filled(rect, radius, track_bg(ui));

    let fill_width = rect.width() * fraction;
    if fill_width >= 0.5 {
        let fill_rect = egui::Rect::from_min_size(rect.min, egui::vec2(fill_width, rect.height()));
        ui.painter().rect_filled(fill_rect, radius, fill);
    }

    ui.painter().rect_stroke(
        rect,
        radius,
        egui::Stroke::new(1.0, ui.visuals().window_stroke.color),
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
            accent(ui),
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
    section_frame(ui).show(ui, |ui| {
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
    section_frame(ui).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            section_title(ui, "Event Log");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(!state.event_log.is_empty(), egui::Button::new("Copy"))
                    .clicked()
                {
                    let text = state
                        .event_log
                        .iter()
                        .map(|entry| {
                            format!("[+{}] {}", format_duration(entry.elapsed), entry.text)
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    ui.ctx().copy_text(text);
                }
            });
        });
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

const BADGE_WIDTH: f32 = 76.0;
const BADGE_HEIGHT: f32 = 20.0;
