use std::path::{Path, PathBuf};
use std::process::Command;

use crate::state::{AppState, backend_name, format_count, format_duration, output_format_name};

use super::{page_title, primary_button, section_frame, section_title};

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    let Some(summary) = ResultSummary::from_state(state) else {
        page_title(ui, "No extraction result");
        if ui.button("Back to Configuration").clicked() {
            state.reset_to_config();
        }
        return;
    };

    // Pin the actions to an always-visible bottom bar so "Extract Again" stays
    // reachable no matter how tall the (scrollable) summary/notes body grows.
    egui::Panel::bottom("results_action_bar")
        .exact_size(58.0)
        .frame(
            egui::Frame::NONE
                .fill(ui.visuals().panel_fill)
                .inner_margin(egui::Margin::symmetric(0, 10)),
        )
        .show_inside(ui, |ui| {
            render_action_bar(ui, state, &summary);
        });

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("results_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    render_results_body(ui, state, &summary);
                    ui.add_space(8.0);
                });
        });
}

fn render_action_bar(ui: &mut egui::Ui, state: &mut AppState, summary: &ResultSummary) {
    ui.horizontal(|ui| {
        if ui
            .add(primary_button(ui, "Extract Again").min_size(egui::vec2(140.0, 34.0)))
            .clicked()
        {
            state.reset_to_config();
        }

        if let Some(path) = &summary.output_path {
            if ui
                .add(egui::Button::new("Open File").min_size(egui::vec2(120.0, 34.0)))
                .clicked()
            {
                match open_path(path) {
                    Ok(()) => state.status_text.clear(),
                    Err(err) => state.status_text = format!("Failed to open output file: {err}"),
                }
            }

            if ui
                .add(egui::Button::new("Open Output Folder").min_size(egui::vec2(170.0, 34.0)))
                .clicked()
            {
                match open_output_folder(path) {
                    Ok(()) => state.status_text.clear(),
                    Err(err) => state.status_text = format!("Failed to open output folder: {err}"),
                }
            }
        }

        if !state.status_text.is_empty() {
            ui.label(egui::RichText::new(&state.status_text).color(ui.visuals().weak_text_color()));
        }
    });
}

fn render_results_body(ui: &mut egui::Ui, state: &AppState, summary: &ResultSummary) {
    page_title(ui, "Extraction Complete");
    ui.add_space(14.0);

    section_frame(ui).show(ui, |ui| {
        ui.set_width(ui.available_width());
        egui::Grid::new("result_summary")
            .num_columns(2)
            .spacing([28.0, 8.0])
            .show(ui, |ui| {
                row(ui, "Backend", &summary.backend);
                row(ui, "Total notes", &summary.note_count);
                row(ui, "Chunks", &summary.chunk_count);
                row(ui, "Audio duration", &summary.audio_duration);
                row(ui, "Frames", &summary.total_frames);
                row(ui, "Total time", &summary.total_time);
                if let Some(output) = &summary.output_display {
                    row(ui, "Output", output);
                }
            });
    });

    ui.add_space(16.0);
    section_frame(ui).show(ui, |ui| {
        ui.set_width(ui.available_width());
        section_title(ui, "Timing Breakdown");
        ui.add_space(8.0);
        egui::Grid::new("result_timings")
            .num_columns(2)
            .spacing([28.0, 8.0])
            .show(ui, |ui| {
                row(ui, "Model load", &summary.model_load);
                row(ui, "Audio prep", &summary.audio_prepare);
                row(ui, "Silence slice", &summary.silence_slice);
                row(ui, "Long chunk split", &summary.long_chunk_split);
                row(ui, "Mel setup", &summary.mel_setup);
                row(ui, "Inference", &summary.inference);
                row(ui, "Output write", &summary.output_write);
            });
    });

    ui.add_space(16.0);
    render_notes_preview(ui, state);
}

fn row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(egui::RichText::new(label).color(ui.visuals().weak_text_color()));
    ui.label(value);
    ui.end_row();
}

struct ResultSummary {
    backend: String,
    note_count: String,
    chunk_count: String,
    audio_duration: String,
    total_frames: String,
    total_time: String,
    model_load: String,
    audio_prepare: String,
    silence_slice: String,
    long_chunk_split: String,
    mel_setup: String,
    inference: String,
    output_write: String,
    output_display: Option<String>,
    output_path: Option<PathBuf>,
}

impl ResultSummary {
    fn from_state(state: &AppState) -> Option<Self> {
        let result = state.result.as_ref()?;
        let backend = if let Some(adapter) = &result.gpu_adapter {
            format!("{} ({})", backend_name(result.backend), adapter.name)
        } else {
            backend_name(result.backend).to_owned()
        };
        let chunk_count = if result.chunks_before_long_split != result.chunk_count {
            format!(
                "{} -> {}",
                result.chunks_before_long_split, result.chunk_count
            )
        } else {
            result.chunk_count.to_string()
        };
        let output_display = result.output.as_ref().map(|output| {
            format!(
                "{} ({})",
                output.path.display(),
                output_format_name(output.format)
            )
        });
        let output_path = result.output.as_ref().map(|output| output.path.clone());

        Some(Self {
            backend,
            note_count: format_count(result.notes.len()),
            chunk_count,
            audio_duration: format_audio_duration(result.audio.duration_seconds()),
            total_frames: format_count(result.total_frames),
            total_time: format_duration(result.timings.total),
            model_load: format_duration(result.timings.model_load),
            audio_prepare: format_duration(result.timings.audio_prepare),
            silence_slice: format_duration(result.timings.silence_slice),
            long_chunk_split: format_duration(result.timings.long_chunk_split),
            mel_setup: format_duration(result.timings.mel_setup),
            inference: format_duration(result.timings.inference),
            output_write: format_duration(result.timings.output_write),
            output_display,
            output_path,
        })
    }
}

fn format_audio_duration(seconds: f64) -> String {
    let total = seconds.round().max(0.0) as u64;
    let minutes = total / 60;
    let seconds = total % 60;
    if minutes == 0 {
        format!("{seconds}s")
    } else {
        format!("{minutes}m {seconds:02}s")
    }
}

fn open_output_folder(path: &Path) -> std::io::Result<()> {
    let folder = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    open_path(folder)
}

/// Opens a file or folder with the OS default handler.
fn open_path(target: &Path) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer").arg(target).spawn()?;
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(target).spawn()?;
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open").arg(target).spawn()?;
    }
    Ok(())
}

const NOTE_PREVIEW_LIMIT: usize = 200;

fn render_notes_preview(ui: &mut egui::Ui, state: &AppState) {
    let Some(result) = state.result.as_ref() else {
        return;
    };
    let notes = &result.notes;

    section_frame(ui).show(ui, |ui| {
        ui.set_width(ui.available_width());
        section_title(ui, "Extracted Notes");
        ui.add_space(8.0);

        if notes.is_empty() {
            ui.label("No notes were produced.");
            return;
        }

        let voiced = notes.iter().filter(|note| note.voiced).count();
        let range_text = voiced_pitch_range(notes)
            .map(|(lo, hi)| format!("{} – {}", midi_note_name(lo), midi_note_name(hi)))
            .unwrap_or_else(|| "—".to_owned());
        ui.label(
            egui::RichText::new(format!(
                "{} notes ({voiced} voiced) · pitch range {range_text}",
                format_count(notes.len())
            ))
            .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(8.0);

        egui::ScrollArea::vertical()
            .id_salt("notes_preview")
            .max_height(260.0)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                egui::Grid::new("notes_grid")
                    .num_columns(4)
                    .striped(true)
                    .spacing([24.0, 4.0])
                    .show(ui, |ui| {
                        for header in ["#", "Offset", "Duration", "Pitch"] {
                            ui.label(
                                egui::RichText::new(header).color(ui.visuals().weak_text_color()),
                            );
                        }
                        ui.end_row();

                        for (index, note) in notes.iter().take(NOTE_PREVIEW_LIMIT).enumerate() {
                            ui.label(format!("{}", index + 1));
                            ui.label(format!("{:.3}s", note.offset_seconds));
                            ui.label(format!("{:.3}s", note.duration_seconds));
                            if note.voiced {
                                ui.label(format!(
                                    "{} ({:.1})",
                                    midi_note_name(note.pitch_midi),
                                    note.pitch_midi
                                ));
                            } else {
                                ui.label("rest");
                            }
                            ui.end_row();
                        }
                    });

                if notes.len() > NOTE_PREVIEW_LIMIT {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "… and {} more",
                            format_count(notes.len() - NOTE_PREVIEW_LIMIT)
                        ))
                        .color(ui.visuals().weak_text_color()),
                    );
                }
            });
    });
}

fn voiced_pitch_range(notes: &[game_service::Note]) -> Option<(f32, f32)> {
    let mut voiced = notes
        .iter()
        .filter(|note| note.voiced)
        .map(|note| note.pitch_midi);
    let first = voiced.next()?;
    Some(voiced.fold((first, first), |(lo, hi), pitch| {
        (lo.min(pitch), hi.max(pitch))
    }))
}

fn midi_note_name(midi: f32) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let rounded = midi.round() as i32;
    let name = NAMES[rounded.rem_euclid(12) as usize];
    let octave = rounded.div_euclid(12) - 1;
    format!("{name}{octave}")
}
