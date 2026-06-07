use std::path::{Path, PathBuf};
use std::process::Command;

use crate::state::{AppState, backend_name, format_count, format_duration, output_format_name};

use super::{TEXT_SECONDARY, page_title, primary_button, section_frame, section_title};

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    let Some(summary) = ResultSummary::from_state(state) else {
        page_title(ui, "No extraction result");
        if ui.button("Back to Configuration").clicked() {
            state.reset_to_config();
        }
        return;
    };

    page_title(ui, "Extraction Complete");
    ui.add_space(14.0);

    section_frame().show(ui, |ui| {
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
    section_frame().show(ui, |ui| {
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

    if !state.status_text.is_empty() {
        ui.add_space(8.0);
        ui.label(&state.status_text);
    }

    ui.add_space(16.0);
    ui.horizontal(|ui| {
        if ui
            .add(primary_button("Extract Again").min_size(egui::vec2(130.0, 32.0)))
            .clicked()
        {
            state.reset_to_config();
        }

        if let Some(path) = &summary.output_path {
            if ui
                .add(egui::Button::new("Open Output Folder").min_size(egui::vec2(160.0, 32.0)))
                .clicked()
            {
                match open_output_folder(path) {
                    Ok(()) => state.status_text.clear(),
                    Err(err) => {
                        state.status_text = format!("Failed to open output folder: {err}");
                    }
                }
            }
        }
    });
}

fn row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(egui::RichText::new(label).color(TEXT_SECONDARY));
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

    #[cfg(target_os = "windows")]
    {
        Command::new("explorer").arg(folder).spawn()?;
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(folder).spawn()?;
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open").arg(folder).spawn()?;
    }

    Ok(())
}
