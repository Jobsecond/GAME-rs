use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::{Note, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextOutputFormat {
    Txt,
    Csv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TextWriteOptions {
    pub round_pitch: bool,
    pub use_pitch_names: bool,
}

const PITCH_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

pub fn format_notes_text(
    notes: &[Note],
    format: TextOutputFormat,
    options: &TextWriteOptions,
) -> String {
    let mut out = String::new();
    if format == TextOutputFormat::Csv {
        out.push_str("offset,duration,pitch\n");
    }

    let separator = match format {
        TextOutputFormat::Txt => '\t',
        TextOutputFormat::Csv => ',',
    };

    for note in notes {
        let _ = writeln!(
            out,
            "{}{}{}{}{}",
            note.offset_seconds,
            separator,
            note.duration_seconds,
            separator,
            format_pitch(note.pitch_midi, note.voiced, options)
        );
    }

    out
}

pub fn write_text_file(
    path: impl AsRef<Path>,
    notes: &[Note],
    format: TextOutputFormat,
    options: &TextWriteOptions,
) -> Result<()> {
    fs::write(path.as_ref(), format_notes_text(notes, format, options))?;
    Ok(())
}

fn format_pitch(pitch: f32, voiced: bool, options: &TextWriteOptions) -> String {
    if !voiced {
        return "rest".to_owned();
    }

    if options.use_pitch_names {
        let rounded = pitch.round() as i32;
        let pitch_class = rounded.rem_euclid(12) as usize;
        let octave = rounded.div_euclid(12) - 1;
        let cents = (pitch - rounded as f32) * 100.0;

        if options.round_pitch || cents.abs() < 0.5 {
            return format!("{}{}", PITCH_NAMES[pitch_class], octave);
        }

        return format!(
            "{}{}{:+}",
            PITCH_NAMES[pitch_class],
            octave,
            cents.round() as i32
        );
    }

    if options.round_pitch {
        format!("{}", pitch.round() as i32)
    } else {
        format!("{pitch:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::{TextOutputFormat, TextWriteOptions, format_notes_text};
    use crate::Note;

    #[test]
    fn txt_and_csv_emit_expected_separators_and_headers() {
        let notes = vec![
            Note {
                offset_seconds: 0.0,
                duration_seconds: 0.5,
                pitch_midi: 60.0,
                voiced: true,
            },
            Note {
                offset_seconds: 0.5,
                duration_seconds: 0.5,
                pitch_midi: 0.0,
                voiced: false,
            },
        ];

        let options = TextWriteOptions {
            round_pitch: true,
            use_pitch_names: false,
        };
        let txt = format_notes_text(&notes, TextOutputFormat::Txt, &options);
        let csv = format_notes_text(&notes, TextOutputFormat::Csv, &options);

        assert!(txt.contains('\t'));
        assert!(txt.contains("rest"));
        assert!(csv.starts_with("offset,duration,pitch\n"));
        assert!(csv.contains("60"));
    }

    #[test]
    fn note_names_include_octave() {
        let notes = vec![Note {
            offset_seconds: 0.0,
            duration_seconds: 1.0,
            pitch_midi: 69.0,
            voiced: true,
        }];

        let options = TextWriteOptions {
            round_pitch: false,
            use_pitch_names: true,
        };
        let txt = format_notes_text(&notes, TextOutputFormat::Txt, &options);

        assert!(txt.contains("A4"));
    }
}
