use std::fs;
use std::path::Path;

use midly::num::{u4, u7, u15, u24, u28};
use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};

use crate::{Error, Note, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MidiWriteOptions {
    pub tempo_bpm: u32,
    pub velocity: u8,
    pub ticks_per_qn: u16,
}

impl Default for MidiWriteOptions {
    fn default() -> Self {
        Self {
            tempo_bpm: 120,
            velocity: 96,
            ticks_per_qn: 480,
        }
    }
}

#[derive(Clone)]
struct TimedEvent {
    tick: u64,
    kind: TrackEventKind<'static>,
}

pub fn encode_midi(notes: &[Note], options: &MidiWriteOptions) -> Result<Vec<u8>> {
    if options.tempo_bpm == 0 {
        return Err(Error::message("MIDI tempo must be > 0 BPM"));
    }
    if options.ticks_per_qn == 0 {
        return Err(Error::message("MIDI ticks_per_qn must be > 0"));
    }

    let velocity = options.velocity.clamp(1, 127);
    let ticks_per_second = f64::from(options.ticks_per_qn) * f64::from(options.tempo_bpm) / 60.0;
    let mut events = Vec::with_capacity(notes.len() * 2 + 2);

    let micros_per_quarter = 60_000_000u32 / options.tempo_bpm;
    events.push(TimedEvent {
        tick: 0,
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::from(micros_per_quarter))),
    });

    for note in notes {
        if !note.voiced {
            continue;
        }

        let pitch = note.pitch_midi.round().clamp(0.0, 127.0) as u8;
        let start_tick = seconds_to_tick(note.offset_seconds, ticks_per_second)?;
        let mut end_tick = seconds_to_tick(
            note.offset_seconds + note.duration_seconds,
            ticks_per_second,
        )?;
        if end_tick <= start_tick {
            end_tick = start_tick + 1;
        }

        events.push(TimedEvent {
            tick: start_tick,
            kind: TrackEventKind::Midi {
                channel: u4::from(0),
                message: MidiMessage::NoteOn {
                    key: u7::from(pitch),
                    vel: u7::from(velocity),
                },
            },
        });
        events.push(TimedEvent {
            tick: end_tick,
            kind: TrackEventKind::Midi {
                channel: u4::from(0),
                message: MidiMessage::NoteOff {
                    key: u7::from(pitch),
                    vel: u7::from(64),
                },
            },
        });
    }

    let last_tick = events.iter().map(|event| event.tick).max().unwrap_or(0);
    events.push(TimedEvent {
        tick: last_tick,
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    events.sort_by(|left, right| left.tick.cmp(&right.tick));

    let mut prev_tick = 0u64;
    let mut track = Vec::with_capacity(events.len());
    for event in events {
        let delta = event
            .tick
            .checked_sub(prev_tick)
            .ok_or_else(|| Error::message("MIDI event ordering produced a negative delta time"))?;
        let delta = u32::try_from(delta)
            .map_err(|_| Error::message(format!("MIDI delta time {delta} exceeds u32::MAX")))?;
        if delta > 0x0fff_ffff {
            return Err(Error::message(format!(
                "MIDI delta time {delta} exceeds the SMF u28 limit"
            )));
        }

        track.push(TrackEvent {
            delta: u28::from(delta),
            kind: event.kind,
        });
        prev_tick = event.tick;
    }

    let smf = Smf {
        header: Header::new(
            Format::SingleTrack,
            Timing::Metrical(u15::from(options.ticks_per_qn)),
        ),
        tracks: vec![track],
    };

    let mut bytes = Vec::new();
    smf.write_std(&mut bytes)
        .map_err(|err| Error::message(format!("failed to encode MIDI: {err}")))?;
    Ok(bytes)
}

pub fn write_midi_file(
    path: impl AsRef<Path>,
    notes: &[Note],
    options: &MidiWriteOptions,
) -> Result<()> {
    let bytes = encode_midi(notes, options)?;
    let path_ref = path.as_ref();
    fs::write(path_ref, bytes)
        .map_err(|err| Error::message(format!("failed to write MIDI {}: {err}", path_ref.display())))?;
    Ok(())
}

fn seconds_to_tick(seconds: f32, ticks_per_second: f64) -> Result<u64> {
    if !seconds.is_finite() {
        return Err(Error::message(format!(
            "MIDI note time must be finite, got {seconds}"
        )));
    }

    let tick = (f64::from(seconds) * ticks_per_second).round();
    if !(0.0..=u64::MAX as f64).contains(&tick) {
        return Err(Error::message(format!(
            "MIDI tick value {tick} is out of range"
        )));
    }

    Ok(tick as u64)
}

#[cfg(test)]
mod tests {
    use midly::{MetaMessage, MidiMessage, Smf, TrackEventKind};

    use super::{MidiWriteOptions, encode_midi};
    use crate::Note;

    #[test]
    fn encodes_single_track_midi_with_note_events() {
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
                pitch_midi: 64.0,
                voiced: true,
            },
        ];

        let bytes = encode_midi(&notes, &MidiWriteOptions::default()).unwrap();
        let smf = Smf::parse(&bytes).unwrap();

        assert_eq!(smf.header.format, midly::Format::SingleTrack);
        assert_eq!(smf.tracks.len(), 1);

        let mut note_on_count = 0usize;
        let mut note_off_count = 0usize;
        let mut saw_tempo = false;
        let mut saw_end = false;

        for event in &smf.tracks[0] {
            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(_)) => saw_tempo = true,
                TrackEventKind::Meta(MetaMessage::EndOfTrack) => saw_end = true,
                TrackEventKind::Midi {
                    message: MidiMessage::NoteOn { .. },
                    ..
                } => note_on_count += 1,
                TrackEventKind::Midi {
                    message: MidiMessage::NoteOff { .. },
                    ..
                } => note_off_count += 1,
                _ => {}
            }
        }

        assert!(saw_tempo);
        assert!(saw_end);
        assert_eq!(note_on_count, 2);
        assert_eq!(note_off_count, 2);
    }

    #[test]
    fn skips_unvoiced_notes() {
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

        let bytes = encode_midi(&notes, &MidiWriteOptions::default()).unwrap();
        let smf = Smf::parse(&bytes).unwrap();
        let note_event_count = smf.tracks[0]
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    TrackEventKind::Midi {
                        message: MidiMessage::NoteOn { .. } | MidiMessage::NoteOff { .. },
                        ..
                    }
                )
            })
            .count();

        assert_eq!(note_event_count, 2);
    }
}
