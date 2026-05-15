pub use game_core::{Error, Note, Result};

mod output;

mod midi_writer;

pub use midi_writer::{MidiWriteOptions, encode_midi, write_midi_file};
pub use output::{TextOutputFormat, TextWriteOptions, format_notes_text, write_text_file};
