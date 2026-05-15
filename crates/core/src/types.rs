#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Note {
    pub offset_seconds: f32,
    pub duration_seconds: f32,
    pub pitch_midi: f32,
    pub voiced: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferParams {
    pub language: i32,
    pub d3pm_ts: Vec<f32>,
    pub d3pm_t0: f32,
    pub d3pm_nsteps: i32,
    pub boundary_threshold: f32,
    pub boundary_radius: i32,
    pub note_threshold: f32,
    pub seed: u64,
}

impl Default for InferParams {
    fn default() -> Self {
        Self {
            language: 0,
            d3pm_ts: Vec::new(),
            d3pm_t0: 0.0,
            d3pm_nsteps: 1,
            boundary_threshold: 0.2,
            boundary_radius: 2,
            note_threshold: 0.2,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct InferResult {
    pub notes: Vec<Note>,
    pub num_frames: i32,
}
