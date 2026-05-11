use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct BackboneConfig {
    pub cls: String,
    pub dim: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub c_kernel_size: i32,
    pub m_kernel_size: i32,
    pub ffn_type: String,
    pub use_ls: bool,
    pub use_out_norm: bool,
    pub skip_first_ffn: bool,
    pub skip_out_ffn: bool,
    pub return_latent: bool,
    pub latent_layer_idx: i32,
    pub latent_out_dim: i32,
    pub region_token_num: i32,
    pub pool_merge_mode: String,
    pub attn_type: String,
    pub rope_mode: String,
    pub qk_norm: bool,
    pub use_region_bias: bool,
    pub c_kernel_size_pool: i32,
    pub m_kernel_size_pool: i32,
    pub c_kernel_size_x: i32,
    pub m_kernel_size_x: i32,
    pub use_rope: bool,
    pub use_pool_offset: bool,
    pub theta: f32,
}

impl Default for BackboneConfig {
    fn default() -> Self {
        Self {
            cls: String::new(),
            dim: 0,
            num_layers: 0,
            num_heads: 0,
            head_dim: 0,
            c_kernel_size: 0,
            m_kernel_size: 0,
            ffn_type: "glu".to_owned(),
            use_ls: true,
            use_out_norm: true,
            skip_first_ffn: false,
            skip_out_ffn: false,
            return_latent: false,
            latent_layer_idx: 0,
            latent_out_dim: 0,
            region_token_num: 1,
            pool_merge_mode: "mean".to_owned(),
            attn_type: "joint".to_owned(),
            rope_mode: "mixed".to_owned(),
            qk_norm: true,
            use_region_bias: false,
            c_kernel_size_pool: 0,
            m_kernel_size_pool: 0,
            c_kernel_size_x: 0,
            m_kernel_size_x: 0,
            use_rope: true,
            use_pool_offset: false,
            theta: 10_000.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceConfig {
    pub audio_sample_rate: i32,
    pub hop_size: i32,
    pub fft_size: i32,
    pub win_size: i32,
    pub n_mels: i32,
    pub fmin: f32,
    pub fmax: f32,
    pub spectrogram_type: String,
    pub midi_min: f32,
    pub midi_max: f32,
    pub midi_num_bins: i32,
    pub midi_std: f32,
    pub lang_map: BTreeMap<String, i32>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            audio_sample_rate: 0,
            hop_size: 0,
            fft_size: 0,
            win_size: 0,
            n_mels: 0,
            fmin: 0.0,
            fmax: 0.0,
            spectrogram_type: "mel".to_owned(),
            midi_min: 0.0,
            midi_max: 0.0,
            midi_num_bins: 0,
            midi_std: 0.0,
            lang_map: BTreeMap::new(),
        }
    }
}

impl InferenceConfig {
    pub fn timestep(&self) -> f32 {
        if self.audio_sample_rate > 0 {
            self.hop_size as f32 / self.audio_sample_rate as f32
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GameModelConfig {
    pub architecture: String,
    pub name: String,
    pub version: String,
    pub mode: String,
    pub embedding_dim: i32,
    pub in_dim: i32,
    pub estimator_out_dim: i32,
    pub region_cycle_len: i32,
    pub use_languages: bool,
    pub num_languages: i32,
    pub encoder: BackboneConfig,
    pub segmenter: BackboneConfig,
    pub estimator: BackboneConfig,
    pub inference: InferenceConfig,
}
