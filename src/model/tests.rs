use std::collections::BTreeMap;
use std::path::PathBuf;

use super::{
    Backend, Model, bind_model_weights, build_joint_attn_mask, run_encoder, run_estimator,
    run_segmenter_step,
};
use crate::config::{BackboneConfig, GameModelConfig, InferenceConfig};
use crate::gguf::{GGMLType, GGUFVersion};
use crate::gguf_loader::{LoadedGgufModel, LoadedTensor};
use crate::{CpuDevice, CpuTensor, InferParams, InjectedRng, Tensor};

#[test]
fn joint_attention_mask_matches_region_rules() {
    let mask = build_joint_attn_mask(&[1, 1, 2, 0], 2);
    let side = 6usize;

    assert_eq!(mask.len(), side * side);
    assert_eq!(mask[0 * side + 1], 0.0);
    assert_eq!(mask[0 * side + 2], 0.0);
    assert_eq!(mask[0 * side + 4], -10_000.0);
    assert_eq!(mask[2 * side + 3], 0.0);
    assert_eq!(mask[4 * side + 5], -10_000.0);
    assert_eq!(mask[5 * side + 5], -10_000.0);
}

#[test]
fn synthetic_forward_passes_produce_expected_shapes() {
    let model = fake_loaded_model();
    let weights = bind_model_weights::<CpuTensor>(&model, &CpuDevice).unwrap();
    let mel = cpu_tensor(
        &[
            0.1, 0.2, 0.3, //
            0.4, 0.5, 0.6, //
            0.7, 0.8, 0.9,
        ],
        &[3, 3],
    );

    let encoder = run_encoder(
        &mel,
        &weights.spectrogram_projection,
        &weights.encoder,
        &model.config,
    )
    .unwrap();
    assert_eq!(encoder.x_seg.shape(), &[3, 4]);
    assert_eq!(encoder.x_est.shape(), &[3, 4]);
    assert_all_finite(&encoder.x_seg);
    assert_all_finite(&encoder.x_est);

    let segmenter = run_segmenter_step(
        &encoder.x_seg,
        &[0, 1, 2],
        Some(0.5),
        Some(1),
        &weights.segmenter,
        &model.config,
    )
    .unwrap();
    assert_eq!(segmenter.logits.shape(), &[3]);
    assert_eq!(segmenter.latent.as_ref().unwrap().shape(), &[3, 2]);
    assert_all_finite(&segmenter.logits);
    assert_all_finite(segmenter.latent.as_ref().unwrap());

    let estimator = run_estimator(
        &encoder.x_est,
        &[1, 1, 2],
        &weights.estimator,
        &model.config,
    )
    .unwrap();
    assert_eq!(estimator.pool_logits.shape(), &[2, 5]);
    assert_all_finite(&estimator.pool_logits);
}

#[test]
fn estimator_returns_empty_logits_for_no_regions() {
    let model = fake_loaded_model();
    let weights = bind_model_weights::<CpuTensor>(&model, &CpuDevice).unwrap();
    let x_est = cpu_tensor(
        &[
            0.1, 0.2, 0.3, 0.4, //
            0.5, 0.6, 0.7, 0.8,
        ],
        &[2, 4],
    );

    let estimator = run_estimator(&x_est, &[0, 0], &weights.estimator, &model.config).unwrap();
    assert_eq!(estimator.pool_logits.shape(), &[0, 5]);
}

#[test]
fn model_infer_runs_end_to_end_on_cpu_backend() {
    let model = Model::from_loaded_model(fake_loaded_model(), Backend::Cpu).unwrap();
    let waveform = vec![0.0f32; 4];
    let params = InferParams {
        seed: 42,
        d3pm_nsteps: 2,
        boundary_threshold: 0.0,
        note_threshold: 0.0,
        ..Default::default()
    };
    let mut rng = InjectedRng::new(vec![0.1, 0.9, 0.2, 0.8]);

    let result = model.infer_with_rng(&waveform, &params, &mut rng).unwrap();

    assert_eq!(model.backend(), Backend::Cpu);
    assert_eq!(result.num_frames, 1);
    assert!(!result.notes.is_empty());
    let total_duration = result
        .notes
        .iter()
        .map(|note| note.duration_seconds)
        .sum::<f32>();
    assert!((total_duration - 0.5).abs() < 1e-6);
    assert!(result.notes.iter().any(|note| note.duration_seconds > 0.0));
    for note in &result.notes {
        assert!(note.duration_seconds >= 0.0);
        assert!(note.pitch_midi.is_finite());
    }
}

#[test]
fn model_infer_rejects_waveform_too_short_for_one_frame() {
    let model = Model::from_loaded_model(fake_loaded_model(), Backend::Cpu).unwrap();
    let params = InferParams::default();
    let mut rng = InjectedRng::new(Vec::new());

    let err = model.infer_with_rng(&[], &params, &mut rng).unwrap_err();
    assert!(err.to_string().contains("waveform too short"));
}

fn cpu_tensor(data: &[f32], shape: &[usize]) -> CpuTensor {
    CpuTensor::from_data(data, shape, &CpuDevice).unwrap()
}

fn assert_all_finite(tensor: &CpuTensor) {
    for value in tensor.to_vec().unwrap() {
        assert!(value.is_finite());
    }
}

fn fake_loaded_model() -> LoadedGgufModel {
    let mut tensors = BTreeMap::new();
    let cfg = fake_config();

    add_linear(&mut tensors, "spectrogram_projection", 4, 3);
    add_linear(&mut tensors, "encoder.input_proj", 4, 4);
    add_ebf_layer(&mut tensors, "encoder.layers.0", 4, 4, 3, 2, 3, 3, true);
    add_norm(&mut tensors, "encoder.output_norm.weight", 4);
    add_linear(&mut tensors, "encoder.output_proj", 8, 4);

    add_embedding(&mut tensors, "noise_embedding.embedding.weight", 3, 4);
    add_embedding(&mut tensors, "language_embedding.weight", 3, 4);
    add_linear(&mut tensors, "time_embedding.0", 16, 1);
    add_linear(&mut tensors, "time_embedding.2", 4, 16);
    add_linear(&mut tensors, "segmenter.input_proj", 4, 4);
    add_ebf_layer(&mut tensors, "segmenter.layers.0", 4, 4, 3, 2, 3, 3, true);
    add_norm(&mut tensors, "segmenter.latent_norm.weight", 4);
    add_linear(&mut tensors, "segmenter.latent_proj", 2, 4);
    add_norm(&mut tensors, "segmenter.output_norm.weight", 4);
    add_linear(&mut tensors, "segmenter.output_proj", 1, 4);

    add_linear(&mut tensors, "estimator.input_proj", 4, 4);
    add_embedding(&mut tensors, "estimator.pool_token_gen.emb", 1, 4);
    add_embedding(&mut tensors, "region_embedding.embedding.weight", 3, 4);
    add_jebf_layer(
        &mut tensors,
        "estimator.layers.0",
        4,
        4,
        4,
        3,
        3,
        3,
        3,
        true,
    );
    add_norm(&mut tensors, "estimator.output_norm_x.weight", 4);
    add_norm(&mut tensors, "estimator.output_norm_pool.weight", 4);
    add_linear(&mut tensors, "estimator.output_proj_x", 5, 4);
    add_linear(&mut tensors, "estimator.output_proj_pool", 5, 4);

    LoadedGgufModel {
        path: PathBuf::from("synthetic-phase6.gguf"),
        gguf_version: GGUFVersion::V3,
        quantization_version: None,
        metadata_count: 0,
        config: cfg,
        tensors,
    }
}

fn fake_config() -> GameModelConfig {
    let encoder = BackboneConfig {
        cls: "modules.backbones.EBF.EBFBackbone".to_owned(),
        dim: 4,
        num_layers: 1,
        num_heads: 1,
        head_dim: 4,
        c_kernel_size: 3,
        m_kernel_size: 3,
        ffn_type: "glu".to_owned(),
        use_ls: true,
        use_out_norm: true,
        use_rope: true,
        theta: 10_000.0,
        ..Default::default()
    };
    let segmenter = BackboneConfig {
        cls: "modules.backbones.EBF.EBFBackbone".to_owned(),
        dim: 4,
        num_layers: 1,
        num_heads: 1,
        head_dim: 4,
        c_kernel_size: 3,
        m_kernel_size: 3,
        ffn_type: "glu".to_owned(),
        use_ls: true,
        use_out_norm: true,
        return_latent: true,
        latent_layer_idx: 1,
        latent_out_dim: 2,
        use_rope: true,
        theta: 10_000.0,
        ..Default::default()
    };
    let estimator = BackboneConfig {
        cls: "modules.backbones.ebf_with_joint_attention.JEBFBackbone".to_owned(),
        dim: 4,
        num_layers: 1,
        num_heads: 1,
        head_dim: 4,
        ffn_type: "glu".to_owned(),
        use_ls: true,
        use_out_norm: true,
        region_token_num: 1,
        pool_merge_mode: "mean".to_owned(),
        attn_type: "joint".to_owned(),
        rope_mode: "mixed".to_owned(),
        qk_norm: true,
        c_kernel_size_pool: 3,
        m_kernel_size_pool: 3,
        c_kernel_size_x: 3,
        m_kernel_size_x: 3,
        use_rope: true,
        theta: 10_000.0,
        ..Default::default()
    };

    GameModelConfig {
        architecture: "game-me".to_owned(),
        name: "synthetic-phase6".to_owned(),
        version: "1".to_owned(),
        mode: "d3pm".to_owned(),
        embedding_dim: 4,
        in_dim: 3,
        estimator_out_dim: 5,
        region_cycle_len: 3,
        use_languages: true,
        num_languages: 2,
        encoder,
        segmenter,
        estimator,
        inference: InferenceConfig {
            audio_sample_rate: 8,
            hop_size: 4,
            fft_size: 4,
            win_size: 4,
            n_mels: 3,
            fmin: 0.0,
            fmax: 4.0,
            spectrogram_type: "mel".to_owned(),
            midi_min: 60.0,
            midi_max: 64.0,
            midi_num_bins: 5,
            midi_std: 1.0,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn add_linear(
    tensors: &mut BTreeMap<String, LoadedTensor>,
    prefix: &str,
    out_dim: usize,
    in_dim: usize,
) {
    add_tensor(tensors, &format!("{prefix}.weight"), &[out_dim, in_dim]);
    add_tensor(tensors, &format!("{prefix}.bias"), &[out_dim]);
}

fn add_pointwise_linear(
    tensors: &mut BTreeMap<String, LoadedTensor>,
    prefix: &str,
    out_dim: usize,
    in_dim: usize,
) {
    add_tensor(tensors, &format!("{prefix}.weight"), &[out_dim, in_dim, 1]);
    add_tensor(tensors, &format!("{prefix}.bias"), &[out_dim]);
}

fn add_embedding(
    tensors: &mut BTreeMap<String, LoadedTensor>,
    name: &str,
    rows: usize,
    cols: usize,
) {
    add_tensor(tensors, name, &[rows, cols]);
}

fn add_norm(tensors: &mut BTreeMap<String, LoadedTensor>, name: &str, dim: usize) {
    add_tensor(tensors, name, &[dim]);
}

fn add_depthwise(
    tensors: &mut BTreeMap<String, LoadedTensor>,
    prefix: &str,
    channels: usize,
    kernel_size: usize,
) {
    add_tensor(
        tensors,
        &format!("{prefix}.weight"),
        &[channels, 1, kernel_size],
    );
    add_tensor(tensors, &format!("{prefix}.bias"), &[channels]);
}

fn add_glu_ffn(
    tensors: &mut BTreeMap<String, LoadedTensor>,
    prefix: &str,
    dim: usize,
    hidden_dim: usize,
) {
    add_linear(tensors, &format!("{prefix}.ln1"), hidden_dim * 2, dim);
    add_linear(tensors, &format!("{prefix}.ln2"), dim, hidden_dim);
}

fn add_cgmlp(
    tensors: &mut BTreeMap<String, LoadedTensor>,
    prefix: &str,
    dim: usize,
    hidden_dim: usize,
    kernel_size: usize,
) {
    add_pointwise_linear(tensors, &format!("{prefix}.pw1"), hidden_dim * 2, dim);
    add_norm(tensors, &format!("{prefix}.norm.weight"), hidden_dim);
    add_depthwise(tensors, &format!("{prefix}.dw"), hidden_dim, kernel_size);
    add_pointwise_linear(tensors, &format!("{prefix}.pw2"), dim, hidden_dim);
}

fn add_attention(
    tensors: &mut BTreeMap<String, LoadedTensor>,
    prefix: &str,
    dim: usize,
    proj_dim: usize,
) {
    add_linear(tensors, &format!("{prefix}.q_linear"), proj_dim, dim);
    add_linear(tensors, &format!("{prefix}.kv_linear"), proj_dim * 2, dim);
    add_linear(tensors, &format!("{prefix}.out_linear"), dim, proj_dim);
}

fn add_merge(
    tensors: &mut BTreeMap<String, LoadedTensor>,
    linear_prefix: &str,
    dw_prefix: &str,
    out_dim: usize,
    in_dim: usize,
    kernel_size: usize,
) {
    add_linear(tensors, linear_prefix, out_dim, in_dim);
    if kernel_size != 0 {
        add_depthwise(tensors, dw_prefix, in_dim, kernel_size);
    }
}

fn add_ebf_layer(
    tensors: &mut BTreeMap<String, LoadedTensor>,
    prefix: &str,
    dim: usize,
    proj_dim: usize,
    ffn_hidden: usize,
    cg_hidden: usize,
    c_kernel_size: usize,
    m_kernel_size: usize,
    use_ls: bool,
) {
    add_norm(tensors, &format!("{prefix}.norm1.weight"), dim);
    add_glu_ffn(tensors, &format!("{prefix}.ffn1"), dim, ffn_hidden);
    add_norm(tensors, &format!("{prefix}.norm2.weight"), dim);
    add_glu_ffn(tensors, &format!("{prefix}.ffn2"), dim, ffn_hidden);
    if use_ls {
        add_norm(tensors, &format!("{prefix}.lay_scale1.scale"), dim);
        add_norm(tensors, &format!("{prefix}.lay_scale2.scale"), dim);
        add_norm(tensors, &format!("{prefix}.lay_scale3.scale"), dim);
    }

    add_norm(tensors, &format!("{prefix}.attn.a_norm.weight"), dim);
    add_norm(tensors, &format!("{prefix}.attn.c_norm.weight"), dim);
    add_attention(tensors, &format!("{prefix}.attn.attn"), dim, proj_dim);
    add_cgmlp(
        tensors,
        &format!("{prefix}.attn.c"),
        dim,
        cg_hidden,
        c_kernel_size,
    );
    add_merge(
        tensors,
        &format!("{prefix}.attn.merge_linear"),
        &format!("{prefix}.attn.merge_dw_conv"),
        dim,
        dim * 2,
        m_kernel_size,
    );
}

fn add_joint_attention_stream(
    tensors: &mut BTreeMap<String, LoadedTensor>,
    prefix: &str,
    label: &str,
    dim: usize,
    proj_dim: usize,
    head_dim: usize,
    qk_norm: bool,
) {
    add_norm(tensors, &format!("{prefix}.{label}_norm.weight"), dim);
    add_linear(tensors, &format!("{prefix}.{label}_qkv"), proj_dim * 3, dim);
    if qk_norm {
        add_norm(
            tensors,
            &format!("{prefix}.{label}_q_norm.weight"),
            head_dim,
        );
        add_norm(
            tensors,
            &format!("{prefix}.{label}_k_norm.weight"),
            head_dim,
        );
    }
    add_linear(tensors, &format!("{prefix}.{label}_out"), dim, proj_dim);
}

fn add_jebf_layer(
    tensors: &mut BTreeMap<String, LoadedTensor>,
    prefix: &str,
    dim: usize,
    proj_dim: usize,
    head_dim: usize,
    c_kernel_size_x: usize,
    c_kernel_size_pool: usize,
    m_kernel_size_x: usize,
    m_kernel_size_pool: usize,
    qk_norm: bool,
) {
    add_norm(tensors, &format!("{prefix}.norm_ffn1_x.weight"), dim);
    add_norm(tensors, &format!("{prefix}.norm_ffn1_pool.weight"), dim);
    add_glu_ffn(tensors, &format!("{prefix}.ffn1_x"), dim, 3);
    add_glu_ffn(tensors, &format!("{prefix}.ffn1_pool"), dim, 3);
    add_norm(tensors, &format!("{prefix}.lay_scale_ffn1_x.scale"), dim);
    add_norm(tensors, &format!("{prefix}.lay_scale_ffn1_pool.scale"), dim);

    add_joint_attention_stream(
        tensors,
        &format!("{prefix}.attn.jattn"),
        "pool",
        dim,
        proj_dim,
        head_dim,
        qk_norm,
    );
    add_joint_attention_stream(
        tensors,
        &format!("{prefix}.attn.jattn"),
        "x",
        dim,
        proj_dim,
        head_dim,
        qk_norm,
    );
    add_norm(tensors, &format!("{prefix}.attn.c_norm_x.weight"), dim);
    add_norm(tensors, &format!("{prefix}.attn.c_norm_pool.weight"), dim);
    add_cgmlp(
        tensors,
        &format!("{prefix}.attn.c_x"),
        dim,
        2,
        c_kernel_size_x,
    );
    add_cgmlp(
        tensors,
        &format!("{prefix}.attn.c_pool"),
        dim,
        2,
        c_kernel_size_pool,
    );
    add_merge(
        tensors,
        &format!("{prefix}.attn.merge_linear_x"),
        &format!("{prefix}.attn.merge_dw_conv_x"),
        dim,
        dim * 2,
        m_kernel_size_x,
    );
    add_merge(
        tensors,
        &format!("{prefix}.attn.merge_linear_pool"),
        &format!("{prefix}.attn.merge_dw_conv_pool"),
        dim,
        dim * 2,
        m_kernel_size_pool,
    );
    add_norm(tensors, &format!("{prefix}.lay_scale_jpac_x.scale"), dim);
    add_norm(tensors, &format!("{prefix}.lay_scale_jpac_pool.scale"), dim);

    add_norm(tensors, &format!("{prefix}.norm_ffn2_x.weight"), dim);
    add_norm(tensors, &format!("{prefix}.norm_ffn2_pool.weight"), dim);
    add_glu_ffn(tensors, &format!("{prefix}.ffn2_x"), dim, 3);
    add_glu_ffn(tensors, &format!("{prefix}.ffn2_pool"), dim, 3);
    add_norm(tensors, &format!("{prefix}.lay_scale_ffn2_x.scale"), dim);
    add_norm(tensors, &format!("{prefix}.lay_scale_ffn2_pool.scale"), dim);
}

fn add_tensor(tensors: &mut BTreeMap<String, LoadedTensor>, name: &str, shape: &[usize]) {
    let len = shape.iter().copied().product::<usize>();
    let data = (0..len)
        .map(|index| index as f32 * 0.01 + name.len() as f32 * 0.001)
        .collect::<Vec<_>>();
    tensors.insert(
        name.to_owned(),
        LoadedTensor {
            shape: shape.to_vec(),
            tensor_type: GGMLType::F32,
            data,
        },
    );
}
