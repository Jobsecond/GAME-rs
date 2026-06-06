use std::path::Path;

use crate::config::{BackboneConfig, GameModelConfig};
use crate::gguf_loader::{LoadedGgufModel, LoadedTensor};
use crate::{CpuDevice, CpuTensor, Error, Result, Tensor, load_gguf};

#[derive(Clone, Debug)]
pub struct LinearWeights<T> {
    pub weight: T,
    pub bias: T,
}

#[derive(Clone, Debug)]
pub struct GluFfnWeights<T> {
    pub ln1: LinearWeights<T>,
    pub ln2: LinearWeights<T>,
}

#[derive(Clone, Debug)]
pub struct DepthwiseConvWeights<T> {
    pub weight: T,
    pub bias: Option<T>,
    pub kernel_size: usize,
}

#[derive(Clone, Debug)]
pub struct CgmlpWeights<T> {
    pub pw1: LinearWeights<T>,
    pub norm: T,
    pub dw: DepthwiseConvWeights<T>,
    pub pw2: LinearWeights<T>,
}

#[derive(Clone, Debug)]
pub struct AttentionWeights<T> {
    pub q: LinearWeights<T>,
    pub kv: LinearWeights<T>,
    pub out: LinearWeights<T>,
}

#[derive(Clone, Debug)]
pub struct MergeWeights<T> {
    pub linear: LinearWeights<T>,
    pub dw: Option<DepthwiseConvWeights<T>>,
}

#[derive(Clone, Debug)]
pub struct PacWeights<T> {
    pub a_norm: T,
    pub c_norm: T,
    pub attn: AttentionWeights<T>,
    pub cgmlp: CgmlpWeights<T>,
    pub merge: MergeWeights<T>,
}

#[derive(Clone, Debug)]
pub struct ResidualGluWeights<T> {
    pub norm: T,
    pub ffn: GluFfnWeights<T>,
    pub layer_scale: Option<T>,
}

#[derive(Clone, Debug)]
pub struct EbfBlockWeights<T> {
    pub ffn1: Option<ResidualGluWeights<T>>,
    pub pac: PacWeights<T>,
    pub pac_layer_scale: Option<T>,
    pub ffn2: Option<ResidualGluWeights<T>>,
}

#[derive(Clone, Debug)]
pub struct EncoderWeights<T> {
    pub input_proj: LinearWeights<T>,
    pub layers: Vec<EbfBlockWeights<T>>,
    pub output_norm: Option<T>,
    pub output_proj: LinearWeights<T>,
}

#[derive(Clone, Debug)]
pub struct TimeEmbeddingWeights<T> {
    pub layer0: LinearWeights<T>,
    pub layer2: LinearWeights<T>,
}

#[derive(Clone, Debug)]
pub struct LatentHeadWeights<T> {
    pub norm: Option<T>,
    pub proj: LinearWeights<T>,
}

#[derive(Clone, Debug)]
pub struct SegmenterWeights<T> {
    pub noise_embedding: T,
    pub language_embedding: Option<T>,
    pub time_embedding: Option<TimeEmbeddingWeights<T>>,
    pub input_proj: LinearWeights<T>,
    pub layers: Vec<EbfBlockWeights<T>>,
    pub latent: Option<LatentHeadWeights<T>>,
    pub output_norm: Option<T>,
    pub output_proj: LinearWeights<T>,
}

#[derive(Clone, Debug)]
pub struct QkNormWeights<T> {
    pub q: T,
    pub k: T,
}

#[derive(Clone, Debug)]
pub struct JointAttentionStreamWeights<T> {
    pub norm: T,
    pub qkv: LinearWeights<T>,
    pub qk_norm: Option<QkNormWeights<T>>,
    pub out: LinearWeights<T>,
}

#[derive(Clone, Debug)]
pub struct JointAttentionWeights<T> {
    pub pool: JointAttentionStreamWeights<T>,
    pub x: JointAttentionStreamWeights<T>,
}

#[derive(Clone, Debug)]
pub struct PjacWeights<T> {
    pub jattn: JointAttentionWeights<T>,
    pub c_norm_x: T,
    pub c_norm_pool: T,
    pub cgmlp_x: CgmlpWeights<T>,
    pub cgmlp_pool: CgmlpWeights<T>,
    pub merge_x: MergeWeights<T>,
    pub merge_pool: MergeWeights<T>,
}

#[derive(Clone, Debug)]
pub struct JebfBlockWeights<T> {
    pub ffn1_x: Option<ResidualGluWeights<T>>,
    pub ffn1_pool: Option<ResidualGluWeights<T>>,
    pub pjac: PjacWeights<T>,
    pub pjac_layer_scale_x: Option<T>,
    pub pjac_layer_scale_pool: Option<T>,
    pub ffn2_x: Option<ResidualGluWeights<T>>,
    pub ffn2_pool: Option<ResidualGluWeights<T>>,
}

#[derive(Clone, Debug)]
pub struct EstimatorWeights<T> {
    pub input_proj: LinearWeights<T>,
    pub pool_token_gen: T,
    pub region_embedding: T,
    pub layers: Vec<JebfBlockWeights<T>>,
    pub output_norm_x: Option<T>,
    pub output_norm_pool: Option<T>,
    pub output_proj_x: LinearWeights<T>,
    pub output_proj_pool: LinearWeights<T>,
}

#[derive(Clone, Debug)]
pub struct GameModelWeights<T> {
    pub spectrogram_projection: LinearWeights<T>,
    pub encoder: EncoderWeights<T>,
    pub segmenter: SegmenterWeights<T>,
    pub estimator: EstimatorWeights<T>,
}

impl<T: Tensor> GameModelWeights<T> {
    pub fn from_loaded(model: &LoadedGgufModel, device: &T::Device) -> Result<Self> {
        validate_supported_config(&model.config)?;

        let embedding_dim = positive_usize("game.model.embedding_dim", model.config.embedding_dim)?;
        let input_dim = positive_usize("game.model.in_dim", model.config.in_dim)?;
        let estimator_out_dim = positive_usize(
            "game.model.estimator_out_dim",
            model.config.estimator_out_dim,
        )?;
        let region_cycle_len =
            positive_usize("game.model.region_cycle_len", model.config.region_cycle_len)?;
        let language_vocab = checked_add_usize(
            "game.model.num_languages + 1",
            non_negative_usize("game.model.num_languages", model.config.num_languages)?,
            1,
        )?;

        let binder = WeightBinder::<T>::new(model, device);
        Ok(Self {
            spectrogram_projection: binder.linear(
                "spectrogram_projection",
                embedding_dim,
                input_dim,
            )?,
            encoder: bind_encoder(&binder, &model.config.encoder, embedding_dim)?,
            segmenter: bind_segmenter(
                &binder,
                &model.config,
                embedding_dim,
                region_cycle_len,
                language_vocab,
            )?,
            estimator: bind_estimator(
                &binder,
                &model.config,
                embedding_dim,
                estimator_out_dim,
                region_cycle_len,
            )?,
        })
    }
}

pub fn bind_model_weights<T: Tensor>(
    model: &LoadedGgufModel,
    device: &T::Device,
) -> Result<GameModelWeights<T>> {
    GameModelWeights::from_loaded(model, device)
}

pub fn bind_cpu_model_weights(model: &LoadedGgufModel) -> Result<GameModelWeights<CpuTensor>> {
    bind_model_weights(model, &CpuDevice)
}

pub fn load_cpu_model_weights(path: impl AsRef<Path>) -> Result<GameModelWeights<CpuTensor>> {
    let model = load_gguf(path)?;
    bind_cpu_model_weights(&model)
}

struct WeightBinder<'a, T: Tensor> {
    model: &'a LoadedGgufModel,
    device: &'a T::Device,
}

impl<'a, T: Tensor> WeightBinder<'a, T> {
    fn new(model: &'a LoadedGgufModel, device: &'a T::Device) -> Self {
        Self { model, device }
    }

    fn require_loaded(&self, name: &str) -> Result<&'a LoadedTensor> {
        self.model
            .tensor(name)
            .ok_or_else(|| Error::message(format!("missing required tensor `{name}`")))
    }

    fn tensor_with_shapes(
        &self,
        name: &str,
        stored_shape: &[usize],
        runtime_shape: &[usize],
    ) -> Result<T> {
        let loaded = self.require_loaded(name)?;
        if loaded.shape != stored_shape {
            return Err(Error::message(format!(
                "tensor `{name}` has shape {:?}, expected {:?}",
                loaded.shape, stored_shape
            )));
        }

        let stored_elems = checked_num_elements(stored_shape)?;
        let runtime_elems = checked_num_elements(runtime_shape)?;
        if stored_elems != runtime_elems {
            return Err(Error::message(format!(
                "internal shape mismatch for `{name}`: stored {:?} has {stored_elems} elements but runtime {:?} has {runtime_elems}",
                stored_shape, runtime_shape
            )));
        }

        T::from_data(&loaded.data, runtime_shape, self.device).map_err(|err| {
            Error::message(format!(
                "failed to materialize tensor `{name}` as {:?}: {err}",
                runtime_shape
            ))
        })
    }

    fn optional_tensor_with_shapes(
        &self,
        name: &str,
        stored_shape: &[usize],
        runtime_shape: &[usize],
    ) -> Result<Option<T>> {
        if self.model.tensor(name).is_none() {
            return Ok(None);
        }

        self.tensor_with_shapes(name, stored_shape, runtime_shape)
            .map(Some)
    }

    fn norm(&self, name: &str, dim: usize) -> Result<T> {
        self.tensor_with_shapes(name, &[dim], &[dim])
    }

    fn optional_norm(&self, name: &str, dim: usize) -> Result<Option<T>> {
        self.optional_tensor_with_shapes(name, &[dim], &[dim])
    }

    fn linear(&self, prefix: &str, out_dim: usize, in_dim: usize) -> Result<LinearWeights<T>> {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        Ok(LinearWeights {
            weight: self.tensor_with_shapes(
                &weight_name,
                &[out_dim, in_dim],
                &[out_dim, in_dim],
            )?,
            bias: self.tensor_with_shapes(&bias_name, &[out_dim], &[out_dim])?,
        })
    }

    fn pointwise_linear(
        &self,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<LinearWeights<T>> {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        Ok(LinearWeights {
            weight: self.tensor_with_shapes(
                &weight_name,
                &[out_dim, in_dim, 1],
                &[out_dim, in_dim],
            )?,
            bias: self.tensor_with_shapes(&bias_name, &[out_dim], &[out_dim])?,
        })
    }

    fn inferred_linear_with_input(
        &self,
        prefix: &str,
        expected_in_dim: usize,
    ) -> Result<(LinearWeights<T>, usize)> {
        let weight_name = format!("{prefix}.weight");
        let weight = self.require_loaded(&weight_name)?;
        if weight.shape.len() != 2 || weight.shape[1] != expected_in_dim {
            return Err(Error::message(format!(
                "tensor `{weight_name}` has shape {:?}, expected [out_dim, {expected_in_dim}]",
                weight.shape
            )));
        }

        let out_dim = weight.shape[0];
        Ok((self.linear(prefix, out_dim, expected_in_dim)?, out_dim))
    }

    fn inferred_pointwise_linear_with_input(
        &self,
        prefix: &str,
        expected_in_dim: usize,
    ) -> Result<(LinearWeights<T>, usize)> {
        let weight_name = format!("{prefix}.weight");
        let weight = self.require_loaded(&weight_name)?;
        if weight.shape.len() != 3 || weight.shape[1] != expected_in_dim || weight.shape[2] != 1 {
            return Err(Error::message(format!(
                "tensor `{weight_name}` has shape {:?}, expected [out_dim, {expected_in_dim}, 1]",
                weight.shape
            )));
        }

        let out_dim = weight.shape[0];
        Ok((
            self.pointwise_linear(prefix, out_dim, expected_in_dim)?,
            out_dim,
        ))
    }

    fn depthwise(
        &self,
        prefix: &str,
        channels: usize,
        kernel_size: usize,
    ) -> Result<DepthwiseConvWeights<T>> {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        Ok(DepthwiseConvWeights {
            weight: self.tensor_with_shapes(
                &weight_name,
                &[channels, 1, kernel_size],
                &[channels, kernel_size],
            )?,
            bias: self.optional_tensor_with_shapes(&bias_name, &[channels], &[channels])?,
            kernel_size,
        })
    }

    fn merge(
        &self,
        linear_prefix: &str,
        dw_prefix: &str,
        out_dim: usize,
        in_dim: usize,
        kernel_size: usize,
    ) -> Result<MergeWeights<T>> {
        Ok(MergeWeights {
            linear: self.linear(linear_prefix, out_dim, in_dim)?,
            dw: if kernel_size == 0 {
                None
            } else {
                Some(self.depthwise(dw_prefix, in_dim, kernel_size)?)
            },
        })
    }
}

fn bind_encoder<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    cfg: &BackboneConfig,
    embedding_dim: usize,
) -> Result<EncoderWeights<T>> {
    let dim = positive_usize("game.encoder.dim", cfg.dim)?;
    let num_layers = non_negative_usize("game.encoder.num_layers", cfg.num_layers)?;
    let proj_dim = checked_mul_usize(
        "game.encoder.num_heads * game.encoder.head_dim",
        positive_usize("game.encoder.num_heads", cfg.num_heads)?,
        positive_usize("game.encoder.head_dim", cfg.head_dim)?,
    )?;
    let c_kernel_size = positive_usize("game.encoder.c_kernel_size", cfg.c_kernel_size)?;
    let m_kernel_size = non_negative_usize("game.encoder.m_kernel_size", cfg.m_kernel_size)?;

    let mut layers = Vec::with_capacity(num_layers);
    for index in 0..num_layers {
        layers.push(bind_ebf_layer(
            binder,
            &format!("encoder.layers.{index}"),
            cfg,
            dim,
            proj_dim,
            c_kernel_size,
            m_kernel_size,
        )?);
    }

    Ok(EncoderWeights {
        input_proj: binder.linear("encoder.input_proj", dim, embedding_dim)?,
        layers,
        output_norm: if cfg.use_out_norm {
            Some(binder.norm("encoder.output_norm.weight", dim)?)
        } else {
            None
        },
        output_proj: binder.linear(
            "encoder.output_proj",
            checked_mul_usize("2 * game.model.embedding_dim", embedding_dim, 2)?,
            dim,
        )?,
    })
}

fn bind_segmenter<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    cfg: &GameModelConfig,
    embedding_dim: usize,
    region_cycle_len: usize,
    language_vocab: usize,
) -> Result<SegmenterWeights<T>> {
    let backbone = &cfg.segmenter;
    let dim = positive_usize("game.segmenter.dim", backbone.dim)?;
    let num_layers = non_negative_usize("game.segmenter.num_layers", backbone.num_layers)?;
    let proj_dim = checked_mul_usize(
        "game.segmenter.num_heads * game.segmenter.head_dim",
        positive_usize("game.segmenter.num_heads", backbone.num_heads)?,
        positive_usize("game.segmenter.head_dim", backbone.head_dim)?,
    )?;
    let c_kernel_size = positive_usize("game.segmenter.c_kernel_size", backbone.c_kernel_size)?;
    let m_kernel_size = non_negative_usize("game.segmenter.m_kernel_size", backbone.m_kernel_size)?;

    let mut layers = Vec::with_capacity(num_layers);
    for index in 0..num_layers {
        layers.push(bind_ebf_layer(
            binder,
            &format!("segmenter.layers.{index}"),
            backbone,
            dim,
            proj_dim,
            c_kernel_size,
            m_kernel_size,
        )?);
    }

    let time_embedding = if cfg.mode == "d3pm" {
        Some(TimeEmbeddingWeights {
            layer0: binder.linear(
                "time_embedding.0",
                checked_mul_usize("4 * game.model.embedding_dim", embedding_dim, 4)?,
                1,
            )?,
            layer2: binder.linear(
                "time_embedding.2",
                embedding_dim,
                checked_mul_usize("4 * game.model.embedding_dim", embedding_dim, 4)?,
            )?,
        })
    } else {
        None
    };

    let latent = if backbone.return_latent {
        let latent_out_dim =
            positive_usize("game.segmenter.latent_out_dim", backbone.latent_out_dim)?;
        Some(LatentHeadWeights {
            norm: binder.optional_norm("segmenter.latent_norm.weight", dim)?,
            proj: binder.linear("segmenter.latent_proj", latent_out_dim, dim)?,
        })
    } else {
        None
    };

    Ok(SegmenterWeights {
        noise_embedding: binder.tensor_with_shapes(
            "noise_embedding.embedding.weight",
            &[region_cycle_len, embedding_dim],
            &[region_cycle_len, embedding_dim],
        )?,
        language_embedding: if cfg.use_languages {
            Some(binder.tensor_with_shapes(
                "language_embedding.weight",
                &[language_vocab, embedding_dim],
                &[language_vocab, embedding_dim],
            )?)
        } else {
            None
        },
        time_embedding,
        input_proj: binder.linear("segmenter.input_proj", dim, embedding_dim)?,
        layers,
        latent,
        output_norm: if backbone.use_out_norm {
            Some(binder.norm("segmenter.output_norm.weight", dim)?)
        } else {
            None
        },
        output_proj: binder.linear("segmenter.output_proj", 1, dim)?,
    })
}

fn bind_estimator<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    cfg: &GameModelConfig,
    embedding_dim: usize,
    estimator_out_dim: usize,
    region_cycle_len: usize,
) -> Result<EstimatorWeights<T>> {
    let backbone = &cfg.estimator;
    let dim = positive_usize("game.estimator.dim", backbone.dim)?;
    let num_layers = non_negative_usize("game.estimator.num_layers", backbone.num_layers)?;
    let num_heads = positive_usize("game.estimator.num_heads", backbone.num_heads)?;
    let head_dim = positive_usize("game.estimator.head_dim", backbone.head_dim)?;
    let proj_dim = checked_mul_usize(
        "game.estimator.num_heads * game.estimator.head_dim",
        num_heads,
        head_dim,
    )?;
    let c_kernel_size_x =
        positive_usize("game.estimator.c_kernel_size_x", backbone.c_kernel_size_x)?;
    let c_kernel_size_pool = positive_usize(
        "game.estimator.c_kernel_size_pool",
        backbone.c_kernel_size_pool,
    )?;
    let m_kernel_size_x =
        non_negative_usize("game.estimator.m_kernel_size_x", backbone.m_kernel_size_x)?;
    let m_kernel_size_pool = non_negative_usize(
        "game.estimator.m_kernel_size_pool",
        backbone.m_kernel_size_pool,
    )?;
    let region_token_num =
        positive_usize("game.estimator.region_token_num", backbone.region_token_num)?;

    let mut layers = Vec::with_capacity(num_layers);
    for index in 0..num_layers {
        layers.push(bind_jebf_layer(
            binder,
            &format!("estimator.layers.{index}"),
            backbone,
            dim,
            proj_dim,
            head_dim,
            c_kernel_size_x,
            c_kernel_size_pool,
            m_kernel_size_x,
            m_kernel_size_pool,
        )?);
    }

    Ok(EstimatorWeights {
        input_proj: binder.linear("estimator.input_proj", dim, embedding_dim)?,
        pool_token_gen: binder.tensor_with_shapes(
            "estimator.pool_token_gen.emb",
            &[region_token_num, dim],
            &[region_token_num, dim],
        )?,
        region_embedding: binder.tensor_with_shapes(
            "region_embedding.embedding.weight",
            &[region_cycle_len, embedding_dim],
            &[region_cycle_len, embedding_dim],
        )?,
        layers,
        output_norm_x: if backbone.use_out_norm {
            Some(binder.norm("estimator.output_norm_x.weight", dim)?)
        } else {
            None
        },
        output_norm_pool: if backbone.use_out_norm {
            Some(binder.norm("estimator.output_norm_pool.weight", dim)?)
        } else {
            None
        },
        output_proj_x: binder.linear("estimator.output_proj_x", estimator_out_dim, dim)?,
        output_proj_pool: binder.linear("estimator.output_proj_pool", estimator_out_dim, dim)?,
    })
}

fn bind_ebf_layer<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    prefix: &str,
    cfg: &BackboneConfig,
    dim: usize,
    proj_dim: usize,
    c_kernel_size: usize,
    m_kernel_size: usize,
) -> Result<EbfBlockWeights<T>> {
    Ok(EbfBlockWeights {
        ffn1: if cfg.skip_first_ffn {
            None
        } else {
            Some(bind_residual_glu(
                binder,
                &format!("{prefix}.norm1.weight"),
                &format!("{prefix}.ffn1"),
                layer_scale_name(cfg.use_ls, &format!("{prefix}.lay_scale1.scale")),
                dim,
            )?)
        },
        pac: bind_pac(
            binder,
            &format!("{prefix}.attn"),
            dim,
            proj_dim,
            c_kernel_size,
            m_kernel_size,
        )?,
        pac_layer_scale: if cfg.use_ls {
            Some(binder.norm(&format!("{prefix}.lay_scale2.scale"), dim)?)
        } else {
            None
        },
        ffn2: if cfg.skip_out_ffn {
            None
        } else {
            Some(bind_residual_glu(
                binder,
                &format!("{prefix}.norm2.weight"),
                &format!("{prefix}.ffn2"),
                layer_scale_name(cfg.use_ls, &format!("{prefix}.lay_scale3.scale")),
                dim,
            )?)
        },
    })
}

fn bind_pac<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    prefix: &str,
    dim: usize,
    proj_dim: usize,
    c_kernel_size: usize,
    m_kernel_size: usize,
) -> Result<PacWeights<T>> {
    Ok(PacWeights {
        a_norm: binder.norm(&format!("{prefix}.a_norm.weight"), dim)?,
        c_norm: binder.norm(&format!("{prefix}.c_norm.weight"), dim)?,
        attn: bind_attention_with_rope(binder, &format!("{prefix}.attn"), dim, proj_dim)?,
        cgmlp: bind_cgmlp(binder, &format!("{prefix}.c"), dim, c_kernel_size)?,
        merge: binder.merge(
            &format!("{prefix}.merge_linear"),
            &format!("{prefix}.merge_dw_conv"),
            dim,
            checked_mul_usize("2 * backbone.dim", dim, 2)?,
            m_kernel_size,
        )?,
    })
}

fn bind_attention_with_rope<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    prefix: &str,
    dim: usize,
    proj_dim: usize,
) -> Result<AttentionWeights<T>> {
    Ok(AttentionWeights {
        q: binder.linear(&format!("{prefix}.q_linear"), proj_dim, dim)?,
        kv: binder.linear(
            &format!("{prefix}.kv_linear"),
            checked_mul_usize("2 * attention projection dim", proj_dim, 2)?,
            dim,
        )?,
        out: binder.linear(&format!("{prefix}.out_linear"), dim, proj_dim)?,
    })
}

fn bind_residual_glu<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    norm_name: &str,
    ffn_prefix: &str,
    layer_scale_name: Option<&str>,
    dim: usize,
) -> Result<ResidualGluWeights<T>> {
    Ok(ResidualGluWeights {
        norm: binder.norm(norm_name, dim)?,
        ffn: bind_glu_ffn(binder, ffn_prefix, dim)?,
        layer_scale: match layer_scale_name {
            Some(name) => Some(binder.norm(name, dim)?),
            None => None,
        },
    })
}

fn bind_glu_ffn<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    prefix: &str,
    dim: usize,
) -> Result<GluFfnWeights<T>> {
    let (ln1, ln1_out_dim) = binder.inferred_linear_with_input(&format!("{prefix}.ln1"), dim)?;
    ensure_even(&format!("{prefix}.ln1.weight"), ln1_out_dim)?;
    let hidden_dim = ln1_out_dim / 2;

    Ok(GluFfnWeights {
        ln1,
        ln2: binder.linear(&format!("{prefix}.ln2"), dim, hidden_dim)?,
    })
}

fn bind_cgmlp<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    prefix: &str,
    dim: usize,
    kernel_size: usize,
) -> Result<CgmlpWeights<T>> {
    let (pw1, pw1_out_dim) =
        binder.inferred_pointwise_linear_with_input(&format!("{prefix}.pw1"), dim)?;
    ensure_even(&format!("{prefix}.pw1.weight"), pw1_out_dim)?;
    let hidden_dim = pw1_out_dim / 2;

    Ok(CgmlpWeights {
        pw1,
        norm: binder.norm(&format!("{prefix}.norm.weight"), hidden_dim)?,
        dw: binder.depthwise(&format!("{prefix}.dw"), hidden_dim, kernel_size)?,
        pw2: binder.pointwise_linear(&format!("{prefix}.pw2"), dim, hidden_dim)?,
    })
}

fn bind_jebf_layer<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    prefix: &str,
    cfg: &BackboneConfig,
    dim: usize,
    proj_dim: usize,
    head_dim: usize,
    c_kernel_size_x: usize,
    c_kernel_size_pool: usize,
    m_kernel_size_x: usize,
    m_kernel_size_pool: usize,
) -> Result<JebfBlockWeights<T>> {
    Ok(JebfBlockWeights {
        ffn1_x: if cfg.skip_first_ffn {
            None
        } else {
            Some(bind_residual_glu(
                binder,
                &format!("{prefix}.norm_ffn1_x.weight"),
                &format!("{prefix}.ffn1_x"),
                layer_scale_name(cfg.use_ls, &format!("{prefix}.lay_scale_ffn1_x.scale")),
                dim,
            )?)
        },
        ffn1_pool: if cfg.skip_first_ffn {
            None
        } else {
            Some(bind_residual_glu(
                binder,
                &format!("{prefix}.norm_ffn1_pool.weight"),
                &format!("{prefix}.ffn1_pool"),
                layer_scale_name(cfg.use_ls, &format!("{prefix}.lay_scale_ffn1_pool.scale")),
                dim,
            )?)
        },
        pjac: bind_pjac(
            binder,
            &format!("{prefix}.attn"),
            dim,
            proj_dim,
            head_dim,
            cfg.qk_norm,
            c_kernel_size_x,
            c_kernel_size_pool,
            m_kernel_size_x,
            m_kernel_size_pool,
        )?,
        pjac_layer_scale_x: if cfg.use_ls {
            binder.optional_norm(&format!("{prefix}.lay_scale_jpac_x.scale"), dim)?
        } else {
            None
        },
        pjac_layer_scale_pool: if cfg.use_ls {
            binder.optional_norm(&format!("{prefix}.lay_scale_jpac_pool.scale"), dim)?
        } else {
            None
        },
        ffn2_x: if cfg.skip_out_ffn {
            None
        } else {
            Some(bind_residual_glu(
                binder,
                &format!("{prefix}.norm_ffn2_x.weight"),
                &format!("{prefix}.ffn2_x"),
                layer_scale_name(cfg.use_ls, &format!("{prefix}.lay_scale_ffn2_x.scale")),
                dim,
            )?)
        },
        ffn2_pool: if cfg.skip_out_ffn {
            None
        } else {
            Some(bind_residual_glu(
                binder,
                &format!("{prefix}.norm_ffn2_pool.weight"),
                &format!("{prefix}.ffn2_pool"),
                layer_scale_name(cfg.use_ls, &format!("{prefix}.lay_scale_ffn2_pool.scale")),
                dim,
            )?)
        },
    })
}

fn bind_pjac<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    prefix: &str,
    dim: usize,
    proj_dim: usize,
    head_dim: usize,
    qk_norm: bool,
    c_kernel_size_x: usize,
    c_kernel_size_pool: usize,
    m_kernel_size_x: usize,
    m_kernel_size_pool: usize,
) -> Result<PjacWeights<T>> {
    Ok(PjacWeights {
        jattn: bind_joint_attention(
            binder,
            &format!("{prefix}.jattn"),
            dim,
            proj_dim,
            head_dim,
            qk_norm,
        )?,
        c_norm_x: binder.norm(&format!("{prefix}.c_norm_x.weight"), dim)?,
        c_norm_pool: binder.norm(&format!("{prefix}.c_norm_pool.weight"), dim)?,
        cgmlp_x: bind_cgmlp(binder, &format!("{prefix}.c_x"), dim, c_kernel_size_x)?,
        cgmlp_pool: bind_cgmlp(binder, &format!("{prefix}.c_pool"), dim, c_kernel_size_pool)?,
        merge_x: binder.merge(
            &format!("{prefix}.merge_linear_x"),
            &format!("{prefix}.merge_dw_conv_x"),
            dim,
            checked_mul_usize("2 * estimator.dim", dim, 2)?,
            m_kernel_size_x,
        )?,
        merge_pool: binder.merge(
            &format!("{prefix}.merge_linear_pool"),
            &format!("{prefix}.merge_dw_conv_pool"),
            dim,
            checked_mul_usize("2 * estimator.dim", dim, 2)?,
            m_kernel_size_pool,
        )?,
    })
}

fn bind_joint_attention<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    prefix: &str,
    dim: usize,
    proj_dim: usize,
    head_dim: usize,
    qk_norm: bool,
) -> Result<JointAttentionWeights<T>> {
    Ok(JointAttentionWeights {
        pool: bind_joint_attention_stream(
            binder, prefix, "pool", dim, proj_dim, head_dim, qk_norm,
        )?,
        x: bind_joint_attention_stream(binder, prefix, "x", dim, proj_dim, head_dim, qk_norm)?,
    })
}

fn bind_joint_attention_stream<T: Tensor>(
    binder: &WeightBinder<'_, T>,
    prefix: &str,
    label: &str,
    dim: usize,
    proj_dim: usize,
    head_dim: usize,
    qk_norm: bool,
) -> Result<JointAttentionStreamWeights<T>> {
    Ok(JointAttentionStreamWeights {
        norm: binder.norm(&format!("{prefix}.{label}_norm.weight"), dim)?,
        qkv: binder.linear(
            &format!("{prefix}.{label}_qkv"),
            checked_mul_usize("3 * joint attention projection dim", proj_dim, 3)?,
            dim,
        )?,
        qk_norm: if qk_norm {
            Some(QkNormWeights {
                q: binder.norm(&format!("{prefix}.{label}_q_norm.weight"), head_dim)?,
                k: binder.norm(&format!("{prefix}.{label}_k_norm.weight"), head_dim)?,
            })
        } else {
            None
        },
        out: binder.linear(&format!("{prefix}.{label}_out"), dim, proj_dim)?,
    })
}

fn validate_supported_config(cfg: &GameModelConfig) -> Result<()> {
    if cfg.encoder.ffn_type != "glu" {
        return Err(Error::message(format!(
            "unsupported encoder ffn_type `{}` (expected `glu`)",
            cfg.encoder.ffn_type
        )));
    }
    if cfg.segmenter.ffn_type != "glu" {
        return Err(Error::message(format!(
            "unsupported segmenter ffn_type `{}` (expected `glu`)",
            cfg.segmenter.ffn_type
        )));
    }
    if cfg.estimator.ffn_type != "glu" {
        return Err(Error::message(format!(
            "unsupported estimator ffn_type `{}` (expected `glu`)",
            cfg.estimator.ffn_type
        )));
    }
    if cfg.estimator.attn_type != "joint" {
        return Err(Error::message(format!(
            "unsupported estimator attn_type `{}` (expected `joint`)",
            cfg.estimator.attn_type
        )));
    }
    if cfg.estimator.rope_mode != "mixed" {
        return Err(Error::message(format!(
            "unsupported estimator rope_mode `{}` (expected `mixed`)",
            cfg.estimator.rope_mode
        )));
    }
    if cfg.estimator.region_token_num != 1 {
        return Err(Error::message(format!(
            "unsupported estimator region_token_num {} (expected 1)",
            cfg.estimator.region_token_num
        )));
    }
    if !cfg.estimator.qk_norm {
        return Err(Error::message(
            "unsupported estimator configuration: qk_norm=false",
        ));
    }
    if cfg.estimator.use_region_bias {
        return Err(Error::message(
            "unsupported estimator configuration: use_region_bias=true",
        ));
    }
    if cfg.estimator.pool_merge_mode != "mean" {
        return Err(Error::message(format!(
            "unsupported estimator pool_merge_mode `{}` (expected `mean`)",
            cfg.estimator.pool_merge_mode
        )));
    }
    if cfg.segmenter.return_latent {
        let num_layers = non_negative_usize("game.segmenter.num_layers", cfg.segmenter.num_layers)?;
        let latent_layer_idx = positive_usize(
            "game.segmenter.latent_layer_idx",
            cfg.segmenter.latent_layer_idx,
        )?;
        if latent_layer_idx > num_layers {
            return Err(Error::message(format!(
                "game.segmenter.latent_layer_idx ({latent_layer_idx}) exceeds num_layers ({num_layers})"
            )));
        }
    }
    Ok(())
}

fn layer_scale_name(enabled: bool, name: &str) -> Option<&str> {
    enabled.then_some(name)
}

fn positive_usize(field: &str, value: i32) -> Result<usize> {
    let value = non_negative_usize(field, value)?;
    if value == 0 {
        return Err(Error::message(format!(
            "configuration field `{field}` must be > 0"
        )));
    }
    Ok(value)
}

fn non_negative_usize(field: &str, value: i32) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        Error::message(format!(
            "configuration field `{field}` must be >= 0, got {value}"
        ))
    })
}

fn checked_mul_usize(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::message(format!("overflow while computing `{label}`")))
}

fn checked_add_usize(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::message(format!("overflow while computing `{label}`")))
}

fn checked_num_elements(shape: &[usize]) -> Result<usize> {
    let mut total = 1usize;
    for &dim in shape {
        total = total
            .checked_mul(dim)
            .ok_or_else(|| Error::message(format!("shape {:?} overflows usize", shape)))?;
    }
    Ok(total)
}

fn ensure_even(name: &str, dim: usize) -> Result<()> {
    if dim % 2 != 0 {
        return Err(Error::message(format!(
            "tensor `{name}` expected an even output dimension, got {dim}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::{
        GameModelWeights, bind_cpu_model_weights, bind_model_weights, checked_num_elements,
    };
    use crate::config::{BackboneConfig, GameModelConfig};
    use crate::gguf::{GGMLType, GGUFVersion};
    use crate::gguf_loader::{LoadedGgufModel, LoadedTensor};
    use crate::tensor::Tensor;
    use crate::{CpuDevice, CpuTensor};

    #[test]
    fn binds_generic_weights_and_canonicalizes_conv_shapes() {
        let model = fake_loaded_model();
        let weights = bind_model_weights::<CpuTensor>(&model, &CpuDevice).unwrap();

        assert_eq!(weights.spectrogram_projection.weight.shape(), &[4, 3]);
        assert_eq!(weights.encoder.layers.len(), 1);
        assert_eq!(
            weights.encoder.layers[0].pac.cgmlp.pw1.weight.shape(),
            &[4, 4]
        );
        assert_eq!(
            weights.encoder.layers[0].pac.cgmlp.dw.weight.shape(),
            &[2, 3]
        );
        assert_eq!(
            weights
                .segmenter
                .time_embedding
                .as_ref()
                .unwrap()
                .layer0
                .weight
                .shape(),
            &[16, 1]
        );
        assert_eq!(
            weights
                .segmenter
                .latent
                .as_ref()
                .unwrap()
                .proj
                .weight
                .shape(),
            &[2, 4]
        );
        assert_eq!(
            weights.estimator.layers[0]
                .pjac
                .jattn
                .pool
                .qkv
                .weight
                .shape(),
            &[6, 4]
        );
        assert_eq!(
            weights.estimator.layers[0]
                .pjac
                .cgmlp_pool
                .dw
                .weight
                .shape(),
            &[2, 3]
        );
        assert_eq!(weights.estimator.output_proj_pool.weight.shape(), &[5, 4]);
    }

    #[test]
    fn missing_required_tensor_is_reported() {
        let mut model = fake_loaded_model();
        model.tensors.remove("segmenter.output_proj.bias");

        let err = bind_cpu_model_weights(&model).unwrap_err();
        assert!(err.to_string().contains("segmenter.output_proj.bias"));
    }

    #[test]
    fn unsupported_estimator_branch_is_rejected_before_binding() {
        let mut model = fake_loaded_model();
        model.config.estimator.qk_norm = false;

        let err = GameModelWeights::<CpuTensor>::from_loaded(&model, &CpuDevice).unwrap_err();
        assert!(err.to_string().contains("qk_norm=false"));
    }

    fn fake_loaded_model() -> LoadedGgufModel {
        let mut tensors = BTreeMap::new();
        let cfg = fake_config();

        add_linear(&mut tensors, "spectrogram_projection", 4, 3);
        add_linear(&mut tensors, "encoder.input_proj", 4, 4);
        add_ebf_layer(&mut tensors, "encoder.layers.0", 4, 2, 3, 2, 3, 3, true);
        add_norm(&mut tensors, "encoder.output_norm.weight", 4);
        add_linear(&mut tensors, "encoder.output_proj", 8, 4);

        add_embedding(&mut tensors, "noise_embedding.embedding.weight", 3, 4);
        add_embedding(&mut tensors, "language_embedding.weight", 3, 4);
        add_linear(&mut tensors, "time_embedding.0", 16, 1);
        add_linear(&mut tensors, "time_embedding.2", 4, 16);
        add_linear(&mut tensors, "segmenter.input_proj", 4, 4);
        add_ebf_layer(&mut tensors, "segmenter.layers.0", 4, 2, 3, 2, 3, 3, true);
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
            2,
            2,
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
            path: PathBuf::from("synthetic.gguf"),
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
            head_dim: 2,
            c_kernel_size: 3,
            m_kernel_size: 3,
            ffn_type: "glu".to_owned(),
            use_ls: true,
            use_out_norm: true,
            ..Default::default()
        };
        let segmenter = BackboneConfig {
            cls: "modules.backbones.EBF.EBFBackbone".to_owned(),
            dim: 4,
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
            c_kernel_size: 3,
            m_kernel_size: 3,
            ffn_type: "glu".to_owned(),
            use_ls: true,
            use_out_norm: true,
            return_latent: true,
            latent_layer_idx: 1,
            latent_out_dim: 2,
            ..Default::default()
        };
        let estimator = BackboneConfig {
            cls: "modules.backbones.ebf_with_joint_attention.JEBFBackbone".to_owned(),
            dim: 4,
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
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
            name: "synthetic".to_owned(),
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
        let len = checked_num_elements(shape).unwrap();
        let data = (0..len)
            .map(|index| index as f32 + (name.len() as f32 * 0.01))
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
}
