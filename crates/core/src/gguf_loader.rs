use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bytemuck::cast_slice;

use crate::config::{BackboneConfig, GameModelConfig};
use crate::error::{Error, Result};
use crate::gguf::{
    GGMLType, GGUFFile, GGUFFileLoader, GGUFMetadata, GGUFMetadataValue, GGUFVersion,
};

#[derive(Debug, Clone, PartialEq)]
pub struct LoadedTensor {
    pub shape: Vec<usize>,
    pub tensor_type: GGMLType,
    pub data: Vec<f32>,
}

impl LoadedTensor {
    pub fn num_elements(&self) -> usize {
        num_elements(&self.shape).expect("LoadedTensor shape must not overflow")
    }

    pub fn byte_len(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadedGgufModel {
    pub path: PathBuf,
    pub gguf_version: GGUFVersion,
    pub quantization_version: Option<u32>,
    pub metadata_count: usize,
    pub config: GameModelConfig,
    pub tensors: BTreeMap<String, LoadedTensor>,
}

impl LoadedGgufModel {
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor(&self, name: &str) -> Option<&LoadedTensor> {
        self.tensors.get(name)
    }

    pub fn total_parameters(&self) -> usize {
        self.tensors.values().map(LoadedTensor::num_elements).sum()
    }

    pub fn total_loaded_bytes(&self) -> usize {
        self.tensors.values().map(LoadedTensor::byte_len).sum()
    }
}

pub fn load_gguf(path: impl AsRef<Path>) -> Result<LoadedGgufModel> {
    let path = path.as_ref();
    let path_str = path
        .to_str()
        .ok_or_else(|| Error::NonUtf8Path(path.to_path_buf()))?;
    let loader = GGUFFileLoader::new(path_str, false)?;
    let file = loader.open()?;

    Ok(LoadedGgufModel {
        path: path.to_path_buf(),
        gguf_version: file.version(),
        quantization_version: file.quantization_version(),
        metadata_count: file.metadata().as_hashmap().len(),
        config: load_config(file.metadata())?,
        tensors: load_tensors(&file)?,
    })
}

fn load_config(metadata: &GGUFMetadata) -> Result<GameModelConfig> {
    let architecture = required_string(metadata, "general.architecture")?;
    if architecture != "game-me" {
        return Err(Error::UnsupportedArchitecture {
            found: architecture,
        });
    }

    let mut config = GameModelConfig {
        architecture,
        name: optional_string(metadata, "general.name")?.unwrap_or_default(),
        version: optional_string(metadata, "general.version")?.unwrap_or_default(),
        mode: required_string(metadata, "game.model.mode")?,
        embedding_dim: to_i32(metadata, "game.model.embedding_dim")?,
        in_dim: to_i32(metadata, "game.model.in_dim")?,
        estimator_out_dim: to_i32(metadata, "game.model.estimator_out_dim")?,
        region_cycle_len: to_i32(metadata, "game.model.region_cycle_len")?,
        use_languages: required_bool(metadata, "game.model.use_languages")?,
        num_languages: to_i32(metadata, "game.model.num_languages")?,
        ..Default::default()
    };

    fill_ebf_backbone(metadata, "encoder", &mut config.encoder, false)?;
    fill_ebf_backbone(metadata, "segmenter", &mut config.segmenter, true)?;
    fill_jebf_backbone(metadata, "estimator", &mut config.estimator)?;

    let inference = &mut config.inference;
    inference.audio_sample_rate = to_i32(metadata, "game.inference.audio_sample_rate")?;
    inference.hop_size = to_i32(metadata, "game.inference.hop_size")?;
    inference.fft_size = to_i32(metadata, "game.inference.fft_size")?;
    inference.win_size = to_i32(metadata, "game.inference.win_size")?;
    inference.n_mels = to_i32(metadata, "game.inference.spectrogram.num_bins")?;
    inference.fmin = required_f32(metadata, "game.inference.spectrogram.fmin")?;
    inference.fmax = required_f32(metadata, "game.inference.spectrogram.fmax")?;
    inference.spectrogram_type = optional_string(metadata, "game.inference.spectrogram.type")?
        .unwrap_or_else(|| "mel".to_owned());
    inference.midi_min = required_f32(metadata, "game.inference.midi_min")?;
    inference.midi_max = required_f32(metadata, "game.inference.midi_max")?;
    inference.midi_num_bins = to_i32(metadata, "game.inference.midi_num_bins")?;
    inference.midi_std = required_f32(metadata, "game.inference.midi_std")?;

    if config.use_languages {
        if let Some(raw) = optional_string(metadata, "game.inference.lang_map")? {
            if !raw.is_empty() {
                inference.lang_map = serde_json::from_str(&raw)?;
            }
        }
    }

    Ok(config)
}

fn fill_ebf_backbone(
    metadata: &GGUFMetadata,
    section: &str,
    backbone: &mut BackboneConfig,
    expect_latent: bool,
) -> Result<()> {
    let prefix = format!("game.{section}.");
    backbone.cls = optional_string(metadata, &format!("{prefix}cls"))?.unwrap_or_default();
    backbone.dim = to_i32(metadata, &format!("{prefix}dim"))?;
    backbone.num_layers = to_i32(metadata, &format!("{prefix}num_layers"))?;
    backbone.num_heads = to_i32(metadata, &format!("{prefix}num_heads"))?;
    backbone.head_dim = to_i32(metadata, &format!("{prefix}head_dim"))?;
    backbone.c_kernel_size = to_i32(metadata, &format!("{prefix}c_kernel_size"))?;
    backbone.m_kernel_size = to_i32(metadata, &format!("{prefix}m_kernel_size"))?;
    backbone.ffn_type =
        optional_string(metadata, &format!("{prefix}ffn_type"))?.unwrap_or_else(|| "glu".into());
    backbone.use_ls = optional_bool(metadata, &format!("{prefix}use_ls"))?.unwrap_or(true);
    backbone.use_out_norm =
        optional_bool(metadata, &format!("{prefix}use_out_norm"))?.unwrap_or(true);
    backbone.skip_first_ffn =
        optional_bool(metadata, &format!("{prefix}skip_first_ffn"))?.unwrap_or(false);
    backbone.skip_out_ffn =
        optional_bool(metadata, &format!("{prefix}skip_out_ffn"))?.unwrap_or(false);

    if expect_latent {
        if let Some(value) = optional_i64(metadata, &format!("{prefix}latent_layer_idx"))? {
            backbone.return_latent = true;
            backbone.latent_layer_idx = to_i32_value(value, &format!("{prefix}latent_layer_idx"))?;
            backbone.latent_out_dim = to_i32(metadata, &format!("{prefix}latent_out_dim"))?;
        }
    }

    Ok(())
}

fn fill_jebf_backbone(
    metadata: &GGUFMetadata,
    section: &str,
    backbone: &mut BackboneConfig,
) -> Result<()> {
    let prefix = format!("game.{section}.");
    backbone.cls = optional_string(metadata, &format!("{prefix}cls"))?.unwrap_or_default();
    backbone.dim = to_i32(metadata, &format!("{prefix}dim"))?;
    backbone.num_layers = to_i32(metadata, &format!("{prefix}num_layers"))?;
    backbone.num_heads = to_i32(metadata, &format!("{prefix}num_heads"))?;
    backbone.head_dim = to_i32(metadata, &format!("{prefix}head_dim"))?;
    backbone.ffn_type =
        optional_string(metadata, &format!("{prefix}ffn_type"))?.unwrap_or_else(|| "glu".into());
    backbone.use_ls = optional_bool(metadata, &format!("{prefix}use_ls"))?.unwrap_or(true);
    backbone.use_out_norm =
        optional_bool(metadata, &format!("{prefix}use_out_norm"))?.unwrap_or(true);
    backbone.skip_first_ffn =
        optional_bool(metadata, &format!("{prefix}skip_first_ffn"))?.unwrap_or(false);
    backbone.skip_out_ffn =
        optional_bool(metadata, &format!("{prefix}skip_out_ffn"))?.unwrap_or(false);
    backbone.region_token_num = to_i32(metadata, &format!("{prefix}region_token_num"))?;
    backbone.pool_merge_mode = optional_string(metadata, &format!("{prefix}pool_merge_mode"))?
        .unwrap_or_else(|| "mean".into());
    backbone.attn_type =
        optional_string(metadata, &format!("{prefix}attn_type"))?.unwrap_or_else(|| "joint".into());
    backbone.rope_mode =
        optional_string(metadata, &format!("{prefix}rope_mode"))?.unwrap_or_else(|| "mixed".into());
    backbone.qk_norm = optional_bool(metadata, &format!("{prefix}qk_norm"))?.unwrap_or(true);
    backbone.use_region_bias =
        optional_bool(metadata, &format!("{prefix}use_region_bias"))?.unwrap_or(false);
    backbone.c_kernel_size_pool = to_i32(metadata, &format!("{prefix}c_kernel_size_pool"))?;
    backbone.m_kernel_size_pool = to_i32(metadata, &format!("{prefix}m_kernel_size_pool"))?;
    backbone.c_kernel_size_x = to_i32(metadata, &format!("{prefix}c_kernel_size_x"))?;
    backbone.m_kernel_size_x = to_i32(metadata, &format!("{prefix}m_kernel_size_x"))?;
    backbone.use_rope = optional_bool(metadata, &format!("{prefix}use_rope"))?.unwrap_or(true);
    backbone.use_pool_offset =
        optional_bool(metadata, &format!("{prefix}use_pool_offset"))?.unwrap_or(false);
    backbone.theta = optional_f32(metadata, &format!("{prefix}theta"))?.unwrap_or(10_000.0);

    Ok(())
}

fn load_tensors(file: &GGUFFile) -> Result<BTreeMap<String, LoadedTensor>> {
    let mut tensors = BTreeMap::new();

    for tensor in file.tensor_infos() {
        let name = tensor.name().to_owned();
        let shape = tensor
            .dimensions()
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let data = decode_tensor_data(&name, tensor.typ(), tensor.data(), &shape)?;
        let replaced = tensors.insert(
            name,
            LoadedTensor {
                shape,
                tensor_type: tensor.typ(),
                data,
            },
        );
        if replaced.is_some() {
            return Err(Error::message(format!(
                "duplicate tensor name `{}` in GGUF file",
                tensor.name()
            )));
        }
    }

    Ok(tensors)
}

fn decode_tensor_data(
    name: &str,
    typ: GGMLType,
    bytes: &[u8],
    shape: &[usize],
) -> Result<Vec<f32>> {
    match typ {
        GGMLType::F32 => {
            let len = num_elements(shape)?;
            let expected_bytes = len
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| Error::message(format!("tensor `{name}` byte size overflow")))?;
            if bytes.len() < expected_bytes {
                return Err(Error::InvalidTensorSize {
                    name: name.to_owned(),
                    expected_bytes,
                    actual_bytes: bytes.len(),
                });
            }

            Ok(cast_slice::<u8, f32>(&bytes[..expected_bytes]).to_vec())
        }
        _ => Err(Error::UnsupportedTensorType {
            name: name.to_owned(),
            typ: typ.to_string(),
        }),
    }
}

fn required_string(metadata: &GGUFMetadata, key: &str) -> Result<String> {
    optional_string(metadata, key)?.ok_or_else(|| Error::MissingMetadata {
        key: key.to_owned(),
    })
}

fn optional_string(metadata: &GGUFMetadata, key: &str) -> Result<Option<String>> {
    match metadata.as_hashmap().get(key) {
        Some(GGUFMetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(other) => Err(Error::InvalidMetadataType {
            key: key.to_owned(),
            expected: "string",
            found: metadata_type_name(other),
        }),
        None => Ok(None),
    }
}

fn required_bool(metadata: &GGUFMetadata, key: &str) -> Result<bool> {
    optional_bool(metadata, key)?.ok_or_else(|| Error::MissingMetadata {
        key: key.to_owned(),
    })
}

fn optional_bool(metadata: &GGUFMetadata, key: &str) -> Result<Option<bool>> {
    match metadata.as_hashmap().get(key) {
        Some(GGUFMetadataValue::Bool(value)) => Ok(Some(*value != 0)),
        Some(other) => Err(Error::InvalidMetadataType {
            key: key.to_owned(),
            expected: "bool",
            found: metadata_type_name(other),
        }),
        None => Ok(None),
    }
}

fn to_i32(metadata: &GGUFMetadata, key: &str) -> Result<i32> {
    let value = optional_i64(metadata, key)?.ok_or_else(|| Error::MissingMetadata {
        key: key.to_owned(),
    })?;
    to_i32_value(value, key)
}

fn optional_i64(metadata: &GGUFMetadata, key: &str) -> Result<Option<i64>> {
    match metadata.as_hashmap().get(key) {
        Some(GGUFMetadataValue::I8(value)) => Ok(Some(i64::from(*value))),
        Some(GGUFMetadataValue::I16(value)) => Ok(Some(i64::from(*value))),
        Some(GGUFMetadataValue::I32(value)) => Ok(Some(i64::from(*value))),
        Some(GGUFMetadataValue::I64(value)) => Ok(Some(*value)),
        Some(GGUFMetadataValue::U8(value)) => Ok(Some(i64::from(*value))),
        Some(GGUFMetadataValue::U16(value)) => Ok(Some(i64::from(*value))),
        Some(GGUFMetadataValue::U32(value)) => Ok(Some(i64::from(*value))),
        Some(GGUFMetadataValue::U64(value)) => {
            let converted = i64::try_from(*value).map_err(|_| Error::InvalidMetadataValue {
                key: key.to_owned(),
                value: value.to_string(),
                reason: "value does not fit into i64",
            })?;
            Ok(Some(converted))
        }
        Some(GGUFMetadataValue::Bool(_)) => Err(Error::InvalidMetadataType {
            key: key.to_owned(),
            expected: "integer",
            found: "bool",
        }),
        Some(other) => Err(Error::InvalidMetadataType {
            key: key.to_owned(),
            expected: "integer",
            found: metadata_type_name(other),
        }),
        None => Ok(None),
    }
}

fn required_f32(metadata: &GGUFMetadata, key: &str) -> Result<f32> {
    optional_f32(metadata, key)?.ok_or_else(|| Error::MissingMetadata {
        key: key.to_owned(),
    })
}

fn optional_f32(metadata: &GGUFMetadata, key: &str) -> Result<Option<f32>> {
    match metadata.as_hashmap().get(key) {
        Some(GGUFMetadataValue::F32(value)) => Ok(Some(*value)),
        Some(GGUFMetadataValue::F64(value)) => {
            let converted = *value as f32;
            if !converted.is_finite() && value.is_finite() {
                return Err(Error::InvalidMetadataValue {
                    key: key.to_owned(),
                    value: value.to_string(),
                    reason: "f64 value overflows f32 range",
                });
            }
            Ok(Some(converted))
        }
        Some(GGUFMetadataValue::I32(value)) => Ok(Some(*value as f32)),
        Some(GGUFMetadataValue::I64(value)) => Ok(Some(*value as f32)),
        Some(other) => Err(Error::InvalidMetadataType {
            key: key.to_owned(),
            expected: "float",
            found: metadata_type_name(other),
        }),
        None => Ok(None),
    }
}

fn to_i32_value(value: i64, key: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InvalidMetadataValue {
        key: key.to_owned(),
        value: value.to_string(),
        reason: "value does not fit into i32",
    })
}

fn num_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::message(format!("tensor shape {:?} overflows usize", shape)))
    })
}

fn metadata_type_name(value: &GGUFMetadataValue) -> &'static str {
    match value {
        GGUFMetadataValue::U8(_) => "u8",
        GGUFMetadataValue::I8(_) => "i8",
        GGUFMetadataValue::U16(_) => "u16",
        GGUFMetadataValue::I16(_) => "i16",
        GGUFMetadataValue::U32(_) => "u32",
        GGUFMetadataValue::I32(_) => "i32",
        GGUFMetadataValue::U64(_) => "u64",
        GGUFMetadataValue::I64(_) => "i64",
        GGUFMetadataValue::F32(_) => "f32",
        GGUFMetadataValue::F64(_) => "f64",
        GGUFMetadataValue::Bool(_) => "bool",
        GGUFMetadataValue::String(_) => "string",
        GGUFMetadataValue::Array(_) => "array",
    }
}

#[cfg(test)]
mod tests {
    use super::decode_tensor_data;
    use crate::config::InferenceConfig;
    use crate::gguf::GGMLType;

    #[test]
    fn timestep_is_derived_from_hop_and_sample_rate() {
        let config = InferenceConfig {
            audio_sample_rate: 16_000,
            hop_size: 320,
            ..Default::default()
        };

        assert_eq!(config.timestep(), 0.02);
    }

    #[test]
    fn f32_tensor_decode_ignores_alignment_padding() {
        let mut bytes = Vec::new();
        for value in [1.0f32, 2.5, -3.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[0u8; 16]);

        let decoded = decode_tensor_data("x", GGMLType::F32, &bytes, &[3]).unwrap();
        assert_eq!(decoded, vec![1.0, 2.5, -3.0]);
    }

    #[test]
    fn non_f32_tensor_decode_is_rejected() {
        let err = decode_tensor_data("x", GGMLType::F16, &[0; 8], &[4]).unwrap_err();
        assert!(err.to_string().contains("unsupported tensor type"));
    }
}
