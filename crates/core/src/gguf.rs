use std::collections::HashMap;
use std::fmt::{self, Display};
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Result};

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

pub const KEY_GENERAL_ARCHITECTURE: &str = "general.architecture";
pub const KEY_GENERAL_QUANTIZATION_VERSION: &str = "general.quantization_version";
pub const KEY_GENERAL_ALIGNMENT: &str = "general.alignment";

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGUFVersion {
    V1 = 1,
    V2 = 2,
    V3 = 3,
}

impl Display for GGUFVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V1 => write!(f, "1"),
            Self::V2 => write!(f, "2"),
            Self::V3 => write!(f, "3"),
        }
    }
}

impl TryFrom<u32> for GGUFVersion {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            1 => Ok(Self::V1),
            2 => Ok(Self::V2),
            3 => Ok(Self::V3),
            _ => Err(Error::Format(format!(
                "unsupported GGUF version {value}, expected 1, 2, or 3"
            ))),
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    Count = 19,
}

impl Display for GGMLType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q5_0 => write!(f, "Q5_0"),
            Self::Q5_1 => write!(f, "Q5_1"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q8_1 => write!(f, "Q8_1"),
            Self::Q2K => write!(f, "Q2_K"),
            Self::Q3K => write!(f, "Q3_K"),
            Self::Q4K => write!(f, "Q4_K"),
            Self::Q5K => write!(f, "Q5_K"),
            Self::Q6K => write!(f, "Q6_K"),
            Self::Q8K => write!(f, "Q8_K"),
            Self::I8 => write!(f, "I8"),
            Self::I16 => write!(f, "I16"),
            Self::I32 => write!(f, "I32"),
            Self::Count => write!(f, "COUNT"),
        }
    }
}

impl TryFrom<u32> for GGMLType {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::I8),
            17 => Ok(Self::I16),
            18 => Ok(Self::I32),
            19 => Ok(Self::Count),
            _ => Err(Error::Format(format!(
                "unsupported GGML tensor type {value}"
            ))),
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GGUFMetadataValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl TryFrom<u32> for GGUFMetadataValueType {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => Err(Error::Format(format!(
                "unsupported GGUF metadata value type {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GGUFMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(u8),
    String(String),
    Array(GGUFMetadataArray),
}

#[derive(Debug, Clone, PartialEq)]
pub enum GGUFMetadataArray {
    U8Array(Vec<u8>),
    I8Array(Vec<i8>),
    U16Array(Vec<u16>),
    I16Array(Vec<i16>),
    U32Array(Vec<u32>),
    I32Array(Vec<i32>),
    U64Array(Vec<u64>),
    I64Array(Vec<i64>),
    F32Array(Vec<f32>),
    F64Array(Vec<f64>),
    BoolArray(Vec<u8>),
    StringArray(Vec<String>),
    NestedArray(Vec<GGUFMetadataArray>),
}

#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    metadata_kv: HashMap<String, GGUFMetadataValue>,
}

impl GGUFMetadata {
    pub fn as_hashmap(&self) -> &HashMap<String, GGUFMetadataValue> {
        &self.metadata_kv
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata_kv.get(key) {
            Some(GGUFMetadataValue::U32(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.metadata_kv.get(key) {
            Some(GGUFMetadataValue::String(value)) => Some(value.as_str()),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GGUFTensorInfo {
    name: String,
    dimensions: Vec<usize>,
    typ: GGMLType,
    bytes: Arc<[u8]>,
    start: usize,
    end: usize,
}

impl GGUFTensorInfo {
    fn new(
        name: String,
        dimensions: Vec<usize>,
        typ: GGMLType,
        bytes: Arc<[u8]>,
        start: usize,
        end: usize,
    ) -> Self {
        Self {
            name,
            dimensions,
            typ,
            bytes,
            start,
            end,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn typ(&self) -> GGMLType {
        self.typ
    }

    pub fn data(&self) -> &[u8] {
        &self.bytes[self.start..self.end]
    }
}

#[derive(Debug, Clone)]
struct GGUFHeader {
    version: GGUFVersion,
    tensor_count: usize,
    metadata: GGUFMetadata,
    architecture: String,
}

impl GGUFHeader {
    fn alignment(&self) -> u64 {
        match self.metadata.as_hashmap().get(KEY_GENERAL_ALIGNMENT) {
            Some(GGUFMetadataValue::U64(value)) => *value,
            Some(GGUFMetadataValue::U32(value)) => u64::from(*value),
            Some(GGUFMetadataValue::U16(value)) => u64::from(*value),
            Some(GGUFMetadataValue::U8(value)) => u64::from(*value),
            Some(GGUFMetadataValue::I64(value)) if *value > 0 => *value as u64,
            Some(GGUFMetadataValue::I32(value)) if *value > 0 => *value as u64,
            Some(GGUFMetadataValue::I16(value)) if *value > 0 => *value as u64,
            Some(GGUFMetadataValue::I8(value)) if *value > 0 => *value as u64,
            _ => GGUF_DEFAULT_ALIGNMENT,
        }
    }

    fn quantization_version(&self) -> Option<u32> {
        self.metadata.get_u32(KEY_GENERAL_QUANTIZATION_VERSION)
    }
}

#[derive(Debug, Clone)]
struct GGUFOnDiskTensorInfo {
    name: String,
    dimensions: Vec<usize>,
    typ: GGMLType,
    offset: u64,
}

#[derive(Debug, Clone)]
pub struct GGUFFile {
    header: GGUFHeader,
    tensor_infos: Vec<GGUFTensorInfo>,
    _bytes: Arc<[u8]>,
}

impl GGUFFile {
    fn decode(bytes: Arc<[u8]>) -> Result<Self> {
        let mut reader = GGUFReader::new(&bytes);
        let header = decode_header(&mut reader)?;

        let mut on_disk_tensor_infos = Vec::with_capacity(header.tensor_count);
        for _ in 0..header.tensor_count {
            on_disk_tensor_infos.push(decode_on_disk_tensor_info(&mut reader, header.version)?);
        }

        let tensor_data_start = align_up(reader.position(), header.alignment() as usize);
        if tensor_data_start > bytes.len() {
            return Err(Error::Format(format!(
                "GGUF tensor data starts past end of file: start={}, file_size={}",
                tensor_data_start,
                bytes.len()
            )));
        }

        let tensor_infos =
            convert_tensor_infos(&on_disk_tensor_infos, bytes.clone(), tensor_data_start)?;
        Ok(Self {
            header,
            tensor_infos,
            _bytes: bytes,
        })
    }

    pub fn architecture(&self) -> &str {
        &self.header.architecture
    }

    pub fn quantization_version(&self) -> Option<u32> {
        self.header.quantization_version()
    }

    pub fn version(&self) -> GGUFVersion {
        self.header.version
    }

    pub fn metadata(&self) -> &GGUFMetadata {
        &self.header.metadata
    }

    pub fn tensor_infos(&self) -> &[GGUFTensorInfo] {
        &self.tensor_infos
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<GGUFTensorInfo> {
        self.tensor_infos
            .iter()
            .find(|info| info.name() == name)
            .cloned()
    }
}

pub struct GGUFFileLoader {
    bytes: Arc<[u8]>,
}

impl GGUFFileLoader {
    pub fn new(path: &str, _mlock: bool) -> Result<Self> {
        let bytes = std::fs::read(Path::new(path))?;
        Ok(Self {
            bytes: Arc::<[u8]>::from(bytes),
        })
    }

    pub fn open(&self) -> Result<GGUFFile> {
        GGUFFile::decode(self.bytes.clone())
    }
}

struct GGUFReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> GGUFReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| Error::Format("GGUF offset overflow".to_owned()))?;
        if end > self.bytes.len() {
            return Err(Error::Format(format!(
                "failed to read {len} bytes from GGUF buffer, only {} bytes remain",
                self.bytes.len().saturating_sub(self.position)
            )));
        }

        let bytes = &self.bytes[self.position..end];
        self.position = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_array::<2>()?))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.read_array::<2>()?))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array::<4>()?))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read_array::<4>()?))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array::<8>()?))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read_array::<8>()?))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.read_array::<4>()?))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.read_array::<8>()?))
    }

    fn read_len(&mut self, version: GGUFVersion) -> Result<usize> {
        match version {
            GGUFVersion::V1 => Ok(self.read_u32()? as usize),
            GGUFVersion::V2 | GGUFVersion::V3 => usize::try_from(self.read_u64()?)
                .map_err(|_| Error::Format("GGUF length does not fit into usize".to_owned())),
        }
    }

    fn read_len_array(&mut self, version: GGUFVersion, count: usize) -> Result<Vec<usize>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_len(version)?);
        }
        Ok(values)
    }

    fn read_string(&mut self, version: GGUFVersion) -> Result<String> {
        let len = self.read_len(version)?;
        let bytes = self.read_exact(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|err| Error::Format(format!("invalid UTF-8 in GGUF string: {err}")))
    }

    fn read_value(&mut self, version: GGUFVersion) -> Result<GGUFMetadataValue> {
        let typ = GGUFMetadataValueType::try_from(self.read_u32()?)?;
        match typ {
            GGUFMetadataValueType::U8 => Ok(GGUFMetadataValue::U8(self.read_u8()?)),
            GGUFMetadataValueType::I8 => Ok(GGUFMetadataValue::I8(self.read_i8()?)),
            GGUFMetadataValueType::U16 => Ok(GGUFMetadataValue::U16(self.read_u16()?)),
            GGUFMetadataValueType::I16 => Ok(GGUFMetadataValue::I16(self.read_i16()?)),
            GGUFMetadataValueType::U32 => Ok(GGUFMetadataValue::U32(self.read_u32()?)),
            GGUFMetadataValueType::I32 => Ok(GGUFMetadataValue::I32(self.read_i32()?)),
            GGUFMetadataValueType::U64 => Ok(GGUFMetadataValue::U64(self.read_u64()?)),
            GGUFMetadataValueType::I64 => Ok(GGUFMetadataValue::I64(self.read_i64()?)),
            GGUFMetadataValueType::F32 => Ok(GGUFMetadataValue::F32(self.read_f32()?)),
            GGUFMetadataValueType::F64 => Ok(GGUFMetadataValue::F64(self.read_f64()?)),
            GGUFMetadataValueType::Bool => Ok(GGUFMetadataValue::Bool(self.read_u8()?)),
            GGUFMetadataValueType::String => {
                Ok(GGUFMetadataValue::String(self.read_string(version)?))
            }
            GGUFMetadataValueType::Array => {
                Ok(GGUFMetadataValue::Array(self.read_array_value(version)?))
            }
        }
    }

    fn read_array_value(&mut self, version: GGUFVersion) -> Result<GGUFMetadataArray> {
        let typ = GGUFMetadataValueType::try_from(self.read_u32()?)?;
        let len = self.read_len(version)?;
        match typ {
            GGUFMetadataValueType::U8 => Ok(GGUFMetadataArray::U8Array(self.read_vec_u8(len)?)),
            GGUFMetadataValueType::I8 => Ok(GGUFMetadataArray::I8Array(self.read_vec_i8(len)?)),
            GGUFMetadataValueType::U16 => Ok(GGUFMetadataArray::U16Array(self.read_vec_u16(len)?)),
            GGUFMetadataValueType::I16 => Ok(GGUFMetadataArray::I16Array(self.read_vec_i16(len)?)),
            GGUFMetadataValueType::U32 => Ok(GGUFMetadataArray::U32Array(self.read_vec_u32(len)?)),
            GGUFMetadataValueType::I32 => Ok(GGUFMetadataArray::I32Array(self.read_vec_i32(len)?)),
            GGUFMetadataValueType::U64 => Ok(GGUFMetadataArray::U64Array(self.read_vec_u64(len)?)),
            GGUFMetadataValueType::I64 => Ok(GGUFMetadataArray::I64Array(self.read_vec_i64(len)?)),
            GGUFMetadataValueType::F32 => Ok(GGUFMetadataArray::F32Array(self.read_vec_f32(len)?)),
            GGUFMetadataValueType::F64 => Ok(GGUFMetadataArray::F64Array(self.read_vec_f64(len)?)),
            GGUFMetadataValueType::Bool => Ok(GGUFMetadataArray::BoolArray(self.read_vec_u8(len)?)),
            GGUFMetadataValueType::String => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.read_string(version)?);
                }
                Ok(GGUFMetadataArray::StringArray(values))
            }
            GGUFMetadataValueType::Array => {
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.read_array_value(version)?);
                }
                Ok(GGUFMetadataArray::NestedArray(values))
            }
        }
    }

    fn read_vec_u8(&mut self, count: usize) -> Result<Vec<u8>> {
        Ok(self.read_exact(count)?.to_vec())
    }

    fn read_vec_i8(&mut self, count: usize) -> Result<Vec<i8>> {
        Ok(self
            .read_exact(count)?
            .iter()
            .map(|byte| *byte as i8)
            .collect())
    }

    fn read_vec_u16(&mut self, count: usize) -> Result<Vec<u16>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_u16()?);
        }
        Ok(values)
    }

    fn read_vec_i16(&mut self, count: usize) -> Result<Vec<i16>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_i16()?);
        }
        Ok(values)
    }

    fn read_vec_u32(&mut self, count: usize) -> Result<Vec<u32>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_u32()?);
        }
        Ok(values)
    }

    fn read_vec_i32(&mut self, count: usize) -> Result<Vec<i32>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_i32()?);
        }
        Ok(values)
    }

    fn read_vec_u64(&mut self, count: usize) -> Result<Vec<u64>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_u64()?);
        }
        Ok(values)
    }

    fn read_vec_i64(&mut self, count: usize) -> Result<Vec<i64>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_i64()?);
        }
        Ok(values)
    }

    fn read_vec_f32(&mut self, count: usize) -> Result<Vec<f32>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_f32()?);
        }
        Ok(values)
    }

    fn read_vec_f64(&mut self, count: usize) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_f64()?);
        }
        Ok(values)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0u8; N];
        bytes.copy_from_slice(self.read_exact(N)?);
        Ok(bytes)
    }
}

fn decode_header(reader: &mut GGUFReader<'_>) -> Result<GGUFHeader> {
    let magic = reader.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(Error::Format(format!(
            "invalid GGUF magic number {magic:#x}, expected {GGUF_MAGIC:#x}"
        )));
    }

    let version = GGUFVersion::try_from(reader.read_u32()?)?;
    let tensor_count = reader.read_len(version)?;
    let metadata_kv_count = reader.read_len(version)?;

    let mut metadata_kv = HashMap::with_capacity(metadata_kv_count);
    for _ in 0..metadata_kv_count {
        let key = reader.read_string(version)?;
        let value = reader.read_value(version)?;
        metadata_kv.insert(key, value);
    }
    let metadata = GGUFMetadata { metadata_kv };
    let architecture = metadata
        .get_string(KEY_GENERAL_ARCHITECTURE)
        .ok_or_else(|| Error::Format("missing string metadata general.architecture".to_owned()))?
        .to_owned();

    Ok(GGUFHeader {
        version,
        tensor_count,
        metadata,
        architecture,
    })
}

fn decode_on_disk_tensor_info(
    reader: &mut GGUFReader<'_>,
    version: GGUFVersion,
) -> Result<GGUFOnDiskTensorInfo> {
    let name = reader.read_string(version)?;
    let n_dimensions = reader.read_u32()? as usize;
    let dimensions = reader.read_len_array(version, n_dimensions)?;
    let typ = GGMLType::try_from(reader.read_u32()?)?;
    let offset = reader.read_u64()?;

    Ok(GGUFOnDiskTensorInfo {
        name,
        dimensions,
        typ,
        offset,
    })
}

fn convert_tensor_infos(
    infos: &[GGUFOnDiskTensorInfo],
    bytes: Arc<[u8]>,
    tensor_data_start: usize,
) -> Result<Vec<GGUFTensorInfo>> {
    let mut result = Vec::with_capacity(infos.len());

    for (index, info) in infos.iter().enumerate() {
        let start = tensor_data_start
            .checked_add(u64_to_usize(info.offset)?)
            .ok_or_else(|| Error::Format(format!("tensor offset overflow for `{}`", info.name)))?;

        let next_offset = if index + 1 < infos.len() {
            infos[index + 1].offset
        } else {
            bytes.len().saturating_sub(tensor_data_start) as u64
        };
        let end = tensor_data_start
            .checked_add(u64_to_usize(next_offset)?)
            .ok_or_else(|| Error::Format(format!("tensor end overflow for `{}`", info.name)))?;

        if start > end || end > bytes.len() {
            return Err(Error::Format(format!(
                "tensor `{}` range [{start}, {end}) is outside file bounds {}",
                info.name,
                bytes.len()
            )));
        }

        result.push(GGUFTensorInfo::new(
            info.name.clone(),
            info.dimensions.clone(),
            info.typ,
            bytes.clone(),
            start,
            end,
        ));
    }

    Ok(result)
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }

    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

fn u64_to_usize(value: u64) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::Format(format!("value {value} does not fit into usize")))
}
