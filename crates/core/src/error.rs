use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Candle(candle_core::Error),
    Format(String),
    MissingMetadata {
        key: String,
    },
    InvalidMetadataType {
        key: String,
        expected: &'static str,
        found: &'static str,
    },
    InvalidMetadataValue {
        key: String,
        value: String,
        reason: &'static str,
    },
    UnsupportedArchitecture {
        found: String,
    },
    UnsupportedTensorType {
        name: String,
        typ: String,
    },
    InvalidTensorSize {
        name: String,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    NonUtf8Path(PathBuf),
    Message(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Json(err) => write!(f, "{err}"),
            Self::Candle(err) => write!(f, "{err}"),
            Self::Format(message) => write!(f, "{message}"),
            Self::MissingMetadata { key } => write!(f, "missing GGUF metadata key `{key}`"),
            Self::InvalidMetadataType {
                key,
                expected,
                found,
            } => write!(
                f,
                "invalid GGUF metadata type for `{key}`: expected {expected}, found {found}"
            ),
            Self::InvalidMetadataValue { key, value, reason } => {
                write!(
                    f,
                    "invalid GGUF metadata value for `{key}` (`{value}`): {reason}"
                )
            }
            Self::UnsupportedArchitecture { found } => write!(
                f,
                "unsupported GGUF architecture `{found}` (expected `game-me`)"
            ),
            Self::UnsupportedTensorType { name, typ } => {
                write!(f, "unsupported tensor type for `{name}`: {typ}")
            }
            Self::InvalidTensorSize {
                name,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "tensor `{name}` is too small: expected at least {expected_bytes} bytes, got {actual_bytes}"
            ),
            Self::NonUtf8Path(path) => write!(f, "path is not valid UTF-8: {}", path.display()),
            Self::Message(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::Candle(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<candle_core::Error> for Error {
    fn from(value: candle_core::Error) -> Self {
        Self::Candle(value)
    }
}
