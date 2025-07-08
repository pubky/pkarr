use thiserror::Error;
use wasm_bindgen::JsValue;

/// Generic error types for the WASM module
#[derive(Error, Debug)]
pub enum ClientError {
    /// Input validation failed
    #[error("validation failed in {context}: {message}")]
    ValidationError { context: String, message: String },

    /// Failed to parse input data
    #[error("parse error for {input_type}: {message}")]
    ParseError { input_type: String, message: String },

    /// Network or connectivity error
    #[error("network error: {0}")]
    NetworkError(String),

    /// Configuration or setup error
    #[error("configuration error: {0}")]
    ConfigurationError(String),

    /// Requested feature is not available
    #[error("feature not enabled: {0}")]
    FeatureNotEnabled(String),

    /// Failed to build or create something
    #[error("build error: {0}")]
    BuildError(String),
}

impl From<ClientError> for JsValue {
    fn from(err: ClientError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}
