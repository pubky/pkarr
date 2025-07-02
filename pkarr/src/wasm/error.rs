use wasm_bindgen::JsValue;

/// Generic error types for the WASM module
#[derive(Debug)]
pub enum ClientError {
    /// Input validation failed
    ValidationError { context: String, message: String },

    /// Failed to parse input data
    ParseError { input_type: String, message: String },

    /// Network or connectivity error
    NetworkError(String),

    /// Configuration or setup error
    ConfigurationError(String),

    /// Requested feature is not available
    FeatureNotEnabled(String),

    /// Failed to build or create something
    BuildError(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::ValidationError { context, message } => {
                write!(f, "Validation error in {}: {}", context, message)
            }
            ClientError::ParseError {
                input_type,
                message,
            } => {
                write!(f, "Parse error for {}: {}", input_type, message)
            }
            ClientError::NetworkError(msg) => {
                write!(f, "Network error: {}", msg)
            }
            ClientError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            ClientError::FeatureNotEnabled(feature) => {
                write!(f, "Feature not available: {}", feature)
            }
            ClientError::BuildError(msg) => {
                write!(f, "Build error: {}", msg)
            }
        }
    }
}

impl From<ClientError> for JsValue {
    fn from(err: ClientError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}
