use js_sys::{Error, Reflect};
use pkarr::errors::{PublishError, ResolveError};
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
        let code = match err {
            ClientError::ValidationError { .. } => "ValidationError",
            ClientError::ParseError { .. } => "ParseError",
            ClientError::NetworkError(_) => "NetworkError",
            ClientError::ConfigurationError(_) => "ConfigurationError",
            ClientError::FeatureNotEnabled(_) => "FeatureNotEnabled",
            ClientError::BuildError(_) => "BuildError",
        };

        js_error("PkarrError", code, err.to_string()).into()
    }
}

pub(crate) fn publish_error(error: PublishError) -> JsValue {
    let message = error.to_string();
    let (code, response) = match error {
        PublishError::NoDhtNodesQueried => ("NoDhtNodesQueried", None),
        PublishError::NoResponses => ("NoResponses", None),
        PublishError::Rejected { code, description } => ("Rejected", Some((code, description))),
        PublishError::NotMostRecent => ("NotMostRecent", None),
        PublishError::UnexpectedResponses => ("UnexpectedResponses", None),
    };

    let error = js_error("PkarrPublishError", code, message);
    if let Some((response_code, description)) = response {
        set_property(
            &error,
            "responseCode",
            &JsValue::from_f64(response_code.into()),
        );
        set_property(&error, "description", &JsValue::from_str(&description));
    }

    error.into()
}

pub(crate) fn resolve_error(error: ResolveError) -> JsValue {
    let message = error.to_string();
    let (code, invalid_sequence) = match error {
        ResolveError::NoDhtNodesQueried => ("NoDhtNodesQueried", None),
        ResolveError::NoResponses => ("NoResponses", None),
        ResolveError::NoUsableResponses => ("NoUsableResponses", None),
        ResolveError::NotFound => ("NotFound", None),
        ResolveError::InvalidSignedPacket { seq } => ("InvalidSignedPacket", Some(seq)),
        ResolveError::UnexpectedResponses => ("UnexpectedResponses", None),
    };

    let error = js_error("PkarrResolveError", code, message);
    if let Some(sequence) = invalid_sequence {
        set_property(&error, "sequence", js_sys::BigInt::from(sequence).as_ref());
    }

    error.into()
}

fn js_error(name: &str, code: &str, message: String) -> Error {
    let error = Error::new(&message);
    error.set_name(name);
    set_property(&error, "code", &JsValue::from_str(code));
    error
}

fn set_property(error: &Error, name: &str, value: &JsValue) {
    let _ = Reflect::set(error.as_ref(), &JsValue::from_str(name), value);
}
