//! Main Crate Error

use axum::{
    extract::rejection::{ExtensionRejection, QueryRejection},
    http::StatusCode,
    response::IntoResponse,
};
use pkarr::dht;

#[derive(Debug, Clone)]
pub struct Error {
    // #[serde(with = "serde_status_code")]
    status: StatusCode,
    detail: Option<String>,
}

impl Default for Error {
    fn default() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: None,
        }
    }
}

impl Error {
    pub fn with_status(status: StatusCode) -> Error {
        Self {
            status,
            detail: None,
        }
    }

    /// Create a new [`Error`].
    pub fn new(status_code: StatusCode, message: impl ToString) -> Error {
        Self {
            status: status_code,
            // title: Self::canonical_reason_to_string(&status_code),
            detail: Some(message.to_string()),
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> axum::response::Response {
        match self.detail {
            Some(detail) => (self.status, detail).into_response(),
            _ => (self.status,).into_response(),
        }
    }
}

impl From<QueryRejection> for Error {
    fn from(value: QueryRejection) -> Self {
        Self::new(StatusCode::BAD_REQUEST, value)
    }
}

impl From<ExtensionRejection> for Error {
    fn from(value: ExtensionRejection) -> Self {
        Self::new(StatusCode::BAD_REQUEST, value)
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, value)
    }
}

impl From<dht::PublishError> for Error {
    fn from(error: dht::PublishError) -> Self {
        match error {
            dht::PublishError::Timeout
            | dht::PublishError::NoClosestNodes
            | dht::PublishError::ErrorResponse { .. } => {
                Self::new(StatusCode::INTERNAL_SERVER_ERROR, error)
            }
            dht::PublishError::CasFailed => Self::new(StatusCode::PRECONDITION_FAILED, error),
            dht::PublishError::ConflictRisk => Self::new(StatusCode::PRECONDITION_REQUIRED, error),
            dht::PublishError::NotMostRecent => Self::new(StatusCode::CONFLICT, error),
        }
    }
}

impl From<dht::ResolveReport> for Error {
    fn from(report: dht::ResolveReport) -> Self {
        if report.queried() == 0 {
            Error::new(StatusCode::SERVICE_UNAVAILABLE, "No DHT nodes were queried")
        } else if report.responded() == 0 {
            Error::new(
                StatusCode::GATEWAY_TIMEOUT,
                "No queried DHT nodes responded",
            )
        } else if report.valid_responses() == 0 {
            Error::new(
                StatusCode::BAD_GATEWAY,
                "No responded DHT nodes returned valid response",
            )
        } else {
            Error::with_status(StatusCode::NOT_FOUND)
        }
    }
}
