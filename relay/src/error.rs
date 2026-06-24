//! Main Crate Error

use axum::{
    extract::rejection::{ExtensionRejection, QueryRejection},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
};
use pkarr::{dht, PKARR_INVALID_SIGNED_PACKET_SEQ};

#[derive(Debug, Clone)]
pub struct Error {
    status: StatusCode,
    detail: Option<String>,
    headers: HeaderMap,
}

impl Default for Error {
    fn default() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: None,
            headers: HeaderMap::new(),
        }
    }
}

impl Error {
    pub fn with_status(status: StatusCode) -> Error {
        Self {
            status,
            detail: None,
            headers: HeaderMap::new(),
        }
    }

    /// Create a new [`Error`].
    pub fn new(status_code: StatusCode, message: impl ToString) -> Error {
        Self {
            status: status_code,
            // title: Self::canonical_reason_to_string(&status_code),
            detail: Some(message.to_string()),
            headers: HeaderMap::new(),
        }
    }

    pub(crate) fn invalid_signed_packet(seq: i64) -> Error {
        let mut error = Self::new(
            StatusCode::NOT_FOUND,
            format!("DHT mutable item at seq {seq} is not a valid signed packet"),
        );
        error.headers.insert(
            PKARR_INVALID_SIGNED_PACKET_SEQ,
            HeaderValue::from_str(&seq.to_string())
                .expect("i64 string is a valid HTTP header value"),
        );
        error
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> axum::response::Response {
        let mut response = match self.detail {
            Some(detail) => (self.status, detail).into_response(),
            _ => (self.status,).into_response(),
        };
        response.headers_mut().extend(self.headers);
        response
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

impl From<dht::ResolveError> for Error {
    fn from(error: dht::ResolveError) -> Self {
        match error {
            dht::ResolveError::NoNodesQueried => {
                Error::new(StatusCode::SERVICE_UNAVAILABLE, "No DHT nodes were queried")
            }
            dht::ResolveError::NoNodesResponded => Error::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "No queried DHT nodes responded",
            ),
            dht::ResolveError::NoUsableResponses => Error::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "No responded DHT nodes returned usable response",
            ),
            dht::ResolveError::NotFound => Error::with_status(StatusCode::NOT_FOUND),
            dht::ResolveError::InvalidSignedPacket { seq } => Error::invalid_signed_packet(seq),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_signed_packet_error_returns_not_found_with_seq_header() {
        let response = Error::invalid_signed_packet(42).into_response();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            response
                .headers()
                .get(PKARR_INVALID_SIGNED_PACKET_SEQ)
                .unwrap(),
            "42"
        );
    }
}
