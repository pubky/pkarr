use std::str::FromStr;

use axum::extract::Path;
use axum::http::HeaderMap;
use axum::{extract::State, response::IntoResponse};

use bytes::Bytes;
use http::{header, StatusCode};
use httpdate::HttpDate;
use pkarr::errors::{ConcurrencyError, PublishError};
use pubky_timestamp::Timestamp;
use tracing::debug;

use pkarr::{PublicKey, DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL};

use crate::error::Error;

use crate::AppState;

pub async fn put(
    State(state): State<AppState>,
    Path(public_key): Path<String>,
    request_headers: HeaderMap,
    body: Bytes,
) -> Result<impl IntoResponse, Error> {
    let public_key = PublicKey::try_from(public_key.as_str())
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, Some(error)))?;

    let signed_packet = pkarr::SignedPacket::from_relay_payload(&public_key, &body)
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, Some(error)))?;

    let cas = if let Some(header_value) = request_headers.get(header::IF_UNMODIFIED_SINCE) {
        let httpdate = header_value.to_str().map_err(|_| {
            Error::new(
                StatusCode::BAD_REQUEST,
                Some("Could not parse `IF_UNMODIFIED_SINCE` header"),
            )
        })?;

        Some(Timestamp::parse_http_date(httpdate).map_err(|_| {
            Error::new(
                StatusCode::BAD_REQUEST,
                Some("Could not parse `IF_UNMODIFIED_SINCE` header"),
            )
        })?)
    } else {
        None
    };

    state
        .client
        .publish(&signed_packet, cas)
        .await
        .map_err(|error| match error {
            PublishError::Concurrency(error) => match error {
                ConcurrencyError::NotMostRecent => Error::new(StatusCode::CONFLICT, Some(error)),
                ConcurrencyError::CasFailed => {
                    Error::new(StatusCode::PRECONDITION_FAILED, Some(error))
                }
                ConcurrencyError::ConflictRisk => {
                    Error::new(StatusCode::PRECONDITION_REQUIRED, Some(error))
                }
            },
            PublishError::Query(_) | PublishError::UnexpectedResponses => {
                debug!(?error, "Query error while publishing");

                Error::new(StatusCode::INTERNAL_SERVER_ERROR, Some(error))
            }
        })?;

    Ok(StatusCode::NO_CONTENT)
}

pub async fn get(
    State(state): State<AppState>,
    Path(public_key): Path<String>,
    request_headers: HeaderMap,
) -> Result<impl IntoResponse, Error> {
    let public_key = PublicKey::try_from(public_key.as_str())
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, Some(error)))?;

    if let Some(signed_packet) = state.client.resolve(&public_key).await {
        tracing::debug!(?public_key, "cache hit responding with packet!");

        let mut response_headers = HeaderMap::new();

        response_headers.insert(
            header::CONTENT_TYPE,
            "application/pkarr.org/relays#payload"
                .try_into()
                .expect("pkarr payload content-type header should be valid"),
        );
        response_headers.insert(
            header::CACHE_CONTROL,
            format!(
                "public, max-age={}",
                signed_packet.ttl(DEFAULT_MINIMUM_TTL, DEFAULT_MAXIMUM_TTL)
            )
            .try_into()
            .expect("pkarr cache-control header should be valid."),
        );
        response_headers.insert(
            header::LAST_MODIFIED,
            signed_packet
                .timestamp()
                .format_http_date()
                .try_into()
                .expect("expect last-modified to be a valid HeaderValue"),
        );

        let mut response = response_headers.into_response();

        // Handle IF_MODIFIED_SINCE
        if let Some(condition_http_date) = request_headers
            .get(header::IF_MODIFIED_SINCE)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| HttpDate::from_str(s).ok())
        {
            let entry_http_date: HttpDate = signed_packet.timestamp().into();

            if condition_http_date >= entry_http_date {
                *response.status_mut() = StatusCode::NOT_MODIFIED;
            }
        } else {
            *response.body_mut() = signed_packet.to_relay_payload().into();
        };

        Ok(response)
    } else {
        Err(Error::with_status(StatusCode::NOT_FOUND))
    }
}
