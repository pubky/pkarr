use std::str::FromStr;

use axum::extract::Path;
use axum::http::HeaderMap;
use axum::{extract::State, response::IntoResponse};

use bytes::Bytes;
use http::{header, StatusCode};
use httpdate::HttpDate;
use tracing::error;

use pkarr::{PublicKey, DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL};

use crate::error::{Error, Result};

use super::http_server::AppState;

pub async fn put(
    State(state): State<AppState>,
    Path(public_key): Path<String>,
    body: Bytes,
) -> Result<impl IntoResponse> {
    let public_key = PublicKey::try_from(public_key.as_str())
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, Some(error)))?;

    let signed_packet = pkarr::SignedPacket::from_relay_payload(&public_key, &body)
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, Some(error)))?;

    state
        .client
        .publish(&signed_packet)
        .await
        .map_err(|error| match error {
            pkarr::errors::PublishError::PublishInflight => {
                Error::new(StatusCode::TOO_MANY_REQUESTS, Some(error))
            }
            pkarr::errors::PublishError::NotMostRecent => {
                Error::new(StatusCode::CONFLICT, Some(error))
            }
            pkarr::errors::PublishError::ClientWasShutdown => {
                error!("Pkarr client was shutdown");
                Error::new(StatusCode::INTERNAL_SERVER_ERROR, Some(error))
            }
            error => {
                error!(?error, "Unexpected error");
                Error::new(StatusCode::INTERNAL_SERVER_ERROR, Some(error))
            }
        })?;

    Ok(StatusCode::OK)
}

pub async fn get(
    State(state): State<AppState>,
    Path(public_key): Path<String>,
    headers: HeaderMap,
) -> Result<impl IntoResponse> {
    let public_key = PublicKey::try_from(public_key.as_str())
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, Some(error)))?;

    let mut response = HeaderMap::new().into_response();

    if let Some(signed_packet) = state.client.resolve(&public_key).await? {
        tracing::debug!(?public_key, "cache hit responding with packet!");

        // Handle IF_MODIFIED_SINCE
        if let Some(condition_http_date) = headers
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

        let mut header_map = HeaderMap::new();

        header_map.insert(
            header::CONTENT_TYPE,
            "application/pkarr.org/relays#payload".try_into().unwrap(),
        );
        header_map.insert(
            header::CACHE_CONTROL,
            format!(
                "public, max-age={}",
                signed_packet.ttl(DEFAULT_MINIMUM_TTL, DEFAULT_MAXIMUM_TTL)
            )
            .try_into()
            .unwrap(),
        );
        header_map.insert(
            header::LAST_MODIFIED,
            signed_packet
                .timestamp()
                .format_http_date()
                .try_into()
                .expect("expect last-modified to be a valid HeaderValue"),
        );

        Ok(response)
    } else {
        Err(Error::with_status(StatusCode::NOT_FOUND))
    }
}
