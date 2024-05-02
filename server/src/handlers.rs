use axum::extract::Path;
use axum::http::HeaderMap;
use axum::{extract::State, response::IntoResponse};

use bytes::Bytes;
use http::{header, StatusCode};
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

    let signed_packet = pkarr::SignedPacket::from_relay_payload(&public_key, &body).map_err(
        |error| match error {
            pkarr::Error::PacketTooLarge(_) => {
                Error::new(StatusCode::PAYLOAD_TOO_LARGE, Some(error))
            }
            _ => Error::new(StatusCode::BAD_REQUEST, Some(error)),
        },
    )?;

    state
        .client
        .publish(&signed_packet)
        .await
        .map_err(|error| match error {
            pkarr::Error::PublishInflight => Error::new(StatusCode::TOO_MANY_REQUESTS, Some(error)),
            pkarr::Error::NotMostRecent => Error::new(StatusCode::CONFLICT, Some(error)),
            pkarr::Error::DhtIsShutdown => {
                error!("Dht is shutdown");
                Error::with_status(StatusCode::INTERNAL_SERVER_ERROR)
            }
            error => {
                error!(?error, "Unexpected error in pkarr relay PUT");
                Error::with_status(StatusCode::INTERNAL_SERVER_ERROR)
            }
        })?;

    Ok(StatusCode::OK)
}

pub async fn get(
    State(state): State<AppState>,
    Path(public_key): Path<String>,
) -> Result<impl IntoResponse> {
    let public_key = PublicKey::try_from(public_key.as_str())
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, Some(error)))?;

    if let Some(signed_packet) =
        state
            .client
            .resolve(&public_key)
            .await
            .map_err(|error| match error {
                pkarr::Error::DhtIsShutdown => {
                    error!("Dht is shutdown");
                    Error::with_status(StatusCode::INTERNAL_SERVER_ERROR)
                }
                error => {
                    // TODO: do we need this explicit "in x", if we add better tracing tower stuff?
                    error!(?error, "Unexpected error in pkarr relay GET");
                    Error::with_status(StatusCode::INTERNAL_SERVER_ERROR)
                }
            })?
    {
        let body = signed_packet.to_relay_payload();

        let ttl = signed_packet.ttl(DEFAULT_MINIMUM_TTL, DEFAULT_MAXIMUM_TTL);

        let mut header_map = HeaderMap::new();

        header_map.insert(
            header::CONTENT_TYPE,
            "application/pkarr.org/relays#payload".try_into().unwrap(),
        );
        header_map.insert(
            header::CACHE_CONTROL,
            format!("public, max-age={}", ttl).try_into().unwrap(),
        );

        Ok((header_map, body))
    } else {
        Err(Error::with_status(StatusCode::NOT_FOUND))
    }
}
