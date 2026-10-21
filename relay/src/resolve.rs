//! PKResolve's current service protocol, backed by the relay's packet cache.

use std::time::Duration;

use axum::{
    extract::{rejection::JsonRejection, State},
    response::IntoResponse,
    Extension, Json,
};
use http::{header::CACHE_CONTROL, StatusCode};
use pkresolve::ResolutionRequest;
use tokio::time::timeout;

use crate::{error::Error, real_ip::RealIp, AppState};

mod packet_provider;

use packet_provider::RelayPacketProvider;

const RESOLUTION_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) async fn resolve(
    State(state): State<AppState>,
    real_ip: Option<Extension<RealIp>>,
    request: Result<Json<ResolutionRequest>, JsonRejection>,
) -> Result<impl IntoResponse, Error> {
    let Json(request) = request?;
    let provider = RelayPacketProvider::new(&state.dht, real_ip.map(|extension| extension.0));

    let resolution = timeout(
        RESOLUTION_TIMEOUT,
        pkresolve::derive_resolution_async(&request, &provider),
    )
    .await
    .map_err(|_| Error::new(StatusCode::GATEWAY_TIMEOUT, "resolution timed out"))?
    .map_err(resolution_error)?;

    let bundle = provider.into_bundle(resolution)?;
    // Reuse is governed by signed record TTLs in the client, not HTTP caches.
    Ok(([(CACHE_CONTROL, "no-store")], Json(bundle)))
}

fn resolution_error(error: pkresolve::Error) -> Error {
    use pkresolve::Error as ResolutionError;

    let status = match error {
        ResolutionError::Provider { source, .. } => {
            // Preserve DHT throttling, availability, and invalid-packet headers.
            return match source.downcast::<Error>() {
                Ok(error) => *error,
                Err(source) => Error::new(StatusCode::BAD_GATEWAY, source),
            };
        }
        ResolutionError::InvalidServiceName(_) | ResolutionError::InvalidRequest(_) => {
            StatusCode::BAD_REQUEST
        }
        ResolutionError::MissingPacket(_) => StatusCode::NOT_FOUND,
        ResolutionError::Unsupported(_)
        | ResolutionError::ServiceUnavailable
        | ResolutionError::NoCompatibleEndpoint
        | ResolutionError::NoAddresses => StatusCode::UNPROCESSABLE_ENTITY,
        ResolutionError::InvalidRecord(_) | ResolutionError::ExpiredRecord => {
            StatusCode::BAD_GATEWAY
        }
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    Error::new(status, error)
}

#[cfg(test)]
mod tests;
