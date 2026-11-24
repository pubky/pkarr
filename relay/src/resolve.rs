//! PKResolve's current service protocol, backed by the relay's packet cache.

use std::time::Duration;

use axum::{
    extract::{rejection::JsonRejection, State},
    response::IntoResponse,
    Extension, Json,
};
use http::{header::CACHE_CONTROL, StatusCode};
use pkresolve::{
    service_protocol::{RelayDiagnostic, ResolveBundle, ResolveErrorBundle, BUNDLE_SCHEMA},
    ResolutionRequest,
};
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

    let result = match timeout(
        RESOLUTION_TIMEOUT,
        pkresolve::derive_resolution_async(&request, &provider),
    )
    .await
    {
        Ok(result) => result.map_err(resolution_error),
        Err(_) => Err(Error::new(
            StatusCode::GATEWAY_TIMEOUT,
            "resolution timed out",
        )),
    };

    let packets = provider.into_packets();
    let response = match result {
        Ok(resolution) => Json(ResolveBundle {
            schema: BUNDLE_SCHEMA,
            packets,
            resolution: resolution.into(),
        })
        .into_response(),
        Err(error) if packets.is_empty() => return Err(error),
        Err(error) => {
            let (status, headers, detail) = error.into_parts();
            let bundle = ResolveErrorBundle {
                schema: BUNDLE_SCHEMA,
                packets,
                error: RelayDiagnostic {
                    status: status.as_u16(),
                    message: detail.unwrap_or_else(|| {
                        status
                            .canonical_reason()
                            .unwrap_or("Resolution failed")
                            .to_owned()
                    }),
                },
            };
            (status, headers, Json(bundle)).into_response()
        }
    };
    Ok(([(CACHE_CONTROL, "no-store")], response))
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
