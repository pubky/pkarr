use axum::extract::{Extension, Path, Query, State};
use axum::response::Html;
use bytes::Bytes;
use http::StatusCode;
use pkarr::{ResolvePolicy, SignedPacket};
use serde::Deserialize;

use crate::error::Error;
use crate::extractors::{IfModifiedSince, PublicKeyParam};
use crate::real_ip::RealIp;
use crate::response::{PutResponse, SignedPacketResponse};
use crate::AppState;

pub async fn put(
    State(state): State<AppState>,
    Path(PublicKeyParam(public_key)): Path<PublicKeyParam>,
    real_ip: Option<Extension<RealIp>>,
    body: Bytes,
) -> Result<PutResponse, Error> {
    let real_ip = real_ip.as_ref().map(|extension| &extension.0);
    let signed_packet = SignedPacket::from_relay_payload(&public_key, &body)
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, error))?;
    let stored_on = state.dht.publish(&signed_packet, real_ip).await?;

    Ok(PutResponse::new(stored_on))
}

pub async fn get(
    State(state): State<AppState>,
    Path(PublicKeyParam(public_key)): Path<PublicKeyParam>,
    Query(query): Query<GetQuery>,
    real_ip: Option<Extension<RealIp>>,
    IfModifiedSince(if_modified_since): IfModifiedSince,
) -> Result<SignedPacketResponse, Error> {
    let real_ip = real_ip.as_ref().map(|extension| &extension.0);
    let policy = query.policy.unwrap_or(ResolvePolicy::CacheFirst);
    let packet = state
        .dht
        .resolve_packet(&public_key, policy, real_ip)
        .await?;
    let ttl = state.dht.ttl(&packet);

    Ok(SignedPacketResponse::new(packet, ttl, if_modified_since))
}

#[derive(Debug, Deserialize)]
pub(crate) struct GetQuery {
    policy: Option<ResolvePolicy>,
}

pub async fn index(State(state): State<AppState>) -> Html<String> {
    crate::index::render(&state).await
}
