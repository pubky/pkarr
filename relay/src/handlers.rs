use axum::extract::{Extension, Path, Query};
use axum::response::Html;
use axum::{extract::State, response::IntoResponse};
use bytes::Bytes;
use http::StatusCode;
use pkarr::dht::{ReportPolicy, ResolveError as DhtResolveError, ResolveReport, ResolveResponse};
use pkarr::mainline::errors::ConcurrencyError;
use pkarr::{Cache, CacheKey, ResolvePolicy, SignedPacket, Timestamp};
use serde::Deserialize;
use tracing::warn;

use crate::error::Error;
use crate::extractors::{IfMatch, IfModifiedSince, PublicKeyParam};
use crate::real_ip::RealIp;
use crate::response::SignedPacketResponse;
use crate::AppState;

pub async fn put(
    State(state): State<AppState>,
    Path(PublicKeyParam(public_key)): Path<PublicKeyParam>,
    IfMatch(cas): IfMatch,
    real_ip: Option<Extension<RealIp>>,
    body: Bytes,
) -> Result<impl IntoResponse, Error> {
    let real_ip = real_ip.as_ref().map(|extension| &extension.0);
    let signed_packet = pkarr::SignedPacket::from_relay_payload(&public_key, &body)
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, error))?;

    let key = CacheKey::from(&public_key);
    if let Some(cached) = state.cache.get_read_only(&key) {
        if cached.more_recent_than(&signed_packet) {
            return Err(Error::new(
                StatusCode::CONFLICT,
                ConcurrencyError::NotMostRecent,
            ));
        }
    }

    enforce_user_dht_rate_limit(&state, real_ip)?;

    let stored_at = state.dht.publish(&signed_packet, cas).await?;
    let warnings = state.report_policy.classify_publish_result(stored_at);
    if !warnings.is_empty() {
        warn!(
            ?public_key,
            ?warnings,
            "DHT publish completed with warnings"
        );
    }
    update_cache_if_needed(&state, &key, &signed_packet);

    Ok(StatusCode::NO_CONTENT)
}

pub async fn get(
    State(state): State<AppState>,
    Path(PublicKeyParam(public_key)): Path<PublicKeyParam>,
    Query(query): Query<GetQuery>,
    real_ip: Option<Extension<RealIp>>,
    IfModifiedSince(if_modified_since): IfModifiedSince,
) -> Result<SignedPacketResponse, Error> {
    let real_ip = real_ip.as_ref().map(|extension| &extension.0);
    let key = CacheKey::from(&public_key);

    let policy = query.policy.unwrap_or(ResolvePolicy::CacheFirst);
    let packet = resolve_packet(&state, &public_key, key, policy, real_ip).await?;

    update_cache_if_needed(&state, &key, &packet);
    let ttl = packet.ttl(state.minimum_ttl, state.maximum_ttl);
    Ok(SignedPacketResponse::new(packet, ttl, if_modified_since))
}

#[derive(Debug, Deserialize)]
pub(crate) struct GetQuery {
    policy: Option<ResolvePolicy>,
}

pub async fn index(State(state): State<AppState>) -> Result<impl IntoResponse, Error> {
    let cache = &state.cache;

    let size = cache.len();
    let capacity = cache.capacity();
    let utilization = 100.0 * size as f32 / capacity as f32;

    let info = state.dht.info().await;

    let html_content = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pkarr Relay</title>
    <style>
        body {{
            background-color: black;
            color: white;
            font-family: monospace;
            margin: 20px;
        }}
        pre {{
            margin: 0;
        }}
        a {{
            color: white;
        }}
    </style>
</head>
<body>
    <h1>Pkarr Relay</h1>
    <p>This server is a <a href="https://github.com/pubky/pkarr/">Pkarr</a> <a href="https://github.com/pubky/pkarr/blob/main/design/relays.md">relay</a>.</p>

    <h2>Versioning</h2>
    <pre>
    version : {version}
    </pre>

    <h2>Cache stats</h2>
    <pre>
    size:       : {size}
    capacity:   : {capacity}
    utilization : {utilization:.2}%
    </pre>

    <h2>Dht Info</h2>
    <pre>
    node port         : {node_port}
    node firewalled   : {firewalled}
    dht size estimate : {dht_size} Nodes ±{confidence:.2}%
    </pre>
</body>
</html>"#,
        version = env!("CARGO_PKG_VERSION"),
        size = format_number(size),
        capacity = format_number(capacity),
        utilization = utilization,
        confidence = info.dht_size_estimate().1,
        dht_size = format_number(info.dht_size_estimate().0),
        node_port = info
            .public_address()
            .map(|addr| addr.port())
            .unwrap_or(info.local_addr().port()),
        firewalled = info.firewalled(),
    );

    Ok(Html(html_content))
}

fn format_number(num: usize) -> String {
    // Handle large numbers and format with suffixes
    if num >= 1_000_000_000 {
        return format!("{:.1}B", num as f64 / 1_000_000_000.0);
    } else if num >= 1_000_000 {
        return format!("{:.1}M", num as f64 / 1_000_000.0);
    } else if num >= 1_000 {
        return format!("{:.1}K", num as f64 / 1_000.0);
    }

    // Format with commas for thousands
    let num_str = num.to_string();
    let mut result = String::new();
    let len = num_str.len();

    for (i, c) in num_str.chars().enumerate() {
        // Add a comma before every three digits, except for the first part
        if i > 0 && (len - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }

    result
}

fn update_cache_if_needed(state: &AppState, key: &CacheKey, signed_packet: &SignedPacket) {
    let _lock = state
        .cache_write_lock
        .lock()
        .expect("AppState cache_write_lock");

    let should_update = match state.cache.get_read_only(key) {
        None => true,
        Some(cached) if signed_packet.more_recent_than(&cached) => true,
        Some(cached) if signed_packet.is_same_as(&cached) => {
            // The packet is the same but we refresh the last seen.
            signed_packet.last_seen() > cached.last_seen()
        }
        _ => false,
    };

    if should_update {
        state.cache.put(key, signed_packet);
    }
}

async fn resolve_packet(
    state: &AppState,
    public_key: &pkarr::PublicKey,
    key: CacheKey,
    policy: ResolvePolicy,
    real_ip: Option<&RealIp>,
) -> Result<SignedPacket, Error> {
    let cached = state.cache.get(&key);

    match policy {
        ResolvePolicy::LocalOrRelayCacheOnly => {
            cached.ok_or_else(|| Error::with_status(StatusCode::NOT_FOUND))
        }
        ResolvePolicy::CacheFirst => {
            resolve_cache_first(state, public_key, key, cached.as_ref(), real_ip).await
        }
        ResolvePolicy::DhtNetworkOnly => resolve_dht_network_only(state, public_key, real_ip).await,
    }
}

async fn resolve_cache_first(
    state: &AppState,
    public_key: &pkarr::PublicKey,
    key: CacheKey,
    cached: Option<&SignedPacket>,
    real_ip: Option<&RealIp>,
) -> Result<SignedPacket, Error> {
    if let Some(packet) = cached {
        if !packet.is_expired(state.minimum_ttl, state.maximum_ttl) {
            return Ok(packet.clone());
        }
    }

    let more_recent_than = cached.map(SignedPacket::timestamp).and_then(decrement);
    enforce_user_dht_rate_limit(state, real_ip)?;
    let response = state.dht.resolve(public_key, more_recent_than).await;

    resolve_cache_first_dht_response(state, public_key, key, cached, response).await
}

async fn resolve_cache_first_dht_response(
    state: &AppState,
    public_key: &pkarr::PublicKey,
    key: CacheKey,
    cached: Option<&SignedPacket>,
    response: ResolveResponse,
) -> Result<SignedPacket, Error> {
    let first_packet = response
        .first()
        .cloned()
        .map(|packet| apply_cached_floor(Ok(packet), cached));

    match first_packet {
        Some(Ok(packet)) => {
            let state = state.clone();
            let public_key = public_key.clone();
            // Do not waste late responses with potentially newer packets.
            tokio::spawn(async move {
                let resolved = response.complete().await;
                log_resolve_warnings(&public_key, &resolved.report, state.report_policy);
                if let Ok(packet) = resolved.most_recent {
                    update_cache_if_needed(&state, &key, &packet);
                }
            });
            Ok(packet)
        }
        Some(Err(DhtResolveError::NotFound)) | None => {
            let resolved = response.complete().await;
            log_resolve_warnings(public_key, &resolved.report, state.report_policy);
            apply_cached_floor(resolved.most_recent, cached).map_err(Error::from)
        }
        Some(Err(error)) => Err(error.into()),
    }
}

async fn resolve_dht_network_only(
    state: &AppState,
    public_key: &pkarr::PublicKey,
    real_ip: Option<&RealIp>,
) -> Result<SignedPacket, Error> {
    enforce_user_dht_rate_limit(state, real_ip)?;

    // DhtNetworkOnly reports the DHT's current state without using the relay
    // cache as a lower bound or as invalid-sequence suppression.
    let resolved = state.dht.resolve(public_key, None).await.complete().await;
    log_resolve_warnings(public_key, &resolved.report, state.report_policy);

    Ok(resolved.most_recent?)
}

fn apply_cached_floor(
    result: Result<SignedPacket, DhtResolveError>,
    cached: Option<&SignedPacket>,
) -> Result<SignedPacket, DhtResolveError> {
    let Some(cached) = cached else {
        return result;
    };

    match result {
        Ok(packet) if cached.more_recent_than(&packet) => Err(DhtResolveError::NotFound),
        Ok(packet) => Ok(packet),
        Err(DhtResolveError::InvalidSignedPacket { seq })
            if u64::try_from(seq).is_ok_and(|seq| cached.timestamp().as_u64() >= seq) =>
        {
            Err(DhtResolveError::NotFound)
        }
        Err(error) => Err(error),
    }
}

fn decrement(timestamp: Timestamp) -> Option<Timestamp> {
    timestamp.as_u64().checked_sub(1).map(Timestamp::from)
}

fn log_resolve_warnings(
    public_key: &pkarr::PublicKey,
    report: &ResolveReport,
    policy: ReportPolicy,
) {
    let warnings = policy.classify_resolve_report(report);
    if !warnings.is_empty() {
        warn!(
            ?public_key,
            ?warnings,
            "DHT resolve completed with warnings"
        );
    }
}

#[allow(clippy::result_large_err)]
fn enforce_user_dht_rate_limit(state: &AppState, real_ip: Option<&RealIp>) -> Result<(), Error> {
    if let (Some(rate_limiter), Some(real_ip)) = (&state.user_dht_rate_limiter, real_ip) {
        if rate_limiter.is_limited(&real_ip.0) {
            return Err(Error::new(
                StatusCode::TOO_MANY_REQUESTS,
                "Too many requests to DHT nodes",
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{apply_cached_floor, decrement};
    use crate::error::Error;
    use axum::response::IntoResponse;
    use http::StatusCode;
    use pkarr::{dht::ResolveError, Keypair, SignedPacket, Timestamp};

    #[test]
    fn zero_timestamp_has_no_dht_lower_bound() {
        assert_eq!(decrement(Timestamp::from(0)), None);
        assert_eq!(decrement(Timestamp::from(1)), Some(Timestamp::from(0)));
    }

    #[test]
    fn cached_packet_suppresses_invalid_seq_that_is_not_newer() {
        let cached = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .sign(&Keypair::random())
            .unwrap();

        let response = Error::from(
            apply_cached_floor(
                Err(ResolveError::InvalidSignedPacket { seq: 10 }),
                Some(&cached),
            )
            .unwrap_err(),
        )
        .into_response();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert!(!response
            .headers()
            .contains_key(pkarr::PKARR_INVALID_SIGNED_PACKET_SEQ));
    }

    #[test]
    fn newer_invalid_seq_is_reported() {
        let cached = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .sign(&Keypair::random())
            .unwrap();

        let response = Error::from(
            apply_cached_floor(
                Err(ResolveError::InvalidSignedPacket { seq: 11 }),
                Some(&cached),
            )
            .unwrap_err(),
        )
        .into_response();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert!(response
            .headers()
            .contains_key(pkarr::PKARR_INVALID_SIGNED_PACKET_SEQ));
    }

    #[test]
    fn cached_packet_suppresses_older_packet_with_same_timestamp() {
        let keypair = Keypair::random();
        let a = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .txt("key".try_into().unwrap(), "a".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();
        let b = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .txt("key".try_into().unwrap(), "b".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();
        let (cached, older) = if a.more_recent_than(&b) {
            (a, b)
        } else {
            (b, a)
        };

        let result = apply_cached_floor(Ok(older), Some(&cached));

        assert!(matches!(result, Err(ResolveError::NotFound)));
    }

    #[test]
    fn invalid_seq_without_cached_floor_is_reported() {
        let response = Error::from(
            apply_cached_floor(Err(ResolveError::InvalidSignedPacket { seq: 10 }), None)
                .unwrap_err(),
        )
        .into_response();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert!(response
            .headers()
            .contains_key(pkarr::PKARR_INVALID_SIGNED_PACKET_SEQ));
    }
}
