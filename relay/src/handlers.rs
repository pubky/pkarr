use std::str::FromStr;

use axum::extract::{Path, Query};
use axum::http::HeaderMap;
use axum::response::Html;
use axum::{extract::State, response::IntoResponse};
use bytes::Bytes;
use futures_lite::StreamExt;
use http::{header, StatusCode};
use httpdate::HttpDate;
use pkarr::mainline::async_dht::AsyncDht;
use pkarr::mainline::errors::ConcurrencyError;
use pkarr::{Cache, CacheKey, PublicKey, ResolvePolicy, SignedPacket, Timestamp};
use serde::Deserialize;

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
    let key = CacheKey::from(&public_key);
    let signed_packet = pkarr::SignedPacket::from_relay_payload(&public_key, &body)
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, Some(error)))?;
    let cas = request_headers
        .get(header::IF_MATCH)
        .map(|h| h.to_str())
        .transpose()
        .map_err(|_| {
            Error::new(
                StatusCode::BAD_REQUEST,
                Some("Invalid IF_MATCH header value"),
            )
        })?
        .map(|s| s.parse::<u64>())
        .transpose()
        .map_err(|_| {
            Error::new(
                StatusCode::BAD_REQUEST,
                Some("Invalid IF_MATCH header value"),
            )
        })?
        .map(Timestamp::from);

    if let Some(cached) = state.cache.get_read_only(&key) {
        if cached.more_recent_than(&signed_packet) {
            return Err(Error::new(
                StatusCode::CONFLICT,
                Some(ConcurrencyError::NotMostRecent),
            ));
        }
    }

    let cas = cas.map(|t| t.as_u64() as i64);
    // TODO: Ratelimit.
    state.dht.put_mutable((&signed_packet).into(), cas).await?;
    update_cache(&state, &key, &signed_packet);

    Ok(StatusCode::NO_CONTENT)
}

pub async fn get(
    State(state): State<AppState>,
    Path(public_key): Path<String>,
    Query(query): Query<GetQuery>,
    request_headers: HeaderMap,
) -> Result<impl IntoResponse, Error> {
    let public_key = PublicKey::try_from(public_key.as_str())
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, Some(error)))?;
    let key = CacheKey::from(&public_key);

    let policy = query.policy.unwrap_or(ResolvePolicy::CacheFirst);
    let cached_packet = state.cache.get_read_only(&key);
    let signed_packet = match policy {
        ResolvePolicy::LocalOrRelayCacheOnly => cached_packet,
        ResolvePolicy::CacheFirst => match cached_packet {
            Some(packet) if !packet.is_expired(state.minimum_ttl, state.maximum_ttl) => {
                Some(packet)
            }
            Some(packet) => {
                resolve(
                    &state.dht,
                    &public_key,
                    Mode::First,
                    Some(packet.timestamp()),
                )
                .await
            }
            None => resolve(&state.dht, &public_key, Mode::First, None).await,
        },
        ResolvePolicy::DhtNetworkOnly => {
            let at_least_that_recent = cached_packet.as_ref().map(SignedPacket::timestamp);
            resolve(
                &state.dht,
                &public_key,
                Mode::MostRecent,
                at_least_that_recent,
            )
            .await
        }
    };

    if let Some(signed_packet) = signed_packet {
        update_cache(&state, &key, &signed_packet);

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
                signed_packet.ttl(state.minimum_ttl, state.maximum_ttl)
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
        response_headers.insert(
            "memento-datetime",
            signed_packet
                .last_seen()
                .format_http_date()
                .try_into()
                .expect("expect memento-datetime to be a valid HeaderValue"),
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

#[derive(Debug, Deserialize)]
pub(crate) struct GetQuery {
    policy: Option<ResolvePolicy>,
}

pub async fn index(State(state): State<AppState>) -> Result<impl IntoResponse, Error> {
    let cache = state.cache;

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

enum Mode {
    First,
    MostRecent,
}

async fn resolve(
    dht: &AsyncDht,
    public_key: &PublicKey,
    mode: Mode,
    at_least_that_recent: Option<Timestamp>,
) -> Option<SignedPacket> {
    // TODO: Ratelimit.
    let key = public_key.as_bytes();
    // It is safe to subtract 1, because even the very first item has a timestamp
    // greater than 0.
    let more_recent_than = at_least_that_recent
        .map(|timestamp| timestamp.as_u64() as i64 - 1)
        .unwrap_or_default();
    let item = match mode {
        Mode::First => {
            dht.get_mutable(key, None, Some(more_recent_than))
                .next()
                .await
        }
        Mode::MostRecent => dht
            .get_mutable_most_recent(key, None)
            .await
            .filter(|item| more_recent_than < item.seq()),
    };
    item.map(SignedPacket::try_from).transpose().ok().flatten()
}

fn update_cache(state: &AppState, key: &CacheKey, signed_packet: &SignedPacket) {
    let _lock = state
        .cache_write_lock
        .lock()
        .expect("AppState cache_write_lock");

    if !state
        .cache
        .get_read_only(key)
        .is_some_and(|cached| cached.more_recent_than(signed_packet))
    {
        state.cache.put(key, signed_packet);
    }
}
