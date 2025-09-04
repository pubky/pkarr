use std::str::FromStr;

use crate::config::RelayMode;
use crate::error::Error;
use crate::AppState;
use axum::extract::{Path, Query};
use axum::http::HeaderMap;
use axum::response::Html;
use axum::{extract::State, response::IntoResponse};
use bytes::Bytes;
use http::{header, StatusCode};
use httpdate::HttpDate;
use pkarr::errors::{ConcurrencyError, PublishError};
use pkarr::Timestamp;
use pkarr::{PublicKey, DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL};
use serde::Deserialize;
use tracing::debug;

#[derive(Deserialize)]
pub struct GetQuery {
    #[serde(default)]
    most_recent: bool,
}

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

    let cas = request_headers
        .get(header::IF_MATCH)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .map(Timestamp::from);

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
    Query(query): Query<GetQuery>,
    request_headers: HeaderMap,
) -> Result<impl IntoResponse, Error> {
    let public_key = PublicKey::try_from(public_key.as_str())
        .map_err(|error| Error::new(StatusCode::BAD_REQUEST, Some(error)))?;

    let signed_packet =
        resolve_packet(&state.client, &public_key, query.most_recent, &state.mode).await;

    if let Some(signed_packet) = signed_packet {
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

async fn resolve_packet(
    client: &pkarr::Client,
    public_key: &PublicKey,
    most_recent: bool,
    mode: &RelayMode,
) -> Option<pkarr::SignedPacket> {
    match mode {
        // Legacy mode ignores most_recent param, always use cache
        RelayMode::LEGACY => client.resolve(public_key).await,
        // Public and Private modes respect most_recent param
        RelayMode::PUBLIC | RelayMode::PRIVATE => {
            if most_recent {
                return client.resolve_most_recent(public_key).await;
            }
            client.resolve(public_key).await
        }
    }
}

pub async fn index(State(state): State<AppState>) -> Result<impl IntoResponse, Error> {
    let cache = state.client.cache().expect("lmdb_cache");

    let size = cache.len();
    let capacity = cache.capacity();
    let utilization = 100.0 * size as f32 / capacity as f32;

    let info = state.client.dht().expect("dht node").info();

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
    <p>This server is a <a href="https://pkarr.org">Pkarr</a> <a href="https://pkarr.org/relays">relay</a>.</p>

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
    dht size estimate : {dht_size} Nodes +-{confidence:.2}% 
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
