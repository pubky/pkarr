//! shared logic between the native and web relay logic.

use bytes::Bytes;
use reqwest::header::HeaderValue;
use reqwest::{header, StatusCode};
use url::Url;

use crate::{Cache, CacheKey, PublicKey, SignedPacket};

macro_rules! cross_debug {
    ($($arg:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        log::debug!($($arg)*);
        #[cfg(not(target_arch = "wasm32"))]
        tracing::debug!($($arg)*);
    };
}

macro_rules! cross_trace {
    ($($arg:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        log::trace!($($arg)*);
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!($($arg)*);
    };
}

pub async fn publish_to_relay(
    http_client: reqwest::Client,
    relay: Url,
    public_key: &PublicKey,
    body: Bytes,
) -> Result<(), reqwest::Error> {
    let url = format_url(&relay, public_key);

    let response = http_client
        .put(url.clone())
        .body(body)
        .send()
        .await
        .inspect_err(|error| {
            cross_debug!("Failed to send a request PUT {url} {error}");
        })?;

    let status = response.status();

    if let Err(error) = response.error_for_status_ref() {
        let text = response.text().await.unwrap_or("".to_string());

        cross_debug!("Got error response for PUT {url} {status} {text}");

        return Err(error);
    };

    if status != StatusCode::OK {
        cross_debug!("Got neither 200 nor >=400 status code {status} for PUT {url}",);
    }

    cross_debug!("Successfully published to {url}");

    Ok(())
}

pub async fn resolve_from_relay(
    http_client: reqwest::Client,
    relay: Url,
    public_key: &PublicKey,
    cache: Option<Box<dyn Cache>>,
    cache_key: &CacheKey,
) -> Result<Option<SignedPacket>, reqwest::Error> {
    let url = format_url(&relay, public_key);

    let mut request = http_client.get(url.clone());

    if let Some(httpdate) = cache
        .as_ref()
        .and_then(|cache| cache.get_read_only(&public_key.into()))
        .map(|c| c.timestamp().format_http_date())
    {
        request = request.header(
            header::IF_MODIFIED_SINCE,
            HeaderValue::from_str(httpdate.as_str()).expect("httpdate to be valid header value"),
        );
    }

    let response = request.send().await.inspect_err(|error| {
        cross_debug!("Failed to send a request GET {url} {error}");
    })?;

    let status = response.status();

    if status == StatusCode::NOT_FOUND {
        cross_trace!("Received a 404 response for GET {url}");

        return Ok(None);
    }

    if let Err(error) = response.error_for_status_ref() {
        let text = response.text().await.unwrap_or("".to_string());

        cross_debug!("Got error response for GET {url} {status} {text}");

        return Err(error);
    };

    if response.status() != StatusCode::OK {
        cross_debug!("Got neither 200 nor >=400 status code {status} for GET {url}",);
    }

    if response.content_length().unwrap_or_default() > SignedPacket::MAX_BYTES {
        cross_debug!("Response too large for GET {url}");

        return Ok(None);
    }

    let payload = response.bytes().await.inspect_err(|error| {
        cross_debug!("Failed to read relay response from GET {url} {error}");
    })?;

    match SignedPacket::from_relay_payload(public_key, &payload) {
        Ok(signed_packet) => {
            if let Some(cache) = cache {
                cache.put(cache_key, &signed_packet);
            }

            Ok(Some(signed_packet))
        }
        Err(error) => {
            cross_debug!("Invalid signed_packet {url}:{error}");

            Ok(None)
        }
    }
}

pub fn format_url(relay: &Url, public_key: &PublicKey) -> Url {
    let mut url = relay.clone();

    let mut segments = url
        .path_segments_mut()
        .expect("Relay url cannot be base, is it http(s)?");

    segments.push(&public_key.to_string());

    drop(segments);

    url
}
