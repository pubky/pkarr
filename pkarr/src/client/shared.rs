//! shared logic between the native and web relay logic.

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

pub async fn resolve_from_relay(
    http_client: reqwest::Client,
    relay: Url,
    public_key: &PublicKey,
    cache: Option<Box<dyn Cache>>,
    cache_key: &CacheKey,
) -> Result<Option<SignedPacket>, reqwest::Error> {
    let url = format_url(&relay, public_key);

    let mut request = http_client.get(url);

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

    let response = request.send().await?;

    if response.status() == StatusCode::NOT_FOUND {
        return Ok(None);
    }

    let response = response.error_for_status()?;

    if response.content_length().unwrap_or_default() > SignedPacket::MAX_BYTES {
        cross_debug!("Response too large {relay}");

        return Ok(None);
    }

    let payload = response.bytes().await?;

    match SignedPacket::from_relay_payload(public_key, &payload) {
        Ok(signed_packet) => {
            if let Some(cache) = cache {
                cache.put(cache_key, &signed_packet);
            }

            Ok(Some(signed_packet))
        }
        Err(error) => {
            cross_debug!("Invalid signed_packet {relay}:{error}");

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
