use pkarr::ResolvePolicy as InnerResolvePolicy;

use super::constants::*;
use super::error::{publish_error, resolve_error, ClientError};
use super::*;
use std::sync::LazyLock;

// Pre-parsed default relays for better performance
// Fails fast if any default relay URL is invalid
static PARSED_DEFAULT_RELAYS: LazyLock<Vec<url::Url>> = LazyLock::new(|| {
    pkarr::DEFAULT_RELAYS
        .iter()
        .map(|&url_str| url::Url::parse(url_str).expect("valid default relays urls"))
        .collect()
});

/// Controls whether resolution uses cached packets or queries the network.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvePolicy {
    /// Return only a locally cached or relay-cached packet, even if expired.
    CacheOnly,
    /// Return a fresh cached packet or query the network for a fresh result.
    CacheFirst,
    /// Bypass cache reads and query the network for its most recent value.
    NetworkOnly,
}

impl From<ResolvePolicy> for InnerResolvePolicy {
    fn from(policy: ResolvePolicy) -> Self {
        match policy {
            ResolvePolicy::CacheOnly => Self::CacheOnly,
            ResolvePolicy::CacheFirst => Self::CacheFirst,
            ResolvePolicy::NetworkOnly => Self::NetworkOnly,
        }
    }
}

/// Pkarr Client for publishing and resolving signed DNS packets
#[wasm_bindgen]
pub struct Client {
    client: pkarr::Client,
    timeout: std::time::Duration,
}

#[wasm_bindgen]
impl Client {
    /// Create a new client with relay endpoints
    ///
    /// # Arguments
    /// * `relays` - Optional array of relay URLs as strings. If not provided, default relays will be used
    /// * `timeout_ms` - Controls the network timeouts for relay responses (optional, defaults to 30000 ms, min: 1000, max: 300000)
    #[wasm_bindgen(constructor)]
    pub fn new(relays: Option<Array>, timeout_ms: Option<u32>) -> Result<Client, JsValue> {
        console_error_panic_hook::set_once();

        let timeout = Self::validate_and_create_timeout(timeout_ms)?;

        let relay_urls = Self::parse_relay_urls(relays)?;
        let client = pkarr::Client::builder()
            .relays(&relay_urls)
            .map_err(|e| {
                ClientError::ConfigurationError(format!("Invalid relay configuration: {e}"))
            })?
            .request_timeout(timeout)
            .build()
            .map_err(|e| ClientError::ConfigurationError(format!("Failed to build client: {e}")))?;

        Ok(Client { client, timeout })
    }

    /// Publish a signed packet to relays
    ///
    /// # Arguments
    /// * `signed_packet` - The signed packet to publish
    ///
    /// # Returns
    /// The minimum number of DHT nodes known to store the packet. When multiple
    /// backends succeed, this is their maximum reported count rather than a sum,
    /// because different backends may store the packet on the same DHT nodes.
    pub async fn publish(&self, signed_packet: &super::SignedPacket) -> Result<u32, JsValue> {
        self.client
            .publish(&signed_packet.inner)
            .await
            .map_err(publish_error)
    }

    /// Resolve a public key to get the latest signed packet
    ///
    /// # Arguments
    /// * `public_key_str` - The public key as a z-base32 string
    ///
    /// * `policy` - Controls whether cached packets may be returned
    ///
    /// # Returns
    /// The resolved [`SignedPacket`], or JavaScript `null` when no packet is found.
    ///
    /// # Errors
    /// Returns a JavaScript error when the public key is invalid or resolution
    /// fails for a reason other than the packet not being found.
    #[wasm_bindgen(unchecked_return_type = "SignedPacket | null")]
    pub async fn resolve(
        &self,
        public_key_str: &str,
        policy: ResolvePolicy,
    ) -> Result<JsValue, JsValue> {
        let public_key = Self::parse_public_key(public_key_str)?;

        match self.client.resolve(&public_key, policy.into()).await {
            Ok(packet) => Ok(super::SignedPacket::from(packet).into()),
            Err(pkarr::errors::ResolveError::NotFound) => Ok(JsValue::NULL),
            Err(error) => Err(resolve_error(error)),
        }
    }

    /// Get default relay URLs as JavaScript array
    #[wasm_bindgen(js_name = "defaultRelays")]
    pub fn default_relays() -> Array {
        let relays = Array::new();
        for relay in PARSED_DEFAULT_RELAYS.iter() {
            relays.push(&JsValue::from_str(relay.as_str()));
        }
        relays
    }

    /// Get the configured timeout in milliseconds
    #[wasm_bindgen(js_name = "getTimeout")]
    pub fn get_timeout(&self) -> u32 {
        self.timeout.as_millis() as u32
    }
}

// Private helper methods
impl Client {
    /// Validate and create timeout duration from milliseconds
    fn validate_and_create_timeout(
        timeout_ms: Option<u32>,
    ) -> Result<std::time::Duration, JsValue> {
        let timeout_ms = timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS);

        if !(MIN_TIMEOUT_MS..=MAX_TIMEOUT_MS).contains(&timeout_ms) {
            return Err(ClientError::ValidationError {
                context: "timeout".to_string(),
                message: format!(
                    "{timeout_ms} ms (must be between {MIN_TIMEOUT_MS} and {MAX_TIMEOUT_MS} ms)",
                ),
            }
            .into());
        }

        Ok(std::time::Duration::from_millis(timeout_ms as u64))
    }

    /// Parse and validate relay URLs
    fn parse_relay_urls(relays: Option<Array>) -> Result<Vec<url::Url>, JsValue> {
        let relay_urls: Result<Vec<url::Url>, ClientError> = if let Some(relays) = relays {
            // Validate that we have at least one relay
            if relays.length() == 0 {
                return Err(ClientError::ConfigurationError(
                    "At least one relay URL is required".to_string(),
                )
                .into());
            }

            relays
                .iter()
                .enumerate()
                .map(|(index, val)| {
                    let url_str = val
                        .as_string()
                        .ok_or_else(|| ClientError::ValidationError {
                            context: format!("relay URL at index {index}"),
                            message: "Relay URL must be a string".to_string(),
                        })?;

                    if url_str.trim().is_empty() {
                        return Err(ClientError::ValidationError {
                            context: format!("relay URL at index {index}"),
                            message: "Relay URL cannot be empty".to_string(),
                        });
                    }

                    url::Url::parse(&url_str).map_err(|e| ClientError::ParseError {
                        input_type: format!("relay URL at index {index}"),
                        message: e.to_string(),
                    })
                })
                .collect()
        } else {
            // Use pre-parsed default relays
            Ok(PARSED_DEFAULT_RELAYS.clone())
        };

        let relay_urls = relay_urls?;

        if relay_urls.is_empty() {
            return Err(
                ClientError::ConfigurationError("No valid relay URLs found".to_string()).into(),
            );
        }

        Ok(relay_urls)
    }

    /// Parse and validate public key string
    fn parse_public_key(public_key_str: &str) -> Result<pkarr::PublicKey, JsValue> {
        if public_key_str.trim().is_empty() {
            return Err(ClientError::ValidationError {
                context: "public key".to_string(),
                message: "Public key cannot be empty".to_string(),
            }
            .into());
        }

        pkarr::PublicKey::try_from(public_key_str).map_err(|e| {
            ClientError::ParseError {
                input_type: "public key".to_string(),
                message: e.to_string(),
            }
            .into()
        })
    }
}
