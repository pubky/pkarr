use super::constants::*;
use super::error::ClientError;
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

/// Pkarr Client for publishing and resolving signed DNS packets
#[wasm_bindgen]
pub struct Client {
    relays: Option<std::sync::Arc<pkarr::Client>>,
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
            .unwrap()
            .request_timeout(timeout)
            .build()
            .unwrap();

        Ok(Client {
            relays: Some(std::sync::Arc::new(client)),
            timeout,
        })
    }

    /// Publish a signed packet to relays
    ///
    /// # Arguments
    /// * `signed_packet` - The signed packet to publish
    /// * `cas_timestamp` - Optional compare-and-swap timestamp in milliseconds
    #[wasm_bindgen(js_name = "publish")]
    pub async fn publish(
        &self,
        signed_packet: &super::SignedPacket,
        cas_timestamp: Option<f64>,
    ) -> Result<(), JsValue> {
        let relays = self.get_relays()?;
        let cas = Self::convert_cas_timestamp(cas_timestamp)?;

        relays
            .publish(&signed_packet.inner, cas)
            .await
            .map_err(|e| ClientError::NetworkError(format!("publish failed: {e}")))?;

        Ok(())
    }

    /// Resolve a public key to get the latest signed packet
    ///
    /// # Arguments
    /// * `public_key_str` - The public key as a z-base32 string
    ///
    /// # Returns
    /// * `Option<SignedPacket>` - The signed packet if found
    #[wasm_bindgen(js_name = "resolve")]
    pub async fn resolve(
        &self,
        public_key_str: &str,
    ) -> Result<Option<super::SignedPacket>, JsValue> {
        self.inner_resolve(public_key_str).await
    }

    /// Resolve the most recent signed packet for a public key
    ///
    /// # Arguments
    /// * `public_key_str` - The public key as a z-base32 string
    ///
    /// # Returns
    /// * `Option<SignedPacket>` - The most recent signed packet if found
    #[wasm_bindgen(js_name = "resolveMostRecent")]
    pub async fn resolve_most_recent(
        &self,
        public_key_str: &str,
    ) -> Result<Option<super::SignedPacket>, JsValue> {
        // TODO: This could implement more sophisticated logic to
        // query multiple relays and find the most recent packet
        self.inner_resolve(public_key_str).await
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
    /// Common implementation for resolve methods to avoid duplication
    async fn inner_resolve(
        &self,
        public_key_str: &str,
    ) -> Result<Option<super::SignedPacket>, JsValue> {
        let relays = self.get_relays()?;
        let public_key = Self::parse_public_key(public_key_str)?;

        Ok(relays
            .resolve(&public_key)
            .await
            .map(super::SignedPacket::from))
    }

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

    /// Convert and validate CAS timestamp
    fn convert_cas_timestamp(
        cas_timestamp: Option<f64>,
    ) -> Result<Option<pkarr::Timestamp>, JsValue> {
        match cas_timestamp {
            Some(ts) => {
                if ts < 0.0 {
                    return Err(ClientError::ValidationError {
                        context: "CAS timestamp".to_string(),
                        message: "Timestamp cannot be negative".to_string(),
                    }
                    .into());
                }
                if ts > u64::MAX as f64 {
                    return Err(ClientError::ValidationError {
                        context: "CAS timestamp".to_string(),
                        message: "Timestamp too large".to_string(),
                    }
                    .into());
                }
                Ok(Some(pkarr::Timestamp::from(ts as u64)))
            }
            None => Ok(None),
        }
    }

    /// Get the relays client or return an error
    fn get_relays(&self) -> Result<&pkarr::Client, JsValue> {
        self.relays.as_ref().map(|arc| arc.as_ref()).ok_or_else(|| {
            ClientError::ConfigurationError("No relays configured".to_string()).into()
        })
    }
}
