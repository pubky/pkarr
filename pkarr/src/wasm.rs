//! WASM bindings for pkarr relay functions

use js_sys::{Array, Uint8Array};
use wasm_bindgen::prelude::*;

use crate::{Keypair, PublicKey, SignedPacket};

#[cfg(feature = "relays")]
use crate::client::relays::RelaysClient;

/// Pkarr Client for publishing and resolving signed DNS packets
#[wasm_bindgen]
pub struct Client {
    #[cfg(feature = "relays")]
    relays: Option<std::sync::Arc<RelaysClient>>,
    timeout: std::time::Duration,
}

#[wasm_bindgen]
impl Client {
    /// Create a new client with relay endpoints
    ///
    /// # Arguments
    /// * `relays` - Optional array of relay URLs as strings. If not provided, default relays will be used
    /// * `timeout_ms` - Controls the networks timeouts for relay responses (optional, defaults to 30000 ms)
    #[wasm_bindgen(constructor)]
    pub fn new(relays: Option<Array>, timeout_ms: Option<u32>) -> Result<Client, JsValue> {
        console_error_panic_hook::set_once();

        let timeout = std::time::Duration::from_millis(timeout_ms.unwrap_or(30000) as u64);

        #[cfg(feature = "relays")]
        {
            let relay_urls: Result<Vec<url::Url>, _> = if let Some(relays) = relays {
                relays
                    .iter()
                    .map(|val| {
                        let url_str = val
                            .as_string()
                            .ok_or_else(|| JsValue::from_str("Relay URL must be a string"))?;
                        url::Url::parse(&url_str)
                            .map_err(|e| JsValue::from_str(&format!("Invalid URL: {}", e)))
                    })
                    .collect()
            } else {
                // Use default relays
                crate::DEFAULT_RELAYS
                    .iter()
                    .map(|&url_str| {
                        url::Url::parse(url_str)
                            .map_err(|e| JsValue::from_str(&format!("Invalid default URL: {}", e)))
                    })
                    .collect()
            };

            let relay_urls = relay_urls?;

            if relay_urls.is_empty() {
                return Err(JsValue::from_str("At least one relay URL is required"));
            }

            let relays_client = RelaysClient::new(relay_urls.into_boxed_slice(), timeout);

            Ok(Client {
                relays: Some(std::sync::Arc::new(relays_client)),
                timeout,
            })
        }

        #[cfg(not(feature = "relays"))]
        {
            Err(JsValue::from_str("Relays feature not enabled"))
        }
    }

    /// Publish a signed packet to relays
    ///
    /// # Arguments
    /// * `signed_packet` - The SignedPacket to publish
    /// * `cas_timestamp` - Optional compare-and-swap timestamp in milliseconds
    pub async fn publish(
        &self,
        signed_packet: &SignedPacket,
        cas_timestamp: Option<f64>,
    ) -> Result<(), JsValue> {
        #[cfg(feature = "relays")]
        {
            if let Some(relays) = &self.relays {
                let cas = cas_timestamp.map(|ts| crate::Timestamp::from(ts as u64));

                relays
                    .publish(signed_packet, cas)
                    .await
                    .map_err(|e| JsValue::from_str(&format!("Publish failed: {}", e)))?;

                Ok(())
            } else {
                Err(JsValue::from_str("No relays configured"))
            }
        }

        #[cfg(not(feature = "relays"))]
        {
            Err(JsValue::from_str("Relays feature not enabled"))
        }
    }

    /// Resolve a public key to get the latest signed packet
    ///
    /// # Arguments
    /// * `public_key_str` - The public key as a z-base32 string
    ///
    /// # Returns
    /// * `Option<SignedPacket>` - The signed packet if found
    pub async fn resolve(&self, public_key_str: &str) -> Result<Option<SignedPacket>, JsValue> {
        #[cfg(feature = "relays")]
        {
            if let Some(relays) = &self.relays {
                let public_key = PublicKey::try_from(public_key_str)
                    .map_err(|e| JsValue::from_str(&format!("Invalid public key: {}", e)))?;

                use futures_lite::StreamExt;
                let mut futures = relays.resolve_futures(&public_key, None);
                let result = futures.next().await.flatten();
                Ok(result)
            } else {
                Err(JsValue::from_str("No relays configured"))
            }
        }

        #[cfg(not(feature = "relays"))]
        {
            Err(JsValue::from_str("Relays feature not enabled"))
        }
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
    ) -> Result<Option<SignedPacket>, JsValue> {
        // For simplicity, this is the same as resolve for relay-only implementation
        self.resolve(public_key_str).await
    }

    /// Get default relay URLs as JavaScript array
    #[wasm_bindgen(js_name = "defaultRelays")]
    pub fn default_relays() -> Array {
        let relays = Array::new();
        for relay in crate::DEFAULT_RELAYS.iter() {
            relays.push(&JsValue::from_str(relay));
        }
        relays
    }
}

/// WASM-compatible wrapper for Keypair
#[wasm_bindgen]
pub struct WasmKeypair {
    pub(crate) keypair: Keypair,
}

#[wasm_bindgen]
impl WasmKeypair {
    /// Generate a random keypair
    #[wasm_bindgen(constructor)]
    pub fn random() -> WasmKeypair {
        WasmKeypair {
            keypair: Keypair::random(),
        }
    }

    /// Create keypair from secret key bytes
    #[wasm_bindgen]
    pub fn from_secret_key(secret_key_bytes: &[u8]) -> Result<WasmKeypair, JsValue> {
        if secret_key_bytes.len() != 32 {
            return Err(JsValue::from_str("Secret key must be 32 bytes"));
        }

        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(secret_key_bytes);

        Ok(WasmKeypair {
            keypair: Keypair::from_secret_key(&bytes),
        })
    }

    /// Get the public key as a z-base32 string
    #[wasm_bindgen]
    pub fn public_key_string(&self) -> String {
        self.keypair.public_key().to_string()
    }

    /// Get the secret key bytes
    #[wasm_bindgen]
    pub fn secret_key_bytes(&self) -> Uint8Array {
        Uint8Array::from(&self.keypair.secret_key()[..])
    }

    /// Get the public key bytes
    #[wasm_bindgen]
    pub fn public_key_bytes(&self) -> Uint8Array {
        Uint8Array::from(&self.keypair.public_key().to_bytes()[..])
    }
}

/// Utility functions
#[wasm_bindgen]
pub struct WasmUtils;

#[wasm_bindgen]
impl WasmUtils {
    /// Parse a signed packet from bytes and return the SignedPacket
    #[wasm_bindgen]
    pub fn parse_signed_packet(bytes: &[u8]) -> Result<SignedPacket, JsValue> {
        SignedPacket::deserialize(bytes)
            .map_err(|e| JsValue::from_str(&format!("Invalid signed packet: {}", e)))
    }

    /// Validate a public key string
    #[wasm_bindgen]
    pub fn validate_public_key(public_key_str: &str) -> bool {
        PublicKey::try_from(public_key_str).is_ok()
    }

    /// Get default relay URLs
    #[wasm_bindgen]
    pub fn default_relays() -> Array {
        let relays = Array::new();
        for relay in crate::DEFAULT_RELAYS.iter() {
            relays.push(&JsValue::from_str(relay));
        }
        relays
    }
}

/// Initialize console error panic hook for better debugging
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}
