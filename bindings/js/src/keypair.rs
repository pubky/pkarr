use super::constants::*;
use super::error::ClientError;
use super::*;

/// WASM-compatible wrapper for Keypair
#[wasm_bindgen]
pub struct Keypair {
    pub(super) keypair: pkarr::Keypair,
}

#[wasm_bindgen]
impl Keypair {
    /// Generate a cryptographically secure random keypair
    #[wasm_bindgen(constructor)]
    pub fn random() -> Keypair {
        Keypair {
            keypair: pkarr::Keypair::random(),
        }
    }

    /// Create keypair from secret key bytes
    ///
    /// # Arguments
    /// * `secret_key_bytes` - The 32-byte secret key
    #[wasm_bindgen]
    pub fn from_secret_key(secret_key_bytes: &[u8]) -> Result<Keypair, JsValue> {
        Self::validate_secret_key_bytes(secret_key_bytes)?;

        let mut bytes = [0u8; SECRET_KEY_SIZE];
        bytes.copy_from_slice(secret_key_bytes);

        Ok(Keypair {
            keypair: pkarr::Keypair::from_secret_key(&bytes),
        })
    }

    /// Get the public key as a z-base32 encoded string
    ///
    /// This is the format used for pkarr public key identifiers
    #[wasm_bindgen]
    pub fn public_key_string(&self) -> String {
        self.keypair.public_key().to_string()
    }

    /// Get the secret key as raw bytes (32 bytes)
    ///
    /// # Security Warning
    /// Keep secret key data secure and never transmit it over insecure channels
    #[wasm_bindgen]
    pub fn secret_key_bytes(&self) -> Uint8Array {
        Uint8Array::from(&self.keypair.secret_key()[..])
    }

    /// Get the public key as raw bytes (32 bytes)
    #[wasm_bindgen]
    pub fn public_key_bytes(&self) -> Uint8Array {
        Uint8Array::from(&self.keypair.public_key().to_bytes()[..])
    }
}

// Private helper methods
impl Keypair {
    /// Validate secret key bytes
    fn validate_secret_key_bytes(secret_key_bytes: &[u8]) -> Result<(), JsValue> {
        // Check length
        if secret_key_bytes.len() != SECRET_KEY_SIZE {
            return Err(ClientError::ValidationError {
                context: "secret key".to_string(),
                message: format!(
                    "Secret key must be exactly {} bytes, got {}",
                    SECRET_KEY_SIZE,
                    secret_key_bytes.len()
                ),
            }
            .into());
        }

        // Check for all-zero key (insecure)
        if secret_key_bytes.iter().all(|&b| b == 0) {
            return Err(ClientError::ValidationError {
                context: "secret key".to_string(),
                message: "Secret key cannot be all zeros".to_string(),
            }
            .into());
        }

        // Check for other obviously invalid patterns
        if secret_key_bytes.iter().all(|&b| b == 0xFF) {
            return Err(ClientError::ValidationError {
                context: "secret key".to_string(),
                message: "Secret key cannot be all 0xFF bytes".to_string(),
            }
            .into());
        }

        Ok(())
    }
}
