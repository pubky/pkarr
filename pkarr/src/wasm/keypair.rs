use super::*;

/// WASM-compatible wrapper for Keypair
#[wasm_bindgen]
pub struct Keypair {
    pub(super) keypair: NativeKeypair,
}

#[wasm_bindgen]
impl Keypair {
    /// Generate a random keypair
    #[wasm_bindgen(constructor)]
    pub fn random() -> Keypair {
        Keypair {
            keypair: NativeKeypair::random(),
        }
    }

    /// Create keypair from secret key bytes
    #[wasm_bindgen]
    pub fn from_secret_key(secret_key_bytes: &[u8]) -> Result<Keypair, JsValue> {
        if secret_key_bytes.len() != 32 {
            return Err(JsValue::from_str("Secret key must be 32 bytes"));
        }

        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(secret_key_bytes);

        Ok(Keypair {
            keypair: NativeKeypair::from_secret_key(&bytes),
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
