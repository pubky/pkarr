//! WASM bindings for pkarr relay functions

use wasm_bindgen::prelude::*;
use js_sys::{Array, Uint8Array};

use crate::{Keypair, PublicKey, SignedPacket};

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