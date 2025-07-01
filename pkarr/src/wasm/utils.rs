use super::*;

/// Utility functions
#[wasm_bindgen]
pub struct Utils;

#[wasm_bindgen]
impl Utils {
    /// Parse a signed packet from bytes and validate it
    #[wasm_bindgen(js_name = "parseSignedPacket")]
    pub fn parse_signed_packet(bytes: &[u8]) -> Result<super::SignedPacket, JsValue> {
        let signed_packet = NativeSignedPacket::deserialize(bytes)
            .map_err(|e| JsValue::from_str(&format!("Invalid signed packet: {}", e)))?;
        Ok(super::SignedPacket::from(signed_packet))
    }

    /// Format a DNS record value for display
    #[wasm_bindgen(js_name = "formatRecordValue")]
    pub fn format_record_value(rdata: &JsValue) -> Result<String, JsValue> {
        let record_type = js_sys::Reflect::get(rdata, &JsValue::from_str("type"))?
            .as_string()
            .unwrap_or_default();

        match record_type.as_str() {
            "A" | "AAAA" => {
                let address = js_sys::Reflect::get(rdata, &JsValue::from_str("address"))?
                    .as_string()
                    .unwrap_or_default();
                Ok(address)
            }
            "CNAME" => {
                let target = js_sys::Reflect::get(rdata, &JsValue::from_str("target"))?
                    .as_string()
                    .unwrap_or_default();
                Ok(target)
            }
            "TXT" => {
                let value = js_sys::Reflect::get(rdata, &JsValue::from_str("value"))?
                    .as_string()
                    .unwrap_or_default();
                Ok(value)
            }
            "HTTPS" | "SVCB" => {
                let priority = js_sys::Reflect::get(rdata, &JsValue::from_str("priority"))?
                    .as_f64()
                    .unwrap_or(0.0) as u16;
                let target = js_sys::Reflect::get(rdata, &JsValue::from_str("target"))?
                    .as_string()
                    .unwrap_or_default();
                Ok(format!("{} {}", priority, target))
            }
            "NS" => {
                let nsdname = js_sys::Reflect::get(rdata, &JsValue::from_str("nsdname"))?
                    .as_string()
                    .unwrap_or_default();
                Ok(nsdname)
            }
            _ => {
                // For unknown types, return a JSON-like representation
                Ok(format!("{:?}", rdata))
            }
        }
    }

    /// Validate a public key string
    #[wasm_bindgen(js_name = "validatePublicKey")]
    pub fn validate_public_key(public_key_str: &str) -> bool {
        PublicKey::try_from(public_key_str).is_ok()
    }

    /// Get default relay URLs
    #[wasm_bindgen(js_name = "defaultRelays")]
    pub fn default_relays() -> Array {
        let relays = Array::new();
        for relay in crate::DEFAULT_RELAYS.iter() {
            relays.push(&JsValue::from_str(relay));
        }
        relays
    }
}
