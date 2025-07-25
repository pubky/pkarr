use super::constants::*;
use super::*;

/// Utility functions
#[wasm_bindgen]
pub struct Utils;

#[wasm_bindgen]
impl Utils {
    /// Format a DNS record value for display
    #[wasm_bindgen(js_name = "formatRecordValue")]
    pub fn format_record_value(rdata: &JsValue) -> Result<String, JsValue> {
        let record_type = js_sys::Reflect::get(rdata, &JsValue::from_str(PROP_TYPE))?
            .as_string()
            .unwrap_or_default();

        match record_type.as_str() {
            TYPE_A | TYPE_AAAA => {
                let address = js_sys::Reflect::get(rdata, &JsValue::from_str(PROP_ADDRESS))?
                    .as_string()
                    .unwrap_or_default();
                Ok(address)
            }
            TYPE_CNAME => {
                let target = js_sys::Reflect::get(rdata, &JsValue::from_str(PROP_TARGET))?
                    .as_string()
                    .unwrap_or_default();
                Ok(target)
            }
            TYPE_TXT => {
                let value = js_sys::Reflect::get(rdata, &JsValue::from_str(PROP_VALUE))?
                    .as_string()
                    .unwrap_or_default();
                Ok(value)
            }
            TYPE_HTTPS | TYPE_SVCB => {
                let priority = js_sys::Reflect::get(rdata, &JsValue::from_str(PROP_PRIORITY))?
                    .as_f64()
                    .unwrap_or(0.0) as u16;
                let target = js_sys::Reflect::get(rdata, &JsValue::from_str(PROP_TARGET))?
                    .as_string()
                    .unwrap_or_default();
                Ok(format!("{priority} {target}"))
            }
            TYPE_NS => {
                let nsdname = js_sys::Reflect::get(rdata, &JsValue::from_str(PROP_NSDNAME))?
                    .as_string()
                    .unwrap_or_default();
                Ok(nsdname)
            }
            _ => {
                // For unknown types, return a JSON-like representation
                Ok(format!("{rdata:?}"))
            }
        }
    }

    /// Validate a public key string
    #[wasm_bindgen(js_name = "validatePublicKey")]
    pub fn validate_public_key(public_key_str: &str) -> bool {
        if public_key_str.trim().is_empty() {
            return false;
        }
        pkarr::PublicKey::try_from(public_key_str).is_ok()
    }

    /// Get default relay URLs
    #[wasm_bindgen(js_name = "defaultRelays")]
    pub fn default_relays() -> Array {
        let relays = Array::new();
        for relay in pkarr::DEFAULT_RELAYS.iter() {
            relays.push(&JsValue::from_str(relay));
        }
        relays
    }
}
