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
                let mut target = js_sys::Reflect::get(rdata, &JsValue::from_str(PROP_TARGET))?
                    .as_string()
                    .unwrap_or_default();

                // For priority 0, only show priority and target
                if priority == 0 {
                    return Ok(format!("{} {}", priority, target));
                }

                // ServiceMode (priority > 0): TargetName must be set; "." means "this host" (RFC 9460 §2.5.2)
                if target.is_empty() {
                    target = ".".to_string();
                }

                // For priority > 0, include parameters
                let params_result = js_sys::Reflect::get(rdata, &JsValue::from_str(PROP_PARAMS));
                let mut params_str = String::new();

                if let Ok(params_obj) = params_result {
                    if !params_obj.is_undefined() {
                        if let Some(obj_ref) = params_obj.dyn_ref::<js_sys::Object>() {
                            let keys = js_sys::Object::keys(obj_ref);
                            let mut param_parts = Vec::new();

                            for i in 0..keys.length() {
                                if let Some(key_str) = keys.get(i).as_string() {
                                    let display_name = &key_str;
                                    let value_result = js_sys::Reflect::get(obj_ref, &keys.get(i));
                                    if let Ok(value_js) = value_result {
                                        if let Some(value_str) = value_js.as_string() {
                                            param_parts
                                                .push(format!("{}={}", display_name, value_str));
                                        } else {
                                            param_parts.push(format!("{}=<invalid>", display_name));
                                        }
                                    } else {
                                        param_parts.push(format!("{}=<error>", display_name));
                                    }
                                }
                            }

                            if !param_parts.is_empty() {
                                params_str = format!(" {}", param_parts.join(" "));
                            }
                        }
                    }
                }

                Ok(format!("{} {}{}", priority, target, params_str))
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
