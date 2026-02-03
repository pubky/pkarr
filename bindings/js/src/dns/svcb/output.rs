//! SVCB parameter output - converting Rust data to JavaScript objects

use crate::error::ClientError;

use super::constants::*;
use super::utils::bytes_to_hex;
use simple_dns::rdata::{SVCParam, SVCB};
use wasm_bindgen::JsValue;

/// Convert SVCB parameters to JavaScript object with descriptive keys and parsed values
pub fn to_js_object(svcb: &SVCB) -> Result<js_sys::Object, JsValue> {
    let params_obj = js_sys::Object::new();

    for param in svcb.iter_params() {
        let key = param.key_code();
        if is_known_param(key) {
            let key_name = param_key_to_name(key);
            let parsed_value = parse_param_value_from_param(param);
            js_sys::Reflect::set(
                &params_obj,
                &JsValue::from_str(key_name),
                &JsValue::from_str(&parsed_value),
            )?;
        } else {
            // For unknown parameters, use the format "param{number}" with hex values
            let unknown_key = format!("param{}", key);
            let SVCParam::Unknown(_, data) = param else {
                return Err(ClientError::ParseError {
                    input_type: "svcb".to_string(),
                    message: "Expected unknown SVCParam for key {key}, got known one".into(),
                }
                .into());
            };
            let hex_value = bytes_to_hex(data.as_ref());
            js_sys::Reflect::set(
                &params_obj,
                &JsValue::from_str(&unknown_key),
                &JsValue::from_str(&hex_value),
            )?;
        }
    }

    Ok(params_obj)
}

/// Parse parameter value from SVCParam
fn parse_param_value_from_param(param: &SVCParam) -> String {
    match param {
        SVCParam::Alpn(alpns) => alpns
            .iter()
            .map(|cs| cs.to_string())
            .collect::<Vec<_>>()
            .join(","),
        SVCParam::Port(port) => port.to_string(),
        SVCParam::Ipv4Hint(ips) => ips
            .iter()
            .map(|&ip| std::net::Ipv4Addr::from(ip).to_string())
            .collect::<Vec<_>>()
            .join(","),
        SVCParam::Ipv6Hint(ips) => ips
            .iter()
            .map(|&ip| std::net::Ipv6Addr::from(ip).to_string())
            .collect::<Vec<_>>()
            .join(","),
        SVCParam::NoDefaultAlpn => String::new(),
        SVCParam::Ech(data) => bytes_to_hex(data.as_ref()),
        SVCParam::Mandatory(keys) => keys
            .iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(","),
        SVCParam::Unknown(_, data) => bytes_to_hex(data.as_ref()),
        SVCParam::InvalidKey => String::from("invalid_key"),
    }
}
