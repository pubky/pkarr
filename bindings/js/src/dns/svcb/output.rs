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

/// Parse ALPN parameter to comma-separated protocol list
///
/// Format: `[length1][protocol1_bytes][length2][protocol2_bytes]...`
fn parse_alpn_param(value: &[u8]) -> String {
    let mut protocols = Vec::new();
    let mut offset = 0;

    while offset < value.len() {
        if let Some(&len) = value.get(offset) {
            offset += 1;
            if offset + len as usize <= value.len() {
                // Convert bytes to string
                let protocol_bytes = &value[offset..offset + len as usize];
                let protocol = String::from_utf8_lossy(protocol_bytes);
                protocols.push(protocol.to_string());
                offset += len as usize;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    protocols.join(",")
}

/// Parse PORT parameter to port number string
fn parse_port_param(value: &[u8]) -> String {
    if value.len() == 2 {
        let port = u16::from_be_bytes([value[0], value[1]]);
        port.to_string()
    } else {
        format!("invalid_port_({}_bytes)", value.len())
    }
}

/// Generic function to parse IP address hints (IPv4 or IPv6)
fn parse_ip_hint_param(
    value: &[u8],
    bytes_per_addr: usize,
    hint_type: &str,
    format_addr: impl Fn(&[u8]) -> String,
) -> String {
    if value.len() % bytes_per_addr != 0 {
        return format!("invalid_{}_({}_bytes)", hint_type, value.len());
    }

    value
        .chunks(bytes_per_addr)
        .map(format_addr)
        .collect::<Vec<_>>()
        .join(",")
}
