//! SVCB parameter output - converting Rust data to JavaScript objects

use super::constants::*;
use super::utils::bytes_to_hex;
use simple_dns::rdata::SVCB;
use wasm_bindgen::JsValue;

/// Convert SVCB parameters to JavaScript object with descriptive keys and parsed values
pub fn to_js_object(svcb: &SVCB) -> Result<js_sys::Object, JsValue> {
    let params_obj = js_sys::Object::new();

    for (key, value) in svcb.iter_params() {
        if is_known_param(key) {
            let key_name = param_key_to_name(key);
            let parsed_value = parse_param_value(key, value);
            js_sys::Reflect::set(
                &params_obj,
                &JsValue::from_str(key_name),
                &JsValue::from_str(&parsed_value),
            )?;
        } else {
            // For unknown parameters, use the format "param{number}" with hex values
            let unknown_key = format!("param{}", key);
            let hex_value = bytes_to_hex(value);
            js_sys::Reflect::set(
                &params_obj,
                &JsValue::from_str(&unknown_key),
                &JsValue::from_str(&hex_value),
            )?;
        }
    }

    Ok(params_obj)
}

/// Parse parameter value based on its type
fn parse_param_value(key: u16, value: &[u8]) -> String {
    match key {
        PARAM_ALPN => parse_alpn_param(value),
        PARAM_PORT => parse_port_param(value),
        PARAM_IPV4HINT => {
            parse_ip_hint_param(value, 4, "ipv4hint", |chunk| {
                format!("{}.{}.{}.{}", chunk[0], chunk[1], chunk[2], chunk[3])
            })
        }
        PARAM_IPV6HINT => {
            parse_ip_hint_param(value, 16, "ipv6hint", |chunk| {
                let mut addr_bytes = [0u8; 16];
                addr_bytes.copy_from_slice(chunk);
                std::net::Ipv6Addr::from(addr_bytes).to_string()
            })
        }
        _ => bytes_to_hex(value),
    }
}

/// Parse ALPN parameter to comma-separated protocol list
/// Format: [length1][protocol1_bytes][length2][protocol2_bytes]...
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
