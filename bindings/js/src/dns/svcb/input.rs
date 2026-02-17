//! SVCB parameter input - converting JavaScript values to Rust SVCB data

use super::constants::*;
use crate::constants::*;
use simple_dns::rdata::SVCB;
use wasm_bindgen::{JsCast, JsValue};

/// Apply JavaScript parameters to SVCB record - supports both numeric and descriptive keys
pub fn apply_svcb_params(svcb: &mut SVCB, params: &js_sys::Object) -> Result<(), JsValue> {
    let keys = js_sys::Object::keys(params);

    for i in 0..keys.length() {
        let key_js = keys.get(i);
        let key_str = key_js.as_string().ok_or_else(|| {
            JsValue::from_str(&format!("Parameter key at index {i} must be a string"))
        })?;

        // Hybrid approach: try numeric first, then descriptive names
        let key_num = if let Ok(num) = key_str.parse::<u16>() {
            num
        } else {
            match key_str.as_str() {
                SVCB_PARAM_ALPN => PARAM_ALPN,
                SVCB_PARAM_NO_DEFAULT_ALPN => PARAM_NO_DEFAULT_ALPN,
                SVCB_PARAM_PORT => PARAM_PORT,
                SVCB_PARAM_IPV4HINT => PARAM_IPV4HINT,
                // TODO: Add ECH parameter
                SVCB_PARAM_ECH => PARAM_ECH,
                SVCB_PARAM_IPV6HINT => PARAM_IPV6HINT,
                _ => {
                    return Err(JsValue::from_str(&format!(
                        "Unknown SVCB parameter: {key_str}"
                    )))
                }
            }
        };

        let value_js = js_sys::Reflect::get(params, &key_js)?;

        match key_num {
            PARAM_ALPN => set_alpn_from_js(svcb, &value_js)?,
            PARAM_PORT => set_port_from_js(svcb, &value_js)?,
            PARAM_IPV4HINT => set_ipv4hint_from_js(svcb, &value_js)?,
            PARAM_IPV6HINT => set_ipv6hint_from_js(svcb, &value_js)?,
            _ => set_unknown_param_from_js(key_num, &value_js)?,
        }
    }

    Ok(())
}

/// Set ALPN parameter from JavaScript value (string or array)
fn set_alpn_from_js(svcb: &mut SVCB, value_js: &JsValue) -> Result<(), JsValue> {
    let protocol_strings = if let Some(string_val) = value_js.as_string() {
        // Comma-separated string like "h2,h3"
        string_val
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    } else if let Some(array) = value_js.dyn_ref::<js_sys::Array>() {
        // Array of protocol strings
        let mut protocols = Vec::new();
        for j in 0..array.length() {
            if let Some(protocol) = array.get(j).as_string() {
                protocols.push(protocol);
            }
        }
        protocols
    } else {
        return Err(JsValue::from_str(
            "ALPN parameter must be a string or array",
        ));
    };

    // Convert protocol strings to CharacterString and set on SVCB
    set_alpn_protocols(svcb, protocol_strings)
}

/// Helper function to convert protocol strings and set them on SVCB
fn set_alpn_protocols(svcb: &mut SVCB, protocol_strings: Vec<String>) -> Result<(), JsValue> {
    // First validate all protocols are known valid ALPN identifiers
    for protocol in &protocol_strings {
        if !is_valid_alpn_protocol(protocol) {
            return Err(JsValue::from_str(&format!(
                "Invalid ALPN protocol: {protocol}"
            )));
        }
    }

    // Then convert to owned CharacterStrings using TryFrom<String>
    let protocols: Result<Vec<simple_dns::CharacterString>, _> =
        protocol_strings.into_iter().map(|s| s.try_into()).collect();

    match protocols {
        Ok(alpn_list) => {
            svcb.set_alpn(&alpn_list);
            Ok(())
        }
        Err(e) => Err(JsValue::from_str(&format!(
            "Invalid ALPN protocol format: {e}"
        ))),
    }
}

/// Set port parameter from JavaScript value (number or 2-byte Uint8Array)
fn set_port_from_js(svcb: &mut SVCB, value_js: &JsValue) -> Result<(), JsValue> {
    if let Some(port_num) = value_js.as_f64() {
        // Validate port range
        if !((0.0..=65535.0).contains(&port_num)) || port_num.fract() != 0.0 {
            return Err(JsValue::from_str(
                "Port must be an integer between 0 and 65535",
            ));
        }
        svcb.set_port(port_num as u16);
    } else if let Some(uint8_array) = value_js.dyn_ref::<js_sys::Uint8Array>() {
        if uint8_array.length() == 2 {
            let mut bytes = [0u8; 2];
            uint8_array.copy_to(&mut bytes);
            let port = u16::from_be_bytes(bytes);
            svcb.set_port(port);
        } else {
            return Err(JsValue::from_str(
                "Port parameter as bytes must be exactly 2 bytes",
            ));
        }
    } else {
        return Err(JsValue::from_str(
            "Port parameter must be a number or 2-byte Uint8Array",
        ));
    }
    Ok(())
}

/// Set IPv4 address hints from JavaScript value (string, array, or Uint8Array)
fn set_ipv4hint_from_js(svcb: &mut SVCB, value_js: &JsValue) -> Result<(), JsValue> {
    let addrs = parse_ip_hint_from_js(
        value_js,
        4, // 4 bytes per IPv4 address
        "IPv4hint",
        |addr_str| addr_str.parse::<std::net::Ipv4Addr>().map(u32::from),
        |bytes| {
            let addr = std::net::Ipv4Addr::from([bytes[0], bytes[1], bytes[2], bytes[3]]);
            u32::from(addr)
        },
    )?;

    svcb.set_ipv4hint(&addrs);
    Ok(())
}

/// Set IPv6 address hints from JavaScript value (string, array, or Uint8Array)
fn set_ipv6hint_from_js(svcb: &mut SVCB, value_js: &JsValue) -> Result<(), JsValue> {
    let addrs = parse_ip_hint_from_js(
        value_js,
        16, // 16 bytes per IPv6 address
        "IPv6hint",
        |addr_str| addr_str.parse::<std::net::Ipv6Addr>().map(u128::from),
        |bytes| {
            let mut addr_bytes = [0u8; 16];
            addr_bytes.copy_from_slice(bytes);
            let addr = std::net::Ipv6Addr::from(addr_bytes);
            u128::from(addr)
        },
    )?;

    svcb.set_ipv6hint(&addrs);
    Ok(())
}

/// Generic helper function for parsing IP address hints (IPv4 or IPv6)
fn parse_ip_hint_from_js<T, ParseFn, BytesFn>(
    value_js: &JsValue,
    bytes_per_addr: usize,
    hint_type: &str,
    parse_string: ParseFn,
    parse_bytes: BytesFn,
) -> Result<Vec<T>, JsValue>
where
    T: Clone,
    ParseFn: Fn(&str) -> Result<T, std::net::AddrParseError>,
    BytesFn: Fn(&[u8]) -> T,
{
    let addrs = if let Some(string_val) = value_js.as_string() {
        // Single IP address as string
        match parse_string(&string_val) {
            Ok(addr) => vec![addr],
            Err(_) => {
                return Err(JsValue::from_str(&format!(
                    "Invalid {hint_type} address format"
                )))
            }
        }
    } else if let Some(array) = value_js.dyn_ref::<js_sys::Array>() {
        // Array of IP addresses
        let mut addrs = Vec::new();
        for j in 0..array.length() {
            if let Some(addr_str) = array.get(j).as_string() {
                match parse_string(&addr_str) {
                    Ok(addr) => addrs.push(addr),
                    Err(_) => {
                        return Err(JsValue::from_str(&format!(
                            "Invalid {hint_type} address: {addr_str}"
                        )));
                    }
                }
            }
        }
        addrs
    } else if let Some(uint8_array) = value_js.dyn_ref::<js_sys::Uint8Array>() {
        // Raw bytes
        let bytes_len = uint8_array.length() as usize;
        if bytes_len % bytes_per_addr != 0 {
            return Err(JsValue::from_str(&format!(
                "{hint_type} raw bytes must be multiple of {bytes_per_addr} bytes"
            )));
        }
        let mut bytes = vec![0u8; bytes_len];
        uint8_array.copy_to(&mut bytes);

        bytes.chunks(bytes_per_addr).map(parse_bytes).collect()
    } else {
        return Err(JsValue::from_str(&format!(
            "{hint_type} parameter must be a string, array, or Uint8Array"
        )));
    };

    if addrs.is_empty() {
        return Err(JsValue::from_str(&format!(
            "{hint_type} parameter cannot be empty"
        )));
    }

    Ok(addrs)
}

/// Handle unknown/unsupported parameters
fn set_unknown_param_from_js(key_num: u16, value_js: &JsValue) -> Result<(), JsValue> {
    if value_js.dyn_ref::<js_sys::Uint8Array>().is_some() {
        // For now, we can't set generic parameters without a generic method
        Err(JsValue::from_str(&format!(
            "Parameter {key_num} is not yet supported - needs generic parameter setting in SVCB"
        )))
    } else {
        Err(JsValue::from_str(&format!(
            "Parameter {key_num} must be provided as Uint8Array"
        )))
    }
}
