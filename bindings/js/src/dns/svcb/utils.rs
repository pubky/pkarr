//! Shared utilities for SVCB parameter handling

/// Convert bytes to hex string for unknown parameters
pub fn bytes_to_hex(value: &[u8]) -> String {
    let hex_chars: String = value.iter().map(|b| format!("{:02x}", b)).collect();
    format!("0x{}", hex_chars)
}
