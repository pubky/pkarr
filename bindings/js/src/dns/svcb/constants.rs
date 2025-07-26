//! SVCB/HTTPS service parameter constants and mappings

/// SVCB parameter type: ALPN (Application-Layer Protocol Negotiation)
pub const PARAM_ALPN: u16 = 1;

/// SVCB parameter type: No default ALPN
pub const PARAM_NO_DEFAULT_ALPN: u16 = 2;

/// SVCB parameter type: PORT number
pub const PARAM_PORT: u16 = 3;

/// SVCB parameter type: IPv4 address hints
pub const PARAM_IPV4HINT: u16 = 4;

/// SVCB parameter type: Encrypted Client Hello configuration
pub const PARAM_ECH: u16 = 5;

/// SVCB parameter type: IPv6 address hints
pub const PARAM_IPV6HINT: u16 = 6;

/// List of valid ALPN protocols for SVCB/HTTPS records
pub const VALID_ALPN_PROTOCOLS: &[&str] = &[
    "h2",       // HTTP/2 (RFC 7540)
    "h3",       // HTTP/3 (RFC 9114)
    "http/1.1", // HTTP/1.1 (RFC 9112)
];

/// Convert numeric parameter key to descriptive name
pub fn param_key_to_name(key: u16) -> &'static str {
    match key {
        PARAM_ALPN => "alpn",
        PARAM_NO_DEFAULT_ALPN => "no-default-alpn",
        PARAM_PORT => "port",
        PARAM_IPV4HINT => "ipv4hint",
        PARAM_ECH => "ech",
        PARAM_IPV6HINT => "ipv6hint",
        _ => "unknown",
    }
}

/// Check if a parameter type is known/supported
pub fn is_known_param(key: u16) -> bool {
    matches!(
        key,
        PARAM_ALPN
            | PARAM_NO_DEFAULT_ALPN
            | PARAM_PORT
            | PARAM_IPV4HINT
            | PARAM_ECH
            | PARAM_IPV6HINT
    )
}

/// Validate if a protocol string is a known valid ALPN identifier
pub fn is_valid_alpn_protocol(protocol: &str) -> bool {
    VALID_ALPN_PROTOCOLS.contains(&protocol)
}
