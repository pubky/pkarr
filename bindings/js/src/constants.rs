//! Constants used throughout the WASM module

// === Size and Time Constants ===

/// Pkarr specification: packets must be <= 1000 bytes
pub const MAX_PACKET_SIZE: usize = 1000;

/// Size of secret key in bytes (Ed25519)
pub const SECRET_KEY_SIZE: usize = 32;

/// Conversion factor from milliseconds to microseconds
pub const MS_TO_MICROSECONDS: u64 = 1000;

// === Client Timeout Constants ===

/// Default timeout for requests in milliseconds (30 seconds)
pub const DEFAULT_TIMEOUT_MS: u32 = 30000;

/// Minimum allowed timeout in milliseconds (1 second)
pub const MIN_TIMEOUT_MS: u32 = 1000;

/// Maximum allowed timeout in milliseconds (5 minutes)
pub const MAX_TIMEOUT_MS: u32 = 300000;

// === DNS Record Property Names ===
// These are used when creating JavaScript objects for DNS records

/// Property name for the DNS record name field
pub const PROP_NAME: &str = "name";

/// Property name for the DNS record TTL field
pub const PROP_TTL: &str = "ttl";

/// Property name for the DNS record data object
pub const PROP_RDATA: &str = "rdata";

/// Property name for the DNS record type field
pub const PROP_TYPE: &str = "type";

/// Property name for IP address in A/AAAA records
pub const PROP_ADDRESS: &str = "address";

/// Property name for target domain in CNAME/HTTPS/SVCB records
pub const PROP_TARGET: &str = "target";

/// Property name for text content in TXT records
pub const PROP_VALUE: &str = "value";

/// Property name for priority in HTTPS/SVCB records
pub const PROP_PRIORITY: &str = "priority";

/// Property name for name server domain in NS records
pub const PROP_NSDNAME: &str = "nsdname";

/// Property name for service parameters in HTTPS/SVCB records
pub const PROP_PARAMS: &str = "params";

// === SVCB/HTTPS Service Parameter Names ===
// These allow using descriptive names instead of just numeric keys

/// SVCB parameter name for ALPN (Application-Layer Protocol Negotiation)
pub const SVCB_PARAM_ALPN: &str = "alpn";

/// SVCB parameter name for no default ALPN
pub const SVCB_PARAM_NO_DEFAULT_ALPN: &str = "no-default-alpn";

/// SVCB parameter name for port number
pub const SVCB_PARAM_PORT: &str = "port";

/// SVCB parameter name for IPv4 address hints
pub const SVCB_PARAM_IPV4HINT: &str = "ipv4hint";

/// SVCB parameter name for Encrypted Client Hello configuration
pub const SVCB_PARAM_ECH: &str = "ech";

/// SVCB parameter name for IPv6 address hints
pub const SVCB_PARAM_IPV6HINT: &str = "ipv6hint";

// === DNS Record Type Values ===
// These are the string values used for the "type" field in DNS record objects

/// DNS record type value for A records (IPv4 address)
pub const TYPE_A: &str = "A";

/// DNS record type value for AAAA records (IPv6 address)
pub const TYPE_AAAA: &str = "AAAA";

/// DNS record type value for CNAME records (canonical name)
pub const TYPE_CNAME: &str = "CNAME";

/// DNS record type value for TXT records (text data)
pub const TYPE_TXT: &str = "TXT";

/// DNS record type value for HTTPS records (HTTPS service binding)
pub const TYPE_HTTPS: &str = "HTTPS";

/// DNS record type value for SVCB records (service binding)
pub const TYPE_SVCB: &str = "SVCB";

/// DNS record type value for NS records (name server)
pub const TYPE_NS: &str = "NS";
