//! Shared request and response types for the PKResolve JSON service protocol.
//!
//! Available with `service-protocol`, also enabled by `service-client`.
//! [`crate::ResolutionRequest`] supports Serde directly and validates its service,
//! port, and protocol preferences during deserialization.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::{EndpointCandidate, Resolution};

/// Supported packet bundle schema version.
pub const BUNDLE_SCHEMA: u32 = 1;

/// JSON response containing the packets and a required resolution summary.
/// Deserialization does not validate the schema or packet contents.
#[derive(Debug, Serialize, Deserialize)]
pub struct ResolveBundle {
    /// Protocol version, checked against [`BUNDLE_SCHEMA`] by the consumer.
    pub schema: u32,
    /// Standard base64 of native `SignedPacket::serialize()` bytes, including
    /// the original `last_seen` timestamp. Each owner occurs once; order is
    /// irrelevant. Decoding these strings does not verify packet signatures.
    pub packets: Vec<String>,
    /// Relay-computed result included for convenience. Clients should verify
    /// it by deriving their own resolution for the original request, using
    /// the bundled packets as the packet source, before trusting routes.
    pub resolution: BundleResolution,
}

/// Serializable snapshot of a relay's resolution result.
/// The local monotonic deadline in [`Resolution`] cannot be sent across hosts.
/// This summary is informational; it does not authenticate routes or TLS keys.
#[derive(Debug, Serialize, Deserialize)]
pub struct BundleResolution {
    /// Original scoped service name, used for SNI and HTTP authority.
    pub service: String,
    /// Original URL port, independent of candidate connection ports.
    pub url_port: u16,
    /// Endpoint candidates in the resolver's priority order.
    pub candidates: Vec<EndpointCandidate>,
    /// Remaining lifetime when the summary was created, rounded down to
    /// milliseconds. Does not account for transport delay; do not restart
    /// this lifetime on receipt or use it instead of checking packet TTLs.
    pub valid_for_ms: u64,
}

impl From<Resolution> for BundleResolution {
    fn from(resolution: Resolution) -> Self {
        Self {
            service: resolution.service.into(),
            url_port: resolution.url_port,
            candidates: resolution.candidates,
            valid_for_ms: resolution
                .expires_at
                .saturating_duration_since(Instant::now())
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
        }
    }
}
