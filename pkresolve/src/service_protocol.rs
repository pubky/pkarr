//! Shared request and response types for the PKResolve JSON service protocol.
//!
//! Available with `service-protocol`, also enabled by `service-client`.
//! [`crate::ResolutionRequest`] supports Serde directly and validates its service,
//! port, and protocol preferences during deserialization.

use serde::{Deserialize, Serialize};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
    /// Z32-encoded raw TLS key, including HTTPS AliasMode delegation.
    pub tls_public_key: String,
    /// Original URL port, independent of candidate connection ports.
    pub url_port: u16,
    /// Endpoint candidates in the resolver's priority order.
    pub candidates: Vec<EndpointCandidate>,
    /// Absolute expiry in Unix milliseconds, rounded down. Derived using the
    /// relay's wall clock; interpreting it depends on clock agreement.
    pub valid_until_unix_ms: u64,
}

impl From<Resolution> for BundleResolution {
    fn from(resolution: Resolution) -> Self {
        // Sample wall time first so time spent sampling does not extend expiry.
        let wall_now = SystemTime::now();
        let monotonic_now = Instant::now();
        Self {
            service: resolution.service.into(),
            tls_public_key: resolution.tls_public_key.to_z32(),
            url_port: resolution.url_port,
            candidates: resolution.candidates,
            valid_until_unix_ms: unix_expiry_ms(resolution.expires_at, monotonic_now, wall_now),
        }
    }
}

/// Convert a monotonic deadline using paired clock samples.
/// Clamp expiry to the unsigned millisecond range; wall times before the Unix
/// epoch return zero. The caller samples wall time before monotonic time.
fn unix_expiry_ms(expires_at: Instant, monotonic_now: Instant, wall_now: SystemTime) -> u64 {
    let Ok(unix_now) = wall_now.duration_since(UNIX_EPOCH) else {
        return 0;
    };
    let unix_expiry = if let Some(remaining) = expires_at.checked_duration_since(monotonic_now) {
        unix_now.saturating_add(remaining)
    } else {
        unix_now.saturating_sub(monotonic_now.duration_since(expires_at))
    };
    unix_expiry.as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn unix_expiry_preserves_future_present_and_past_deadlines() {
        let monotonic_now = Instant::now();
        let wall_now = UNIX_EPOCH + Duration::from_millis(10_000);
        for (expires_at, expected) in [
            (monotonic_now + Duration::from_micros(1500), 10_001),
            (monotonic_now, 10_000),
            (monotonic_now - Duration::from_micros(1500), 9_998),
            (monotonic_now - Duration::from_secs(20), 0),
        ] {
            assert_eq!(
                unix_expiry_ms(expires_at, monotonic_now, wall_now),
                expected
            );
        }
    }

    #[test]
    fn unix_expiry_saturates_overflow_and_rejects_pre_epoch_clocks() {
        let monotonic_now = Instant::now();
        let wall_now = UNIX_EPOCH + Duration::from_millis(u64::MAX);
        assert_eq!(
            unix_expiry_ms(
                monotonic_now + Duration::from_secs(1),
                monotonic_now,
                wall_now
            ),
            u64::MAX
        );
        assert_eq!(
            unix_expiry_ms(
                monotonic_now + Duration::from_secs(10),
                monotonic_now,
                UNIX_EPOCH - Duration::from_secs(1)
            ),
            0
        );
    }
}
