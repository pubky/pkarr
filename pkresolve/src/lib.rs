//! Pubky routing and direct raw-key TLS with pluggable packet sources.
//!
//! Enable `service-client` to use `ServiceClient` for cache-first resolution
//! and whole-bundle processing. This feature is disabled by default.
//! The caller supplies HTTP; the library performs no network I/O.
//! Enable `service-protocol` for shared Serde request and bundle types without
//! the client cache. `service-client` includes this feature.
//!
//! Call [`derive_resolution`] with a
//! [`PacketProvider`] backed by PKARR, a local cache, or a response bundle.
//! For asynchronous packet sources, use [`derive_resolution_async`] with an
//! [`AsyncPacketProvider`].
//! Missing evidence is never a negative answer.
//! The host authenticates every candidate with `Resolution::service.public_key()`
//! and uses that service name as TLS SNI.
//!
//! Supports same-name HTTPS ServiceMode records and direct raw-key TLS using
//! the destination Pubky key. TLS delegation records are ignored. HTTPS aliases,
//! routing CNAMEs, alternate targets, CA TLS, certificates, ECH, and HTTP/3 are
//! not implemented.

mod error;
mod provider;
mod resolution;
#[cfg(feature = "service-client")]
pub mod service_client;
mod service_name;
#[cfg(feature = "service-protocol")]
pub mod service_protocol;

pub use error::Error;
pub use pkarr::{PublicKey, SignedPacket};
pub use provider::{AsyncPacketProvider, PacketProvider};
pub use resolution::{
    derive_resolution, derive_resolution_async, EndpointCandidate, HttpProtocol, Resolution,
    ResolutionRequest,
};
pub use service_name::ServiceName;

#[cfg(feature = "service-client")]
pub use service_client::ServiceClient;
