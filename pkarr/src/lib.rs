#![doc = include_str!("../README.md")]
//! ## Feature flags
#![doc = document_features::document_features!()]
//!

#![cfg_attr(not(test), warn(unused_crate_dependencies))]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

// Modules
#[cfg(client)]
mod client;
#[cfg(dht)]
pub mod dht;
#[cfg(client)]
pub mod extra;
#[cfg(relays)]
pub mod relay_client;
pub mod types;

/// Default minimum TTL: 5 minutes.
pub const DEFAULT_MINIMUM_TTL: u32 = 300;
/// Default maximum TTL: 24 hours.
pub const DEFAULT_MAXIMUM_TTL: u32 = 24 * 60 * 60;
/// Default [Relays](https://github.com/pubky/pkarr/blob/main/design/relays.md).
pub const DEFAULT_RELAYS: [&str; 2] = ["https://pkarr.pubky.app", "https://pkarr.pubky.org"];
/// Relay response header carrying the mutable item sequence number when the
/// DHT has a newer mutable item that is not a valid signed packet for a key.
pub const PKARR_INVALID_SIGNED_PACKET_SEQ: &str = "Pkarr-Invalid-Signed-Packet-Seq";
/// Relay response header carrying the number of DHT nodes that acknowledged
/// storing a published packet.
pub const PKARR_DHT_STORED_NODES: &str = "Pkarr-Dht-Stored-Nodes";
#[cfg(feature = "__client")]
/// Default cache size: 1000
pub const DEFAULT_CACHE_SIZE: usize = 1000;

// Exports
#[cfg(client)]
pub use client::cache::{Cache, CacheKey, InMemoryCache};
#[cfg(client)]
pub use client::{builder::DEFAULT_REQUEST_TIMEOUT, Client, ClientBuilder};
pub use types::*;

// Rexports
#[cfg(dht)]
pub use mainline;
#[cfg(feature = "signed_packet")]
pub use ntimestamp::Timestamp;
#[cfg(feature = "signed_packet")]
pub use simple_dns as dns;

pub mod errors {
    //! Exported errors

    pub use super::types::errors::*;

    #[cfg(relays)]
    pub use super::client::InvalidRelayUrl;
    #[cfg(client)]
    pub use super::client::{BuildError, PublishError, ResolveError};
}
