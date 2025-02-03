#![doc = include_str!("../README.md")]
//! ## Feature flags
#![doc = document_features::document_features!()]
//!

// TODO: deny missing_docs
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

// Modules
#[cfg(all(
    feature = "__client",
    not(all(target_family = "wasm", not(feature = "relays")))
))]
pub mod client;
pub mod extra;
#[cfg(feature = "keys")]
mod keys;
#[cfg(feature = "signed_packet")]
mod signed_packet;

/// Default minimum TTL: 5 minutes
pub const DEFAULT_MINIMUM_TTL: u32 = 300;
/// Default maximum TTL: 24 hours
pub const DEFAULT_MAXIMUM_TTL: u32 = 24 * 60 * 60;
/// Default cache size: 1000
pub const DEFAULT_CACHE_SIZE: usize = 1000;
/// Default [relay](https://pkarr.org/relays)s
pub const DEFAULT_RELAYS: [&str; 2] = ["https://relay.pkarr.org", "https://pkarr.pubky.org"];
#[cfg(all(feature = "dht", not(target_family = "wasm")))]
/// Default [resolver](https://pkarr.org/resolvers)s
pub const DEFAULT_RESOLVERS: [&str; 2] = ["resolver.pkarr.org:6881", "pkarr.pubky.org:6881"];

// Exports
#[cfg(all(
    feature = "__client",
    not(all(target_family = "wasm", not(feature = "relays")))
))]
pub use client::cache::{Cache, CacheKey, InMemoryCache};
#[cfg(feature = "keys")]
pub use keys::{Keypair, PublicKey};
#[cfg(feature = "signed_packet")]
pub use signed_packet::SignedPacket;

#[cfg(all(
    feature = "__client",
    not(all(target_family = "wasm", not(feature = "relays")))
))]
pub use client::{Client, ClientBuilder};

// Rexports
#[cfg(feature = "signed_packet")]
pub use simple_dns as dns;

pub mod errors {
    //! Exported errors

    #[cfg(all(
        feature = "__client",
        not(all(target_family = "wasm", not(feature = "relays")))
    ))]
    pub use super::client::{BuildError, ClientWasShutdown, ConcurrencyError, PublishError};

    #[cfg(feature = "keys")]
    pub use super::keys::PublicKeyError;

    #[cfg(feature = "signed_packet")]
    pub use super::signed_packet::{SignedPacketBuildError, SignedPacketVerifyError};
}
