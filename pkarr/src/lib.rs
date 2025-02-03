#![doc = include_str!("../README.md")]
//! ## Feature flags
#![doc = document_features::document_features!()]
//!

// TODO: deny missing_docs
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

// Modules
#[cfg(feature = "__client")]
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
#[cfg(feature = "__client")]
pub use client::cache::{Cache, CacheKey, InMemoryCache};
#[cfg(feature = "keys")]
pub use keys::{Keypair, PublicKey};
#[cfg(feature = "signed_packet")]
pub use signed_packet::SignedPacket;

#[cfg(feature = "__client")]
pub use client::{Client, ClientBuilder};

// Rexports
pub use simple_dns as dns;

pub mod errors {
    //! Exported errors

    #[cfg(feature = "__client")]
    pub use super::client::native::{
        BuildError, ClientWasShutdown, ConcurrencyError, PublishError,
    };

    #[cfg(feature = "keys")]
    pub use super::keys::PublicKeyError;
    #[cfg(feature = "signed_packet")]
    pub use super::signed_packet::{SignedPacketBuildError, SignedPacketVerifyError};
}
