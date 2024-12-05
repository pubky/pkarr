#![doc = include_str!("../README.md")]
//! ## Feature flags
#![doc = document_features::document_features!()]
//!

// Modules
mod client;
pub mod extra;
mod keys;
mod signed_packet;

// Exports
pub use client::cache;
pub use keys::{Keypair, PublicKey};
pub use signed_packet::SignedPacket;

/// Default minimum TTL: 5 minutes
pub const DEFAULT_MINIMUM_TTL: u32 = 300;
/// Default maximum TTL: 24 hours
pub const DEFAULT_MAXIMUM_TTL: u32 = 24 * 60 * 60;
/// Default cache size: 1000
pub const DEFAULT_CACHE_SIZE: usize = 1000;
/// Default [relay](https://pkarr.org/relays)s
pub const DEFAULT_RELAYS: [&str; 2] = ["https://relay.pkarr.org", "https://pkarr.pubky.org"];
/// Default [resolver](https://pkarr.org/resolvers)s
pub const DEFAULT_RESOLVERS: [&str; 2] = ["resolver.pkarr.org:6881", "pkarr.pubky.org:6881"];

#[cfg(all(not(target_arch = "wasm32"), any(feature = "dht", feature = "relay")))]
pub use client::native::Info;
#[cfg(any(target_arch = "wasm32", feature = "dht"))]
pub use client::{Client, Settings};

pub mod errors {
    //! Exported errors
    #[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
    pub use super::client::native::{ClientWasShutdown, PublishError};

    // TODO: wasm
    #[cfg(any(target_arch = "wasm32", feature = "relay"))]
    // pub use super::client::relay::{EmptyListOfRelays, PublishToRelayError};
    //
    pub use super::keys::PublicKeyError;
    pub use super::signed_packet::SignedPacketError;
}

// Rexports
pub use bytes;
#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
pub use mainline;
pub use simple_dns as dns;
