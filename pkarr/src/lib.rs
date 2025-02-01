#![doc = include_str!("../README.md")]
//! ## Feature flags
#![doc = document_features::document_features!()]
//!

macro_rules! cross_debug {
    ($($arg:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        log::debug!($($arg)*);
        #[cfg(not(target_arch = "wasm32"))]
        tracing::debug!($($arg)*);
        #[cfg(test)]
        eprintln!($($arg)*);
    };
}

// Modules
#[cfg(feature = "client")]
pub mod client;
pub mod extra;
mod keys;
mod signed_packet;

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

// Exports
#[cfg(feature = "client")]
pub use client::cache::{Cache, CacheKey, InMemoryCache};
pub use keys::{Keypair, PublicKey};
pub use signed_packet::SignedPacket;

#[cfg(feature = "client")]
pub use client::{Client, ClientBuilder};

// Rexports
pub use simple_dns as dns;

pub mod errors {
    //! Exported errors

    #[cfg(all(feature = "client", not(target_arch = "wasm32")))]
    pub use super::client::native::{
        BuildError, ClientWasShutdown, ConcurrencyError, PublishError,
    };

    #[cfg(all(feature = "client", target_arch = "wasm32"))]
    pub use super::client::web::{AllGetRequestsFailed, EmptyListOfRelays, PublishError};

    pub use super::keys::PublicKeyError;
    pub use super::signed_packet::SignedPacketBuildError;
    pub use super::signed_packet::SignedPacketVerifyError;
}
