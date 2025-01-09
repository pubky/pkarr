#![doc = include_str!("../README.md")]
//! ## Feature flags
#![doc = document_features::document_features!()]
//!

macro_rules! if_dht {
    ($($item:item)*) => {$(
        #[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
        $item
    )*}
}

// Modules
mod error;
mod keys;
mod signed_packet;

// Common exports
pub use crate::error::{Error, Result};
pub use crate::keys::{Keypair, PublicKey};
pub use crate::signed_packet::{system_time, SignedPacket};

/// Default minimum TTL: 5 minutes
pub const DEFAULT_MINIMUM_TTL: u32 = 300;
/// Default maximum TTL: 24 hours
pub const DEFAULT_MAXIMUM_TTL: u32 = 24 * 60 * 60;
/// Default cache size: 1000
pub const DEFAULT_CACHE_SIZE: usize = 1000;

pub const DEFAULT_RELAYS: [&str; 1] = ["https://relay.pkarr.org"];

pub const DEFAULT_RESOLVERS: [&str; 1] = ["resolver.pkarr.org:6881"];

// Rexports
pub use bytes;

#[cfg(not(target_arch = "wasm32"))]
macro_rules! if_async {
    ($($item:item)*) => {$(
        #[cfg(all(not(target_arch = "wasm32"), feature = "async"))]
        $item
    )*}
}

macro_rules! if_relay {
    ($($item:item)*) => {$(
        #[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
        $item
    )*}
}

if_dht! {
    mod cache;
    mod client;

    if_async! {
        mod client_async;
        pub use client_async::PkarrClientAsync;
    }

    pub use client::{PkarrClientBuilder, PkarrClient, Settings};
    pub use cache::{PkarrCache, PkarrCacheKey, InMemoryPkarrCache};

    // Rexports
    pub use mainline;
}

if_relay! {
    mod relay_client;
    pub use relay_client::{PkarrRelayClient, RelaySettings};

    if_async! {
        mod relay_client_async;
        pub use relay_client_async::PkarrRelayClientAsync;
    }
}

#[cfg(target_arch = "wasm32")]
mod relay_client_web;
#[cfg(target_arch = "wasm32")]
pub use relay_client_web::PkarrRelayClient;
