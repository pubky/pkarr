#![doc = include_str!("../README.md")]
//! ## Feature flags
#![doc = document_features::document_features!()]
//!

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

// Modules
#[cfg(client)]
mod client;
#[cfg(client)]
pub mod extra;
#[cfg(feature = "keys")]
mod keys;
#[cfg(feature = "signed_packet")]
mod signed_packet;

/// Default minimum TTL: 5 minutes.
pub const DEFAULT_MINIMUM_TTL: u32 = 300;
/// Default maximum TTL: 24 hours.
pub const DEFAULT_MAXIMUM_TTL: u32 = 24 * 60 * 60;
/// Default [Relays](https://pkarr.org/relays).
pub const DEFAULT_RELAYS: [&str; 2] = ["https://pkarr.pubky.app", "https://pkarr.pubky.org"];
#[cfg(feature = "__client")]
/// Default cache size: 1000
pub const DEFAULT_CACHE_SIZE: usize = 1000;

// Exports
#[cfg(all(client, not(wasm_browser)))]
pub use client::blocking::ClientBlocking;
#[cfg(client)]
pub use client::cache::{Cache, CacheKey, InMemoryCache};
#[cfg(feature = "relays")]
pub use client::relays::RelaysClient;
#[cfg(client)]
pub use client::{builder::ClientBuilder, Client};
#[cfg(feature = "keys")]
pub use keys::{Keypair, PublicKey};
#[cfg(feature = "signed_packet")]
pub use signed_packet::{SignedPacket, SignedPacketBuilder};

// Rexports
#[cfg(dht)]
pub use mainline;
#[cfg(feature = "signed_packet")]
pub use ntimestamp::Timestamp;
#[cfg(feature = "signed_packet")]
pub use simple_dns as dns;

pub mod errors {
    //! Exported errors

    #[cfg(feature = "keys")]
    pub use super::keys::PublicKeyError;

    #[cfg(feature = "signed_packet")]
    pub use super::signed_packet::{SignedPacketBuildError, SignedPacketVerifyError};

    #[cfg(client)]
    pub use super::client::{BuildError, ConcurrencyError, PublishError, QueryError};
}

// --- Compile-time guards for WASM targets -----------------------------------
// Ensure correct feature combinations at compile time rather than silently
// producing an unusable build. These protect users who use the main pkarr crate
// directly in WASM environments (not through language bindings)

// If we are compiling for a WASM target and *did not* enable the `relays`
// feature, emit a helpful error.
#[cfg(all(target_family = "wasm", not(feature = "relays")))]
compile_error!(
    "When targeting WebAssembly, the `relays` feature must be enabled (e.g. `--features relays`)."
);

// Conversely, the `dht` feature is not supported in WASM builds; make that
// explicit so accidental `--all-features` builds fail fast.
#[cfg(all(target_family = "wasm", feature = "dht"))]
compile_error!(
    "The `dht` feature is not available for WebAssembly targets. Disable it when building for WASM."
);
