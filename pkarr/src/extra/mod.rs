//! Extra features that might benefit most but not all developers building apps using Pkarr.
//!

#[cfg(feature = "endpoints")]
pub mod endpoints;

#[cfg(all(not(wasm_browser), feature = "reqwest-resolve"))]
pub mod reqwest;

#[cfg(all(not(wasm_browser), feature = "tls"))]
pub mod tls;

#[cfg(all(not(wasm_browser), feature = "lmdb-cache"))]
pub mod lmdb_cache;
