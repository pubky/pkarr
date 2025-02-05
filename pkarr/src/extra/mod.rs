#[cfg(feature = "endpoints")]
pub mod endpoints;

#[cfg(all(not(target_family = "wasm"), feature = "reqwest-resolve"))]
pub mod reqwest;

#[cfg(all(not(target_family = "wasm"), feature = "tls"))]
pub mod tls;

#[cfg(all(not(target_family = "wasm"), feature = "lmdb-cache"))]
pub mod lmdb_cache;
