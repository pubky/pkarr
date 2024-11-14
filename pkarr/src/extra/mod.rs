#[cfg(feature = "endpoints")]
pub mod endpoints;

#[cfg(feature = "reqwest-resolve")]
pub mod reqwest;

#[cfg(all(not(target_arch = "wasm32"), feature = "lmdb-cache"))]
pub mod lmdb_cache;
