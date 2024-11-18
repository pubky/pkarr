use std::sync::Arc;

#[cfg(feature = "endpoints")]
pub mod endpoints;

#[cfg(all(not(target_arch = "wasm32"), feature = "reqwest-resolve"))]
pub mod reqwest;

#[cfg(all(not(target_arch = "wasm32"), feature = "tls"))]
pub mod tls;

#[cfg(all(not(target_arch = "wasm32"), feature = "lmdb-cache"))]
pub mod lmdb_cache;

#[cfg(all(not(target_arch = "wasm32"), feature = "reqwest-builder"))]
impl From<crate::Client> for ::reqwest::ClientBuilder {
    fn from(client: crate::Client) -> Self {
        ::reqwest::ClientBuilder::new()
            .dns_resolver(Arc::new(client.clone()))
            .use_preconfigured_tls(rustls::ClientConfig::from(client))
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "reqwest-builder"))]
impl From<crate::client::relay::Client> for ::reqwest::ClientBuilder {
    fn from(client: crate::client::relay::Client) -> Self {
        ::reqwest::ClientBuilder::new()
            .dns_resolver(Arc::new(client.clone()))
            .use_preconfigured_tls(rustls::ClientConfig::from(client))
    }
}
