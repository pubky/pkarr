mod backend;
pub mod builder;
pub mod cache;
mod client_impl;
mod errors;

#[cfg(all(test, not(wasm_browser)))]
mod tests;
#[cfg(all(test, wasm_browser))]
mod tests_web;

pub use builder::ClientBuilder;
#[cfg(relays)]
pub use builder::InvalidRelayUrl;
pub use client_impl::Client;
pub use errors::{BuildError, PublishError, ResolveError};
