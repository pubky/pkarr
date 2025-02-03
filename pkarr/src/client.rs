//! Client implementation.

macro_rules! cross_debug {
    ($($arg:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        log::debug!($($arg)*);
        #[cfg(not(target_arch = "wasm32"))]
        tracing::debug!($($arg)*);
    };
}

pub mod cache;

mod builder;
pub(crate) mod native;
#[cfg(feature = "relays")]
mod relays;

pub use native::{Client, ClientBuilder};

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;
#[cfg(all(test, target_family = "wasm"))]
mod tests_web;
