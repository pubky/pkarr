#![doc = include_str!("../README.md")]

mod identity;
mod proxy_protocol;

pub use identity::{ClientAddr, ConfigError, IdentifyError, IdentityMode};
pub use proxy_protocol::ProxyProtocol;
