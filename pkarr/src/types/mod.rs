//! Protocol data types.

mod keys;
mod resolve_policy;
#[cfg(feature = "signed_packet")]
pub(crate) mod signed_packet;

pub use keys::{Keypair, PublicKey};
pub use resolve_policy::ResolvePolicy;
#[cfg(feature = "signed_packet")]
pub use signed_packet::{SignedPacket, SignedPacketBuilder};

/// Documentation alias for the number of DHT nodes that acknowledged storing a
/// published packet.
pub type StoredNodeCount = u32;

/// Exported type errors.
pub mod errors {
    pub use super::keys::PublicKeyError;

    #[cfg(feature = "signed_packet")]
    pub use super::signed_packet::{SignedPacketBuildError, SignedPacketVerifyError};
}
