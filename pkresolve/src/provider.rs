//! Packet sources supplied by the host.

use crate::{PublicKey, SignedPacket};
use std::{
    collections::HashMap, convert::Infallible, error::Error as StdError, future::Future, sync::Arc,
};

/// Supplies valid packets for the requested key from a cache or network source.
/// Providers own validation, caching, and proof collection. During a traversal,
/// every lookup for a key must return the same packet version. Use a fresh
/// provider snapshot for each traversal if the underlying cache can change.
/// Preserve the packet's `last_seen` on cache reads. Providers own I/O timeouts.
/// `None` means missing evidence. Use [`AsyncPacketProvider`] for async I/O.
pub trait PacketProvider {
    /// The adapter's underlying lookup failure.
    type Error: StdError + Send + Sync + 'static;

    /// Retrieve the requested owner's packet synchronously.
    fn get_packet(&self, key: &PublicKey) -> Result<Option<Arc<SignedPacket>>, Self::Error>;
}

/// Supplies packets asynchronously, with the same validation, snapshot, and
/// `last_seen` requirements as [`PacketProvider`]. Providers own I/O timeouts;
/// dropping a pending lookup should cancel it. `None` means missing evidence.
pub trait AsyncPacketProvider {
    /// The adapter's underlying lookup failure.
    type Error: StdError + Send + Sync + 'static;

    /// Retrieve the requested owner's packet.
    fn get_packet(
        &self,
        key: &PublicKey,
    ) -> impl Future<Output = Result<Option<Arc<SignedPacket>>, Self::Error>> + Send;
}

/// A local cache or verified service response needs no custom provider wrapper.
impl PacketProvider for HashMap<PublicKey, Arc<SignedPacket>> {
    type Error = Infallible;

    fn get_packet(&self, key: &PublicKey) -> Result<Option<Arc<SignedPacket>>, Infallible> {
        Ok(self.get(key).cloned())
    }
}
