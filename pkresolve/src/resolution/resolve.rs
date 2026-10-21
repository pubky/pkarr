//! Drive the shared traversal with synchronous or asynchronous packet retrieval.

use crate::{AsyncPacketProvider, Error, PacketProvider};

use super::{
    traversal::{ResolutionStep, ResolutionTraversal},
    Resolution, ResolutionRequest,
};

/// Derive routes with direct raw-key TLS for the requested Pubky service.
/// The provider supplies the validated packet and owns caching, proof collection,
/// and I/O timeouts. The result expires at the earliest packet deadline in the
/// alias chain. Traversal visits at most 32 names and rejects loops.
/// This function calls the provider synchronously and needs no async runtime.
/// Use [`derive_resolution_async`] for an asynchronous packet source.
///
/// # Errors
/// Rejects missing packets, expired records, ambiguous DNS sets, and unsupported
/// routing. No compatible ServiceMode record means failure, even if address
/// records exist.
/// Callers must check `expires_at` before reusing the result.
pub fn derive_resolution(
    request: &ResolutionRequest,
    provider: &(impl PacketProvider + ?Sized),
) -> Result<Resolution, Error> {
    let mut step = ResolutionTraversal::start(request)?;
    loop {
        match step {
            ResolutionStep::NeedPacket(traversal) => {
                let packet = provider.get_packet(traversal.packet_key());
                step = traversal.accept_packet(packet)?;
            }
            ResolutionStep::Complete(resolution) => return Ok(resolution),
        }
    }
}

/// Derive routes using an asynchronous packet source.
/// Uses the same routing and freshness rules as [`derive_resolution`]. Only
/// packet retrieval is asynchronous; dropping this future drops the lookup.
///
/// # Errors
/// Returns the same provider and resolution errors as [`derive_resolution`].
pub async fn derive_resolution_async(
    request: &ResolutionRequest,
    provider: &(impl AsyncPacketProvider + ?Sized),
) -> Result<Resolution, Error> {
    let mut step = ResolutionTraversal::start(request)?;
    loop {
        match step {
            ResolutionStep::NeedPacket(traversal) => {
                let packet = provider.get_packet(traversal.packet_key()).await;
                step = traversal.accept_packet(packet)?;
            }
            ResolutionStep::Complete(resolution) => return Ok(resolution),
        }
    }
}
