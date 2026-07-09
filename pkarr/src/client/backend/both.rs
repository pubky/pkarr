use ntimestamp::Timestamp;

use crate::{PublicKey, ResolvePolicy, SignedPacket, StoredNodeCount};

use super::dht::DhtBackend;
use super::publish_result_accumulator::PublishResultAccumulator;
use super::relays::RelaysClient;
use super::resolve_result_accumulator::ResolveResultAccumulator;
use crate::client::{PublishError, ResolveError};

#[derive(Debug)]
pub(in crate::client) struct BothBackend {
    relays: RelaysClient,
    dht: DhtBackend,
}

impl BothBackend {
    pub(super) fn new(relays: RelaysClient, dht: DhtBackend) -> Self {
        Self { relays, dht }
    }

    pub(super) async fn publish(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<StoredNodeCount, PublishError> {
        let relay_publish = self.relays.publish(signed_packet, cas);
        let dht_publish = self.dht.publish(signed_packet, cas);
        let (relay_result, dht_result) = tokio::join!(relay_publish, dht_publish);

        let mut accumulator = PublishResultAccumulator::default();
        accumulator.record_result(relay_result);
        accumulator.record_result(dht_result);
        accumulator.into_result()
    }

    pub(super) async fn resolve(
        &self,
        public_key: &PublicKey,
        policy: ResolvePolicy,
        more_recent_than: Option<Timestamp>,
    ) -> Result<SignedPacket, ResolveError> {
        let r1 = self.relays.resolve(public_key, policy, more_recent_than);
        let r2 = self.dht.resolve(public_key, policy, more_recent_than);

        match policy {
            ResolvePolicy::LocalOrRelayCacheOnly => r1.await,
            ResolvePolicy::CacheFirst => {
                tokio::pin!(r1);
                tokio::pin!(r2);
                tokio::select!(
                    result = &mut r1 => ok_or_wait_second(result, r2).await,
                    result = &mut r2 => ok_or_wait_second(result, r1).await,
                )
            }
            ResolvePolicy::DhtNetworkOnly => {
                let (r1, r2) = tokio::join!(r1, r2);
                merge_resolve_results(r1, r2)
            }
        }
    }
}

async fn ok_or_wait_second(
    first: Result<SignedPacket, ResolveError>,
    second: impl std::future::Future<Output = Result<SignedPacket, ResolveError>>,
) -> Result<SignedPacket, ResolveError> {
    if first.is_ok() {
        return first;
    }
    merge_resolve_results(first, second.await)
}

fn merge_resolve_results(
    r1: Result<SignedPacket, ResolveError>,
    r2: Result<SignedPacket, ResolveError>,
) -> Result<SignedPacket, ResolveError> {
    let mut accumulator = ResolveResultAccumulator::default();
    accumulator.record_result(r1);
    accumulator.record_result(r2);
    accumulator.into_result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Keypair;

    fn signed_packet(timestamp: u64) -> SignedPacket {
        SignedPacket::builder()
            .timestamp(Timestamp::from(timestamp))
            .sign(&Keypair::random())
            .unwrap()
    }

    #[test]
    fn merge_resolve_results_preserves_packet_when_invalid_seq_is_not_newer() {
        let packet = signed_packet(10);

        assert_eq!(
            merge_resolve_results(
                Ok(packet.clone()),
                Err(ResolveError::InvalidSignedPacket { seq: 10 }),
            ),
            Ok(packet)
        );
    }

    #[test]
    fn merge_resolve_results_returns_invalid_signed_packet_when_seq_is_newer() {
        assert_eq!(
            merge_resolve_results(
                Ok(signed_packet(10)),
                Err(ResolveError::InvalidSignedPacket { seq: 11 }),
            ),
            Err(ResolveError::InvalidSignedPacket { seq: 11 })
        );
    }

    #[test]
    fn merge_resolve_results_keeps_highest_invalid_signed_packet_seq() {
        assert_eq!(
            merge_resolve_results(
                Err(ResolveError::InvalidSignedPacket { seq: 10 }),
                Err(ResolveError::InvalidSignedPacket { seq: 11 }),
            ),
            Err(ResolveError::InvalidSignedPacket { seq: 11 })
        );
    }
}
