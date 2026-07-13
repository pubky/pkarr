use crate::{PublicKey, SignedPacket, StoredNodeCount};

use super::dht::DhtBackend;
use super::publish_result_accumulator::PublishResultAccumulator;
use super::relay::RelayBackend;
use super::resolve_result_accumulator::ResolveResultAccumulator;
use super::{BackendResolvePolicy, CacheContext};
use crate::client::{PublishError, ResolveError};

#[derive(Debug)]
pub(in crate::client) struct CombinedBackend {
    relay: RelayBackend,
    dht: DhtBackend,
}

impl CombinedBackend {
    pub(super) fn new(relay: RelayBackend, dht: DhtBackend) -> Self {
        Self { relay, dht }
    }

    pub(super) async fn publish(
        &self,
        signed_packet: &SignedPacket,
    ) -> Result<StoredNodeCount, PublishError> {
        let relay_publish = self.relay.publish(signed_packet);
        let dht_publish = self.dht.publish(signed_packet);
        let (relay_result, dht_result) = tokio::join!(relay_publish, dht_publish);

        let mut accumulator = PublishResultAccumulator::default();
        accumulator.record_result(relay_result);
        accumulator.record_result(dht_result);
        accumulator.into_result()
    }

    pub(super) async fn resolve(
        &self,
        public_key: &PublicKey,
        policy: BackendResolvePolicy<'_>,
    ) -> Result<SignedPacket, ResolveError> {
        let first_resolve = self.relay.resolve(public_key, policy);
        let second_resolve = self.dht.resolve(public_key, policy);

        match policy {
            BackendResolvePolicy::LocalOrRelayCacheOnly => first_resolve.await,
            BackendResolvePolicy::CacheFirst(context) => {
                tokio::pin!(first_resolve);
                tokio::pin!(second_resolve);
                tokio::select!(
                    result = &mut first_resolve => {
                        first_acceptable_or_wait_second(result, second_resolve, context).await
                    }
                    result = &mut second_resolve => {
                        first_acceptable_or_wait_second(result, first_resolve, context).await
                    }
                )
            }
            BackendResolvePolicy::DhtNetworkOnly => {
                let (first_result, second_result) = tokio::join!(first_resolve, second_resolve);
                merge_resolve_results(first_result, second_result)
            }
        }
    }
}

async fn first_acceptable_or_wait_second(
    first: Result<SignedPacket, ResolveError>,
    second: impl std::future::Future<Output = Result<SignedPacket, ResolveError>>,
    context: CacheContext<'_>,
) -> Result<SignedPacket, ResolveError> {
    let mut accumulator = ResolveResultAccumulator::new(Some(context));
    if accumulator.record_result(first) {
        return accumulator.into_result();
    }
    accumulator.record_result(second.await);
    accumulator.into_result()
}

fn merge_resolve_results(
    first_result: Result<SignedPacket, ResolveError>,
    second_result: Result<SignedPacket, ResolveError>,
) -> Result<SignedPacket, ResolveError> {
    let mut accumulator = ResolveResultAccumulator::default();
    accumulator.record_result(first_result);
    accumulator.record_result(second_result);
    accumulator.into_result()
}

#[cfg(test)]
mod tests {
    use ntimestamp::Timestamp;

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

    #[tokio::test]
    async fn cache_first_waits_for_fresh_second_backend() {
        let keypair = Keypair::random();
        let mut expired = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .sign(&keypair)
            .unwrap();
        expired.set_last_seen(&(Timestamp::now() - 60 * 1_000_000_u64));
        let fresh = SignedPacket::builder()
            .timestamp(Timestamp::from(11))
            .sign(&keypair)
            .unwrap();

        let resolved = first_acceptable_or_wait_second(
            Ok(expired),
            std::future::ready(Ok(fresh.clone())),
            CacheContext::new(None, 30, 30),
        )
        .await
        .unwrap();

        assert_eq!(resolved, fresh);
    }
}
