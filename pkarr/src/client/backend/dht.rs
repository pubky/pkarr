use crate::dht::{DhtClient, ReportPolicy, ResolveOutcome, ResolveReport, ResolveResponse};
use crate::{PublicKey, SignedPacket, StoredNodeCount};

use crate::client::{PublishError, ResolveError};

use super::{
    resolve_result_accumulator::ResolveResultAccumulator, BackendResolvePolicy, CacheContext,
};

#[derive(Debug)]
pub(in crate::client) struct DhtBackend {
    client: DhtClient,
    report_policy: ReportPolicy,
}

impl DhtBackend {
    pub(super) fn new(client: DhtClient, report_policy: ReportPolicy) -> Self {
        Self {
            client,
            report_policy,
        }
    }

    pub(super) async fn publish(
        &self,
        signed_packet: &SignedPacket,
    ) -> Result<StoredNodeCount, PublishError> {
        let stored_on = self.client.publish(signed_packet).await?;
        self.log_publish_warnings(&signed_packet.public_key(), stored_on);
        Ok(stored_on)
    }

    pub(super) async fn resolve(
        &self,
        public_key: &PublicKey,
        policy: BackendResolvePolicy<'_>,
    ) -> Result<SignedPacket, ResolveError> {
        match policy {
            BackendResolvePolicy::LocalOrRelayCacheOnly => Err(ResolveError::NotFound),
            BackendResolvePolicy::CacheFirst(context) => {
                self.resolve_cache_first(public_key, context).await
            }
            BackendResolvePolicy::DhtNetworkOnly => {
                let response = self.client.resolve(public_key, None).await;
                let outcome = self.complete_resolve(public_key, response).await;
                outcome.most_recent.map_err(Into::into)
            }
        }
    }

    async fn resolve_cache_first(
        &self,
        public_key: &PublicKey,
        cache_context: CacheContext<'_>,
    ) -> Result<SignedPacket, ResolveError> {
        let response = self
            .client
            .resolve(public_key, cache_context.dht_request_lower_bound())
            .await;

        let mut accumulator = ResolveResultAccumulator::new(Some(cache_context));
        if let Some(packet) = response.first() {
            if accumulator.record_result(Ok(packet.clone())) {
                return accumulator.into_result();
            }
        }

        let outcome = self.complete_resolve(public_key, response).await;
        accumulator.record_result(outcome.most_recent.map_err(Into::into));
        accumulator.into_result()
    }

    async fn complete_resolve(
        &self,
        public_key: &PublicKey,
        response: ResolveResponse,
    ) -> ResolveOutcome {
        let outcome = response.complete().await;
        self.log_resolve_warnings(public_key, &outcome.report);
        outcome
    }

    fn log_publish_warnings(&self, public_key: &PublicKey, stored_on: StoredNodeCount) {
        let warnings = self.report_policy.classify_publish_result(stored_on);
        if !warnings.is_empty() {
            tracing::warn!(
                ?public_key,
                ?warnings,
                "DHT publish completed with warnings"
            );
        }
    }

    fn log_resolve_warnings(&self, public_key: &PublicKey, report: &ResolveReport) {
        let warnings = self.report_policy.classify_resolve_report(report);
        if !warnings.is_empty() {
            tracing::warn!(
                ?public_key,
                ?warnings,
                "DHT resolve completed with warnings"
            );
        }
    }
}
