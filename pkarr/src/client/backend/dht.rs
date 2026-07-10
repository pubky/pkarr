use ntimestamp::Timestamp;

use crate::dht::{DhtClient, ReportPolicy, ResolveReport};
use crate::{PublicKey, ResolvePolicy, SignedPacket, StoredNodeCount};

use crate::client::{PublishError, ResolveError};

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
        policy: ResolvePolicy,
        more_recent_than: Option<Timestamp>,
    ) -> Result<SignedPacket, ResolveError> {
        if matches!(policy, ResolvePolicy::LocalOrRelayCacheOnly) {
            return Err(ResolveError::NotFound);
        }

        let response = self.client.resolve(public_key, more_recent_than).await;

        if matches!(policy, ResolvePolicy::CacheFirst) {
            if let Some(packet) = response.first() {
                return Ok(packet.clone());
            }
        }

        let outcome = response.complete().await;
        self.log_resolve_warnings(public_key, &outcome.report);
        outcome.most_recent.map_err(Into::into)
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
