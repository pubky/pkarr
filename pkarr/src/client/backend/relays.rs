use std::time::Duration;

use futures_buffered::FuturesUnorderedBounded;
use futures_lite::StreamExt;
use ntimestamp::Timestamp;
use url::Url;

use super::publish_results::PublishResults;
use super::resolve_results::ResolveResults;
use super::ResolveError;
use crate::client::PublishError;
use crate::relay_client::{RelayClient, RelayError};
use crate::{PublicKey, ResolvePolicy, SignedPacket};

#[derive(Clone, Debug)]
pub(in crate::client) struct RelaysClient {
    relays: Vec<RelayClient>,
}

impl RelaysClient {
    pub(super) fn new(
        relays: Vec<Url>,
        timeout: Duration,
        client: Option<reqwest::Client>,
    ) -> Result<Self, RelayError> {
        let client = match client {
            Some(client) => client,
            None => reqwest::Client::builder()
                .build()
                .map_err(|e| RelayError::Build(e.to_string()))?,
        };
        let relays = relays
            .into_iter()
            .map(|url| RelayClient::new(url, client.clone(), timeout))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { relays })
    }

    pub(super) async fn publish(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<u32, PublishError> {
        let mut futures = FuturesUnorderedBounded::new(self.relays.len());
        for relay in &self.relays {
            futures.push(relay.publish(signed_packet, cas));
        }

        let mut results = PublishResults::default();
        while let Some(result) = futures.next().await {
            results.record(result.map_err(Into::into));
        }
        results.finish()
    }

    pub(super) async fn resolve(
        &self,
        public_key: &PublicKey,
        policy: ResolvePolicy,
        more_recent_than: Option<Timestamp>,
    ) -> Result<SignedPacket, ResolveError> {
        let mut futures = FuturesUnorderedBounded::new(self.relays.len());
        for relay in &self.relays {
            futures.push(relay.resolve(public_key, policy, more_recent_than));
        }

        let finish_on_first_packet = matches!(
            policy,
            ResolvePolicy::LocalOrRelayCacheOnly | ResolvePolicy::CacheFirst
        );

        let mut results = ResolveResults::default();
        while let Some(result) = futures.next().await {
            if results.record(result.map_err(Into::into)) && finish_on_first_packet {
                return results.finish();
            }
        }
        results.finish()
    }
}
