use std::time::Duration;

use futures_buffered::FuturesUnorderedBounded;
use futures_lite::StreamExt;
use ntimestamp::Timestamp;
use url::Url;

use super::publish_result_accumulator::PublishResultAccumulator;
use super::resolve_result_accumulator::ResolveResultAccumulator;
use crate::client::{PublishError, ResolveError};
use crate::relay_client::{RelayClient, RelayError};
use crate::{PublicKey, ResolvePolicy, SignedPacket, StoredNodeCount};

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
    ) -> Result<StoredNodeCount, PublishError> {
        let mut futures = FuturesUnorderedBounded::new(self.relays.len());
        for relay in &self.relays {
            futures.push(relay.publish(signed_packet));
        }

        let mut accumulator = PublishResultAccumulator::default();
        while let Some(result) = futures.next().await {
            accumulator.record_result(result.map_err(Into::into));
        }
        accumulator.into_result()
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

        let mut accumulator = ResolveResultAccumulator::default();
        while let Some(result) = futures.next().await {
            if accumulator.record_result(result.map_err(Into::into)) && finish_on_first_packet {
                return accumulator.into_result();
            }
        }
        accumulator.into_result()
    }
}
