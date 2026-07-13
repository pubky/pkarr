use std::time::Duration;

use futures_buffered::FuturesUnorderedBounded;
use futures_lite::StreamExt;
use url::Url;

use super::publish_result_accumulator::PublishResultAccumulator;
use super::resolve_result_accumulator::ResolveResultAccumulator;
use super::BackendResolvePolicy;
use crate::client::{PublishError, ResolveError};
use crate::relay_client::{RelayClient, RelayError};
use crate::{PublicKey, SignedPacket, StoredNodeCount};

#[derive(Clone, Debug)]
pub(in crate::client) struct RelayBackend {
    relays: Vec<RelayClient>,
}

impl RelayBackend {
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
        packet: &SignedPacket,
    ) -> Result<StoredNodeCount, PublishError> {
        let mut requests: FuturesUnorderedBounded<_> = self
            .relays
            .iter()
            .map(|relay| relay.publish(packet))
            .collect();

        let mut accumulator = PublishResultAccumulator::default();
        while let Some(result) = requests.next().await {
            accumulator.record_result(result.map_err(Into::into));
        }
        accumulator.into_result()
    }

    pub(super) async fn resolve(
        &self,
        key: &PublicKey,
        policy: BackendResolvePolicy<'_>,
    ) -> Result<SignedPacket, ResolveError> {
        let relay_policy = policy.as_relay_policy();
        let request_lower_bound = policy.relay_request_lower_bound();
        let complete_on_first = policy.completes_on_first_acceptable();
        let mut requests: FuturesUnorderedBounded<_> = self
            .relays
            .iter()
            .map(|relay| relay.resolve(key, relay_policy, request_lower_bound))
            .collect();

        let mut accumulator = ResolveResultAccumulator::new(policy.cache_context());
        while let Some(result) = requests.next().await {
            let acceptable = accumulator.record_result(result.map_err(Into::into));
            if acceptable && complete_on_first {
                break;
            }
        }
        accumulator.into_result()
    }
}
