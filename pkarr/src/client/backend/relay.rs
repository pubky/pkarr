use std::time::Duration;

use futures_buffered::FuturesUnorderedBounded;
use futures_lite::StreamExt;
use ntimestamp::Timestamp;
use url::Url;

use super::publish_result_accumulator::PublishResultAccumulator;
use super::resolve_result_accumulator::ResolveResultAccumulator;
use super::{BackendResolvePolicy, CacheContext};
use crate::client::{PublishError, ResolveError};
use crate::relay_client::{RelayClient, RelayError};
use crate::{PublicKey, ResolvePolicy, SignedPacket, StoredNodeCount};

#[derive(Clone, Debug)]
pub(in crate::client) struct RelayBackend {
    relays: Vec<RelayClient>,
}

struct RelayResolveOptions<'a> {
    relay_policy: ResolvePolicy,
    cache_context: Option<CacheContext<'a>>,
    request_lower_bound: Option<Timestamp>,
    complete_on_first_acceptable: bool,
}

impl<'a> From<BackendResolvePolicy<'a>> for RelayResolveOptions<'a> {
    fn from(policy: BackendResolvePolicy<'a>) -> Self {
        match policy {
            BackendResolvePolicy::LocalOrRelayCacheOnly => Self {
                relay_policy: ResolvePolicy::LocalOrRelayCacheOnly,
                cache_context: None,
                request_lower_bound: None,
                complete_on_first_acceptable: true,
            },
            BackendResolvePolicy::CacheFirst(context) => Self {
                relay_policy: ResolvePolicy::CacheFirst,
                cache_context: Some(context),
                request_lower_bound: context.relay_request_lower_bound(),
                complete_on_first_acceptable: true,
            },
            BackendResolvePolicy::DhtNetworkOnly => Self {
                relay_policy: ResolvePolicy::DhtNetworkOnly,
                cache_context: None,
                request_lower_bound: None,
                complete_on_first_acceptable: false,
            },
        }
    }
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
        let options = RelayResolveOptions::from(policy);
        let mut requests: FuturesUnorderedBounded<_> = self
            .relays
            .iter()
            .map(|relay| relay.resolve(key, options.relay_policy, options.request_lower_bound))
            .collect();

        let mut accumulator = ResolveResultAccumulator::new(options.cache_context);
        while let Some(result) = requests.next().await {
            let acceptable = accumulator.record_result(result.map_err(Into::into));
            if acceptable && options.complete_on_first_acceptable {
                break;
            }
        }
        accumulator.into_result()
    }
}
