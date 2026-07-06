use ntimestamp::Timestamp;
#[cfg(relays)]
use std::time::Duration;
#[cfg(relays)]
use url::Url;

#[cfg(dht)]
use crate::dht::{DhtClient, ReportPolicy};
use crate::{PublicKey, ResolvePolicy, SignedPacket};

#[cfg(all(dht, relays))]
mod both;
#[cfg(dht)]
mod dht;
#[cfg(relays)]
mod publish_results;
#[cfg(relays)]
mod relays;
#[cfg(relays)]
mod resolve_results;

use super::errors::ResolveError;
use super::{BuildError, PublishError};
#[cfg(relays)]
use relays::RelaysClient;

#[derive(Debug)]
pub(super) enum Backend {
    #[cfg(relays)]
    Relays(RelaysClient),
    #[cfg(dht)]
    Dht(dht::DhtBackend),
    #[cfg(all(dht, relays))]
    Both(RelaysClient, dht::DhtBackend),
}

impl Backend {
    #[cfg(relays)]
    pub(super) fn relays(
        relays: Vec<Url>,
        request_timeout: Duration,
        reqwest_client: Option<reqwest::Client>,
    ) -> Result<Self, BuildError> {
        if relays.is_empty() {
            return Err(BuildError::EmptyListOfRelays);
        }
        RelaysClient::new(relays, request_timeout, reqwest_client)
            .map(Self::Relays)
            .map_err(BuildError::RelayBuildError)
    }

    #[cfg(dht)]
    pub(super) fn dht(
        config: mainline::Config,
        report_policy: ReportPolicy,
    ) -> Result<Self, BuildError> {
        DhtClient::build(config)
            .map(|client| Self::Dht(dht::DhtBackend::new(client, report_policy)))
            .map_err(BuildError::DhtBuildError)
    }

    pub(super) fn try_merge(self, other: Backend) -> Option<Self> {
        match (self, other) {
            #[cfg(all(dht, relays))]
            (Backend::Dht(dht), Backend::Relays(relays))
            | (Backend::Relays(relays), Backend::Dht(dht)) => Some(Backend::Both(relays, dht)),
            _ => None,
        }
    }

    pub(super) async fn publish(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<u32, PublishError> {
        match self {
            #[cfg(relays)]
            Self::Relays(relays) => relays.publish(signed_packet, cas).await,
            #[cfg(dht)]
            Self::Dht(dht) => dht.publish(signed_packet, cas).await,
            #[cfg(all(dht, relays))]
            Self::Both(relays, dht) => both::publish(relays, dht, signed_packet, cas).await,
        }
    }

    pub(super) async fn resolve(
        &self,
        public_key: &PublicKey,
        policy: ResolvePolicy,
        more_recent_than: Option<Timestamp>,
    ) -> Result<SignedPacket, ResolveError> {
        match self {
            #[cfg(relays)]
            Self::Relays(relays) => relays.resolve(public_key, policy, more_recent_than).await,
            #[cfg(dht)]
            Self::Dht(dht) => dht.resolve(public_key, policy, more_recent_than).await,
            #[cfg(all(dht, relays))]
            Self::Both(relays, dht) => {
                both::resolve(relays, dht, public_key, policy, more_recent_than).await
            }
        }
    }
}
