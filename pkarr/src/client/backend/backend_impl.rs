use ntimestamp::Timestamp;
#[cfg(relays)]
use std::time::Duration;
#[cfg(relays)]
use url::Url;

use crate::client::errors::ResolveError;
use crate::client::{BuildError, PublishError};
#[cfg(dht)]
use crate::dht::{DhtClient, ReportPolicy};
use crate::{PublicKey, ResolvePolicy, SignedPacket, StoredNodeCount};

#[cfg(all(dht, relays))]
use super::both::BothBackend;
#[cfg(dht)]
use super::dht::DhtBackend;
#[cfg(relays)]
use super::relays::RelaysClient;

#[derive(Debug)]
pub(in crate::client) enum Backend {
    #[cfg(relays)]
    Relays(RelaysClient),
    #[cfg(dht)]
    Dht(DhtBackend),
    #[cfg(all(dht, relays))]
    Both(BothBackend),
}

impl Backend {
    #[cfg(relays)]
    pub(in crate::client) fn relays(
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
    pub(in crate::client) fn dht(
        config: mainline::Config,
        report_policy: ReportPolicy,
    ) -> Result<Self, BuildError> {
        DhtClient::build(config)
            .map(|client| Self::Dht(super::dht::DhtBackend::new(client, report_policy)))
            .map_err(BuildError::DhtBuildError)
    }

    /// Merge a DHT backend with a relays backend into a combined backend.
    ///
    /// Returns [`Some`] when one backend is [`Backend::Dht`] and the other is
    /// [`Backend::Relays`], preserving both as a [`Backend::Both`]. Returns
    /// [`None`] for any unsupported pairing.
    pub(in crate::client) fn checked_merge(self, other: Self) -> Option<Self> {
        match (self, other) {
            #[cfg(all(dht, relays))]
            (Self::Dht(dht), Self::Relays(relays)) | (Self::Relays(relays), Self::Dht(dht)) => {
                Some(Self::Both(BothBackend::new(relays, dht)))
            }
            _ => None,
        }
    }

    pub(in crate::client) async fn publish(
        &self,
        signed_packet: &SignedPacket,
    ) -> Result<StoredNodeCount, PublishError> {
        match self {
            #[cfg(relays)]
            Self::Relays(relays) => relays.publish(signed_packet).await,
            #[cfg(dht)]
            Self::Dht(dht) => dht.publish(signed_packet).await,
            #[cfg(all(dht, relays))]
            Self::Both(both) => both.publish(signed_packet).await,
        }
    }

    pub(in crate::client) async fn resolve(
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
            Self::Both(both) => both.resolve(public_key, policy, more_recent_than).await,
        }
    }
}
