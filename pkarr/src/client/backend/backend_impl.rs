#[cfg(relays)]
use std::time::Duration;
#[cfg(relays)]
use url::Url;

use crate::client::{BuildError, PublishError, ResolveError};
#[cfg(dht)]
use crate::dht::{DhtClient, ReportPolicy};
use crate::{PublicKey, SignedPacket, StoredNodeCount};

#[cfg(all(dht, relays))]
use super::combined::CombinedBackend;
#[cfg(dht)]
use super::dht::DhtBackend;
#[cfg(relays)]
use super::relay::RelayBackend;
use super::BackendResolvePolicy;

#[derive(Debug)]
pub(in crate::client) enum Backend {
    #[cfg(relays)]
    Relay(RelayBackend),
    #[cfg(dht)]
    Dht(DhtBackend),
    #[cfg(all(dht, relays))]
    Combined(CombinedBackend),
}

impl Backend {
    #[cfg(relays)]
    pub(in crate::client) fn relay(
        relays: Vec<Url>,
        request_timeout: Duration,
        reqwest_client: Option<reqwest::Client>,
    ) -> Result<Self, BuildError> {
        if relays.is_empty() {
            return Err(BuildError::EmptyListOfRelays);
        }
        let backend = RelayBackend::new(relays, request_timeout, reqwest_client)
            .map_err(BuildError::RelayBuildError)?;
        Ok(Self::Relay(backend))
    }

    #[cfg(dht)]
    pub(in crate::client) fn dht(
        config: mainline::Config,
        report_policy: ReportPolicy,
    ) -> Result<Self, BuildError> {
        let client = DhtClient::build(config).map_err(BuildError::DhtBuildError)?;
        Ok(Self::Dht(DhtBackend::new(client, report_policy)))
    }

    /// Combine a DHT backend with a relay backend into a combined backend.
    ///
    /// Returns [`Some`] when one backend is [`Backend::Dht`] and the other is
    /// [`Backend::Relay`], preserving both as a [`Backend::Combined`]. Returns
    /// [`None`] for any unsupported pairing.
    pub(in crate::client) fn checked_combine(self, other: Self) -> Option<Self> {
        match (self, other) {
            #[cfg(all(dht, relays))]
            (Self::Dht(dht), Self::Relay(relay)) | (Self::Relay(relay), Self::Dht(dht)) => {
                Some(Self::Combined(CombinedBackend::new(relay, dht)))
            }
            _ => None,
        }
    }

    pub(in crate::client) async fn publish(
        &self,
        packet: &SignedPacket,
    ) -> Result<StoredNodeCount, PublishError> {
        match self {
            #[cfg(relays)]
            Self::Relay(relay) => relay.publish(packet).await,
            #[cfg(dht)]
            Self::Dht(dht) => dht.publish(packet).await,
            #[cfg(all(dht, relays))]
            Self::Combined(combined) => combined.publish(packet).await,
        }
    }

    pub(in crate::client) async fn resolve(
        &self,
        public_key: &PublicKey,
        policy: BackendResolvePolicy<'_>,
    ) -> Result<SignedPacket, ResolveError> {
        match self {
            #[cfg(relays)]
            Self::Relay(relay) => relay.resolve(public_key, policy).await,
            #[cfg(dht)]
            Self::Dht(dht) => dht.resolve(public_key, policy).await,
            #[cfg(all(dht, relays))]
            Self::Combined(combined) => combined.resolve(public_key, policy).await,
        }
    }
}
