//! Packet lookup and bundle collection for a single resolution request.

use std::{collections::HashMap, sync::Arc};

use base64::{engine::general_purpose::STANDARD, Engine};
use http::StatusCode;
use pkarr::{PublicKey, ResolvePolicy, SignedPacket};
use pkresolve::AsyncPacketProvider;
use tokio::sync::Mutex;

use crate::{dht_service::DhtService, error::Error, real_ip::RealIp};

pub(super) const MAX_RESOLUTION_PACKETS: usize = 32;

/// Collect the exact packet versions used by one traversal. Shared caching
/// remains in DhtService; these references keep the proof consistent if that
/// cache changes during resolution. Each owner appears once in the bundle.
pub(super) struct RelayPacketProvider<'a> {
    dht: &'a DhtService,
    real_ip: Option<RealIp>,
    lookups: Mutex<HashMap<PublicKey, Result<Arc<SignedPacket>, Error>>>,
}

impl<'a> RelayPacketProvider<'a> {
    pub(super) fn new(dht: &'a DhtService, real_ip: Option<RealIp>) -> Self {
        Self {
            dht,
            real_ip,
            lookups: Mutex::new(HashMap::new()),
        }
    }

    /// Include completed packet fetches even when a later lookup failed or timed out.
    pub(super) fn into_packets(self) -> Vec<String> {
        self.lookups
            .into_inner()
            .into_values()
            .filter_map(Result::ok)
            .map(|packet| STANDARD.encode(packet.serialize()))
            .collect()
    }
}

impl AsyncPacketProvider for RelayPacketProvider<'_> {
    type Error = Error;

    async fn get_packet(&self, key: &PublicKey) -> Result<Option<Arc<SignedPacket>>, Error> {
        // Serialize lookups within this traversal, including I/O, so each key
        // is fetched once. Other requests use their own provider and lock.
        let mut lookups = self.lookups.lock().await;
        if let Some(result) = lookups.get(key) {
            return result.clone().map(Some);
        }
        if lookups.len() >= MAX_RESOLUTION_PACKETS {
            return Err(Error::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "resolution packet limit exceeded",
            ));
        }

        let result = self
            .dht
            .resolve_packet(key, ResolvePolicy::CacheFirst, self.real_ip.as_ref())
            .await
            .map(Arc::new);
        lookups.insert(key.clone(), result.clone());
        result.map(Some)
    }
}
