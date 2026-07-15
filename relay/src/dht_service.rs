use std::sync::{Arc, Mutex};

use http::StatusCode;
use pkarr::dht::{
    DhtClient, DhtInfo, PublishError as DhtPublishError, ReportPolicy,
    ResolveError as DhtResolveError, ResolveReport, ResolveResponse,
};
use pkarr::extra::lmdb_cache::LmdbCache;
use pkarr::{Cache, CacheKey, PublicKey, ResolvePolicy, SignedPacket, StoredNodeCount, Timestamp};
use tracing::warn;

use crate::error::Error;
use crate::rate_limiting::UserDhtRateLimiter;
use crate::real_ip::RealIp;

/// Cache-aware DHT service used by relay handlers.
#[derive(Debug, Clone)]
pub(crate) struct DhtService {
    minimum_ttl: u32,
    maximum_ttl: u32,
    cache: Arc<LmdbCache>,
    cache_write_lock: Arc<Mutex<()>>,
    dht: DhtClient,
    report_policy: ReportPolicy,
    rate_limiter: Option<UserDhtRateLimiter>,
}

impl DhtService {
    /// Create a cache-aware DHT service.
    pub(crate) fn new(
        minimum_ttl: u32,
        maximum_ttl: u32,
        cache: Arc<LmdbCache>,
        dht: DhtClient,
        report_policy: ReportPolicy,
        rate_limiter: Option<UserDhtRateLimiter>,
    ) -> Self {
        Self {
            minimum_ttl,
            maximum_ttl,
            cache,
            cache_write_lock: Arc::new(Mutex::new(())),
            dht,
            report_policy,
            rate_limiter,
        }
    }

    /// Publish a signed packet through the DHT and update the relay cache.
    ///
    /// Returns the stored-node count reported back to relay clients.
    pub(crate) async fn publish(
        &self,
        signed_packet: &SignedPacket,
        real_ip: Option<&RealIp>,
    ) -> Result<StoredNodeCount, Error> {
        let public_key = signed_packet.public_key();
        let key = CacheKey::from(&public_key);
        if let Some(cached) = self.cache.get_read_only(&key) {
            if cached.more_recent_than(signed_packet) {
                return Err(Error::new(
                    StatusCode::CONFLICT,
                    DhtPublishError::NotMostRecent,
                ));
            }
        }

        self.enforce_user_dht_rate_limit(real_ip)?;

        let stored_on = self.dht.publish(signed_packet).await?;
        let warnings = self.report_policy.classify_publish_result(stored_on);
        if !warnings.is_empty() {
            warn!(
                ?public_key,
                ?warnings,
                "DHT publish completed with warnings"
            );
        }

        self.update_cache_if_needed(&key, signed_packet);

        Ok(stored_on)
    }

    /// Resolve a packet using relay cache and DHT policy semantics.
    pub(crate) async fn resolve_packet(
        &self,
        public_key: &PublicKey,
        policy: ResolvePolicy,
        real_ip: Option<&RealIp>,
    ) -> Result<SignedPacket, Error> {
        let key = CacheKey::from(public_key);
        let cached = self.cache.get(&key);

        let packet = match policy {
            ResolvePolicy::CacheOnly => {
                cached.ok_or_else(|| Error::with_status(StatusCode::NOT_FOUND))
            }
            ResolvePolicy::CacheFirst => {
                self.resolve_cache_first(public_key, key, cached.as_ref(), real_ip)
                    .await
            }
            ResolvePolicy::NetworkOnly => self.resolve_network_only(public_key, real_ip).await,
        }?;

        self.update_cache_if_needed(&key, &packet);
        Ok(packet)
    }

    /// Return cache size and capacity.
    pub(crate) fn cache_stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
            capacity: self.cache.capacity(),
        }
    }

    /// Return the relay TTL for a signed packet.
    pub(crate) fn ttl(&self, packet: &SignedPacket) -> u32 {
        packet.ttl(self.minimum_ttl, self.maximum_ttl)
    }

    /// Return information about the underlying DHT node.
    pub(crate) async fn info(&self) -> DhtInfo {
        self.dht.info().await
    }

    async fn resolve_cache_first(
        &self,
        public_key: &PublicKey,
        key: CacheKey,
        cached: Option<&SignedPacket>,
        real_ip: Option<&RealIp>,
    ) -> Result<SignedPacket, Error> {
        if let Some(packet) = cached {
            if !packet.is_expired(self.minimum_ttl, self.maximum_ttl) {
                return Ok(packet.clone());
            }
        }

        let more_recent_than = cached.map(SignedPacket::timestamp).and_then(decrement);
        self.enforce_user_dht_rate_limit(real_ip)?;

        let response = self.dht.resolve(public_key, more_recent_than).await;

        self.resolve_cache_first_dht_response(public_key, key, cached, response)
            .await
    }

    async fn resolve_cache_first_dht_response(
        &self,
        public_key: &PublicKey,
        key: CacheKey,
        cached: Option<&SignedPacket>,
        response: ResolveResponse,
    ) -> Result<SignedPacket, Error> {
        let first_packet = response
            .first()
            .cloned()
            .map(|packet| apply_cached_floor(Ok(packet), cached));

        match first_packet {
            Some(Ok(packet)) => {
                let service = self.clone();
                let public_key = public_key.clone();
                // Do not waste late responses with potentially newer packets.
                tokio::spawn(async move {
                    let resolved = response.complete().await;
                    service.log_resolve_warnings(&public_key, &resolved.report);
                    if let Ok(packet) = resolved.most_recent {
                        service.update_cache_if_needed(&key, &packet);
                    }
                });
                Ok(packet)
            }
            Some(Err(DhtResolveError::NotFound)) | None => {
                let resolved = response.complete().await;
                self.log_resolve_warnings(public_key, &resolved.report);
                apply_cached_floor(resolved.most_recent, cached).map_err(Error::from)
            }
            Some(Err(error)) => Err(error.into()),
        }
    }

    async fn resolve_network_only(
        &self,
        public_key: &PublicKey,
        real_ip: Option<&RealIp>,
    ) -> Result<SignedPacket, Error> {
        // NetworkOnly reports the DHT's current state without using the relay
        // cache as a lower bound or as invalid-sequence suppression.
        self.enforce_user_dht_rate_limit(real_ip)?;

        let resolved = self.dht.resolve(public_key, None).await.complete().await;
        self.log_resolve_warnings(public_key, &resolved.report);

        Ok(resolved.most_recent?)
    }

    fn log_resolve_warnings(&self, public_key: &PublicKey, report: &ResolveReport) {
        let warnings = self.report_policy.classify_resolve_report(report);
        if !warnings.is_empty() {
            warn!(
                ?public_key,
                ?warnings,
                "DHT resolve completed with warnings"
            );
        }
    }

    fn update_cache_if_needed(&self, key: &CacheKey, signed_packet: &SignedPacket) {
        let _lock = self
            .cache_write_lock
            .lock()
            .expect("DhtService cache_write_lock");

        let should_update = match self.cache.get_read_only(key) {
            None => true,
            Some(cached) if signed_packet.more_recent_than(&cached) => true,
            Some(cached) if signed_packet.is_same_as(&cached) => {
                // The packet is the same but we refresh the last seen.
                signed_packet.last_seen() > cached.last_seen()
            }
            _ => false,
        };

        if should_update {
            self.cache.put(key, signed_packet);
        }
    }

    #[allow(clippy::result_large_err)]
    fn enforce_user_dht_rate_limit(&self, real_ip: Option<&RealIp>) -> Result<(), Error> {
        if let (Some(rate_limiter), Some(real_ip)) = (&self.rate_limiter, real_ip) {
            if rate_limiter.is_limited(&real_ip.0) {
                return Err(Error::new(
                    StatusCode::TOO_MANY_REQUESTS,
                    "Too many requests to DHT nodes",
                ));
            }
        }

        Ok(())
    }
}

/// Cache size metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CacheStats {
    /// Number of packets in the cache.
    pub(crate) size: usize,
    /// Maximum cache capacity.
    pub(crate) capacity: usize,
}

fn apply_cached_floor(
    result: Result<SignedPacket, DhtResolveError>,
    cached: Option<&SignedPacket>,
) -> Result<SignedPacket, DhtResolveError> {
    let Some(cached) = cached else {
        return result;
    };

    match result {
        Ok(packet) if cached.more_recent_than(&packet) => Err(DhtResolveError::NotFound),
        Ok(packet) => Ok(packet),
        Err(DhtResolveError::InvalidSignedPacket { seq })
            if u64::try_from(seq).is_ok_and(|seq| cached.timestamp().as_u64() >= seq) =>
        {
            Err(DhtResolveError::NotFound)
        }
        Err(error) => Err(error),
    }
}

fn decrement(timestamp: Timestamp) -> Option<Timestamp> {
    timestamp.as_u64().checked_sub(1).map(Timestamp::from)
}

#[cfg(test)]
mod tests {
    use super::{apply_cached_floor, decrement};
    use crate::error::Error;
    use axum::response::IntoResponse;
    use http::StatusCode;
    use pkarr::{dht::ResolveError, Keypair, SignedPacket, Timestamp};

    #[test]
    fn zero_timestamp_has_no_dht_lower_bound() {
        assert_eq!(decrement(Timestamp::from(0)), None);
        assert_eq!(decrement(Timestamp::from(1)), Some(Timestamp::from(0)));
    }

    #[test]
    fn cached_packet_suppresses_invalid_seq_that_is_not_newer() {
        let cached = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .sign(&Keypair::random())
            .unwrap();

        let response = Error::from(
            apply_cached_floor(
                Err(ResolveError::InvalidSignedPacket { seq: 10 }),
                Some(&cached),
            )
            .unwrap_err(),
        )
        .into_response();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert!(!response
            .headers()
            .contains_key(pkarr::PKARR_INVALID_SIGNED_PACKET_SEQ));
    }

    #[test]
    fn newer_invalid_seq_is_reported() {
        let cached = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .sign(&Keypair::random())
            .unwrap();

        let response = Error::from(
            apply_cached_floor(
                Err(ResolveError::InvalidSignedPacket { seq: 11 }),
                Some(&cached),
            )
            .unwrap_err(),
        )
        .into_response();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert!(response
            .headers()
            .contains_key(pkarr::PKARR_INVALID_SIGNED_PACKET_SEQ));
    }

    #[test]
    fn cached_packet_suppresses_older_packet_with_same_timestamp() {
        let keypair = Keypair::random();
        let a = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .txt("key".try_into().unwrap(), "a".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();
        let b = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .txt("key".try_into().unwrap(), "b".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();
        let (cached, older) = if a.more_recent_than(&b) {
            (a, b)
        } else {
            (b, a)
        };

        let result = apply_cached_floor(Ok(older), Some(&cached));

        assert!(matches!(result, Err(ResolveError::NotFound)));
    }

    #[test]
    fn invalid_seq_without_cached_floor_is_reported() {
        let response = Error::from(
            apply_cached_floor(Err(ResolveError::InvalidSignedPacket { seq: 10 }), None)
                .unwrap_err(),
        )
        .into_response();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert!(response
            .headers()
            .contains_key(pkarr::PKARR_INVALID_SIGNED_PACKET_SEQ));
    }
}
