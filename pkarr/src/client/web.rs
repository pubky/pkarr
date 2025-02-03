//! Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).

use reqwest::header::HeaderValue;
use reqwest::{header, Response, StatusCode};
use std::fmt::{self, Debug, Display, Formatter};

use std::num::NonZeroUsize;

use url::Url;

use futures::future::select_ok;

use crate::{
    Cache, InMemoryCache, PublicKey, SignedPacket, DEFAULT_CACHE_SIZE, DEFAULT_MAXIMUM_TTL,
    DEFAULT_MINIMUM_TTL, DEFAULT_RELAYS,
};

#[derive(Debug, Clone)]
/// [Client]'s Config
pub struct Config {
    // Relays base URLs
    pub relays: Vec<Url>,
    /// Defaults to [DEFAULT_CACHE_SIZE]
    pub cache_size: NonZeroUsize,
    /// Used in the `min` parameter in [SignedPacket::expires_in].
    ///
    /// Defaults to [DEFAULT_MINIMUM_TTL]
    pub minimum_ttl: u32,
    /// Used in the `max` parametere in [SignedPacket::expires_in].
    ///
    /// Defaults to [DEFAULT_MAXIMUM_TTL]
    pub maximum_ttl: u32,
    /// Custom [reqwest::Client]
    pub http_client: reqwest::Client,
    /// Custom [Cache] implementation, defaults to [InMemoryCache]
    pub cache: Option<Box<dyn Cache>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            relays: DEFAULT_RELAYS
                .map(|s| Url::parse(s).expect("DEFAULT_RELAYS should be parsed correctly"))
                .to_vec(),

            cache_size: NonZeroUsize::new(DEFAULT_CACHE_SIZE)
                .expect("NonZeroUsize from DEFAULT_CACHE_SIZE"),
            minimum_ttl: DEFAULT_MINIMUM_TTL,
            maximum_ttl: DEFAULT_MAXIMUM_TTL,
            http_client: reqwest::Client::new(),
            cache: None,
        }
    }
}

#[derive(Debug, Default)]
pub struct ClientBuilder(Config);

impl ClientBuilder {
    /// Set the relays to publish and resolve [SignedPacket]s to and from.
    pub fn relays(mut self, relays: Vec<Url>) -> Self {
        self.0.relays = relays;

        self
    }

    /// Set the [Config::cache_size].
    ///
    /// Controls the capacity of [Cache].
    pub fn cache_size(mut self, cache_size: NonZeroUsize) -> Self {
        self.0.cache_size = cache_size;

        self
    }

    /// Set the [Config::minimum_ttl] value.
    ///
    /// Limits how soon a [SignedPacket] is considered expired.
    pub fn minimum_ttl(mut self, ttl: u32) -> Self {
        self.0.minimum_ttl = ttl;

        self
    }

    /// Set the [Config::maximum_ttl] value.
    ///
    /// Limits how long it takes before a [SignedPacket] is considered expired.
    pub fn maximum_ttl(mut self, ttl: u32) -> Self {
        self.0.maximum_ttl = ttl;

        self
    }

    /// Set a custom implementation of [Cache].
    pub fn cache(mut self, cache: Box<dyn Cache>) -> Self {
        self.0.cache = Some(cache);

        self
    }

    pub fn build(self) -> Result<Client, EmptyListOfRelays> {
        Client::new(self.0)
    }
}

#[derive(Debug, Clone)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).
pub struct Client {
    http_client: reqwest::Client,
    relays: Vec<Url>,
    cache: Box<dyn Cache>,
    minimum_ttl: u32,
    maximum_ttl: u32,
}

impl Default for Client {
    fn default() -> Self {
        Self::builder()
            .build()
            .expect("default config should be infallible")
    }
}

impl Client {
    pub fn new(config: Config) -> Result<Self, EmptyListOfRelays> {
        if config.relays.is_empty() {
            return Err(EmptyListOfRelays);
        }

        let cache = config
            .cache
            .clone()
            .unwrap_or(Box::new(InMemoryCache::new(config.cache_size)));

        Ok(Self {
            http_client: config.http_client,
            relays: config.relays,
            cache,
            minimum_ttl: config.minimum_ttl.min(config.maximum_ttl),
            maximum_ttl: config.maximum_ttl.max(config.minimum_ttl),
        })
    }

    /// Returns a builder to edit config before creating Client.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// Returns a reference to the internal cache.
    pub fn cache(&self) -> &dyn Cache {
        self.cache.as_ref()
    }

    /// Publishes a [SignedPacket] to this client's relays.
    ///
    /// Return the first successful completion, or the last failure.
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<(), PublishError> {
        let public_key = signed_packet.public_key();

        if let Some(current) = self.cache.get(&public_key.as_ref().into()) {
            if current.timestamp() > signed_packet.timestamp() {
                return Err(PublishError::NotMostRecent);
            }
        };

        self.cache.put(&public_key.as_ref().into(), signed_packet);

        self.race_publish(signed_packet)
            .await
            .map_err(|_| PublishError::AllRequestsFailed)
    }

    /// Returns a [SignedPacket] from the cache even if it is expired.
    /// If there is no packet in the cache, or if the cached packet is expired,
    /// it will make GET requests to all client's relays in the background,
    /// and caches any more recent packets it receieves.
    ///
    /// If you want to have more control, you can call [Self::resolve_rx] directly,
    /// and then [iterate](flume::Receiver::recv) over or [stream](flume::Receiver::recv_async)
    /// incoming [SignedPacket]s until your lookup criteria is satisfied.
    pub async fn resolve(&self, public_key: &PublicKey) -> Option<SignedPacket> {
        Ok(self.resolve_rx(public_key)?.recv_async().await.ok())
    }

    /// Returns a [flume::Receiver<SignedPacket>] that allows [iterating](flume::Receiver::recv) over or
    /// [streaming](flume::Receiver::recv_async) incoming [SignedPacket]s, in case you need more control over your
    /// caching strategy and when resolution should terminate, as well as filtering [SignedPacket]s according to a custom criteria.
    pub async fn resolve_rx(&self, public_key: &PublicKey) -> flume::Receiver<SignedPacket> {
        let target = MutableItem::target_from_key(public_key.as_bytes(), None);

        let cached_packet = self.cache.get(target.as_bytes());

        let (tx, rx) = flume::bounded::<SignedPacket>(1);

        let as_ref = cached_packet.as_ref();

        // Should query?
        if as_ref
            .as_ref()
            .map(|c| c.is_expired(self.minimum_ttl, self.maximum_ttl))
            .unwrap_or(true)
        {
            log::debug!(
                "Found expired cached packet, querying relays to hydrate our cache for later. public_key: {}",
                public_key,
            );

            for relay in self.relays {
                wasm_bindgen_futures::spawn_local(async move {
                    let response = self.resolve_from_relay(public_key, relay);

                    // If the receiver was dropped.. no harm.
                    let _ = tx.send(this.race_resolve(&pubky, None).await);
                });
            }
        }

        if let Some(cached_packet) = cached_packet {
            log::debug!("responding with cached packet even if expired. {public_key}",);

            // If the receiver was dropped.. no harm.
            let _ = tx.send(cached_packet);
        }

        Ok(rx)
    }

    // === Private Methods ===

    async fn race_publish(&self, signed_packet: &SignedPacket) -> Result<(), ()> {
        let futures = self.relays.iter().map(|relay| {
            let signed_packet = signed_packet.clone();
            let this = self.clone();

            Box::pin(async move { this.publish_to_relay(relay, signed_packet).await })
        });

        match select_ok(futures).await {
            Ok((_, _)) => Ok(()),
            Err(_) => Err(()),
        }
    }

    async fn publish_to_relay(
        &self,
        relay: &Url,
        signed_packet: SignedPacket,
    ) -> Result<Response, reqwest::Error> {
        let url = format_url(relay, &signed_packet.public_key());

        self.http_client
            .put(url.clone())
            .body(signed_packet.as_relay_payload().to_vec())
            .send()
            .await
    }

    async fn resolve_from_relay(
        &self,
        relay: &Url,
        public_key: PublicKey,
        cached_packet: Option<SignedPacket>,
    ) -> Result<Option<SignedPacket>, reqwest::Error> {
        let url = format_url(relay, &public_key);

        let mut request = self.http_client.get(url.clone());

        if let Some(httpdate) = cached_packet
            .as_ref()
            .map(|c| c.timestamp().format_http_date())
        {
            request = request.header(
                header::IF_MODIFIED_SINCE,
                HeaderValue::from_str(httpdate.as_str())
                    .expect("httpdate to be valid header value"),
            );
        }

        let response = request.send().await?;

        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }

        let response = response.error_for_status()?;

        if response.content_length().unwrap_or_default() > SignedPacket::MAX_BYTES {
            log::debug!("Response too large {url}");

            return Ok(None);
        }

        let payload = response.bytes().await?;

        match SignedPacket::from_relay_payload(&public_key, &payload) {
            Ok(signed_packet) => Ok(choose_most_recent(signed_packet, cached_packet)),
            Err(error) => {
                log::debug!("Invalid signed_packet {url}:{error}");

                Ok(None)
            }
        }
    }
}

fn format_url(url: &Url, public_key: &PublicKey) -> Url {
    let mut url = url.clone();

    let mut segments = url
        .path_segments_mut()
        .expect("Relay url cannot be base, is it http(s)?");

    segments.push(&public_key.to_string());

    drop(segments);

    url
}

fn choose_most_recent(
    signed_packet: SignedPacket,
    cached_packet: Option<SignedPacket>,
) -> Option<SignedPacket> {
    if let Some(ref cached) = cached_packet {
        if signed_packet.more_recent_than(cached) {
            log::debug!(
                "Received more recent packet than in cache for public_key: {}",
                signed_packet.public_key()
            );
            Some(signed_packet)
        } else {
            None
        }
    } else {
        log::debug!(
            "Received new packet after cache miss for public_key: {}",
            signed_packet.public_key()
        );
        Some(signed_packet)
    }
}

#[derive(thiserror::Error, Debug)]
/// Errors during publishing a [SignedPacket] to a list of relays
pub enum PublishError {
    #[error("SignedPacket's timestamp is not the most recent")]
    /// Failed to publish because there is a more recent packet.
    NotMostRecent,

    #[error("All PUT requests to Pkarr relays failed, check your console and/or Network tab.")]
    /// All relays responded with non-2xx status code, or something wrong with the transport.
    AllRequestsFailed,
}

#[derive(Debug)]
pub struct AllGetRequestsFailed;

impl std::error::Error for AllGetRequestsFailed {}

impl Display for AllGetRequestsFailed {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "All GET requests to Pkarr relays failed, check your console and/or Network tab."
        )
    }
}

#[derive(Debug)]
pub struct EmptyListOfRelays;

impl std::error::Error for EmptyListOfRelays {}

impl Display for EmptyListOfRelays {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Can not create a Pkarr relay Client with an empty list of relays"
        )
    }
}

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::*;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    use super::*;

    use crate::Keypair;

    #[wasm_bindgen_test]
    async fn publish_resolve() {
        console_log::init_with_level(log::Level::Debug).unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        let relays = vec![Url::parse("http://localhost:15411").unwrap()];

        let a = Client::builder().relays(relays.clone()).build().unwrap();
        let b = Client::builder().relays(relays).build().unwrap();

        a.publish(&signed_packet).await.unwrap();

        let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }

    #[wasm_bindgen_test]
    async fn not_found() {
        let keypair = Keypair::random();

        let relays = vec![Url::parse("http://localhost:15411").unwrap()];

        let client = Client::builder().relays(relays).build().unwrap();

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert!(resolved.is_none());
    }

    #[wasm_bindgen_test]
    async fn return_expired_packet_fallback() {
        let keypair = Keypair::random();

        let relays = vec![Url::parse("http://localhost:15411").unwrap()];

        let client = Client::builder()
            .relays(relays)
            .maximum_ttl(0)
            .build()
            .unwrap();

        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();

        client
            .cache()
            .put(&keypair.public_key().into(), &signed_packet);

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert_eq!(resolved, Some(signed_packet));
    }
}
