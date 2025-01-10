//! Pkarr client for publishing and resolving [SignedPacket]s over [mainline] and/or [Relays](https://pkarr.org/relays).

use flume::{Receiver, Sender};
use mainline::{errors::PutError, rpc::DEFAULT_BOOTSTRAP_NODES};
use std::{
    net::{SocketAddr, SocketAddrV4, ToSocketAddrs},
    num::NonZeroUsize,
    thread,
};
use tracing::debug;
#[cfg(feature = "relays")]
use url::Url;

use crate::{
    Cache, CacheKey, InMemoryCache, DEFAULT_CACHE_SIZE, DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL,
    DEFAULT_RESOLVERS,
};
use crate::{PublicKey, SignedPacket};

mod actor_thread;
#[cfg(feature = "dht")]
mod dht;
#[cfg(feature = "relays")]
mod relays;

pub use actor_thread::Info;
use actor_thread::{actor_thread, ActorMessage};

// TODO: recognize when the cache is updated since `publish` was called,
// and return an error ... a CAS error maybe.

#[derive(Debug)]
/// [Client]'s Config
pub struct Config {
    pub dht_config: mainline::rpc::Config,
    /// A set of [resolver](https://pkarr.org/resolvers)s
    /// to be queried alongside the Dht routing table, to
    /// lower the latency on cold starts, and help if the
    /// Dht is missing values not't republished often enough.
    ///
    /// Defaults to [DEFAULT_RESOLVERS]
    pub resolvers: Option<Vec<SocketAddrV4>>,
    /// Defaults to [DEFAULT_CACHE_SIZE]
    pub cache_size: usize,
    /// Used in the `min` parameter in [SignedPacket::expires_in].
    ///
    /// Defaults to [DEFAULT_MINIMUM_TTL]
    pub minimum_ttl: u32,
    /// Used in the `max` parameter in [SignedPacket::expires_in].
    ///
    /// Defaults to [DEFAULT_MAXIMUM_TTL]
    pub maximum_ttl: u32,
    /// Custom [Cache] implementation, defaults to [InMemoryCache]
    pub cache: Option<Box<dyn Cache>>,

    /// Pkarr [Relays](https://pkarr.org/relays) Urls
    #[cfg(feature = "relays")]
    pub relays: Option<Vec<Url>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dht_config: mainline::rpc::Config::default(),
            cache_size: DEFAULT_CACHE_SIZE,
            resolvers: Some(resolvers_to_socket_addrs(&DEFAULT_RESOLVERS)),
            minimum_ttl: DEFAULT_MINIMUM_TTL,
            maximum_ttl: DEFAULT_MAXIMUM_TTL,
            cache: None,
            #[cfg(feature = "relays")]
            relays: Some(
                crate::DEFAULT_RELAYS
                    .iter()
                    .map(|s| {
                        Url::parse(s).expect("DEFAULT_RELAYS should be parsed to Url successfully.")
                    })
                    .collect(),
            ),
        }
    }
}

#[derive(Debug, Default)]
pub struct ClientBuilder(Config);

impl ClientBuilder {
    /// Similar to crates `no-default-features`, this method will remove the default
    /// [mainline::Config::bootstrap], [Config::resolvers], and [Config::relays],
    /// effectively disabling the use of both [mainline] and [Relays](https://pkarr.org/relays).
    ///
    /// You can [Self::use_mainline] or [Self::use_relays] to add both or either
    /// back with the default configurations for each.
    ///
    /// Or you can use [Self::relays] to use custom [Relays](https://pkarr.org/relays).
    ///
    /// Similarly you can use [Self::resolvers] and / or [Self::bootstrap] to use [mainline]
    /// with custom configurations.
    pub fn no_default_network(mut self) -> Self {
        #[cfg(feature = "dht")]
        {
            self.0.resolvers = None;
            self.0.dht_config.bootstrap = vec![];
        }
        #[cfg(feature = "relays")]
        {
            self.0.relays = None;
        }

        self
    }

    /// Reenable using [mainline] with [DEFAULT_RESOLVERS] and [DEFAULT_BOOTSTRAP_NODES].
    #[cfg(feature = "dht")]
    pub fn use_mainline(mut self) -> Self {
        self.0.resolvers = Some(resolvers_to_socket_addrs(&DEFAULT_RESOLVERS));
        self.0.dht_config.bootstrap = DEFAULT_BOOTSTRAP_NODES.map(|s| s.to_string()).to_vec();

        self
    }

    /// Convienent method to set the [mainline::Config::bootstrap].
    ///
    /// You can start a separate Dht network by setting this to an empty array.
    ///
    /// If you want to extend [Config::dht_config::bootstrap][mainline::Config::bootstrap] nodes with more nodes, you can
    /// use [Self::extra_bootstrap].
    #[cfg(feature = "dht")]
    pub fn bootstrap(mut self, bootstrap: &[String]) -> Self {
        self.0.dht_config.bootstrap = bootstrap.to_vec();

        self
    }

    #[cfg(feature = "dht")]
    /// Extend the [Config::dht_config::bootstrap][mainline::Config::bootstrap] nodes.
    ///
    /// If you want to set (override) the [Config::dht_config::bootsrtap][mainline::Config::bootstrap],
    /// use [Self::bootstrap]
    pub fn extra_resolvers(mut self, resolvers: Vec<String>) -> Self {
        let resolvers = resolvers_to_socket_addrs(&resolvers);

        if let Some(ref mut existing) = self.0.resolvers {
            existing.extend_from_slice(&resolvers);
        };

        self
    }

    /// Set custom set of [resolvers](Config::resolvers).
    ///
    /// You can disable using resolvers by passing `None`.
    ///
    /// If you want to extend the [Config::resolvers] with more nodes, you can
    /// use [Self::extra_resolvers].
    #[cfg(feature = "dht")]
    pub fn resolvers(mut self, resolvers: Option<Vec<String>>) -> Self {
        self.0.resolvers = resolvers.map(|resolvers| resolvers_to_socket_addrs(&resolvers));

        self
    }

    #[cfg(feature = "dht")]
    /// Extend the current [Config::resolvers] with extra resolvers.
    ///
    /// If you want to set (override) the [Config::resolvers], use [Self::resolvers]
    pub fn extra_bootstrap(mut self, bootstrap: &[String]) -> Self {
        for node in bootstrap {
            if !self.0.dht_config.bootstrap.contains(node) {
                self.0.dht_config.bootstrap.push(node.clone())
            }
        }

        self
    }

    #[cfg(feature = "relays")]
    /// Reenable using [Config::relays] with [crate::DEFAULT_RELAYS].
    pub fn use_relays(mut self) -> Self {
        self.0.relays = Some(
            crate::DEFAULT_RELAYS
                .iter()
                .map(|s| {
                    Url::parse(s).expect("DEFAULT_RELAYS should be parsed to Url successfully.")
                })
                .collect(),
        );

        self
    }

    /// Set custom set of [relays](Config::relays)
    #[cfg(feature = "relays")]
    pub fn relays(mut self, relays: Option<Vec<Url>>) -> Self {
        self.0.relays = relays;

        self
    }

    #[cfg(feature = "dht")]
    /// Extend the current [Config::relays] with extra relays.
    ///
    /// If you want to set (override) the [Config::relays], use [Self::relays]
    pub fn extra_relays(mut self, relays: Vec<Url>) -> Self {
        if let Some(ref mut existing) = self.0.relays {
            for relay in relays {
                if !existing.contains(&relay) {
                    existing.push(relay)
                }
            }
        }

        self
    }

    /// Set the [Config::cache_size].
    ///
    /// Controls the capacity of [Cache].
    ///
    /// If set to `0` cache will be disabled.
    pub fn cache_size(mut self, cache_size: usize) -> Self {
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

    /// Set [Config::dht_config]
    #[cfg(feature = "dht")]
    pub fn dht_config(mut self, config: mainline::rpc::Config) -> Self {
        self.0.dht_config = config;

        self
    }

    pub fn build(self) -> Result<Client, std::io::Error> {
        Client::new(self.0)
    }
}

#[derive(Clone, Debug)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [mainline] and/or
/// [Relays](https://pkarr.org/relays).
pub struct Client {
    sender: Sender<ActorMessage>,
    cache: Option<Box<dyn Cache>>,
}

impl Client {
    pub fn new(config: Config) -> Result<Client, std::io::Error> {
        let (sender, receiver) = flume::bounded(32);

        let cache = if config.cache_size == 0 {
            None
        } else {
            Some(
                config.cache.clone().unwrap_or(Box::new(InMemoryCache::new(
                    NonZeroUsize::new(config.cache_size)
                        .expect("if cache size is zero cache should be disabled."),
                ))),
            )
        };

        let cache_clone = cache.clone();

        let client = Client { sender, cache };

        debug!(?config, "Starting Client main loop..");

        thread::Builder::new()
            .name("Pkarr Dht actor thread".to_string())
            .spawn(move || actor_thread(receiver, cache_clone, config))?;

        let (tx, rx) = flume::bounded(1);

        client
            .sender
            .send(ActorMessage::Check(tx))
            .expect("actor thread unexpectedly shutdown");

        rx.recv().expect("infallible")?;

        Ok(client)
    }

    /// Returns a builder to edit config before creating Client.
    ///
    /// You can use [ClientBuilder::no_default_network] to start from a clean slate and
    /// decide which networks to use.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    // === Getters ===

    /// Returns [Info] about the running session from the actor thread.
    pub fn info(&self) -> Result<Info, ClientWasShutdown> {
        let (tx, rx) = flume::bounded(1);

        self.sender
            .send(ActorMessage::Info(tx))
            .map_err(|_| ClientWasShutdown)?;

        rx.recv().map_err(|_| ClientWasShutdown)
    }

    /// Returns a reference to the internal cache.
    pub fn cache(&self) -> Option<&dyn Cache> {
        self.cache.as_deref()
    }

    // === Public Methods ===

    /// Publishes a [SignedPacket] to the Dht.
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<(), PublishError> {
        self.publish_inner(signed_packet)?
            .recv_async()
            .await
            .expect("Query was dropped before sending a response, please open an issue.")
            .map_err(|_error| {
                // TODO: do better.
                PublishError::PublishInflight
            })?;

        Ok(())
    }

    /// Publishes a [SignedPacket] to the Dht.
    pub fn publish_sync(&self, signed_packet: &SignedPacket) -> Result<(), PublishError> {
        self.publish_inner(signed_packet)?
            .recv()
            .expect("Query was dropped before sending a response, please open an issue.")
            .map_err(|_error| {
                // TODO: do better.
                PublishError::PublishInflight
            })?;

        Ok(())
    }

    /// Returns a [SignedPacket] from the cache even if it is expired.
    /// If there is no packet in the cache, or if the cached packet is expired,
    /// it will make a DHT query in a background query and caches any more recent packets it receieves.
    ///
    /// If you want to have more control, you can call [Self::resolve_rx] directly,
    /// and then [iterate](flume::Receiver::recv) over or [stream](flume::Receiver::recv_async)
    /// incoming [SignedPacket]s until your lookup criteria is satisfied.
    pub async fn resolve(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        Ok(self.resolve_rx(public_key)?.recv_async().await.ok())
    }

    /// Returns a [SignedPacket] from the cache even if it is expired.
    /// If there is no packet in the cache, or if the cached packet is expired,
    /// it will make a DHT query in a background query and caches any more recent packets it receieves.
    ///
    /// If you want to have more control, you can call [Self::resolve_rx] directly,
    /// and then [iterate](flume::Receiver::recv) over or [stream](flume::Receiver::recv_async)
    /// incoming [SignedPacket]s until your lookup criteria is satisfied.
    pub fn resolve_sync(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        Ok(self.resolve_rx(public_key)?.recv().ok())
    }

    /// Returns a `flume::Receiver<SignedPacket>` that allows [iterating](flume::Receiver::recv) over or
    /// [streaming](flume::Receiver::recv_async) incoming [SignedPacket]s, in case you need more control over your
    /// caching strategy and when resolution should terminate, as well as filtering [SignedPacket]s according to a custom criteria.
    pub fn resolve_rx(
        &self,
        public_key: &PublicKey,
    ) -> Result<Receiver<SignedPacket>, ClientWasShutdown> {
        let (tx, rx) = flume::bounded::<SignedPacket>(1);

        self.sender
            .send(ActorMessage::Resolve(public_key.clone(), tx.clone()))
            .map_err(|_| ClientWasShutdown)?;

        Ok(rx)
    }

    /// Shutdown the actor thread loop.
    pub async fn shutdown(&mut self) {
        let (sender, receiver) = flume::bounded(1);

        let _ = self.sender.send(ActorMessage::Shutdown(sender));
        let _ = receiver.recv_async().await;
    }

    /// Shutdown the actor thread loop.
    pub fn shutdown_sync(&self) {
        let (sender, receiver) = flume::bounded(1);

        let _ = self.sender.send(ActorMessage::Shutdown(sender));
        let _ = receiver.recv();
    }

    // === Private Methods ===

    pub(crate) fn publish_inner(
        &self,
        signed_packet: &SignedPacket,
    ) -> Result<Receiver<Result<(), ()>>, PublishError> {
        let cache_key = CacheKey::from(signed_packet.public_key());

        if let Some(cache) = &self.cache {
            if let Some(current) = cache.get(&cache_key) {
                if current.timestamp() > signed_packet.timestamp() {
                    return Err(PublishError::NotMostRecent);
                }
            };

            cache.put(&cache_key, signed_packet);
        }

        let (sender, receiver) = flume::bounded::<Result<(), ()>>(1);

        self.sender
            .send(ActorMessage::Publish(signed_packet.clone(), sender))
            .map_err(|_| PublishError::ClientWasShutdown)?;

        Ok(receiver)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ClientWasShutdown;

impl std::error::Error for ClientWasShutdown {}

impl std::fmt::Display for ClientWasShutdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pkarr Client was shutdown")
    }
}

#[derive(thiserror::Error, Debug)]
/// Errors occuring during publishing a [SignedPacket]
pub enum PublishError {
    #[error("Found a more recent SignedPacket in the client's cache")]
    /// Found a more recent SignedPacket in the client's cache
    NotMostRecent,

    #[error("Pkarr Client was shutdown")]
    ClientWasShutdown,

    #[error("Publish query is already inflight for the same public_key")]
    /// [crate::Client::publish] is already inflight to the same public_key
    PublishInflight,

    #[error(transparent)]
    MainlinePutError(#[from] PutError),
}

pub fn resolvers_to_socket_addrs<T: ToSocketAddrs>(resolvers: &[T]) -> Vec<SocketAddrV4> {
    resolvers
        .iter()
        .flat_map(|resolver| {
            resolver.to_socket_addrs().map(|iter| {
                iter.filter_map(|a| match a {
                    SocketAddr::V4(a) => Some(a),
                    _ => None,
                })
            })
        })
        .flatten()
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    //! Combined client tests

    use std::{thread, time::Duration};

    use pkarr_relay::Relay;

    use super::super::*;
    use crate::{Keypair, SignedPacket};

    #[tokio::test]
    async fn publish_resolve() {
        let relay = Relay::start_test().await.unwrap();

        let a = Client::builder()
            .bootstrap(&[relay.resolver_address().to_string()])
            .relays(Some(vec![relay.local_url()]))
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = Client::builder()
            .bootstrap(&[relay.resolver_address().to_string()])
            .relays(Some(vec![relay.local_url()]))
            .build()
            .unwrap();

        let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }

    #[tokio::test]
    async fn thread_safe() {
        let relay = Relay::start_test().await.unwrap();

        let a = Client::builder()
            .bootstrap(&[relay.resolver_address().to_string()])
            .relays(Some(vec![relay.local_url()]))
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = Client::builder()
            .bootstrap(&[relay.resolver_address().to_string()])
            .relays(Some(vec![relay.local_url()]))
            .build()
            .unwrap();

        tokio::spawn(async move {
            let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
            assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

            let from_cache = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
            assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
            assert_eq!(from_cache.last_seen(), resolved.last_seen());
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn return_expired_packet_fallback() {
        let relay = Relay::start_test().await.unwrap();

        let client = Client::builder()
            .bootstrap(&[relay.resolver_address().to_string()])
            .relays(Some(vec![relay.local_url()]))
            .dht_config(mainline::Config {
                request_timeout: Duration::from_millis(10),
                ..Default::default()
            })
            // Everything is expired
            .maximum_ttl(0)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();

        client
            .cache()
            .unwrap()
            .put(&keypair.public_key().into(), &signed_packet);

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert_eq!(resolved, Some(signed_packet));
    }

    #[tokio::test]
    async fn ttl_0_test() {
        let relay = Relay::start_test().await.unwrap();

        let client = Client::builder()
            .bootstrap(&[relay.resolver_address().to_string()])
            .relays(Some(vec![relay.local_url()]))
            .maximum_ttl(0)
            .build()
            .unwrap();

        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();

        client.publish(&signed_packet).await.unwrap();

        // First Call
        let resolved = client
            .resolve(&signed_packet.public_key())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved.encoded_packet(), signed_packet.encoded_packet());

        thread::sleep(Duration::from_millis(10));

        let second = client
            .resolve(&signed_packet.public_key())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second.encoded_packet(), signed_packet.encoded_packet());
    }
}
