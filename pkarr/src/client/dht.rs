//! Pkarr client for publishing and resolving [SignedPacket]s over [mainline].

use flume::{Receiver, Sender};
use mainline::{
    errors::PutError,
    rpc::{messages, Response, Rpc},
    Id, MutableItem,
};
use std::{
    collections::HashMap,
    net::{SocketAddr, SocketAddrV4, ToSocketAddrs},
    num::NonZeroUsize,
    thread,
};
use tracing::debug;

use crate::{
    Cache, InMemoryCache, DEFAULT_CACHE_SIZE, DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL,
    DEFAULT_RESOLVERS,
};
use crate::{PublicKey, SignedPacket};

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
    pub resolvers: Option<Box<[SocketAddrV4]>>,
    /// Defaults to [DEFAULT_CACHE_SIZE]
    pub cache_size: NonZeroUsize,
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dht_config: mainline::rpc::Config::default(),
            cache_size: NonZeroUsize::new(DEFAULT_CACHE_SIZE)
                .expect("NonZeroUsize from DEFAULT_CACHE_SIZE"),
            resolvers: Some(resolvres_to_socket_addrs(&DEFAULT_RESOLVERS)),
            minimum_ttl: DEFAULT_MINIMUM_TTL,
            maximum_ttl: DEFAULT_MAXIMUM_TTL,
            cache: None,
        }
    }
}

#[derive(Debug, Default)]
pub struct ClientBuilder(Config);

impl ClientBuilder {
    /// Set custom set of [resolvers](Config::resolvers).
    pub fn resolvers(mut self, resolvers: Option<Vec<String>>) -> Self {
        self.0.resolvers = resolvers.map(|resolvers| resolvres_to_socket_addrs(&resolvers));

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

    /// Set [Config::dht_config]
    pub fn dht_config(mut self, config: mainline::rpc::Config) -> Self {
        self.0.dht_config = config;

        self
    }

    /// Convienent method to set the [mainline::Config::bootstrap].
    pub fn bootstrap(mut self, bootstrap: &[String]) -> Self {
        self.0.dht_config.bootstrap = bootstrap.to_vec();

        self
    }

    pub fn build(self) -> Result<Client, std::io::Error> {
        Client::new(self.0)
    }
}

#[derive(Clone, Debug)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [mainline].
pub struct Client {
    sender: Sender<ActorMessage>,
    cache: Box<dyn Cache>,
    minimum_ttl: u32,
    maximum_ttl: u32,
}

impl Client {
    pub fn new(config: Config) -> Result<Client, std::io::Error> {
        let (sender, receiver) = flume::bounded(32);

        let cache = config
            .cache
            .clone()
            .unwrap_or(Box::new(InMemoryCache::new(config.cache_size)));
        let cache_clone = cache.clone();

        let client = Client {
            sender,
            cache,
            minimum_ttl: config.minimum_ttl.min(config.maximum_ttl),
            maximum_ttl: config.maximum_ttl.max(config.minimum_ttl),
        };

        debug!(?config, "Starting Client main loop..");

        thread::Builder::new()
            .name("Pkarr Dht actor thread".to_string())
            .spawn(move || run(cache_clone, config, receiver))?;

        let (tx, rx) = flume::bounded(1);

        client
            .sender
            .send(ActorMessage::Check(tx))
            .expect("actor thread unexpectedly shutdown");

        rx.recv().expect("infallible")?;

        Ok(client)
    }

    /// Returns a builder to edit config before creating Client.
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
    pub fn cache(&self) -> &dyn Cache {
        self.cache.as_ref()
    }

    // === Public Methods ===

    /// Publishes a [SignedPacket] to the Dht.
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<(), PublishError> {
        self.publish_inner(signed_packet)?
            .recv_async()
            .await
            .expect("Query was dropped before sending a response, please open an issue.")
            .map_err(|error| match error {
                PutError::PutQueryIsInflight(_) => PublishError::PublishInflight,
                _ => PublishError::MainlinePutError(error),
            })?;

        Ok(())
    }

    /// Returns a [SignedPacket] from the cache even if it is expired.
    /// If there is no packet in the cache, or if the cached packet is expired,
    /// it will make a DHT in the background query and caches any more recent packets it receieves.
    ///
    /// If you want to have more control, you can call [Self::resolve_rx] directly,
    /// and then [iterate](flume::Receiver::recv) over or [stream](flume::Receiver::recv_async)
    /// incoming [SignedPacket]s until your lookup criteria is satisfied.
    ///
    /// # Errors
    /// - Returns a [ClientWasShutdown] if [Client::shutdown] was called, or
    ///   the loop in the actor thread is stopped for any reason (like thread panic).
    pub async fn resolve(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        Ok(self.resolve_rx(public_key)?.recv_async().await.ok())
    }

    /// Shutdown the actor thread loop.
    pub async fn shutdown(&mut self) {
        let (sender, receiver) = flume::bounded(1);

        let _ = self.sender.send(ActorMessage::Shutdown(sender));
        let _ = receiver.recv_async().await;
    }

    // === Sync ===

    /// Publishes a [SignedPacket] to the Dht.
    pub fn publish_sync(&self, signed_packet: &SignedPacket) -> Result<(), PublishError> {
        self.publish_inner(signed_packet)?
            .recv()
            .expect("Query was dropped before sending a response, please open an issue.")
            .map_err(|error| match error {
                PutError::PutQueryIsInflight(_) => PublishError::PublishInflight,
                _ => PublishError::MainlinePutError(error),
            })?;

        Ok(())
    }

    /// Returns a [SignedPacket] from the cache even if it is expired.
    /// If there is no packet in the cache, or if the cached packet is expired,
    /// it will make a DHT in the background query and caches any more recent packets it receieves.
    ///
    /// If you want to have more control, you can call [Self::resolve_rx] directly,
    /// and then [iterate](flume::Receiver::recv) over or [stream](flume::Receiver::recv_async)
    /// incoming [SignedPacket]s until your lookup criteria is satisfied.
    ///
    /// # Errors
    /// - Returns a [ClientWasShutdown] if [Client::shutdown] was called, or
    ///   the loop in the actor thread is stopped for any reason (like thread panic).
    pub fn resolve_sync(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        Ok(self.resolve_rx(public_key)?.recv().ok())
    }

    /// Returns a [flume::Receiver<SignedPacket>] that allows [iterating](flume::Receiver::recv) over or
    /// [streaming](flume::Receiver::recv_async) incoming [SignedPacket]s, in case you need more control over your
    /// caching strategy and when resolution should terminate, as well as filtering [SignedPacket]s according to a custom criteria.
    ///
    /// # Errors
    /// - Returns a [ClientWasShutdown] if [Client::shutdown] was called, or
    ///   the loop in the actor thread is stopped for any reason (like thread panic).
    pub fn resolve_rx(
        &self,
        public_key: &PublicKey,
    ) -> Result<Receiver<SignedPacket>, ClientWasShutdown> {
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
            debug!(
                ?public_key,
                "querying the DHT to hydrate our cache for later."
            );

            self.sender
                .send(ActorMessage::Resolve(
                    target,
                    tx.clone(),
                    // Sending the `timestamp` of the known cache, help save some bandwith,
                    // since remote nodes will not send the encoded packet if they don't know
                    // any more recent versions.
                    // most_recent_known_timestamp,
                    as_ref.map(|cached| cached.timestamp().as_u64()),
                ))
                .map_err(|_| ClientWasShutdown)?;
        }

        if let Some(cached_packet) = cached_packet {
            debug!(
                public_key = ?cached_packet.public_key(),
                "responding with cached packet even if expired"
            );

            // If the receiver was dropped.. no harm.
            let _ = tx.send(cached_packet);
        }

        Ok(rx)
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
    ) -> Result<Receiver<Result<Id, PutError>>, PublishError> {
        let mutable_item: MutableItem = (signed_packet).into();

        if let Some(current) = self.cache.get(mutable_item.target().as_bytes()) {
            if current.timestamp() > signed_packet.timestamp() {
                return Err(PublishError::NotMostRecent);
            }
        };

        self.cache
            .put(mutable_item.target().as_bytes(), signed_packet);

        let (sender, receiver) = flume::bounded::<Result<Id, PutError>>(1);

        self.sender
            .send(ActorMessage::Publish(mutable_item, sender))
            .map_err(|_| PublishError::ClientWasShutdown)?;

        Ok(receiver)
    }
}

#[derive(Debug)]
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

fn run(cache: Box<dyn Cache>, config: Config, receiver: Receiver<ActorMessage>) {
    match Rpc::new(config.dht_config) {
        Ok(mut rpc) => actor_thread(&mut rpc, cache, receiver, config.resolvers),
        Err(err) => {
            if let Ok(ActorMessage::Check(sender)) = receiver.try_recv() {
                let _ = sender.send(Err(err));
            }
        }
    }
}

fn actor_thread(
    rpc: &mut Rpc,
    cache: Box<dyn Cache>,
    receiver: Receiver<ActorMessage>,
    resolvers: Option<Box<[SocketAddrV4]>>,
) {
    let mut resolve_senders: HashMap<Id, Vec<Sender<SignedPacket>>> = HashMap::new();
    let mut publish_senders: HashMap<Id, Sender<Result<Id, PutError>>> = HashMap::new();

    loop {
        // === Receive actor messages ===
        if let Ok(actor_message) = receiver.try_recv() {
            match actor_message {
                ActorMessage::Shutdown(sender) => {
                    drop(receiver);
                    let _ = sender.send(());
                    break;
                }
                ActorMessage::Publish(mutable_item, sender) => {
                    let target = *mutable_item.target();

                    if let Err(put_error) = rpc.put(messages::PutRequestSpecific::PutMutable(
                        mutable_item.into(),
                    )) {
                        let _ = sender.send(Err(put_error));
                    } else {
                        publish_senders.insert(target, sender);
                    };
                }
                ActorMessage::Resolve(target, sender, most_recent_known_timestamp) => {
                    if let Some(senders) = resolve_senders.get_mut(&target) {
                        senders.push(sender);
                    } else {
                        resolve_senders.insert(target, vec![sender]);
                    };

                    if let Some(responses) = rpc.get(
                        messages::RequestTypeSpecific::GetValue(
                            messages::GetValueRequestArguments {
                                target,
                                seq: most_recent_known_timestamp.map(|t| t as i64),
                                salt: None,
                            },
                        ),
                        resolvers.clone(),
                    ) {
                        for response in responses {
                            if let Response::Mutable(mutable_item) = response {
                                if let Ok(signed_packet) = SignedPacket::try_from(mutable_item) {
                                    if let Some(senders) = resolve_senders.get(&target) {
                                        for sender in senders {
                                            let _ = sender.send(signed_packet.clone());
                                        }
                                    }
                                }
                            }
                        }
                    };
                }
                ActorMessage::Info(sender) => {
                    let _ = sender.send(Info {
                        dht_info: rpc.info(),
                    });
                }
                ActorMessage::Check(sender) => {
                    let _ = sender.send(Ok(()));
                }
            }
        }

        // === Dht Tick ===

        let report = rpc.tick();

        // === Receive and handle incoming mutable item from the DHT ===

        if let Some((target, Response::Mutable(mutable_item))) = &report.query_response {
            if let Ok(signed_packet) = &SignedPacket::try_from(mutable_item) {
                let new_packet = if let Some(ref cached) = cache.get_read_only(target.as_bytes()) {
                    if signed_packet.more_recent_than(cached) {
                        debug!(?target, "Received more recent packet than in cache");

                        Some(signed_packet)
                    } else {
                        None
                    }
                } else {
                    debug!(?target, "Received new packet after cache miss");
                    Some(signed_packet)
                };

                if let Some(packet) = new_packet {
                    cache.put(target.as_bytes(), packet);

                    if let Some(senders) = resolve_senders.get(target) {
                        for sender in senders {
                            let _ = sender.send(packet.clone());
                        }
                    }
                }
            };
        }

        // TODO: Handle relay messages before removing the senders.

        // === Drop senders to done queries ===
        for id in &report.done_get_queries {
            resolve_senders.remove(id);
        }

        for (id, error) in &report.done_put_queries {
            if let Some(sender) = publish_senders.remove(id) {
                let _ = sender.send(if let Some(error) = error.to_owned() {
                    Err(error)
                } else {
                    Ok(*id)
                });
            };
        }
    }

    debug!("Client main loop terminated");
}

enum ActorMessage {
    Publish(MutableItem, Sender<Result<Id, PutError>>),
    Resolve(Id, Sender<SignedPacket>, Option<u64>),
    Shutdown(Sender<()>),
    Info(Sender<Info>),
    Check(Sender<Result<(), std::io::Error>>),
}

pub struct Info {
    dht_info: mainline::rpc::Info,
}

// TODO: add more infor like Mainline
impl Info {
    //
    pub fn dht_info(&self) -> &mainline::rpc::Info {
        &self.dht_info
    }
}

pub fn resolvres_to_socket_addrs<T: ToSocketAddrs>(resolvers: &[T]) -> Box<[SocketAddrV4]> {
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
        .into()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use mainline::Testnet;

    use super::*;
    use crate::{Keypair, SignedPacket};

    #[test]
    fn shutdown_sync() {
        let testnet = Testnet::new(3).unwrap();

        let client = Client::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        client.shutdown_sync();

        assert!(client.info().is_err());
    }

    #[test]
    fn publish_resolve_sync() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish_sync(&signed_packet).unwrap();

        let b = Client::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let resolved = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }

    #[test]
    fn thread_safe_sync() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish_sync(&signed_packet).unwrap();

        let b = Client::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        thread::spawn(move || {
            let resolved = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
            assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

            let from_cache = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
            assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
            assert_eq!(from_cache.last_seen(), resolved.last_seen());
        })
        .join()
        .unwrap();
    }

    #[tokio::test]
    async fn shutdown() {
        let testnet = Testnet::new(3).unwrap();

        let mut a = Client::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        a.shutdown().await;

        assert!(a.info().is_err());
    }

    #[tokio::test]
    async fn publish_resolve() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = Client::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let resolved = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }

    #[tokio::test]
    async fn thread_safe() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = Client::builder()
            .bootstrap(&testnet.bootstrap)
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
        let testnet = Testnet::new(10).unwrap();

        let client = Client::builder()
            .bootstrap(&testnet.bootstrap)
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
            .put(&keypair.public_key().into(), &signed_packet);

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert_eq!(resolved, Some(signed_packet));
    }

    #[tokio::test]
    async fn ttl_0_test() {
        let testnet = Testnet::new(10).unwrap();

        let client = Client::builder()
            .bootstrap(&testnet.bootstrap)
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
