//! Pkarr client for publishing and resolving [SignedPacket]s over [mainline].

use flume::{Receiver, Sender};
use mainline::{
    errors::PutError,
    rpc::{
        messages, QueryResponse, QueryResponseSpecific, ReceivedFrom, ReceivedMessage, Response,
        Rpc,
    },
    Id, MutableItem, Testnet,
};
use std::{
    collections::HashMap,
    net::{SocketAddr, ToSocketAddrs},
    num::NonZeroUsize,
    thread,
};
use tracing::{debug, trace};

use crate::{
    Cache, InMemoryCache, DEFAULT_CACHE_SIZE, DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL,
    DEFAULT_RESOLVERS,
};
use crate::{PublicKey, SignedPacket};

#[derive(Debug)]
/// [Client]'s Config
pub struct Config {
    pub dht_config: mainline::Config,
    /// A set of [resolver](https://pkarr.org/resolvers)s
    /// to be queried alongside the Dht routing table, to
    /// lower the latency on cold starts, and help if the
    /// Dht is missing values not't republished often enough.
    ///
    /// Defaults to [DEFAULT_RESOLVERS]
    pub resolvers: Option<Vec<SocketAddr>>,
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
            dht_config: mainline::Config::default(),
            cache_size: NonZeroUsize::new(DEFAULT_CACHE_SIZE)
                .expect("NonZeroUsize from DEFAULT_CACHE_SIZE"),
            resolvers: Some(
                DEFAULT_RESOLVERS
                    .iter()
                    .flat_map(|resolver| resolver.to_socket_addrs())
                    .flatten()
                    .collect::<Vec<_>>(),
            ),
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
        self.0.resolvers = resolvers.map(|resolvers| {
            resolvers
                .iter()
                .flat_map(|resolver| resolver.to_socket_addrs())
                .flatten()
                .collect::<Vec<_>>()
        });
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
        self.0.maximum_ttl = self.0.maximum_ttl.max(ttl);
        self
    }

    /// Set the [Config::maximum_ttl] value.
    ///
    /// Limits how long it takes before a [SignedPacket] is considered expired.
    pub fn maximum_ttl(mut self, ttl: u32) -> Self {
        self.0.maximum_ttl = ttl;
        self.0.minimum_ttl = self.0.minimum_ttl.min(ttl);
        self
    }

    /// Set a custom implementation of [Cache].
    pub fn cache(mut self, cache: Box<dyn Cache>) -> Self {
        self.0.cache = Some(cache);
        self
    }

    /// Set [Config::dht_config]
    pub fn dht_config(mut self, settings: mainline::Config) -> Self {
        self.0.dht_config = settings;
        self
    }

    /// Convienent methot to set the [mainline::Settings::bootstrap] from [mainline::Testnet::bootstrap]
    pub fn testnet(mut self, testnet: &Testnet) -> Self {
        self.0.dht_config.bootstrap = testnet.bootstrap.clone();

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

    /// Returns a [SignedPacket] from cache if it is not expired, otherwise,
    /// it will query the Dht, and return the first valid response, which may
    /// or may not be expired itself.
    ///
    /// If the Dht was called, in the background, it continues receiving responses
    /// and updating the cache with any more recent valid packets it receives.
    ///
    /// # Errors
    /// - Returns a [ClientWasShutdown] if [Client::shutdown] was called, or
    ///   the loop in the actor thread is stopped for any reason (like thread panic).
    pub async fn resolve(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        Ok(self.resolve_inner(public_key)?.recv_async().await.ok())
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

    /// Returns a [SignedPacket] from cache if it is not expired, otherwise,
    /// it will query the Dht, and return the first valid response, which may
    /// or may not be expired itself.
    ///
    /// If the Dht was called, in the background, it continues receiving responses
    /// and updating the cache with any more recent valid packets it receives.
    ///
    /// # Errors
    /// - Returns a [ClientWasShutdown] if [Client::shutdown] was called, or
    ///   the loop in the actor thread is stopped for any reason (like thread panic).
    pub fn resolve_sync(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        Ok(self.resolve_inner(public_key)?.recv().ok())
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

    pub(crate) fn resolve_inner(
        &self,
        public_key: &PublicKey,
    ) -> Result<Receiver<SignedPacket>, ClientWasShutdown> {
        let target = MutableItem::target_from_key(public_key.as_bytes(), &None);

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
    match Rpc::new(
        config
            .dht_config
            .bootstrap
            .to_owned()
            .iter()
            .flat_map(|s| s.to_socket_addrs().map(|addrs| addrs.collect::<Vec<_>>()))
            .flatten()
            .collect::<Vec<_>>(),
        config.dht_config.server.is_none(),
        config.dht_config.request_timeout,
        config.dht_config.port,
    ) {
        Ok(mut rpc) => {
            let mut server = config.dht_config.server;
            actor_thread(&mut rpc, &mut server, cache, receiver, config.resolvers)
        }
        Err(err) => {
            if let Ok(ActorMessage::Check(sender)) = receiver.try_recv() {
                let _ = sender.send(Err(err));
            }
        }
    }
}

fn actor_thread(
    rpc: &mut Rpc,
    server: &mut Option<Box<dyn mainline::server::Server>>,
    cache: Box<dyn Cache>,
    receiver: Receiver<ActorMessage>,
    resolvers: Option<Vec<SocketAddr>>,
) {
    let mut senders: HashMap<Id, Vec<Sender<SignedPacket>>> = HashMap::new();

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
                    let target = mutable_item.target();

                    let request = messages::PutRequestSpecific::PutMutable(
                        messages::PutMutableRequestArguments {
                            target: *target,
                            v: mutable_item.value().to_vec(),
                            k: mutable_item.key().to_vec(),
                            seq: *mutable_item.seq(),
                            sig: mutable_item.signature().to_vec(),
                            salt: None,
                            cas: None,
                        },
                    );

                    rpc.put(*target, request, Some(sender))
                }
                ActorMessage::Resolve(target, sender, most_recent_known_timestamp) => {
                    if let Some(set) = senders.get_mut(&target) {
                        set.push(sender);
                    } else {
                        senders.insert(target, vec![sender]);
                    };

                    let request = messages::RequestTypeSpecific::GetValue(
                        messages::GetValueRequestArguments {
                            target,
                            seq: most_recent_known_timestamp.map(|t| t as i64),
                            // seq: None,
                            salt: None,
                        },
                    );

                    rpc.get(target, request, None, resolvers.clone())
                }
                ActorMessage::Info(sender) => {
                    let local_addr = rpc.local_addr();

                    let _ = sender.send(Info { local_addr });
                }
                ActorMessage::Check(sender) => {
                    let _ = sender.send(Ok(()));
                }
            }
        }

        // === Dht Tick ===

        let report = rpc.tick();

        // === Drop senders to done queries ===
        for id in &report.done_get_queries {
            if let Some(senders) = senders.remove(id) {
                if let Some(cached) = cache.get(id.as_bytes()) {
                    debug!(public_key = ?cached.public_key(), "Returning expired cache as a fallback");
                    // Send cached packets if available
                    for sender in senders {
                        let _ = sender.send(cached.clone());
                    }
                }
            };
        }

        // === Receive and handle incoming messages ===
        if let Some(ReceivedFrom { from, message }) = &report.received_from {
            match message {
                // === Responses ===
                ReceivedMessage::QueryResponse(response) => {
                    match response {
                        // === Got Mutable Value ===
                        QueryResponse {
                            target,
                            response: QueryResponseSpecific::Value(Response::Mutable(mutable_item)),
                        } => {
                            if let Ok(signed_packet) = &SignedPacket::try_from(mutable_item) {
                                let new_packet = if let Some(ref cached) =
                                    cache.get_read_only(target.as_bytes())
                                {
                                    if signed_packet.more_recent_than(cached) {
                                        debug!(
                                            ?target,
                                            "Received more recent packet than in cache"
                                        );
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

                                    if let Some(set) = senders.get(target) {
                                        for sender in set {
                                            let _ = sender.send(packet.clone());
                                        }
                                    }
                                }
                            }
                        }
                        // === Got NoMoreRecentValue ===
                        QueryResponse {
                            target,
                            response: QueryResponseSpecific::Value(Response::NoMoreRecentValue(seq)),
                        } => {
                            if let Some(mut cached) = cache.get_read_only(target.as_bytes()) {
                                if (*seq as u64) == cached.timestamp().as_u64() {
                                    trace!("Remote node has the a packet with same timestamp, refreshing cached packet.");

                                    cached.refresh();
                                    cache.put(target.as_bytes(), &cached);

                                    // Send the found sequence as a timestamp to the caller to decide what to do
                                    // with it.
                                    if let Some(set) = senders.get(target) {
                                        for sender in set {
                                            let _ = sender.send(cached.clone());
                                        }
                                    }
                                }
                            };
                        }
                        // Ignoring errors, as they are logged in `mainline` crate already.
                        _ => {}
                    };
                }
                // === Requests ===
                ReceivedMessage::Request((transaction_id, request)) => {
                    if let Some(server) = server.as_mut() {
                        server.handle_request(rpc, *from, *transaction_id, request);
                    }
                }
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
    local_addr: Result<SocketAddr, std::io::Error>,
}

impl Info {
    /// Local UDP Ipv4 socket address that this node is listening on.
    pub fn local_addr(&self) -> Result<&SocketAddr, std::io::Error> {
        self.local_addr
            .as_ref()
            .map_err(|e| std::io::Error::new(e.kind(), e.to_string()))
    }
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

        let client = Client::builder().testnet(&testnet).build().unwrap();

        let local_addr = client.info().unwrap().local_addr().unwrap().to_string();

        println!("{}", local_addr);

        assert!(client.info().unwrap().local_addr().is_ok());

        client.shutdown_sync();

        assert!(client.info().is_err());
    }

    #[test]
    fn publish_resolve_sync() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder().testnet(&testnet).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish_sync(&signed_packet).unwrap();

        let b = Client::builder().testnet(&testnet).build().unwrap();

        let resolved = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }

    #[test]
    fn thread_safe_sync() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder().testnet(&testnet).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish_sync(&signed_packet).unwrap();

        let b = Client::builder().testnet(&testnet).build().unwrap();

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

        let mut a = Client::builder().testnet(&testnet).build().unwrap();

        assert!(a.info().unwrap().local_addr().is_ok());

        a.shutdown().await;

        assert!(a.info().is_err());
    }

    #[tokio::test]
    async fn publish_resolve() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder().testnet(&testnet).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = Client::builder().testnet(&testnet).build().unwrap();

        let resolved = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }

    #[tokio::test]
    async fn thread_safe() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder().testnet(&testnet).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = Client::builder().testnet(&testnet).build().unwrap();

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
            .testnet(&testnet)
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
}
