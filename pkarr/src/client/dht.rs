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
/// [Client]'s settings
pub struct Settings {
    pub dht: mainline::Settings,
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

impl Default for Settings {
    fn default() -> Self {
        Self {
            dht: mainline::Settings::default(),
            cache_size: NonZeroUsize::new(DEFAULT_CACHE_SIZE).unwrap(),
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
/// Builder for [Client]
pub struct ClientBuilder {
    settings: Settings,
}

impl ClientBuilder {
    /// Set custom set of [resolvers](Settings::resolvers).
    pub fn resolvers(mut self, resolvers: Option<Vec<String>>) -> Self {
        self.settings.resolvers = resolvers.map(|resolvers| {
            resolvers
                .iter()
                .flat_map(|resolver| resolver.to_socket_addrs())
                .flatten()
                .collect::<Vec<_>>()
        });
        self
    }

    /// Set the [Settings::cache_size].
    ///
    /// Controls the capacity of [Cache].
    pub fn cache_size(mut self, cache_size: NonZeroUsize) -> Self {
        self.settings.cache_size = cache_size;
        self
    }

    /// Set the [Settings::minimum_ttl] value.
    ///
    /// Limits how soon a [SignedPacket] is considered expired.
    pub fn minimum_ttl(mut self, ttl: u32) -> Self {
        self.settings.minimum_ttl = ttl;
        self.settings.maximum_ttl = self.settings.maximum_ttl.clamp(ttl, u32::MAX);
        self
    }

    /// Set the [Settings::maximum_ttl] value.
    ///
    /// Limits how long it takes before a [SignedPacket] is considered expired.
    pub fn maximum_ttl(mut self, ttl: u32) -> Self {
        self.settings.maximum_ttl = ttl;
        self.settings.minimum_ttl = self.settings.minimum_ttl.clamp(0, ttl);
        self
    }

    /// Set a custom implementation of [Cache].
    pub fn cache(mut self, cache: Box<dyn Cache>) -> Self {
        self.settings.cache = Some(cache);
        self
    }

    /// Set [Settings::dht]
    pub fn dht_settings(mut self, settings: mainline::Settings) -> Self {
        self.settings.dht = settings;
        self
    }

    /// Convienent methot to set the [mainline::Settings::bootstrap] from [mainline::Testnet::bootstrap]
    pub fn testnet(mut self, testnet: &Testnet) -> Self {
        self.settings.dht.bootstrap = testnet.bootstrap.clone().into();
        self
    }

    pub fn build(self) -> Result<Client, std::io::Error> {
        Client::new(self.settings)
    }
}

#[derive(Clone, Debug)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [mainline].
pub struct Client {
    address: Option<SocketAddr>,
    sender: Sender<ActorMessage>,
    cache: Box<dyn Cache>,
    minimum_ttl: u32,
    maximum_ttl: u32,
}

impl Client {
    pub fn new(settings: Settings) -> Result<Client, std::io::Error> {
        let (sender, receiver) = flume::bounded(32);

        let rpc = Rpc::new(&settings.dht)?;

        let local_addr = rpc.local_addr()?;

        let cache = settings
            .cache
            .clone()
            .unwrap_or(Box::new(InMemoryCache::new(settings.cache_size)));
        let cache_clone = cache.clone();

        let client = Client {
            address: Some(local_addr),
            sender,
            cache,
            minimum_ttl: settings.minimum_ttl,
            maximum_ttl: settings.maximum_ttl,
        };

        thread::Builder::new()
            .name("Client loop".to_string())
            .spawn(move || run(rpc, cache_clone, settings, receiver))?;

        Ok(client)
    }

    /// Returns a builder to edit settings before creating Client.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    // === Getters ===

    /// Returns the local address of the udp socket this node is listening on.
    ///
    /// Returns `None` if the node is shutdown
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.address
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

        self.address = None;
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
    pub fn shutdown_sync(&mut self) {
        let (sender, receiver) = flume::bounded(1);

        let _ = self.sender.send(ActorMessage::Shutdown(sender));
        let _ = receiver.recv();

        self.address = None;
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
                    as_ref.map(|cached| cached.timestamp()),
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

fn run(mut rpc: Rpc, cache: Box<dyn Cache>, settings: Settings, receiver: Receiver<ActorMessage>) {
    debug!(?settings, "Starting Client main loop..");

    let mut server = settings.dht.server;
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

                    rpc.get(target, request, None, settings.resolvers.clone())
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
                                if (*seq as u64) == cached.timestamp() {
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
                        server.handle_request(&mut rpc, *from, *transaction_id, request);
                    }
                }
            };
        }
    }

    debug!("Client main loop terminated");
}

pub enum ActorMessage {
    Publish(MutableItem, Sender<Result<Id, PutError>>),
    Resolve(Id, Sender<SignedPacket>, Option<u64>),
    Shutdown(Sender<()>),
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use mainline::Testnet;

    use super::*;
    use crate::{dns, Keypair, SignedPacket};

    #[test]
    fn shutdown_sync() {
        let testnet = Testnet::new(3).unwrap();

        let mut a = Client::builder().testnet(&testnet).build().unwrap();

        assert_ne!(a.local_addr(), None);

        a.shutdown_sync();

        assert_eq!(a.local_addr(), None);
    }

    #[test]
    fn publish_resolve_sync() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder().testnet(&testnet).build().unwrap();

        let keypair = Keypair::random();

        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("foo").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::TXT("bar".try_into().unwrap()),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

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

        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("foo").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::TXT("bar".try_into().unwrap()),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

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

        assert_ne!(a.local_addr(), None);

        a.shutdown().await;

        assert_eq!(a.local_addr(), None);
    }

    #[tokio::test]
    async fn publish_resolve() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder().testnet(&testnet).build().unwrap();

        let keypair = Keypair::random();

        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("foo").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::TXT("bar".try_into().unwrap()),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

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

        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("foo").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::TXT("bar".try_into().unwrap()),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

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
            .dht_settings(mainline::Settings {
                request_timeout: Duration::from_millis(10).into(),
                ..Default::default()
            })
            // Everything is expired
            .maximum_ttl(0)
            .build()
            .unwrap();

        let keypair = Keypair::random();
        let packet = dns::Packet::new_reply(0);
        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

        client
            .cache()
            .put(&keypair.public_key().into(), &signed_packet);

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert_eq!(resolved, Some(signed_packet));
    }
}
