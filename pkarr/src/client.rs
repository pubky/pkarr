use flume::{Receiver, Sender};
use mainline::{
    dht::DhtSettings,
    rpc::{
        messages::{
            self, GetValueRequestArguments, RequestSpecific, RequestTypeSpecific, ResponseSpecific,
        },
        QueryResponse, QueryResponseSpecific, ReceivedFrom, ReceivedMessage, Response, Rpc,
    },
    Id, MutableItem,
};
use std::{
    collections::HashMap,
    net::{SocketAddr, ToSocketAddrs},
    num::NonZeroUsize,
    thread,
    time::Instant,
};
use tracing::{debug, info, instrument};

use crate::{
    cache::PkarrCache, Error, PublicKey, Result, SignedPacket, DEFAULT_MAXIMUM_TTL,
    DEFAULT_MINIMUM_TTL,
};

pub const DEFAULT_CACHE_SIZE: usize = 1000;

pub const DEFAULT_RESOLVERS: [&str; 1] = ["resolver.pkarr.org:7101"];

#[derive(Debug, Clone)]
pub struct Settings {
    pub dht: DhtSettings,
    /// If set to `true`, run as a [resolver](https://pkarr.org/resolvers)s.
    ///
    /// Defaults to `false`
    pub resolver: bool,
    /// Controls the Resolver behavior when querying the Dht on cache miss.
    /// if set to `true`, it will query [Settings::resolvers] alongside the closest nodes.
    ///
    /// If [Settings::resolver] is `false` (default), this will have no effect.
    ///
    /// Defaults to `false`
    pub recursive: bool,
    /// A set of [resolver](https://pkarr.org/resolvers)s
    /// to be queried alongside the Dht routing table, to
    /// lower the latency on cold starts, and help if the
    /// Dht is missing values not't republished often enough.
    ///
    /// Defaults to [DEFAULT_RESOLVERS]
    pub resolvers: Vec<SocketAddr>,
    /// Defaults to [DEFAULT_CACHE_SIZE]
    pub cache_size: NonZeroUsize,
    /// Used in the `min` parametere in [SignedPacket::ttl].
    ///
    /// Defaults to [DEFAULT_MINIMUM_TTL]
    pub minimum_ttl: u32,
    /// Used in the `max` parametere in [SignedPacket::ttl].
    ///
    /// Defaults to [DEFAULT_MAXIMUM_TTL]
    pub maximum_ttl: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            dht: DhtSettings::default(),
            cache_size: NonZeroUsize::new(DEFAULT_CACHE_SIZE).unwrap(),
            resolver: false,
            recursive: false,
            resolvers: DEFAULT_RESOLVERS
                .iter()
                .flat_map(|resolver| resolver.to_socket_addrs())
                .flatten()
                .collect::<Vec<_>>(),
            minimum_ttl: DEFAULT_MINIMUM_TTL,
            maximum_ttl: DEFAULT_MAXIMUM_TTL,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PkarrClientBuilder {
    settings: Settings,
}

impl PkarrClientBuilder {
    /// Set [Settings::resolver] to true.
    ///
    /// Run this client as a [resolver](https://pkarr.org/resolvers)
    pub fn resolver(mut self) -> Self {
        self.settings.resolver = true;
        self.settings.dht.server = true;
        self
    }

    /// Set [Settings::recursive] to true.
    ///
    /// Set to true to help your resolver leverage even bigger resolvers.
    pub fn recursive(mut self) -> Self {
        self.settings.recursive = true;
        self
    }

    /// Set custom set of [resolvers](Settings::resolvers).
    pub fn resolvers(mut self, resolvers: Vec<String>) -> Self {
        self.settings.resolvers = resolvers
            .iter()
            .flat_map(|resolver| resolver.to_socket_addrs())
            .flatten()
            .collect::<Vec<_>>();
        self
    }

    /// Set the Dht bootstrapping nodes.
    pub fn bootstrap(mut self, bootstrap: &[String]) -> Self {
        self.settings.dht.bootstrap = Some(bootstrap.to_vec());
        self
    }

    /// Set the port to listen on.
    pub fn port(mut self, port: u16) -> Self {
        self.settings.dht.port = Some(port);
        self
    }

    /// Set the [Settings::cache_size].
    ///
    /// Controls the capacity of [PkarrCache].
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

    pub fn build(self) -> Result<PkarrClient> {
        PkarrClient::new(self.settings)
    }
}

#[derive(Clone, Debug)]
/// Native Pkarr client for publishing and resolving [SignedPacket]s.
pub struct PkarrClient {
    pub(crate) address: Option<SocketAddr>,
    pub(crate) sender: Sender<ActorMessage>,
    pub(crate) cache: PkarrCache,
    pub(crate) minimum_ttl: u32,
    pub(crate) maximum_ttl: u32,
}

impl PkarrClient {
    pub fn new(settings: Settings) -> Result<PkarrClient> {
        let (sender, receiver) = flume::bounded(32);

        debug!(?settings, "Starting PkarrClient..");

        let rpc = Rpc::new(settings.dht.to_owned())?;

        let local_addr = rpc.local_addr();

        info!(?local_addr, "Running PkarrClient");

        let cache = PkarrCache::new(settings.cache_size);
        let moved_cache = cache.clone();

        let client = PkarrClient {
            address: Some(local_addr),
            sender,
            cache,
            minimum_ttl: settings.minimum_ttl,
            maximum_ttl: settings.maximum_ttl,
        };

        thread::spawn(move || run(rpc, moved_cache, settings, receiver));

        Ok(client)
    }

    /// Returns a builder to edit settings before creating PkarrClient.
    pub fn builder() -> PkarrClientBuilder {
        PkarrClientBuilder::default()
    }

    // === Getters ===

    /// Returns the local address of the udp socket this node is listening on.
    ///
    /// Returns `None` if the node is shutdown
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.address
    }

    /// Returns a reference to the internal cache.
    pub fn cache(&self) -> &PkarrCache {
        &self.cache
    }

    // === Public Methods ===

    /// Publishes a [SignedPacket] to the Dht.
    ///
    /// # Errors
    /// - Returns a [Error::DhtIsShutdown] if [PkarrClient::shutdown] was called, or
    /// the loop in the actor thread is stopped for any reason (like thread panic).
    /// - Returns a [Error::PublishInflight] if the client is currently publishing the same public_key.
    /// - Returns a [Error::NotMostRecent] if the provided signed packet is older than most recent.
    /// - Returns a [Error::MainlineError] if the Dht received an unexpected error otherwise.
    pub fn publish(&self, signed_packet: &SignedPacket) -> Result<()> {
        let mutable_item: MutableItem = (signed_packet).into();

        if let Some(current) = self.cache.get(mutable_item.target()) {
            if current.timestamp() > signed_packet.timestamp() {
                return Err(Error::NotMostRecent);
            }
        };

        self.cache.put(mutable_item.target(), signed_packet);

        let (sender, receiver) = flume::bounded::<mainline::Result<Id>>(1);

        self.sender
            .send(ActorMessage::Publish(mutable_item, sender))
            .map_err(|_| Error::DhtIsShutdown)?;

        match receiver.recv() {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(error)) => match error {
                mainline::Error::PutQueryIsInflight(_) => Err(Error::PublishInflight),
                _ => Err(Error::MainlineError(error)),
            },
            // Since we pass this sender to `Rpc::put`, the only reason the sender,
            // would be dropped, is if `Rpc` is dropped, which should only happeng on shutdown.
            Err(_) => Err(Error::DhtIsShutdown),
        }
    }

    /// Returns a [SignedPacket] from cache if it is not expired, otherwise,
    /// it will query the Dht, and return the first valid response, which may
    /// or may not be expired itself.
    ///
    /// If the Dht was called, in the background, it continues receiving responses
    /// and updating the cache with any more recent valid packets it receives.
    ///
    /// # Errors
    /// - Returns a [Error::DhtIsShutdown] if [PkarrClient::shutdown] was called, or
    /// the loop in the actor thread is stopped for any reason (like thread panic).
    #[instrument(skip(self))]
    pub fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>> {
        let target = MutableItem::target_from_key(public_key.as_bytes(), &None);

        let cached_packet = self.cache.get(&target);

        if let Some(ref cached) = cached_packet {
            let expires_in = cached.expires_in(self.minimum_ttl, self.maximum_ttl);

            if expires_in > 0 {
                debug!(expires_in, "Have fresh signed_packet in cache.");
                return Ok(Some(cached.clone()));
            }
        }

        // Cache miss

        let (sender, receiver) = flume::bounded::<SignedPacket>(1);

        debug!("Cache miss, asking the network for a fresh signed_packet");

        self.sender
            .send(ActorMessage::Resolve(
                target,
                sender,
                // Sending the `timestamp` of the known cache, help save some bandwith,
                // since remote nodes will not send the encoded packet if they don't know
                // any more recent versions.
                cached_packet.as_ref().map(|cached| cached.timestamp()),
            ))
            .map_err(|_| Error::DhtIsShutdown)?;

        Ok(receiver.recv().ok())
    }

    /// Shutdown the actor thread loop.
    pub fn shutdown(&mut self) -> Result<()> {
        let (sender, receiver) = flume::bounded(1);

        self.sender
            .send(ActorMessage::Shutdown(sender))
            .map_err(|_| Error::DhtIsShutdown)?;

        receiver.recv()?;

        self.address = None;

        Ok(())
    }
}

fn run(mut rpc: Rpc, cache: PkarrCache, settings: Settings, receiver: Receiver<ActorMessage>) {
    let mut server = mainline::server::Server::new(&settings.dht.server_settings);
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

                    rpc.get(target, request, None, Some(settings.resolvers.clone()))
                }
            }
        }

        // === Dht Tick ===

        let report = rpc.tick();

        // === Drop senders to done queries ===
        for id in &report.done_get_queries {
            senders.remove(id);
        }

        // === Receive and handle incoming messages ===
        match &report.received_from {
            Some(ReceivedFrom {
                message:
                    ReceivedMessage::QueryResponse(QueryResponse {
                        target,
                        response: QueryResponseSpecific::Value(Response::Mutable(mutable_item)),
                    }),
                ..
            }) => {
                if let Ok(signed_packet) = &SignedPacket::try_from(mutable_item) {
                    let new_packet = if let Some(ref cached) = cache.get(target) {
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
                        cache.put(target, packet);

                        if let Some(set) = senders.get(target) {
                            for sender in set {
                                let _ = sender.send(packet.clone());
                            }
                        }
                    }
                }
            }
            Some(ReceivedFrom {
                message:
                    ReceivedMessage::QueryResponse(QueryResponse {
                        target,
                        response: QueryResponseSpecific::Value(Response::NoMoreRecentValue(seq)),
                    }),
                ..
            }) => {
                if let Some(mut cached) = cache.get(target) {
                    if (*seq as u64) == cached.timestamp() {
                        debug!("Remote node has the a packet with same timestamp, refreshing cached packet.");

                        cached.set_last_seen(&Instant::now());
                        cache.put(target, &cached);

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
            Some(ReceivedFrom {
                from,
                message: ReceivedMessage::Request((transaction_id, request)),
            }) => {
                // === Resolver Logic ===

                // We shouldn't reach this if [Settings.resolver] is not set to true,
                // because that sets [DhtSettings::server].
                if let RequestSpecific {
                    request_type:
                        RequestTypeSpecific::GetValue(GetValueRequestArguments { target, .. }),
                    ..
                } = request
                {
                    if let Some(cached) = cache.get(target) {
                        // We don't care about expiry, if the client doesn't want it, they
                        // can discard it. But it is useful to return whatever we have.
                        let mutable_item = MutableItem::from(&cached);
                        debug!(?target, "Resolver: cache hit responding with packet!");

                        rpc.response(
                            *from,
                            *transaction_id,
                            ResponseSpecific::GetMutable(messages::GetMutableResponseArguments {
                                responder_id: *rpc.id(),
                                // Token doesn't matter much, as we are most likely _not_ the
                                // closest nodes, so we shouldn't expect an PUT requests based on
                                // this response.
                                token: vec![0, 0, 0, 0],
                                nodes: None,
                                v: mutable_item.value().to_vec(),
                                k: mutable_item.key().to_vec(),
                                seq: *mutable_item.seq(),
                                sig: mutable_item.signature().to_vec(),
                            }),
                        )
                    } else {
                        debug!(?target, "Resolver: cache miss, requesting from network.");
                        rpc.get(
                            *target,
                            RequestTypeSpecific::GetValue(GetValueRequestArguments {
                                target: *target,
                                seq: None,
                                salt: None,
                            }),
                            None,
                            if settings.recursive {
                                Some(settings.resolvers.clone())
                            } else {
                                None
                            },
                        );
                    }
                };

                server.handle_request(&mut rpc, *from, *transaction_id, request);
            }
            _ => {}
        }
    }
}

pub enum ActorMessage {
    Publish(MutableItem, Sender<mainline::Result<Id>>),
    Resolve(Id, Sender<SignedPacket>, Option<u64>),
    Shutdown(Sender<()>),
}

#[cfg(test)]
mod tests {
    use mainline::Testnet;

    use super::*;
    use crate::{dns, Keypair, SignedPacket};

    #[test]
    fn shutdown() {
        let testnet = Testnet::new(3);

        let mut a = PkarrClient::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        assert_ne!(a.local_addr(), None);

        a.shutdown().unwrap();

        assert_eq!(a.local_addr(), None);
    }

    #[test]
    fn publish_resolve() {
        let testnet = Testnet::new(10);

        let a = PkarrClient::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("foo").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::TXT("bar".try_into().unwrap()),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

        let _ = a.publish(&signed_packet);

        let b = PkarrClient::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let resolved = b.resolve(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }
}
