//! Pkarr client for publishing and resolving [SignedPacket]s over [mainline].

use flume::{Receiver, Sender};
use mainline::{
    dht::DhtSettings,
    rpc::{
        messages, QueryResponse, QueryResponseSpecific, ReceivedFrom, ReceivedMessage, Response,
        Rpc,
    },
    Id, MutableItem,
};
use std::{
    collections::HashMap,
    net::{SocketAddr, ToSocketAddrs},
    num::NonZeroUsize,
    thread,
};
use tracing::{debug, trace};

use crate::{
    cache::{InMemoryPkarrCache, PkarrCache},
    DEFAULT_CACHE_SIZE, DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL, DEFAULT_RESOLVERS,
};
use crate::{Error, PublicKey, Result, SignedPacket};

#[derive(Debug)]
/// [PkarrClient]'s settings
pub struct Settings {
    pub dht: DhtSettings,
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
    /// Custom [PkarrCache] implementation, defaults to [InMemoryPkarrCache]
    pub cache: Option<Box<dyn PkarrCache>>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            dht: DhtSettings::default(),
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
/// Builder for [PkarrClient]
pub struct PkarrClientBuilder {
    settings: Settings,
}

impl PkarrClientBuilder {
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

    /// Set a custom implementation of [PkarrCache].
    pub fn cache(mut self, cache: Box<dyn PkarrCache>) -> Self {
        self.settings.cache = Some(cache);
        self
    }

    /// Set [DhtSettings]
    pub fn dht_settings(mut self, settings: DhtSettings) -> Self {
        self.settings.dht = settings;
        self
    }

    pub fn build(self) -> Result<PkarrClient> {
        PkarrClient::new(self.settings)
    }
}

#[derive(Clone, Debug)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [mainline].
pub struct PkarrClient {
    pub(crate) address: Option<SocketAddr>,
    pub(crate) sender: Sender<ActorMessage>,
    pub(crate) cache: Box<dyn PkarrCache>,
    pub(crate) minimum_ttl: u32,
    pub(crate) maximum_ttl: u32,
}

impl PkarrClient {
    pub fn new(settings: Settings) -> Result<PkarrClient> {
        let (sender, receiver) = flume::bounded(32);

        let rpc = Rpc::new(&settings.dht)?;

        let local_addr = rpc.local_addr();

        let cache = settings
            .cache
            .clone()
            .unwrap_or(Box::new(InMemoryPkarrCache::new(settings.cache_size)));
        let cache_clone = cache.clone();

        let client = PkarrClient {
            address: Some(local_addr),
            sender,
            cache,
            minimum_ttl: settings.minimum_ttl,
            maximum_ttl: settings.maximum_ttl,
        };

        thread::Builder::new()
            .name("PkarrClient loop".to_string())
            .spawn(move || run(rpc, cache_clone, settings, receiver))?;

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
    pub fn cache(&self) -> &dyn PkarrCache {
        self.cache.as_ref()
    }

    // === Public Methods ===

    /// Publishes a [SignedPacket] to the Dht.
    ///
    /// # Errors
    /// - Returns a [Error::DhtIsShutdown] if [PkarrClient::shutdown] was called, or
    ///   the loop in the actor thread is stopped for any reason (like thread panic).
    /// - Returns a [Error::PublishInflight] if the client is currently publishing the same public_key.
    /// - Returns a [Error::NotMostRecent] if the provided signed packet is older than most recent.
    /// - Returns a [Error::MainlineError] if the Dht received an unexpected error otherwise.
    pub fn publish(&self, signed_packet: &SignedPacket) -> Result<()> {
        match self.publish_inner(signed_packet)?.recv() {
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
    ///   the loop in the actor thread is stopped for any reason (like thread panic).
    pub fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>> {
        Ok(self.resolve_inner(public_key)?.recv().ok())
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

    // === Private Methods ===

    pub(crate) fn publish_inner(
        &self,
        signed_packet: &SignedPacket,
    ) -> Result<Receiver<mainline::Result<Id>>> {
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

        Ok(receiver)
    }

    pub(crate) fn resolve_inner(&self, public_key: &PublicKey) -> Result<Receiver<SignedPacket>> {
        let target = MutableItem::target_from_key(public_key.as_bytes(), &None);

        let (sender, receiver) = flume::bounded::<SignedPacket>(1);

        let cached_packet = self.cache.get(&target);

        if let Some(ref cached) = cached_packet {
            let expires_in = cached.expires_in(self.minimum_ttl, self.maximum_ttl);

            if expires_in > 0 {
                debug!(expires_in, "Have fresh signed_packet in cache.");

                sender
                    .send(cached.clone())
                    .map_err(|_| Error::DhtIsShutdown)?;

                return Ok(receiver);
            }

            debug!(expires_in, "Have expired signed_packet in cache.");
        } else {
            debug!("Cache miss");
        }

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

        Ok(receiver)
    }
}

fn run(
    mut rpc: Rpc,
    cache: Box<dyn PkarrCache>,
    settings: Settings,
    receiver: Receiver<ActorMessage>,
) {
    debug!(?settings, "Starting PkarrClient main loop..");

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
            // Send cached signed packet one more time in case none of the incoming
            // packets were more recent than cache.
            if let Some(senders) = senders.get(id) {
                if let Some(signed_packet) = cache.get_read_only(id) {
                    for sender in senders {
                        let _ = sender.send(signed_packet.clone());
                    }
                }
            }

            senders.remove(id);
        }

        // === Receive and handle incoming messages ===
        if let Some(ReceivedFrom { from, message }) = &report.received_from {
            // match &report.received_from {
            // Some(ReceivedFrom { from, message }) => match message {
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
                                let new_packet =
                                    if let Some(ref cached) = cache.get_read_only(target) {
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
                                    cache.put(target, packet);

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
                            if let Some(mut cached) = cache.get_read_only(target) {
                                if (*seq as u64) == cached.timestamp() {
                                    trace!("Remote node has the a packet with same timestamp, refreshing cached packet.");

                                    cached.refresh();
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

    debug!("PkarrClient main terminated");
}

pub enum ActorMessage {
    Publish(MutableItem, Sender<mainline::Result<Id>>),
    Resolve(Id, Sender<SignedPacket>, Option<u64>),
    Shutdown(Sender<()>),
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use mainline::Testnet;

    use super::*;
    use crate::{Keypair, SignedPacket};

    #[test]
    fn shutdown() {
        let testnet = Testnet::new(3);

        let mut a = PkarrClient::builder()
            .dht_settings(DhtSettings {
                bootstrap: Some(testnet.bootstrap),
                request_timeout: None,
                server: None,
                port: None,
            })
            .build()
            .unwrap();

        assert_ne!(a.local_addr(), None);

        a.shutdown().unwrap();

        assert_eq!(a.local_addr(), None);
    }

    // #[test]
    // fn publish_resolve() {
    //     let testnet = Testnet::new(10);

    //     let a = PkarrClient::builder()
    //         .dht_settings(DhtSettings {
    //             bootstrap: Some(testnet.bootstrap.clone()),
    //             request_timeout: None,
    //             server: None,
    //             port: None,
    //         })
    //         .build()
    //         .unwrap();

    //     let keypair = Keypair::random();

    //     let mut packet = dns::Packet::new_reply(0);
    //     packet.answers.push(dns::ResourceRecord::new(
    //         dns::Name::new("foo").unwrap(),
    //         dns::CLASS::IN,
    //         30,
    //         dns::rdata::RData::TXT("bar".try_into().unwrap()),
    //     ));

    //     let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

    //     let _ = a.publish(&signed_packet);

    //     let b = PkarrClient::builder()
    //         .dht_settings(DhtSettings {
    //             bootstrap: Some(testnet.bootstrap),
    //             request_timeout: None,
    //             server: None,
    //             port: None,
    //         })
    //         .build()
    //         .unwrap();

    //     let resolved = b.resolve(&keypair.public_key()).unwrap().unwrap();
    //     assert_eq!(resolved.to_vec(), signed_packet.to_vec());

    //     let from_cache = b.resolve(&keypair.public_key()).unwrap().unwrap();
    //     assert_eq!(from_cache.to_vec(), signed_packet.to_vec());
    //     assert_eq!(from_cache.last_seen(), resolved.last_seen());
    // }

    // #[test]
    // fn thread_safe() {
    //     let testnet = Testnet::new(10);

    //     let a = PkarrClient::builder()
    //         .dht_settings(DhtSettings {
    //             bootstrap: Some(testnet.bootstrap.clone()),
    //             request_timeout: None,
    //             server: None,
    //             port: None,
    //         })
    //         .build()
    //         .unwrap();

    //     let keypair = Keypair::random();

    //     let mut packet = dns::Packet::new_reply(0);
    //     packet.answers.push(dns::ResourceRecord::new(
    //         dns::Name::new("foo").unwrap(),
    //         dns::CLASS::IN,
    //         30,
    //         dns::rdata::RData::TXT("bar".try_into().unwrap()),
    //     ));

    //     let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

    //     let _ = a.publish(&signed_packet);

    //     let b = PkarrClient::builder()
    //         .dht_settings(DhtSettings {
    //             bootstrap: Some(testnet.bootstrap),
    //             request_timeout: None,
    //             server: None,
    //             port: None,
    //         })
    //         .build()
    //         .unwrap();

    //     thread::spawn(move || {
    //         let resolved = b.resolve(&keypair.public_key()).unwrap().unwrap();
    //         assert_eq!(resolved.to_vec(), signed_packet.to_vec());

    //         let from_cache = b.resolve(&keypair.public_key()).unwrap().unwrap();
    //         assert_eq!(from_cache.to_vec(), signed_packet.to_vec());
    //         assert_eq!(from_cache.last_seen(), resolved.last_seen());
    //     })
    //     .join()
    //     .unwrap();
    // }

    // #[test]
    // fn ttl_0_test() {
    //     let testnet = Testnet::new(10);

    //     let client = PkarrClient::builder()
    //         .dht_settings(DhtSettings {
    //             bootstrap: Some(testnet.bootstrap),
    //             request_timeout: Some(Duration::from_millis(50)),
    //             server: None,
    //             port: None,
    //         })
    //         .maximum_ttl(0)
    //         .build()
    //         .unwrap();

    //     let keypair = Keypair::random();
    //     let mut packet = dns::Packet::new_reply(0);
    //     packet.answers.push(dns::ResourceRecord::new(
    //         dns::Name::new("foo").unwrap(),
    //         dns::CLASS::IN,
    //         30,
    //         dns::rdata::RData::TXT("bar".try_into().unwrap()),
    //     ));

    //     let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

    //     client.publish(&signed_packet).unwrap();

    //     // First Call
    //     let resolved = client
    //         .resolve(&signed_packet.public_key())
    //         .unwrap()
    //         .unwrap();

    //     assert_eq!(resolved.encoded_packet(), signed_packet.encoded_packet());

    //     thread::sleep(Duration::from_millis(10));

    //     let second = client
    //         .resolve(&signed_packet.public_key())
    //         .unwrap()
    //         .unwrap();
    //     assert_eq!(second.encoded_packet(), signed_packet.encoded_packet());
    // }
}
