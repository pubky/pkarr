use flume::{Receiver, Sender};
use mainline::{
    dht::DhtSettings,
    rpc::{
        messages::{self},
        QueryResponse, QueryResponseSpecific, ReceivedFrom, ReceivedMessage, Response, Rpc,
    },
    Id, MutableItem,
};
use std::{collections::HashMap, num::NonZeroUsize, thread};
use tracing::{debug, instrument};

use crate::{cache::PkarrCache, Error, PublicKey, Result, SignedPacket};

pub const DEFAULT_CACHE_SIZE: usize = 1000;

// TODO: Use NoMoreRecentValue even if the data is not in cache
// TODO: Figure out how or should we use Cas
// TODO: Resolver
// TODO: test shutdown
// TODO: examine errors (failed to publish, failed to bind socket, unused errors...)
// TODO: logs (info for binding, debug for steps)
// TODO: HTTP relay should return some caching headers.
// TODO: add server settings to mainline DhtSettings

#[derive(Debug, Clone)]
pub struct Settings {
    dht: DhtSettings,
    cache_size: usize,
    minimum_ttl: Option<u32>,
    maximum_ttl: Option<u32>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            dht: DhtSettings::default(),
            cache_size: DEFAULT_CACHE_SIZE,
            minimum_ttl: None,
            maximum_ttl: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PkarrClientBuilder {
    settings: Settings,
}

impl PkarrClientBuilder {
    /// Create a full DHT node that accepts requests, and acts as a routing and storage node.
    pub fn server(mut self) -> Self {
        self.settings.dht.read_only = false;
        self
    }

    /// Set the Dht bootstrapping nodes
    pub fn bootstrap(mut self, bootstrap: &[String]) -> Self {
        self.settings.dht.bootstrap = Some(bootstrap.to_vec());
        self
    }

    /// Set the port to listen on.
    pub fn port(mut self, port: u16) -> Self {
        self.settings.dht.port = Some(port);
        self
    }

    // TODO: allow custom Cache with traits.
    /// Set the [SignedPacket] cache size.
    /// Defaults to [DEFAULT_CACHE_SIZE].
    pub fn cache_size(mut self, cache_size: usize) -> Self {
        self.settings.cache_size = cache_size;
        self
    }

    /// Set the minimum ttl for a cached [SignedPacket].
    /// Defaults to [crate::signed_packet::DEFAULT_MINIMUM_TTL].
    ///
    /// Internally the cache will expire after the smallest ttl in
    /// the resource records, unless it is smaller than this minimum.
    pub fn minimum_ttl(mut self, ttl: u32) -> Self {
        self.settings.minimum_ttl = Some(ttl);
        self
    }

    /// Set the maximum ttl for a cached [SignedPacket].
    /// Defaults to [crate::signed_packet::DEFAULT_MAXIMUM_TTL].
    ///
    /// Internally the cache will expire after the smallest ttl in
    /// the resource records, unless it is bigger than this maximum.
    pub fn maximum_ttl(mut self, ttl: u32) -> Self {
        self.settings.maximum_ttl = Some(ttl);
        self
    }

    pub fn build(self) -> Result<PkarrClient> {
        PkarrClient::new(self.settings)
    }
}

#[derive(Clone, Debug)]
/// Main client for publishing and resolving [SignedPacket]s.
pub struct PkarrClient {
    sender: Sender<ActorMessage>,
    cache: PkarrCache,
    minimum_ttl: Option<u32>,
    maximum_ttl: Option<u32>,
}

impl PkarrClient {
    pub fn new(settings: Settings) -> Result<PkarrClient> {
        let (sender, receiver) = flume::bounded(32);

        let mut rpc = Rpc::new()?.with_read_only(settings.dht.read_only);

        if let Some(bootstrap) = settings.dht.bootstrap.clone() {
            rpc = rpc.with_bootstrap(bootstrap);
        }

        if let Some(port) = settings.dht.port {
            rpc = rpc.with_port(port)?;
        }

        let cache = PkarrCache::new(NonZeroUsize::new(DEFAULT_CACHE_SIZE).unwrap());

        let cloned = cache.clone();
        thread::spawn(move || run(rpc, settings.dht, cloned, receiver));

        Ok(PkarrClient {
            sender,
            cache,
            minimum_ttl: settings.minimum_ttl,
            maximum_ttl: settings.maximum_ttl,
        })
    }

    /// Returns a builder to edit settings before creating PkarrClient.
    pub fn builder() -> PkarrClientBuilder {
        PkarrClientBuilder::default()
    }

    /// Publishes a [SignedPacket] to the Dht.
    ///
    /// # Errors
    /// - Returns a [Error::DhtIsShutdown] if [PkarrClient::shutdown] was called, or
    /// the loop in the actor thread is stopped for any reason (like thread panic).
    /// - Returns a [Error::PublishInflight] if the client is currently publishing the same public_key.
    /// - Returns a [Error::FailedToPublish] if all the closest nodes in the Dht responded with
    /// errors.
    /// - Returns a [Error::MainlineError] if the Dht received an unexpected error otherwise.
    pub fn publish(&self, signed_packet: &SignedPacket) -> Result<()> {
        let (sender, receiver) = flume::bounded::<mainline::Result<Id>>(1);

        self.sender
            .send(ActorMessage::Publish(signed_packet.clone(), sender))
            .map_err(|_| Error::DhtIsShutdown)?;

        match receiver.recv() {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(error)) => match error {
                mainline::Error::PutQueryIsInflight(_) => Err(Error::PublishInflight),
                // Should not be reachable unless all nodes responded with a QueryError,
                // which is either a bug in mainline crate, or just malicious responses.
                mainline::Error::QueryError(error) => Err(Error::FailedToPublish(error)),
                // Should not be reachable unless we did something wrong!
                _ => Err(Error::MainlineError(error)),
            },
            // Since we pass this sender to `Rpc::put`, the only reason the sender,
            // would be dropped, is if `Rpc` is dropped, which should only happeng on shutdown.
            Err(_) => Err(Error::DhtIsShutdown),
        }
    }

    /// Returns the first valid [SignedPacket] available from cache, or the Dht.
    ///
    /// If the Dht was called, in the background, it continues receiving responses
    /// and updating the cache.
    ///
    /// # Errors
    /// - Returns a [Error::DhtIsShutdown] if [PkarrClient::shutdown] was called, or
    /// the loop in the actor thread is stopped for any reason (like thread panic).
    /// - Returns a [Error::NotFound] if no packet was resolved.
    #[instrument(skip(self))]
    pub fn resolve(&self, public_key: &PublicKey) -> Result<SignedPacket> {
        let target = MutableItem::target_from_key(public_key.as_bytes(), &None);

        let mut cached_packet = self.cache.get(&target);

        if let Some(ref cached) = cached_packet {
            let ttl = cached.ttl(self.minimum_ttl, self.maximum_ttl);

            if ttl > 0 {
                debug!(ttl, "Have fresh signed_packet in cache.");
                return Ok(cached.clone());
            }

            debug!("Cached packet is not fresh. Asking the network for more recent values.");
        }

        // Cache miss

        let (sender, receiver) = flume::bounded::<ResolveResponse>(1);

        let most_recent_known_seq = cached_packet
            .as_ref()
            .map(|cached| cached.timestamp() as i64);

        self.sender
            .send(ActorMessage::Resolve(
                target,
                sender,
                // Sending the `seq` of the known cache, help save some bandwith,
                // since remote nodes will not send the encoded packet if they don't know
                // any more recent versions.
                most_recent_known_seq,
            ))
            .map_err(|_| Error::DhtIsShutdown)?;

        while let Ok(response) = receiver.recv() {
            match response {
                ResolveResponse::SignedPacket(signed_packet) => {
                    debug!("Resolve: resolved more recent signed_packet.");
                    return Ok(signed_packet);
                }
                ResolveResponse::NoMoreRecentValue(network_seq) => {
                    if let Some(ref mut cached) = cached_packet {
                        if network_seq == most_recent_known_seq.unwrap() {
                            debug!("Remote node has no more recent value, refreshing cached signed_packet.");

                            cached.refresh();
                            self.cache.put(target, cached.clone());

                            return Ok(cached.clone());
                        }
                    }
                }
            };
        }

        Err(Error::NotFound(public_key.to_string()))
    }

    /// Shutdown the actor thread loop.
    pub fn shutdown(&self) -> Result<()> {
        self.sender
            .send(ActorMessage::Shutdown)
            .map_err(|_| Error::DhtIsShutdown)?;

        Ok(())
    }
}

fn run(mut rpc: Rpc, _settings: DhtSettings, cache: PkarrCache, receiver: Receiver<ActorMessage>) {
    let mut server = mainline::server::Server::default();
    let mut senders: HashMap<Id, Vec<Sender<ResolveResponse>>> = HashMap::new();

    loop {
        // === Receive actor messages ===
        if let Ok(actor_message) = receiver.try_recv() {
            match actor_message {
                ActorMessage::Shutdown => {
                    drop(receiver);
                    break;
                }
                ActorMessage::Publish(signed_packet, sender) => {
                    let mutable_item: MutableItem = (&signed_packet).into();
                    let target = mutable_item.target();

                    cache.put(*target, signed_packet);

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
                ActorMessage::Resolve(target, sender, most_recent_known_seq) => {
                    if let Some(set) = senders.get_mut(&target) {
                        set.push(sender);
                    } else {
                        senders.insert(target, vec![sender]);
                    };

                    let request = messages::RequestTypeSpecific::GetValue(
                        messages::GetValueRequestArguments {
                            target,
                            seq: most_recent_known_seq,
                            // seq: None,
                            salt: None,
                        },
                    );

                    rpc.get(target, request, None)
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
                    let is_most_recent = cache
                        .get(target)
                        // Do not filter expired cached packets, because we care more that
                        // they are more recent.
                        .map_or(true, |cached| signed_packet.more_recent_than(&cached));

                    if is_most_recent {
                        // Save at cache, and then send empty notification.

                        cache.put(*target, signed_packet.clone());

                        if let Some(set) = senders.get_mut(target) {
                            for sender in set {
                                let _ = sender
                                    .send(ResolveResponse::SignedPacket(signed_packet.clone()));
                            }

                            // Removing all senders, because they don't need to wait anymore.
                            senders.remove(target);
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
                // Send the found sequence as a timestamp to the caller to decide what to do
                // with it.
                if let Some(set) = senders.get(target) {
                    for sender in set {
                        let _ = sender.send(ResolveResponse::NoMoreRecentValue(*seq));
                    }
                }

                // We shouldn't remove the senders yet, since the incoming seq might not be
                // good enough for the caller of `resolve()`.
                //
                // They will be gced when the query is exhausted.
            }
            Some(ReceivedFrom {
                from,
                message: ReceivedMessage::Request((transaction_id, request)),
            }) => {
                // TODO: investigate why; not handling a request causes a hang in tests?
                server.handle_request(&mut rpc, *from, *transaction_id, request);
            }
            _ => {}
        }
    }
}

enum ActorMessage {
    Publish(SignedPacket, Sender<mainline::Result<Id>>),
    Resolve(Id, Sender<ResolveResponse>, Option<i64>),
    Shutdown,
}

enum ResolveResponse {
    SignedPacket(SignedPacket),
    /// A node doesn't have anything more recent than what we have,
    /// returning the `seq` of what it has
    NoMoreRecentValue(i64),
}

#[cfg(test)]
mod tests {
    use mainline::Testnet;

    use super::*;
    use crate::{dns, Keypair, SignedPacket};

    #[test]
    fn publish_resolve() {
        let testnet = Testnet::new(10);

        let a = PkarrClient::builder()
            .bootstrap(&testnet.bootstrap)
            .server()
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

        let resolved = b.resolve(&keypair.public_key()).unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve(&keypair.public_key()).unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }
}
