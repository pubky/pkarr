use flume::{Receiver, Sender};
use lru::LruCache;
use mainline::{
    dht::DhtSettings,
    rpc::{
        messages, QueryResponse, QueryResponseSpecific, ReceivedFrom, ReceivedMessage, Response,
        Rpc, RpcTickReport,
    },
    Id, MutableItem,
};
use std::{
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    thread,
    time::{Duration, Instant},
};

use crate::{PublicKey, Result, SignedPacket};

const DEFAULT_CACHE_SIZE: usize = 1000;
/// Default minimum TTL 30 seconds
const DEFAULT_MINIMUM_TTL: u64 = 30;

#[derive(Debug, Clone)]
pub struct Settings {
    dht: DhtSettings,
    cache_size: usize,
    // TODO: add cusotmization of minimum ttl and maximum ttl
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            dht: DhtSettings::default(),
            cache_size: DEFAULT_CACHE_SIZE,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PkarrClientBuilder {
    settings: Settings,
}

impl PkarrClientBuilder {
    pub fn server(mut self) -> Self {
        self.settings.dht.read_only = false;
        self
    }

    pub fn bootstrap(mut self, bootstrap: &[String]) -> Self {
        self.settings.dht.bootstrap = Some(bootstrap.to_owned());
        self
    }

    pub fn cache_size(mut self, cache_size: usize) -> Self {
        self.settings.cache_size = cache_size;
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
}

impl PkarrClient {
    pub fn new(settings: Settings) -> Result<PkarrClient> {
        let (sender, receiver) = flume::bounded(32);

        let mut rpc = Rpc::new()?.with_read_only(settings.dht.read_only);

        if let Some(bootstrap) = settings.dht.bootstrap {
            rpc = rpc.with_bootstrap(bootstrap);
        }

        if let Some(port) = settings.dht.port {
            rpc = rpc.with_port(port)?;
        }

        let state = State {
            rpc,
            resolve_senders: HashMap::new(),
            cache: LruCache::new(NonZeroUsize::new(DEFAULT_CACHE_SIZE).unwrap()),
        };

        thread::spawn(move || run(state, receiver));

        Ok(PkarrClient { sender })
    }

    /// Returns a builder to edit settings before creating PkarrClient.
    pub fn builder() -> PkarrClientBuilder {
        PkarrClientBuilder::default()
    }

    pub fn publish(&self, signed_packet: &SignedPacket) -> mainline::Result<Id> {
        let (sender, receiver) = flume::unbounded::<mainline::Result<Id>>();

        let _ = self
            .sender
            .send(ActorMessage::Publish(signed_packet.to_owned(), sender));

        receiver.recv()?
    }

    /// Returns the first valid [SignedPacket] available from cache, or relays, or the Dht.
    pub fn resolve(&self, public_key: &PublicKey) -> Result<SignedPacket> {
        let (sender, receiver) = flume::unbounded::<SignedPacket>();

        let _ = self
            .sender
            .send(ActorMessage::Resolve(public_key.clone(), sender));

        let first = receiver.recv()?;

        Ok(first)
    }

    /// Shutdown the actor thread loop.
    pub fn shutdown(&self) -> Result<()> {
        let (sender, receiver) = flume::bounded::<()>(1);

        self.sender.send(ActorMessage::Shutdown(sender))?;

        Ok(receiver.recv()?)
    }
}

struct CachedPacket {
    expires_at: Instant,
    signed_packet: SignedPacket,
}

impl CachedPacket {
    fn new(signed_packet: &SignedPacket, ttl: u64) -> Self {
        Self {
            // TODO: set the ttl from signed_packet records' ttls
            expires_at: Instant::now() + Duration::from_secs(ttl),
            signed_packet: signed_packet.to_owned(),
        }
    }

    fn is_fresh(&self) -> bool {
        let now = Instant::now();
        now < self.expires_at
    }
}

struct State {
    rpc: Rpc,
    resolve_senders: HashMap<Id, Vec<Sender<SignedPacket>>>,
    cache: LruCache<PublicKey, CachedPacket>,
}

fn run(mut state: State, receiver: Receiver<ActorMessage>) {
    let mut server = mainline::server::Server::default();

    loop {
        dht_tick(&mut state, &mut server);

        // === Receive actor messages ===

        if let Ok(actor_message) = receiver.try_recv() {
            match actor_message {
                ActorMessage::Shutdown(sender) => {
                    let _ = sender.send(());
                    drop(receiver);
                    break;
                }
                ActorMessage::Publish(signed_packet, sender) => {
                    let mutable_item: MutableItem = (&signed_packet).into();
                    let target = mutable_item.target();

                    state.cache.put(
                        signed_packet.public_key().to_owned(),
                        CachedPacket::new(&signed_packet, DEFAULT_MINIMUM_TTL),
                    );

                    let request = messages::PutRequestSpecific::PutMutable(
                        messages::PutMutableRequestArguments {
                            target: target.to_owned(),
                            v: mutable_item.value().to_vec(),
                            k: mutable_item.key().to_vec(),
                            seq: *mutable_item.seq(),
                            sig: mutable_item.signature().to_vec(),
                            salt: None,
                            // TODO: figure out how or should we use Cas
                            cas: None,
                        },
                    );

                    state.rpc.put(*target, request, Some(sender))
                }
                ActorMessage::Resolve(public_key, sender) => {
                    let target = MutableItem::target_from_key(public_key.as_bytes(), &None);

                    if let Some(cached) = state.cache.get(&public_key) {
                        if cached.is_fresh() {
                            sender.send(cached.signed_packet.clone());

                            return;
                        }

                        let timestamp = cached.signed_packet.timestamp();

                        dht_request(&mut state, target, Some(timestamp));
                    } else {
                        if let Some(set) = state.resolve_senders.get_mut(&target) {
                            set.push(sender);
                        } else {
                            state.resolve_senders.insert(target, vec![sender]);
                        };

                        dht_request(&mut state, target, None);
                    };
                }
            }
        }
    }
}

/// Make a dht request if the Dht is enabled by feature gate and builder settings
fn dht_request(state: &mut State, target: Id, most_recent_known_timestamp: Option<u64>) {
    let request = messages::RequestTypeSpecific::GetValue(messages::GetValueRequestArguments {
        target,
        seq: most_recent_known_timestamp.map(|t| t as i64),
        salt: None,
    });

    state.rpc.get(target, request, None)
}

/// Perform the Dht tick, and receive incoming mutable items, and cache the [SignedPacket]
fn dht_tick(state: &mut State, server: &mut mainline::server::Server) {
    let report = state.rpc.tick();

    // === Drop senders to done queries ===
    for id in &report.done_get_queries {
        state.resolve_senders.remove(id);
    }

    match &report.received_from {
        Some(ReceivedFrom {
            message:
                ReceivedMessage::QueryResponse(QueryResponse {
                    target,
                    response: QueryResponseSpecific::Value(Response::Mutable(mutable_item)),
                }),
            ..
        }) => {
            if let Ok(signed_packet) = SignedPacket::try_from(mutable_item) {
                let timestamp = signed_packet.timestamp();
                let public_key = signed_packet.public_key();

                let is_most_recent = state
                    .cache
                    .get(&public_key)
                    .filter(|cached| cached.is_fresh())
                    .map_or(true, |cached| {
                        signed_packet.more_recent_than(&cached.signed_packet)
                    });

                if is_most_recent {
                    state.cache.put(
                        public_key.clone(),
                        // TODO: compare the packet TTLs and the DEFAULT_MINIMUM_TTL
                        CachedPacket::new(&signed_packet, DEFAULT_MINIMUM_TTL),
                    );

                    if let Some(set) = state.resolve_senders.get_mut(target) {
                        while let Some(sender) = set.pop() {
                            sender.send(signed_packet.to_owned());
                        }

                        state.resolve_senders.remove(target);
                    }
                }
            }
        }
        Some(ReceivedFrom {
            from,
            message: ReceivedMessage::Request((transaction_id, request)),
        }) => {
            // TODO: investigate why; not handling a request causes a hang in tests?
            server.handle_request(&mut state.rpc, *from, *transaction_id, request);
        }
        _ => {}
    }
}

pub enum ActorMessage {
    Publish(SignedPacket, Sender<mainline::Result<Id>>),
    Resolve(PublicKey, Sender<SignedPacket>),
    Shutdown(Sender<()>),
}

#[cfg(test)]
mod tests {
    use mainline::Testnet;

    use super::*;
    use crate::{dns, Keypair, SignedPacket};

    #[test]
    fn publish_resolve() {
        let testnet = Testnet::new(10);

        let mut a = PkarrClient::builder()
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

        let x = a.publish(&signed_packet);

        let mut b = PkarrClient::builder()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();
        let resolved = b.resolve(&keypair.public_key()).unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }
}
