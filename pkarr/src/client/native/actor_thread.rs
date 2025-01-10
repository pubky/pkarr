use std::collections::HashMap;

use flume::{Receiver, Sender};
use mainline::rpc::Rpc;
use tracing::debug;

use crate::{client::native::dht::DhtClient, Cache, CacheKey, Config, PublicKey, SignedPacket};

#[cfg(feature = "relays")]
use super::relays::RelaysClient;

pub fn actor_thread(
    receiver: Receiver<ActorMessage>,
    cache: Option<Box<dyn Cache>>,
    config: Config,
) {
    let minimum_ttl = config.minimum_ttl.min(config.maximum_ttl);
    let maximum_ttl = config.maximum_ttl.max(config.minimum_ttl);

    #[cfg(feature = "dht")]
    let mut dht_client = if !config.dht_config.bootstrap.is_empty() || config.resolvers.is_some() {
        match Rpc::new(config.dht_config) {
            Ok(rpc) => {
                let resolvers = config.resolvers.map(|r| r.into());

                Some(DhtClient {
                    rpc,
                    cache: cache.clone(),
                    resolvers,
                    resolve_senders: HashMap::new(),
                    publish_senders: HashMap::new(),
                })
            }
            Err(err) => {
                if let Ok(ActorMessage::Check(sender)) = receiver.try_recv() {
                    let _ = sender.send(Err(err));
                }

                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "dht"))]
    let mut dht_client: Option<()> = None;

    #[cfg(feature = "relays")]
    let mut relays_client = config
        .relays
        .map(|r| RelaysClient::new(r.into(), cache.clone()));

    #[cfg(not(feature = "relays"))]
    let mut relays_client: Option<()> = None;

    loop {
        // === Receive actor messages ===
        if let Ok(actor_message) = receiver.try_recv() {
            match actor_message {
                ActorMessage::Shutdown(sender) => {
                    drop(receiver);
                    let _ = sender.send(());
                    break;
                }
                ActorMessage::Publish(signed_packet, sender) => {
                    #[cfg(feature = "relays")]
                    {
                        if let Some(relays) = &relays_client {
                            relays.publish(&signed_packet, sender.clone());
                        }
                    }

                    #[cfg(feature = "dht")]
                    {
                        if let Some(ref mut dht_client) = dht_client {
                            // TODO: add an if condition in case dht is disabled.
                            dht_client.publish(&signed_packet, sender);
                        }
                    }
                }
                ActorMessage::Resolve(public_key, sender) => {
                    let cache_key: CacheKey = (&public_key).into();

                    let cached_packet = cache.as_ref().and_then(|cache| cache.get(&cache_key));

                    // Sending the `timestamp` of the known cache, help save some bandwith,
                    // since remote nodes will not send the encoded packet if they don't know
                    // any more recent versions.
                    let most_recent_known_timestamp = cached_packet
                        .as_ref()
                        .map(|cached| cached.timestamp().as_u64());

                    // Should query?
                    if cached_packet
                        .as_ref()
                        .map(|c| c.is_expired(minimum_ttl, maximum_ttl))
                        .unwrap_or(true)
                    {
                        debug!(
                            ?public_key,
                            "querying the DHT to hydrate our cache for later."
                        );

                        #[cfg(feature = "relays")]
                        if let Some(ref mut relays_client) = relays_client {
                            relays_client.resolve(&public_key, &cache_key, sender.clone());
                        }

                        #[cfg(feature = "dht")]
                        if let Some(ref mut dht_client) = dht_client {
                            dht_client.resolve(
                                cache_key.into(),
                                most_recent_known_timestamp,
                                sender.clone(),
                            );
                        }
                    }

                    if let Some(cached_packet) = cached_packet {
                        debug!(
                            public_key = ?cached_packet.public_key(),
                            "responding with cached packet even if expired"
                        );

                        // If the receiver was dropped.. no harm.
                        let _ = sender.send(cached_packet);
                    }
                }
                ActorMessage::Info(sender) => {
                    let dht_info = dht_client.as_ref().map(|dht_client| dht_client.rpc.info());

                    let _ = sender.send(Info {
                        // TODO: figure out Info with or without dht and or relay.
                        dht_info: dht_info.unwrap(),
                    });
                }
                ActorMessage::Check(sender) => {
                    let _ = sender.send(Ok(()));
                }
            }
        }

        #[cfg(feature = "dht")]
        if let Some(ref mut dht_client) = dht_client {
            dht_client.tick();
        }
    }

    debug!("Client main loop terminated");
}

pub enum ActorMessage {
    Publish(SignedPacket, Sender<Result<(), ()>>),
    Resolve(PublicKey, Sender<SignedPacket>),
    Shutdown(Sender<()>),
    Info(Sender<Info>),
    Check(Sender<Result<(), std::io::Error>>),
}

pub struct Info {
    #[cfg(feature = "dht")]
    dht_info: mainline::rpc::Info,
}

// TODO: add more infor like Mainline
impl Info {
    #[cfg(feature = "dht")]
    pub fn dht_info(&self) -> &mainline::rpc::Info {
        &self.dht_info
    }
}
