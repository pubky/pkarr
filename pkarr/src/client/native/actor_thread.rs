use std::collections::HashMap;

use flume::{Receiver, Sender};
use mainline::rpc::Rpc;
use pubky_timestamp::Timestamp;
use tracing::debug;

use crate::{
    client::native::{dht::DhtClient, BuildError},
    Cache, CacheKey, Config, PublicKey, SignedPacket,
};

#[cfg(feature = "relays")]
use super::relays::RelaysClient;
use super::PublishError;

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
                    #[cfg(not(feature = "relays"))]
                    {
                        let _ = sender.send(Err(BuildError::MainlineUdpSocket(err)));
                    }
                    #[cfg(feature = "relays")]
                    if config.relays.as_ref().map(|r| r.is_empty()).unwrap_or(true) {
                        let _ = sender.send(Err(BuildError::MainlineUdpSocket(err)));
                    }
                }

                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "dht"))]
    let dht_client: Option<()> = None;

    #[cfg(feature = "relays")]
    let mut relays_client = config.relays.map(|r| {
        RelaysClient::new(
            r.into(),
            cache.clone(),
            config.request_timeout,
            config.relays_runtime,
        )
    });

    #[cfg(not(feature = "relays"))]
    let relays_client: Option<()> = None;

    loop {
        // === Receive actor messages ===
        if let Ok(actor_message) = receiver.try_recv() {
            match actor_message {
                ActorMessage::Shutdown(sender) => {
                    drop(receiver);
                    let _ = sender.send(());
                    break;
                }
                ActorMessage::Publish(signed_packet, sender, cas) => {
                    #[cfg(feature = "dht")]
                    {
                        if let Some(ref mut dht_client) = dht_client {
                            dht_client.publish(&signed_packet, sender.clone(), cas);
                        }
                    }

                    #[cfg(feature = "relays")]
                    {
                        if let Some(relays) = &relays_client {
                            relays.publish(&signed_packet, sender, cas);
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

                    let _ = sender.send(Info { dht_info });
                }
                #[cfg(feature = "dht")]
                ActorMessage::ToBootstrap(sender) => {
                    let bootstrap = dht_client
                        .as_ref()
                        .map(|dht_client| dht_client.rpc.routing_table().to_bootstrap());

                    let _ = sender.send(bootstrap);
                }
                ActorMessage::Check(sender) => {
                    let _ = sender.send(if dht_client.is_none() && relays_client.is_none() {
                        Err(BuildError::NoNetwork)
                    } else {
                        Ok(())
                    });
                }
            }
        }

        #[cfg(feature = "dht")]
        if let Some(ref mut dht_client) = dht_client {
            dht_client.tick(relays_client.is_none());
        }
    }

    debug!("Client main loop terminated");
}

pub enum ActorMessage {
    Publish(
        SignedPacket,
        Sender<Result<(), PublishError>>,
        Option<Timestamp>,
    ),
    Resolve(PublicKey, Sender<SignedPacket>),
    Shutdown(Sender<()>),
    Info(Sender<Info>),
    Check(Sender<Result<(), BuildError>>),
    #[cfg(feature = "dht")]
    ToBootstrap(Sender<Option<Vec<String>>>),
}

pub struct Info {
    #[cfg(feature = "dht")]
    dht_info: Option<mainline::rpc::Info>,
}

impl Info {
    #[cfg(feature = "dht")]
    /// Returns [mainline::rpc::Info] if the dht client is enabled.
    pub fn dht_info(&self) -> Option<&mainline::rpc::Info> {
        self.dht_info.as_ref()
    }
}
