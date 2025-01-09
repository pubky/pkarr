//! A DHT client for the native client.

use std::{collections::HashMap, net::SocketAddrV4};

use flume::Sender;
use mainline::{
    rpc::{messages, Response, Rpc},
    Id,
};
use tracing::debug;

use crate::{Cache, SignedPacket};

pub struct DhtClient {
    pub rpc: Rpc,
    pub cache: Option<Box<dyn Cache>>,
    pub resolve_senders: HashMap<Id, Vec<Sender<SignedPacket>>>,
    pub publish_senders: HashMap<Id, Sender<Result<(), ()>>>,
    pub resolvers: Option<Box<[SocketAddrV4]>>,
}

impl DhtClient {
    pub fn publish(&mut self, signed_packet: &SignedPacket, sender: Sender<Result<(), ()>>) {
        let mutable_item = mainline::MutableItem::from(signed_packet);
        let target = *mutable_item.target();

        if let Err(_put_error) = self.rpc.put(messages::PutRequestSpecific::PutMutable(
            mutable_item.into(),
        )) {
            // TODO: do better
            let _ = sender.send(Err(()));
        } else {
            self.publish_senders.insert(target, sender);
        };
    }

    pub fn resolve(
        &mut self,
        target: Id,
        most_recent_known_timestamp: Option<u64>,
        sender: Sender<SignedPacket>,
    ) {
        if let Some(senders) = self.resolve_senders.get_mut(&target) {
            senders.push(sender);
        } else {
            self.resolve_senders.insert(target, vec![sender]);
        };

        if let Some(responses) = self.rpc.get(
            messages::RequestTypeSpecific::GetValue(messages::GetValueRequestArguments {
                target,
                seq: most_recent_known_timestamp.map(|t| t as i64),
                salt: None,
            }),
            self.resolvers.as_deref(),
        ) {
            for response in responses {
                if let Response::Mutable(mutable_item) = response {
                    if let Ok(signed_packet) = SignedPacket::try_from(mutable_item) {
                        if let Some(senders) = self.resolve_senders.get(&target) {
                            for sender in senders {
                                let _ = sender.send(signed_packet.clone());
                            }
                        }
                    }
                }
            }
        };
    }

    pub fn tick(&mut self) {
        let report = self.rpc.tick();

        // === Receive and handle incoming mutable item from the DHT ===

        if let Some((target, Response::Mutable(mutable_item))) = &report.query_response {
            if let Ok(signed_packet) = &SignedPacket::try_from(mutable_item) {
                let new_packet = if let Some(ref cached) = self
                    .cache
                    .as_ref()
                    .and_then(|cache| cache.get_read_only(target.as_bytes()))
                {
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
                    if let Some(cache) = &self.cache {
                        cache.put(target.as_bytes(), packet)
                    };

                    if let Some(senders) = self.resolve_senders.get(target) {
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
            self.resolve_senders.remove(id);
        }

        for (id, error) in &report.done_put_queries {
            if let Some(sender) = self.publish_senders.remove(id) {
                let _ = sender.send(if let Some(_error) = error.to_owned() {
                    Err(())
                } else {
                    Ok(())
                });
            };
        }
    }
}
