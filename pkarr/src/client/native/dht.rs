//! A DHT client for the native client.

use std::{collections::HashMap, net::SocketAddrV4};

use flume::Sender;
use mainline::{
    errors::PutQueryError,
    rpc::{
        messages::{self, PutMutableRequestArguments},
        GetRequestSpecific, PutError, Response, Rpc,
    },
    Id,
};
use pubky_timestamp::Timestamp;
use tracing::debug;

use crate::{errors::ConcurrencyError, Cache, SignedPacket};

use super::PublishError;

pub struct DhtClient {
    pub rpc: Rpc,
    pub cache: Option<Box<dyn Cache>>,
    pub resolve_senders: HashMap<Id, Vec<Sender<SignedPacket>>>,
    pub publish_senders: HashMap<Id, Vec<Sender<Result<(), PublishError>>>>,
    pub resolvers: Option<Box<[SocketAddrV4]>>,
}

impl DhtClient {
    pub fn publish(
        &mut self,
        signed_packet: &SignedPacket,
        sender: Sender<Result<(), PublishError>>,
        cas: Option<Timestamp>,
    ) {
        let mutable_item = mainline::MutableItem::from(signed_packet);
        let target = *mutable_item.target();

        let mut put_mutable_request: PutMutableRequestArguments = mutable_item.into();

        put_mutable_request.cas = cas.map(|cas| cas.as_u64() as i64);

        self.rpc
            .put(messages::PutRequestSpecific::PutMutable(
                put_mutable_request,
            ))
            .expect("should be infallible");

        let senders = self.publish_senders.entry(target).or_default();

        senders.push(sender)
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
            GetRequestSpecific::GetValue(messages::GetValueRequestArguments {
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

    // TODO: instead of no_relays check if there are no more pending relays responses.
    pub fn tick(&mut self, no_relays: bool) {
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

        // === Drop senders to done queries ===
        for id in &report.done_get_queries {
            self.resolve_senders.remove(id);
        }

        for (id, error) in &report.done_put_queries {
            if let Some(senders) = self.publish_senders.remove(id) {
                if let Some(put_error) = error.to_owned() {
                    debug!(?put_error, "mainline PUT mutable query failed");

                    if let Some(error) = match put_error {
                        // Return Concurrency errors regardless of the state of the
                        // publish query in the relays client.
                        PutError::Concurrency(error) => Some(match error {
                            mainline::errors::ConcurrencyError::ConflictRisk => {
                                PublishError::Concurrency(ConcurrencyError::ConflictRisk)
                            }
                            mainline::errors::ConcurrencyError::NotMostRecent => {
                                PublishError::Concurrency(ConcurrencyError::NotMostRecent)
                            }
                            mainline::errors::ConcurrencyError::CasFailed => {
                                PublishError::Concurrency(ConcurrencyError::CasFailed)
                            }
                        }),
                        PutError::Query(error) => {
                            if no_relays {
                                Some(match error {
                                    PutQueryError::Timeout => PublishError::Timeout,
                                    // TODO: Maybe return a unified error response (unexpected)?
                                    PutQueryError::NoClosestNodes => PublishError::NoClosestNodes,
                                    PutQueryError::ErrorResponse(error) => {
                                        PublishError::MainlineErrorResponse(error)
                                    }
                                })
                            } else {
                                None
                            }
                        }
                    } {
                        for sender in senders {
                            let _ = sender.send(Err(error.clone()));
                        }
                    }
                } else {
                    for sender in senders {
                        let _ = sender.send(Ok(()));
                    }
                };
            };
        }
    }
}
