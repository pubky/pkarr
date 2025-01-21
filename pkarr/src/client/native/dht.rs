//! A DHT client for the native client.

use std::{collections::HashMap, net::SocketAddrV4};

use flume::Sender;
use mainline::{
    rpc::{
        messages::{self, PutMutableRequestArguments},
        PutError, Response, Rpc,
    },
    Id,
};
use pubky_timestamp::Timestamp;
use tracing::debug;

use crate::{Cache, SignedPacket};

use super::PublishError;

pub struct DhtClient {
    pub rpc: Rpc,
    pub cache: Option<Box<dyn Cache>>,
    pub resolve_senders: HashMap<Id, Vec<Sender<SignedPacket>>>,
    pub publish_senders: HashMap<Id, Sender<Result<(), PublishError>>>,
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

        if let Err(put_error) = self.rpc.put(messages::PutRequestSpecific::PutMutable(
            put_mutable_request,
        )) {
            if let PutError::PutQueryIsInflight(_) = put_error {
                let _ = sender.send(Err(PublishError::PublishInflight));

                return;
            }

            handle_put_error(put_error);
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

        // === Drop senders to done queries ===
        for id in &report.done_get_queries {
            self.resolve_senders.remove(id);
        }

        for (id, error) in &report.done_put_queries {
            if let Some(sender) = self.publish_senders.remove(id) {
                if let Some(put_error) = error.to_owned() {
                    handle_put_error(put_error);
                } else {
                    let _ = sender.send(Ok(()));
                };
            };
        }
    }
}

fn handle_put_error(error: PutError) {
    match error {
        mainline::rpc::PutError::NoClosestNodes => {
            debug!("mainline failed to find closest nodes (usually means UDP and or Mainline packets are firewalled)");
        }
        mainline::rpc::PutError::PutQueryIsInflight(_) => {
            unreachable!("Should not make two publish queries at the same time!");
        }
        mainline::rpc::PutError::ErrorResponse(error) => {
            debug!(?error, "mainline nodes responded with error for PUT query");
        }
    }
}

#[cfg(test)]
mod tests {
    //! Dht only tests

    use std::{thread, time::Duration};

    use mainline::Testnet;

    use super::super::*;
    use crate::{Keypair, SignedPacket};

    #[test]
    fn shutdown_sync() {
        let testnet = Testnet::new(3).unwrap();

        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        client.shutdown_sync();

        assert!(client.info().is_err());
    }

    #[test]
    fn publish_resolve_sync() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish_sync(&signed_packet).unwrap();

        let b = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let resolved = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }

    #[test]
    fn thread_safe_sync() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish_sync(&signed_packet).unwrap();

        let b = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

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

        let mut a = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        a.shutdown().await;

        assert!(a.info().is_err());
    }

    #[tokio::test]
    async fn publish_resolve() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }

    #[tokio::test]
    async fn thread_safe() {
        let testnet = Testnet::new(10).unwrap();

        let a = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

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
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
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
            .unwrap()
            .put(&keypair.public_key().into(), &signed_packet);

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert_eq!(resolved, Some(signed_packet));
    }

    #[tokio::test]
    async fn ttl_0_test() {
        let testnet = Testnet::new(10).unwrap();

        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .maximum_ttl(0)
            .build()
            .unwrap();

        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();

        client.publish(&signed_packet).await.unwrap();

        // First Call
        let resolved = client
            .resolve(&signed_packet.public_key())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved.encoded_packet(), signed_packet.encoded_packet());

        thread::sleep(Duration::from_millis(10));

        let second = client
            .resolve(&signed_packet.public_key())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second.encoded_packet(), signed_packet.encoded_packet());
    }

    #[tokio::test]
    async fn not_found() {
        let testnet = Testnet::new(10).unwrap();

        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert_eq!(resolved, None);
    }

    #[tokio::test]
    async fn inflight() {
        let testnet = Testnet::new(10).unwrap();

        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .request_timeout(Duration::from_millis(100))
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        let clone = client.clone();
        let packet = signed_packet.clone();

        let handle = tokio::spawn(async move {
            let result = clone.publish(&packet).await;

            assert!(matches!(result, Err(PublishError::PublishInflight)));
        });

        client.publish(&signed_packet).await.unwrap();

        handle.await.unwrap()
    }
}
