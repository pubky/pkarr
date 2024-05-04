//! Async version of [PkarrClient]

use mainline::{Id, MutableItem};
use std::net::SocketAddr;
use tracing::{debug, instrument};

use super::{
    cache::PkarrCache,
    client::{ActorMessage, PkarrClient},
};
use crate::{Error, PublicKey, Result, SignedPacket};

#[derive(Clone, Debug)]
/// Async version of [PkarrClient]
pub struct PkarrClientAsync(PkarrClient);

impl PkarrClient {
    /// Returns [PkarrClientAsync]
    pub fn as_async(self) -> PkarrClientAsync {
        PkarrClientAsync(self)
    }
}

impl PkarrClientAsync {
    // === Getters ===

    /// Returns the local address of the udp socket this node is listening on.
    ///
    /// Returns `None` if the node is shutdown
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.0.address
    }

    /// Returns a reference to the internal cache.
    pub fn cache(&self) -> &dyn PkarrCache {
        self.0.cache.as_ref()
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
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<()> {
        let mutable_item: MutableItem = (signed_packet).into();

        if let Some(current) = self.0.cache.get(mutable_item.target()) {
            if current.timestamp() > signed_packet.timestamp() {
                return Err(Error::NotMostRecent);
            }
        };

        self.0.cache.put(mutable_item.target(), signed_packet);

        let (sender, receiver) = flume::bounded::<mainline::Result<Id>>(1);

        self.0
            .sender
            .send(ActorMessage::Publish(mutable_item, sender))
            .map_err(|_| Error::DhtIsShutdown)?;

        match receiver.recv_async().await {
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

    /// Returns the first valid [SignedPacket] available from cache, or the Dht.
    ///
    /// If the Dht was called, in the background, it continues receiving responses
    /// and updating the cache.
    ///
    /// # Errors
    /// - Returns a [Error::DhtIsShutdown] if [PkarrClient::shutdown] was called, or
    /// the loop in the actor thread is stopped for any reason (like thread panic).
    #[instrument(skip(self))]
    pub async fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>> {
        let target = MutableItem::target_from_key(public_key.as_bytes(), &None);

        let cached_packet = self.0.cache.get(&target);

        if let Some(ref cached) = cached_packet {
            let expires_in = cached.expires_in(self.0.minimum_ttl, self.0.maximum_ttl);

            if expires_in > 0 {
                debug!(expires_in, "Have fresh signed_packet in cache.");
                return Ok(Some(cached.clone()));
            }

            debug!(expires_in, "Have expired signed_packet in cache.");
        } else {
            debug!("Cache mess");
        }

        // Cache miss

        let (sender, receiver) = flume::bounded::<SignedPacket>(1);

        debug!("Cache miss, asking the network for a fresh signed_packet");

        self.0
            .sender
            .send(ActorMessage::Resolve(
                target,
                sender,
                // Sending the `timestamp` of the known cache, help save some bandwith,
                // since remote nodes will not send the encoded packet if they don't know
                // any more recent versions.
                cached_packet.as_ref().map(|cached| cached.timestamp()),
            ))
            .map_err(|_| Error::DhtIsShutdown)?;

        Ok(receiver.recv_async().await.ok())
    }

    /// Shutdown the actor thread loop.
    pub async fn shutdown(&mut self) -> Result<()> {
        let (sender, receiver) = flume::bounded(1);

        self.0
            .sender
            .send(ActorMessage::Shutdown(sender))
            .map_err(|_| Error::DhtIsShutdown)?;

        receiver.recv_async().await?;

        self.0.address = None;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use mainline::{dht::DhtSettings, Testnet};

    use super::*;
    use crate::{dns, Keypair, SignedPacket};

    #[test]
    fn shutdown() {
        async fn test() {
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

        futures::executor::block_on(test());
    }

    #[test]
    fn publish_resolve() {
        async fn test() {
            let testnet = Testnet::new(10);

            let a = PkarrClient::builder()
                .dht_settings(DhtSettings {
                    bootstrap: Some(testnet.bootstrap.clone()),
                    request_timeout: None,
                    server: None,
                    port: None,
                })
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
                .dht_settings(DhtSettings {
                    bootstrap: Some(testnet.bootstrap),
                    request_timeout: None,
                    server: None,
                    port: None,
                })
                .build()
                .unwrap();

            let resolved = b.resolve(&keypair.public_key()).unwrap().unwrap();
            assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

            let from_cache = b.resolve(&keypair.public_key()).unwrap().unwrap();
            assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
            assert_eq!(from_cache.last_seen(), resolved.last_seen());
        }

        futures::executor::block_on(test());
    }
}
