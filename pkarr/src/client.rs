use lru::LruCache;
use std::num::NonZeroUsize;

use crate::keys::PublicKey;
use crate::signed_packet::SignedPacket;
use crate::{Error, Result};

use mainline::{common::MutableItem, Dht, DhtSettings, StoreQueryMetdata};

pub const DEFAULT_CACHE_SIZE: usize = 1000;

pub struct PkarrClientBuilder {
    dht_settings: DhtSettings,
    cache_size: usize,
}

impl PkarrClientBuilder {
    pub fn bootstrap(mut self, bootstrap: &[String]) -> Self {
        self.dht_settings.bootstrap = Some(bootstrap.to_owned());
        self
    }

    pub fn cache_size(mut self, cache_size: usize) -> Self {
        self.cache_size = cache_size;
        self
    }

    pub fn build(self) -> PkarrClient {
        PkarrClient {
            cache: LruCache::new(NonZeroUsize::new(self.cache_size.max(1)).unwrap()),
            dht: Dht::new(self.dht_settings),
        }
    }
}

impl Default for PkarrClientBuilder {
    fn default() -> Self {
        Self {
            dht_settings: DhtSettings::default(),
            cache_size: DEFAULT_CACHE_SIZE,
        }
    }
}

#[derive(Clone, Debug)]
/// Main client for publishing and resolving [SignedPacket]s.
pub struct PkarrClient {
    // TODO: make the cache thread safe.
    cache: LruCache<PublicKey, SignedPacket>,
    dht: Dht,
}

impl PkarrClient {
    pub fn new() -> PkarrClient {
        PkarrClient::default()
    }

    pub fn builder() -> PkarrClientBuilder {
        PkarrClientBuilder::default()
    }

    pub fn cache(&self) -> &LruCache<PublicKey, SignedPacket> {
        &self.cache
    }

    /// Publish a [SignedPacket].
    pub fn publish(&mut self, signed_packet: &SignedPacket) -> Result<StoreQueryMetdata> {
        let item: MutableItem = signed_packet.into();

        // TODO: Backoff from unnecessarily republishing to the DHT
        // if we know that the packet has been seen recently on many nodes.
        // see get_mutable and put_mutable_to

        self.cache_put(signed_packet);

        // TODO: How to react if there are relays?
        self.dht.put_mutable(item).map_err(Error::MainlineError)
    }

    /// Returns a stream of [SignedPacket]s as they are resolved from local cache, relays, or DHT nodes.
    pub fn resolve(&mut self, public_key: PublicKey) -> flume::Receiver<SignedPacket> {
        let (sender, receiver) = flume::unbounded::<SignedPacket>();

        if let Some(cached) = self.cache.get(&public_key) {
            let _ = sender.send(cached.to_owned());
        };

        // // TODO: Backoff from querying the DHT too often!
        let mut response = self.dht.get_mutable(public_key.as_bytes(), None);

        for res in &mut response {
            let signed_packet: Result<SignedPacket> = res.item.try_into();

            if let Ok(signed_packet) = signed_packet {
                self.cache_put(&signed_packet);

                let _ = sender.send(signed_packet);
            };
        }

        receiver
    }

    /// Update the cache with the [SignedPacket] if it is [more recent than](SignedPacket::more_recent_than) the cached version.
    ///
    /// Returns `true` if the cache was updated, or `false` if not.
    pub fn cache_put(&mut self, signed_packet: &SignedPacket) -> bool {
        let public_key = signed_packet.public_key();

        if let Some(most_recent) = self.cache.get(public_key) {
            if most_recent.more_recent_than(signed_packet) {
                return false;
            }
        };

        self.cache
            .put(signed_packet.public_key().to_owned(), signed_packet.clone());

        true
    }

    /// Returns the cached [SignedPacket] for a `public_key` if it exists.
    pub fn cache_get(&mut self, public_key: &PublicKey) -> Option<&SignedPacket> {
        self.cache.get(public_key)
    }
}

impl Default for PkarrClient {
    fn default() -> Self {
        PkarrClient::builder().build()
    }
}

#[cfg(test)]
mod tests {
    use mainline::Testnet;

    use super::*;
    use crate::{dns, Keypair};

    #[test]
    fn publish_resolve() {
        let testnet = Testnet::new(10);

        let mut a = PkarrClient::builder().bootstrap(&testnet.bootstrap).build();

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

        let mut b = PkarrClient::builder().bootstrap(&testnet.bootstrap).build();
        let resolved = b.resolve(keypair.public_key()).recv().unwrap();

        assert_eq!(resolved.to_bytes(), signed_packet.to_bytes());

        assert_eq!(b.cache().len(), 1);
    }
}
