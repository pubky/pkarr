use lru::LruCache;
use std::num::NonZeroUsize;

use crate::keys::PublicKey;
use crate::signed_packet::SignedPacket;
use crate::{Error, Result};

use mainline::{
    common::MutableItem, Dht, DhtSettings, GetMutableResponse, Response, StoreQueryMetdata,
};

pub const DEFAULT_PKARR_RELAY: &str = "https://relay.pkarr.org";
pub const DEFAULT_CACHE_SIZE: usize = 1000;

#[derive(Default)]
pub struct PkarrClientBuilder {
    relays: Vec<String>,
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

#[derive(Clone, Debug)]
/// Main client for publishing and resolving [SignedPacket]s.
pub struct PkarrClient {
    cache: LruCache<PublicKey, SignedPacket>,
    dht: Dht,
}

impl PkarrClient {
    pub fn builder() -> PkarrClientBuilder {
        PkarrClientBuilder::default()
    }

    pub fn cache(&self) -> &LruCache<PublicKey, SignedPacket> {
        &self.cache
    }

    /// Publish a [SignedPacket].
    pub fn publish(&self, signed_packet: &SignedPacket) -> Result<StoreQueryMetdata> {
        let item: MutableItem = signed_packet.into();

        self.dht.put_mutable(item).map_err(Error::MainlineError)
    }

    /// The recommended way to resolve [SignedPacket]s.
    ///
    /// Returns the first value from [resolve_stream](PkarrClient::resolve_stream).
    pub fn resolve(&mut self, public_key: PublicKey) -> Option<SignedPacket> {
        self.resolve_stream(public_key).recv().ok()
    }

    /// Returns a stream of [SignedPacket]s as they are resolved local cache, relays, or DHT nodes.
    pub fn resolve_stream(&mut self, public_key: PublicKey) -> flume::Receiver<SignedPacket> {
        let (sender, receiver) = flume::unbounded::<SignedPacket>();

        if let Some(cached) = self.cache.get(&public_key) {
            let _ = sender.send(cached.to_owned());
        };

        // TODO: Backoff from querying the DHT too often!
        let mut response = self.dht.get_mutable(public_key.as_bytes(), None);

        for res in &mut response {
            let signed_packet: Result<SignedPacket> = res.item.try_into();

            if let Ok(signed_packet) = signed_packet {
                self.cache.put(public_key.clone(), signed_packet.clone());

                let _ = sender.send(signed_packet);
            };
        }

        receiver
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
    fn testnet() {
        let testnet = Testnet::new(10);

        let a = PkarrClient::builder().bootstrap(&testnet.bootstrap).build();

        let keypair = Keypair::random();

        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::TXT("bar".try_into().unwrap()),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

        let x = a.publish(&signed_packet);

        let mut b = PkarrClient::builder().bootstrap(&testnet.bootstrap).build();
        let resolved = b.resolve(keypair.public_key()).unwrap();

        assert_eq!(resolved.to_bytes(), signed_packet.to_bytes());

        assert_eq!(b.cache().len(), 1);
    }
}
