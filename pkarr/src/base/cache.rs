//! Trait and inmemory implementation of [Cache]

use dyn_clone::DynClone;
use lru::LruCache;
use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
use mainline::MutableItem;

use crate::SignedPacket;

/// The sha1 hash of the [crate::PublicKey] used as the key in [Cache].
pub type CacheKey = [u8; 20];

#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
impl From<&crate::PublicKey> for CacheKey {
    fn from(public_key: &crate::PublicKey) -> CacheKey {
        MutableItem::target_from_key(public_key.as_bytes(), &None).into()
    }
}

#[cfg(any(target_arch = "wasm32", all(not(feature = "dht"), feature = "relay")))]
impl From<&crate::PublicKey> for CacheKey {
    fn from(public_key: &crate::PublicKey) -> CacheKey {
        let mut encoded = vec![];

        encoded.extend(public_key.as_bytes());

        let mut hasher = sha1_smol::Sha1::new();
        hasher.update(&encoded);
        hasher.digest().bytes()
    }
}

#[cfg(any(target_arch = "wasm32", feature = "dht", feature = "relay"))]
impl From<crate::PublicKey> for CacheKey {
    fn from(value: crate::PublicKey) -> Self {
        (&value).into()
    }
}

pub trait Cache: Debug + Send + Sync + DynClone {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Puts [SignedPacket] into cache.
    fn put(&self, key: &CacheKey, signed_packet: &SignedPacket);
    /// Reads [SignedPacket] from cache, while moving it to the head of the LRU list.
    fn get(&self, key: &CacheKey) -> Option<SignedPacket>;
    /// Reads [SignedPacket] from cache, without changing the LRU list.
    ///
    /// Used for internal reads that are not initiated by the user directly,
    /// like comparing an received signed packet with existing one.
    ///
    /// Useful to implement differently from [Cache::get], if you are implementing
    /// persistent cache where writes are slower than reads.
    ///
    /// Otherwise it will just use [Cache::get].
    fn get_read_only(&self, key: &CacheKey) -> Option<SignedPacket> {
        self.get(key)
    }
}

dyn_clone::clone_trait_object!(Cache);

/// A thread safe wrapper around [lru::LruCache]
#[derive(Debug, Clone)]
pub struct InMemoryCache {
    inner: Arc<Mutex<LruCache<CacheKey, SignedPacket>>>,
}

impl InMemoryCache {
    /// Creats a new `LRU` cache that holds at most `cap` items.
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LruCache::new(capacity))),
        }
    }
}

impl Cache for InMemoryCache {
    fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Puts [SignedPacket], if a version of the  packet already exists,
    /// and it has the same [SignedPacket::as_bytes], then only [SignedPacket::last_seen] will be
    /// updated, otherwise the input will be cloned.
    fn put(&self, key: &CacheKey, signed_packet: &SignedPacket) {
        let mut lock = self.inner.lock().unwrap();

        match lock.get_mut(key) {
            Some(existing) => {
                if existing.as_bytes() == signed_packet.as_bytes() {
                    // just refresh the last_seen
                    existing.set_last_seen(signed_packet.last_seen())
                } else {
                    lock.put(*key, signed_packet.clone());
                }
            }
            None => {
                lock.put(*key, signed_packet.clone());
            }
        }
    }

    fn get(&self, key: &CacheKey) -> Option<SignedPacket> {
        self.inner.lock().unwrap().get(key).cloned()
    }
}
