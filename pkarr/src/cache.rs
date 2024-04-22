use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use crate::{PublicKey, SignedPacket};

/// A wrapper around `LruCache`. This struct is thread safe, doesn't return any references to any
/// elements inside.
#[derive(Debug, Clone)]
pub struct PkarrCache {
    inner: Arc<Mutex<LruCache<PublicKey, SignedPacket>>>,
}

impl PkarrCache {
    /// Creats a new `LRU` cache that holds at most `cap` items.
    pub fn new(cap: NonZeroUsize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LruCache::new(cap))),
        }
    }

    /// Puts [SignedPacket], if a version of the  packet already exists,
    /// and it has the same [SignedPacket::as_bytes], then only [SignedPacket::last_seen] will be
    /// updated, otherwise the input will be cloned.
    pub fn put(&self, signed_packet: &SignedPacket) {
        let mut lock = self.inner.lock().unwrap();

        let public_key = signed_packet.public_key();

        match lock.get_mut(&public_key) {
            Some(existing) => {
                if existing.as_bytes() == signed_packet.as_bytes() {
                    // just refresh the last_seen
                    existing.set_last_seen(signed_packet.last_seen())
                } else {
                    lock.put(public_key, signed_packet.clone());
                }
            }
            None => {
                lock.put(public_key, signed_packet.clone());
            }
        }
    }

    /// Returns the [SignedPacket] for the public_key in the cache or None if it is not present in the cache.
    /// Moves the key to the head of the LRU list if it exists.
    pub fn get(&self, public_key: &PublicKey) -> Option<SignedPacket> {
        self.inner.lock().unwrap().get(public_key).cloned()
    }
}
