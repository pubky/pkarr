//! Trait and inmemory implementation of [PkarrCache]

use dyn_clone::DynClone;
use lru::LruCache;
use mainline::Id;
use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use crate::SignedPacket;

/// The sha1 hash of the [crate::PublicKey] used as the key in [PkarrCache].
pub type PkarrCacheKey = Id;

pub trait PkarrCache: Debug + Send + Sync + DynClone {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn put(&self, key: &PkarrCacheKey, signed_packet: &SignedPacket);
    fn get(&self, key: &PkarrCacheKey) -> Option<SignedPacket>;
}

dyn_clone::clone_trait_object!(PkarrCache);

/// A thread safe wrapper around `LruCache`
#[derive(Debug, Clone)]
pub struct InMemoryPkarrCache {
    inner: Arc<Mutex<LruCache<Id, SignedPacket>>>,
}

impl InMemoryPkarrCache {
    /// Creats a new `LRU` cache that holds at most `cap` items.
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LruCache::new(capacity))),
        }
    }
}

impl PkarrCache for InMemoryPkarrCache {
    fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Puts [SignedPacket], if a version of the  packet already exists,
    /// and it has the same [SignedPacket::as_bytes], then only [SignedPacket::last_seen] will be
    /// updated, otherwise the input will be cloned.
    fn put(&self, target: &Id, signed_packet: &SignedPacket) {
        let mut lock = self.inner.lock().unwrap();

        match lock.get_mut(target) {
            Some(existing) => {
                if existing.as_bytes() == signed_packet.as_bytes() {
                    // just refresh the last_seen
                    existing.set_last_seen(signed_packet.last_seen())
                } else {
                    lock.put(*target, signed_packet.clone());
                }
            }
            None => {
                lock.put(*target, signed_packet.clone());
            }
        }
    }

    /// Returns the [SignedPacket] for the public_key in the cache or None if it is not present in the cache.
    /// Moves the key to the head of the LRU list if it exists.
    fn get(&self, key: &PkarrCacheKey) -> Option<SignedPacket> {
        self.inner.lock().unwrap().get(key).cloned()
    }
}
