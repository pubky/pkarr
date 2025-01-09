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
    /// Puts [SignedPacket] into cache.
    fn put(&self, key: &PkarrCacheKey, signed_packet: &SignedPacket);
    /// Reads [SignedPacket] from cache, while moving it to the head of the LRU list.
    fn get(&self, key: &PkarrCacheKey) -> Option<SignedPacket>;
    /// Reads [SignedPacket] from cache, without changing the LRU list.
    ///
    /// Used for internal reads that are not initiated by the user directly,
    /// like comparing an received signed packet with existing one.
    ///
    /// Useful to implement differently from [PkarrCache::get], if you are implementing
    /// persistent cache where writes are slower than reads.
    ///
    /// Otherwise it will just use [PkarrCache::get].
    fn get_read_only(&self, key: &PkarrCacheKey) -> Option<SignedPacket> {
        self.get(key)
    }
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
                if existing == signed_packet {
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

    fn get(&self, key: &PkarrCacheKey) -> Option<SignedPacket> {
        self.inner.lock().unwrap().get(key).cloned()
    }
}
