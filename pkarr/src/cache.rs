use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use mainline::Id;

use crate::SignedPacket;

/// A wrapper around `LruCache`. This struct is thread safe, doesn't return any references to any
/// elements inside.
#[derive(Debug, Clone)]
pub struct PkarrCache {
    inner: Arc<Mutex<LruCache<Id, SignedPacket>>>,
}

impl PkarrCache {
    /// Creats a new `LRU` cache that holds at most `cap` items.
    pub fn new(cap: NonZeroUsize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LruCache::new(cap))),
        }
    }

    /// Puts a key-value pair into cache. If the key already exists in the cache,
    /// then it updates the key's value.
    pub fn put(&self, key: &Id, value: &SignedPacket) {
        let mut lock = self.inner.lock().unwrap();

        match lock.get_mut(key) {
            Some(existing) => {
                if existing.as_bytes() == value.as_bytes() {
                    // just refresh the last_seen
                    existing.set_last_seen(*value.last_seen())
                } else {
                    lock.put(*key, value.clone());
                }
            }
            None => {
                lock.put(*key, value.clone());
            }
        }
    }

    /// Returns the value of the key in the cache or None if it is not present in the cache.
    /// Moves the key to the head of the LRU list if it exists.
    pub fn get(&self, key: &Id) -> Option<SignedPacket> {
        self.inner.lock().unwrap().get(key).cloned()
    }
}
