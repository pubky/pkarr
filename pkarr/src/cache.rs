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

    /// Returns the number of key-value pairs that are currently in the cache.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Returns true if the cache is empty and false otherwise.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }

    /// Puts a key-value pair into cache. If the key already exists in the cache,
    /// then it updates the key's value.
    pub fn put(&self, key: Id, value: SignedPacket) {
        self.inner.lock().unwrap().put(key, value);
    }

    /// Returns the value of the key in the cache or None if it is not present in the cache.
    /// Moves the key to the head of the LRU list if it exists.
    pub fn get(&self, key: &Id) -> Option<SignedPacket> {
        self.inner.lock().unwrap().get(key).cloned()
    }

    /// Returns the lock over underlying LRU cache.
    pub fn lock(&self) -> std::sync::MutexGuard<LruCache<Id, SignedPacket>> {
        self.inner.lock().unwrap()
    }
}
