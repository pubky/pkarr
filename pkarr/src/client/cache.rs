//! Trait and inmemory implementation of [Cache]

use core::fmt;
use dyn_clone::DynClone;
use lru::LruCache;
use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::hash::{BuildHasher, Hash};
use std::io::{Read, Write};
use std::path::Path;

use crate::SignedPacket;

/// The sha1 hash of the [crate::PublicKey] used as the key in [Cache].
pub type CacheKey = [u8; 20];

impl From<&crate::PublicKey> for CacheKey {
    fn from(public_key: &crate::PublicKey) -> CacheKey {
        let mut encoded = vec![];

        encoded.extend(public_key.as_bytes());

        let mut hasher = sha1_smol::Sha1::new();
        hasher.update(&encoded);
        hasher.digest().bytes()
    }
}

impl From<crate::PublicKey> for CacheKey {
    fn from(value: crate::PublicKey) -> Self {
        (&value).into()
    }
}

/// A trait for a [SignedPacket]s cache for Pkarr [Client][crate::Client].
pub trait Cache: Debug + Send + Sync + DynClone {
    /// Returns the maximum capacity of [SignedPacket]s allowed in this cache.
    fn capacity(&self) -> usize {
        // backward compatibility
        0
    }
    /// Returns the number of [SignedPacket]s in this cache.
    fn len(&self) -> usize;
    /// Returns true if this cache is empty.
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
    inner: Arc<RwLock<LruCache<CacheKey, SignedPacket>>>,
}

impl InMemoryCache {
    /// Creates a new `LRU` cache that holds at most `cap` items.
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(LruCache::new(capacity))),
        }
    }
}

impl Cache for InMemoryCache {
    fn capacity(&self) -> usize {
        self.inner
            .read()
            .expect("InMemoryCache RwLock")
            .cap()
            .into()
    }

    fn len(&self) -> usize {
        self.inner.read().expect("InMemoryCache RwLock").len()
    }

    /// Puts [SignedPacket], if a version of the  packet already exists,
    /// and it has the same [SignedPacket::as_bytes], then only [SignedPacket::last_seen] will be
    /// updated, otherwise the input will be cloned.
    fn put(&self, key: &CacheKey, signed_packet: &SignedPacket) {
        let mut lock = self.inner.write().expect("InMemoryCache RwLock");

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
        self.inner
            .write()
            .expect("InMemoryCache RwLock")
            .get(key)
            .cloned()
    }

    fn get_read_only(&self, key: &CacheKey) -> Option<SignedPacket> {
        self.inner
            .read()
            .expect("InMemoryCache RwLock")
            .peek(key)
            .cloned()
    }
}

struct PersistentLruCache<K, V>
where
    K: Eq + std::hash::Hash + Serialize + for<'de> Deserialize<'de>,
    V: Clone + Serialize + for<'de> Deserialize<'de>
{
    cache: LruCache<K, V>,
    path: String,
}

impl<K: Hash + Eq + Serialize + for<'de> serde::Deserialize<'de>, V: Clone + Serialize + for<'de> serde::Deserialize<'de>> fmt::Debug for PersistentLruCache<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LruCache")
            .field("len", &self.cache.len())
            .field("cap", &self.cache.cap())
            .finish()
    }
}

impl<K, V> PersistentLruCache<K, V>
where
    K: Eq + std::hash::Hash + Serialize + for<'de> Deserialize<'de> + Clone,
    V: Clone + Serialize + for<'de> Deserialize<'de> + Clone
{
    // Create new cache or load from disk if file exists
    pub fn new(capacity: NonZeroUsize, path: &str) -> Self {
        let cache = if Path::new(path).exists() {
            Self::load_from_disk(capacity, path).unwrap_or_else(|_| {
                LruCache::new(capacity)
            })
        } else {
            LruCache::new(capacity)
        };

        Self {
            cache,
            path: path.to_string(),
        }
    }

    // Save current cache state to disk
    pub fn save_to_disk(&self) -> Result<(), Box<dyn std::error::Error>> {
        let pairs: Vec<(K, V)> = self.cache
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let serialized = serde_json::to_string(&pairs)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;

        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    // Load cache from disk
    fn load_from_disk(capacity: NonZeroUsize, path: &str) -> Result<LruCache<K, V>, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let pairs: Vec<(K, V)> = serde_json::from_str(&contents)?;
        let mut cache = LruCache::new(capacity);

        for (k, v) in pairs {
            cache.put(k, v);
        }

        Ok(cache)
    }

    // Delegate methods to the underlying LruCache
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.cache.get(key)
    }

    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        let result = self.cache.put(key, value);
        // Optionally save to disk after each update
        // self.save_to_disk().unwrap_or_else(|e| eprintln!("Error saving cache: {}", e));
        result
    }

    // Add other methods you need...
}

#[derive(Debug, Clone)]
pub struct PersistentMemoryCache {
    inner: Arc<RwLock<PersistentLruCache<CacheKey, SignedPacket>>>,
}

impl PersistentMemoryCache {
    /// Creates a new `LRU` cache that holds at most `cap` items.
    pub fn new(capacity: NonZeroUsize, path: &str) -> Self {
        Self {
            inner: Arc::new(RwLock::new(PersistentLruCache::new(capacity, path))),
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;
    use std::fs;
    use tempfile::tempdir;
    use serde::{Serialize, Deserialize};

    // Test types that implement required traits
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct TestKey(String);

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestValue(i32);

    // Helper function to create a temporary file path
    fn temp_cache_path() -> (tempfile::TempDir, String) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_cache.json").to_string_lossy().to_string();
        (dir, path)
    }

    #[test]
    fn test_new_cache_creation() {
        let (_temp_dir, path) = temp_cache_path();
        let capacity = NonZeroUsize::new(5).unwrap();

        // File should not exist yet
        assert!(!Path::new(&path).exists());

        let cache = PersistentLruCache::<TestKey, TestValue>::new(capacity, &path);

        // Verify cache is empty initially
        assert_eq!(cache.cache.len(), 0);
        assert_eq!(cache.cache.cap().get(), 5);
        assert_eq!(cache.path, path);

        // File should still not exist until we save
        assert!(!Path::new(&path).exists());

        let result = cache.save_to_disk().expect("Should save cache");
        assert!(Path::new(&path).exists());
    }

    /*
    #[test]
    fn test_put_and_get() {
        let (_temp_dir, path) = temp_cache_path();
        let capacity = NonZeroUsize::new(3).unwrap();
        let mut cache = PersistentLruCache::new(capacity, &path);

        let key1 = TestKey("key1".to_string());
        let value1 = TestValue(42);

        // Test putting a value
        let old_value = cache.put(key1.clone(), value1.clone());
        assert!(old_value.is_none());

        // Test getting the value
        let retrieved = cache.get(&key1);
        assert_eq!(retrieved, Some(&value1));

        // Test getting non-existent key
        let non_existent = TestKey("missing".to_string());
        assert!(cache.get(&non_existent).is_none());
    }

    #[test]
    fn test_cache_capacity_enforcement() {
        let (_temp_dir, path) = temp_cache_path();
        let capacity = NonZeroUsize::new(2).unwrap();
        let mut cache = PersistentLruCache::new(capacity, &path);

        let key1 = TestKey("key1".to_string());
        let key2 = TestKey("key2".to_string());
        let key3 = TestKey("key3".to_string());

        // Fill cache to capacity
        cache.put(key1.clone(), TestValue(1));
        cache.put(key2.clone(), TestValue(2));
        assert_eq!(cache.cache.len(), 2);

        // Adding third item should evict the first
        cache.put(key3.clone(), TestValue(3));
        assert_eq!(cache.cache.len(), 2);

        // key1 should be evicted, key2 and key3 should remain
        assert!(cache.get(&key1).is_none());
        assert_eq!(cache.get(&key2), Some(&TestValue(2)));
        assert_eq!(cache.get(&key3), Some(&TestValue(3)));
    }

    #[test]
    fn test_save_to_disk() {
        let (_temp_dir, path) = temp_cache_path();
        let capacity = NonZeroUsize::new(3).unwrap();
        let mut cache = PersistentLruCache::new(capacity, &path);

        // Add some data
        cache.put(TestKey("key1".to_string()), TestValue(10));
        cache.put(TestKey("key2".to_string()), TestValue(20));

        // Save to disk
        let result = cache.save_to_disk();
        assert!(result.is_ok());

        // Verify file was created
        assert!(std::path::Path::new(&path).exists());

        // Clean up
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_load_from_disk() {
        let (_temp_dir, path) = temp_cache_path();
        let capacity = NonZeroUsize::new(3).unwrap();

        // Create and populate first cache
        {
            let mut cache1 = PersistentLruCache::new(capacity, &path);
            cache1.put(TestKey("persistent_key".to_string()), TestValue(100));
            cache1.put(TestKey("another_key".to_string()), TestValue(200));
            cache1.save_to_disk().unwrap();
        }

        // Create new cache that should load from disk
        let mut cache2 = PersistentLruCache::new(capacity, &path);

        // Verify data was loaded
        assert_eq!(cache2.get(&TestKey("persistent_key".to_string())), Some(&TestValue(100)));
        assert_eq!(cache2.get(&TestKey("another_key".to_string())), Some(&TestValue(200)));

        // Clean up
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_debug_format() {
        let (_temp_dir, path) = temp_cache_path();
        let capacity = NonZeroUsize::new(2).unwrap();
        let mut cache = PersistentLruCache::new(capacity, &path);

        cache.put(TestKey("debug_key".to_string()), TestValue(42));

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("LruCache"));
        assert!(debug_str.contains("len"));
        assert!(debug_str.contains("cap"));
    }

    #[test]
    fn test_persistent_memory_cache() {
        let (_temp_dir, path) = temp_cache_path();
        let capacity = NonZeroUsize::new(5).unwrap();

        let cache = PersistentMemoryCache::new(capacity, &path);

        // Verify it was created successfully
        assert!(cache.inner.read().is_ok());

        // Test basic thread safety by cloning
        let cache_clone = cache.clone();
        assert!(cache_clone.inner.read().is_ok());
    }

    #[test]
    fn test_corrupted_cache_file_fallback() {
        let (_temp_dir, path) = temp_cache_path();

        // Create a corrupted cache file
        fs::write(&path, "invalid json content").unwrap();

        let capacity = NonZeroUsize::new(3).unwrap();
        let cache = PersistentLruCache::<TestKey, TestValue>::new(capacity, &path);

        // Should create empty cache when file is corrupted
        assert_eq!(cache.cache.len(), 0);

        // Clean up
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_lru_behavior() {
        let (_temp_dir, path) = temp_cache_path();
        let capacity = NonZeroUsize::new(2).unwrap();
        let mut cache = PersistentLruCache::new(capacity, &path);

        let key1 = TestKey("old".to_string());
        let key2 = TestKey("new".to_string());
        let key3 = TestKey("newest".to_string());

        // Fill cache
        cache.put(key1.clone(), TestValue(1));
        cache.put(key2.clone(), TestValue(2));

        // Access key1 to make it recently used
        cache.get(&key1);

        // Add key3 - should evict key2 (least recently used)
        cache.put(key3.clone(), TestValue(3));

        // key1 and key3 should remain, key2 should be evicted
        assert_eq!(cache.get(&key1), Some(&TestValue(1)));
        assert!(cache.get(&key2).is_none());
        assert_eq!(cache.get(&key3), Some(&TestValue(3)));
    }

    #[test]
    fn test_overwrite_existing_key() {
        let (_temp_dir, path) = temp_cache_path();
        let capacity = NonZeroUsize::new(3).unwrap();
        let mut cache = PersistentLruCache::new(capacity, &path);

        let key = TestKey("overwrite_test".to_string());

        // Put initial value
        let old_value = cache.put(key.clone(), TestValue(100));
        assert!(old_value.is_none());

        // Overwrite with new value
        let old_value = cache.put(key.clone(), TestValue(200));
        assert_eq!(old_value, Some(TestValue(100)));

        // Verify new value
        assert_eq!(cache.get(&key), Some(&TestValue(200)));
        assert_eq!(cache.cache.len(), 1); // Should still be just one item
    }
    */
}