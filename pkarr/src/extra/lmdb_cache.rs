//! Persistent [crate::Cache] implementation using LMDB's bindings [heed]

use std::{
    borrow::Cow,
    fmt::Debug,
    fs,
    path::Path,
    sync::{Arc, RwLock},
};

use byteorder::BigEndian;
use heed::{
    types::U64, BoxedError, BytesDecode, BytesEncode, Database, Env, EnvOpenOptions, RwTxn,
};

use tracing::debug;

use pubky_timestamp::Timestamp;

use crate::{Cache, CacheKey, SignedPacket};

const MAX_MAP_SIZE: usize = 10995116277760; // 10 TB
const MIN_MAP_SIZE: usize = 10 * 1024 * 1024; // 10 mb

const SIGNED_PACKET_TABLE: &str = "pkarrcache:signed_packet";
const KEY_TO_TIME_TABLE: &str = "pkarrcache:key_to_time";
const TIME_TO_KEY_TABLE: &str = "pkarrcache:time_to_key";

type SignedPacketsTable = Database<CacheKeyCodec, SignedPacket>;
type KeyToTimeTable = Database<CacheKeyCodec, U64<BigEndian>>;
type TimeToKeyTable = Database<U64<BigEndian>, CacheKeyCodec>;

/// A wrapper for [CacheKey] to implement [BytesEncode] and [BytesDecode].
pub struct CacheKeyCodec;

impl<'a> BytesEncode<'a> for CacheKeyCodec {
    type EItem = CacheKey;

    fn bytes_encode(key: &Self::EItem) -> Result<Cow<[u8]>, BoxedError> {
        Ok(Cow::Owned(key.to_vec()))
    }
}

impl<'a> BytesDecode<'a> for CacheKeyCodec {
    type DItem = CacheKey;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        let key: [u8; 20] = bytes.try_into()?;
        Ok(key)
    }
}

impl<'a> BytesEncode<'a> for SignedPacket {
    type EItem = SignedPacket;

    fn bytes_encode(signed_packet: &Self::EItem) -> Result<Cow<[u8]>, BoxedError> {
        Ok(Cow::Owned(signed_packet.serialize().to_vec()))
    }
}

impl<'a> BytesDecode<'a> for SignedPacket {
    type DItem = SignedPacket;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        Ok(SignedPacket::deserialize(bytes)?)
    }
}

#[derive(Clone)]
/// Persistent [crate::Cache] implementation using LMDB's bindings [heed]
pub struct LmdbCache {
    capacity: usize,
    env: Env,
    signed_packets_table: SignedPacketsTable,
    key_to_time_table: KeyToTimeTable,
    time_to_key_table: TimeToKeyTable,
    batch: Arc<RwLock<Vec<CacheKey>>>,
}

impl Debug for LmdbCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LmdbCache")
            .field("capacity", &self.capacity)
            .field("env", &self.env)
            .finish_non_exhaustive()
    }
}

impl LmdbCache {
    /// Creates a new [LmdbCache] at the `env_path` and set the [heed::EnvOpenOptions::map_size]
    /// to a multiple of the `capacity` by [SignedPacket::MAX_BYTES], aligned to system's page size,
    /// a maximum of 10 TB, and a minimum of 10 MB.
    ///
    /// # Safety
    /// LmdbCache uses LMDB, [opening][heed::EnvOpenOptions::open] which is marked unsafe,
    /// because the possible Undefined Behavior (UB) if the lock file is broken.
    pub unsafe fn open(env_path: &Path, capacity: usize) -> Result<Self, Error> {
        let page_size = page_size::get();

        // Page aligned but more than enough bytes for `capacity` many SignedPacket
        let map_size = capacity
            .checked_mul(SignedPacket::MAX_BYTES as usize)
            .and_then(|x| x.checked_add(page_size))
            .and_then(|x| x.checked_div(page_size))
            .and_then(|x| x.checked_mul(page_size))
            .unwrap_or(MAX_MAP_SIZE)
            .max(MIN_MAP_SIZE);

        fs::create_dir_all(env_path)?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(map_size)
                .max_dbs(3)
                .open(env_path)?
        };

        let mut wtxn = env.write_txn()?;

        let signed_packets_table: SignedPacketsTable =
            env.create_database(&mut wtxn, Some(SIGNED_PACKET_TABLE))?;
        let key_to_time_table: KeyToTimeTable =
            env.create_database(&mut wtxn, Some(KEY_TO_TIME_TABLE))?;
        let time_to_key_table: TimeToKeyTable =
            env.create_database(&mut wtxn, Some(TIME_TO_KEY_TABLE))?;

        wtxn.commit()?;

        let instance = Self {
            capacity,
            env,
            signed_packets_table,
            key_to_time_table,
            time_to_key_table,
            batch: Arc::new(RwLock::new(vec![])),
        };

        Ok(instance)
    }

    /// Convenient wrapper around [Self::open].
    ///
    /// Make sure to read the safety section in [Self::open]
    pub fn open_unsafe(env_path: &Path, capacity: usize) -> Result<Self, Error> {
        unsafe { Self::open(env_path, capacity) }
    }

    fn internal_len(&self) -> Result<usize, heed::Error> {
        let rtxn = self.env.read_txn()?;
        let len = self.signed_packets_table.len(&rtxn)? as usize;
        rtxn.commit()?;

        Ok(len)
    }

    fn internal_put(
        &self,
        key: &CacheKey,
        signed_packet: &SignedPacket,
    ) -> Result<(), heed::Error> {
        if self.capacity == 0 {
            return Ok(());
        }

        let mut wtxn = self.env.write_txn()?;

        let packets = self.signed_packets_table;
        let key_to_time = self.key_to_time_table;
        let time_to_key = self.time_to_key_table;

        let mut batch = self.batch.write().expect("LmdbCache::batch.write()");
        update_lru(&mut wtxn, packets, key_to_time, time_to_key, &batch)?;

        let len = packets.len(&wtxn)? as usize;

        if len >= self.capacity {
            debug!(?len, ?self.capacity, "Reached cache capacity, deleting extra item.");

            let mut iter = time_to_key.iter(&wtxn)?;

            if let Some((time, key)) = iter.next().transpose()? {
                drop(iter);

                time_to_key.delete(&mut wtxn, &time)?;
                key_to_time.delete(&mut wtxn, &key)?;
                packets.delete(&mut wtxn, &key)?;
            };
        }

        batch.clear();

        if let Some(old_time) = key_to_time.get(&wtxn, key)? {
            time_to_key.delete(&mut wtxn, &old_time)?;
        }

        let new_time = Timestamp::now();

        time_to_key.put(&mut wtxn, &new_time.as_u64(), key)?;
        key_to_time.put(&mut wtxn, key, &new_time.as_u64())?;

        packets.put(&mut wtxn, key, signed_packet)?;

        wtxn.commit()?;

        Ok(())
    }

    fn internal_get(&self, key: &CacheKey) -> Result<Option<SignedPacket>, heed::Error> {
        self.batch
            .write()
            .expect("LmdbCache::batch.write()")
            .push(*key);

        self.internal_get_read_only(key)
    }

    fn internal_get_read_only(&self, key: &CacheKey) -> Result<Option<SignedPacket>, heed::Error> {
        let rtxn = self.env.read_txn()?;

        if let Some(signed_packet) = self.signed_packets_table.get(&rtxn, key)? {
            return Ok(Some(signed_packet));
        }

        rtxn.commit()?;

        Ok(None)
    }
}

fn update_lru(
    wtxn: &mut RwTxn,
    packets: SignedPacketsTable,
    key_to_time: KeyToTimeTable,
    time_to_key: TimeToKeyTable,
    to_update: &[CacheKey],
) -> Result<(), heed::Error> {
    for key in to_update {
        if packets.get(wtxn, key)?.is_some() {
            if let Some(time) = key_to_time.get(wtxn, key)? {
                time_to_key.delete(wtxn, &time)?;
            };

            let new_time = Timestamp::now();

            time_to_key.put(wtxn, &new_time.as_u64(), key)?;
            key_to_time.put(wtxn, key, &new_time.as_u64())?;
        }
    }

    Ok(())
}

impl Cache for LmdbCache {
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        match self.internal_len() {
            Ok(result) => result,
            Err(error) => {
                debug!(?error, "Error in LmdbCache::len");
                0
            }
        }
    }

    fn put(&self, key: &CacheKey, signed_packet: &SignedPacket) {
        if let Err(error) = self.internal_put(key, signed_packet) {
            debug!(?error, "Error in LmdbCache::put");
        };
    }

    fn get(&self, key: &CacheKey) -> Option<SignedPacket> {
        match self.internal_get(key) {
            Ok(result) => result,
            Err(error) => {
                debug!(?error, "Error in LmdbCache::get");

                None
            }
        }
    }

    fn get_read_only(&self, key: &CacheKey) -> Option<SignedPacket> {
        match self.internal_get_read_only(key) {
            Ok(result) => result,
            Err(error) => {
                debug!(?error, "Error in LmdbCache::get");

                None
            }
        }
    }
}

#[derive(thiserror::Error, Debug)]
/// Pkarr crate error enum.
pub enum Error {
    #[error(transparent)]
    /// Transparent [heed::Error]
    Lmdb(#[from] heed::Error),

    #[error(transparent)]
    /// Transparent [std::io::Error]
    IO(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use crate::Keypair;

    use super::*;

    #[test]
    fn max_map_size() {
        let env_path = std::env::temp_dir().join(Timestamp::now().to_string());

        LmdbCache::open_unsafe(&env_path, usize::MAX).unwrap();
    }

    #[test]
    fn lru_capacity() {
        let env_path = std::env::temp_dir().join(Timestamp::now().to_string());

        let cache = LmdbCache::open_unsafe(&env_path, 2).unwrap();

        let mut keys = vec![];

        for i in 0..2 {
            let signed_packet = SignedPacket::builder()
                .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), i)
                .sign(&Keypair::random())
                .unwrap();

            let key = CacheKey::from(signed_packet.public_key());
            cache.put(&key, &signed_packet);

            keys.push((key, signed_packet));
        }

        assert_eq!(
            cache.get_read_only(&keys.first().unwrap().0).unwrap(),
            keys.first().unwrap().1,
            "first key saved"
        );
        assert_eq!(
            cache.get_read_only(&keys.last().unwrap().0).unwrap(),
            keys.last().unwrap().1,
            "second key saved"
        );

        assert_eq!(cache.len(), 2);

        // Put another one, effectively deleting the oldest.
        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 3)
            .sign(&Keypair::random())
            .unwrap();
        let key = CacheKey::from(signed_packet.public_key());
        cache.put(&key, &signed_packet);

        assert_eq!(cache.len(), 2);

        assert!(
            cache.get_read_only(&keys.first().unwrap().0).is_none(),
            "oldest key dropped"
        );
        assert_eq!(
            cache.get_read_only(&keys.last().unwrap().0).unwrap(),
            keys.last().unwrap().1,
            "more recent key survived"
        );
        assert_eq!(
            cache.get_read_only(&key).unwrap(),
            signed_packet,
            "most recent key survived"
        )
    }

    #[test]
    fn lru_capacity_refresh_oldest() {
        let env_path = std::env::temp_dir().join(Timestamp::now().to_string());

        let cache = LmdbCache::open_unsafe(&env_path, 2).unwrap();

        let mut keys = vec![];

        for i in 0..2 {
            let signed_packet = SignedPacket::builder()
                .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), i)
                .sign(&Keypair::random())
                .unwrap();

            let key = CacheKey::from(signed_packet.public_key());
            cache.put(&key, &signed_packet);

            keys.push((key, signed_packet));
        }

        assert_eq!(
            cache.get_read_only(&keys.first().unwrap().0).unwrap(),
            keys.first().unwrap().1,
            "first key saved"
        );
        assert_eq!(
            cache.get_read_only(&keys.last().unwrap().0).unwrap(),
            keys.last().unwrap().1,
            "second key saved"
        );

        // refresh the oldest
        cache.get(&keys.first().unwrap().0).unwrap();

        assert_eq!(cache.len(), 2);

        // Put another one, effectively deleting the oldest.
        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 3)
            .sign(&Keypair::random())
            .unwrap();
        let key = CacheKey::from(signed_packet.public_key());
        cache.put(&key, &signed_packet);

        assert_eq!(cache.len(), 2);

        assert!(
            cache.get_read_only(&keys.last().unwrap().0).is_none(),
            "oldest key dropped"
        );
        assert_eq!(
            cache.get_read_only(&keys.first().unwrap().0).unwrap(),
            keys.first().unwrap().1,
            "refreshed key survived"
        );
        assert_eq!(
            cache.get_read_only(&key).unwrap(),
            signed_packet,
            "most recent key survived"
        )
    }
}
