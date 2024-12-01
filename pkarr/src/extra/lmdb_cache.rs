//! Persistent [crate::base::cache::Cache] implementation using LMDB's bindings [heed]

use std::{borrow::Cow, fs, path::Path, time::Duration};

use byteorder::LittleEndian;
use heed::{types::U64, BoxedError, BytesDecode, BytesEncode, Database, Env, EnvOpenOptions};
use libc::{sysconf, _SC_PAGESIZE};

use tracing::debug;

use pubky_timestamp::Timestamp;

use crate::{
    base::cache::{Cache, CacheKey},
    SignedPacket,
};

const MAX_MAP_SIZE: usize = 10995116277760; // 10 TB
const MIN_MAP_SIZE: usize = 10 * 1024 * 1024; // 10 mb

const SIGNED_PACKET_TABLE: &str = "pkarrcache:signed_packet";
const KEY_TO_TIME_TABLE: &str = "pkarrcache:key_to_time";
const TIME_TO_KEY_TABLE: &str = "pkarrcache:time_to_key";

type SignedPacketsTable = Database<CacheKeyCodec, SignedPacketCodec>;
type KeyToTimeTable = Database<CacheKeyCodec, U64<LittleEndian>>;
type TimeToKeyTable = Database<U64<LittleEndian>, CacheKeyCodec>;

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

pub struct SignedPacketCodec;

impl<'a> BytesEncode<'a> for SignedPacketCodec {
    type EItem = SignedPacket;

    fn bytes_encode(signed_packet: &Self::EItem) -> Result<Cow<[u8]>, BoxedError> {
        Ok(Cow::Owned(signed_packet.serialize().to_vec()))
    }
}

impl<'a> BytesDecode<'a> for SignedPacketCodec {
    type DItem = SignedPacket;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        Ok(SignedPacket::deserialize(bytes)?)
    }
}

#[derive(Debug, Clone)]
/// Persistent [crate::base::cache::Cache] implementation using LMDB's bindings [heed]
pub struct LmdbCache {
    capacity: usize,
    env: Env,
    signed_packets_table: SignedPacketsTable,
    key_to_time_table: KeyToTimeTable,
    time_to_key_table: TimeToKeyTable,
}

impl LmdbCache {
    /// Creates a new [LmdbCache] at the `env_path` and set the [heed::EnvOpenOptions::map_size]
    /// to a multiple of the `capacity` by [SignedPacket::MAX_BYTES], aligned to system's page size,
    /// a maximum of 10 TB, and a minimum of 10 MB.
    pub fn new(env_path: &Path, capacity: usize) -> Result<Self, Error> {
        let page_size = unsafe { sysconf(_SC_PAGESIZE) as usize };

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
        };

        let clone = instance.clone();
        std::thread::spawn(move || loop {
            debug!(size = ?clone.len(), "Cache stats");
            std::thread::sleep(Duration::from_secs(60));
        });

        Ok(instance)
    }

    pub fn internal_len(&self) -> Result<usize, heed::Error> {
        let rtxn = self.env.read_txn()?;
        let len = self.signed_packets_table.len(&rtxn)? as usize;
        rtxn.commit()?;

        Ok(len)
    }

    pub fn internal_put(
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

        let len = packets.len(&wtxn)? as usize;

        if len >= self.capacity {
            debug!(?len, ?self.capacity, "Reached cache capacity, deleting extra item.");

            let mut iter = time_to_key.rev_iter(&wtxn)?;

            if let Some((time, key)) = iter.next().transpose()? {
                drop(iter);

                time_to_key.delete(&mut wtxn, &time)?;
                key_to_time.delete(&mut wtxn, &key)?;
                packets.delete(&mut wtxn, &key)?;
            };
        }

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

    pub fn internal_get(&self, key: &CacheKey) -> Result<Option<SignedPacket>, heed::Error> {
        let mut wtxn = self.env.write_txn()?;

        let packets = self.signed_packets_table;
        let key_to_time = self.key_to_time_table;
        let time_to_key = self.time_to_key_table;

        if let Some(signed_packet) = packets.get(&wtxn, key)? {
            if let Some(time) = key_to_time.get(&wtxn, key)? {
                time_to_key.delete(&mut wtxn, &time)?;
            };

            let new_time = Timestamp::now();

            time_to_key.put(&mut wtxn, &new_time.as_u64(), key)?;
            key_to_time.put(&mut wtxn, key, &new_time.as_u64())?;

            wtxn.commit()?;

            return Ok(Some(signed_packet));
        }

        wtxn.commit()?;

        Ok(None)
    }

    pub fn internal_get_read_only(
        &self,
        key: &CacheKey,
    ) -> Result<Option<SignedPacket>, heed::Error> {
        let rtxn = self.env.read_txn()?;

        if let Some(signed_packet) = self.signed_packets_table.get(&rtxn, key)? {
            return Ok(Some(signed_packet));
        }

        rtxn.commit()?;

        Ok(None)
    }
}

impl Cache for LmdbCache {
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
    use std::usize;

    use super::*;

    #[test]
    fn max_map_size() {
        let env_path = std::env::temp_dir().join(Timestamp::now().to_string());

        LmdbCache::new(&env_path, usize::MAX).unwrap();
    }
}
