use std::{borrow::Cow, path::Path, time::Duration};

use pkarr::{Cache, CacheKey, SignedPacket, Timestamp};

use byteorder::LittleEndian;
use heed::{types::U64, BoxedError, BytesDecode, BytesEncode, Database, Env, EnvOpenOptions};

use anyhow::Result;
use tracing::debug;

const PKARR_CACHE_TABLE_NAME_SIGNED_PACKET: &str = "pkarrcache:signed_packet";
const PKARR_CACHE_TABLE_NAME_KEY_TO_TIME: &str = "pkarrcache:key_to_time";
const PKARR_CACHE_TABLE_NAME_TIME_TO_KEY: &str = "pkarrcache:time_to_key";

type CacheSignedPacketsTable = Database<CacheKeyCodec, SignedPacketCodec>;
type CacheKeyToTimeTable = Database<CacheKeyCodec, U64<LittleEndian>>;
type CacheTimeToKeyTable = Database<U64<LittleEndian>, CacheKeyCodec>;

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
        let bytes = signed_packet.as_bytes();

        let mut vec = Vec::with_capacity(bytes.len() + 8);

        vec.extend(
            <U64<LittleEndian>>::bytes_encode(&signed_packet.last_seen().into_u64())?.as_ref(),
        );
        vec.extend(bytes);

        Ok(Cow::Owned(vec))
    }
}

impl<'a> BytesDecode<'a> for SignedPacketCodec {
    type DItem = SignedPacket;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        let last_seen = <U64<LittleEndian>>::bytes_decode(bytes)?;

        Ok(SignedPacket::from_bytes_unchecked(
            &bytes[8..].to_vec().into(),
            last_seen,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct HeedCache {
    capacity: usize,
    env: Env,
}

impl HeedCache {
    pub fn new(env_path: &Path, capacity: usize) -> Result<Self> {
        // Page aligned but more than enough bytes for `capacity` many SignedPacket
        let map_size = (((capacity * 1112) + 4095) / 4096) * 4096;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(map_size)
                .max_dbs(3)
                .open(env_path)?
        };

        let mut wtxn = env.write_txn()?;
        let _: CacheSignedPacketsTable =
            env.create_database(&mut wtxn, Some(PKARR_CACHE_TABLE_NAME_SIGNED_PACKET))?;
        let _: CacheKeyToTimeTable =
            env.create_database(&mut wtxn, Some(PKARR_CACHE_TABLE_NAME_KEY_TO_TIME))?;
        let _: CacheTimeToKeyTable =
            env.create_database(&mut wtxn, Some(PKARR_CACHE_TABLE_NAME_TIME_TO_KEY))?;

        wtxn.commit()?;

        let instance = Self { capacity, env };

        let clone = instance.clone();
        std::thread::spawn(move || loop {
            debug!(size = ?clone.len(), "Cache stats");
            std::thread::sleep(Duration::from_secs(60));
        });

        Ok(instance)
    }

    pub fn internal_len(&self) -> Result<usize> {
        let rtxn = self.env.read_txn()?;

        let db: CacheSignedPacketsTable = self
            .env
            .open_database(&rtxn, Some(PKARR_CACHE_TABLE_NAME_SIGNED_PACKET))?
            .unwrap();

        Ok(db.len(&rtxn)? as usize)
    }

    pub fn internal_put(&self, key: &CacheKey, signed_packet: &SignedPacket) -> Result<()> {
        if self.capacity == 0 {
            return Ok(());
        }

        let mut wtxn = self.env.write_txn()?;

        let packets: CacheSignedPacketsTable = self
            .env
            .open_database(&wtxn, Some(PKARR_CACHE_TABLE_NAME_SIGNED_PACKET))?
            .unwrap();

        let key_to_time: CacheKeyToTimeTable = self
            .env
            .open_database(&wtxn, Some(PKARR_CACHE_TABLE_NAME_KEY_TO_TIME))?
            .unwrap();

        let time_to_key: CacheTimeToKeyTable = self
            .env
            .open_database(&wtxn, Some(PKARR_CACHE_TABLE_NAME_TIME_TO_KEY))?
            .unwrap();

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

        time_to_key.put(&mut wtxn, &new_time.into_u64(), key)?;
        key_to_time.put(&mut wtxn, key, &new_time.into_u64())?;

        packets.put(&mut wtxn, key, signed_packet)?;

        wtxn.commit()?;

        Ok(())
    }

    pub fn internal_get(&self, key: &CacheKey) -> Result<Option<SignedPacket>> {
        let mut wtxn = self.env.write_txn()?;

        let packets: CacheSignedPacketsTable = self
            .env
            .open_database(&wtxn, Some(PKARR_CACHE_TABLE_NAME_SIGNED_PACKET))?
            .unwrap();

        let key_to_time: CacheKeyToTimeTable = self
            .env
            .open_database(&wtxn, Some(PKARR_CACHE_TABLE_NAME_KEY_TO_TIME))?
            .unwrap();
        let time_to_key: CacheTimeToKeyTable = self
            .env
            .open_database(&wtxn, Some(PKARR_CACHE_TABLE_NAME_TIME_TO_KEY))?
            .unwrap();

        if let Some(signed_packet) = packets.get(&wtxn, key)? {
            if let Some(time) = key_to_time.get(&wtxn, key)? {
                time_to_key.delete(&mut wtxn, &time)?;
            };

            let new_time = Timestamp::now();

            time_to_key.put(&mut wtxn, &new_time.into_u64(), key)?;
            key_to_time.put(&mut wtxn, key, &new_time.into_u64())?;

            wtxn.commit()?;

            return Ok(Some(signed_packet));
        }

        wtxn.commit()?;

        Ok(None)
    }

    pub fn internal_get_read_only(&self, key: &CacheKey) -> Result<Option<SignedPacket>> {
        let rtxn = self.env.read_txn()?;

        let packets: CacheSignedPacketsTable = self
            .env
            .open_database(&rtxn, Some(PKARR_CACHE_TABLE_NAME_SIGNED_PACKET))?
            .unwrap();

        if let Some(signed_packet) = packets.get(&rtxn, key)? {
            return Ok(Some(signed_packet));
        }

        rtxn.commit()?;

        Ok(None)
    }
}

impl Cache for HeedCache {
    fn len(&self) -> usize {
        match self.internal_len() {
            Ok(result) => result,
            Err(error) => {
                debug!(?error, "Error in HeedCache::len");
                0
            }
        }
    }

    fn put(&self, key: &CacheKey, signed_packet: &SignedPacket) {
        if let Err(error) = self.internal_put(key, signed_packet) {
            debug!(?error, "Error in HeedCache::put");
        };
    }

    fn get(&self, key: &CacheKey) -> Option<SignedPacket> {
        match self.internal_get(key) {
            Ok(result) => result,
            Err(error) => {
                debug!(?error, "Error in HeedCache::get");

                None
            }
        }
    }

    fn get_read_only(&self, key: &CacheKey) -> Option<SignedPacket> {
        match self.internal_get_read_only(key) {
            Ok(result) => result,
            Err(error) => {
                debug!(?error, "Error in HeedCache::get");

                None
            }
        }
    }
}
