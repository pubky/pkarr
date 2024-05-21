use std::{borrow::Cow, path::Path, time::Duration};

use pkarr::{system_time, PkarrCache, PkarrCacheKey, SignedPacket};

use byteorder::LittleEndian;
use heed::{types::U64, BoxedError, BytesDecode, BytesEncode, Database, Env, EnvOpenOptions};

use anyhow::Result;
use tracing::debug;

const PKARR_CACHE_TABLE_NAME_SIGNED_PACKET: &str = "pkarrcache:signed_packet";
const PKARR_CACHE_TABLE_NAME_KEY_TO_TIME: &str = "pkarrcache:key_to_time";
const PKARR_CACHE_TABLE_NAME_TIME_TO_KEY: &str = "pkarrcache:time_to_key";

type PkarrCacheSignedPacketsTable = Database<PkarrCacheKeyCodec, SignedPacketCodec>;
type PkarrCacheKeyToTimeTable = Database<PkarrCacheKeyCodec, U64<LittleEndian>>;
type PkarrCacheTimeToKeyTable = Database<U64<LittleEndian>, PkarrCacheKeyCodec>;

pub struct PkarrCacheKeyCodec;

impl<'a> BytesEncode<'a> for PkarrCacheKeyCodec {
    type EItem = PkarrCacheKey;

    fn bytes_encode(key: &Self::EItem) -> Result<Cow<[u8]>, BoxedError> {
        Ok(Cow::Owned(key.bytes.to_vec()))
    }
}

impl<'a> BytesDecode<'a> for PkarrCacheKeyCodec {
    type DItem = PkarrCacheKey;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        Ok(PkarrCacheKey::from_bytes(bytes)?)
    }
}

pub struct SignedPacketCodec;

impl<'a> BytesEncode<'a> for SignedPacketCodec {
    type EItem = SignedPacket;

    fn bytes_encode(signed_packet: &Self::EItem) -> Result<Cow<[u8]>, BoxedError> {
        let bytes = signed_packet.as_bytes();

        let mut vec = Vec::with_capacity(bytes.len() + 8);

        vec.extend(<U64<LittleEndian>>::bytes_encode(signed_packet.last_seen())?.as_ref());
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
pub struct HeedPkarrCache {
    capacity: usize,
    env: Env,
}

impl HeedPkarrCache {
    pub fn new(env_path: &Path, capacity: usize) -> Result<Self> {
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(10 * 1024 * 1024) // 10MB
                .max_dbs(3)
                .open(env_path)?
        };

        let mut wtxn = env.write_txn()?;
        let _: PkarrCacheSignedPacketsTable =
            env.create_database(&mut wtxn, Some(PKARR_CACHE_TABLE_NAME_SIGNED_PACKET))?;
        let _: PkarrCacheKeyToTimeTable =
            env.create_database(&mut wtxn, Some(PKARR_CACHE_TABLE_NAME_KEY_TO_TIME))?;
        let _: PkarrCacheTimeToKeyTable =
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

        let db: PkarrCacheSignedPacketsTable = self
            .env
            .open_database(&rtxn, Some(PKARR_CACHE_TABLE_NAME_SIGNED_PACKET))?
            .unwrap();

        Ok(db.len(&rtxn)? as usize)
    }

    pub fn internal_put(&self, key: &PkarrCacheKey, signed_packet: &SignedPacket) -> Result<()> {
        if self.capacity == 0 {
            return Ok(());
        }

        let mut wtxn = self.env.write_txn()?;

        let packets: PkarrCacheSignedPacketsTable = self
            .env
            .open_database(&wtxn, Some(PKARR_CACHE_TABLE_NAME_SIGNED_PACKET))?
            .unwrap();

        let key_to_time: PkarrCacheKeyToTimeTable = self
            .env
            .open_database(&wtxn, Some(PKARR_CACHE_TABLE_NAME_KEY_TO_TIME))?
            .unwrap();

        let time_to_key: PkarrCacheTimeToKeyTable = self
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

        let new_time = system_time();

        time_to_key.put(&mut wtxn, &new_time, key)?;
        key_to_time.put(&mut wtxn, key, &new_time)?;

        packets.put(&mut wtxn, key, signed_packet)?;

        wtxn.commit()?;

        Ok(())
    }

    pub fn internal_get(&self, key: &PkarrCacheKey) -> Result<Option<SignedPacket>> {
        let mut wtxn = self.env.write_txn()?;

        let packets: PkarrCacheSignedPacketsTable = self
            .env
            .open_database(&wtxn, Some(PKARR_CACHE_TABLE_NAME_SIGNED_PACKET))?
            .unwrap();

        let key_to_time: PkarrCacheKeyToTimeTable = self
            .env
            .open_database(&wtxn, Some(PKARR_CACHE_TABLE_NAME_KEY_TO_TIME))?
            .unwrap();
        let time_to_key: PkarrCacheTimeToKeyTable = self
            .env
            .open_database(&wtxn, Some(PKARR_CACHE_TABLE_NAME_TIME_TO_KEY))?
            .unwrap();

        if let Some(signed_packet) = packets.get(&wtxn, key)? {
            if let Some(time) = key_to_time.get(&wtxn, key)? {
                time_to_key.delete(&mut wtxn, &time)?;
            };

            let new_time = system_time();

            time_to_key.put(&mut wtxn, &new_time, key)?;
            key_to_time.put(&mut wtxn, key, &new_time)?;

            wtxn.commit()?;

            return Ok(Some(signed_packet));
        }

        wtxn.commit()?;

        Ok(None)
    }

    pub fn internal_get_read_only(&self, key: &PkarrCacheKey) -> Result<Option<SignedPacket>> {
        let rtxn = self.env.read_txn()?;

        let packets: PkarrCacheSignedPacketsTable = self
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

impl PkarrCache for HeedPkarrCache {
    fn len(&self) -> usize {
        match self.internal_len() {
            Ok(result) => result,
            Err(error) => {
                debug!(?error, "Error in HeedPkarrCache::len");
                0
            }
        }
    }

    fn put(&self, key: &PkarrCacheKey, signed_packet: &SignedPacket) {
        if let Err(error) = self.internal_put(key, signed_packet) {
            debug!(?error, "Error in HeedPkarrCache::put");
        };
    }

    fn get(&self, key: &PkarrCacheKey) -> Option<SignedPacket> {
        match self.internal_get(key) {
            Ok(result) => result,
            Err(error) => {
                debug!(?error, "Error in HeedPkarrCache::get");

                None
            }
        }
    }

    fn get_read_only(&self, key: &PkarrCacheKey) -> Option<SignedPacket> {
        match self.internal_get_read_only(key) {
            Ok(result) => result,
            Err(error) => {
                debug!(?error, "Error in HeedPkarrCache::get");

                None
            }
        }
    }
}
