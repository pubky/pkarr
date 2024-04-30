use std::{
    borrow::Cow,
    path::Path,
    time::{Duration, Instant, SystemTime},
};

use governor::clock::Reference;
use pkarr::{
    cache::{PkarrCache, PkarrCacheKey},
    SignedPacket,
};

use heed::{BoxedError, BytesDecode, BytesEncode, Env};
use heed::{Database, EnvOpenOptions};

use anyhow::Result;
use tracing::debug;

const PKARR_CACHE_TABLE_NAME_SIGNED_PACKET: &str = "pkarrcache:signed_packet";
const PKARR_CACHE_TABLE_NAME_KEY_TO_TIME: &str = "pkarrcache:key_to_time";
const PKARR_CACHE_TABLE_NAME_TIME_TO_KEY: &str = "pkarrcache:time_to_key";

type PkarrCacheSignedPacketsTable = Database<PkarrCacheKeyCodec, SignedPacketCodec>;
type PkarrCacheKeyToTimeTable = Database<PkarrCacheKeyCodec, InstantCodec>;
type PkarrCacheTimeToKeyTable = Database<InstantCodec, PkarrCacheKeyCodec>;

pub struct PkarrCacheKeyCodec;

impl<'a> BytesEncode<'a> for PkarrCacheKeyCodec {
    type EItem = PkarrCacheKey;

    fn bytes_encode(key: &Self::EItem) -> Result<Cow<[u8]>, BoxedError> {
        Ok(Cow::Owned(key.bytes.to_vec()))
    }
}

pub struct InstantCodec;

impl<'a> BytesEncode<'a> for InstantCodec {
    type EItem = Instant;

    fn bytes_encode(instant: &Self::EItem) -> Result<Cow<[u8]>, BoxedError> {
        let system_now = SystemTime::now();
        let instant_now = Instant::now();
        let approx = system_now - (instant_now - *instant);

        let vec = (approx
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("time drift")
            // Casting as u64 is safe for ~500_000 years.
            .as_micros() as u64)
            .to_be_bytes()
            .to_vec();

        Ok(Cow::Owned(vec))
    }
}

impl<'a> BytesDecode<'a> for InstantCodec {
    type DItem = Instant;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        let mut first_8_bytes: [u8; 8] = [0; 8];
        first_8_bytes.copy_from_slice(&bytes[0..8]);
        let micros = u64::from_be_bytes(first_8_bytes);

        let duration_since_unix = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("time drift");
        let encoded_duration = Duration::from_micros(micros);

        let instant = Instant::now() - duration_since_unix + encoded_duration;

        Ok(instant)
    }
}

pub struct SignedPacketCodec;

impl<'a> BytesEncode<'a> for SignedPacketCodec {
    type EItem = SignedPacket;

    fn bytes_encode(signed_packet: &Self::EItem) -> Result<Cow<[u8]>, BoxedError> {
        let instant = signed_packet.last_seen();
        let bytes = signed_packet.as_bytes();

        let mut vec = Vec::with_capacity(bytes.len() + 8);

        vec.extend(InstantCodec::bytes_encode(instant)?.as_ref());
        vec.extend(bytes);

        Ok(Cow::Owned(vec))
    }
}

impl<'a> BytesDecode<'a> for SignedPacketCodec {
    type DItem = SignedPacket;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        let last_seen = InstantCodec::bytes_decode(&bytes)?;

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
        let _: Database<PkarrCacheKeyCodec, InstantCodec> =
            env.create_database(&mut wtxn, Some(PKARR_CACHE_TABLE_NAME_KEY_TO_TIME))?;
        let _: Database<InstantCodec, PkarrCacheKeyCodec> =
            env.create_database(&mut wtxn, Some(PKARR_CACHE_TABLE_NAME_TIME_TO_KEY))?;

        wtxn.commit()?;

        Ok(Self { capacity, env })
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
        // TODO: delete with capactiy
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

        if let Some(old_time) = key_to_time.get(&wtxn, &key)? {
            time_to_key.delete(&mut wtxn, &old_time)?;
        }

        let new_time = Instant::now();

        time_to_key.put(&mut wtxn, &new_time, &key)?;
        key_to_time.put(&mut wtxn, &key, &new_time)?;

        packets.put(&mut wtxn, &key, &signed_packet)?;

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

        if let Some(signed_packet) = packets.get(&wtxn, &key)? {
            if let Some(time) = key_to_time.get(&wtxn, key)? {
                time_to_key.delete(&mut wtxn, &time)?;
            };

            let new_time = Instant::now();

            time_to_key.put(&mut wtxn, &new_time, key)?;
            key_to_time.put(&mut wtxn, &key, &new_time)?;

            wtxn.commit()?;

            return Ok(Some(signed_packet));
        }

        wtxn.commit()?;

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
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::InstantCodec;
    use heed::BytesDecode;
    use std::time::Instant;

    #[test]
    fn instant_encoding() {
        let instant = Instant::now();
        let encoded = InstantCodec::bytes_encode(&instant).unwrap();
        let decoded = InstantCodec::bytes_decode(&encoded).unwrap();

        assert_eq!(decoded - instant, Duration::from_secs(0))
    }
}
