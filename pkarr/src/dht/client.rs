use ed25519_dalek::Signature;
use futures_lite::{Stream, StreamExt};
use mainline::{async_dht::AsyncDht, Config, Dht, MutableItem};
use ntimestamp::Timestamp;

use crate::{PublicKey, SignedPacket};

use super::{DhtInfo, PublishError};

/// Pkarr DHT client.
#[derive(Clone, Debug)]
pub struct DhtClient {
    inner: AsyncDht,
}

impl DhtClient {
    /// Build a DHT client from a Mainline configuration.
    pub fn build(config: Config) -> Result<Self, std::io::Error> {
        let dht = Dht::new(config)?;
        Ok(Self {
            inner: dht.as_async(),
        })
    }

    /// Publish a signed packet to the DHT.
    pub async fn publish(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        let cas = cas.map(|timestamp| timestamp.as_u64() as i64);
        self.inner.put_mutable(signed_packet.into(), cas).await?;
        Ok(())
    }

    /// Resolve signed packets newer than the given timestamp.
    pub fn resolve(
        &self,
        public_key: &PublicKey,
        more_recent_than: Option<Timestamp>,
    ) -> impl Stream<Item = SignedPacket> + Send {
        let more_recent_than = more_recent_than.map(|timestamp| timestamp.as_u64() as i64);

        self.inner
            .get_mutable(public_key.as_bytes(), None, more_recent_than)
            .filter_map(|item| SignedPacket::try_from(item).ok())
    }

    /// Return information about the underlying DHT node.
    pub async fn info(&self) -> DhtInfo {
        let info = self.inner.info().await;

        DhtInfo::new(
            info.local_addr(),
            info.public_address(),
            info.firewalled(),
            info.dht_size_estimate(),
        )
    }
}

impl From<&SignedPacket> for MutableItem {
    fn from(s: &SignedPacket) -> Self {
        Self::new_signed_unchecked(
            s.public_key().to_bytes(),
            s.signature().to_bytes(),
            // Packet
            s.inner.borrow_owner()[104..].into(),
            s.timestamp().as_u64() as i64,
            None,
        )
    }
}

impl TryFrom<&MutableItem> for SignedPacket {
    type Error = crate::errors::SignedPacketVerifyError;

    fn try_from(i: &MutableItem) -> Result<Self, crate::errors::SignedPacketVerifyError> {
        let public_key = PublicKey::try_from(i.key())?;
        let seq = i.seq() as u64;
        let signature: Signature = i.signature().into();

        Ok(Self {
            inner: crate::types::signed_packet::Inner::try_from_parts(
                &public_key,
                &signature,
                seq,
                i.value(),
            )?,
            last_seen: Timestamp::now(),
        })
    }
}

impl TryFrom<MutableItem> for SignedPacket {
    type Error = crate::errors::SignedPacketVerifyError;

    fn try_from(i: MutableItem) -> Result<Self, crate::errors::SignedPacketVerifyError> {
        SignedPacket::try_from(&i)
    }
}

#[cfg(test)]
mod tests {
    use mainline::MutableItem;

    use crate::{Keypair, SignedPacket};

    #[test]
    fn signed_packet_converts_to_mutable_item() {
        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .address(
                "_derp_region.iroh.".try_into().unwrap(),
                "1.1.1.1".parse().unwrap(),
                30,
            )
            .sign(&keypair)
            .unwrap();

        let item: MutableItem = (&signed_packet).into();
        let seq = signed_packet.timestamp().as_u64() as i64;

        let expected = MutableItem::new(
            keypair.secret_key().into(),
            &signed_packet.packet().build_bytes_vec_compressed().unwrap(),
            seq,
            None,
        );

        assert_eq!(item, expected);
    }
}
