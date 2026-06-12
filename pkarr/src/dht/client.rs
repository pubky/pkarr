use std::future::Future;

use ed25519_dalek::Signature;
use futures_lite::{Stream, StreamExt};
use mainline::{
    async_dht::{AsyncDht, GetMutableDetailed},
    Config, Dht, MutableItem,
};
use ntimestamp::Timestamp;

use crate::{PublicKey, SignedPacket};

use super::{DhtInfo, PublishError, ResolveFound, ResolveReport};

/// Minimum DHT nodes expected to acknowledge storing a published packet.
pub const MINIMUM_PUBLISH_STORED_NODES: u32 = 10;

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
    ) -> Result<u32, PublishError> {
        let cas = cas.map(|timestamp| timestamp.as_u64() as i64);
        Ok(self
            .inner
            .put_mutable(signed_packet.into(), cas)
            .await?
            .stored_at)
    }

    /// Resolve signed packets newer than the given timestamp.
    pub fn resolve_stream(
        &self,
        public_key: &PublicKey,
        more_recent_than: Option<Timestamp>,
    ) -> impl Stream<Item = SignedPacket> + Send {
        let more_recent_than = more_recent_than.map(|timestamp| timestamp.as_u64() as i64);

        self.inner
            .get_mutable(public_key.as_bytes(), None, more_recent_than)
            .filter_map(|item| SignedPacket::try_from(item).ok())
    }

    /// Resolve signed packets newer than the given timestamp and return query diagnostics.
    pub async fn resolve(
        &self,
        public_key: &PublicKey,
        more_recent_than: Option<Timestamp>,
    ) -> Result<
        ResolveFound<impl Future<Output = (SignedPacket, ResolveReport)> + Send>,
        ResolveReport,
    > {
        let more_recent_than = more_recent_than.map(|timestamp| timestamp.as_u64() as i64);
        let mut detailed =
            self.inner
                .get_mutable_detailed(public_key.as_bytes(), None, more_recent_than);

        let mut invalid_signed_packet_count = 0;
        while let Some(item) = detailed.items.next().await {
            match SignedPacket::try_from(item) {
                Ok(first) => {
                    let most_recent = first.clone();
                    return Ok(ResolveFound {
                        first,
                        completion: finish_resolve(
                            detailed,
                            most_recent,
                            invalid_signed_packet_count,
                        ),
                    });
                }
                Err(_) => invalid_signed_packet_count += 1,
            }
        }

        let report = ResolveReport::with_invalid_signed_packets(
            detailed.outcome.recv().await,
            invalid_signed_packet_count,
        );
        Err(report)
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

async fn finish_resolve(
    mut detailed: GetMutableDetailed,
    mut most_recent: SignedPacket,
    mut invalid_signed_packet_count: u32,
) -> (SignedPacket, ResolveReport) {
    while let Some(item) = detailed.items.next().await {
        match SignedPacket::try_from(item) {
            Ok(packet) if packet.more_recent_than(&most_recent) => most_recent = packet,
            Ok(_) => {}
            Err(_) => invalid_signed_packet_count += 1,
        }
    }

    let report = ResolveReport::with_invalid_signed_packets(
        detailed.outcome.recv().await,
        invalid_signed_packet_count,
    );

    (most_recent, report)
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
