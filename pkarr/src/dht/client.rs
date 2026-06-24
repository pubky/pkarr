use ed25519_dalek::Signature;
use futures_lite::{Stream, StreamExt};
use mainline::{
    async_dht::{AsyncDht, GetMutableDetailed},
    Config, Dht, MutableItem,
};
use ntimestamp::Timestamp;

use crate::{PublicKey, SignedPacket};

use super::{DhtInfo, PublishError, ResolveError, ResolveOutcome, ResolveReport, ResolveResponse};

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
    ///
    /// # Returns
    ///
    /// The number of DHT nodes that acknowledged storing the packet.
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
    ///
    /// The first valid signed packet, if any, is available immediately from
    /// the returned [`ResolveResponse`]. Completing the response returns the
    /// most recent valid signed packet or a resolve error with diagnostics.
    ///
    /// BEP44 sequence numbers define mutable-item freshness. If a mutable item
    /// with a higher sequence number is not a valid signed packet, that item is
    /// still treated as the current DHT state and supersedes older valid signed
    /// packets for the same key.
    ///
    pub async fn resolve(
        &self,
        public_key: &PublicKey,
        more_recent_than: Option<Timestamp>,
    ) -> ResolveResponse {
        let more_recent_than = more_recent_than.map(|timestamp| timestamp.as_u64() as i64);
        let mut detailed =
            self.inner
                .get_mutable_detailed(public_key.as_bytes(), None, more_recent_than);

        let mut highest_seq = None;
        while let Some(item) = detailed.items.next().await {
            let seq = item.seq();
            // BEP44 sequence numbers define the mutable item's freshness.
            // A newer item that is not a valid signed packet still supersedes
            // older valid signed packets for this key.
            if seq >= highest_seq.unwrap_or(seq) {
                highest_seq = Some(seq);
                if let Ok(packet) = SignedPacket::try_from(&item) {
                    let most_recent = packet.clone();
                    return ResolveResponse::new(
                        Some(packet),
                        finish_resolve(detailed, Some(most_recent), highest_seq),
                    );
                }
            }
        }

        ResolveResponse::new(None, finish_resolve(detailed, None, highest_seq))
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
    mut most_recent: Option<SignedPacket>,
    mut highest_seq: Option<i64>,
) -> ResolveOutcome {
    while let Some(item) = detailed.items.next().await {
        let seq = item.seq();
        if seq >= highest_seq.unwrap_or(seq) {
            highest_seq = Some(seq);
            match SignedPacket::try_from(&item) {
                Ok(packet)
                    if most_recent
                        .as_ref()
                        .is_none_or(|most_recent| packet.more_recent_than(most_recent)) =>
                {
                    most_recent = Some(packet)
                }
                _ => (),
            }
        }
    }

    let report = ResolveReport::new(detailed.outcome.recv().await);
    let most_recent = completed_resolve_result(most_recent, highest_seq, &report);

    ResolveOutcome {
        most_recent,
        report,
    }
}

fn completed_resolve_result(
    most_recent: Option<SignedPacket>,
    highest_seq: Option<i64>,
    report: &ResolveReport,
) -> Result<SignedPacket, ResolveError> {
    let Some(most_recent) = most_recent else {
        return Err(if let Some(seq) = highest_seq {
            ResolveError::InvalidSignedPacket { seq }
        } else {
            ResolveError::from(report.clone())
        });
    };

    let Some(highest_seq) = highest_seq else {
        return Ok(most_recent);
    };

    let packet_seq = most_recent.timestamp().as_u64() as i64;
    if packet_seq >= highest_seq {
        Ok(most_recent)
    } else {
        Err(ResolveError::InvalidSignedPacket { seq: highest_seq })
    }
}

#[cfg(test)]
mod tests {
    use mainline::MutableItem;

    use crate::{Keypair, SignedPacket};

    fn report(outcome: mainline::GetMutableOutcome) -> super::ResolveReport {
        super::ResolveReport::new(outcome)
    }

    fn healthy_report() -> super::ResolveReport {
        report(mainline::GetMutableOutcome {
            queried: 20,
            values: 1,
            ..Default::default()
        })
    }

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

    #[test]
    fn completed_resolve_result_preserves_valid_signed_packet_when_invalid_seq_is_not_newer() {
        let signed_packet = SignedPacket::builder()
            .timestamp(10.into())
            .address(
                "_derp_region.iroh.".try_into().unwrap(),
                "1.1.1.1".parse().unwrap(),
                30,
            )
            .sign(&Keypair::random())
            .unwrap();

        assert_eq!(
            super::completed_resolve_result(
                Some(signed_packet.clone()),
                Some(10),
                &healthy_report()
            ),
            Ok(signed_packet)
        );
    }

    #[test]
    fn completed_resolve_result_respects_newer_invalid_signed_packet_seq() {
        let signed_packet = SignedPacket::builder()
            .timestamp(10.into())
            .address(
                "_derp_region.iroh.".try_into().unwrap(),
                "1.1.1.1".parse().unwrap(),
                30,
            )
            .sign(&Keypair::random())
            .unwrap();

        assert_eq!(
            super::completed_resolve_result(Some(signed_packet), Some(11), &healthy_report()),
            Err(super::ResolveError::InvalidSignedPacket { seq: 11 })
        );
    }

    #[test]
    fn completed_resolve_result_reports_invalid_signed_packet_without_valid_packet() {
        assert_eq!(
            super::completed_resolve_result(None, Some(11), &healthy_report()),
            Err(super::ResolveError::InvalidSignedPacket { seq: 11 })
        );
    }

    #[test]
    fn completed_resolve_result_classifies_report_without_any_packet() {
        assert_eq!(
            super::completed_resolve_result(
                None,
                None,
                &report(mainline::GetMutableOutcome::default())
            ),
            Err(super::ResolveError::NoNodesQueried)
        );
    }
}
