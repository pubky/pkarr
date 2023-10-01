use crate::{prelude::*, Keypair, PublicKey};
use bytes::{Bytes, BytesMut};
use ed25519_dalek::Signature;
use ouroboros::*;
use self_cell::self_cell;
use simple_dns::Packet;
use std::time::{Instant, SystemTime};

self_cell!(
    struct DnsPacket {
        owner: Bytes,

        #[covariant]
        dependent: Packet,
    }

    impl {Debug}
);

#[derive(Debug)]
pub struct SignedPacket {
    /// The public key of the signer of this packet.
    pub public_key: PublicKey,
    /// Time of signing in milliseconds since the UNIX epoch
    timestamp: u64,
    /// Signature over the bencode encoded timestamp and packet_bytes as `seq` and `v` respectively
    /// according to [BEP0044](https://www.bittorrent.org/beps/bep_0044.html) message.
    signature: Signature,
    /// DNS [Packet].
    /// and its encoded compressed form serving as the `v` value in the [BEP0044](https://www.bittorrent.org/beps/bep_0044.html) message.
    packet: DnsPacket,
}

impl SignedPacket {
    /// Create a new [SignedPacket] from a DNS [Packet] and a [Keypair].
    pub fn new<'a>(keypair: &Keypair, packet: &Packet<'a>) -> Result<SignedPacket> {
        let packet_bytes = packet.build_bytes_vec_compressed()?;
        let timestamp = system_time_now();

        let signable = signable(&timestamp, &packet_bytes);

        let signature = keypair.sign(&signable);

        let packet_bytes = Bytes::from(packet_bytes);

        Ok(SignedPacket {
            public_key: keypair.public_key(),
            signature,
            timestamp,
            packet: DnsPacket::new(packet_bytes, |packet_bytes| {
                Packet::parse(packet_bytes).unwrap()
            }),
        })
    }

    /// Try parsing a relay's GET response and verify the signature to create a [SignedPacket].
    ///
    /// Read more about [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md)
    pub fn try_from_relay_response(public_key: PublicKey, bytes: Bytes) -> Result<SignedPacket> {
        let bytes_length = bytes.len();

        let signature = if bytes_length < 64 {
            return Err(Error::RelayPayloadInvalidSignatureLength(bytes_length));
        } else {
            // Unwrap is safe ase we already checked for the length of `sig_bytes` above.
            Signature::try_from(bytes.slice(..64).as_ref()).unwrap()
        };

        let timestamp = if bytes_length < 72 {
            return Err(Error::RelayPayloadInvalidSequenceLength(bytes_length - 64));
        } else {
            // Unwrap is safe ase we already checked for the length of `seq_bytes` above.
            u64::from_be_bytes(bytes.slice(64..72).as_ref().try_into().unwrap())
        };

        let packet_bytes = bytes.slice(72..);
        let signable = signable(&timestamp, packet_bytes.as_ref());

        public_key.verify(&signable, &signature)?;

        Ok(SignedPacket {
            public_key: PublicKey(public_key.0),
            timestamp,
            signature,
            packet: DnsPacket::try_new(packet_bytes, |packet_bytes| Packet::parse(packet_bytes))?,
        })
    }

    /// Convert the [SignedPacket] int the body of a relay's PUT request.
    ///
    /// Read more about [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md)
    pub fn into_relay_payload(&self) -> Bytes {
        let packet_bytes = self.packet.borrow_owner();
        let mut body = BytesMut::with_capacity((64 + 8 + packet_bytes.len()));

        body.extend_from_slice(&self.signature.to_bytes());
        body.extend_from_slice(&self.timestamp.to_be_bytes());
        body.extend_from_slice(packet_bytes);

        body.into()
    }
}

fn system_time_now() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("time drift")
        .as_micros() as u64
}

fn signable(seq: &u64, v: &[u8]) -> Vec<u8> {
    let mut signable = format!("3:seqi{}e1:v{}:", seq, v.len()).as_bytes().to_vec();

    signable.extend(v);

    signable
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use crate::Keypair;
    use bytes::Bytes;
    use simple_dns::{
        rdata::{RData, A},
        Name, Packet, ResourceRecord, CLASS,
    };

    use super::*;

    #[test]
    fn from_relay_payload() {
        let keypair = Keypair::random();

        let invalid_sig_len = Bytes::from("");

        assert!(
            SignedPacket::try_from_relay_response(keypair.public_key(), invalid_sig_len).is_err()
        );
    }

    #[test]
    fn try_from_packet() {
        let keypair = Keypair::random();

        let mut packet = Packet::new_reply(0);
        packet.answers.push(ResourceRecord::new(
            Name::new("_derp_region.iroh.").unwrap(),
            CLASS::IN,
            30,
            RData::A(A {
                address: Ipv4Addr::new(1, 1, 1, 1).into(),
            }),
        ));

        let args = SignedPacket::new(&keypair, &packet).unwrap();
    }

    #[test]
    fn sign_verify() {
        let keypair = Keypair::random();

        let mut packet = Packet::new_reply(0);
        packet.answers.push(ResourceRecord::new(
            Name::new("_derp_region.iroh.").unwrap(),
            CLASS::IN,
            30,
            RData::A(A {
                address: Ipv4Addr::new(1, 1, 1, 1).into(),
            }),
        ));

        let packet = SignedPacket::new(&keypair, &packet).unwrap();
        let payload = Bytes::from(packet.into_relay_payload());

        assert!(SignedPacket::try_from_relay_response(keypair.public_key(), payload).is_ok());
    }
}
