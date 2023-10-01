use crate::{prelude::*, Keypair, PublicKey};
use bytes::{Bytes, BytesMut};
use ed25519_dalek::Signature;
use simple_dns::Packet;
use std::time::{Instant, SystemTime};

#[derive(Debug)]
pub struct SignedPacket {
    pub k: PublicKey,
    seq: u64,
    v: Bytes,
    sig: Signature,
}

impl SignedPacket {
    /// Create a new [SignedPacket] from a DNS [Packet] and a [Keypair].
    pub fn new<'a>(keypair: &'a Keypair, packet: &Packet<'a>) -> Result<SignedPacket> {
        let v = packet.build_bytes_vec_compressed()?;
        let seq = system_time_now();

        let signable = signable(&seq, &v);

        let signature = keypair.sign(&signable);

        let v = Bytes::from(v);

        Ok(SignedPacket {
            k: keypair.public_key(),
            sig: signature,
            seq,
            v,
        })
    }

    /// Try parsing a relay's GET response and verify the signature to create a [SignedPacket].
    ///
    /// Read more about [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md)
    pub fn try_from_relay_response<'a>(
        public_key: &'a PublicKey,
        bytes: Bytes,
    ) -> Result<SignedPacket> {
        let bytes_length = bytes.len();

        let sig = if bytes_length < 64 {
            return Err(Error::RelayPayloadInvalidSignatureLength(bytes_length));
        } else {
            // Unwrap is safe ase we already checked for the length of `sig_bytes` above.
            Signature::try_from(bytes.slice(..64).as_ref()).unwrap()
        };

        let seq = if bytes_length < 72 {
            return Err(Error::RelayPayloadInvalidSequenceLength(bytes_length - 64));
        } else {
            // Unwrap is safe ase we already checked for the length of `seq_bytes` above.
            u64::from_be_bytes(bytes.slice(64..72).as_ref().try_into().unwrap())
        };

        let v = bytes.slice(72..);
        let signable = signable(&seq, v.as_ref());

        public_key.verify(&signable, &sig)?;

        Ok(SignedPacket {
            k: PublicKey(public_key.0.clone()),
            seq,
            sig,
            v,
        })
    }

    /// Convert the [SignedPacket] int the body of a relay's PUT request.
    ///
    /// Read more about [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md)
    pub fn into_relay_payload(&self) -> Bytes {
        let mut body = BytesMut::with_capacity((64 + 8 + self.v.len()));

        body.extend_from_slice(&self.sig.to_bytes());
        body.extend_from_slice(&self.seq.to_be_bytes());
        body.extend_from_slice(&self.v);

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
            SignedPacket::try_from_relay_response(&keypair.public_key(), invalid_sig_len).is_err()
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

        let args = SignedPacket::new(&keypair, &packet).unwrap();
        let payload = Bytes::from(args.into_relay_payload());

        assert!(SignedPacket::try_from_relay_response(&args.k, payload).is_ok());
    }
}
