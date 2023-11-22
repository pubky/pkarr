use crate::{Error, Keypair, PublicKey, Result};
use bytes::{Bytes, BytesMut};
use ed25519_dalek::Signature;
use mainline::common::MutableItem;
use self_cell::self_cell;
use simple_dns::{
    rdata::{RData, A, AAAA},
    Name, Packet, ResourceRecord,
};
use std::{
    char,
    fmt::{self, Display, Formatter},
    net::{Ipv4Addr, Ipv6Addr},
    time::SystemTime,
};

const DOT: char = '.';

self_cell!(
    struct PacketBytes {
        owner: Bytes,

        #[covariant]
        dependent: Packet,
    }

    impl{Debug}
);

#[derive(Debug)]
/// Signed DNS packet as defined in [BEP_0044](https://www.bittorrent.org/beps/bep_0044.html).
///
/// `timestamp` is the number of microseconds since the [UNIX_EPOCH](std::time::UNIX_EPOCH).
pub struct SignedPacket {
    public_key: PublicKey,
    signature: Signature,
    timestamp: u64,
    packet_bytes: PacketBytes,
}

impl SignedPacket {
    /// Returns the [PublicKey] of the signer of this [SignedPacket]
    pub fn public_key(&self) -> &PublicKey {
        &self.public_key
    }

    /// Returns the timestamp in microseconds since the UNIX epoch
    pub fn timestamp(&self) -> &u64 {
        &self.timestamp
    }

    /// Return the DNS [Packet].
    pub fn packet(&self) -> &Packet {
        self.packet_bytes.borrow_dependent()
    }

    /// Return and iterator over the [ResourceRecord]s in the Answers section of the DNS [Packet]
    /// that matches the given name. The name will be normalized to the origin TLD of this packet.
    pub fn resource_records(&self, name: &str) -> impl Iterator<Item = &ResourceRecord> {
        let origin = self.public_key().to_z32();
        let normalized_name = normalize_name(&origin, name.to_string());
        self.packet()
            .answers
            .iter()
            .filter(move |rr| rr.name == Name::new(&normalized_name).unwrap())
    }

    /// Returns the [Signature] of the the bencoded sequence number concatenated with the
    /// encoded and compressed packet, as defined in [BEP_0044](https://www.bittorrent.org/beps/bep_0044.html)
    pub fn signature(&self) -> &Signature {
        &self.signature
    }

    /// Creates a new [SignedPacket] from a [PublicKey] and the concatenated 64 bytes Signature, 8
    /// bytes timestamp and encoded [Packet] as defined in the [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md) spec.
    pub fn from_bytes(public_key: PublicKey, bytes: Bytes) -> Result<SignedPacket> {
        if bytes.len() < 72 {
            return Err(Error::InvalidSingedPacketBytes(bytes.len()));
        }
        if bytes.len() > 1072 {
            return Err(Error::PacketTooLarge(bytes.len()));
        }

        let signature =
            Signature::from_bytes(bytes[..64].try_into().expect("signature is 64 bytes"));
        let timestamp = u64::from_be_bytes(bytes[64..72].try_into().expect("seq is 8 bytes"));

        public_key.verify(&signable(timestamp, &bytes[72..]), &signature)?;

        Ok(SignedPacket {
            public_key,
            signature,
            timestamp,
            packet_bytes: PacketBytes::try_new(bytes, |bytes| Packet::parse(&bytes[72..]))?,
        })
    }

    /// Creates a new [SignedPacket] from a [Keypair] and a DNS [Packet].
    ///
    /// It will also normalize the names of the [ResourceRecord]s to be relative to the origin,
    /// which would be the [zbase32](z32) encoded [PublicKey] of the [Keypair] used to sign the Packet.
    pub fn from_packet(keypair: &Keypair, packet: &Packet) -> Result<SignedPacket> {
        // Normalize names to the origin TLD
        let mut inner = Packet::new_reply(0);

        let origin = keypair.public_key().to_z32();

        let normalized_names: Vec<String> = packet
            .answers
            .iter()
            .map(|answer| normalize_name(&origin, answer.name.to_string()))
            .collect();

        packet
            .answers
            .iter()
            .enumerate()
            .for_each(|(index, answer)| {
                let new_new_name = Name::new_unchecked(&normalized_names[index]);

                inner.answers.push(ResourceRecord::new(
                    new_new_name.clone(),
                    answer.class,
                    answer.ttl,
                    answer.rdata.clone(),
                ))
            });

        // Encode the packet as `v` and verify its length
        let encoded_packet = inner.build_bytes_vec_compressed()?;

        if encoded_packet.len() > 1000 {
            return Err(Error::PacketTooLarge(encoded_packet.len()));
        }

        // Create the inner bytes from <public_key><signature>timestamp><v>
        let mut bytes = BytesMut::with_capacity(encoded_packet.len() + 72);

        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("time drift")
            .as_micros() as u64;

        let sig = keypair.sign(&signable(timestamp, &encoded_packet));

        bytes.extend_from_slice(&sig.to_bytes());
        bytes.extend_from_slice(&timestamp.to_be_bytes());
        bytes.extend_from_slice(&encoded_packet);

        SignedPacket::from_bytes(keypair.public_key(), bytes.into())
    }
}

fn signable(seq: u64, v: &[u8]) -> Vec<u8> {
    let mut signable = format!("3:seqi{}e1:v{}:", seq, v.len()).into_bytes();
    signable.extend(v);
    signable
}

impl From<SignedPacket> for Bytes {
    fn from(s: SignedPacket) -> Self {
        s.packet_bytes.borrow_owner().clone()
    }
}

impl From<SignedPacket> for MutableItem {
    fn from(s: SignedPacket) -> Self {
        let seq: i64 = *s.timestamp() as i64;

        Self::new_signed(
            s.public_key.0.to_bytes(),
            s.signature.to_bytes(),
            s.packet_bytes.borrow_owner().to_vec(),
            seq,
            None,
        )
    }
}

impl TryFrom<MutableItem> for SignedPacket {
    type Error = Error;

    fn try_from(i: MutableItem) -> Result<Self> {
        let encoded_packet = i.value();
        let signature = i.signature();

        // Create the inner bytes from <public_key><signature>timestamp><v>
        let mut bytes = BytesMut::with_capacity(encoded_packet.len() + 72);

        let seq = i.seq().to_owned() as u64;

        bytes.extend_from_slice(signature);
        bytes.extend_from_slice(&seq.to_be_bytes());
        bytes.extend_from_slice(&encoded_packet);

        let packet_bytes = PacketBytes::try_new(bytes.into(), |bytes| Packet::parse(&bytes[72..]))?;

        Ok(Self {
            public_key: i.key().to_owned().try_into().unwrap(),
            signature: signature.into(),
            timestamp: seq,
            packet_bytes,
        })
    }
}

impl AsRef<[u8]> for SignedPacket {
    fn as_ref(&self) -> &[u8] {
        self.packet_bytes.borrow_owner()
    }
}

impl Display for SignedPacket {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SignedPacket ({}):\n    timestamp: {},\n    signature: {}\n    records:\n",
            &self.public_key,
            &self.timestamp(),
            &self.signature(),
        )?;

        for answer in &self.packet().answers {
            writeln!(
                f,
                "        {}  IN  {}  {}\n",
                &answer.name,
                &answer.ttl,
                match &answer.rdata {
                    RData::A(A { address }) => format!("A  {}", Ipv4Addr::from(*address)),
                    RData::AAAA(AAAA { address }) => format!("AAAA  {}", Ipv6Addr::from(*address)),
                    #[allow(clippy::to_string_in_format_args)]
                    RData::CNAME(name) => format!("CNAME  {}", name.to_string()),
                    RData::TXT(txt) => {
                        format!(
                            "TXT  \"{}\"",
                            txt.clone()
                                .try_into()
                                .unwrap_or("__INVALID_TXT_VALUE_".to_string())
                        )
                    }
                    _ => format!("{:?}", answer.rdata),
                }
            )?;
        }

        writeln!(f)?;

        Ok(())
    }
}

fn normalize_name(origin: &str, name: String) -> String {
    let name = if name.ends_with(DOT) {
        name[..name.len() - 1].to_string()
    } else {
        name
    };

    let parts: Vec<&str> = name.split('.').collect();
    let last = *parts.last().unwrap_or(&"");

    if last == origin {
        // Already normalized.
        return name.to_string();
    } else if last == "@" || last.is_empty() {
        // Shorthand of origin
        return origin.to_string();
    }

    format!("{}.{}", name, origin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_names() {
        let origin = "ed4mn3aoazuf1ahpy9rz1nyswhukbj5483ryefwkue7fbp3egkzo";

        assert_eq!(normalize_name(origin, ".".to_string()), origin);
        assert_eq!(normalize_name(origin, "@".to_string()), origin);
        assert_eq!(normalize_name(origin, "@.".to_string()), origin);
        assert_eq!(normalize_name(origin, origin.to_string()), origin);
        assert_eq!(
            normalize_name(origin, "_derp_region.irorh".to_string()),
            format!("_derp_region.irorh.{}", origin)
        );
        assert_eq!(
            normalize_name(origin, format!("_derp_region.irorh.{}", origin)),
            format!("_derp_region.irorh.{}", origin)
        );
        assert_eq!(
            normalize_name(origin, format!("_derp_region.irorh.{}.", origin)),
            format!("_derp_region.irorh.{}", origin)
        );
    }

    #[test]
    fn sign_verify() {
        let keypair = Keypair::random();

        let mut packet = Packet::new_reply(0);
        packet.answers.push(ResourceRecord::new(
            Name::new("_derp_region.iroh.").unwrap(),
            simple_dns::CLASS::IN,
            30,
            RData::A(A {
                address: Ipv4Addr::new(1, 1, 1, 1).into(),
            }),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();
        assert!(
            SignedPacket::from_bytes(signed_packet.public_key().clone(), signed_packet.into())
                .is_ok()
        );
    }

    #[test]
    fn from_too_large_bytes() {
        let keypair = Keypair::random();

        let bytes = Bytes::from(vec![0; 1073]);
        let error = SignedPacket::from_bytes(keypair.public_key().clone(), bytes);

        assert!(error.is_err());
    }

    #[test]
    fn from_too_large_packet() {
        let keypair = Keypair::random();

        let mut packet = Packet::new_reply(0);
        for _ in 0..100 {
            packet.answers.push(ResourceRecord::new(
                Name::new("_derp_region.iroh.").unwrap(),
                simple_dns::CLASS::IN,
                30,
                RData::A(A {
                    address: Ipv4Addr::new(1, 1, 1, 1).into(),
                }),
            ));
        }

        let error = SignedPacket::from_packet(&keypair, &packet);

        assert!(error.is_err());
    }

    #[test]
    fn resource_records_iterator() {
        let keypair = Keypair::random();

        let target = ResourceRecord::new(
            Name::new("_derp_region.iroh.").unwrap(),
            simple_dns::CLASS::IN,
            30,
            RData::A(A {
                address: Ipv4Addr::new(1, 1, 1, 1).into(),
            }),
        );

        let mut packet = Packet::new_reply(0);
        packet.answers.push(target.clone());
        packet.answers.push(ResourceRecord::new(
            Name::new("something else").unwrap(),
            simple_dns::CLASS::IN,
            30,
            RData::A(A {
                address: Ipv4Addr::new(1, 1, 1, 1).into(),
            }),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

        let iter = signed_packet.resource_records("_derp_region.iroh");
        assert_eq!(iter.count(), 1);

        for record in signed_packet.resource_records("_derp_region.iroh") {
            assert_eq!(record.rdata, target.rdata);
        }
    }
}
