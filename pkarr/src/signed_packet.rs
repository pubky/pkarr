use crate::{Error, Keypair, PublicKey, Result};
use bytes::{Bytes, BytesMut};
use ed25519_dalek::Signature;
#[cfg(feature = "dht")]
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
    struct Inner {
        owner: Bytes,

        #[covariant]
        dependent: InnerParsed,
    }

    impl{Debug}
);

#[derive(Debug)]
struct InnerParsed<'a> {
    public_key: PublicKey,
    timestamp: u64,
    signature: Signature,
    packet: Packet<'a>,
}

impl Inner {
    fn try_from_response(public_key: PublicKey, response: Bytes) -> Result<Self> {
        Self::try_new(response, |response| {
            if response.len() < 72 {
                return Err(Error::InvalidSingedPacketBytes(response.len()));
            }
            if response.len() > 1072 {
                return Err(Error::PacketTooLarge(response.len()));
            }

            let signature =
                Signature::from_bytes(response[..64].try_into().expect("signature is 64 bytes"));
            let timestamp =
                u64::from_be_bytes(response[64..72].try_into().expect("seq is 8 bytes"));

            let encoded_packet = &response.slice(72..);

            public_key.verify(&signable(timestamp, encoded_packet), &signature)?;

            match Packet::parse(&response[72..]) {
                Ok(packet) => Ok(InnerParsed {
                    public_key,
                    timestamp,
                    signature,
                    packet,
                }),
                Err(e) => Err(e.into()),
            }
        })
    }

    fn try_from_parts(
        public_key: PublicKey,
        encoded_packet: Bytes,
        timestamp: u64,
        signature: Signature,
    ) -> Result<Self> {
        // Create the inner bytes from <public_key><signature>timestamp><v>
        let mut bytes = BytesMut::with_capacity(encoded_packet.len() + 104);

        bytes.extend_from_slice(&public_key.to_bytes());
        bytes.extend_from_slice(&signature.to_bytes());
        bytes.extend_from_slice(&timestamp.to_be_bytes());
        bytes.extend_from_slice(&encoded_packet);

        Self::try_new(bytes.into(), |bytes| match Packet::parse(&bytes[104..]) {
            Ok(packet) => Ok(InnerParsed {
                public_key,
                timestamp,
                signature,
                packet,
            }),
            Err(e) => Err(e),
        })
        .map_err(|e| e.into())
    }
}

#[derive(Debug)]
/// Signed DNS packet as defined in [BEP_0044](https://www.bittorrent.org/beps/bep_0044.html).
///
/// `timestamp` is the number of microseconds since the [UNIX_EPOCH](std::time::UNIX_EPOCH).
pub struct SignedPacket {
    inner: Inner,
}

impl SignedPacket {
    /// Creates a new [SignedPacket] from a [PublicKey] and the 64 bytes Signature
    /// concatenated with 8 bytes timestamp and encoded [Packet] as defined in the [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md) spec.
    pub fn from_relay_response(public_key: PublicKey, response: Bytes) -> Result<SignedPacket> {
        let inner = Inner::try_from_response(public_key, response)?;

        Ok(SignedPacket { inner })
    }

    /// Returns the 64 bytes Signature concatenated with 8 bytes timestamp and
    /// encoded [Packet] as defined in the [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md) spec.
    pub fn as_relay_request(&self) -> Bytes {
        self.inner.borrow_owner().slice(32..)
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
        let encoded_packet: Bytes = inner.build_bytes_vec_compressed()?.into();

        if encoded_packet.len() > 1000 {
            return Err(Error::PacketTooLarge(encoded_packet.len()));
        }

        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("time drift")
            .as_micros() as u64;

        let signature = keypair.sign(&signable(timestamp, &encoded_packet));

        Ok(SignedPacket {
            inner: Inner::try_from_parts(
                keypair.public_key(),
                encoded_packet,
                timestamp,
                signature,
            )?,
        })
    }

    // === Getters ===

    /// Returns the [PublicKey] of the signer of this [SignedPacket]
    pub fn public_key(&self) -> &PublicKey {
        &self.inner.borrow_dependent().public_key
    }

    /// Returns the timestamp in microseconds since the UNIX epoch
    pub fn timestamp(&self) -> &u64 {
        &self.inner.borrow_dependent().timestamp
    }

    /// Return the DNS [Packet].
    pub fn packet(&self) -> &Packet {
        &self.inner.borrow_dependent().packet
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
        &self.inner.borrow_dependent().signature
    }

    pub fn encoded_packet(&self) -> Bytes {
        self.inner.borrow_owner().slice(104..)
    }
}

fn signable(timestamp: u64, v: &Bytes) -> Bytes {
    let mut signable = format!("3:seqi{}e1:v{}:", timestamp, v.len()).into_bytes();
    signable.extend(v);
    signable.into()
}

#[cfg(feature = "dht")]
impl From<&SignedPacket> for MutableItem {
    fn from(s: &SignedPacket) -> Self {
        let seq: i64 = *s.timestamp() as i64;
        let packet = s.inner.borrow_owner().slice(104..);

        Self::new_signed_unchecked(
            s.public_key().to_bytes(),
            s.signature().to_bytes(),
            packet,
            seq,
            None,
        )
    }
}

#[cfg(feature = "dht")]
impl TryFrom<MutableItem> for SignedPacket {
    type Error = Error;

    fn try_from(i: MutableItem) -> Result<Self> {
        let public_key: PublicKey = i.key().to_owned().try_into().unwrap();
        let encoded_packet: Bytes = i.value().to_vec().into();
        let seq = i.seq().to_owned() as u64;
        let signature: Signature = i.signature().into();

        Ok(Self {
            inner: Inner::try_from_parts(public_key, encoded_packet, seq, signature)?,
        })
    }
}

impl AsRef<[u8]> for SignedPacket {
    fn as_ref(&self) -> &[u8] {
        self.inner.borrow_owner()
    }
}

impl Display for SignedPacket {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SignedPacket ({}):\n    timestamp: {},\n    signature: {}\n    records:\n",
            &self.public_key(),
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

        assert!(SignedPacket::from_relay_response(
            signed_packet.public_key().clone(),
            signed_packet.as_relay_request()
        )
        .is_ok());
    }

    #[test]
    fn from_too_large_bytes() {
        let keypair = Keypair::random();

        let bytes = Bytes::from(vec![0; 1073]);
        let error = SignedPacket::from_relay_response(keypair.public_key().clone(), bytes);

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

    #[test]
    fn to_mutable() {
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
        let item: MutableItem = (&signed_packet).into();
        let seq: i64 = *signed_packet.timestamp() as i64;

        let expected = MutableItem::new(
            keypair.secret_key().into(),
            signed_packet
                .packet()
                .build_bytes_vec_compressed()
                .unwrap()
                .into(),
            seq,
            None,
        );

        assert_eq!(item, expected);
    }
}
