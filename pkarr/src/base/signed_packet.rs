//! Signed DNS packet

use crate::{Keypair, PublicKey};
use bytes::{Bytes, BytesMut};
use ed25519_dalek::{Signature, SignatureError};
use self_cell::self_cell;
use simple_dns::{
    rdata::{RData, A, AAAA},
    Name, Packet, ResourceRecord, SimpleDnsError,
};
use std::{
    char,
    fmt::{self, Display, Formatter},
    net::{Ipv4Addr, Ipv6Addr},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::timestamp::Timestamp;

const DOT: char = '.';

self_cell!(
    struct Inner {
        owner: Bytes,

        #[covariant]
        dependent: Packet,
    }

    impl{Debug, PartialEq, Eq}
);

impl Inner {
    fn try_from_parts(
        public_key: &PublicKey,
        signature: &Signature,
        timestamp: u64,
        encoded_packet: &Bytes,
    ) -> Result<Self, SimpleDnsError> {
        // Create the inner bytes from <public_key><signature>timestamp><v>
        let mut bytes = BytesMut::with_capacity(encoded_packet.len() + 104);

        bytes.extend_from_slice(public_key.as_bytes());
        bytes.extend_from_slice(&signature.to_bytes());
        bytes.extend_from_slice(&timestamp.to_be_bytes());
        bytes.extend_from_slice(encoded_packet);

        Self::try_new(bytes.into(), |bytes| Packet::parse(&bytes[104..]))
    }

    fn try_from_bytes(bytes: Bytes) -> Result<Self, SimpleDnsError> {
        Inner::try_new(bytes.to_owned(), |bytes| Packet::parse(&bytes[104..]))
    }
}

#[derive(Debug, PartialEq, Eq)]
/// Signed DNS packet
pub struct SignedPacket {
    inner: Inner,
    last_seen: Timestamp,
}

impl SignedPacket {
    pub const MAX_BYTES: u64 = 1104;

    /// Creates a [Self] from the serialized representation:
    /// `<32 bytes public_key><64 bytes signature><8 bytes big-endian timestamp in microseconds><encoded DNS packet>`
    ///
    /// Performs the following validations:
    /// - Bytes minimum length
    /// - Validates the PublicKey
    /// - Verifies the Signature
    /// - Validates the DNS packet encoding
    ///
    /// You can skip all these validations by using [Self::from_bytes_unchecked] instead.
    ///
    /// You can use [Self::from_relay_payload] instead if you are receiving a response from an HTTP relay.
    pub fn from_bytes(bytes: &Bytes) -> Result<SignedPacket, SignedPacketError> {
        if bytes.len() < 104 {
            return Err(SignedPacketError::InvalidSignedPacketBytesLength(
                bytes.len(),
            ));
        }
        if (bytes.len() as u64) > SignedPacket::MAX_BYTES {
            return Err(SignedPacketError::PacketTooLarge(bytes.len()));
        }
        let public_key = PublicKey::try_from(&bytes[..32])?;
        let signature = Signature::from_bytes(bytes[32..96].try_into().unwrap());
        let timestamp = u64::from_be_bytes(bytes[96..104].try_into().unwrap());

        let encoded_packet = &bytes.slice(104..);

        public_key.verify(&signable(timestamp, encoded_packet), &signature)?;

        Ok(SignedPacket {
            inner: Inner::try_from_bytes(bytes.to_owned())?,
            last_seen: Timestamp::now(),
        })
    }

    /// Useful for cloning a [SignedPacket], or cerating one from a previously checked bytes,
    /// like ones stored on disk or in a database.
    pub fn from_bytes_unchecked(bytes: &Bytes, last_seen: impl Into<Timestamp>) -> SignedPacket {
        SignedPacket {
            inner: Inner::try_from_bytes(bytes.to_owned()).unwrap(),
            last_seen: last_seen.into(),
        }
    }

    /// Creates a [SignedPacket] from a [PublicKey] and the [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md) payload.
    pub fn from_relay_payload(
        public_key: &PublicKey,
        payload: &Bytes,
    ) -> Result<SignedPacket, SignedPacketError> {
        let mut bytes = BytesMut::with_capacity(payload.len() + 32);

        bytes.extend_from_slice(public_key.as_bytes());
        bytes.extend_from_slice(payload);

        SignedPacket::from_bytes(&bytes.into())
    }

    /// Creates a new [SignedPacket] from a [Keypair] and a DNS [Packet].
    ///
    /// It will also normalize the names of the [ResourceRecord]s to be relative to the origin,
    /// which would be the z-base32 encoded [PublicKey] of the [Keypair] used to sign the Packet.
    pub fn from_packet(
        keypair: &Keypair,
        packet: &Packet,
    ) -> Result<SignedPacket, SignedPacketError> {
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
            return Err(SignedPacketError::PacketTooLarge(encoded_packet.len()));
        }

        let timestamp = Timestamp::now().into();

        let signature = keypair.sign(&signable(timestamp, &encoded_packet));

        Ok(SignedPacket {
            inner: Inner::try_from_parts(
                &keypair.public_key(),
                &signature,
                timestamp,
                &encoded_packet,
            )?,
            last_seen: Timestamp::now(),
        })
    }

    // === Getters ===

    /// Returns the serialized signed packet:
    /// `<32 bytes public_key><64 bytes signature><8 bytes big-endian timestamp in microseconds><encoded DNS packet>`
    pub fn as_bytes(&self) -> &Bytes {
        self.inner.borrow_owner()
    }

    /// Returns a serialized representation of this [SignedPacket] including
    /// the [SignedPacket::last_seen] timestamp followed by the returned value from [SignedPacket::as_bytes].
    pub fn serialize(&self) -> Bytes {
        let mut bytes = Vec::with_capacity(SignedPacket::MAX_BYTES as usize);
        bytes.extend_from_slice(&self.last_seen.to_bytes());
        bytes.extend_from_slice(self.as_bytes());

        bytes.into()
    }

    /// Deserialize [SignedPacket] from a serialized version for persistent storage using
    /// [SignedPacket::serialize].
    ///
    /// If deserializing the [SignedPacket::last_seen] failed, or is far in the future,
    /// it will be unwrapped to default, i.e the UNIX_EPOCH.
    ///
    /// That is useful for backwards compatibility if you
    /// ever stored the [SignedPacket::last_seen] as Little Endian in previous versions.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, SimpleDnsError> {
        let mut last_seen = Timestamp::try_from(&bytes[0..8]).unwrap_or_default();

        if last_seen > (&Timestamp::now() + 60_000_000) {
            last_seen = Timestamp::from(0)
        }

        Ok(SignedPacket {
            inner: Inner::try_from_bytes(bytes[8..].to_owned().into())?,
            last_seen,
        })
    }

    /// Returns a slice of the serialized [SignedPacket] omitting the leading public_key,
    /// to be sent as a request/response body to or from [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md).
    pub fn to_relay_payload(&self) -> Bytes {
        self.inner.borrow_owner().slice(32..)
    }

    /// Returns the [PublicKey] of the signer of this [SignedPacket]
    pub fn public_key(&self) -> PublicKey {
        PublicKey::try_from(&self.inner.borrow_owner()[0..32]).unwrap()
    }

    /// Returns the [Signature] of the the bencoded sequence number concatenated with the
    /// encoded and compressed packet, as defined in [BEP_0044](https://www.bittorrent.org/beps/bep_0044.html)
    pub fn signature(&self) -> Signature {
        Signature::try_from(&self.inner.borrow_owner()[32..96]).unwrap()
    }

    /// Returns the timestamp in microseconds since the [UNIX_EPOCH](std::time::UNIX_EPOCH).
    ///
    /// This timestamp is authored by the controller of the keypair,
    /// and it is trusted as a way to order which packets where authored after which,
    /// but it shouldn't be used for caching for example, instead, use [Self::last_seen]
    /// which is set when you create a new packet.
    pub fn timestamp(&self) -> u64 {
        let bytes = self.inner.borrow_owner();
        let slice: [u8; 8] = bytes[96..104].try_into().unwrap();

        u64::from_be_bytes(slice)
    }

    /// Returns the DNS [Packet] compressed and encoded.
    pub fn encoded_packet(&self) -> Bytes {
        self.inner.borrow_owner().slice(104..)
    }

    /// Return the DNS [Packet].
    pub fn packet(&self) -> &Packet {
        self.inner.borrow_dependent()
    }

    /// Unix last_seen time in microseconds
    pub fn last_seen(&self) -> &Timestamp {
        &self.last_seen
    }

    // === Setters ===

    /// Set the [Self::last_seen] property
    pub fn set_last_seen(&mut self, last_seen: &Timestamp) {
        self.last_seen = last_seen.into();
    }

    // === Public Methods ===

    /// Set the [Self::last_seen] to the current system time
    pub fn refresh(&mut self) {
        self.last_seen = Timestamp::now();
    }

    /// Return whether this [SignedPacket] is more recent than the given one.
    /// If the timestamps are erqual, the one with the largest value is considered more recent.
    /// Usefel for determining which packet contains the latest information from the Dht.
    /// Assumes that both packets have the same [PublicKey], you shouldn't compare packets from
    /// different keys.
    pub fn more_recent_than(&self, other: &SignedPacket) -> bool {
        // In the rare ocasion of timestamp collission,
        // we use the one with the largest value
        if self.timestamp() == other.timestamp() {
            self.encoded_packet() > other.encoded_packet()
        } else {
            self.timestamp() > other.timestamp()
        }
    }

    /// Returns true if both packets have the same timestamp and packet,
    /// and only differ in [Self::last_seen]
    pub fn is_same_as(&self, other: &SignedPacket) -> bool {
        self.as_bytes() == other.as_bytes()
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

    /// Similar to [resource_records](SignedPacket::resource_records), but filters out
    /// expired records, according the the [Self::last_seen] value and each record's `ttl`.
    pub fn fresh_resource_records(&self, name: &str) -> impl Iterator<Item = &ResourceRecord> {
        let origin = self.public_key().to_z32();
        let normalized_name = normalize_name(&origin, name.to_string());

        self.packet().answers.iter().filter(move |rr| {
            rr.name == Name::new(&normalized_name).unwrap() && rr.ttl > self.elapsed()
        })
    }

    /// calculates the remaining seconds by comparing the [Self::ttl] (clamped by `min` and `max`)
    /// to the [Self::last_seen].
    ///
    /// # Panics
    ///
    /// Panics if `min` < `max`
    pub fn expires_in(&self, min: u32, max: u32) -> u32 {
        match self.ttl(min, max).overflowing_sub(self.elapsed()) {
            (_, true) => 0,
            (ttl, false) => ttl,
        }
    }

    /// Returns the smallest `ttl` in the [Self::packet] resource records,
    /// calmped with `min` and `max`.
    ///
    /// # Panics
    ///
    /// Panics if `min` < `max`
    pub fn ttl(&self, min: u32, max: u32) -> u32 {
        self.packet()
            .answers
            .iter()
            .map(|rr| rr.ttl)
            .min()
            .map_or(min, |v| v.clamp(min, max))
    }

    // === Private Methods ===

    /// Time since the [Self::last_seen] in seconds
    fn elapsed(&self) -> u32 {
        ((Timestamp::now().as_u64() - self.last_seen.as_u64()) / 1_000_000) as u32
    }
}

fn signable(timestamp: u64, v: &Bytes) -> Bytes {
    let mut signable = format!("3:seqi{}e1:v{}:", timestamp, v.len()).into_bytes();
    signable.extend(v);
    signable.into()
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
    }

    if last == "@" || last.is_empty() {
        // Shorthand of origin
        return origin.to_string();
    }

    format!("{}.{}", name, origin)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
use mainline::MutableItem;

use super::keys::PublicKeyError;

#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
impl From<&SignedPacket> for MutableItem {
    fn from(s: &SignedPacket) -> Self {
        let seq: i64 = s.timestamp() as i64;
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

#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
impl TryFrom<&MutableItem> for SignedPacket {
    type Error = SignedPacketError;

    fn try_from(i: &MutableItem) -> Result<Self, SignedPacketError> {
        let public_key = PublicKey::try_from(i.key())?;
        let seq = *i.seq() as u64;
        let signature: Signature = i.signature().into();

        Ok(Self {
            inner: Inner::try_from_parts(&public_key, &signature, seq, i.value())?,
            last_seen: Timestamp::now(),
        })
    }
}

impl AsRef<[u8]> for SignedPacket {
    /// Returns the SignedPacket as a bytes slice with the format:
    /// `<public_key><signature><6 bytes timestamp in microseconds><compressed dns packet>`
    fn as_ref(&self) -> &[u8] {
        self.inner.borrow_owner()
    }
}

impl Clone for SignedPacket {
    fn clone(&self) -> Self {
        Self::from_bytes_unchecked(self.as_bytes(), &self.last_seen)
    }
}

impl Display for SignedPacket {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SignedPacket ({}):\n    last_seen: {} seconds ago\n    timestamp: {},\n    signature: {}\n    records:\n",
            &self.public_key(),
            &self.elapsed(),
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

// === Serialization ===

#[cfg(feature = "serde")]
impl Serialize for SignedPacket {
    /// Serialize a [SignedPacket] for persistent storage.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.serialize().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for SignedPacket {
    /// Deserialize a [SignedPacket] from persistent storage.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;

        SignedPacket::deserialize(&bytes).map_err(serde::de::Error::custom)
    }
}

#[derive(thiserror::Error, Debug)]
/// Errors trying to parse or create a [SignedPacket]
pub enum SignedPacketError {
    #[error(transparent)]
    SignatureError(#[from] SignatureError),

    #[error(transparent)]
    PublicKeyError(#[from] PublicKeyError),

    #[error(transparent)]
    /// Transparent [simple_dns::SimpleDnsError]
    DnsError(#[from] simple_dns::SimpleDnsError),

    #[error("Invalid SignedPacket bytes length, expected at least 104 bytes but got: {0}")]
    /// Serialized signed packets are `<32 bytes publickey><64 bytes signature><8 bytes
    /// timestamp><less than or equal to 1000 bytes encoded dns packet>`.
    InvalidSignedPacketBytesLength(usize),

    #[error("Invalid relay payload size, expected at least 72 bytes but got: {0}")]
    /// Relay api http-body should be `<64 bytes signature><8 bytes timestamp>
    /// <less than or equal to 1000 bytes encoded dns packet>`.
    InvalidRelayPayloadSize(usize),

    #[error("DNS Packet is too large, expected max 1000 bytes but got: {0}")]
    // DNS packet endocded and compressed is larger than 1000 bytes
    PacketTooLarge(usize),
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::dns;

    use crate::{DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL};

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

        assert!(SignedPacket::from_relay_payload(
            &signed_packet.public_key(),
            &signed_packet.to_relay_payload()
        )
        .is_ok());
    }

    #[test]
    fn from_too_large_bytes() {
        let keypair = Keypair::random();

        let bytes = Bytes::from(vec![0; 1073]);
        let error = SignedPacket::from_relay_payload(&keypair.public_key(), &bytes);

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

    #[cfg(feature = "dht")]
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
        let seq: i64 = signed_packet.timestamp() as i64;

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

    #[test]
    fn compressed_names() {
        let keypair = Keypair::random();

        let name = "foobar";
        let dup = name;

        let mut packet = Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("@").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::CNAME(dns::Name::new(name).unwrap().into()),
        ));
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("@").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::CNAME(dns::Name::new(dup).unwrap().into()),
        ));

        let signed = SignedPacket::from_packet(&keypair, &packet).unwrap();

        assert_eq!(
            signed
                .resource_records("@")
                .map(|r| r.rdata.clone())
                .collect::<Vec<_>>(),
            packet
                .answers
                .iter()
                .map(|r| r.rdata.clone())
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn to_bytes_from_bytes() {
        let keypair = Keypair::random();
        let mut packet = Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            30,
            RData::TXT("hello".try_into().unwrap()),
        ));
        let signed = SignedPacket::from_packet(&keypair, &packet).unwrap();
        let bytes = signed.as_bytes();
        let from_bytes = SignedPacket::from_bytes(bytes).unwrap();
        assert_eq!(signed.as_bytes(), from_bytes.as_bytes());
        let from_bytes2 = SignedPacket::from_bytes_unchecked(bytes, &signed.last_seen);
        assert_eq!(signed.as_bytes(), from_bytes2.as_bytes());

        let public_key = keypair.public_key();
        let payload = signed.to_relay_payload();
        let from_relay_payload = SignedPacket::from_relay_payload(&public_key, &payload).unwrap();
        assert_eq!(signed.as_bytes(), from_relay_payload.as_bytes());
    }

    #[test]
    fn clone() {
        let keypair = Keypair::random();
        let mut packet = Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            30,
            RData::TXT("hello".try_into().unwrap()),
        ));

        let signed = SignedPacket::from_packet(&keypair, &packet).unwrap();
        let cloned = signed.clone();

        assert_eq!(cloned.as_bytes(), signed.as_bytes());
    }

    #[test]
    fn expires_in_minimum_ttl() {
        let keypair = Keypair::random();
        let mut packet = Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            10,
            RData::TXT("hello".try_into().unwrap()),
        ));

        let mut signed = SignedPacket::from_packet(&keypair, &packet).unwrap();

        signed.last_seen -= 20 * 1_000_000_u64;

        assert!(
            signed.expires_in(30, u32::MAX) > 0,
            "input minimum_ttl is 30 so ttl = 30"
        );

        assert!(
            signed.expires_in(0, u32::MAX) == 0,
            "input minimum_ttl is 0 so ttl = 10 (smallest in resource records)"
        );
    }

    #[test]
    fn expires_in_maximum_ttl() {
        let keypair = Keypair::random();
        let mut packet = Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            3 * DEFAULT_MAXIMUM_TTL,
            RData::TXT("hello".try_into().unwrap()),
        ));

        let mut signed = SignedPacket::from_packet(&keypair, &packet).unwrap();

        signed.last_seen -= 2 * (DEFAULT_MAXIMUM_TTL as u64) * 1_000_000;

        assert!(
            signed.expires_in(0, DEFAULT_MAXIMUM_TTL) == 0,
            "input maximum_ttl is the dfeault 86400 so maximum ttl = 86400"
        );

        assert!(
            signed.expires_in(0, 7 * DEFAULT_MAXIMUM_TTL) > 0,
            "input maximum_ttl is 7 * 86400 so ttl = 3 * 86400 (smallest in resource records)"
        );
    }

    #[test]
    fn fresh_resource_records() {
        let keypair = Keypair::random();
        let mut packet = Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            30,
            RData::TXT("hello".try_into().unwrap()),
        ));
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            60,
            RData::TXT("world".try_into().unwrap()),
        ));

        let mut signed = SignedPacket::from_packet(&keypair, &packet).unwrap();

        signed.last_seen -= 30 * 1_000_000;

        assert_eq!(signed.fresh_resource_records("_foo").count(), 1);
    }

    #[test]
    fn ttl_empty() {
        let keypair = Keypair::random();
        let packet = Packet::new_reply(0);

        let signed = SignedPacket::from_packet(&keypair, &packet).unwrap();

        assert_eq!(signed.ttl(DEFAULT_MINIMUM_TTL, DEFAULT_MAXIMUM_TTL), 300);
    }

    #[test]
    fn ttl_with_records_less_than_minimum() {
        let keypair = Keypair::random();
        let mut packet = Packet::new_reply(0);

        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            DEFAULT_MINIMUM_TTL / 2,
            RData::TXT("hello".try_into().unwrap()),
        ));
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            DEFAULT_MINIMUM_TTL / 4,
            RData::TXT("world".try_into().unwrap()),
        ));

        let signed = SignedPacket::from_packet(&keypair, &packet).unwrap();

        assert_eq!(
            signed.ttl(DEFAULT_MINIMUM_TTL, DEFAULT_MAXIMUM_TTL),
            DEFAULT_MINIMUM_TTL
        );

        assert_eq!(signed.ttl(0, DEFAULT_MAXIMUM_TTL), DEFAULT_MINIMUM_TTL / 4);
    }

    #[test]
    fn ttl_with_records_more_than_maximum() {
        let keypair = Keypair::random();
        let mut packet = Packet::new_reply(0);

        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            DEFAULT_MAXIMUM_TTL * 2,
            RData::TXT("world".try_into().unwrap()),
        ));

        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            DEFAULT_MAXIMUM_TTL * 4,
            RData::TXT("world".try_into().unwrap()),
        ));

        let signed = SignedPacket::from_packet(&keypair, &packet).unwrap();

        assert_eq!(
            signed.ttl(DEFAULT_MINIMUM_TTL, DEFAULT_MAXIMUM_TTL),
            DEFAULT_MAXIMUM_TTL
        );

        assert_eq!(
            signed.ttl(0, DEFAULT_MAXIMUM_TTL * 8),
            DEFAULT_MAXIMUM_TTL * 2
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde() {
        use postcard::{from_bytes, to_allocvec};

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

        let serialized = to_allocvec(&signed_packet).unwrap();
        let deserialized: SignedPacket = from_bytes(&serialized).unwrap();

        assert_eq!(deserialized, signed_packet);

        // Backwards compat
        {
            let mut bytes = vec![];

            bytes.extend_from_slice(&[210, 1]);
            bytes.extend_from_slice(&signed_packet.last_seen().as_u64().to_le_bytes());
            bytes.extend_from_slice(signed_packet.as_bytes());

            let deserialized: SignedPacket = from_bytes(&bytes).unwrap();

            assert_eq!(deserialized.as_bytes(), signed_packet.as_bytes());
            assert_eq!(deserialized.last_seen(), &Timestamp::from(0));
        }
    }
}
