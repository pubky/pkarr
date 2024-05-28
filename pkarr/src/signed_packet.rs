//! Signed DNS packet

use crate::{Error, Keypair, PublicKey, Result};
use bytes::{Bytes, BytesMut};
use ed25519_dalek::Signature;
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
        dependent: Packet,
    }

    impl{Debug}
);

impl Inner {
    fn try_from_parts(
        public_key: &PublicKey,
        signature: &Signature,
        timestamp: u64,
        encoded_packet: &Bytes,
    ) -> Result<Self> {
        // Create the inner bytes from <public_key><signature>timestamp><v>
        let mut bytes = BytesMut::with_capacity(encoded_packet.len() + 104);

        bytes.extend_from_slice(public_key.as_bytes());
        bytes.extend_from_slice(&signature.to_bytes());
        bytes.extend_from_slice(&timestamp.to_be_bytes());
        bytes.extend_from_slice(encoded_packet);

        Ok(Self::try_new(bytes.into(), |bytes| {
            Packet::parse(&bytes[104..])
        })?)
    }

    fn try_from_bytes(bytes: &Bytes) -> Result<Self> {
        Ok(Inner::try_new(bytes.to_owned(), |bytes| {
            Packet::parse(&bytes[104..])
        })?)
    }
}

#[derive(Debug)]
/// Signed DNS packet
pub struct SignedPacket {
    inner: Inner,
    last_seen: u64,
}

impl SignedPacket {
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
    ///
    /// # Errors
    /// - Returns [crate::Error::InvalidSignedPacketBytesLength] if `bytes.len()` is smaller than 104 bytes
    /// - Returns [crate::Error::PacketTooLarge] if `bytes.len()` is bigger than 1104 bytes
    /// - Returns [crate::Error::InvalidEd25519PublicKey] if the first 32 bytes are invalid `ed25519` public key
    /// - Returns [crate::Error::InvalidEd25519Signature] if the following 64 bytes are invalid `ed25519` signature
    /// - Returns [crate::Error::DnsError] if it failed to parse the DNS Packet after the first 104 bytes
    pub fn from_bytes(bytes: &Bytes) -> Result<SignedPacket> {
        if bytes.len() < 104 {
            return Err(Error::InvalidSignedPacketBytesLength(bytes.len()));
        }
        if bytes.len() > 1104 {
            return Err(Error::PacketTooLarge(bytes.len()));
        }
        let public_key = PublicKey::try_from(&bytes[..32])?;
        let signature = Signature::from_bytes(bytes[32..96].try_into().unwrap());
        let timestamp = u64::from_be_bytes(bytes[96..104].try_into().unwrap());

        let encoded_packet = &bytes.slice(104..);

        public_key.verify(&signable(timestamp, encoded_packet), &signature)?;

        Ok(SignedPacket {
            inner: Inner::try_from_bytes(bytes)?,
            last_seen: system_time(),
        })
    }

    /// Useful for cloning a [SignedPacket], or cerating one from a previously checked bytes,
    /// like ones stored on disk or in a database.
    pub fn from_bytes_unchecked(bytes: &Bytes, last_seen: u64) -> SignedPacket {
        SignedPacket {
            inner: Inner::try_from_bytes(bytes).unwrap(),
            last_seen,
        }
    }

    /// Creates a [SignedPacket] from a [PublicKey] and the [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md) payload.
    ///
    /// # Errors
    /// - Returns [crate::Error::InvalidSignedPacketBytesLength] if `payload` is too small
    /// - Returns [crate::Error::PacketTooLarge] if the payload is too large.
    /// - Returns [crate::Error::InvalidEd25519Signature] if the signature in the payload is invalid
    /// - Returns [crate::Error::DnsError] if it failed to parse the DNS Packet
    pub fn from_relay_payload(public_key: &PublicKey, payload: &Bytes) -> Result<SignedPacket> {
        let mut bytes = BytesMut::with_capacity(payload.len() + 32);

        bytes.extend_from_slice(public_key.as_bytes());
        bytes.extend_from_slice(payload);

        SignedPacket::from_bytes(&bytes.into())
    }

    /// Creates a new [SignedPacket] from a [Keypair] and a DNS [Packet].
    ///
    /// It will also normalize the names of the [ResourceRecord]s to be relative to the origin,
    /// which would be the z-base32 encoded [PublicKey] of the [Keypair] used to sign the Packet.
    ///
    /// # Errors
    /// - Returns [crate::Error::DnsError] if the packet is invalid or it failed to compress or encode it.
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

        let timestamp = system_time();

        let signature = keypair.sign(&signable(timestamp, &encoded_packet));

        Ok(SignedPacket {
            inner: Inner::try_from_parts(
                &keypair.public_key(),
                &signature,
                timestamp,
                &encoded_packet,
            )?,
            last_seen: system_time(),
        })
    }

    // === Getters ===

    /// Returns the serialized signed packet:
    /// `<32 bytes public_key><64 bytes signature><8 bytes big-endian timestamp in microseconds><encoded DNS packet>`
    pub fn as_bytes(&self) -> &Bytes {
        self.inner.borrow_owner()
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
    pub fn last_seen(&self) -> &u64 {
        &self.last_seen
    }

    // === Setters ===

    /// Set the [Self::last_seen] property
    pub fn set_last_seen(&mut self, last_seen: &u64) {
        self.last_seen = *last_seen;
    }

    // === Public Methods ===

    /// Set the [Self::last_seen] to the current system time
    pub fn refresh(&mut self) {
        self.last_seen = system_time();
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
        ((system_time() - self.last_seen) / 1_000_000) as u32
    }
}

fn signable(timestamp: u64, v: &Bytes) -> Bytes {
    let mut signable = format!("3:seqi{}e1:v{}:", timestamp, v.len()).into_bytes();
    signable.extend(v);
    signable.into()
}

#[cfg(not(target_arch = "wasm32"))]
/// Return the number of microseconds since [SystemTime::UNIX_EPOCH]
pub fn system_time() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("time drift")
        .as_micros() as u64
}

#[cfg(target_arch = "wasm32")]
/// Return the number of microseconds since [SystemTime::UNIX_EPOCH]
pub fn system_time() -> u64 {
    // Won't be an issue for more than 5000 years!
    (js_sys::Date::now() as u64 )
    // Turn miliseconds to microseconds
    * 1000
}

if_dht! {
    use mainline::MutableItem;

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

    impl TryFrom<&MutableItem> for SignedPacket {
        type Error = Error;

        fn try_from(i: &MutableItem) -> Result<Self> {
            let public_key = PublicKey::try_from(i.key()).unwrap();
            let seq = *i.seq() as u64;
            let signature: Signature = i.signature().into();

            Ok(Self {
                inner: Inner::try_from_parts(&public_key, &signature, seq, i.value())?,
                last_seen: system_time(),
            })
        }
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
        Self::from_bytes_unchecked(self.as_bytes(), self.last_seen)
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
        let from_bytes2 = SignedPacket::from_bytes_unchecked(bytes, signed.last_seen);
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

        signed.last_seen = system_time() - (20 * 1_000_000);

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

        signed.last_seen = system_time() - (2 * (DEFAULT_MAXIMUM_TTL as u64) * 1_000_000);

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

        signed.last_seen = system_time() - (30 * 1_000_000);

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
}
