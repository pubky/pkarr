//! Signed DNS packet

use crate::{Error, Keypair, PublicKey, Result};
use ed25519_dalek::Signature;
use hickory_proto::{
    op::{header::MessageType, Message},
    rr::{Name, RData, Record},
    serialize::binary::{BinEncodable, BinEncoder},
};
use std::{
    char,
    fmt::{self, Display, Formatter},
    sync::Arc,
};

#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;

const DOT: char = '.';

const OFFSET: usize = 104;

#[derive(Debug, PartialEq, Eq, Clone)]
/// Signed DNS packet
pub struct SignedPacket {
    inner: Arc<Inner>,
    last_seen: u64,
}

#[derive(Debug, PartialEq, Eq)]
struct Inner {
    message: Message,
    public_key: PublicKey,
    signature: Signature,
    timestamp: u64,
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
    pub fn from_bytes(bytes: &[u8]) -> Result<SignedPacket> {
        Self::from_bytes_with_last_seen(bytes, system_time())
    }

    /// Document ME
    pub fn from_bytes_with_last_seen(bytes: &[u8], last_seen: u64) -> Result<SignedPacket> {
        if bytes.len() < OFFSET {
            return Err(Error::InvalidSignedPacketBytesLength(bytes.len()));
        }
        if bytes.len() > 1104 {
            return Err(Error::PacketTooLarge(bytes.len()));
        }
        let public_key = PublicKey::try_from(&bytes[..32])?;
        let signature = Signature::from_bytes(bytes[32..96].try_into().unwrap());
        let timestamp = u64::from_be_bytes(bytes[96..104].try_into().unwrap());

        let raw_message = &bytes[OFFSET..];
        public_key.verify(&signable(timestamp, raw_message), &signature)?;
        let message = Message::from_vec(raw_message)?;

        Ok(SignedPacket {
            inner: Arc::new(Inner {
                public_key,
                signature,
                timestamp,
                message,
            }),
            last_seen,
        })
    }

    /// Creates a [SignedPacket] from a [PublicKey] and the [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md) payload.
    ///
    /// # Errors
    /// - Returns [crate::Error::InvalidSignedPacketBytesLength] if `payload` is too small
    /// - Returns [crate::Error::PacketTooLarge] if the payload is too large.
    /// - Returns [crate::Error::InvalidEd25519Signature] if the signature in the payload is invalid
    /// - Returns [crate::Error::DnsError] if it failed to parse the DNS Packet
    pub fn from_relay_payload(public_key: PublicKey, bytes: &[u8]) -> Result<SignedPacket> {
        if bytes.len() < 72 {
            return Err(Error::InvalidSignedPacketBytesLength(bytes.len()));
        }
        if bytes.len() > 1104 - 32 {
            return Err(Error::PacketTooLarge(bytes.len()));
        }

        let signature = Signature::from_bytes(bytes[..64].try_into().unwrap());
        let timestamp = u64::from_be_bytes(bytes[64..72].try_into().unwrap());

        let raw_message = &bytes[72..];
        public_key.verify(&signable(timestamp, raw_message), &signature)?;
        let message = Message::from_vec(raw_message)?;

        Ok(SignedPacket {
            inner: Arc::new(Inner {
                public_key,
                signature,
                timestamp,
                message,
            }),
            last_seen: system_time(),
        })
    }

    /// Creates a new [SignedPacket] from a [Keypair] and a DNS [Packet].
    ///
    /// It will also normalize the names of the [ResourceRecord]s to be relative to the origin,
    /// which would be the [zbase32](z32) encoded [PublicKey] of the [Keypair] used to sign the Packet.
    ///
    /// # Errors
    /// - Returns [crate::Error::DnsError] if the packet is invalid or it failed to compress or encode it.
    pub fn from_packet(keypair: &Keypair, in_message: &Message) -> Result<SignedPacket> {
        // Normalize names to the origin TLD
        let mut message = Message::new();
        message.set_message_type(MessageType::Response);

        let origin = keypair.public_key().to_z32();

        for answer in in_message.answers() {
            let name = normalize_name(&origin, answer.name().to_string());
            let new_name = Name::from_str_relaxed(name)?;

            let mut record: Record = Record::with(new_name, answer.record_type(), answer.ttl());
            record.set_dns_class(answer.dns_class());
            record.set_data(answer.data().cloned());
            message.add_answer(record);
        }

        // Encode the packet as `v` and verify its length
        let encoded_packet: Vec<u8> = message.to_vec()?;

        if encoded_packet.len() > 1000 {
            return Err(Error::PacketTooLarge(encoded_packet.len()));
        }

        let timestamp = system_time();

        let signature = keypair.sign(&signable(timestamp, &encoded_packet));

        Ok(SignedPacket {
            inner: Arc::new(Inner {
                public_key: keypair.public_key(),
                signature,
                timestamp,
                message,
            }),
            last_seen: system_time(),
        })
    }

    // === Encoding ===

    /// Writes the serialized signed packet to the given vector.
    /// `<32 bytes public_key><64 bytes signature><8 bytes big-endian timestamp in microseconds><encoded DNS packet>`
    pub fn to_writer(&self, writer: &mut Vec<u8>) -> Result<()> {
        let mut encoder = BinEncoder::new(writer);

        encoder.emit_vec(self.inner.public_key.as_bytes())?;
        encoder.emit_vec(&self.inner.signature.to_bytes())?;
        encoder.emit_vec(&self.inner.timestamp.to_be_bytes())?;
        self.inner.message.emit(&mut encoder)?;

        Ok(())
    }

    /// Returns the serialized signed packet:
    /// `<32 bytes public_key><64 bytes signature><8 bytes big-endian timestamp in microseconds><encoded DNS packet>`
    pub fn to_vec(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(1104);
        self.to_writer(&mut bytes).expect("vec");

        bytes
    }

    /// Writes the serialized [SignedPacket], omitting the leading public_key, to the
    /// given vector.
    pub fn to_relay_payload_writer(&self, bytes: &mut Vec<u8>) -> Result<()> {
        let mut encoder = BinEncoder::new(bytes);

        encoder.emit_vec(&self.inner.signature.to_bytes())?;
        encoder.emit_vec(&self.inner.timestamp.to_be_bytes())?;
        self.inner.message.emit(&mut encoder)?;
        Ok(())
    }

    /// Returns the serialized [SignedPacket] omitting the leading public_key,
    /// to be sent as a request/response body to or from [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md).
    pub fn to_relay_payload(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(1000);
        self.to_relay_payload_writer(&mut bytes)
            .expect("valid message");

        bytes
    }

    // === Getters ===

    /// Returns the [PublicKey] of the signer of this [SignedPacket]
    pub fn public_key(&self) -> &PublicKey {
        &self.inner.public_key
    }

    /// Returns the [Signature] of the the bencoded sequence number concatenated with the
    /// encoded and compressed packet, as defined in [BEP_0044](https://www.bittorrent.org/beps/bep_0044.html)
    pub fn signature(&self) -> &Signature {
        &self.inner.signature
    }

    /// Returns the timestamp in microseconds since the [UNIX_EPOCH](std::time::UNIX_EPOCH).
    ///
    /// This timestamp is authored by the controller of the keypair,
    /// and it is trusted as a way to order which packets where authored after which,
    /// but it shouldn't be used for caching for example, instead, use [Self::last_seen]
    /// which is set when you create a new packet.
    pub fn timestamp(&self) -> u64 {
        self.inner.timestamp
    }

    /// Returns the DNS [Message] compressed and encoded.
    pub fn encoded_message(&self) -> Vec<u8> {
        self.inner.message.to_vec().expect("valid message")
    }

    /// Return the DNS [Message].
    pub fn message(&self) -> &Message {
        &self.inner.message
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
            self.encoded_message() > other.encoded_message()
        } else {
            self.timestamp() > other.timestamp()
        }
    }

    /// Returns true if both packets have the same timestamp and packet,
    /// and only differ in [Self::last_seen]
    pub fn is_same_as(&self, other: &SignedPacket) -> bool {
        self == other
    }

    /// Return and iterator over the [ResourceRecord]s in the Answers section of the DNS [Packet]
    /// that matches the given name. The name will be normalized to the origin TLD of this packet.
    pub fn resource_records(&self, name: &str) -> impl Iterator<Item = &Record> {
        let origin = self.public_key().to_z32();
        let normalized_name = normalize_name(&origin, name.to_string());
        self.inner
            .message
            .answers()
            .iter()
            .filter(move |rr| rr.name().to_string() == normalized_name)
    }

    /// Similar to [resource_records](SignedPacket::resource_records), but filters out
    /// expired records, according the the [Self::last_seen] value and each record's `ttl`.
    pub fn fresh_resource_records(&self, name: &str) -> impl Iterator<Item = &Record> {
        let origin = self.public_key().to_z32();
        let normalized_name = normalize_name(&origin, name.to_string());

        self.inner
            .message
            .answers()
            .iter()
            .filter(move |rr| rr.name().to_string() == normalized_name && rr.ttl() > self.elapsed())
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
        self.inner
            .message
            .answers()
            .iter()
            .map(|rr| rr.ttl())
            .min()
            .map_or(min, |v| v.clamp(min, max))
    }

    // === Private Methods ===

    /// Time since the [Self::last_seen] in seconds
    fn elapsed(&self) -> u32 {
        ((system_time() - self.last_seen) / 1_000_000) as u32
    }
}

fn signable(timestamp: u64, v: &[u8]) -> Vec<u8> {
    let mut signable = format!("3:seqi{}e1:v{}:", timestamp, v.len())
        .as_bytes()
        .to_vec();
    signable.extend_from_slice(v);
    signable
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
            let packet = s.encoded_message();

            Self::new_signed_unchecked(
                s.public_key().to_bytes(),
                s.signature().to_bytes(),
                packet.into(),
                seq,
                None,
            )
        }
    }

    impl TryFrom<&MutableItem> for SignedPacket {
        type Error = Error;

        fn try_from(i: &MutableItem) -> Result<Self> {
            let public_key = PublicKey::try_from(i.key()).unwrap();
            let timestamp = *i.seq() as u64;
            let signature: Signature = i.signature().into();

            let raw_message = i.value();
            public_key.verify(&signable(timestamp, raw_message), &signature)?;
            let message = Message::from_vec(raw_message)?;

            Ok(Self {
                inner: Arc::new(Inner {
                    public_key,
                    signature,
                    message,
                    timestamp,
                }),
                last_seen: system_time(),
            })
        }
    }
}
impl Display for SignedPacket {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SignedPacket ({}):\n    last_seen: {} seconds ago\n    timestamp: {},\n    signature: {}\n    records:\n",
            self.inner.public_key,
            self.elapsed(),
            self.inner.timestamp,
            self.inner.signature,
        )?;

        for answer in self.inner.message.answers() {
            writeln!(
                f,
                "        {}  IN  {}  {}\n",
                answer.name(),
                answer.ttl(),
                match answer.data() {
                    Some(RData::A(val)) => format!("A  {}", val.0),
                    Some(RData::AAAA(val)) => format!("AAAA  {}", val.0),
                    #[allow(clippy::to_string_in_format_args)]
                    Some(RData::CNAME(name)) => format!("CNAME  {}", name.to_string()),
                    Some(RData::TXT(txt)) => {
                        format!("TXT  \"{}\"", txt)
                    }
                    data => format!("{:?}", data),
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
    use std::net::Ipv4Addr;

    use hickory_proto::rr::{rdata, DNSClass, RecordType};

    use super::*;

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

        let message = new_message_a(Ipv4Addr::new(1, 1, 1, 1));

        let signed_packet = SignedPacket::from_packet(&keypair, &message).unwrap();

        assert!(SignedPacket::from_relay_payload(
            signed_packet.public_key().clone(),
            &signed_packet.to_relay_payload()
        )
        .is_ok());
    }

    #[test]
    fn from_too_large_bytes() {
        let keypair = Keypair::random();

        let bytes = vec![0; 1073];
        let error = SignedPacket::from_relay_payload(keypair.public_key(), &bytes);

        assert!(error.is_err());
    }

    fn new_message_a(addr: Ipv4Addr) -> Message {
        let mut message = Message::new();
        message.set_message_type(MessageType::Response);
        let record = new_record_a("_derp_region.iroh.", addr);
        message.add_answer(record);
        message
    }

    fn new_record_a(name: &str, addr: Ipv4Addr) -> Record {
        let mut record = Record::with(Name::from_ascii(name).unwrap(), RecordType::A, 30);
        record.set_dns_class(DNSClass::IN);
        record.set_data(Some(RData::A(rdata::a::A(addr))));
        record
    }

    fn new_record_txt(name: &str, txt: &str) -> Record {
        let mut record = Record::with(Name::from_ascii(name).unwrap(), RecordType::TXT, 30);
        record.set_dns_class(DNSClass::IN);
        record.set_data(Some(RData::TXT(rdata::TXT::new(vec![txt.to_string()]))));
        record
    }

    #[test]
    fn from_too_large_message() {
        let keypair = Keypair::random();

        let mut message = Message::new();
        message.set_message_type(MessageType::Response);
        for _ in 0..100 {
            let record = new_record_a("_derp_region.iroh.", Ipv4Addr::new(1, 1, 1, 1));
            message.add_answer(record);
        }

        let error = SignedPacket::from_packet(&keypair, &message);

        assert!(error.is_err());
    }

    #[test]
    fn resource_records_iterator() {
        let keypair = Keypair::random();

        let target = new_record_a("_derp_region.iroh.", Ipv4Addr::new(1, 1, 1, 1));
        let mut message = Message::new();
        message.set_message_type(MessageType::Response);

        message.add_answer(target.clone());
        message.add_answer(new_record_a("something_else", Ipv4Addr::new(1, 1, 1, 1)));
        let signed_packet = SignedPacket::from_packet(&keypair, &message).unwrap();

        let iter = signed_packet.resource_records("_derp_region.iroh");
        assert_eq!(iter.count(), 1);

        for record in signed_packet.resource_records("_derp_region.iroh") {
            assert_eq!(record.data(), target.data());
        }
    }

    #[test]
    fn to_mutable() {
        let keypair = Keypair::random();

        let mut message = Message::new();
        message.set_message_type(MessageType::Response);
        message.add_answer(new_record_a(
            "_derp_region.iroh.",
            Ipv4Addr::new(1, 1, 1, 1),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &message).unwrap();
        let item: MutableItem = (&signed_packet).into();
        let seq: i64 = signed_packet.timestamp() as i64;

        let expected = MutableItem::new(
            keypair.secret_key().into(),
            signed_packet.message().to_vec().unwrap().into(),
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

        let mut message = Message::new();
        message.set_message_type(MessageType::Response);

        // First record
        let mut record = Record::with(Name::from_ascii(".").unwrap(), RecordType::CNAME, 30);
        record.set_dns_class(DNSClass::IN);
        record.set_data(Some(RData::CNAME(rdata::CNAME(
            Name::from_ascii(name).unwrap(),
        ))));
        message.add_answer(record);

        // Second record
        let mut record = Record::with(Name::from_ascii(".").unwrap(), RecordType::CNAME, 30);
        record.set_dns_class(DNSClass::IN);
        record.set_data(Some(RData::CNAME(rdata::CNAME(
            Name::from_ascii(dup).unwrap(),
        ))));
        message.add_answer(record);

        let signed = SignedPacket::from_packet(&keypair, &message).unwrap();

        assert_eq!(
            signed
                .resource_records("@")
                .map(|r| r.data())
                .collect::<Vec<_>>(),
            message
                .answers()
                .iter()
                .map(|r| r.data())
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn to_bytes_from_bytes() {
        let keypair = Keypair::random();

        let mut message = Message::new();
        message.set_message_type(MessageType::Response);
        let record = new_record_txt("_foo", "hello");
        message.add_answer(record);

        let signed = SignedPacket::from_packet(&keypair, &message).unwrap();
        let bytes = signed.to_vec();
        let from_bytes = SignedPacket::from_bytes(&bytes).unwrap();
        assert_eq!(signed.to_vec(), from_bytes.to_vec());

        let from_bytes2 =
            SignedPacket::from_bytes_with_last_seen(&bytes, signed.last_seen).unwrap();
        assert_eq!(signed.to_vec(), from_bytes2.to_vec());

        let public_key = keypair.public_key();
        let payload = signed.to_relay_payload();
        let from_relay_payload = SignedPacket::from_relay_payload(public_key, &payload).unwrap();
        assert_eq!(signed.to_vec(), from_relay_payload.to_vec());
    }

    #[test]
    fn clone() {
        let keypair = Keypair::random();
        let mut message = Message::new();
        message.set_message_type(MessageType::Response);
        let record = new_record_txt("_foo", "hello");
        message.add_answer(record);

        let signed = SignedPacket::from_packet(&keypair, &message).unwrap();
        let cloned = signed.clone();

        assert_eq!(cloned.to_vec(), signed.to_vec());
    }

    #[test]
    fn expires_in_minimum_ttl() {
        let keypair = Keypair::random();
        let mut message = Message::new();
        message.set_message_type(MessageType::Response);
        let mut record = new_record_txt("_foo", "hello");
        record.set_ttl(10);
        message.add_answer(record);

        let mut signed = SignedPacket::from_packet(&keypair, &message).unwrap();

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
        let mut message = Message::new();
        message.set_message_type(MessageType::Response);
        let mut record = new_record_txt("_foo", "hello");
        record.set_ttl(3 * DEFAULT_MAXIMUM_TTL);
        message.add_answer(record);

        let mut signed = SignedPacket::from_packet(&keypair, &message).unwrap();

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
        let mut message = Message::new();
        message.set_message_type(MessageType::Response);
        let record = new_record_txt("_foo", "hello");
        message.add_answer(record);
        let mut record = new_record_txt("_foo", "world");
        record.set_ttl(60);
        message.add_answer(record);

        let mut signed = SignedPacket::from_packet(&keypair, &message).unwrap();

        signed.last_seen = system_time() - (30 * 1_000_000);

        assert_eq!(signed.fresh_resource_records("_foo").count(), 1);
    }

    #[test]
    fn ttl_empty() {
        let keypair = Keypair::random();
        let mut message = Message::new();
        message.set_message_type(MessageType::Response);
        let signed = SignedPacket::from_packet(&keypair, &message).unwrap();

        assert_eq!(signed.ttl(DEFAULT_MINIMUM_TTL, DEFAULT_MAXIMUM_TTL), 300);
    }

    #[test]
    fn ttl_with_records_less_than_minimum() {
        let keypair = Keypair::random();

        let mut message = Message::new();
        message.set_message_type(MessageType::Response);
        let mut record = new_record_txt("_foo", "hello");
        record.set_ttl(DEFAULT_MINIMUM_TTL / 2);
        message.add_answer(record);
        let mut record = new_record_txt("_foo", "world");
        record.set_ttl(DEFAULT_MINIMUM_TTL / 4);
        message.add_answer(record);

        let signed = SignedPacket::from_packet(&keypair, &message).unwrap();

        assert_eq!(
            signed.ttl(DEFAULT_MINIMUM_TTL, DEFAULT_MAXIMUM_TTL),
            DEFAULT_MINIMUM_TTL
        );

        assert_eq!(signed.ttl(0, DEFAULT_MAXIMUM_TTL), DEFAULT_MINIMUM_TTL / 4);
    }

    #[test]
    fn ttl_with_records_more_than_maximum() {
        let keypair = Keypair::random();

        let mut message = Message::new();
        message.set_message_type(MessageType::Response);
        let mut record = new_record_txt("_foo", "hello");
        record.set_ttl(DEFAULT_MAXIMUM_TTL * 2);
        message.add_answer(record);
        let mut record = new_record_txt("_foo", "world");
        record.set_ttl(DEFAULT_MAXIMUM_TTL * 4);
        message.add_answer(record);
        let signed = SignedPacket::from_packet(&keypair, &message).unwrap();

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
