use crate::{
    dns::{
        rdata::{RData, SVCB},
        ResourceRecord,
    },
    SignedPacket,
};
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};

use crate::Timestamp;

#[derive(Debug, Clone)]
/// An alternative Endpoint for a `qname`, from either [RData::SVCB] or [RData::HTTPS] dns records
pub struct Endpoint {
    pub(crate) target: String,
    // public_key: PublicKey,
    pub(crate) port: u16,
    pub(crate) addrs: Vec<IpAddr>,
}

impl Endpoint {
    /// 1. Find the SVCB or HTTPS records with the lowest priority
    /// 2. Choose a random one of the list of the above
    /// 3. If the target is `.`, check A and AAAA records see [rfc9460](https://www.rfc-editor.org/rfc/rfc9460#name-special-handling-of-in-targ)
    pub(crate) fn find(
        signed_packet: &SignedPacket,
        target: &str,
        is_svcb: bool,
    ) -> Option<Endpoint> {
        let mut lowest_priority = u16::MAX;
        let mut lowest_priority_index = 0;
        let mut records = vec![];

        for record in signed_packet.resource_records(target) {
            if let Some(svcb) = get_svcb(record, is_svcb) {
                match svcb.priority.cmp(&lowest_priority) {
                    std::cmp::Ordering::Equal => records.push(svcb),
                    std::cmp::Ordering::Less => {
                        lowest_priority_index = records.len();
                        lowest_priority = svcb.priority;
                        records.push(svcb)
                    }
                    _ => {}
                }
            }
        }

        // Good enough random selection
        let now = Timestamp::now().into_u64();
        let slice = &records[lowest_priority_index..];
        let index = if slice.is_empty() {
            0
        } else {
            (now as usize) % slice.len()
        };

        slice.get(index).map(|s| {
            let target = s.target.to_string();

            let mut addrs: Vec<IpAddr> = vec![];

            if &target == "." {
                for record in signed_packet.resource_records("@") {
                    match &record.rdata {
                        RData::A(ip) => addrs.push(IpAddr::V4(ip.address.into())),
                        RData::AAAA(ip) => addrs.push(IpAddr::V6(ip.address.into())),
                        _ => {}
                    }
                }
            }

            Endpoint {
                target,
                // public_key: signed_packet.public_key(),
                port: u16::from_be_bytes(
                    s.get_param(SVCB::PORT)
                        .unwrap_or_default()
                        .try_into()
                        .unwrap_or([0, 0]),
                ),
                addrs,
            }
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Return an iterator of [SocketAddr], either by resolving the [Endpoint::target] using normal DNS,
    /// or, if the target is ".", return the [RData::A] or [RData::AAAA] records
    /// from the endpoint's [SignedPacket], if available.
    pub fn to_socket_addrs(&self) -> std::io::Result<std::vec::IntoIter<SocketAddr>> {
        if self.target == "." {
            let port = self.port;
            return Ok(self
                .addrs
                .iter()
                .map(|addr| SocketAddr::from((*addr, port)))
                .collect::<Vec<_>>()
                .into_iter());
        }

        format!("{}:{}", self.target, self.port).to_socket_addrs()
    }
}

fn get_svcb<'a>(record: &'a ResourceRecord, is_svcb: bool) -> Option<&'a SVCB<'a>> {
    match &record.rdata {
        RData::SVCB(svcb) => {
            if is_svcb {
                Some(svcb)
            } else {
                None
            }
        }

        RData::HTTPS(curr) => {
            if is_svcb {
                None
            } else {
                Some(&curr.0)
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, Ipv6Addr};
    use std::str::FromStr;

    use super::*;

    use crate::{dns, Keypair};

    #[tokio::test]
    async fn endpoint_target() {
        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("foo").unwrap(),
            dns::CLASS::IN,
            3600,
            RData::HTTPS(SVCB::new(0, "https.example.com".try_into().unwrap()).into()),
        ));
        // Make sure HTTPS only follows HTTPs
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("foo").unwrap(),
            dns::CLASS::IN,
            3600,
            RData::SVCB(SVCB::new(0, "protocol.example.com".try_into().unwrap())),
        ));
        // Make sure SVCB only follows SVCB
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("foo").unwrap(),
            dns::CLASS::IN,
            3600,
            RData::HTTPS(SVCB::new(0, "https.example.com".try_into().unwrap()).into()),
        ));
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            3600,
            RData::SVCB(SVCB::new(0, "protocol.example.com".try_into().unwrap())),
        ));
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

        let tld = keypair.public_key();

        // Follow foo.tld HTTPS records
        let endpoint = Endpoint::find(&signed_packet, &format!("foo.{tld}"), false).unwrap();
        assert_eq!(endpoint.target, "https.example.com");

        // Follow _foo.tld SVCB records
        let endpoint = Endpoint::find(&signed_packet, &format!("_foo.{tld}"), true).unwrap();
        assert_eq!(endpoint.target, "protocol.example.com");
    }

    #[test]
    fn endpoint_to_socket_addrs() {
        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("@").unwrap(),
            dns::CLASS::IN,
            3600,
            RData::A(Ipv4Addr::from_str("209.151.148.15").unwrap().into()),
        ));
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("@").unwrap(),
            dns::CLASS::IN,
            3600,
            RData::AAAA(Ipv6Addr::from_str("2a05:d014:275:6201::64").unwrap().into()),
        ));

        let mut svcb = SVCB::new(1, ".".try_into().unwrap());
        svcb.set_port(6881);

        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("@").unwrap(),
            dns::CLASS::IN,
            3600,
            RData::HTTPS(svcb.into()),
        ));
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

        // Follow foo.tld HTTPS records
        let endpoint = Endpoint::find(
            &signed_packet,
            &signed_packet.public_key().to_string(),
            false,
        )
        .unwrap();

        assert_eq!(endpoint.target, ".");

        let addrs = endpoint.to_socket_addrs().unwrap();
        assert_eq!(
            addrs.map(|s| s.to_string()).collect::<Vec<_>>(),
            vec!["209.151.148.15:6881", "[2a05:d014:275:6201::64]:6881"]
        )
    }
}
