use simple_dns::rdata::{RData, SVCB};

use crate::{
    error::{Error, Result},
    PkarrClient, PublicKey, SignedPacket,
};

const MAX_ENDPOINT_RESOLUTION_RECURSION: u8 = 3;

impl PkarrClient {
    pub fn resolve_endpoint(&self, target: &str) -> Result<Endpoint> {
        // TODO: cache the result of this function?

        let mut step = 0;
        let mut svcb: Option<Endpoint> = None;

        loop {
            let current = svcb.clone().map_or(target.to_string(), |s| s.target);
            if let Ok(tld) = PublicKey::try_from(current.clone()) {
                if let Ok(Some(signed_packet)) = self.resolve(&tld) {
                    if step >= MAX_ENDPOINT_RESOLUTION_RECURSION {
                        break;
                    };
                    step += 1;

                    // Choose most prior SVCB record
                    svcb = getx(&signed_packet, &current);

                    // Try wildcards
                    if svcb.is_none() {
                        let parts: Vec<&str> = current.split('.').collect();

                        for i in 1..parts.len() {
                            let xx = format!("*.{}", parts[i..].join("."));

                            svcb = getx(&signed_packet, &xx);

                            if svcb.is_some() {
                                break;
                            }
                        }
                    }

                    if step >= MAX_ENDPOINT_RESOLUTION_RECURSION {
                        break;
                    };
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if let Some(svcb) = svcb {
            if PublicKey::try_from(svcb.target.as_str()).is_err() {
                return Ok(svcb.clone());
            }
        }

        Err(Error::ResolveEndpoint(target.into()))
    }
}

#[derive(Debug, Clone)]
pub struct Endpoint {
    target: String,
}

fn getx(signed_packet: &SignedPacket, target: &str) -> Option<Endpoint> {
    signed_packet
        .resource_records(target)
        .fold(None, |prev: Option<SVCB>, answer| {
            if let Some(svcb) = match &answer.rdata {
                RData::SVCB(svcb) => Some(svcb),
                RData::HTTPS(curr) => Some(&curr.0),
                _ => None,
            } {
                let curr = svcb.clone();

                if curr.priority == 0 {
                    return Some(curr);
                }
                if let Some(prev) = &prev {
                    if curr.priority >= prev.priority {
                        return Some(curr);
                    }
                } else {
                    return Some(curr);
                }
            }

            prev
        })
        .map(|s| Endpoint {
            target: s.target.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        dns::{self, rdata::RData},
        mainline::Testnet,
        Keypair,
    };

    fn publish_packets(client: &PkarrClient, tree: Vec<Vec<(&str, RData)>>) -> Vec<Keypair> {
        let mut keypairs: Vec<Keypair> = Vec::with_capacity(tree.len());

        for node in tree {
            let mut packet = dns::Packet::new_reply(0);

            for record in node {
                packet.answers.push(dns::ResourceRecord::new(
                    dns::Name::new(record.0).unwrap(),
                    dns::CLASS::IN,
                    3600,
                    record.1,
                ));
            }

            let keypair = Keypair::random();

            let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

            keypairs.push(keypair);

            client.publish(&signed_packet).unwrap();
        }

        keypairs
    }

    #[test]
    fn resolve_direct_endpoint() {
        let testnet = Testnet::new(3);

        let client = PkarrClient::builder().testnet(&testnet).build().unwrap();

        let keypairs = publish_packets(
            &client,
            vec![vec![
                (
                    "foo",
                    RData::HTTPS(SVCB::new(0, "https.example.com".try_into().unwrap()).into()),
                ),
                // // Make sure HTTPS only follows HTTPs
                // (
                //     "foo",
                //     RData::SVCB(SVCB::new(0, "protocol.example.com".try_into().unwrap())),
                // ),
                (
                    "_foo",
                    RData::SVCB(SVCB::new(0, "protocol.example.com".try_into().unwrap())),
                ),
            ]],
        );

        let tld = keypairs.first().unwrap().public_key();

        let endpoint = client.resolve_endpoint(&format!("foo.{tld}")).unwrap();

        assert_eq!(endpoint.target, "https.example.com");

        let endpoint = client.resolve_endpoint(&format!("_foo.{tld}")).unwrap();

        assert_eq!(endpoint.target, "protocol.example.com");
    }
}
