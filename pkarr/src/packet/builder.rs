use crate::prelude::*;

use crate::keys::Keypair;
use crate::SignedPacket;
use simple_dns::{
    rdata::{RData, A, AAAA, CNAME, TXT},
    Name, Packet, ResourceRecord, CLASS,
};
use std::net::IpAddr;

pub struct PacketBuilder<'a> {
    packet: Packet<'a>,
    keypair: &'a Keypair,
}

const DEFAULT_TTL: u32 = 3600;

impl<'a> PacketBuilder<'a> {
    pub fn new(keypair: &'a Keypair) -> PacketBuilder<'a> {
        PacketBuilder {
            packet: Packet::new_reply(0),
            keypair,
        }
    }

    /// Add an A or AAAA record from an IP addresso.
    ///
    /// Sets the TTL to [DEFAULT_TTL], to customize the TTL use [PacketBuilder::add_ip_with_ttl]
    ///
    /// # Examples
    ///
    /// ```
    /// use pkarr::{Keypair, PacketBuilder};
    /// use std::net::{Ipv4Addr, Ipv6Addr};
    ///
    /// let keypair = Keypair::random();
    ///
    /// let signed_packet = PacketBuilder::new(&keypair)
    ///     .add_ip("@", Ipv4Addr::new(1,1,1,1).into())
    ///     .add_ip("_foo", Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0).into())
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn add_ip(&mut self, name: &'a str, addr: IpAddr) -> &mut Self {
        self.add_ip_with_ttl(name, addr, DEFAULT_TTL)
    }

    /// Add an A or AAAA record from an IP addresso.
    ///
    /// # Examples
    ///
    /// ```
    /// use pkarr::{Keypair, PacketBuilder};
    /// use std::net::{Ipv4Addr};
    ///
    /// let keypair = Keypair::random();
    ///
    /// let signed_packet = PacketBuilder::new(&keypair)
    ///     .add_ip_with_ttl("@", Ipv4Addr::new(1,1,1,1).into(), 30)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn add_ip_with_ttl(&mut self, name: &'a str, addr: IpAddr, ttl: u32) -> &mut Self {
        let name = Name::new(name).unwrap();

        let record = match addr {
            IpAddr::V4(ip) => ResourceRecord::new(name, CLASS::IN, ttl, RData::A(A::from(ip))),
            IpAddr::V6(ip) => {
                ResourceRecord::new(name, CLASS::IN, ttl, RData::AAAA(AAAA::from(ip)))
            }
        };

        self.packet.answers.push(record);

        self
    }

    pub fn build(&self) -> Result<SignedPacket> {
        SignedPacket::new(&self.keypair, &self.packet)
    }
}
