//! Interpret exact-name routing records in a packet.

use std::{collections::HashSet, net::IpAddr, num::NonZeroU16};

use pkarr::dns::{
    rdata::{RData, SVCParam, SVCB},
    Name, ResourceRecord,
};

use crate::{Error, ServiceName, SignedPacket};

use super::{
    https_records::{https_alias, https_services},
    EndpointCandidate, ResolutionRequest,
};

pub(super) enum PacketRouting {
    Alias(ServiceName),
    Endpoints(Vec<EndpointCandidate>),
}

pub(super) fn route_packet(
    request: &ResolutionRequest,
    routing_owner: &ServiceName,
    packet: &SignedPacket,
) -> Result<PacketRouting, Error> {
    let records = records(packet, routing_owner.as_str());
    check_routing_cname(&records)?;
    if let Some(alias) = https_alias(&records) {
        if alias.target.iter().next().is_none() {
            return Err(Error::ServiceUnavailable);
        }
        let target =
            alias.target.to_string().parse().map_err(|_| {
                Error::Unsupported("HTTPS AliasMode target outside Pubky namespace")
            })?;
        return Ok(PacketRouting::Alias(target));
    }
    let services = https_services(&records, request.supported_protocols())?;
    if services.is_empty() {
        return Err(Error::NoCompatibleEndpoint);
    }
    let mut candidates = Vec::with_capacity(services.len());
    let mut has_supported_target = false;
    for (service, protocols) in services {
        if service.target.iter().next().is_some()
            && !same_name(&service.target, routing_owner.as_str())
        {
            // Unsupported targets must not suppress usable same-service routes.
            continue;
        }
        has_supported_target = true;
        let port = match service.get_param(3) {
            Some(SVCParam::Port(port)) => {
                NonZeroU16::new(*port).ok_or(Error::InvalidRecord("port zero"))?
            }
            _ => request.nonzero_url_port(),
        };
        let addresses = addresses(&records, service);
        if addresses.is_empty() {
            continue;
        }
        candidates.push(EndpointCandidate {
            addresses,
            port,
            priority: service.priority,
            protocols,
        });
    }
    if !has_supported_target {
        return Err(Error::Unsupported("alternate HTTPS target"));
    }
    if candidates.is_empty() {
        return Err(Error::NoAddresses);
    }
    candidates.sort_by_key(|candidate| candidate.priority);
    Ok(PacketRouting::Endpoints(candidates))
}

fn check_routing_cname(records: &[&ResourceRecord<'_>]) -> Result<(), Error> {
    if !records
        .iter()
        .any(|record| matches!(record.rdata, RData::CNAME(_)))
    {
        return Ok(());
    }
    if records.len() != 1 {
        return Err(Error::InvalidRecord("CNAME with other routing records"));
    }
    Err(Error::Unsupported("routing CNAME"))
}

pub(super) fn https_record_name(request: &ResolutionRequest) -> Result<ServiceName, Error> {
    // HTTPS uses the service name on port 443 and a port-prefixed name otherwise.
    if request.url_port() == 443 {
        Ok(request.service().clone())
    } else {
        format!("_{}._https.{}", request.url_port(), request.service()).parse()
    }
}

fn records<'a>(packet: &'a SignedPacket, name: &str) -> Vec<&'a ResourceRecord<'a>> {
    // Match exact names; SDK relative-name and wildcard lookup conveniences
    // must not cross the requested service's authorization boundary.
    packet
        .all_resource_records()
        .filter(|record| same_name(&record.name, name))
        .collect()
}

fn same_name(name: &Name<'_>, expected: &str) -> bool {
    let labels = name.iter();
    let expected = expected.split('.');
    labels.len() == expected.clone().count()
        && labels
            .zip(expected)
            .all(|(label, expected)| label.as_ref().eq_ignore_ascii_case(expected.as_bytes()))
}

fn addresses(records: &[&ResourceRecord<'_>], service: &SVCB<'_>) -> Vec<IpAddr> {
    let mut result: Vec<_> = records
        .iter()
        .filter_map(|record| match &record.rdata {
            RData::A(address) => Some(IpAddr::V4(address.address.into())),
            RData::AAAA(address) => Some(IpAddr::V6(address.address.into())),
            _ => None,
        })
        .collect();
    // Hints are signed by the same routing authority and do not affect TLS.
    // Scheduling address records versus hints remains the host's responsibility.
    for param in service.iter_params() {
        match param {
            SVCParam::Ipv4Hint(ips) => result.extend(ips.iter().map(|ip| IpAddr::V4((*ip).into()))),
            SVCParam::Ipv6Hint(ips) => result.extend(ips.iter().map(|ip| IpAddr::V6((*ip).into()))),
            _ => {}
        }
    }
    let mut seen = HashSet::new();
    result.retain(|address| seen.insert(*address));
    result
}
