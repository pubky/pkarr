//! Select HTTPS ServiceMode records and compatible application protocols.

use pkarr::dns::{
    rdata::{RData, SVCParam, SVCB},
    ResourceRecord,
};

use crate::Error;

use super::HttpProtocol;

pub(super) fn https_services<'a>(
    records: &[&'a ResourceRecord<'a>],
    supported_protocols: &[HttpProtocol],
) -> Result<Vec<(&'a SVCB<'a>, Vec<HttpProtocol>)>, Error> {
    let mut services = Vec::new();
    for record in records {
        let RData::HTTPS(https) = &record.rdata else {
            continue;
        };
        let protocols = compatible_protocols(&https.0, supported_protocols)?;
        if protocols.is_empty() {
            continue;
        }
        services.push((&https.0, protocols));
    }
    Ok(services)
}

/// Select the first alias in signed packet order so bundle replay follows the
/// same path. Alias parameters and all ServiceMode records are ignored.
pub(super) fn https_alias<'a>(records: &[&'a ResourceRecord<'a>]) -> Option<&'a SVCB<'a>> {
    records.iter().find_map(|record| match &record.rdata {
        RData::HTTPS(https) if https.0.priority == 0 => Some(&https.0),
        _ => None,
    })
}

fn compatible_protocols(
    service: &SVCB<'_>,
    supported_protocols: &[HttpProtocol],
) -> Result<Vec<HttpProtocol>, Error> {
    let no_default = service.get_param(2).is_some();
    let alpn = match service.get_param(1) {
        Some(SVCParam::Alpn(alpn)) => Some(alpn),
        _ => None,
    };
    if no_default && alpn.is_none() {
        return Err(Error::InvalidRecord("no-default-alpn requires alpn"));
    }
    if let Some(SVCParam::Mandatory(keys)) = service.get_param(0) {
        if keys.iter().any(|key| service.get_param(*key).is_none()) {
            return Err(Error::InvalidRecord("mandatory parameter is absent"));
        }
        // ECH and unknown keys cannot be declared supported merely because
        // their bytes can be parsed. Optional unsupported parameters are ignored.
        if keys.iter().any(|key| !matches!(key, 1..=4 | 6)) {
            return Ok(Vec::new());
        }
    }
    Ok(supported_protocols
        .iter()
        .copied()
        .filter(|protocol| {
            let alpn_id = protocol
                .alpn()
                .try_into()
                .expect("static ALPN fits a DNS character string");
            (!no_default && *protocol == HttpProtocol::Http11)
                || alpn.is_some_and(|ids| ids.contains(&alpn_id))
        })
        .collect())
}
