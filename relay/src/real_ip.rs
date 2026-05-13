use std::{future::Future, net::IpAddr, pin::Pin};

use axum::{
    extract::Request,
    middleware::{from_fn, FromFnLayer, Next},
    response::Response,
};
use http::request;
use tower_governor::{
    errors::GovernorError,
    key_extractor::{KeyExtractor, PeerIpKeyExtractor, SmartIpKeyExtractor},
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RealIp(pub(crate) IpAddr);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Extracts a request's normalized real IP from extensions.
pub struct RealIpKeyExtractor;

impl KeyExtractor for RealIpKeyExtractor {
    type Key = IpAddr;

    fn extract<T>(&self, req: &request::Request<T>) -> Result<Self::Key, GovernorError> {
        req.extensions()
            .get::<RealIp>()
            .map(|real_ip| real_ip.0)
            .ok_or(GovernorError::UnableToExtractKey)
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn middleware(
    behind_proxy: bool,
) -> FromFnLayer<
    impl Fn(Request, Next) -> Pin<Box<dyn Future<Output = Response> + Send>>
        + Clone
        + Send
        + Sync
        + 'static,
    (),
    (Request,),
> {
    from_fn(move |request, next| {
        let future: Pin<Box<dyn Future<Output = Response> + Send>> =
            Box::pin(handle_request(request, next, behind_proxy));

        future
    })
}

async fn handle_request(mut request: Request, next: Next, behind_proxy: bool) -> Response {
    if let Some(real_ip) = extract_real_ip(&request, behind_proxy) {
        request.extensions_mut().insert(RealIp(real_ip));
    }

    next.run(request).await
}

fn extract_real_ip(request: &Request, behind_proxy: bool) -> Option<IpAddr> {
    let extracted_ip = if behind_proxy {
        SmartIpKeyExtractor.extract(request)
    } else {
        PeerIpKeyExtractor.extract(request)
    };

    extracted_ip.ok()
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    use axum::{body::Body, extract::ConnectInfo};
    use http::Request;
    use tower_governor::key_extractor::KeyExtractor;

    use super::{extract_real_ip, RealIp, RealIpKeyExtractor};

    fn request_with_peer(peer_ip: Ipv4Addr) -> Request<Body> {
        let mut request = Request::new(Body::empty());
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from((peer_ip, 8080))));
        request
    }

    #[test]
    fn real_ip_uses_peer_ip_without_proxy_mode() {
        let mut request = request_with_peer(Ipv4Addr::new(192, 0, 2, 1));
        request
            .headers_mut()
            .insert("x-forwarded-for", "198.51.100.1".parse().unwrap());

        assert_eq!(
            extract_real_ip(&request, false),
            Some(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)))
        );
    }

    #[test]
    fn real_ip_uses_forwarded_ip_with_proxy_mode() {
        let mut request = request_with_peer(Ipv4Addr::new(192, 0, 2, 1));
        request
            .headers_mut()
            .insert("x-forwarded-for", "198.51.100.1".parse().unwrap());

        assert_eq!(
            extract_real_ip(&request, true),
            Some(IpAddr::V4(Ipv4Addr::new(198, 51, 100, 1)))
        );
    }

    #[test]
    fn real_ip_key_extractor_reads_extension() {
        let mut request = Request::new(Body::empty());
        let ip = IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1));
        request.extensions_mut().insert(RealIp(ip));

        assert_eq!(RealIpKeyExtractor.extract(&request).unwrap(), ip);
    }
}
