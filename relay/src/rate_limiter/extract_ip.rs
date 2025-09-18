use axum::extract::Request;
use axum::http::HeaderMap;
use std::net::{IpAddr, SocketAddr};

// From https://github.com/benwis/tower-governor/blob/main/src/key_extractor.rs#L121
const X_REAL_IP: &str = "x-real-ip";
const X_FORWARDED_FOR: &str = "x-forwarded-for";

/// Tries to parse the `x-forwarded-for` header
fn maybe_x_forwarded_for(headers: &HeaderMap) -> Option<IpAddr> {
    headers
        .get(X_FORWARDED_FOR)
        .and_then(|hv| hv.to_str().ok())
        .and_then(|s| s.split(',').find_map(|s| s.trim().parse::<IpAddr>().ok()))
}

/// Tries to parse the `x-real-ip` header
fn maybe_x_real_ip(headers: &HeaderMap) -> Option<IpAddr> {
    headers
        .get(X_REAL_IP)
        .and_then(|hv| hv.to_str().ok())
        .and_then(|s| s.parse::<IpAddr>().ok())
}

fn maybe_connect_info<T>(req: &Request<T>) -> Option<IpAddr> {
    req.extensions()
        .get::<axum::extract::ConnectInfo<SocketAddr>>()
        .map(|addr| addr.ip())
}

pub fn extract_ip<T>(req: &Request<T>) -> anyhow::Result<IpAddr> {
    let headers = req.headers();
    maybe_x_forwarded_for(headers)
        .or_else(|| maybe_x_real_ip(headers))
        .or_else(|| maybe_connect_info(req))
        .ok_or(anyhow::anyhow!("Failed to extract ip."))
}
