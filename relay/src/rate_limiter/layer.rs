use axum::response::{IntoResponse, Response};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use futures_util::future::BoxFuture;
use governor::clock::QuantaClock;
use governor::state::keyed::DashMapStateStore;
use std::num::NonZero;
use std::sync::Arc;
use std::time::Duration;
use std::{convert::Infallible, task::Poll};
use tower::{Layer, Service};

use crate::error::Error as HttpError;
use axum::http::Method;
use serde::{Deserialize, Deserializer, Serialize};
use std::net::IpAddr;
use std::str::FromStr;

/// Custom deserializer for whitelist that converts strings to LimitKey::IpNetwork
fn deserialize_whitelist<'de, D>(deserializer: D) -> Result<Vec<LimitKey>, D::Error>
where
    D: Deserializer<'de>,
{
    let strings: Vec<String> = Vec::deserialize(deserializer)?;
    strings
        .into_iter()
        .map(|s| LimitKey::from_str(&s).map_err(serde::de::Error::custom))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// Key used for rate limiting
pub enum LimitKey {
    /// IP network (supports individual IPs and CIDR blocks)
    IpNetwork(ipnet::IpNet),
}

impl LimitKey {
    /// Check if an IP address matches this limit key
    pub fn matches_ip(&self, ip: &IpAddr) -> bool {
        match self {
            LimitKey::IpNetwork(network) => network.contains(ip),
        }
    }
}

impl std::fmt::Display for LimitKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LimitKey::IpNetwork(network) => write!(f, "{}", network),
        }
    }
}

impl std::str::FromStr for LimitKey {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Try parsing as CIDR first
        if let Ok(network) = s.parse::<ipnet::IpNet>() {
            return Ok(LimitKey::IpNetwork(network));
        }

        // If that fails, try parsing as individual IP and convert to single-host network
        if let Ok(ip) = s.parse::<IpAddr>() {
            let network = match ip {
                IpAddr::V4(ipv4) => ipnet::IpNet::from(
                    ipnet::Ipv4Net::new(ipv4, 32)
                        .map_err(|e| format!("Failed to create IPv4 network: {}", e))?,
                ),
                IpAddr::V6(ipv6) => ipnet::IpNet::from(
                    ipnet::Ipv6Net::new(ipv6, 128)
                        .map_err(|e| format!("Failed to create IPv6 network: {}", e))?,
                ),
            };
            return Ok(LimitKey::IpNetwork(network));
        }

        Err(format!("Invalid IP address or CIDR block: {}", s))
    }
}

/// Speed rate unit for rate limiting
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpeedRateUnit {
    /// Kilobyte
    Kilobyte,
    /// Megabyte
    Megabyte,
    /// Gigabyte
    Gigabyte,
}

impl SpeedRateUnit {
    /// Returns the number of bytes for this unit
    pub const fn multiplier(&self) -> std::num::NonZeroU32 {
        match self {
            // Speed quotas are always in kilobytes.
            // Because counting bytes is not practical and we are limited to u32 = 4GB max.
            // Counting in kb as more practical and we can count up to 4GB*1024 = 4TB.
            SpeedRateUnit::Kilobyte => std::num::NonZeroU32::new(1).expect("Is always non-zero"),
            SpeedRateUnit::Megabyte => std::num::NonZeroU32::new(1024).expect("Is always non-zero"),
            SpeedRateUnit::Gigabyte => {
                std::num::NonZeroU32::new(1024 * 1024).expect("Is always non-zero")
            }
        }
    }
}

impl std::fmt::Display for SpeedRateUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                SpeedRateUnit::Kilobyte => "kb",
                SpeedRateUnit::Megabyte => "mb",
                SpeedRateUnit::Gigabyte => "gb",
            }
        )
    }
}

impl FromStr for SpeedRateUnit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "kb" => Ok(SpeedRateUnit::Kilobyte),
            "mb" => Ok(SpeedRateUnit::Megabyte),
            "gb" => Ok(SpeedRateUnit::Gigabyte),
            _ => Err(format!("Invalid speedrate unit: {}", s)),
        }
    }
}

/// The unit of the rate.
///
/// Examples:
/// - "r" -> request
/// - "kb" -> kilobyte
/// - "mb" -> megabyte
/// - "gb" -> gigabyte
/// - "tb" -> terabyte
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RateUnit {
    /// Request
    Request,
    /// Speed rate unit
    SpeedRateUnit(SpeedRateUnit),
}

impl RateUnit {
    /// Returns true if the rate unit is a speed rate unit.
    pub fn is_speed_rate_unit(&self) -> bool {
        matches!(self, RateUnit::SpeedRateUnit(_))
    }
}

impl std::fmt::Display for RateUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RateUnit::Request => "r".to_string(),
                RateUnit::SpeedRateUnit(unit) => unit.to_string(),
            }
        )
    }
}

impl FromStr for RateUnit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "r" => Ok(RateUnit::Request),
            other => match SpeedRateUnit::from_str(other) {
                Ok(unit) => Ok(RateUnit::SpeedRateUnit(unit)),
                Err(_) => Err(format!("Invalid rate unit: {}", s)),
            },
        }
    }
}

impl RateUnit {
    /// Returns the number of bytes for this unit
    pub const fn multiplier(&self) -> std::num::NonZeroU32 {
        match self {
            RateUnit::Request => std::num::NonZeroU32::new(1).expect("Is always non-zero"),
            RateUnit::SpeedRateUnit(unit) => unit.multiplier(),
        }
    }
}

/// The time unit of the quota.
///
/// Examples:
/// - "s" -> second
/// - "m" -> minute
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    /// Second
    Second,
    /// Minute
    Minute,
}

impl std::fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                TimeUnit::Second => "s",
                TimeUnit::Minute => "m",
            }
        )
    }
}

impl std::str::FromStr for TimeUnit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "s" => Ok(TimeUnit::Second),
            "m" => Ok(TimeUnit::Minute),
            _ => Err(format!("Invalid time unit: {}", s)),
        }
    }
}

impl TimeUnit {
    /// Returns the number of seconds for each unit
    pub const fn multiplier_in_seconds(&self) -> std::num::NonZeroU32 {
        match self {
            TimeUnit::Second => std::num::NonZeroU32::new(1).expect("Is always non-zero"),
            TimeUnit::Minute => std::num::NonZeroU32::new(60).expect("Is always non-zero"),
        }
    }
}

/// Quota value
///
/// Examples:
/// - 5r/m
/// - 5r/s
/// - 5kb/m
/// - 5mb/m
/// - 5gb/s
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuotaValue {
    /// The rate.
    pub rate: std::num::NonZeroU32,
    /// The unit of the rate.
    pub rate_unit: RateUnit,
    /// The unit of the time.
    pub time_unit: TimeUnit,
}

impl From<QuotaValue> for governor::Quota {
    /// Get the quota to do the actual rate limiting.
    ///
    /// Important: The speed quotas are always in kilobytes, not bytes.
    /// Counting bytes is not practical.
    ///
    fn from(value: QuotaValue) -> Self {
        let rate_count = value.rate.get();
        let rate_unit = value.rate_unit.multiplier().get();
        let rate = std::num::NonZeroU32::new(rate_count * rate_unit)
            .expect("Is always non-zero because rate count and rate unit multiplier are non-zero");
        let time_unit = Duration::from_secs(value.time_unit.multiplier_in_seconds().get() as u64);
        let replenish_1_per = time_unit / rate.get();

        let base_quota = governor::Quota::with_period(replenish_1_per)
            .expect("Is always non-zero because replenish_1_per is non-zero");
        base_quota.allow_burst(rate)
    }
}

impl std::fmt::Display for QuotaValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}/{}", self.rate, self.rate_unit, self.time_unit)
    }
}

impl FromStr for QuotaValue {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Split rate part by '/' to get rate+unit and time unit
        let rate_parts: Vec<&str> = s.split('/').collect();
        if rate_parts.len() != 2 {
            return Err(format!(
                "Invalid rate format: '{}', expected {{rate}}{{unit}}/{{time}}",
                s
            ));
        }

        let rate_with_unit = rate_parts[0];
        let time_unit = TimeUnit::from_str(rate_parts[1])?;

        // Find the boundary between rate digits and unit
        let rate_digit_end = rate_with_unit
            .chars()
            .position(|c| !c.is_ascii_digit())
            .unwrap_or(rate_with_unit.len());

        if rate_digit_end == 0 {
            return Err(format!("Missing rate value in '{}'", rate_with_unit));
        }

        let rate_str = &rate_with_unit[..rate_digit_end];
        let rate_unit_str = &rate_with_unit[rate_digit_end..];

        let rate = rate_str
            .parse::<std::num::NonZeroU32>()
            .map_err(|_| format!("Failed to parse rate from '{}'", rate_str))?;
        let rate_unit = RateUnit::from_str(rate_unit_str)?;

        Ok(QuotaValue {
            rate,
            rate_unit,
            time_unit,
        })
    }
}

impl serde::Serialize for QuotaValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for QuotaValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;

        // Parse the quota string
        QuotaValue::from_str(&s).map_err(serde::de::Error::custom)
    }
}

/// Relay operation type for rate limiting
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    /// Publishing records (PUT requests)
    Publish,
    /// Resolving records (GET requests without most_recent=true)
    Resolve,
    /// Resolving most recent records (GET requests with most_recent=true)
    ResolveMostRecent,
}

impl Operation {
    /// Check if a request matches this operation
    pub fn matches_request(&self, req: &Request<Body>) -> bool {
        match self {
            Operation::Publish => req.method() == Method::PUT,
            Operation::Resolve => {
                if req.method() != Method::GET {
                    return false;
                }
                // Check if most_recent query parameter is present and true
                if let Some(query_string) = req.uri().query() {
                    !query_string.contains("most_recent=true")
                } else {
                    true // No query parameters, so it's a regular resolve
                }
            }
            Operation::ResolveMostRecent => {
                if req.method() != Method::GET {
                    return false;
                }
                // Check if most_recent query parameter is present and true
                if let Some(query_string) = req.uri().query() {
                    query_string.contains("most_recent=true")
                } else {
                    false
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
/// Operation-based rate limit configuration
pub struct OperationLimit {
    /// Operation type
    pub operation: Operation,
    /// Rate quota
    pub quota: QuotaValue,
    /// Burst capacity
    pub burst: Option<NonZero<u32>>,
    /// Whitelisted keys (IP networks/CIDR blocks)
    #[serde(default, deserialize_with = "deserialize_whitelist")]
    pub whitelist: Vec<LimitKey>,
}

impl OperationLimit {
    /// Create new operation limit
    pub fn new(operation: Operation, quota: QuotaValue, burst: Option<NonZero<u32>>) -> Self {
        Self {
            operation,
            quota,
            burst,
            whitelist: vec![],
        }
    }

    /// Check if an IP address is whitelisted
    pub fn is_whitelisted(&self, ip: &IpAddr) -> bool {
        self.whitelist
            .iter()
            .any(|whitelist_entry| whitelist_entry.matches_ip(ip))
    }
}

impl std::fmt::Display for OperationLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let burst_str = self
            .burst
            .map(|b| format!(" burst {}", b))
            .unwrap_or("".to_string());
        write!(
            f,
            "{:?}: {}{}. whitelist: {:?}",
            self.operation, self.quota, burst_str, self.whitelist
        )
    }
}

impl From<OperationLimit> for governor::Quota {
    fn from(value: OperationLimit) -> Self {
        let quota: governor::Quota = value.quota.into();
        if let Some(burst) = value.burst {
            quota.allow_burst(burst);
        }
        quota
    }
}
use futures_util::StreamExt;
use governor::{Jitter, Quota, RateLimiter};

use super::extract_ip::extract_ip;

/// A Tower Layer to handle general rate limiting.
///
/// Supports rate limiting by request count and by upload/download speed.
///
/// Requires a `PubkyHostLayer` to be applied first.
/// Used to extract the user pubkey as the key for the rate limiter.
///
/// Returns 400 BAD REQUEST if the user pubkey aka pubky-host cannot be extracted.
///
#[derive(Debug, Clone)]
pub struct RateLimiterLayer {
    limits: Vec<OperationLimit>,
}

impl RateLimiterLayer {
    /// Create a new rate limiter layer with operation-based limits.
    pub fn new(limits: Vec<OperationLimit>) -> Self {
        if limits.is_empty() {
            tracing::info!("Rate limiting is disabled.");
        } else {
            let limits_str = limits
                .iter()
                .map(|limit| format!("\"{limit}\""))
                .collect::<Vec<String>>();
            tracing::info!("Rate limits configured: {}", limits_str.join(", "));
        }
        Self { limits }
    }
}

impl<S> Layer<S> for RateLimiterLayer {
    type Service = RateLimiterMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        let tuples = self
            .limits
            .iter()
            .map(|op| LimitTuple::new(op.clone()))
            .collect::<Vec<_>>();

        RateLimiterMiddleware {
            inner,
            limits: tuples,
        }
    }
}

/// A tuple of an operation limit and the actual governor rate limiter.
#[derive(Debug, Clone)]
struct LimitTuple {
    pub limit: OperationLimit,
    pub limiter: Arc<RateLimiter<LimitKey, DashMapStateStore<LimitKey>, QuantaClock>>,
}

impl LimitTuple {
    /// Create new operation tuple
    pub fn new(operation_limit: OperationLimit) -> Self {
        let quota: Quota = operation_limit.clone().into();
        let limiter = Arc::new(RateLimiter::keyed(quota));

        // Forget keys that are not used anymore. This is to prevent memory leaks.
        let limiter_clone = limiter.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            interval.tick().await;
            loop {
                interval.tick().await;
                limiter_clone.retain_recent();
                limiter_clone.shrink_to_fit();
            }
        });

        Self {
            limit: operation_limit,
            limiter,
        }
    }

    /// Extract the key from the request.
    ///
    /// The key is the IP address of the client as a single-host network.
    fn extract_key(&self, req: &Request<Body>) -> anyhow::Result<LimitKey> {
        let ip = extract_ip(req)?;
        // Convert IP to a /32 or /128 network for consistent matching
        let network = match ip {
            IpAddr::V4(ipv4) => ipnet::IpNet::from(ipnet::Ipv4Net::new(ipv4, 32)?),
            IpAddr::V6(ipv6) => ipnet::IpNet::from(ipnet::Ipv6Net::new(ipv6, 128)?),
        };
        Ok(LimitKey::IpNetwork(network))
    }

    /// Check if the request matches the limit.
    pub fn is_match(&self, req: &Request<Body>) -> bool {
        self.limit.operation.matches_request(req)
    }
}

#[derive(Debug, Clone)]
/// Rate limiter middleware
pub struct RateLimiterMiddleware<S> {
    inner: S,
    limits: Vec<LimitTuple>,
}

impl<S> RateLimiterMiddleware<S> {
    /// Throttle the download body.
    fn throttle_download(
        res: Response<Body>,
        key: &LimitKey,
        limiter: &Arc<RateLimiter<LimitKey, DashMapStateStore<LimitKey>, QuantaClock>>,
    ) -> Response<Body> {
        let (parts, body) = res.into_parts();
        let new_body = Self::throttle_body(body, key, limiter);
        Response::from_parts(parts, new_body)
    }

    /// Throttle the up or download body.
    ///
    /// Important: The speed quotas are always in kilobytes, not bytes.
    /// Counting bytes is not practical.
    ///
    fn throttle_body(
        body: Body,
        key: &LimitKey,
        limiter: &Arc<RateLimiter<LimitKey, DashMapStateStore<LimitKey>, QuantaClock>>,
    ) -> Body {
        let body_stream = body.into_data_stream();
        let limiter = limiter.clone();
        let key = key.clone();
        let throttled = body_stream
            .map(move |chunk| {
                let limiter = limiter.clone();
                let key = key.clone();
                // When the rate limit is exceeded, we wait between 25ms and 500ms before retrying.
                // This is to avoid overwhelming the server with requests when the rate limit is exceeded.
                // Randomization is used to avoid thundering herd problem.
                let jitter = Jitter::new(
                    Duration::from_millis(25),
                    Duration::from_millis(500),
                );
                async move {
                    let bytes = match chunk {
                        Ok(actual_chunk) => {
                            actual_chunk
                        }
                        Err(e) => return Err(e),
                    };

                    // --- Round up to the nearest kilobyte. ---
                    // Important: If the chunk is < 1KB, it will be rounded up to 1 kb.
                    // Many small uploads will be counted as more than they actually are.
                    // I am not too concerned about this though because small random disk writes are stressing 
                    // the disk more anyway compared to larger writes.
                    // Why are we doing this? governor::Quota is defined as a u32. u32 can only count up to 4GB.
                    // To support 4GB/s+ limits we need to count in kilobytes.
                    //
                    // --- Chunk Size ---
                    // The chunk size is determined by the client library.
                    // Common chunk sizes: 16KB to 10MB. 
                    // HTTP based uploads are usually between 256KB and 1MB.
                    // Asking the limiter for 1KB packets is tradeoff between
                    // - Not calling the limiter too much
                    // - Guaranteeing the call size (1kb) is low enough to not cause race condition issues.
                    let chunk_kilobytes = bytes.len().div_ceil(1024);
                    for _ in 0..chunk_kilobytes {
                        // Check each kilobyte
                        if limiter
                            .until_key_n_ready_with_jitter(
                                &key,
                                NonZero::new(1).expect("1 is always non zero"),
                                jitter,
                            )
                            .await.is_err()
                        {
                            // Requested rate (1kb) is higher then the set limit (x kb/s).
                            // This should never happen.
                            unreachable!("Rate limiting is based on the number of kilobytes, not bytes. So 1 kb should always be allowed.");
                        };
                    }
                    Ok(bytes)
                }
            })
            .buffered(1);

        Body::from_stream(throttled)
    }

    /// Get the limits that match the request.
    fn get_limit_matches(&self, req: &Request<Body>) -> Vec<&LimitTuple> {
        self.limits
            .iter()
            .filter(|limit| limit.is_match(req))
            .collect()
    }
}

impl<S> Service<Request<Body>> for RateLimiterMiddleware<S>
where
    S: Service<Request<Body>, Response = axum::response::Response, Error = Infallible>
        + Send
        + 'static
        + Clone,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut std::task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(|_| unreachable!()) // `Infallible` conversion
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let mut inner = self.inner.clone();

        // Match the request to the defined limits.
        let limits = self.get_limit_matches(&req);
        if limits.is_empty() {
            // No limits matched, so we can just call the next layer.
            return Box::pin(async move { inner.call(req).await.map_err(|_| unreachable!()) });
        }

        // Go through all the limits and check if we need to throttle or reject the request.
        for limit in limits.clone() {
            let key = match limit.extract_key(&req) {
                Ok(key) => key,
                Err(e) => {
                    tracing::warn!(
                        "{:?} Failed to extract key for rate limiting: {}",
                        limit.limit.operation,
                        e
                    );
                    return Box::pin(async move {
                        Ok(HttpError::new(
                            StatusCode::BAD_REQUEST,
                            Some("Failed to extract key for rate limiting"),
                        )
                        .into_response())
                    });
                }
            };

            // Check whitelist using the original IP address
            let ip = extract_ip(&req)
                .unwrap_or_else(|_| "127.0.0.1".parse().expect("localhost IP is valid"));
            if limit.limit.is_whitelisted(&ip) {
                continue;
            }

            if !limit.limit.quota.rate_unit.is_speed_rate_unit() {
                // Request limiting is enabled, so we need to limit the number of requests.
                if let Err(_e) = limit.limiter.check_key(&key) {
                    tracing::debug!("Rate limit of {} exceeded for {:?}", limit.limit.quota, key);
                    return Box::pin(async move {
                        Ok(HttpError::new(
                            StatusCode::TOO_MANY_REQUESTS,
                            Some("Rate limit exceeded"),
                        )
                        .into_response())
                    });
                };
            }
        }

        // Create a clone of the request without the body.
        // This way, we can extract the keys for the response too.
        let (parts, body) = req.into_parts();
        let req_clone = Request::from_parts(parts.clone(), Body::empty());
        let req = Request::from_parts(parts, body);

        let speed_limits = limits
            .into_iter()
            .filter(|limit| limit.limit.quota.rate_unit.is_speed_rate_unit())
            .cloned()
            .collect::<Vec<_>>();
        Box::pin(async move {
            // Call the next layer and receive the response.
            let mut response = match inner.call(req).await.map_err(|_| unreachable!()) {
                Ok(response) => response,
                Err(e) => return Err(e),
            };
            // Rate limit the download speed.
            for limit in speed_limits {
                if let Ok(key) = limit.extract_key(&req_clone) {
                    response = Self::throttle_download(response, &key, &limit.limiter);
                };
            }
            Ok(response)
        })
    }
}
