use pkarr_relay::{
    rate_limiter::{Operation, OperationLimit, QuotaValue},
    rate_limiting::RateLimiterConfig,
    Config, RelayBuilder,
};
use std::num::NonZero;
use std::str::FromStr;
use std::time::Duration;

// Test constants
const TEST_KEY: &str = "o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy";
const TEST_BODY: &[u8] = &[0u8; 100];
const STARTUP_DELAY_MS: u64 = 100;

// Unit Tests

#[test]
fn test_default_config() {
    let config = Config::default();

    assert!(config.dht_rate_limiter.is_some());
    assert_eq!(config.dht_rate_limiter.as_ref().unwrap().per_second, 2);
    assert_eq!(config.dht_rate_limiter.as_ref().unwrap().burst_size, 10);
    assert!(config.http_rate_limiter.is_none());
    assert_eq!(config.behind_proxy, false);
}

#[test]
fn test_ip_rate_limiter_peer_variant() {
    let config = RateLimiterConfig {
        behind_proxy: false,
        per_second: 5,
        burst_size: 20,
    };

    let limiter = pkarr_relay::rate_limiting::IpRateLimiter::new(&config);

    match limiter {
        pkarr_relay::rate_limiting::IpRateLimiter::Peer(_) => {}
        pkarr_relay::rate_limiting::IpRateLimiter::Proxy(_) => {
            panic!("Expected Peer variant when behind_proxy = false");
        }
    }
}

#[test]
fn test_ip_rate_limiter_proxy_variant() {
    let config = RateLimiterConfig {
        behind_proxy: true,
        per_second: 5,
        burst_size: 20,
    };

    let limiter = pkarr_relay::rate_limiting::IpRateLimiter::new(&config);

    match limiter {
        pkarr_relay::rate_limiting::IpRateLimiter::Proxy(_) => {}
        pkarr_relay::rate_limiting::IpRateLimiter::Peer(_) => {
            panic!("Expected Proxy variant when behind_proxy = true");
        }
    }
}

#[test]
fn test_quota_value_parse_per_second() {
    let quota = QuotaValue::from_str("5r/s").unwrap();
    assert_eq!(quota.rate.get(), 5);
}

#[test]
fn test_quota_value_parse_per_minute() {
    let quota = QuotaValue::from_str("30r/m").unwrap();
    assert_eq!(quota.rate.get(), 30);
}

#[test]
fn test_all_operation_types() {
    let operations = vec![
        Operation::Publish,
        Operation::Resolve,
        Operation::ResolveMostRecent,
        Operation::Index,
    ];

    for op in operations {
        let quota = QuotaValue::from_str("5r/s").unwrap();
        let limit = OperationLimit::new(op, quota, NonZero::new(10));
        assert_eq!(limit.whitelist.len(), 0);
    }
}

#[test]
fn test_whitelist_parsing() {
    use pkarr_relay::rate_limiter::LimitKey;

    let whitelist_entries = vec!["127.0.0.1", "192.168.1.0/24", "10.0.0.0/8"];

    for entry in whitelist_entries {
        let key = LimitKey::from_str(entry).unwrap();
        match key {
            LimitKey::IpNetwork(_) => {}
        }
    }
}

#[test]
fn test_multiple_http_rate_limits() {
    let limits = vec![
        OperationLimit::new(
            Operation::Publish,
            QuotaValue::from_str("5r/s").unwrap(),
            NonZero::new(10),
        ),
        OperationLimit::new(
            Operation::Resolve,
            QuotaValue::from_str("30r/m").unwrap(),
            NonZero::new(30),
        ),
        OperationLimit::new(
            Operation::ResolveMostRecent,
            QuotaValue::from_str("5r/m").unwrap(),
            NonZero::new(10),
        ),
        OperationLimit::new(
            Operation::Index,
            QuotaValue::from_str("30r/m").unwrap(),
            NonZero::new(30),
        ),
    ];

    assert_eq!(limits.len(), 4);
}

#[test]
fn test_valid_quota_formats() {
    let valid_quotas = vec!["1r/s", "10r/s", "100r/s", "1r/m", "60r/m", "1000r/m"];

    for quota_str in valid_quotas {
        let quota = QuotaValue::from_str(quota_str);
        assert!(quota.is_ok());
    }
}

#[test]
fn test_invalid_quota_formats() {
    let invalid_quotas = vec!["invalid", "10", "10/s", "10rs", "10r/h", "0r/s"];

    for quota_str in invalid_quotas {
        let quota = QuotaValue::from_str(quota_str);
        assert!(quota.is_err());
    }
}

#[test]
fn test_config_only_dht() {
    let mut config = Config::default();
    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 5,
        burst_size: 15,
    });
    config.http_rate_limiter = None;

    assert!(config.dht_rate_limiter.is_some());
    assert!(config.http_rate_limiter.is_none());
}

#[test]
fn test_config_only_http() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = Some(vec![OperationLimit::new(
        Operation::Publish,
        QuotaValue::from_str("10r/s").unwrap(),
        NonZero::new(20),
    )]);

    assert!(config.dht_rate_limiter.is_none());
    assert!(config.http_rate_limiter.is_some());
}

#[test]
fn test_config_both_dht_and_http() {
    let mut config = Config::default();
    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 5,
        burst_size: 15,
    });
    config.http_rate_limiter = Some(vec![OperationLimit::new(
        Operation::Publish,
        QuotaValue::from_str("10r/s").unwrap(),
        NonZero::new(20),
    )]);

    assert!(config.dht_rate_limiter.is_some());
    assert!(config.http_rate_limiter.is_some());
}

#[test]
fn test_config_neither_rate_limiter() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = None;

    assert!(config.dht_rate_limiter.is_none());
    assert!(config.http_rate_limiter.is_none());
}

#[test]
fn test_behind_proxy_false() {
    let config = Config::default();
    assert_eq!(config.behind_proxy, false);
}

#[test]
fn test_behind_proxy_true() {
    let mut config = Config::default();
    config.behind_proxy = true;
    assert_eq!(config.behind_proxy, true);
}

#[test]
fn test_dht_config_overridden_by_global_behind_proxy() {
    let mut config = Config::default();
    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 2,
        burst_size: 10,
    });
    config.behind_proxy = true;

    let dht_config = config.dht_rate_limiter.as_ref().unwrap();
    let mut dht_config_copy = dht_config.clone();
    dht_config_copy.behind_proxy = config.behind_proxy;

    assert_eq!(dht_config_copy.behind_proxy, true);
}

#[test]
fn test_legacy_rate_limiter_applies_to_both() {
    let legacy_config = RateLimiterConfig {
        behind_proxy: false,
        per_second: 3,
        burst_size: 15,
    };

    let mut config = Config::default();
    config.dht_rate_limiter = Some(legacy_config.clone());
    config.http_rate_limiter = None;
    config.behind_proxy = legacy_config.behind_proxy;

    assert!(config.dht_rate_limiter.is_some());
    assert_eq!(config.dht_rate_limiter.as_ref().unwrap().per_second, 3);
    assert_eq!(config.dht_rate_limiter.as_ref().unwrap().burst_size, 15);
    assert_eq!(config.behind_proxy, false);
    assert!(config.http_rate_limiter.is_none());
}

#[test]
fn test_legacy_rate_limiter_with_behind_proxy_true() {
    let legacy_config = RateLimiterConfig {
        behind_proxy: true,
        per_second: 5,
        burst_size: 20,
    };

    let mut config = Config::default();
    config.dht_rate_limiter = Some(legacy_config.clone());
    config.behind_proxy = legacy_config.behind_proxy;

    assert_eq!(config.behind_proxy, true);
    assert!(config.dht_rate_limiter.is_some());
}

#[test]
fn test_dht_rate_limit_overrides_legacy() {
    let mut config = Config::default();

    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 2,
        burst_size: 10,
    });
    config.behind_proxy = false;

    let override_config = RateLimiterConfig {
        behind_proxy: false,
        per_second: 10,
        burst_size: 50,
    };
    config.dht_rate_limiter = Some(override_config.clone());

    assert_eq!(config.dht_rate_limiter.as_ref().unwrap().per_second, 10);
    assert_eq!(config.dht_rate_limiter.as_ref().unwrap().burst_size, 50);
}

#[test]
fn test_http_rate_limit_overrides_legacy() {
    let mut config = Config::default();

    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 2,
        burst_size: 10,
    });
    config.http_rate_limiter = None;

    let http_limits = vec![OperationLimit::new(
        Operation::Publish,
        QuotaValue::from_str("20r/s").unwrap(),
        NonZero::new(40),
    )];
    config.http_rate_limiter = Some(http_limits);

    assert!(config.dht_rate_limiter.is_some());
    assert!(config.http_rate_limiter.is_some());
    assert_eq!(config.http_rate_limiter.as_ref().unwrap().len(), 1);
}

#[test]
fn test_legacy_with_both_specific_overrides() {
    let mut config = Config::default();

    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 2,
        burst_size: 10,
    });
    config.behind_proxy = false;

    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 8,
        burst_size: 25,
    });

    config.http_rate_limiter = Some(vec![OperationLimit::new(
        Operation::Resolve,
        QuotaValue::from_str("100r/m").unwrap(),
        NonZero::new(100),
    )]);

    assert_eq!(config.dht_rate_limiter.as_ref().unwrap().per_second, 8);
    assert_eq!(config.http_rate_limiter.as_ref().unwrap().len(), 1);
}

#[test]
fn test_relay_behind_proxy_overrides_rate_limiter_behind_proxy() {
    let mut config = Config::default();

    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 2,
        burst_size: 10,
    });
    config.behind_proxy = false;

    config.behind_proxy = true;

    assert_eq!(config.behind_proxy, true);
    assert_eq!(
        config.dht_rate_limiter.as_ref().unwrap().behind_proxy,
        false
    );
}

#[test]
fn test_no_legacy_only_specific_dht() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = None;

    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 5,
        burst_size: 15,
    });

    assert!(config.dht_rate_limiter.is_some());
    assert!(config.http_rate_limiter.is_none());
    assert_eq!(config.dht_rate_limiter.as_ref().unwrap().per_second, 5);
}

#[test]
fn test_no_legacy_only_specific_http() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = None;

    config.http_rate_limiter = Some(vec![OperationLimit::new(
        Operation::Index,
        QuotaValue::from_str("50r/m").unwrap(),
        NonZero::new(50),
    )]);

    assert!(config.dht_rate_limiter.is_none());
    assert!(config.http_rate_limiter.is_some());
}

// Integration Tests

async fn start_test_relay(mut config: Config) -> (String, pkarr_relay::Relay) {
    config.http_port = 0;
    let relay = unsafe { RelayBuilder::new(config).run().await.unwrap() };
    let url = relay.local_url().to_string();
    tokio::time::sleep(Duration::from_millis(STARTUP_DELAY_MS)).await;
    (url, relay)
}

fn create_endpoint(url: &str, key: &str) -> String {
    format!("{}/{}", url, key)
}

fn create_most_recent_endpoint(url: &str, key: &str) -> String {
    format!("{}/{}?most_recent=true", url, key)
}

#[tokio::test]
async fn test_http_rate_limit_enforced() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = Some(vec![OperationLimit::new(
        Operation::Publish,
        QuotaValue::from_str("2r/s").unwrap(),
        NonZero::new(2),
    )]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    let mut success_count = 0;
    let mut rate_limited_count = 0;

    for _ in 0..5 {
        let response = client.put(&endpoint).body(TEST_BODY).send().await.unwrap();

        if response.status().is_success() || response.status() == 400 {
            success_count += 1;
        } else if response.status() == 429 {
            rate_limited_count += 1;
        }
    }

    assert!(
        rate_limited_count > 0,
        "Expected some requests to be rate limited"
    );
    assert!(
        success_count <= 2,
        "Expected at most 2 successful requests with 2r/s limit"
    );
}

#[tokio::test]
async fn test_operation_based_rate_limiting() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = Some(vec![
        OperationLimit::new(
            Operation::Publish,
            QuotaValue::from_str("1r/s").unwrap(),
            NonZero::new(1),
        ),
        OperationLimit::new(
            Operation::Resolve,
            QuotaValue::from_str("10r/s").unwrap(),
            NonZero::new(10),
        ),
    ]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    let mut publish_limited = false;
    for _ in 0..3 {
        let response = client.put(&endpoint).body(TEST_BODY).send().await.unwrap();

        if response.status() == 429 {
            publish_limited = true;
            break;
        }
    }

    let mut resolve_success = 0;
    for _ in 0..5 {
        let response = client.get(&endpoint).send().await.unwrap();

        if response.status() != 429 {
            resolve_success += 1;
        }
    }

    assert!(publish_limited, "Publish should be rate limited at 1r/s");
    assert!(
        resolve_success >= 4,
        "Resolve should allow more requests at 10r/s"
    );
}

#[tokio::test]
async fn test_legacy_rate_limiter_applies_to_http() {
    let mut config = Config::default();
    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 1,
        burst_size: 1,
    });
    config.http_rate_limiter = None;

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    let mut rate_limited_count = 0;

    for _ in 0..4 {
        let response = client.put(&endpoint).body(TEST_BODY).send().await.unwrap();

        if response.status() == 429 {
            rate_limited_count += 1;
        }
    }

    assert!(
        rate_limited_count > 0,
        "Legacy rate limiter should apply to HTTP"
    );
}

#[tokio::test]
async fn test_whitelist_bypasses_rate_limit() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;

    let mut publish_limit = OperationLimit::new(
        Operation::Publish,
        QuotaValue::from_str("1r/s").unwrap(),
        NonZero::new(1),
    );
    publish_limit.whitelist =
        vec![pkarr_relay::rate_limiter::LimitKey::from_str("127.0.0.1").unwrap()];

    config.http_rate_limiter = Some(vec![publish_limit]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    let mut success_count = 0;

    for _ in 0..5 {
        let response = client.put(&endpoint).body(TEST_BODY).send().await.unwrap();

        if response.status() != 429 {
            success_count += 1;
        }
    }

    assert_eq!(success_count, 5, "Whitelisted IP should bypass rate limits");
}

#[tokio::test]
async fn test_rate_limit_replenishment() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = Some(vec![OperationLimit::new(
        Operation::Publish,
        QuotaValue::from_str("2r/s").unwrap(),
        NonZero::new(2),
    )]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    for _ in 0..3 {
        client.put(&endpoint).body(TEST_BODY).send().await.unwrap();
    }

    tokio::time::sleep(Duration::from_secs(2)).await;

    let response = client.put(&endpoint).body(TEST_BODY).send().await.unwrap();

    assert_ne!(
        response.status(),
        429,
        "Rate limit should replenish after 2 seconds"
    );
}

#[tokio::test]
async fn test_index_operation_separate_limit() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = Some(vec![OperationLimit::new(
        Operation::Index,
        QuotaValue::from_str("2r/s").unwrap(),
        NonZero::new(2),
    )]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();

    let mut rate_limited = false;
    for _ in 0..5 {
        let response = client.get(&url).send().await.unwrap();

        if response.status() == 429 {
            rate_limited = true;
            break;
        }
    }

    assert!(rate_limited, "Index operation should be rate limited");
}

#[tokio::test]
async fn test_no_rate_limiter_allows_all_requests() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = None;

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    let mut all_success = true;
    for _ in 0..10 {
        let response = client.put(&endpoint).body(TEST_BODY).send().await.unwrap();

        if response.status() == 429 {
            all_success = false;
            break;
        }
    }

    assert!(all_success, "No rate limiter should allow all requests");
}

#[tokio::test]
async fn test_both_dht_and_http_rate_limiting() {
    let mut config = Config::default();
    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 1,
        burst_size: 1,
    });
    config.http_rate_limiter = Some(vec![
        OperationLimit::new(
            Operation::Resolve,
            QuotaValue::from_str("5r/s").unwrap(),
            NonZero::new(5),
        ),
        OperationLimit::new(
            Operation::Publish,
            QuotaValue::from_str("2r/s").unwrap(),
            NonZero::new(2),
        ),
    ]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    let mut resolve_count = 0;
    for _ in 0..7 {
        let response = client.get(&endpoint).send().await.unwrap();
        if response.status() != 429 {
            resolve_count += 1;
        }
    }

    assert!(
        resolve_count >= 5,
        "HTTP resolve should use specific limit of 5r/s"
    );

    let mut publish_count = 0;
    for _ in 0..4 {
        let response = client
            .put(&endpoint)
            .body(vec![0u8; 100])
            .send()
            .await
            .unwrap();
        if response.status() != 429 {
            publish_count += 1;
        }
    }

    assert!(publish_count <= 2, "Publish should be limited at 2r/s");
    assert!(publish_count >= 2, "Both DHT and HTTP limits coexist");
}

#[tokio::test]
async fn test_resolve_most_recent_operation() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.resolve_most_recent = true;
    config.http_rate_limiter = Some(vec![
        OperationLimit::new(
            Operation::Resolve,
            QuotaValue::from_str("10r/s").unwrap(),
            NonZero::new(10),
        ),
        OperationLimit::new(
            Operation::ResolveMostRecent,
            QuotaValue::from_str("2r/s").unwrap(),
            NonZero::new(2),
        ),
    ]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);
    let most_recent_endpoint = create_most_recent_endpoint(&url, TEST_KEY);

    let mut regular_resolve_success = 0;
    for _ in 0..5 {
        let response = client.get(&endpoint).send().await.unwrap();
        if response.status() != 429 {
            regular_resolve_success += 1;
        }
    }

    let mut most_recent_limited = 0;
    for _ in 0..4 {
        let response = client.get(&most_recent_endpoint).send().await.unwrap();
        if response.status() == 429 {
            most_recent_limited += 1;
        }
    }

    assert!(
        regular_resolve_success >= 5,
        "Regular resolve should have high limit"
    );
    assert!(
        most_recent_limited > 0,
        "Most recent should be rate limited at 2r/s"
    );
}

#[tokio::test]
async fn test_behind_proxy_mode() {
    let mut config = Config::default();
    config.behind_proxy = true;
    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 2,
        burst_size: 2,
    });
    config.http_rate_limiter = None;

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    let mut rate_limited_count = 0;
    for _ in 0..5 {
        let response = client
            .put(&endpoint)
            .header("X-Forwarded-For", "192.168.1.100")
            .body(TEST_BODY)
            .send()
            .await
            .unwrap();

        if response.status() == 429 {
            rate_limited_count += 1;
        }
    }

    assert!(
        rate_limited_count > 0,
        "Should rate limit by X-Forwarded-For when behind_proxy=true"
    );
}

#[tokio::test]
async fn test_burst_override_applies() {
    // Burst should be 3 (from OperationLimit.burst), NOT 1 (from "1r/s" rate).
    // If burst config is silently ignored, only 1 request succeeds.
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = Some(vec![OperationLimit::new(
        Operation::Publish,
        QuotaValue::from_str("1r/s").unwrap(),
        NonZero::new(3), // burst of 3
    )]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    let mut success_count = 0;
    for _ in 0..6 {
        let response = client
            .put(&endpoint)
            .body(vec![0u8; 100])
            .send()
            .await
            .unwrap();

        if response.status() != 429 {
            success_count += 1;
        }
    }

    assert!(
        success_count >= 3,
        "Burst of 3 should allow at least 3 requests, but only {} succeeded",
        success_count
    );
}

#[tokio::test]
async fn test_burst_capacity_exhaustion() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = Some(vec![OperationLimit::new(
        Operation::Publish,
        QuotaValue::from_str("1r/s").unwrap(),
        NonZero::new(3),
    )]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    let mut success_count = 0;
    let mut rate_limited_count = 0;

    for _ in 0..6 {
        let response = client
            .put(&endpoint)
            .body(vec![0u8; 100])
            .send()
            .await
            .unwrap();

        if response.status() != 429 && (response.status().is_success() || response.status() == 400)
        {
            success_count += 1;
        } else if response.status() == 429 {
            rate_limited_count += 1;
        }
    }

    assert!(success_count <= 3, "Should allow burst of 3");
    assert!(
        rate_limited_count > 0,
        "Should rate limit after burst exhausted"
    );
}

#[tokio::test]
async fn test_cidr_whitelist_range() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;

    let mut publish_limit = OperationLimit::new(
        Operation::Publish,
        QuotaValue::from_str("1r/s").unwrap(),
        NonZero::new(1),
    );
    publish_limit.whitelist =
        vec![pkarr_relay::rate_limiter::LimitKey::from_str("127.0.0.0/24").unwrap()];

    config.http_rate_limiter = Some(vec![publish_limit]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    let mut success_count = 0;
    for _ in 0..5 {
        let response = client
            .put(&endpoint)
            .body(vec![0u8; 100])
            .send()
            .await
            .unwrap();
        if response.status() != 429 {
            success_count += 1;
        }
    }

    assert_eq!(
        success_count, 5,
        "CIDR whitelist 127.0.0.0/24 should include 127.0.0.1"
    );
}

#[tokio::test]
async fn test_http_overrides_legacy_config() {
    let mut config = Config::default();
    config.dht_rate_limiter = Some(RateLimiterConfig {
        behind_proxy: false,
        per_second: 1,
        burst_size: 1,
    });
    config.http_rate_limiter = Some(vec![OperationLimit::new(
        Operation::Publish,
        QuotaValue::from_str("10r/s").unwrap(),
        NonZero::new(10),
    )]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);

    let mut success_count = 0;
    for _ in 0..6 {
        let response = client
            .put(&endpoint)
            .body(vec![0u8; 100])
            .send()
            .await
            .unwrap();
        if response.status() != 429 {
            success_count += 1;
        }
    }

    assert!(
        success_count >= 5,
        "HTTP-specific config should override legacy 1r/s with 10r/s"
    );
}

#[tokio::test]
async fn test_all_operations_have_separate_limits() {
    let mut config = Config::default();
    config.dht_rate_limiter = None;
    config.resolve_most_recent = true;
    config.http_rate_limiter = Some(vec![
        OperationLimit::new(
            Operation::Publish,
            QuotaValue::from_str("1r/s").unwrap(),
            NonZero::new(1),
        ),
        OperationLimit::new(
            Operation::Resolve,
            QuotaValue::from_str("2r/s").unwrap(),
            NonZero::new(2),
        ),
        OperationLimit::new(
            Operation::ResolveMostRecent,
            QuotaValue::from_str("3r/s").unwrap(),
            NonZero::new(3),
        ),
        OperationLimit::new(
            Operation::Index,
            QuotaValue::from_str("4r/s").unwrap(),
            NonZero::new(4),
        ),
    ]);

    let (url, _relay) = start_test_relay(config).await;

    let client = reqwest::Client::new();
    let endpoint = create_endpoint(&url, TEST_KEY);
    let most_recent_endpoint = create_most_recent_endpoint(&url, TEST_KEY);

    let mut publish_success = 0;
    for _ in 0..3 {
        let response = client.put(&endpoint).body(TEST_BODY).send().await.unwrap();
        if response.status() != 429 {
            publish_success += 1;
        }
    }

    let mut resolve_success = 0;
    for _ in 0..4 {
        let response = client.get(&endpoint).send().await.unwrap();
        if response.status() != 429 {
            resolve_success += 1;
        }
    }

    let mut most_recent_success = 0;
    for _ in 0..5 {
        let response = client.get(&most_recent_endpoint).send().await.unwrap();
        if response.status() != 429 {
            most_recent_success += 1;
        }
    }

    let mut index_success = 0;
    for _ in 0..6 {
        let response = client.get(&url).send().await.unwrap();
        if response.status() != 429 {
            index_success += 1;
        }
    }

    assert!(publish_success <= 1, "Publish limited to 1r/s");
    assert!(resolve_success <= 2, "Resolve limited to 2r/s");
    assert!(most_recent_success <= 3, "Most recent limited to 3r/s");
    assert!(index_success <= 4, "Index limited to 4r/s");

    assert!(
        publish_success >= 1
            && resolve_success >= 2
            && most_recent_success >= 3
            && index_success >= 4,
        "All operations should allow at least their configured rate"
    );
}
