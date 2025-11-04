use pkarr_relay::{
    rate_limiter::{Operation, OperationLimit, QuotaValue},
    rate_limiting::RateLimiterConfig,
};
use std::num::NonZero;
use std::str::FromStr;

#[test]
fn test_default_config() {
    let config = pkarr_relay::Config::default();

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
    let mut config = pkarr_relay::Config::default();
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
    let mut config = pkarr_relay::Config::default();
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
    let mut config = pkarr_relay::Config::default();
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
    let mut config = pkarr_relay::Config::default();
    config.dht_rate_limiter = None;
    config.http_rate_limiter = None;

    assert!(config.dht_rate_limiter.is_none());
    assert!(config.http_rate_limiter.is_none());
}

#[test]
fn test_behind_proxy_false() {
    let config = pkarr_relay::Config::default();
    assert_eq!(config.behind_proxy, false);
}

#[test]
fn test_behind_proxy_true() {
    let mut config = pkarr_relay::Config::default();
    config.behind_proxy = true;
    assert_eq!(config.behind_proxy, true);
}

#[test]
fn test_dht_config_overridden_by_global_behind_proxy() {
    let mut config = pkarr_relay::Config::default();
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

    let mut config = pkarr_relay::Config::default();
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

    let mut config = pkarr_relay::Config::default();
    config.dht_rate_limiter = Some(legacy_config.clone());
    config.behind_proxy = legacy_config.behind_proxy;

    assert_eq!(config.behind_proxy, true);
    assert!(config.dht_rate_limiter.is_some());
}

#[test]
fn test_dht_rate_limit_overrides_legacy() {
    let mut config = pkarr_relay::Config::default();

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
    let mut config = pkarr_relay::Config::default();

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
    let mut config = pkarr_relay::Config::default();

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
    let mut config = pkarr_relay::Config::default();

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
    let mut config = pkarr_relay::Config::default();
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
    let mut config = pkarr_relay::Config::default();
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
