# Pkarr Integration Guide

This guide covers production integration patterns for pkarr. For basic usage, see the [quickstart](./quickstart.md). For feature flag selection, see [features](./features.md).

## Feature Flag Decision Tree

Choose features based on your deployment environment and requirements.

### Environment-Based Selection

```
What is your target platform?
|
+-- Server/Native Application
|   |
|   +-- Need persistent cache? --> Add `lmdb-cache`
|   +-- Need HTTPS/SVCB endpoint discovery? --> Add `endpoints`
|   +-- Need reqwest DNS resolver integration? --> Add `reqwest-resolve`
|   +-- Need a preconfigured reqwest client with Pkarr DNS/TLS? --> Add `reqwest-builder`
|   +-- Otherwise --> Use default (`full-client`)
|
+-- Browser/WASM
|   |
|   +-- Use `relays` as the network feature (DHT is unavailable in browsers)
|
+-- Key management
|   |
|   +-- Just key generation? --> `default-features = false`
|   +-- Signing packets offline? --> `signed_packet`
|
+-- Relay-only deployment
    |
    +-- `relays` (no DHT, smaller binary)
```

### Minimal Configurations

| Use Case | Cargo.toml |
|----------|------------|
| Full native client | `pkarr = "7"` |
| Native client + persistence | `pkarr = { version = "7", features = ["lmdb-cache"] }` |
| Browser/WASM | `pkarr = { version = "7", default-features = false, features = ["relays"] }` |
| Key utilities only | `pkarr = { version = "7", default-features = false }` |
| Everything for native apps | `pkarr = { version = "7", features = ["full"] }` |

Endpoint-related extras require the client API. With default features this is
already enabled on native targets. If you disable default features, combine
`endpoints`, `tls`, `reqwest-resolve`, or `reqwest-builder` with `dht` and/or
`relays` as appropriate for your target.

## Client Abstractions

The client is organized in layers so applications choose behavior without
depending on a concrete network implementation:

```text
ClientBuilder
├── cache: InMemoryCache | custom Cache | disabled
└── backend: DHT | HTTP relays | combined DHT + relays
        ↓
      Client
      ├── publish(&SignedPacket)
      └── resolve(&PublicKey, ResolvePolicy)
```

`ClientBuilder` owns configuration: network selection, cache choice, TTL
bounds, request timeout, and backend-specific settings. `build()` validates
that at least one network is available and constructs a cloneable `Client`.
The concrete DHT, relay, and combined backend types are internal; use builder
methods and Cargo features to select them.

`Client` is the public I/O facade. It coordinates the optional local cache with
the configured backend and maps backend-specific outcomes into `BuildError`,
`PublishError`, and `ResolveError`. The `Cache` trait is the extension point for
custom storage, while `ResolvePolicy` controls how a particular lookup balances
cache latency against network freshness.

With the default native features, the combined backend has these semantics:

| Operation | Combined-backend behavior |
|-----------|---------------------------|
| `publish` | Publishes through DHT and relays concurrently. A successful result reports the maximum known stored-node count, not a sum. |
| `resolve(..., CacheOnly)` | Checks the local cache, then relay caches; it never queries DHT nodes. |
| `resolve(..., CacheFirst)` | Returns a fresh local hit immediately. Otherwise it races configured networks and can finish on the first fresh result that is not older than the cached packet. |
| `resolve(..., NetworkOnly)` | Bypasses local cache reads, waits for configured networks, and selects the most recent observed network state. |

Successful publishes and network resolutions update the local cache without
replacing a newer cached packet. If `CacheFirst` selects an expired packet as
its best backend result, that packet may still advance the cache before the
client returns `ResolveError::NotFound`. Other outcomes are preserved: for
example, a newer malformed mutable item returns
`ResolveError::InvalidSignedPacket`, while backend failures may return their
corresponding `ResolveError`.

## Client Configuration

Use `Client::builder()` to customize the client for your environment.

### Network Configuration

```rust
use pkarr::Client;

// Default: both DHT and relays enabled
let client = Client::builder().build()?;

// DHT only (no relay fallback)
let client = Client::builder()
    .no_relays()
    .build()?;

// Relays only (required for WASM, optional for firewall-restricted environments)
let client = Client::builder()
    .no_dht()
    .build()?;

// Custom relays
let client = Client::builder()
    .relays(&["https://pkarr.pubky.app", "https://my-relay.example.com"])?
    .build()?;

// Start without either default backend, then enable only your relays
let client = Client::builder()
    .no_default_network()
    .relays(&["https://my-relay.example.com"])?
    .build()?;

// Custom DHT bootstrap nodes
let client = Client::builder()
    .bootstrap(&["router.bittorrent.com:6881"])
    .build()?;

// Extend defaults with additional nodes
let client = Client::builder()
    .extra_bootstrap(&["my-bootstrap.example.com:6881"])
    .extra_relays(&["https://my-relay.example.com"])?
    .build()?;
```

Calling `build()` after `no_default_network()` without configuring at least one
backend returns `BuildError::NoNetwork`.

For a private DHT, use testnet diagnostic thresholds and customize the
underlying `mainline::Config` when needed:

```rust
use pkarr::{dht::ReportPolicy, Client};

let client = Client::builder()
    .no_default_network()
    .bootstrap(&["127.0.0.1:6881"])
    .dht_report_policy(ReportPolicy::testnet())
    .dht(|config| {
        config.port = Some(0);
        config
    })
    .build()?;
```

`ReportPolicy` controls warning thresholds for DHT diagnostics; it does not
change which packets are accepted.

### Cache Configuration

```rust
use pkarr::Client;
use std::sync::Arc;

// Adjust in-memory cache size (default: 1000 entries)
let client = Client::builder()
    .cache_size(5000)
    .build()?;

// Disable caching
let client = Client::builder()
    .cache_size(0)
    .build()?;

// Custom cache implementation
let custom_cache: Arc<dyn pkarr::Cache> = /* ... */;
let client = Client::builder()
    .cache(custom_cache)
    .build()?;
```

`cache_size(0)` disables all caching, including a custom cache supplied with
`cache()`. A custom `Cache` implementation must also override `capacity()` and
return a nonzero value; the trait's default capacity is zero and is treated as
disabled.

The `lmdb-cache` feature provides a persistent cache implementation, but the
client will not use it automatically. Enable the feature, create an
`LmdbCache`, and provide it to the builder:

```toml
[dependencies]
pkarr = { version = "7", features = ["lmdb-cache"] }
```

```rust
use pkarr::{extra::lmdb_cache::LmdbCache, Client};
use std::{path::Path, sync::Arc};

let cache = unsafe { LmdbCache::open(Path::new("./pkarr-cache"), 1000)? };
let client = Client::builder()
    .cache(Arc::new(cache))
    .build()?;
```

### TTL Configuration

TTL bounds control when cached packets are considered expired.

```rust
use pkarr::Client;

let client = Client::builder()
    // Minimum time before a packet is considered expired
    .minimum_ttl(60)      // At least 60 seconds
    // Maximum time before forcing a refresh
    .maximum_ttl(86400)   // At most 24 hours
    .build()?;
```

### Timeout Configuration

```rust
use pkarr::Client;
use std::time::Duration;

let client = Client::builder()
    .request_timeout(Duration::from_secs(5))
    .build()?;
```

The timeout applies to both DHT and relay requests. A custom `reqwest::Client`
can configure relay HTTP behavior such as proxies or default headers, while
Pkarr continues to apply `request_timeout` to each request:

```rust
use pkarr::Client;
use std::time::Duration;

let http_client = reqwest::Client::builder()
    .user_agent("my-pkarr-client/1.0")
    .build()?;

let client = Client::builder()
    .no_dht()
    .reqwest_client(http_client)
    .request_timeout(Duration::from_secs(5))
    .build()?;
```

Using this example requires a compatible direct dependency such as
`reqwest = { version = "0.13", default-features = false, features = ["rustls"] }`
in the application.

### Endpoints Configuration

When using the `endpoints` feature for service discovery:

```rust
use pkarr::Client;

let client = Client::builder()
    .max_recursion_depth(7)  // Limit CNAME/alias chain depth
    .build()?;
```

## Async Runtime Support

Pkarr provides an async API.

```rust
use pkarr::{Client, PublicKey, ResolvePolicy};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder().build()?;
    let public_key: PublicKey = "pk:...".try_into()?;

    let packet = client.resolve(&public_key, ResolvePolicy::CacheFirst).await;
    Ok(())
}
```

Pkarr uses `async_compat` internally. If no Tokio runtime is detected, it automatically wraps futures for compatibility. This means pkarr works with async-std, smol, or any other executor.

## WASM/Browser Integration

### Setup

```toml
[dependencies]
pkarr = { version = "7", default-features = false, features = ["relays"] }
```

### Constraints

- **DHT unavailable**: Browsers cannot open UDP sockets. Only HTTP relays work.
- **Uses Fetch API**: Relay requests use the browser's native fetch.
- **Not WASI compatible**: Only `wasm32-unknown-unknown` target is supported.

## Republishing Patterns

DHT records are ephemeral. Nodes drop records after a few hours. For persistent availability, implement periodic republishing.

### Basic Republishing Loop

```rust
use pkarr::{errors::PublishError, Client, SignedPacket};
use std::time::Duration;

async fn republish_loop(
    client: Client,
    build_packet: impl Fn() -> SignedPacket,
) {
    let interval = Duration::from_secs(3600); // Republish hourly

    loop {
        let packet = build_packet();

        match client.publish(&packet).await {
            Ok(stored_on) => {
                println!("Republished successfully; stored on at least {stored_on} DHT nodes")
            }
            Err(PublishError::NotMostRecent) => {
                eprintln!("A newer packet exists; resolve NetworkOnly before retrying")
            }
            Err(error) => eprintln!("Republish failed: {error}"),
        }

        tokio::time::sleep(interval).await;
    }
}
```

### Updating a Published Packet

The client does not serialize concurrent publishes for the same public key. If
your application can build multiple different packets for one key at the same
time, serialize the complete resolve, build, and publish sequence, for example
with a single-permit semaphore per public key.

```rust
use pkarr::{
    errors::ResolveError, Client, Keypair, ResolvePolicy, SignedPacket, Timestamp,
};
use std::io;

async fn republish_update(
    client: &Client,
    keypair: &Keypair,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get the highest observed sequence before building the replacement.
    let observed_sequence = match client
        .resolve(&keypair.public_key(), ResolvePolicy::NetworkOnly)
        .await
    {
        Ok(existing) => existing.timestamp().as_u64(),
        Err(ResolveError::NotFound) => 0,
        Err(ResolveError::InvalidSignedPacket { seq }) => u64::try_from(seq)?,
        Err(error) => return Err(error.into()),
    };

    let next_sequence = observed_sequence
        .checked_add(1)
        .filter(|sequence| *sequence <= i64::MAX as u64)
        .ok_or_else(|| io::Error::other("Pkarr sequence is exhausted"))?;
    let timestamp = Timestamp::from(Timestamp::now().as_u64().max(next_sequence));

    let packet = SignedPacket::builder()
        // Build the complete desired record set; omitted records are removed.
        .txt("key".try_into()?, "updated_value".try_into()?, 300)
        .timestamp(timestamp)
        .sign(keypair)?;

    let stored_on = client.publish(&packet).await?;
    println!("Published successfully; stored on at least {stored_on} DHT nodes");
    Ok(())
}
```

## Resolve Policies

Use `ResolvePolicy::CacheFirst` for normal application lookups. It returns
fresh cached packets, or queries the network on a cache miss or expired cache
entry. It does not fall back to expired local cache entries.

Use `ResolvePolicy::CacheOnly` when a locally cached or relay-cached packet is
acceptable even if it is expired. It may make an HTTP relay request after a
local cache miss, but it never queries DHT nodes.

Use `ResolvePolicy::NetworkOnly` when you need the most recent network state,
for example before rebuilding and publishing an updated packet. Handle
`ResolveError::InvalidSignedPacket` separately from `ResolveError::NotFound`:
the former reports a newer mutable-item sequence that did not contain a valid
PKARR packet and must not be interpreted as an unused key.

## Next Steps

- [API Documentation](https://docs.rs/pkarr/latest/pkarr/) - Complete API reference
- [Examples](https://github.com/Pubky/pkarr/tree/main/pkarr/examples) - Working code samples
- [Pkarr Relay](https://github.com/Pubky/pkarr/tree/main/relay) - Run your own relay
