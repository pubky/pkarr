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
|   +-- Need HTTPS endpoints?  --> Add `endpoints`
|   +-- Need HTTP client integration? --> Add `reqwest-builder`
|   +-- Otherwise --> Use default (`full-client`)
|
+-- Browser/WASM
|   |
|   +-- `relays` only (DHT unavailable in browsers)
|
+-- Key management
|   |
|   +-- Just key generation? --> `keys` only
|   +-- Signing packets offline? --> `signed_packet`
|
+-- Relay-only deployment
    |
    +-- `relays` (no DHT, smaller binary)
```

### Minimal Configurations

| Use Case | Cargo.toml |
|----------|------------|
| Full server | `pkarr = "5"` |
| Server + persistence | `pkarr = { version = "5", features = ["lmdb-cache"] }` |
| Browser/WASM | `pkarr = { version = "5", default-features = false, features = ["relays"] }` |
| Key utilities only | `pkarr = { version = "5", default-features = false, features = ["keys"] }` |
| Everything | `pkarr = { version = "5", features = ["full"] }` |

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
pkarr = { version = "5", default-features = false, features = ["relays"] }
```

### Constraints

- **DHT unavailable**: Browsers cannot open UDP sockets. Only HTTP relays work.
- **Uses Fetch API**: Relay requests use the browser's native fetch.
- **Not WASI compatible**: Only `wasm32-unknown-unknown` target is supported.

## Republishing Patterns

DHT records are ephemeral. Nodes drop records after a few hours. For persistent availability, implement periodic republishing.

### Basic Republishing Loop

```rust
use pkarr::{Client, SignedPacket, Keypair};
use std::time::Duration;

async fn republish_loop(
    client: Client,
    keypair: Keypair,
    build_packet: impl Fn() -> SignedPacket,
) {
    let interval = Duration::from_secs(3600); // Republish hourly

    loop {
        let packet = build_packet();

        match client.publish(&packet, None).await {
            Ok(stored_on) => {
                println!("Republished successfully; stored on at least {stored_on} DHT nodes")
            }
            Err(e) => eprintln!("Republish failed: {e}"),
        }

        tokio::time::sleep(interval).await;
    }
}
```

### Safe Republishing with CAS

When multiple processes might update the same key, use CAS (compare-and-swap) to prevent lost updates.
The client does not serialize concurrent publishes for the same public key; if your application can build multiple different packets for one key at the same time, serialize those publishes in your own code before calling `publish`.

```rust
use pkarr::{Client, Keypair, ResolvePolicy, SignedPacket};
use ntimestamp::Timestamp;

async fn safe_republish(
    client: &Client,
    keypair: &Keypair,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get the most recent version
    let (current_packet, cas): (SignedPacket, Option<Timestamp>) =
        match client
            .resolve(&keypair.public_key(), ResolvePolicy::DhtNetworkOnly)
            .await
        {
            Ok(existing) => {
                // Rebuild packet with same or updated records
                let new_packet = SignedPacket::builder()
                    // Copy existing records you want to keep
                    .txt("key".try_into()?, "updated_value".try_into()?, 300)
                    .sign(keypair)?;
                (new_packet, Some(existing.timestamp()))
            }
            Err(pkarr::errors::ResolveError::NotFound) => {
                let new_packet = SignedPacket::builder()
                    .txt("key".try_into()?, "value".try_into()?, 300)
                    .sign(keypair)?;
                (new_packet, None)
            }
            Err(error) => return Err(error.into()),
        };

    // Publish with CAS - fails if someone else published in between
    let stored_on = client.publish(&current_packet, cas).await?;
    println!("Published successfully; stored on at least {stored_on} DHT nodes");
    Ok(())
}
```

## Resolve Policies

Use [`ResolvePolicy::CacheFirst`] for normal application lookups. It returns fresh cached packets, or queries the network on a cache miss or expired cache entry. It does not fall back to expired local cache entries.

Use [`ResolvePolicy::LocalOrRelayCacheOnly`] when a cached packet is acceptable even if it is expired.

Use [`ResolvePolicy::DhtNetworkOnly`] when you need the most recent packet observed from the network, for example before publishing with CAS.

## Next Steps

- [API Documentation](https://docs.rs/pkarr/latest/pkarr/) - Complete API reference
- [Examples](https://github.com/Pubky/pkarr/tree/main/pkarr/examples) - Working code samples
- [Pkarr Relay](https://github.com/Pubky/pkarr/tree/main/relay) - Run your own relay
