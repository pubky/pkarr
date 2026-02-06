# Pkarr Features Reference

Pkarr uses Cargo feature flags to customize functionality for different use cases.

## Default Features

With a basic dependency declaration, you get the `full-client` feature:

```toml
[dependencies]
pkarr = "5"
```

This includes both DHT and relay support, suitable for most server applications.

## Core Features

### `keys`

Ed25519 key utilities for identity management.

```toml
pkarr = { version = "5", default-features = false, features = ["keys"] }
```

Provides:
- `Keypair` - Generate and manage Ed25519 keypairs
- `PublicKey` - Handle public keys and derive z-base32 identifiers

### `signed_packet`

DNS packet signing and verification. Automatically enables `keys`.

```toml
pkarr = { version = "5", default-features = false, features = ["signed_packet"] }
```

Provides:
- `SignedPacket` - Create and verify signed DNS resource records

## Network Features

### `dht`

Direct Mainline DHT access for publishing and resolving records.

```toml
pkarr = { version = "5", default-features = false, features = ["dht"] }
```

**Not available in WASM environments.**

### `relays`

HTTP relay support for environments without direct DHT access.

```toml
pkarr = { version = "5", default-features = false, features = ["relays"] }
```

**Required for WASM/browser applications.**

### `full-client` (default)

Both DHT and relay support combined.

```toml
pkarr = "5"  # Equivalent to features = ["full-client"]
```

## Extra Features

### `endpoints`

RFC 9460 HTTPS/SVCB service discovery for resolving service endpoints.

This feature enables URLs like `https://<pkarr-key>` by implementing the full
[endpoints resolution algorithm](../design/endpoints.md). It recursively
interprets and resolves SVCB records - when a target points to another Pkarr
key, it fetches that packet and continues resolution. Records are sorted by
priority for failover and shuffled within priority levels for load balancing.

Returns an async `Stream` of endpoints. Downstream applications need `futures-lite`
(or equivalent) to consume the stream with `.next()` and `StreamExt`.

```toml
pkarr = { version = "5", features = ["endpoints"] }
```

### `lmdb-cache`

Persistent LMDB cache backend for the client.

```toml
pkarr = { version = "5", features = ["lmdb-cache"] }
```

### `tls`

TLS certificate support for secure connections. Enables `endpoints`.

```toml
pkarr = { version = "5", features = ["tls"] }
```

### `reqwest-resolve`

Implement `reqwest::dns::Resolve` trait for the Client. Enables `endpoints`.

```toml
pkarr = { version = "5", features = ["reqwest-resolve"] }
```

### `reqwest-builder`

Create a `reqwest::ClientBuilder` from the Pkarr Client. Enables `tls` and `reqwest-resolve`.

```toml
pkarr = { version = "5", features = ["reqwest-builder"] }
```

## Feature Combinations

### `extra`

All extra features: `endpoints`, `lmdb-cache`, `reqwest-resolve`, `tls`, `reqwest-builder`.

```toml
pkarr = { version = "5", features = ["extra"] }
```

### `full`

Everything: `full-client` + `extra`.

```toml
pkarr = { version = "5", features = ["full"] }
```

## Decision Guide

| Use Case | Recommended Features |
|----------|---------------------|
| Server application | `full-client` (default) |
| Browser/WASM | `relays` only |
| Key handling only | `keys` |
| Packet signing only | `signed_packet` |
| With persistent cache | `lmdb-cache` |
| Service discovery | `endpoints` |
| HTTP client integration | `reqwest-builder` |
| Everything | `full` |

## Platform Compatibility

| Feature | Native | WASM |
|---------|--------|------|
| `keys` | Yes | Yes |
| `signed_packet` | Yes | Yes |
| `dht` | Yes | No |
| `relays` | Yes | Yes |
| `lmdb-cache` | Yes | No |
| `endpoints` | Yes | Yes |
| `tls` | Yes | No |

## Minimal Configurations

**Smallest footprint (keys only):**
```toml
pkarr = { version = "5", default-features = false, features = ["keys"] }
```

**WASM browser client:**
```toml
pkarr = { version = "5", default-features = false, features = ["relays"] }
```

**Server with persistence:**
```toml
pkarr = { version = "5", features = ["lmdb-cache"] }
```
