# Pkarr

Pkarr turns Ed25519 public keys into sovereign domain names. This crate creates
and verifies signed DNS packets, then publishes and resolves them through the
[Mainline DHT](https://github.com/Pubky/mainline), HTTP relays, or both.

## Installation

```bash
cargo add pkarr
```

The example below uses Tokio as its async runtime:

```bash
cargo add tokio --features macros,rt-multi-thread
```

## Quick Start

```rust,no_run
use pkarr::{Client, Keypair, ResolvePolicy, SignedPacket};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let keypair = Keypair::random();
    println!("Public key: {}", keypair.public_key());

    let packet = SignedPacket::builder()
        .txt("_hello".try_into()?, "world".try_into()?, 300)
        .sign(&keypair)?;

    let client = Client::builder().build()?;
    let stored_on = client.publish(&packet).await?;
    println!("Stored on at least {stored_on} DHT nodes");

    let resolved = client
        .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
        .await?;
    println!("Resolved:\n{resolved}");

    Ok(())
}
```

The client API is asynchronous. Tokio is used here to run the example; on
native targets, Pkarr also supports other executors through `async_compat`.

## Choosing Features

| Use case | Dependency |
|----------|------------|
| Native application using DHT and relays | `pkarr = "7"` |
| DHT only | `pkarr = { version = "7", default-features = false, features = ["dht"] }` |
| Relay only or browser/WASM | `pkarr = { version = "7", default-features = false, features = ["relays"] }` |
| Sign and verify packets without networking | `pkarr = { version = "7", default-features = false, features = ["signed_packet"] }` |
| Key generation and parsing only | `pkarr = { version = "7", default-features = false }` |

The default `full-client` feature enables both DHT and relay support. Browsers
cannot access the UDP DHT directly and must use `relays`; WASI is not supported.
Optional features also provide persistent LMDB caching, endpoint discovery,
and reqwest integration. See the
[feature reference](https://github.com/pubky/pkarr/blob/main/docs/features.md)
for the complete list.

## Next Steps

- [Quickstart](https://github.com/pubky/pkarr/blob/main/docs/quickstart.md)
- [Integration guide](https://github.com/pubky/pkarr/blob/main/docs/integration.md)
- [Examples](https://github.com/pubky/pkarr/tree/main/pkarr/examples)
- [API documentation](https://docs.rs/pkarr/latest/pkarr/)
