# Pkarr Quickstart

Get up and running with Pkarr in 5 minutes.


## Add to Project

```toml
[dependencies]
pkarr = "5"
tokio = { version = "1", features = ["full"] }
```

## Generate a Keypair

```rust
use pkarr::Keypair;

let keypair = Keypair::random();

// Get the public key as a z-base32 string (this is your "domain name")
let public_key_string = keypair.public_key().to_string();
println!("Public key: {}", public_key_string);
```

## Create and Sign DNS Records

Use `SignedPacket::builder()` to create DNS records and sign them with your keypair.

```rust
use pkarr::{Keypair, SignedPacket};

let keypair = Keypair::random();

let signed_packet = SignedPacket::builder()
    // A record (IPv4 address)
    .a(
        "www".try_into().unwrap(),
        "93.184.216.34".parse().unwrap(),
        3600,
    )
    // TXT record
    .txt(
        "_foo".try_into().unwrap(),
        "bar".try_into().unwrap(),
        30,
    )
    // Sign with your keypair
    .sign(&keypair)
    .unwrap();
```

### Record Types

The builder supports these common record types:

- `.a(name, ipv4, ttl)` - A record (IPv4)
- `.aaaa(name, ipv6, ttl)` - AAAA record (IPv6)
- `.address(name, ip, ttl)` - Auto-selects A or AAAA based on IP type
- `.txt(name, text, ttl)` - TXT record
- `.cname(name, target, ttl)` - CNAME record

Use `.` as the name to create records at the apex (the public key itself).

## Publish

```rust
use pkarr::{Client, Keypair, SignedPacket};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder().build()?;
    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder()
        .txt("_foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)?;

    println!("Publishing {} ...", keypair.public_key());

    client.publish(&signed_packet, None).await?;

    println!("Published successfully!");
    Ok(())
}
```

Publishing sends your signed packet to the Mainline DHT and configured relays. The second argument to `publish()` is an optional CAS (compare-and-swap) timestamp for conflict detection.

## Resolve

```rust
use pkarr::{Client, PublicKey};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder().build()?;

    // Parse public key from z-base32 string
    let public_key: PublicKey = "pk:yqrx81zchh6aotjj85s96gdqbmsoprxr3ks6ks6y8eccpj8b7oiy"
        .try_into()
        .expect("Invalid public key");

    match client.resolve(&public_key).await {
        Some(signed_packet) => {
            println!("Resolved packet:");
            println!("{}", signed_packet);

            // Iterate over specific records
            for record in signed_packet.resource_records("_foo") {
                println!("Record: {:?}", record.rdata);
            }
        }
        None => {
            println!("No packet found for {}", public_key);
        }
    }

    Ok(())
}
```

Use `resolve_most_recent()` instead of `resolve()` when you need the latest version (e.g., before publishing updates).

## Complete Example

A copy-paste example that generates a keypair, publishes a record, resolves it, and prints the result.

```rust
use pkarr::{Client, Keypair, SignedPacket};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create client and keypair
    let client = Client::builder().build()?;
    let keypair = Keypair::random();

    println!("Generated keypair with public key: {}", keypair.public_key());

    // 2. Create a signed packet with an A record
    let signed_packet = SignedPacket::builder()
        .a(
            "www".try_into().unwrap(),
            "93.184.216.34".parse().unwrap(),
            3600,
        )
        .txt(
            "_hello".try_into().unwrap(),
            "world".try_into().unwrap(),
            30,
        )
        .sign(&keypair)?;

    // 3. Publish to DHT and relays
    println!("Publishing...");
    client.publish(&signed_packet, None).await?;
    println!("Published successfully!");

    // 4. Resolve it back
    println!("Resolving...");
    match client.resolve(&keypair.public_key()).await {
        Some(resolved) => {
            // 5. Print results
            println!("\nResolved packet:\n{}", resolved);
        }
        None => {
            println!("Failed to resolve (this can happen on first publish)");
        }
    }

    Ok(())
}
```

Expected output:

```
Generated keypair with public key: <52-character z-base32 string>
Publishing...
Published successfully!
Resolving...

Resolved packet:
SignedPacket (<public_key>):
    last_seen: 0 seconds ago
    timestamp: <timestamp>
    signature: <signature>
    records:
        www.<public_key>  IN  3600  A  93.184.216.34
        _hello.<public_key>  IN  30  TXT  "world"
```

## Next Steps

- [API Documentation](https://docs.rs/pkarr/latest/pkarr/) - Full reference
- [Examples](https://github.com/Pubky/pkarr/tree/main/pkarr/examples) - More code samples
- [Features](./features.md) - Available feature flags and configurations
