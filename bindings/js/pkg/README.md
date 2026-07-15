# Pkarr

Public-Key Addressable Resource Records for publishing and resolving DNS packets over [Mainline DHT](https://github.com/Pubky/mainline)

### 📦 Installation

This package is generated via `wasm-pack` and includes TypeScript definitions.

```bash
npm install @synonymdev/pkarr
```

### 🚀 Quick Start

```javascript
const { Client, Keypair, ResolvePolicy, SignedPacket } = require('@synonymdev/pkarr');

// Create a new keypair and client
const keypair = new Keypair();
const client = new Client();

// Build a DNS packet
const builder = SignedPacket.builder();
builder.addTxtRecord("_service", "pkarr=v1.0", 3600);
builder.addARecord("www", "192.168.1.1", 3600);

// Sign and publish
const packet = builder.buildAndSign(keypair);
const storedNodeCount = await client.publish(packet);
console.log(`Stored on at least ${storedNodeCount} DHT nodes`);

// Resolve later
const resolved = await client.resolve(
    keypair.publicKeyString(),
    ResolvePolicy.CacheFirst,
);
console.log('Records:', resolved.records);
```

### 🏗️ API Overview

#### Core Classes

- **`Client`** - Publish and resolve packets via Pubky relays
- **`ResolvePolicy`** - Select cache-only, cache-first, or network-only resolution
- **`Keypair`** - Generate and manage Ed25519 keypairs
- **`SignedPacket`** - Build and sign DNS packets
- **`Utils`** - Utility functions for validation

#### Client Methods

```javascript
const { Client, ResolvePolicy } = require('@synonymdev/pkarr');

const client = new Client();                          // Default relays and 2s timeout
const customClient = new Client(relays, timeout);     // Custom configuration

const storedNodeCount = await client.publish(packet); // Minimum known DHT node count
const cached = await client.resolve(publicKey, ResolvePolicy.CacheOnly);
const fresh = await client.resolve(publicKey, ResolvePolicy.CacheFirst);
const latest = await client.resolve(publicKey, ResolvePolicy.NetworkOnly);

const relays = Client.defaultRelays();          // Get default relay list
```

#### Keypair Operations

```javascript
const { Keypair } = require('@synonymdev/pkarr');

const keypair = new Keypair();                    // Generate new keypair
const restored = Keypair.fromSecretKey(bytes);    // Restore from an existing secret

const publicKey = keypair.publicKeyString();      // Base32 public key
const publicBytes = keypair.publicKeyBytes();     // Raw public key bytes
const secretBytes = keypair.secretKeyBytes();     // Raw secret key bytes
```

#### Packet Building

```javascript
const { SignedPacket } = require('@synonymdev/pkarr');

const builder = SignedPacket.builder();

// Add DNS records (all 7 supported types)
builder.addTxtRecord(name, value, ttl);         // TXT records
builder.addARecord(name, ipv4, ttl);            // IPv4 addresses
builder.addAAAARecord(name, ipv6, ttl);         // IPv6 addresses
builder.addCnameRecord(name, target, ttl);      // Canonical names
builder.addNsRecord(name, nameserver, ttl);     // Name server records

// SVCB/HTTPS Parameters
const params = {
    port: 443,                           // Port number
    ipv4hint: "192.0.2.1",              // IPv4 hints (string or array)
    ipv6hint: ["2001:db8::1"],          // IPv6 hints (string or array)
    alpn: ["h2", "http/1.1"],           // ALPN protocol IDs
};

builder.addHttpsRecord(name, priority, target, ttl, params);  // HTTPS service records
builder.addSvcbRecord(name, priority, target, ttl, params);   // Service binding records

// Optional: Set custom timestamp
builder.setTimestamp(Date.now());

// Build and sign
const packet = builder.buildAndSign(keypair);

// Access packet data
console.log(packet.publicKeyString);
console.log(packet.timestampMs);
console.log(packet.records);
```

`resolve` rejects when no packet is found or the configured relays fail. Pkarr errors are
JavaScript `Error` objects with a stable `code`, such as `NotFound`, `NoResponses`, or
`NotMostRecent`.

### 🧪 Examples

Run the example to see Pkarr in action:

```bash
npm run example          # Basic publish/resolve workflow
```

### 🌐 Network Operations

The client by **default** uses the following relays:
- `https://pkarr.pubky.app`
- `https://pkarr.pubky.org`

#### Custom Configuration

To publish the records to a custom relay, compile the `pkarr-relay` binary from source. Navigate to the pkarr [repository](https://github.com/pubky/pkarr) and:

```bash
cargo build
./target/debug/pkarr-relay --testnet
```

After configuring your custom relay, you can initialize a client with custom relay URLs:

```javascript
const { Client } = require('@synonymdev/pkarr');

const customRelays = ['http://localhost:15411'];
const client = new Client(customRelays, 10000); // 10s timeout
```

### 🧪 Testing and Examples

To run tests and examples:

```bash
# Run tests. NOTE: Integration tests require a local pkarr-relay server (see Custom Configuration above)
cd pkg
npm run test

# Run examples
npm run example
```

### 📄 License

MIT License - see LICENSE file for details.
