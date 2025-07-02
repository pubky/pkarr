# Pkarr WASM

WebAssembly bindings for [Pkarr](https://pkarr.org) - Public-Key Addressable Resource Records for publishing and resolving DNS packets over [Mainline DHT](https://github.com/Pubky/mainline).

## 🔨 Building from Source

To build the WASM package from source, run:

```bash
./build-wasm.sh

# Test your changes
cd pkg && npm run test

# Run examples
npm run example
npm run example:advanced
```

## 🚀 Quick Start

```javascript
const { Client, Keypair, SignedPacket } = require('pkarr');

// Create a new keypair and client
const keypair = new Keypair();
const client = new Client();

// Build a DNS packet
const builder = SignedPacket.builder();
builder.addTxtRecord("_service", "pkarr=v1.0", 3600);
builder.addARecord("www", "192.168.1.1", 3600);

// Sign and publish
const packet = builder.buildAndSign(keypair);
await client.publish(packet);

// Resolve later
const resolved = await client.resolve(keypair.public_key_string());
console.log('Records:', resolved.records);
```

## 📦 Installation

This package is generated via `wasm-pack` and includes TypeScript definitions.

```bash
npm install pkarr
```

## 🏗️ API Overview

### Core Classes

- **`Client`** - Publish and resolve packets via Pubky relays
- **`Keypair`** - Generate and manage Ed25519 keypairs
- **`SignedPacket`** - Build and sign DNS packets
- **`Utils`** - Utility functions for validation

### Client Methods

```javascript
const client = new Client();                    // Default relays
const client = new Client(relays, timeout);     // Custom configuration

await client.publish(packet);                   // Publish a signed packet
await client.publish(packet, casTimestamp);     // Compare-and-swap publish
const packet = await client.resolve(publicKey); // Resolve most recent packet
const packet = await client.resolveMostRecent(publicKey); // Alternative resolve method

const relays = Client.defaultRelays();          // Get default relay list
```

### Keypair Operations

```javascript
const keypair = new Keypair();                    // Generate new keypair
const keypair = Keypair.from_secret_key(bytes);   // From existing secret

const publicKey = keypair.public_key_string();    // Base32 public key
const publicBytes = keypair.public_key_bytes();   // Raw public key bytes
const secretBytes = keypair.secret_key_bytes();   // Raw secret key bytes
```

### Packet Building

```javascript
const builder = SignedPacket.builder();

// Add DNS records (all 7 supported types)
builder.addTxtRecord(name, value, ttl);         // TXT records
builder.addARecord(name, ipv4, ttl);            // IPv4 addresses
builder.addAAAARecord(name, ipv6, ttl);         // IPv6 addresses
builder.addCnameRecord(name, target, ttl);      // Canonical names
builder.addHttpsRecord(name, priority, target, ttl);  // HTTPS service records
builder.addSvcbRecord(name, priority, target, ttl);   // Service binding records
builder.addNsRecord(name, nameserver, ttl);     // Name server records

// Optional: Set custom timestamp
builder.setTimestamp(Date.now());

// Build and sign
const packet = builder.buildAndSign(keypair);

// Access packet data
console.log(packet.publicKeyString);
console.log(packet.timestampMs);
console.log(packet.records);
```

### Compare-and-Swap (CAS) Publishing

Prevent race conditions when multiple clients update the same DNS records:

```javascript
// Step 1: Read current state
const currentPacket = await client.resolveMostRecent(publicKey);

// Step 2: Create updated packet
const builder = SignedPacket.builder();
builder.addTxtRecord("version", "2.0", 3600);
const updatedPacket = builder.buildAndSign(keypair);

// Step 3: Conditional publish
try {
    const casTimestamp = currentPacket ? currentPacket.timestampMs / 1000 : null;
    await client.publish(updatedPacket, casTimestamp);
    console.log('✅ CAS publish successful');
} catch (error) {
    console.log('❌ CAS failed - state was modified by another client');
    // Re-read current state and retry
}
```

CAS ensures your update only succeeds if the server state hasn't changed since you last read it.

## 🧪 Examples

Run the included examples to see Pkarr in action:

```bash
npm run example          # Basic publish/resolve workflow
npm run example:advanced # Advanced usage with multiple record types
```

### Example Files

- **`examples/base.js`** - Basic usage patterns
- **`examples/advanced.js`** - Complex scenarios and best practices

## 🔬 Testing

Comprehensive test suite with 100% pass rate across all areas:

```bash
npm run test                  # Run all test suites (recommended)
npm run test:unit            # Unit tests (29 tests) - Core functionality
npm run test:integration     # Integration tests (7 tests) - Live network
npm run test:performance     # Performance benchmarks - Speed analysis  
npm run test:edge-cases      # Edge cases (21 tests) - Error handling
```

### Test Coverage

- ✅ **Unit Tests** (29/29): Core WASM functionality, keypairs, packet building, API compatibility
- ✅ **Integration Tests** (7/7): Live network operations with Pubky relays
- ✅ **Performance Tests**: Benchmarks and memory analysis across all operations
- ✅ **Edge Cases** (21/21): Error handling, input validation, boundary conditions

### Performance Characteristics

- **Keypair Generation**: Fast cryptographic key generation using Ed25519
- **Packet Building**: Efficient DNS packet construction and signing
- **Record Operations**: Optimized handling of all 7 DNS record types
- **Network Operations**: Live relay communication with configurable timeouts
- **Memory Usage**: Efficient WASM memory management with stress testing
- **Concurrent Operations**: Full async/await support with parallel publish/resolve

## 🌐 Network Operations

### Default Relays

The client by default uses the following relays:
- `https://pkarr.pubky.app`
- `https://pkarr.pubky.org`

### Custom Configuration

```javascript
const customRelays = ['http://localhost:15411'];
const client = new Client(customRelays, 10000); // 10s timeout
```

## 🛡️ Error Handling

The WASM bindings include robust error handling with comprehensive input validation:

```javascript
try {
    const packet = await client.resolve(publicKey);
    if (packet) {
        console.log('Found packet:', packet.records);
    } else {
        console.log('No packet found for key');
    }
} catch (error) {
    console.error('Resolution failed:', error.message);
}
```

### Enhanced Validation

- **Input Validation**: All DNS names, IP addresses, and parameters are validated
- **Timeout Validation**: Client timeouts must be between 1-300 seconds
- **Key Validation**: Ed25519 keys are validated for proper length and format
- **Packet Size Limits**: Enforces 1000-byte pkarr specification limit
- **Error Context**: Detailed error messages with context for debugging

## 🔧 TypeScript Support

Full TypeScript definitions included:

```typescript
import { Client, Keypair, SignedPacket } from 'pkarr';

const client: Client = new Client();
const keypair: Keypair = new Keypair();
const builder = SignedPacket.builder();
```

## 📄 License

MIT License - see LICENSE file for details.
