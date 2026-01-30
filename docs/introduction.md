# Introduction to Pkarr

> Public-Key Addressable Resource Records

Pkarr is a system that turns Ed25519 public keys into sovereign, censorship-resistant
top-level domains. By publishing DNS records to the Mainline DHT, a 15-year-old
peer-to-peer network with over 10 million nodes, Pkarr enables anyone with a
private key to own their online identity without relying on registrars,
platforms, or centralized authorities.

## The Problem

The current internet has a fundamental ownership problem. Despite the promise of
the web as a decentralized medium, most online identities depend on centralized
gatekeepers.

### Domain Names Require Permission

Traditional DNS is hierarchical. To own a domain, you need a registrar. That
registrar operates under the authority of ICANN and various governments. Your
domain can be seized, suspended, or transferred without your consent. History
is full of examples: political dissidents losing their domains, businesses
disrupted by registrar disputes, and entire country-code TLDs becoming
unreliable due to geopolitical conflicts.

### Platforms Own Your Identity

When you create an account on a social platform, you are borrowing an identity.
Your username, your follower graph, your content, and your reputation all live
on servers you do not control. Deplatforming is not a hypothetical risk; it
happens routinely. When it does, you lose everything: your audience, your
history, and often your ability to redirect people to where you went next.

### Links Break When Services Disappear

Every URL that points to a centralized service is a promise that service will
keep. When platforms shut down, pivot, or simply change their URL structure,
those links die. The web accumulates link rot at an alarming rate, eroding the
connective tissue of human knowledge.

### Users Never Truly Own Anything

The combination of these factors means that most users have no real sovereignty
online. You cannot take your identity with you. You cannot guarantee your
content remains accessible. You are always renting, never owning.

## The Solution

Pkarr addresses these problems by making public keys function as domain names
and using a distributed network for discovery.

### Public Keys as Domains

In Pkarr, your identity is an Ed25519 public key. This key becomes your
top-level domain, no registrar required. Because you generated the key yourself
and you alone hold the private key, no one can take it from you. Your identity
is cryptographic, not bureaucratic.

A Pkarr public key looks like this:

```
o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy
```

This 52-character string is a z-base32 encoded Ed25519 public key. It can
function as a TLD in URIs, allowing URLs like:

```
https://o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy
```

### Mainline DHT for Discovery

Pkarr stores DNS records on the Mainline DHT, the same distributed hash table
that powers trackerless BitTorrent. This network has been running for 15 years,
hosts an estimated 10 million nodes, and has proven remarkably resilient. By
building on existing infrastructure rather than bootstrapping a new network,
Pkarr inherits battle-tested reliability.

### Standard DNS Records, Self-Signed

Pkarr packets contain ordinary DNS resource records: A records, AAAA records,
TXT records, CNAME records, and so on. The difference is that you sign these
records yourself with your private key. Anyone who retrieves your records can
verify the signature against your public key, ensuring authenticity without
trusting any intermediary.

### Change Providers Without Losing Identity

Because you control your DNS records, you can point them anywhere. If a hosting
provider becomes unreliable, you update your records to point elsewhere. If a
platform bans you, you redirect your identity to a new home. Your public key
remains constant; only the records change.

## Key Concepts

### SignedPacket

The fundamental data unit in Pkarr is the SignedPacket. It contains everything
needed to verify and use a set of DNS records:

| Component | Size | Description |
|-----------|------|-------------|
| Public Key | 32 bytes | Ed25519 public key identifying the owner |
| Signature | 64 bytes | Ed25519 signature proving authenticity |
| Timestamp | 8 bytes | Microsecond UNIX timestamp for versioning |
| DNS Packet | Up to 1000 bytes | Compressed DNS resource records |

The timestamp ensures that newer records supersede older ones. The signature
covers both the timestamp and the DNS packet, preventing tampering. The 1000-byte
limit on the DNS packet is a constraint of the Mainline DHT.

### PublicKey Encoding

Pkarr uses z-base32 encoding to represent public keys as human-typeable strings.
The encoding produces 52-character strings using only lowercase letters and
digits, making them suitable for use in domain names and URIs.

### Ephemeral Storage and Republishing

The Mainline DHT does not store data permanently. Records are dropped after
hours or days if not refreshed. This is a feature, not a bug: it prevents the
network from accumulating stale data indefinitely.

However, it means that active records must be republished periodically. This
can be done by:

- The user themselves, running a client that republishes automatically
- Friends or associates who care about keeping the records alive
- Hosting providers mentioned in the records, who are incentivized to keep
  their customers discoverable

### Relays

Relays are HTTP gateways to the DHT.

**Browser access**: Web browsers cannot directly participate in the DHT.
   Relays provide an HTTP API that browser-based applications can use.

Relays also serve as caching layers, reducing DHT traffic and improving
response times for frequently-accessed records.

## Why Not Alternatives?

### Blockchain-Based Domain Systems

Systems like ENS (Ethereum Name Service) or Handshake offer human-readable
names on blockchains. However, they introduce significant drawbacks:

- **Cost**: Registering and updating records requires transaction fees
- **Speed**: Blockchain confirmations take time
- **Environmental impact**: Proof-of-work chains consume substantial energy
- **Chain lock-in**: Your identity becomes tied to a specific blockchain's
  fortunes
- **Artificial scarcity**: Human-readable names create speculation and
  rent-seeking behavior around desirable names

Pkarr avoids all of these by using keys directly. There is no scarcity in
public keys, no fees to update records, and no dependency on any specific
chain.

### GNU Name System

GNS is a sophisticated system with native support for petnames and complex
identity delegation. However, it requires adopting the entire GNU net stack
and has seen limited adoption. It also requires more storage and bandwidth
from its DHT than the 1000-byte limit that Mainline enforces.

Pkarr takes a minimalist approach: leverage existing infrastructure and leave
petname systems to the application layer.

### Ad-Hoc Protocol Solutions

Many open social protocols attempt to solve discovery within their own network
of participants. This approach has problems:

- Service providers may resist, preferring to keep users locked in
- The protocol infrastructure must become a gossip network
- Achieving consistency effectively means reinventing a DHT
- Each protocol reinvents the wheel independently

Pkarr provides a shared discovery layer that any protocol can use, avoiding
fragmentation.

## The Trade-offs

Pkarr makes explicit trade-offs in pursuit of simplicity and sovereignty.

### 1000-Byte Packet Limit

The Mainline DHT limits mutable data to 1000 bytes. This is enough for several
DNS records but not for storing large amounts of data. Pkarr is a discovery
layer, not a storage platform. Use your records to point to where your data
actually lives.

### Ephemeral Records

Records must be republished periodically or they disappear. This requires
either running your own republishing client or relying on others to republish
for you. For most users, hosting providers and relays handle this automatically.

### No Human-Readable Names

Public keys are not memorable. You cannot tell someone your Pkarr address over
the phone. This is the price of true sovereignty: names that humans find
convenient require some form of registry, and registries introduce
centralization.

However, you can build human-readable naming on top of Pkarr. A petname system,
a personal phonebook, or even a traditional DNS record that points to a Pkarr
key can all provide human-friendly interfaces while preserving the underlying
sovereignty.

### DHT Latency

Traversing the DHT can take several seconds on a cache miss. Pkarr mitigates
this through aggressive caching at multiple layers: clients cache locally,
relays cache for their users, and frequently-accessed records stay warm in
the DHT itself.

## Philosophy

### Sovereignty First

The core principle of Pkarr is that users should control their own identity.
No registrar, no platform, no government should be able to revoke your ability
to be found online. Cryptographic keys make this possible.

### Build on Existing Infrastructure

Rather than bootstrapping a new network and hoping for adoption, Pkarr builds
on the Mainline DHT. This network already exists, already works, and already
has millions of participants. Pkarr simply uses it for a new purpose.

### Minimalism Over Features

Pkarr does one thing: map public keys to DNS records via the DHT. It does not
try to solve human-readable naming, data storage, social graphs, or identity
semantics. Each of those problems can be solved in layers above Pkarr, using
Pkarr as the foundation.

### Interoperability Through Standards

By using standard DNS records, Pkarr can integrate with existing systems. A
DNS resolver that understands Pkarr can serve records to any application that
speaks DNS. This means Pkarr can work with existing software, not just
purpose-built clients.

## Getting Started

To explore Pkarr:

- Try the [web demo](https://pkdns.net) to resolve records
- Read the [specification](../design/base.md) for implementation details
- Check the [examples](../pkarr/examples/README.md) for code samples
- Run your own [relay](../design/relays.md) to support the network

Pkarr gives you a permanent, portable, sovereign identity. What you build on
top of it is up to you.
