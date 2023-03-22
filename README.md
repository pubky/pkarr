# Pkarr

> Public-Key Addressable Resource Records

This project aims to investigate the possibility of using Distributed Hash Tables (DHTs) and/or other overlay networks to create a distributed alternative DNS root where public keys serve as the self-certifying top-level domain.

Where we are going, this https://j9afjgmrb65bipi6wreogf8b1emczatecuy9tuzbbwnzsdacpohy resolves!

## Why

In pursuit of a sovereign, distributed, and open web, we identify three challenges:

1. **Distributed Semantics** `Everything expressed as keys and metadata`
Developing interoperable semantics for verifiable metadata about a set of public-keys that form a digital identity, complete with reputation, social graph, credentials, and more.

2. **Distributed Database(s)** `Anyone can host the data`
Verifiable data alone is insufficient; a host-agnostic database is essential for an open web, as opposed to walled gardens.

3. **Distributed Discovery** `Where is the data?`
But before that, you need to efficiently and consistently discover the multiple hosts for a given data-set.

Addressing Distributed Discovery first makes the most sense for several reasons:

- The difficulty of these three challenges is inversely correlated with their order.
- The marginal utility of solving these challenges positively correlates with their order.

    In existing and emerging open social network protocols, users do tolerate limited interoperability between clients, second-class identifiers controlled by hosting or domain servers, inefficient conflict-free replication between data stores, and the absence of local-first or offline support. However, their most common complaints involve unavailability, censorship, deplatforming, and difficulty in securely managing keys.

- Distributed Discovery offers the greatest assured advantage by abstracting over current and emerging solutions for (1) and (2) as they compete, complement, and develop independently. This approach also ensures that users' identifiers (and those of their contacts) remain stable for as long as possible since their primary secret key is used infrequently and is easier to secure.

### Why not [insert ad hoc solution] instead?
Open social networks often attempt to solve discovery natively within their network of participants. However, this approach has several issues:

- It may conflict with participants' (usually service providers) self-interest in keeping users locked in.
- Their infrastructure would need to become a gossip overlay network, which may not be desirable.
- Achieving consistency and load balancing would require further optimization, effectively reinventing a DHT.
- If an overlay network is developed that surpasses the performance of a 10-million-node DHT with a 15-year track record, Pkarr should still be capable of utilizing your network as a backend, either as an alternative or alongside existing solutions.

## How

### Architecture

There are 3 main classes of participants in Pkarr:
1. **P2P Nodes**: are direct participants of the underlying p2p overlay network(s).
2. **Resolvers**: which are nodes that double as "Recursive and caching name server", as well as accepting requests to publish signed records.
3. **Issuers**: creates and signs records and submit them to resolvers.
4. **Refreshers**: a server that refreshes signed records periodically to keep them alive. can be resolvers or issuers.

### Legacy support

For clients unaware of Pkarr, Users _have_ to add a trusted resolver to their operating system's DNS servers configuration.

That means resolvers need to offer the same API expected from any other recursive name server.
Preferably 

### Queries
    
For applications that want to use Pkarr without requiring manual setup from the user, they can directly make requests to a hardcoded set of resolvers.

Resolvers need to implement [DoH](https://www.rfc-editor.org/rfc/rfc8484) to support web apps.

Additional HTTP endpoint needs to be added to resolvers to pass the raw data and its signature to allow applications to verify signatures themselves instead of trusting resolvers.

### Publishing

Clients with access to private keys, can generate records, and send them to any available resolvers, but for maximum censorship resistance they should be running their own resolvers.

Mobile wallets can have embedded resolvers, especially if Nat traversal and holepunching is possible in the underlying overlay network, or through other means.

Resolvers need to have another endpoint that transparently exposes the options of the publishing operation in the overlay network through an HTTP endpoint.

#### use cases

1. Alice is hosting her blog on a paid web hosting service that offers refreshing Alice's records.
2. Bob mentiones their LN node address in a TXT record, so they run a refresher along side said node.
3. Carol wants to migrate from a service provider to another in an open network, so she updates here corrisponding record.

### Scaling

To ensure a good chance of scalability and resilience, a few expectations need to be set straight:

1. Records are ephemeral, without refreshing them, they will be dropped by the underlying p2p network.
2. Resolvers need to be good citizens that follow the recommendations and optimizations needed to reduce the traffic on their peers, including heavily caching records and respecting TTLs.
3. Resolvers that spam their peers will probably be blocked.
4. On cache miss, queries might take few seconds to traverse the DHT.
5. Records are expected to be read often, and rarely updated mostly when users need to switch service providers.
6. Records neeed to be very small to fit in an MTU, so a max size of 1000 bytes is enforced.

### Bootstrapping and Incentives

We will utilize established networks that have proven their scalability to avoid relying on nascent or speculative solutions. By maintaining our expectations for a permissionless, efficient, and distributed relay of ephemeral small data, we ensure that the cost of running a node remains low.

The remaining infrastructure components are the Pkarr resolvers, which can operate as permissioned nodes, volunteers, or be incentivized through Bitcoin donations.

Crucially, these resolvers benefit from existing optimizations of DNS caching at the operating system and router levels.

## What

Pkarr is a work in progress, but these are the choices made so far.

### Overlay network
Pkarr will use [Mainline_DHT](https://en.wikipedia.org/wiki/Mainline_DHT) as the overlay network.
Specifically [BEP44](https://www.bittorrent.org/beps/bep_0044.html) for storing ephemeral arbitrary data.

Reasons for choosing Mainline include:
1. 15 years of proven track record facilitating trackerless torrent for people around the world.
2. Biggest DHT in existence with estimated 10 million nodes.
3. It is fairly generous with its retaining of mutable data, reducing the need to frequently refresh records, thus reducing traffic.
4. It has implementation in most languagues, well understood, simple and stable enough to make a minimal implementation from scratch if we need to.

### Keys

ED25519 keys, mostly because that is what Mainline users, but so does almost every overlay network we may consider.

There are ways to support other keys, but they will need more complex solutions than BEP44.

Supporting multiple keys is not neccessary though, as you can advertise other cryptogarphic primitives in TXT records.


### Endpoints

TODO

### Encoding

[CBOR](https://cbor.io/) with special replacements for known keys, for concisness.

TODO: add more details

### URL

To keep urls human readable and short enough to fit in url limitations, we encode the public key using [z-base32](https://philzimmermann.com/docs/human-oriented-base-32-encoding.txt).

example: `https://j9afjgmrb65bipi6wreogf8b1emczatecuy9tuzbbwnzsdacpohy`
