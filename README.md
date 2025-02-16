# Pkarr

> Public-Key Addressable Resource Records

[![Try the demo](https://img.shields.io/badge/Try%20the-Demo-blue)](https://app.pkarr.org) [![View Examples](https://img.shields.io/badge/View-Examples-green)](./pkarr/examples/README.md) [![Crates.io](https://img.shields.io/crates/v/pkarr)](https://crates.io/crates/pkarr) [![Documentation](https://img.shields.io/badge/docs-design-orange)](./design/README.md) [![License](https://img.shields.io/badge/license-MIT-purple)](./LICENSE)

*The simplest possible integration between DNS and P2P networks, enabling self-issued public keys to function as sovereign, censorship-resistant top-level domains.*

[Quick Start](#tldr) • [Architecture](#architecture) • [Documentation](./design/README.md) • [FAQ](#faq)

--- 

The simplest possible streamlined integration between the Domain Name System and peer-to-peer overlay networks, enabling self-issued public keys to function as sovereign, publicly addressable, censorship-resistant top-level domains. This system is accessible to anyone capable of maintaining a private key.

Where we are going, this [https://o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy](https://app.pkarr.org/?pk=o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy) resolves everywhere!

## TLDR
- To publish resource records for your key, sign a small encoded DNS packet (<= 1000 bytes) and publish it on the DHT (through a relay if necessary).
- To resolve a key's resources, applications query the DHT directly or through a [relay](./design/relays.md), and verify the signature themselves. 
- Clients and Relays extensively cache records and minimize DHT traffic for improved scalability. 
- The DHT drops records after a few hours, so users, their friends, or service providers need to periodically republish their records. Additionally, Pkarr relays can republish recently requested records to keep popular records alive.
- Optional: Existing applications unaware of Pkarr can still function if the user adds Pkarr-aware DNS servers to their operating system's DNS configuration. 

## Demo

Try the [web app demo](https://app.pkarr.org)

Or if you prefer Rust, check out our [Examples](./pkarr/examples/README.md) 

## Contents
- [Architecture](#architecture)
- [Expectations](#expectations)
- [Why](#why)
- [FAQ](#faq)
 
## Architecture

```mermaid
sequenceDiagram
    participant Client
    participant Relay
    participant DHT
    participant Republisher

    Client->>Relay: Publish
    note over Relay: Optional Pkarr Relay
    Relay->>DHT: Put
    Note over Relay,DHT: Store signed DNS packet

    Client->>Republisher: Republish request
    note over Client, Republisher: Notify Hosting provider mentioned in RRs

    loop Periodic Republish
        Republisher->>DHT: Republish
    end

    Client->>Relay: Resolve
    Relay->>DHT: Get
    DHT->>Relay: Response
    Relay->>Client: Response
```

### Clients
#### Pkarr-enabled Applications

Native applications can directly query and verify signed records from the DHT if they are not behind NAT. Otherwise, they will need to use a Pkarr Relay.

Browser web apps should try calling the local Pkarr relay at the default port `6881`. If not accessible, they must query a remote relay as a fallback. In either case, these apps should allow users to configure relays of their choice.
 
Clients with private keys can also submit signed records either directly to the DHT or through a Pkarr relay to update their records when needed.
 
#### Existing Applications
To support existing applications that are unaware of Pkarr, users will need to (manually or programmatically) edit their OS DNS servers to add one or more DNS servers that recognize Pkarr and query the DHT. However, the ideal outcome would be adoption by existing widely used resolvers like `1.1.1.1` (Cloudflare) and `8.8.8.8` (Google).

### Relays

Pkarr relays are optional but they:
1. Enable web applications to query the DHT through [relays](https://pkarr.org/relays)
2. Act as a large caching layer for many users to provide lower latency, better reliability, and improved scalability

Relays are very light and cheap to operate, making them easy to run altruistically. Private and paid relays are also possible.

### Republishers

Services and hosting providers mentioned in a user's Resource Records are incentivized to republish these records and keep them alive on the DHT, for the same reasons they are incentivized to gain that user in the first place.

### DHT

Pkarr uses [Mainline DHT](https://en.wikipedia.org/wiki/Mainline_DHT) as the overlay network,
specifically [BEP44](https://www.bittorrent.org/beps/bep_0044.html) for storing ephemeral arbitrary data.

Reasons for choosing Mainline include:
1. 15 years of proven track record facilitating trackerless torrents worldwide
2. Largest DHT in existence with an estimated 10 million nodes
3. Generous retention of mutable data, reducing the need for frequent record refreshes
4. Implementations available in most languages, well understood by many experts, and stable enough for minimal custom implementation if needed

## Expectations

To ensure good scalability and resilience, a few expectations need to be set:

1. This is **not a storage platform**
    - Records are ephemeral and will be dropped by the DHT without regular refreshing
    - Popular records may be refreshed by DNS servers as they receive queries
2. This is **not a realtime communication** medium
    - Records are heavily cached like in any DNS system
    - Record updates should be infrequent, and relays enforce strict rate-limiting
    - Record updates may take time to propagate due to extensive caching, even with a 1-second TTL
    - In case of a cache miss, traversing the DHT might take few seconds.

## Why?

> Why would you need resource records for keys?

In pursuit of a sovereign, distributed, and open web, we identify three challenges:

1. **Distributed Semantics**: `Everything expressed as keys and metadata`Developing interoperable semantics for verifiable metadata about a set of public keys that form a digital identity, complete with reputation, social graph, credentials, and more.

2. **Distributed Database(s)**: `Anyone can host the data`
Verifiable data alone is insufficient; a host-agnostic database is essential for an open web, as opposed to walled gardens.

3. **Distributed Discovery**: `Where is the data?`
Before anything else, you need to efficiently and consistently discover the multiple hosts for a given dataset.

Addressing Distributed Discovery first makes the most sense for several reasons:

- The difficulty of these three challenges inversely correlates with their order.
- The marginal utility of solving these challenges positively correlates with their order.

    In existing and emerging open social network protocols, users do tolerate limited interoperability between clients, second-class identifiers controlled by hosting or domain servers, inefficient or non-existent conflict-free replication between data stores, and the absence of local-first or offline support. However, their most common complaints involve unavailability, censorship, deplatforming, and difficulty in securely managing keys.

- Distributed Discovery offers the greatest assured leverage by abstracting over current and emerging solutions for (1) and (2) as they compete, complement, and develop independently, all while maintaining the same long lasting identifier, so you don't have to start from scratch or be locked in.

### Leverage

**Solve the most issues...**

Pkarr solves **unavailability** by turning public keys into resolvable URLs: resource **locator**.
Pkarr solves **censorship and deplatforming** by allowing users to conveniently change DNS records to point to other providers or platforms. While there are other ways to do this, it is never as reliable and authoritative as DNS.
Pkarr helps with **key management** by enabling users to maintain a long-lasting identity tied to one key, rarely used, and hopefully kept offline at all times.

Finally, by solving censorship and deplatforming in a sovereign way, the need for signed data becomes less urgent, and we buy more time to figure out the UX of signing everything everywhere all the time.

**... with least work possible ...**

Pkarr doesn't need to bootstrap anything or invent anything, instead using 15 years old battle tested Distributed Hash Table (Mainline DHT) with millions of nodes, and good old web servers.

## FAQ

<details>
<summary><strong>Why not human readable domains on a blockchain?</strong></summary>

Introducing scarcity to names, arguably the most subjective and personal thing in our lives, serves noone except rent seekers. We already know how to use phonebooks, we just need to upgrade small numbers, to bigger sovereign keys.
</details>

<details>
<summary><strong>Why not GNU Name System?</strong></summary>

The GNU net is exciting and impressive, but I didn't have enough time to test it or understand how hard it would be to build a PoC on top of it.

GNU name system seems to support [Petname system](http://www.skyhunter.com/marcs/petnames/IntroPetNames.html) natively, which means it does require more storage and bandwidth from the DHT than a 1000 bytes max size enforced by Mainline DHT. I believe that petnameing should be left to application layer. 

Luckily GNU net uses ed25519 key as well, so there is always a path for migration if we are careful.
</details>

<details>
<summary><strong>Why not [insert ad hoc solution] instead?</strong></summary>

Open social networks often attempt to solve discovery natively within their network of participants. However, this approach has several issues:
- It may conflict with participants' (usually service providers) self-interest in keeping users locked in.
- Their infrastructure would need to become a gossip overlay network, which may not be desirable.
- Achieving consistency and load balancing would require further optimization, effectively reinventing a DHT.
- If an overlay network is developed that surpasses the performance of a 10-million-node DHT with a 15-year track record, Pkarr should still be capable of utilizing your network as a backend, either as an alternative or alongside existing solutions.
</details>

<details>
<summary><strong>How can I run the Pkarr server?</strong></summary>

You can find building instruction [here](./server/README.md).
</details>

