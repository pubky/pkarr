# Pkarr

> Public-Key Addressable Resource Records

This project explores the potential of using Distributed Hash Tables (DHTs) and other overlay networks to map keys to small [resource records](https://en.wikipedia.org/wiki/Domain_Name_System#Resource_records), effectively turning any ed25519 key into a sovereign Top-Level Domain.

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
