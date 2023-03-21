# Pkarr

> Public-Key addressable Resource Records

Exploration of the viability of using Distributed Hast Tables (DHTs) and other overlay networks to map keys to small [resource records](https://en.wikipedia.org/wiki/Domain_Name_System#Resource_records), effictively turning any ed25519 key into a sovereign Top Level Domain.

## Why

In pursuit of an sovereign yet distributed and open web, we identify three challenges:

1. **Distributed Semantics**`Your are keys and metadata` 
Interoperable semantics of verifiable metadata about set of public-keys forming a digital identity, with reputation, social graph, credentials, etc.

2. **Distributed Database(s)** `Anyone can host the data`
Verifiable data is not enough, a host agnostic database is a must for an open web instead of walled gardens.

3. **Distributed Discovery** `Where is the data?`
Finally, or rather fundamentally, you still need to discover the multitude of hosts for a given data-set, and you need to do so efficiently, and despite possible censorship attempts, either for political reasons, or bad incintives.


Starting with solving Distributed Discovery makes the most sense for few reasons:
    
- The difficulty of these three challenges inversly correlate with their order. 
  Hopefully that is self-evident.

- The marginal utility of solving these challenges positively correlates with their order.
 
    In existing and emerging open social network protocols users easily tolerate:
    
    - Limited interoperability between clients.
    - Second class identifier controlled by hosting or domain servers.
    - Lack of efficient conflict free replication between data stores.
    - Lack of local first or offline support and instead often relying on one server to control writes.
    
  But some of their most common complaints are:
    - Unavailability (can't find who currently hosts the data) 
    - Censorship and deplatforming (by centralized or federated servers)
    - Difficulty of securely managing keys
    
- Distributed Discovery offers the biggest guaranteed leverage:
    - DNS based identifiers is a stable occurunce in every protocol for a reason.
    - Abstract over existing and emerging solutions for (1) and (2) as they compete, compliment and evolve indenpendently.

### Why not [insert adhoc solution] instead?
Open social networks often attempt to solve discovery natively within their network of participants.
However, there are few issues with that:
- It usually goes against these participants (usually service providers) self interest of keeping users locked in.
- To fix that (if possible) their infrastructure will need to turn into a gossip overlay network, even if that is not desirable.
- Consistentncy and load balancing, will require more optimizations though, efficively reinventing a DHT. 
- Maybe you think you can do better than the biggest DHT with its 10 million nodes and 15 years of track record. If that happened, Pkarr should still be able to use your network as a backend instead of or in parallel with what already works now. 
