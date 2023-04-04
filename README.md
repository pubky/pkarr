# Pkarr

> Public-Key Addressable Resource Records

The simplest possible streamlined integration between the Domain Name System and peer-to-peer overlay networks, enabling self-issued public keys to function as sovereign, publicly addressable domains. This system would be accessible to anyone capable of maintaining a private key.

Where we are going, this https://j9afjgmrb65bipi6wreogf8b1emczatecuy9tuzbbwnzsdacpohy resolves!

## TLDR
To publish DNS records for your key, sign a small payload (<1000 bytes) and send it to a custom DNS server that commits it to a DHT. To resolve websites or resources belonging to others' keys, applications send regular [DNS Queries over HTTPS (DoH)](https://www.rfc-editor.org/rfc/rfc8484) to Pkarr DNS servers or request the signed payload to verify themselves. Pkarr servers cache records extensively and minimize DHT traffic as much as possible for improved scalability. The DHT drops records after a few hours, but if a refresher (you manually, or the services mentioned in the records, or a volunteer) recommits the signed payload periodically, high availability is maintained for DNS resolvers making first uncached requests.

## DEMO 

Try the [web app demo](https://pkarr.nuhvi.com).

Or if you prefer a [CLI](./js/README.md#cli) 

## Why would you need resource records for keys?

In pursuit of a sovereign, distributed, and open web, we identify three challenges:

1. **Distributed Semantics** `Everything expressed as keys and metadata`
Developing interoperable semantics for verifiable metadata about a set of public-keys that form a digital identity, complete with reputation, social graph, credentials, and more.

2. **Distributed Database(s)** `Anyone can host the data`
Verifiable data alone is insufficient; a host-agnostic database is essential for an open web, as opposed to walled gardens.

3. **Distributed Discovery** `Where is the data?`
But before that, you need to efficiently and consistently discover the multiple hosts for a given data-set.

Addressing Distributed Discovery first makes the most sense for several reasons:

- The difficulty of these three challenges inversely correlates with their order.
- The marginal utility of solving these challenges positively correlates with their order.

    In existing and emerging open social network protocols, users do tolerate limited interoperability between clients, second-class identifiers controlled by hosting or domain servers, inefficient or non-existent conflict-free replication between data stores, and the absence of local-first or offline support. However, their most common complaints involve unavailability, censorship, deplatforming, and difficulty in securely managing keys.

- Distributed Discovery offers the greatest assured leverage by abstracting over current and emerging solutions for (1) and (2) as they compete, complement, and develop independently, all while maintaining the same long lasting identifier, so you don't have to start from scratch or be locked in.

### leverage

**Solve the most issues...**

Pkarr solves **unavailability** by turning public-keys to resolvable URL: resource **locator**.
Pkarr solves **censorship and deplatforming** by allowing users to conveniently change DNS records to point to other providers or platforms. While there are other ways to do that, it is never as reliable and authoritative as DNS.
Pkarr help with **key management** by enabling users to maintain a long lasting identity tied to one key, rarely used, and hopefully kept offline at all times.

Finally, by solving censorship and deplatforming ina sovereign way, the need for signed data becomes less urgent, and we buy more time to figure out the UX of signing everything everywhere all the time.

**with least work**

Pkarr doesn't need to bootstrap anything or invent anything, instead using 15 years old battle tested Distributed Hash Table with millions of nodes, and good old web servers.
 
## Architecture

### Clients
 #### Recursive DNS resolvers.
 
These are existing resolvers in every operating system, completely oblivious to Pkarr, yet by adding one or more trusted Pkarr servers (maybe even running your own), URLs like https://j9afjgmrb65bipi6wreogf8b1emczatecuy9tuzbbwnzsdacpohy should work in browsers and any other application on your device.

 #### Pkarr enabled applications.
 
 For applications aware of Pkarr, they can directly query the signed payload and verify it themselves in a trustless manner.
 To increase user's privacy they should allow users to use a custom server of their choosing, including a local server if that's available.
 
 These clients are also capable of submitting signed records to any server to be published on the DHT on their behalf.

### Servers

Good old DNS over HTTPs servers, with couple additional responsilities:
    
 1. Relaying Get and Put messages from clients to the DHT.
 2. Implement Caching and backoff to reduce traffic on the DHT.

## Use cases

1. Alice is hosting her blog on a paid web hosting service that offers refreshing Alice's records.
2. Bob mentiones their LN node address in a TXT record, so they run a refresher along side said node.
3. Carol wants to migrate from a service provider to another in an open network, so she updates here corrisponding record.

If you think of more, please open a PR.

## Expectations

To ensure a good chance of scalability and resilience, a few expectations need to be set straight:

1. This is **not a storage platform**
    - Records are ephemeral, and without refreshing them regularly they will be dropped by the DHT.
    - Popular records may or may not be refreshed by the DNS servers as they get queries for them.
2. This is **not a realtime communication** medium
    - Records are heavily cached like in any DNS system.
    - You are expected to update your records rarely, so you should expect servers to enforce harsh rate-limiting and maybe demand proof of work.
    - Records are going to be cached heavily to reduce traffic on the DHT, so updates might take some time to propagate, even if you set TTL to 1 second.
    - In case of a chache miss, traversing the DHT might take few seconds.

## Bootstrapping and Incentives

### clients

By conforming to DNS spec, we ensure that exsiting clients like browsers don't need to do anything, once the user setup their DNS settings to use a set Pkarr servers.

For applications that want to opt-in without requiring such manual configuration, or if they need to verify the signed records, they only need an extra HTTPs call and few lines of code.

### Servers

Pkarr server should be light enough that you can run one on your own, locally or on your VPS.
They should also be be cheap enough to volunteer running one for your friends or community.
Donations can keep more popular and beefy servers running.

Businesses can offer private servers that offer resolving records, and regular refreshing to keep records alive on the DHT.

Service providers (e.g a web hosting service) are also incintivized to refresh their customer's records as long as they are pointing to them.

Crucially, thanks to the long history of optimization and infrastructure around DNS, these servers benefit from many layers of caching DNS queries.

### DHT

Pkarr will use [Mainline_DHT](https://en.wikipedia.org/wiki/Mainline_DHT) as the overlay network.
Specifically [BEP44](https://www.bittorrent.org/beps/bep_0044.html) for storing ephemeral arbitrary data.

Reasons for choosing Mainline include:
1. 15 years of proven track record facilitating trackerless torrent for people around the world.
2. Biggest DHT in existence with estimated 10 million nodes.
3. It is fairly generous with its retaining of mutable data, reducing the need to frequently refresh records, thus reducing traffic.
4. It has implementation in most languagues, well understood (by many smart people, that may be willing to guide us), and stable enough to make a minimal implementation from scratch if we need to.

Servers must diligently stay good citizens within the Mainline DHT and strive to minimize traffic across it. This ensures that the marginal increase in the cost of operating DHT nodes, resulting from Pkarr, remains insignificant.

The client-server architecture enables the coordination and potential migration to more efficient alternatives, or even the parallel use of other DHTs alongside Mainline DHT. However, it is challenging to surpass the performance of a system that has proven to be effective and reliable for such an extended period.


## Roadmap

1. [x] Implement a web server as a mere proxy over DHT.
2. [x] Implement client-side tools for making queries and validating responses.
3. [x] Implement a web app to represent the UX and DX of client side.
4. [ ] Add optional caching and republishing of records on GET requests **<< we ar here**
5. [ ] Implement a standalone republisher and test keeping 100s of thousands of records alive.
6. [ ] Reimplement everything in Rust once the JS implementation is tested with some scale.
7. [ ] Add the DNS over HTTPS features and documentations for adding Pkarr servers in OS and browser's configurations.

## FAQ

1. **Why not human readable domains on a blockchain?**

    Introducing scarcity to names, arguably the most subjective and personal thing in our lives, serves noone except rent seekers. We already know how to use phonebooks, we just need to upgrade small numbers, to bigger sovereign keys.
    
1. **Why not GNU Name System?**

    The GNU net is exciting and impressive, but I didn't have enough time to test it or understand how hard it would be to build a PoC on top of it.
    
    GNU name system seems to support [Petname system](http://www.skyhunter.com/marcs/petnames/IntroPetNames.html)  natively, which means it does require more storage and bandwith from the DHT than a 1000 bytes max size enforced by Mainline DHT. I believe that petnameing should be left to application layer. 

    Luckily GNU net uses ed25519 key as well, so there is always a path for migration if we are careful.

2. **Why not [insert ad hoc solution] instead?**
Open social networks often attempt to solve discovery natively within their network of participants. However, this approach has several issues:
    - It may conflict with participants' (usually service providers) self-interest in keeping users locked in.
    - Their infrastructure would need to become a gossip overlay network, which may not be desirable.
    - Achieving consistency and load balancing would require further optimization, effectively reinventing a DHT.
    - If an overlay network is developed that surpasses the performance of a 10-million-node DHT with a 15-year track record, Pkarr should still be capable of utilizing your network as a backend, either as an alternative or alongside existing solutions.
