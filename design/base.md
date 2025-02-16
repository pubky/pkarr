# Pkarr Basic Specification

This document specifies the minimum required features for interoperability between Pkarr implementations.

## TLD

Pkarr uses [z-base32](https://philzimmermann.com/docs/human-oriented-base-32-encoding.txt) encoding to turn the 32 bytes of ed25519 public keys into Top-level domains compatible with DNS and URIs.

While a TLD should work with any URI scheme, sometimes printing the encoded public key alone lacks context, so it is advisable to add `pk:`:
```
pk:o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy
```

Implementations should be able to parse both `pk:<zbase32 encoded key>`, standalone `<zbase32 encoded key>`, and preferably any 52-character TLD in any valid URI.


## SignedPacket

### Encoding

The canonical serialization for a Signed Pkarr packet is as follows:

```abnf
SignedPacket = public-key signature timestamp dns-packet

public-key  = 32 OCTET ; ed25519 public key
signature   = 64 OCTET ; ed25519 signature over the timestamp and encoded DNS packet
timestamp   =  8 OCTET ; big-endian UNIX timestamp in microseconds
dns-packet  =  * OCTET ; compressed encoded DNS answer packet, less than 1000 bytes
```

### DNS packet

All resource records in the packet should be relative to the TLD (public key) that signs the packet, for example:
```
foo 300 A 104.21.59.30
```
should be converted to
```
foo.o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy 300 A 104.21.59.30
```

Because the TLD in Pkarr is so long, packets should be [compressed](https://datatracker.ietf.org/doc/html/rfc1035#section-4.1.4) when encoded. However, implementations should be able to parse uncompressed packets.

### Signing

Signing follows the [bep_0044](https://www.bittorrent.org/beps/bep_0044.html) specification for signing Mutable Items, enabling SignedPackets to be published on the Mainline DHT.

The signable timestamp and dns packet is bencoded as follows:

```abnf
signable          = prefix dns-packet

prefix            = "3:seqi" timestamp "e1:v" dns-packet-length ":"
dns-packet        = * OCTET ; compressed encoded DNS answer packet, less than 1000 bytes

timestamp         = 1*DIGIT ; Integer representing the timestamp
dns-packet-length = 1*DIGIT ; Integer representing the length of the encoded DNS packet
```

### Verification

Implementations should verify the following upon receiving a candidate signed packet for a public key:

1. Timestamp is more recent than what they already have in cache
2. Signature over the signable bencoded timestamp and DNS packet is valid for the public key
3. DNS packet can be parsed correctly

## Publishing and resolving

Signed packets are published and resolved through the [Mainline DHT](https://www.bittorrent.org/beps/bep_0005.html), using the extension to store [mutable arbitrary data](https://www.bittorrent.org/beps/bep_0044.html).

To publish a signed packet, it should be converted to DHT `put message` arguments:

| DHT message field | Signed packet field | 
| ----------------- | ------------------- |
| k                 | public_key bytes    |
| seq               | timestamp           |
| sig               | signature           |
| v                 | encoded dns packet  |

The `cas`, and `salt` fields are ignored.

To look up a `public_key`, send a DHT `get message` with the `sha1` hash of the public key (ignoring `salt`), and optionally using the `timestamp` of the most recent known signed packet for that public key as a `seq` argument:

| DHT message argument | Signed packet field |
| -------------------- | ------------------- |
| target               | sha1(public_key)    |
| seq                  | timestamp           |
