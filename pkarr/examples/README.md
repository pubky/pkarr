# Examples

## Publish 

Publish a SignedPacket for a randomly generated keypair.

```sh
cargo run --example publish [MODE] [RELAYS]
```
```
Arguments:
  [MODE]       Publish to DHT only, Relays only, or default to both [possible values: dht, relays, both]
  [RELAYS]...  List of relays (only valid if mode is 'relays')
```

## Resolve

Resolve a SignedPacket for a given public key.

```sh
cargo run --example resolve <PUBLIC_KEY> [MODE] [RELAYS]
```
```
Arguments:
  <PUBLIC_KEY>  Pkarr public key (z-base32 encoded) or a URL where the TLD is a Pkarr key
  [MODE]        Resolve from DHT only, Relays only, or default to both [possible values: dht, relays, both]
  [RELAYS]...   List of relays (only valid if mode is 'relays')
```

## HTTP

### Serve

Run an `HTTP` server listening on a `Pkarr` key

```sh
cargo run --features=tls --example http-serve <IP> <PORT>
```
```
Arguments:
  <IP>    IP address to listen on (needs to be a public IP address)
  <PORT>  Port number to listen on (needs to be an open port)
```

An `HTTPS` `URL` will be printed with the `Pkarr` key as the `TLD`. Paste it in another terminal window with the next command.

### Get

```sh
cargo run --features=reqwest-builder --example http-get <URL>
```
```
Arguments:
  <URL>  URL to GET from
```

And you should see a `Hello, World!` response.

### Decode

Checks whether a binary file is a `SignedPacket` from Pkarr. You can obtain such a file with:

```sh
curl --output packet.bin -X GET https://pkarr.pubky.app/57ipp7cyxoghrxqk1koh3n5bk5ke1hsz9oo3cn93n1r3htxcpz1o
```

Place `packet.bin` in the project root, or provide a custom path as an argument:

```sh
cargo run --example decode -- 57ipp7cyxoghrxqk1koh3n5bk5ke1hsz9oo3cn93n1r3htxcpz1o packet.bin
```
```
Arguments:
  <Public key ed25519>  <File with Pkarr signed packet>
```

```
Public key: 57ipp7cyxoghrxqk1koh3n5bk5ke1hsz9oo3cn93n1r3htxcpz1o
Reading file: packet.bin

Successfully verified and deserialized packet!

Public Key: 57ipp7cyxoghrxqk1koh3n5bk5ke1hsz9oo3cn93n1r3htxcpz1o
Timestamp: 1750175589279005 (Tue, 17 Jun 2025 15:53:09 GMT)
Last Seen: 0 seconds ago
Signature: C8AC96B30B2879B3386AEDCACB9EAAF6A26E3066C021DB3E73EABC0C27965E84B71F5FF5C2C4947AFF6403EAC8407F64C1429BA1367D6C5B47F5E680A89AEB03

DNS Records:
_foo.57ipp7cyxoghrxqk1koh3n5bk5ke1hsz9oo3cn93n1r3htxcpz1o 30 IN IN TXT "bar"

Check if TTL in between 300s min, 86400s max

Packet expires in 300 seconds
```

