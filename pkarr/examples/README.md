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
