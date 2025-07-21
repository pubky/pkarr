# Pkarr cli

## Install

```bash
cargo install pkarr-cli
```

## Commands

### Resolve

Resolve a SignedPacket for a given public key.

```sh
pkarr resolve <PUBLIC_KEY> [MODE] [RELAYS]
```
```
Arguments:
  <PUBLIC_KEY>  Pkarr public key (z-base32 encoded) or a URL where the TLD is a Pkarr key
  [MODE]        Resolve from DHT only, Relays only, or default to both [possible values: dht, relays, both]
  [RELAYS]...   List of relays (only valid if mode is 'relays')
```
