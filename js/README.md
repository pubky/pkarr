# Pkarr

JavaScript implementation of [Pkarr](https://github.com/nuhvi/pkarr).

## Usage

### CLI 

Publish resource records by passing a [passphrase](https://www.useapassphrase.com/).

```bash
$ pkarr publish
◌  Enter a passphrase (learn more at https://www.useapassphrase.com/)
   Passphrase: ***********
   ❯ pk:54ftp7om3nkc619oaxwbz4mg4btzesnnu1k63pukonzt36xq144y
◌  Enter records to publish:
   ❯ foo bar
   ❯ answer 42
   ❯ Add record or press enter to submit
 ✔ Published Resource Records (8.29 seconds)
```

Resolve resource records of a public key

```bash
$ pkarr resolve pk:54ftp7om3nkc619oaxwbz4mg4btzesnnu1k63pukonzt36xq144y
 ✔ Resolved (1.51 seconds)
  ❯ foo    bar
  ❯ answer 42

  › updated_at: 4/9/2023, 8:48:21 AM
  › size      : 39/1000 bytes
  › from      : 129.159.143.235:8999
```

Keep your or your friend's records alive

```bash
$ pkarr keepalive add [url1] [url2] ...
 ✔  Successfully updated list of keys!

$ pkarr keepalive remove [url1] [url2] ...
 ✔  Successfully updated list of keys!

$ pkarr keepalive list
  ❯  yqrx81zchh6aotjj85s96gdqbmsoprxr3ks6ks6y8eccpj8b7oiy

$ pkarr keepalive
Starting republisher...
```

### Relay

Run an HTTP relay to use Pkarr in UDP-less environments. [Read more](../design/relays.md)

```bash
pkarr relay
```
Consider adding your relay to the [list of free and public relays](../relays.txt)

### Client 

In browsers and devices behind NAT, you can make HTTP requests to a any pkarr server.

```js
import { Pkarr, SignedPacket, generateKeyPair } from '../index.js'

const keypair = generateKeyPair()

{
  // Create a DNS packet
  const packet = {
    id: 0,
    type: 'response',
    flags: 0,
    answers: [
      { name: '_matrix', type: 'TXT', class: 'IN', data: '@foobar:example.com' }
    ]
  }

  // Create a Sgined Packet
  const signedPacket = SignedPacket.fromPacket(keyPair, packet)

  // Send it to a relay
  const response = await Pkarr.relayPut(serverA.address, signedPacket)
}

{
  // === Resolve a packet ===
  const signedPacket = await Pkarr.relayGet(serverB.address, keyPair.publicKey)

  // Get a dns packet corrisponding to a public key from another server
  const {seq, packet} = Pkarr.resolve(key, ["relay.pkarr.org"])
}
```

## API

### Pkarr.relayPut(relayAddress, signedPacket)

Send a `SignedPacket` to a the DHT through an HTTP [relay](../design/relays.md).

### const signedPacket = Pkarr.relayGet(relayAddress, publicKey)

Get a `SignedPacket`  from the DHT through an HTTP [relay](../design/relays.md).

If returned, `SignedPacket` is valid and verified.

### const signedPacket = SignedPacket.fromPacket(keyPair, dnsPacket)

Create a `SignedPacket` from a dns packet and a KeyPair

Notice that internally it will add the zbase32 encoded publicKey as the TLD of every answer in the packet, for example record `{name: 'foo', data: 'bar'}` will become `{name: 'foo.8pinxxgqs41n4aididenw5apqp1urfmzdztr8jt4abrkdn435ewo', data: 'bar'}`.

### const signedPacket = SignedPacket.fromBytes(publicKey, bytes)

Create a `SignedPacket` from bytes received from the HTTP [relay](../design/relays.md), after verifying the signature.

### const publicKey = signedPacket.publicKey()

Returns the publicKey

### const bytes = signedPacket.bytes()

Returns the bytes of the `signature`, encoded `timestamp` and encoded `packet` as defined in the [relay](../design/relays.md) spec.

### const signature = signedPacket.signature()

Returns the Signature of the signed packet

### const packet = signedPacket.packet()

Returns the DNS packet

### const records = signedPacket.resourceRecord(name)

Returns the resource records from the internal dns packet, that matches `name`, which can be relative to the TLD so you don't have to pass the full name, so `foo` instead of `foo.8pinxxgqs41n4aididenw5apqp1urfmzdztr8jt4abrkdn435ewo`
