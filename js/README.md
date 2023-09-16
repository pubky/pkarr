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

Run an HTTP relay to use Pkarr in UDP-less environments.

```bash
pkarr relay
```
Consider adding your relay to the [list of free and public relays](../relays.txt)

### Client 

In browsers and devices behind NAT, you can make HTTP requests to a any pkarr server.

```js
import Pkarr from 'pkarr/relayed.js'

const records = [
  ['_matrix', '@foobar:example.com'],
  ['A', 'nuhvi.com'],
  ["_lud16.alice", "https://my.ln-node/.well-known/lnurlp/alice"],
  ["_btc.bob", "https://my.ln-node/.well-known/lnurlp/bob"]
]

// Genearate a keyPair from a 32 bytes seed
const keyPair = Pkarr.generateKeyPair(seed)

// Create a recrord, sign it and submit it to one or more servers
await Pkarr.publish(keyPair, records, ["relay.pkarr.org"])

// Get records of a public key from another server
const {seq, records} = Pkarr.resolve(key, ["relay.pkarr.org"])
```
