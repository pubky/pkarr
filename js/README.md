# Pkarr

JavaScript implementation of [Pkarr](https://github.com/nuhvi/pkarr).

## Usage

### Server

clone this repo, navigate to js/ directory, then run start script.

```
$ git clone https://github.com/Nuhvi/pkarr.git
$ cd pkarr
```

By default it will run on port `7527` but you can override it using environment variables

```
$ PORT=3000 npm start
> [21:45:58.826] INFO (9863): Server listening at http://0.0.0.0:3000
```

Consider adding your server to the [list of free and public servers](../servers.txt)

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
 ✔ Resolved (8.51 seconds)
  ❯ foo    bar
  ❯ answer 42

  › updated_at: 4/9/2023, 8:48:21 AM
  › size      : 39/1000 bytes
  › nodes     : 5
```

### Client 

In browsers and devices behind NAT, you can make HTTP requests to a any pkarr server.

```js
import { pkarr } from 'pkarr'

const records = [
  ['_matrix', '@foobar:example.com'],
  ['A', 'nuhvi.com'],
  ["_lud16.alice", "https://my.ln-node/.well-known/lnurlp/alice"],
  ["_btc.bob", "https://my.ln-node/.well-known/lnurlp/bob"]
]

// Genearate a keyPair from a 32 bytes seed
const keyPair = pkarr.generateKeyPair(seed)

// Create a recrord, sign it and submit it to one or more servers
await pkarr.put(keyPair, records, ["pkarr1.nuhvi.com"])

// Get records of a public key from another server
const response = pkarr.get(key, ["pkarr2.nuhvi.com"])
// { 
//   ok: true, 
//   seq: 423412341, unix timestamp in seconds
//   records: [...]  same as records above same as records above same as records above same as records above
// }
```

## HTTP API

HTTP endpoints expected from server implementation

#### PUT `/pkarr/:key`

Simple proxy to the relevant parts of [BEP 44](https://www.bittorrent.org/beps/bep_0044.html) mutable put request/response.

```json
{
  "params": {
    "type": "object",
    "required": ["key"],
    "properties": {
      "key": { "type": "string", "pattern": "^[a-fA-F0-9]{64}$" }
    }
  },
  "body": {
    "type": "object",
    "required": ["seq", "sig", "v"],
    "properties": {
      "seq": { "type": "number" },
      "sig": { "type": "string", "pattern": "^[a-fA-F0-9]{128}$" },
      "v": { "type": "string", "contentEncoding": "base64" }
    }
  },
  "response": {
    "200": {
      "type": "object",
      "properties": {
        "hash": { "type": "string", "pattern": "^[a-fA-F0-9]{128}$" }
      }
    }
  }
}
```

#### GET `/pkarr:key`

```json
{
  "params": {
    "type": "object",
    "required": ["key"],
    "properties": {
      "key": { "type": "string", "pattern": "^[a-fA-F0-9]{64}$" }
    }
  },
  "response": {
    "200": {
      "type": "object",
      "properties": {
        "seq": { "type": "number" },
        "sig": { "type": "string", "pattern": "^[a-fA-F0-9]{128}$" },
        "v": { "type": "string", "contentEncoding": "base64" }
      },
      "required": ["seq", "sig", "v"]
    }
  }
}
```
