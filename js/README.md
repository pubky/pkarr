# Pkarr

JavaScript implementation of [Pkarr](https://github.com/nuhvi/pkarr).

## Usage

### Server

Install and run a server on a publicly addressable machine

```
npm install -g pkarr
```

By default it will run on port `7527` but you can override it using environment variables

```
$ PORT=3000 pkarr
> [21:45:58.826] INFO (9863): Server listening at http://0.0.0.0:3000
```

Consider adding your server to the [list of free and public servers](../servers.txt)

### CLI 

after installing you can init a new keypair, add records, or resolve records.

```
pkarr run

pkarr resolve <z-base32 key>
```

### Client 

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
  "querystring": {
    "type": "object",
    "properties": {
      "after": { "type": "number" }
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
