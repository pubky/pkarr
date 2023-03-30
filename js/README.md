# Pkarr

JavaScript implementation of [Pkarr](https://github.com/nuhvi/pkarr).

## Install
```
npm install pkarr
```

## Usage

### Client 

```
import { pkarr } from 'pkarr'

const records = [
  ['_matrix', '@foobar:example.com'],
  ['A', 'nuhvi.com'],
  ["_lud16.alice", "https://my.ln-node/.well-known/lnurlp/alice"],
  ["_btc.bob", "https://my.ln-node/.well-known/lnurlp/bob"]
]

const servers = [example.com]

// Genearate a keyPair from a 32 bytes seed
const keyPair = pkarr.generateKeyPair(seed)

// Create a recrord, sign it and submit it to one or more servers
await pkarr.put(keyPair, records, servers)

// Get records or a public key
const response = pkarr.get(key, servers)
```

## HTTP API

HTTP endpoints expected from server implementation

#### PUT /pkarr/:key

Simple proxy to the relevant parts of [BEP 44](https://www.bittorrent.org/beps/bep_0044.html) mutable put request/response.

```json
{
    params: {
      type: 'object',
      required: ['key'],
      properties: {
        key: { type: 'string', pattern: '^[a-fA-F0-9]{64}$' }
      }
    },
    body: {
      type: 'object',
      required: ['seq', 'sig', 'v'],
      properties: {
        seq: { type: 'number' },
        sig: { type: 'string', pattern: '^[a-fA-F0-9]{128}$' },
        v: { type: "string", contentEncoding: "base64" }
      }
    },
    response: {
      200: {
        type: 'object',
        properties: {
          hash: { type: 'string', pattern: '^[a-fA-F0-9]{128}$' }
        }
      }
    }
}
```
