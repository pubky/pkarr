import test from 'brittle'
import fetch from 'node-fetch'
global.fetch = fetch

import pkarr from '../index.js'

const SERVERS = [
  // Add servers to test against
]

if (SERVERS.length === 0) throw new Error('Need some publicly addressable servers to test against')

test('successful put - get', async function(t) {
  const records = [
    ['_matrix', '@foobar:example.com'],
    ['A', 'nuhvi.com'],
    ["_lud16.alice", "https://my.ln-node/.well-known/lnurlp/alice"],
    ["_btc.bob", "https://my.ln-node/.well-known/lnurlp/bob"]
  ]

  const keyPair = pkarr.generateKeyPair()
  const result = await pkarr.put(keyPair, records, SERVERS)

  const expected = 'ARupAACM1GEtMTc7XBfTpGtt0BP9BRTk1g0OmAsw05wDzDxdBg9ZUPVvTCjKaTJg9tfRP+agsq7BHzA1y8U8MZiD3+3punEw3Ylp7KcesxTw43SafzwtaFlTJjipkWm5j2kjxBN5siBcEYc1SEbDyhk='

  t.ok(result.ok)
  t.alike(result.request?.v, expected)

  const resolved = await pkarr.get(keyPair.publicKey, SERVERS)
  t.ok(resolved.ok)
  t.ok(resolved.seq)
  t.alike(resolved.records, records)
})
