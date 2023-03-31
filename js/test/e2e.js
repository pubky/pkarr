import test from 'brittle'
import fetch from 'node-fetch'

import pkarr from '../index.js'
import codec from '../lib/codec.js'
global.fetch = fetch

test('successful put - get', async function (t) {
  const records = [
    ['_matrix', '@foobar:example.com'],
    ['A', 'nuhvi.com'],
    ['_lud16.alice', 'https://my.ln-node/.well-known/lnurlp/alice'],
    ['_btc.bob', 'https://my.ln-node/.well-known/lnurlp/bob']
  ]

  const keyPair = pkarr.generateKeyPair()
  const result = await pkarr.put(keyPair, records, ['pkarr1.nuhvi.com'])

  t.ok(result.ok)
  t.alike(await codec.decode(Buffer.from(result.request?.v, 'base64')), records)

  const resolved = await pkarr.get(keyPair.publicKey, ['pkarr2.nuhvi.com'])
  t.ok(resolved.ok)
  t.ok(resolved.seq)
  t.alike(resolved.records, records)
})
