import test from 'brittle'
import fetch from 'node-fetch'

import pkarr from '../index.js'
import codec from '../lib/codec.js'
import Server from '../lib/server.js'

global.fetch = fetch

test('successful put - get', async function (t) {
  class MockDHT {
    async put (key, opts) {
      this._saved = {
        k: key,
        ...opts
      }
      return {
        nodes: Array(3).fill(0)
      }
    }

    async get () {
      return {
        ...this._saved,
        nodes: Array(5).fill(0)
      }
    }

    destroy () { }
  }

  const dht = new MockDHT()

  const serverA = await Server.start({ dht, log: false })
  const serverB = await Server.start({ dht, log: false })

  const keyPair = pkarr.generateKeyPair()
  const records = [
    ['_matrix', '@foobar:example.com'],
    ['A', 'nuhvi.com'],
    ['_lud16.alice', 'https://my.ln-node/.well-known/lnurlp/alice'],
    ['_btc.bob', 'https://my.ln-node/.well-known/lnurlp/bob']
  ]
  const result = await pkarr.put(keyPair, records, [serverA.address().toString()])

  t.ok(result.ok)
  if (result.ok) {
    t.is(result.response.query.nodes, 3, 'same nodes as the mock')
    t.alike(
      await codec.decode(Buffer.from(result.response.record.v, 'base64')),
      records
    )
  }

  const resolved = await pkarr.get(keyPair.publicKey, [serverB.address().toString()])
  t.ok(resolved.ok)
  t.ok(resolved.seq)
  t.alike(resolved.records, records)

  serverA.destroy()
  serverB.destroy()
})
