import test from 'brittle'

import Pkarr from '../relayed.js'
import Server from '../lib/relay/server.js'
import DHT from '../lib/dht.js'

test('successful put - get', async (t) => {
  const serverA = await Server.start({ dht: new DHT({ storage: null }) })
  const serverB = await Server.start({ dht: new DHT({ storage: null }) })

  const keyPair = Pkarr.generateKeyPair()
  const records = [
    ['_matrix', '@foobar:example.com'],
    ['A', 'nuhvi.com'],
    ['_lud16.alice', 'https://my.ln-node/.well-known/lnurlp/alice'],
    ['_btc.bob', 'https://my.ln-node/.well-known/lnurlp/bob']
  ]

  const published = await Pkarr.publish(keyPair, records, [serverA.address])

  t.ok(published)

  const resolved = await Pkarr.resolve(keyPair.publicKey, [serverB.address])

  t.ok(resolved)
  t.ok(resolved?.seq)
  t.alike(resolved?.records, records)

  {
    const updated = await Pkarr.publish(keyPair, records.slice(0, 2), [serverA.address])
    t.ok(updated)

    const resolved = await Pkarr.resolve(keyPair.publicKey, [serverB.address])

    t.ok(resolved)
    t.ok(resolved?.seq)
    t.alike(resolved?.records, records.slice(0, 2))
  }

  serverA.close()
  serverB.close()
})
