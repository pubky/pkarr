import test from 'brittle'

import Pkarr from '../lib/relay/client.js'
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

  const publisher = new Pkarr({ relays: [serverA.address] })
  const published = await publisher.publish(keyPair, records)

  t.ok(published)

  const resolver = new Pkarr({ relays: [serverB.address] })
  const resolved = await resolver.resolve(keyPair.publicKey)

  t.ok(resolved)
  t.ok(resolved?.seq)
  t.alike(resolved?.records, records)

  serverA.close()
  serverB.close()
})
