import test from 'brittle'
import dns from 'dns-packet'

import Pkarr from '../relayed.js'
import Server from '../lib/relay/server.js'
import DHT from '../lib/dht.js'

test('successful put - get', async (t) => {
  const serverA = await Server.start({ dht: new DHT({ storage: null }) })
  const serverB = await Server.start({ dht: new DHT({ storage: null }) })

  const keyPair = Pkarr.generateKeyPair()

  /** @type {import('dns-packet').Answer[]} */
  const records = [
    { name: '_matrix', type: 'TXT', class: 'IN', data: '@foobar:example.com' }
  ]

  /** @type {import('dns-packet').Packet} */
  const packet = {
    id: 0,
    type: 'response',
    flags: 0,
    answers: records
  }

  const target = dns.decode(dns.encode(packet))

  const published = await Pkarr.publish(keyPair, packet, [serverA.address])

  t.ok(published)

  const resolved = await Pkarr.resolve(keyPair.publicKey, [serverB.address])

  t.ok(resolved)
  t.ok(resolved?.seq)
  t.alike(resolved?.packet?.answers, target.answers)

  serverA.close()
  serverB.close()
})
