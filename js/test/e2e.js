import test from 'brittle'

import { Pkarr, SignedPacket, generateKeyPair } from '../index.js'
import Server from '../lib/relay/server.js'
import DHT from '../lib/dht.js'

test('successful put - get', async (t) => {
  const serverA = await Server.start({ dht: new DHT({ storage: null }) })
  const serverB = await Server.start({ dht: new DHT({ storage: null }) })

  const keyPair = generateKeyPair()

  /** @type {import('dns-packet').Packet} */
  const packet = {
    id: 0,
    type: 'response',
    flags: 0,
    answers: [
      { name: '_matrix', type: 'TXT', data: '@foobar:example.com' }
    ]
  }

  const signedPacket = SignedPacket.fromPacket(keyPair, packet)
  const response = await Pkarr.relayPut(serverA.address, signedPacket)

  t.ok(response)

  const resolved = await Pkarr.relayGet(serverB.address, keyPair.publicKey)

  t.alike(resolved.publicKey(), keyPair.publicKey)
  t.alike(resolved.signature(), signedPacket.signature())
  t.alike(resolved.timestamp(), signedPacket.timestamp())
  t.alike(resolved.packet, signedPacket.packet)

  serverA.close()
  serverB.close()
})
