import test from 'brittle'

import DHT from '../lib/dht.js'
import * as pkarr from '../lib/tools.js'

test.skip('dht - put and get (skip behind NAT)', async function (t) {
  const keyPair = pkarr.generateKeyPair()
  const request = await pkarr.createPutRequest(keyPair, [['foo', 'bar']])

  const dht = new DHT()
  await dht.put(keyPair.publicKey, request)

  const response = await dht.get(keyPair.publicKey)

  t.alike(response.v, request.v)

  dht.destroy()
})
