import test from 'brittle'

import DHT from '../lib/dht.js'
import * as pkarr from '../lib/tools.js'

test.solo('what', async function(t) {
  const keyPair = pkarr.generateKeyPair()
  const request = await pkarr.createPutRequest(keyPair, [['foo', 'bar']])

  console.log(request)

  const dht = new DHT()

  const result = await dht.put(keyPair.publicKey, request)
  console.log(result)

  dht.destroy()
})
