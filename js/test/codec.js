import test from 'brittle'

import codec from '../lib/codec.js'

test('encode + decode - uncompressable bencode', async function (t) {
  const uncompressable = [[3], ['3']]

  const encoded = await codec.encode(uncompressable)
  t.is(encoded[0], 0, 'version 0')
  t.is(encoded.toString('hex'), '000b05805b5b335d2c5b2233225d5d03')

  const decoded = await codec.decode(encoded)
  t.alike(decoded, uncompressable)
})

test('encode + decode - compressed bencode', async function (t) {
  const records = `
    cloudflare.com. 146 IN A 104.16.132.229
    cloudflare.com. 146 IN A 104.16.133.229
  `.split(/\n/g).filter(Boolean).map(
      row => row.split(/\s/g).filter(Boolean)
        .map(s => Number.isNaN(Number(s)) ? s : Number(s))
    )

  const encoded = await codec.encode(records)
  t.is(encoded[0], 0, 'version 0')
  t.is(
    encoded.toString('hex'),
    '001b6700f845e796faf3c704189bd16831144730d84cc2061c58e090278bc319042821b5379e63effc9b0712502389b9400c10da423445ddc4aca308b70a'
  )

  const decoded = await codec.decode(encoded)
  t.alike(decoded, records)
})
