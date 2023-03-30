import test from 'brittle'

import codec from '../lib/codec.js'

test('encode + decode - uncompressable bencode', function (t) {
  const uncompressable = [[3], ['3']]

  const encoded = codec.encode(uncompressable)
  t.is(encoded[0], 0, '(un)compressed bencode starts with a prefix (0)')

  const decoded = codec.decode(encoded)
  t.alike(decoded, uncompressable)
})

test('encode + decode - compressed bencode', function (t) {
  const records = `
    cloudflare.com. 146 IN A 104.16.132.229
    cloudflare.com. 146 IN A 104.16.133.229
  `.split(/\n/g).filter(Boolean).map(
      row => row.split(/\s/g).filter(Boolean)
        .map(s => Number.isNaN(Number(s)) ? s : Number(s))
    )

  const encoded = codec.encode(records)
  t.is(encoded[0], 1, 'compressed bencode starts with a prefix (1)')

  const decoded = codec.decode(encoded)
  t.alike(decoded, records)
})
