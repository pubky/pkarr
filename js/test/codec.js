import test from 'brittle'

import codec from '../lib/codec.js'

test('encode + decode - uncompressable cbor', function(t) {
  const uncompressable = [[], []]

  const encoded = codec.encode(uncompressable)
  t.is(encoded[0], 0, '(un)compressed cbor starts with a prefix (0)')

  const decoded = codec.decode(encoded)
  t.alike(decoded, uncompressable)
})

test('encode + decode - compressed cbor', function(t) {
  const records = `
    cloudflare.com.		146 IN A 104.16.132.229
    cloudflare.com.		146 IN A 104.16.133.229
  `.split(/\n/g).filter(Boolean).map(
    row => row.split(/\s/g).filter(Boolean)
  )

  const encoded = codec.encode(records)
  t.is(encoded[0], 1, 'compressed cbor starts with a prefix (1)')

  const decoded = codec.decode(encoded)
  t.alike(decoded, records)
})
