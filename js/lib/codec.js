import bencode from 'bencode'
import brotli from 'brotli'

export default {
  encode(json) {
    const encoded = bencode.encode(json)
    const compressed = brotli.compress(encoded)
    const version = Buffer.alloc(1).fill(compressed ? 1 : 0);
    return Buffer.concat([version, compressed || encoded])
  },
  decode(encoded) {
    const version = encoded[0]
    const rest = encoded.subarray(1)

    return bencode.decode(
      version === 1 ? brotli.decompress(rest) : rest,
      'utf-8'
    )
  }
}
