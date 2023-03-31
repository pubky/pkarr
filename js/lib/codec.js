import * as _brotli from 'brotli-compress'
import b4a from 'b4a'

const brotli = _brotli.default || _brotli

export default {
  /**
   * @param {Uint8Array} records
   */
  async encode (records) {
    const encoded = b4a.from(JSON.stringify(records))
    const compressed = await brotli.compress(encoded)
    const version = b4a.from([0])

    return b4a.concat([version, compressed])
  },
  /**
   * @param {Uint8Array} encoded
   */
  async decode (encoded) {
    // const version = encoded[0]
    const rest = encoded.subarray(1)
    const decoded = await brotli.decompress(rest)
    const string = b4a.toString(b4a.from(decoded, 'hex'))

    try {
      return JSON.parse(string)
    } catch (error) {
      console.log('pkarr::codec::decode failed to parse decompressed value', string)
    }
  }
}
