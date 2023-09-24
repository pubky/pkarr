// @ts-nocheck
import * as _brotli from 'brotli-compress'
import b4a from 'b4a'

const brotli = _brotli.default || _brotli

export default {
  /**
   * @param {Array<Array<string | number>> | string} records
   */
  async encode (records) {
    const string = typeof records === 'string' ? records : JSON.stringify(records)
    const encoded = b4a.from(string)
    const compressed = await brotli.compress(encoded)
    const version = b4a.from([0])

    return b4a.concat([version, compressed])
  },
  /**
   * @param {Uint8Array} encoded
   */
  async decode (encoded) {
    try {
      const rest = encoded.subarray(1)
      const decoded = await brotli.decompress(rest)
      const string = b4a.toString(b4a.from(decoded, 'hex'))

      return JSON.parse(string)
    } catch (error) {
      // tolerate change of encoding in the coming days.
      return [['error', 'failed to parse'], ['raw', b4a.toString(encoded)]]
    }
  }
}
