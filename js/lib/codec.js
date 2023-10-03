// @ts-nocheck
import * as _brotli from 'brotli-compress'
import b4a from 'b4a'
import dns from 'dns-packet'

const brotli = _brotli.default || _brotli

export default {
  /**
   * @param {import('dns-packet').Packet} records
   */
  async encode (packet) {
    return dns.encode(packet)
  },

  /**
  * @param {Uint8Array} encoded
  * @returns {Promise<import('dns-packet').Packet}
  */
  async decode (encoded) {
    try {
      const packet = dns.decode(encoded)
      return packet
    } catch (error) {
      // Tolerate old records that aren't realy DNS packets

      const rest = encoded.subarray(1)
      const decoded = await brotli.decompress(rest)
      const string = b4a.toString(b4a.from(decoded, 'hex'))

      let records = []
      try {
        records = [
          { name: 'NOTICE', type: 'TXT', class: 'IN', ttl: 3600, data: '"This is a legacy Pkarr value, please update your packet using the Rust implementation."' },
          ...(JSON.parse(string))
            .map(r => ({
              name: r[0],
              type: 'TXT',
              class: 'IN',
              ttl: 3600,
              data: `"${r[1]}"`
            }))
        ]
      } catch (error) {

      }

      return {
        id: 0,
        flags: 0,
        type: 'response',
        questions: [],
        answers: records,
        additionals: [],
        authorities: []
      }
    }
  }
}
