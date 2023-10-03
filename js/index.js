import DHT from './lib/dht.js'
import { createPutRequest, generateKeyPair, codec, randomBytes, decodeKey } from './lib/tools.js'

export class Pkarr {
  static generateKeyPair = generateKeyPair
  static generateSeed = randomBytes
  static codec = codec

  /**
   * @param {import('./lib/tools.js').KeyPair} keyPair
   * @param {any} records
   *
   * @returns {Promise<boolean>}
   */
  static async publish (keyPair, records) {
    const dht = new DHT()

    const request = await createPutRequest(keyPair, records)

    return dht.put(keyPair.publicKey, request)
      .then(() => true)
      .catch(() => false)
      .finally(() => dht.destroy())
  }

  /**
   * @param {Uint8Array | string} key
   * @param {object} [options]
   * @param {boolean} [options.fullLookup]
   *
   * @throws {Error<'Invalid key'>}
   */
  static async resolve (key, options = {}) {
    const dht = new DHT()
    try {
      const result = await dht.get(decodeKey(key), options)
        .finally(() => dht.destroy())

      if (!result) return null

      return {
        seq: result.seq,
        packet: await codec.decode(result.v),
        nodes: result.nodes
      }
    } catch (error) {
      dht.destroy()

      throw error
    }
  }
}

export default Pkarr
