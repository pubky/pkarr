import DHT from './lib/dht.js'
import _z32 from 'z32'
import _dns from 'dns-packet'

import { generateKeyPair as _generateKeyPair, randomBytes, decodeKey } from './lib/tools.js'
import _SignedPacket from './lib/signed_packet.js'

export const SignedPacket = _SignedPacket
export const z32 = _z32
export const dns = _dns
export const generateKeyPair = _generateKeyPair

export class Pkarr {
  static generateKeyPair = _generateKeyPair
  static generateSeed = randomBytes

  /**
   * Publishes a signed packet using relay. returns the Fetch response.
   *
   * @param {string} relay - Relay url
   * @param {SignedPacket} signedPacket
   */
  static async relayPut (relay, signedPacket) {
    const id = z32.encode(signedPacket.publicKey())
    const url = relay.replace(/\/+$/, '') + '/' + id

    return fetch(
      url,
      { method: 'PUT', body: signedPacket.bytes() }
    )
  }

  /**
   * Publishes a signed packet using relay. returns the Fetch response.
   *
   * @param {string} relay - Relay url
   * @param {Uint8Array} publicKey
   */
  static async relayGet (relay, publicKey) {
    const id = z32.encode(publicKey)
    const url = relay.replace(/\/+$/, '') + '/' + id

    const response = await fetch(url)
    const bytes = Buffer.from(await response.arrayBuffer())

    return SignedPacket.fromBytes(publicKey, bytes)
  }

  /**
   * Publishes a signed packet to the DHT.
   * Throws an error in browser environment.
   *
   * @param {SignedPacket} signedPacket
   *
   * @returns {Promise<boolean>}
   */
  static async publish (signedPacket) {
    const dht = new DHT()

    return dht.put(signedPacket.bep44Args())
      .then(() => true)
      .catch(() => false)
      .finally(() => dht.destroy())
  }

  /**
   * Resolves a signed packet from the DHT.
   * Throws an error in browser environment.
   *
   * @param {Uint8Array | string} key
   * @param {object} [options]
   * @param {boolean} [options.fullLookup=false] - perform a full lookup through the DHT, defaults to false, meaning it will return the first result it finds
   *
   * @throws {Error<'Invalid key'>}
   * @returns {Promise<{signedPacket: SignedPacket, nodes: {host: string, port: number}[]} | null>}
   */
  static async resolve (key, options = {}) {
    const dht = new DHT()
    try {
      const result = await dht.get(decodeKey(key), options)
        .finally(() => dht.destroy())

      if (!result) return null

      return {
        signedPacket: SignedPacket.fromBep44Args(result),
        nodes: result.nodes
      }
    } catch (error) {
      dht.destroy()

      throw error
    }
  }
}

export default Pkarr

/**
 * @typedef {import('./lib/signed_packet.js').Packet} Packet
 * @typedef {import('./lib/signed_packet.js').default} SignedPacket
 */
