import z32 from 'z32'
import dns from 'dns-packet'
import b4a from 'b4a'
import sodium from 'sodium-universal'

import { encodeSigData } from './tools.js'

const verify = sodium.crypto_sign_verify_detached

export default class SignedPacket {
  /** @type {Uint8Array} */
  #publicKey
  /** @type {Packet} */
  #packet
  /** @type {Uint8Array} */
  #bytes
  /** @type {number} */
  #timestamp
  /** @type {Uint8Array} */
  #signature

  /**
   * Creates a new SignedPacket from a Keypair and a DNS Packet.
   *
   * It will also normalize the names of the ResourceRecords to be relative to the origin, which would be the zbase32 encoded PublicKey of the Keypair used to sign the Packet.
   *
   * @param {Keypair} keypair
   * @param {Packet} packet
   *
   * @param {object} [options] - Optional arguments mostly useful for unit testing
   * @param {object} [options.timestamp] - timestamp in microseconds
   *
   * @returns {SignedPacket}
   */
  static fromPacket (keypair, packet, options = {}) {
    const origin = z32.encode(keypair.publicKey)

    packet.answers = packet.answers.map(answer => {
      answer.name = normalizeName(origin, answer.name)
      return answer
    })

    const signedPacket = new SignedPacket()
    signedPacket.#packet = packet

    const timestamp = Math.ceil(options.timestamp || (Date.now() * 1000)) // Micro seconds
    const encodedPacket = dns.encode(packet)

    const signable = encodeSigData({ seq: timestamp, v: encodedPacket })

    const signature = b4a.alloc(sodium.crypto_sign_BYTES)
    sodium.crypto_sign_detached(signature, signable, keypair.secretKey)

    signedPacket.#signature = signature
    signedPacket.#timestamp = timestamp
    signedPacket.#publicKey = keypair.publicKey

    const bytes = Buffer.alloc(encodedPacket.length + 72)
    bytes.set(signature)
    bytes.writeBigUInt64BE(BigInt(timestamp), 64)
    bytes.set(encodedPacket, 72)

    signedPacket.#bytes = bytes

    return signedPacket
  }

  /**
   * Creates a new [SignedPacket] from a [PublicKey] and the concatenated 64 bytes Signature,
   * 8 bytes timestamp and encoded [Packet] as defined in the [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md) spec.
   *
   * @param {Uint8Array} publicKey
   * @param {Uint8Array} bytes
   */
  static fromBytes (publicKey, bytes) {
    const buffer = Buffer.from(bytes)

    if (buffer.length < 72) {
      throw new Error(`Invalid SignedPacket bytes length, expected at least 72 bytes but got: ${buffer.length}`)
    }
    if (buffer.length > 1072) {
      throw new Error(`Encoded and compressed DNS Packet is too large, expected max 1000 bytes but got: ${buffer.length}`)
    }

    const signature = buffer.subarray(0, 64)
    const encodedPacket = buffer.subarray(72)

    /** @type {Number} */
    const timestamp = Number(buffer.readBigUInt64BE(64))

    const signable = encodeSigData({ seq: timestamp, v: encodedPacket })

    const valid = verify(signature, signable, publicKey)

    if (!valid) {
      throw Error('Invalid signature')
    }

    const packet = dns.decode(encodedPacket)

    const signedPacket = new SignedPacket()

    signedPacket.#publicKey = publicKey
    signedPacket.#signature = signature
    signedPacket.#packet = packet
    signedPacket.#timestamp = timestamp
    signedPacket.#bytes = bytes

    return signedPacket
  }

  /**
   * Return a list of resource records with the target `name`.
   * `name` can be not normalized, for example `@` or `subdomain.`.
   *
   * @param {string} name
   */
  resourceRecords (name) {
    const origin = z32.encode(this.publicKey())
    const normalizedName = normalizeName(origin, name)

    return this.packet()
      .answers
      .filter((rr) => rr.name === normalizedName)
  }

  /**
   * Return the publicKey singing this packet
   *
   * @returns{Uint8Array}
   * */
  publicKey () {
    return this.#publicKey
  }

  /**
   * Return the DNS packet
   *
   * @returns{Packet}
   * */
  packet () {
    return this.#packet
  }

  /**
   * Returns the timestamp of the creation of the signed packet
   *
   * @returns {number}
   */
  timestamp () {
    return this.#timestamp
  }

  /**
   * Returns the signature over the encoded DNS packet and timestamp as defined by BEP44
   *
   * @returns{Uint8Array}
   * */
  signature () {
    return this.#signature
  }

  /**
   * Return the encoded signature, timestamp and packet as defined in the [relays](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md) spec.
   *
   * @returns {Uint8Array}
   */
  bytes () {
    return this.#bytes
  }
}

/**
 *
 * @param {string} origin
 * @param {string} name
 *
 * @returns {string}
 */
function normalizeName (origin, name) {
  if (name.endsWith('.')) {
    name = name.slice(0, -1)
  };

  const parts = name.split('.')
  const last = parts[parts.length - 1]

  if (last === origin) {
    // Already normalized.
    return name
  } else if (last === '@' || last.length === 0) {
    // Shorthand of origin
    return origin
  }

  return name.concat('.').concat(origin)
}

/**
 * @typedef {import('dns-packet').Packet} Packet
 * @typedef {{secretKey: Uint8Array, publicKey: Uint8Array}} Keypair
 */
