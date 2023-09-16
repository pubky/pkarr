import b4a from 'b4a'
import sodium from 'sodium-universal'
import bencode from 'bencode'
import z32 from 'z32'
import _codec from './codec.js'

export const verify = sodium.crypto_sign_verify_detached

export const codec = _codec

export function randomBytes (n = 32) {
  const buf = Buffer.alloc(n)
  sodium.randombytes_buf(buf)
  return buf
}

/**
 * Endoced records, sign it and create a put request
 * @param {{publicKey: Uint8Array, secretKey: Uint8Array}} keyPair
 * @param {object} records
 */
export const createPutRequest = async (keyPair, records) => {
  const seq = Math.ceil(Date.now() / 1000)
  const v = await codec.encode(records)
  const msg = encodeSigData({ seq, v })
  const sig = sign(msg, keyPair.secretKey)

  return { seq, v, sig, msg }
}

// Copied from bittorrent-dht
/**
 * @param {{seq: number, v: Uint8Array}} msg
 */
const encodeSigData = (msg) => {
  const ref = { seq: msg.seq || 0, v: msg.v }
  const bencoded = bencode.encode(ref).subarray(1, -1)
  return bencoded
}

/**
 * @param {Uint8Array} sigData
 *
 * @returns {{seq:number, v: Uint8Array}}
 */
export const decodeSigData = (sigData) => {
  const dict = new Uint8Array(sigData.length + 2)
  dict[0] = 100 // d
  dict.set(sigData, 1)
  dict[sigData.length + 1] = 101 // e

  return bencode.decode(dict)
}

/**
 * Sign a message with an secret key
 * @param {Uint8Array} message
 * @param {Uint8Array} secretKey
 */
const sign = (message, secretKey) => {
  const signature = b4a.alloc(sodium.crypto_sign_BYTES)
  sodium.crypto_sign_detached(signature, message, secretKey)
  return signature
}

/**
 * Generate a keypair
 * @param {Uint8Array} [seed]
 *
 * @returns {KeyPair}
 */
export const generateKeyPair = (seed) => {
  const publicKey = b4a.allocUnsafe(sodium.crypto_sign_PUBLICKEYBYTES)
  const secretKey = b4a.allocUnsafe(sodium.crypto_sign_SECRETKEYBYTES)

  if (seed) sodium.crypto_sign_seed_keypair(publicKey, secretKey, seed)
  else sodium.crypto_sign_keypair(publicKey, secretKey)

  return {
    publicKey,
    secretKey
  }
}

/**
 * @param {string | Uint8Array} url
 *
 * @returns {Uint8Array}
 */
export const decodeKey = (url) => {
  if (typeof url !== 'string') return url

  const keyBytes = z32.decode(url.replace('pk:', ''))

  if (keyBytes.byteLength !== 32) {
    throw new Error('Invalid key')
  }

  return keyBytes
}

/**
 * @typedef {{secretKey: Uint8Array, publicKey: Uint8Array}} KeyPair
 */
