import b4a from 'b4a'
import sodium from 'sodium-universal'
import bencode from 'bencode'
import codec from './codec.js'

export function randomBytes(n = 32) {
  const buf = Buffer.alloc(n)
  sodium.randombytes_buf(buf)
  return buf
}

/**
 * Returns seq as timestamp in seconds
 * @returns {Promise<
 *   { ok: false, request: PutRequest, errors: Array<{server: string, message: string}>} |
 *   { ok: true , request: PutRequest, server: string, response: Response }
 * >}
 */
export async function put(keyPair, records, servers) {
  const req = createPutRequest(keyPair, records)
  const key = b4a.toString(keyPair.publicKey, 'hex')

  const promises = servers.map(
    server => fetch(makeURL(server, key).slice(0,), {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(req)
    })
  )

  return new Promise((resolve, reject) => {
    promises.forEach((promise, index) => {
      promise.then((response) => {
        if (response.ok) resolve({
          ok: true,
          response,
          server: servers[index],
          request: req,
        })
      })
        .catch((error) => error)
    })

    return Promise.allSettled(promises)
      // Assume all failed at this point
      .then(responses => {
        resolve({
          ok: false,
          errors: responses.map((response, index) => {
            return {
              server: servers[index],
              response: response.reason
            }
          })
        })
      })
  })
}

/**
 * Sign and create a put request
 * @returns {PutRequest}
 */
export function createPutRequest(keyPair, records) {
  const msg = {
    seq: Math.ceil(Date.now() / 1000),
    v: codec.encode(records)
  }
  const signature = _sign(encodeSigData(msg), keyPair.secretKey)
  return {
    ...msg,
    sig: b4a.toString(signature, 'hex'),
    v: b4a.toString(msg.v, 'base64')
  }
}

// Copied from bittorrent-dht
function encodeSigData(msg) {
  const ref = { seq: msg.seq || 0, v: msg.v }
  if (msg.salt) ref.salt = msg.salt
  const bencoded = bencode.encode(ref).subarray(1, -1)
  return bencoded
}

/**
 * Sign a message with an secret key
 * @param {Uint8Array} message
 * @param {Uint8Array} secretKey
 */
function _sign(message, secretKey) {
  const signature = b4a.alloc(sodium.crypto_sign_BYTES)
  sodium.crypto_sign_detached(signature, message, secretKey)
  return signature
};

function makeURL(server, key) {
  if (!server.startsWith('http')) server = 'https://' + server
  return `${server}/pkarr/${key}`
}

/**
 * Generate a keypair
 * @param {Uint8Array} secretKey
 */
export function generateKeyPair(seed) {
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
 * @typedef {
 *  seq: number,
 *  v: string,
 *  sig: string
 * } PutRequest
 */
