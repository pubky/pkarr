import b4a from 'b4a'
import sodium from 'sodium-universal'
import bencode from 'bencode'
import codec from './codec.js'

/**
 * Returns seq as timestamp in seconds
 * @returns {Promise<
 *   { statusCode: number, error: string, message: string } |
 *   { statusCode: number, error: string, message: string, server: string }[] |
 *   { hash: string }
 * >}
 */
export async function put(keyPair, records, servers) {
  const req = createPutRequest(keyPair, records)
  const key = b4a.toString(keyPair.publicKey, 'hex')

  const promises = servers.map(
    server => fetch(makeURL(server, key), {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(req)
    })
  )

  return new Promise((resolve, reject) => {
    for (const promise of promises) {
      promise.then((response) => {
        if (response.ok) {
          response.text().then(resolve)
        }
      })
    }

    return Promise.all(promises)
      .then(responses => {
        if (responses.every(response => !response.ok)) {
          Promise.all(
            responses.map(response => response.json())
          )
            .then(responses => responses.map((r, i) => ({ ...r, server: servers[i] })))
            .then(resolve)
            .catch(reject)
        }
      })
  })
}

/**
 * Sign and create a put request
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
