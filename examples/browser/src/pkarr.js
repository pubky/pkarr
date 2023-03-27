import b4a from 'b4a'
import z32 from 'z32'
import sodium from 'sodium-universal'
import bencode from 'bencode'

export const verify = sodium.crypto_sign_verify_detached

/**
 * Generate a keypair
 * @param {Uint8Array} secretKey
 */
export function generateKeyPair (seed) {
  const publicKey = b4a.allocUnsafe(sodium.crypto_sign_PUBLICKEYBYTES)
  const secretKey = b4a.allocUnsafe(sodium.crypto_sign_SECRETKEYBYTES)

  if (seed) sodium.crypto_sign_seed_keypair(publicKey, secretKey, seed)
  else sodium.crypto_sign_keypair(publicKey, secretKey)

  return {
    publicKey,
    secretKey
  }
}

export function randomBytes (n = 32) {
  const buf = Buffer.alloc(n)
  sodium.randombytes_buf(buf)
  return buf
}

/**
 * Sign and create a put request
 */
export function createPutRequest (keyPair, records) {
  const msg = {
    seq: Math.ceil(Date.now() / 1000),
    v: encodeValue(records)
  }
  const signature = _sign(encodeSigData(msg), keyPair.secretKey)
  return {
    ...msg,
    sig: b4a.toString(signature, 'hex'),
    v: b4a.toString(msg.v, 'hex')
  }
}

// Copied from bittorrent-dht
function encodeSigData (msg) {
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
function _sign (message, secretKey) {
  const signature = b4a.alloc(sodium.crypto_sign_BYTES)
  sodium.crypto_sign_detached(signature, message, secretKey)
  return signature
};

/**
 * Returns seq as timestamp in seconds
 * @returns {}
 * @throws {[{server: string, reason: Error}]}
 */
export async function put (keyPair, records, servers) {
  const req = createPutRequest(keyPair, records)
  const key = b4a.toString(keyPair.publicKey, 'hex')

  const promises = servers.map(server => {
    return fetch(`${server}/pkarr/${key}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(req)
    })
      .then(async res => {
        const response = await res.json()
        if (!res.ok || response.error) throw new Error(response.error)

        return { status: 'ok', seq: req.seq }
      })
  })

  return new Promise((resolve, reject) => {
    for (const promise of promises) {
      promise.then(resolve)
    }

    return Promise.allSettled(promises)
      .then(results => {
        if (results.every(r => r.status === 'rejected')) {
          reject(
            results.map((result, index) => {
              return { ...result, server: servers[index] }
            })
          )
        }
      })
  })
}

/*
 * Versioned encoding / decoding of the value field in DHT record
 */
const VERSIONS = {
  'simple-json': 0
}

/**
 * @param {object} value
 */
export function encodeValue (value) {
  const string = JSON.stringify(value)
  const buf = b4a.from(string)
  return b4a.concat([Buffer.alloc(1).fill(VERSIONS['simple-json']), buf])
}

/**
 * If the input is a string, it is assumed to be hex encoded
 * @param {Uint8Array | string} input
 */
export function decodeValue(input) {
  input = b4a.isBuffer(input) ? input : b4a.from(input, 'hex')
  const version = input[0]
  const buf = input.slice(1)

  switch (version) {
    case VERSIONS['simple-json']:
      return JSON.parse(b4a.toString(buf))
  }
}

export function encodeID (publicKey) {
  return z32.encode(publicKey)
}

export function decodeID (id) {
  return z32.decode(id)
}
