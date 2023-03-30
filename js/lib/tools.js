import b4a from 'b4a'
import sodium from 'sodium-universal'
import bencode from 'bencode'
import codec from './codec.js'

export const verify = sodium.crypto_sign_verify_detached

export function randomBytes(n = 32) {
  const buf = Buffer.alloc(n)
  sodium.randombytes_buf(buf)
  return buf
}

/**
 * Send PUT request to multiple sevrers.
 * Resolves as soon as one server returns a 200 response
 *
 * @param {{publicKey: Uint8Array, secretKey: Uint8Array}} keyPair
 * @param {Array<string | number>[]} records
 * @param {string[]} servers 
 *
 * @returns {Promise<
 *   { ok: false, request: PutRequest, errors: Array<{server: string, error: { status?: string, statusCode?: number, message: string}}> }
 *   { ok: true , request: PutRequest, server: string, response: {hash: string} }
 * >}
 */
export async function put(keyPair, records, servers) {
  const req = createPutRequest(keyPair, records)
  const key = b4a.toString(keyPair.publicKey, 'hex')

  return raceToSuccess(
    servers.map((server) =>
      fetch(makeURL(server, key), {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(req)
      })
        .then(async (response) => {
          const body = await response.json()
          if (response.ok)
            return {
              ok: response.ok,
              response: body,
              request: req,
            }
          throw body
        })
        .catch(error => ({
          ok: false,
          server,
          error
        }))
    )
  )
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
/**
 * @param {{seq: number, v: Uint8Array}} msg
 */
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
 * Request records from multiple servers.
 * Returns the first answer.
 * @param {Uint8Array} key
 * @param {string[]} servers
 * @returns {Promise<
 *  {ok: true, seq: number, records: object} |
 *  {ok: false, errors: {server: string, error: { status?: string, statusCode?: number, message: string}} }
 * >}
 */
export async function get(key, servers) {
  const keyHex = b4a.toString(key, 'hex')

  return raceToSuccess(
    servers.map((server) =>
      fetch(makeURL(server, keyHex), {
        method: 'GET',
      })
        .then(async (response) => {
          const body = await response.json()

          if (response.ok) {
            const seq = body.seq
            const v = Buffer.from(body.v, 'base64')
            const sig = b4a.from(body.sig, 'hex')
            const sigData = encodeSigData({ seq, v })
            const valid = verify(sig, b4a.from(sigData), key)
            if (!valid) throw "Invalid signature"

            const records = codec.decode(v)

            return {
              ok: response.ok,
              seq: body.seq,
              records,
            }
          }
          throw body
        })
        .catch((error) => {
          throw {
            ok: false,
            server,
            error
          }
        })
    )
  )
}

function raceToSuccess(promises) {
  return new Promise((resolve) => {
    const errors = [];

    // Helper function to handle rejection
    function handleRejection(reason) {
      errors.push(reason);
      if (errors.length === promises.length) {
        resolve({
          errors,
          ok: false,
        });
      }
    }

    function handleSuccess(value) {
      resolve({
        ...value,
        ok: true,
      })
    }

    // Wrap each promise with a custom error handler and race them
    const wrappedPromises = promises.map(promise =>
      promise.then(handleSuccess, handleRejection)
    );

    Promise.race(wrappedPromises).catch(() => {
      // Catch unhandled rejections, do nothing
      resolve()
    });
  });
}

/**
 * @typedef {
 *  seq: number,
 *  v: string,
 *  sig: string
 * } PutRequest
 */
