import b4a from 'b4a'
import z32 from 'z32'
import sodium from 'sodium-universal'
import bencode from 'bencode'

export const verify = sodium.crypto_sign_verify_detached

/**
 * Generate a keypair
 * @param {Uint8Array} secretKey
 */
export function keygen (secretKey) {
  const publicKey = Buffer.alloc(sodium.crypto_sign_PUBLICKEYBYTES)
  if (secretKey == null) {
    secretKey = sodium.sodium_malloc(sodium.crypto_sign_SECRETKEYBYTES)
    sodium.crypto_sign_keypair(publicKey, secretKey)
  } else {
    sodium.crypto_sign_ed25519_sk_to_pk(publicKey, secretKey)
  }

  return { publicKey, secretKey }
};

/**
 * Encode public key as a z-base32 string
 * @param {Uint8Array} buf
 * @param {'z32' | 'hex'} encoding
 */
export function encode (buf, encoding = 'z32') {
  return encoding === 'hex'
    ? b4a.toString(buf, 'hex')
    : z32.encode(buf)
}

/**
 * Universal decoder of all the public keys and signature used in Pkarr
 * @param {Uint8Array} string
 * @param {boolean} arbitrary - if true, the string is assumed to be a hex encoded arbitrary data
 */
export function decode (string, arbitrary = false) {
  try {
    if (!string) return;
    if (arbitrary) return b4a.from(string, 'hex');
      
    switch (string.length) {
      // z-base32 encoded public key
      case 52:
        return z32.decode(string)

      // hex encoded public key
      case 64:
        return b4a.from(string, 'hex')
        
      default:
        // hex encoded public-key / signature / arbitrary data
        return b4a.from(string, 'hex')
    }
  } catch (error) {
    console.log("Error decoding string", string)
    throw error
  }
}

/**
 * Sign and create a put request
 */
export function putRequest (keyPair, records) {
  const msg = {
    seq: Math.ceil(Date.now()/1000),
    v: b4a.from(JSON.stringify(records))
  }
  const signature = _sign(encodeSigData(msg), keyPair.secretKey)
  return {
    ...msg,
    v: b4a.toString(msg.v, 'hex'),
    sig: b4a.toString(signature, 'hex'),
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
  const signature = Buffer.alloc(sodium.crypto_sign_BYTES)
  sodium.crypto_sign_detached(signature, message, secretKey)
  return signature
};

export async function put (keyPair, record, servers ) {
  const body = putRequest(keyPair, record)
  return fetch(`${servers[0]}/pkarr/${b4a.toString(keyPair.publicKey, 'hex')}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  })
}
