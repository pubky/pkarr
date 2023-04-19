import pkarr from 'pkarr'
import bencode from 'bencode'
import sodium from 'sodium-universal'
import fs from 'fs'

// Count of keys to generate
const COUNT = process.argv[2] || 30 * 1000
// Seed to generate a different set of keys
const SEED = process.argv[3] || '2'

const v = Buffer.from('000b0c805b5b225f74657374222c227374696c6c20616c697665225d5d03', 'hex')
const seq = 1681656800
const msg = encodeSigData({ seq, v })

const PATH = './data/users.csv'

fs.writeFileSync(PATH, '')

for (let i = 0; i < COUNT; i++) {
  const keyPair = pkarr.generateKeyPair(Buffer.alloc(32).fill(SEED + i))
  const sig = _sign(msg, keyPair.secretKey)
  const line = [keyPair.publicKey, sig].map(b => b.toString('hex')).join(',') + '\n'
  fs.appendFileSync(PATH, line)
}

// Copied from bittorrent-dht
/**
 * @param {{seq: number, v: Uint8Array}} msg
 */
export function encodeSigData(msg) {
  const ref = { seq: msg.seq || 0, v: msg.v }
  const bencoded = bencode.encode(ref).subarray(1, -1)
  return bencoded
}

/**
 * Sign a message with an secret key
 * @param {Uint8Array} message
 * @param {Uint8Array} secretKey
 */
function _sign(message, secretKey) {
  const signature = Buffer.alloc(sodium.crypto_sign_BYTES)
  sodium.crypto_sign_detached(signature, message, secretKey)
  return signature
};
