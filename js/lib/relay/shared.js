import b4a from 'b4a'
import { verify, encodeSigData } from '../tools.js'

/**
 * @param {Uint8Array} key
 * @param {Uint8Array} body
 *
 * @returns {{seq:number,v:Uint8Array, sig:Uint8Array } | Error}
 */
export const verifyBody = (key, body) => {
  if (body.length < 64) {
    return new Error('Signature should be 64 bytes')
  }
  if (body.length < 72) {
    return new Error('Sequence should be 8 bytes')
  }

  const sig = body.subarray(0, 64)
  const v = body.subarray(72)

  /** @type {Number} */
  let seq

  try {
    seq = Number(b4a.from(body.subarray(64, 72)).readBigInt64BE())
  } catch (error) {
    return new Error('Invalid sequence number')
  }

  const sigData = encodeSigData({ seq, v })

  const valid = verify(sig, sigData, key)

  if (!valid) {
    return new Error('Invalid signature')
  }

  return valid && { seq, v, sig }
}

/**
 * @param {{seq:number, v:Uint8Array, sig:Uint8Array}} request
 */
export const writeBody = (request) => {
  const body = b4a.alloc(request.v.length + 72)

  body.set(request.sig)
  body.writeBigInt64BE(BigInt(request.seq), 64)
  body.set(request.v, 72)

  return body
}
