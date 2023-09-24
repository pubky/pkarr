import { verify, encodeSigData } from '../tools.js'

/**
 * @param {Uint8Array} key
 * @param {Uint8Array} body
 *
 * @returns {{seq:number,v:Uint8Array, sig:Uint8Array } | Error}
 */
export const verifyBody = (key, body) => {
  // Just use Buffer because b4a doesn't support readUInt64BE
  const buffer = Buffer.from(body)

  if (buffer.length < 64) {
    return new Error('Signature should be 64 bytes')
  }
  if (buffer.length < 72) {
    return new Error('Sequence should be 8 bytes')
  }

  const sig = buffer.subarray(0, 64)
  const v = buffer.subarray(72)

  /** @type {Number} */
  let seq

  try {
    seq = Number(buffer.readBigUInt64BE(64))
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
  const body = Buffer.alloc(request.v.length + 72)

  body.set(request.sig)
  body.writeBigUInt64BE(BigInt(request.seq), 64)
  body.set(request.v, 72)

  return body
}
