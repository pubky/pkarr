import z32 from 'z32'
import b4a from 'b4a'

import { createPutRequest, generateKeyPair, verify, codec, decodeSigData, randomBytes } from '../tools.js'

export class Pkarr {
  static generateKeyPair = generateKeyPair
  static generateSeed = randomBytes
  static codec = codec

  /**
   * @param {import('../tools.js').KeyPair} keyPair
   * @param {any} records
   * @param {string[]} relays
   *
   * @returns {Promise<boolean>}
   */
  static async publish (keyPair, records, relays) {
    const request = await createPutRequest(keyPair, records)

    const body = b4a.alloc(request.v.length + 72)
    body.set(request.sig)
    body.writeBigInt64BE(BigInt(request.seq), 64)
    body.set(request.v, 72)

    return new Promise(resolve => {
      let count = 0

      const notOk = () => {
        count += 1
        if (count >= relays.length) {
          resolve(false)
        }
      }

      relays.forEach((relay) =>
        fetch(
          relay.replace(/\/+$/, '') + '/' + z32.encode(keyPair.publicKey),
          { method: 'PUT', body }
        )
          .then(async (response) => {
            if (!response.ok) return notOk()
            resolve(true)
          })
          .catch(notOk)
      )
    })
  }

  /**
   * @param {Uint8Array} key
   * @param {string[]} relays
   *
   * @returns {Promise<{seq:number, records: any[]} | null>}
   */
  static async resolve (key, relays) {
    return new Promise(resolve => {
      let count = 0

      const notOk = () => {
        count += 1
        if (count >= relays.length) {
          resolve(null)
        }
      }

      relays.forEach((relay) =>
        fetch(
          relay.replace(/\/+$/, '') + '/' + z32.encode(key),
          { method: 'GET' }
        )
          .then(async (response) => {
            if (!response.ok) return notOk()

            /** @type {Uint8Array} */
            const body = b4a.from(await response.arrayBuffer())

            const sig = body.subarray(0, 64)
            const msg = body.subarray(64)

            const valid = verify(sig, msg, key)
            if (!valid) return notOk()

            const { seq, v } = decodeSigData(msg)

            resolve({
              seq,
              records: await codec.decode(v)
            })
          })
          .catch(notOk)
      )
    })
  }
}

export default Pkarr
