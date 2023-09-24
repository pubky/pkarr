import z32 from 'z32'
import b4a from 'b4a'

import { createPutRequest, generateKeyPair, codec, randomBytes } from '../tools.js'

import { verifyBody, writeBody } from './shared.js'

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

    const body = writeBody(request)

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

            const result = verifyBody(key, body)

            if (result instanceof Error) return notOk()

            resolve({
              seq: result.seq,
              records: await codec.decode(result.v)
            })
          })
          .catch(notOk)
      )
    })
  }
}

export default Pkarr
