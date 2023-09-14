import z32 from 'z32'
import b4a from 'b4a'

import { createPutRequest, generateKeyPair, verify, codec, decodeSigData } from '../tools.js'

export class Pkarr {
  /**
   * @param {object} options
   * @param {string[]} options.relays
   */
  constructor (options) {
    this._relays = options.relays
      .map(relay => !relay.startsWith('http') ? 'https://' + relay : relay)
  }

  static generateKeyPair = generateKeyPair

  /**
   * @param {import('../tools.js').KeyPair} keyPair
   * @param {any} records
   *
   * @returns {Promise<boolean>}
   */
  async publish (keyPair, records) {
    const request = await createPutRequest(keyPair, records)

    const body = b4a.concat([request.sig, request.msg])

    return new Promise(resolve => {
      let count = 0

      const notOk = () => {
        count += 1
        if (count >= this._relays.length) {
          resolve(false)
        }
      }

      this._relays.forEach((relay) =>
        fetch(
          relay + '/' + z32.encode(keyPair.publicKey),
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
   */
  async resolve (key) {
    return new Promise(resolve => {
      let count = 0

      const notOk = () => {
        count += 1
        if (count >= this._relays.length) {
          resolve(null)
        }
      }

      this._relays.forEach((relay) =>
        fetch(
          relay + '/' + z32.encode(key),
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
