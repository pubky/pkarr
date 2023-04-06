import _DHT from 'bittorrent-dht'
import sodium from 'sodium-universal'

export class DHT {
  constructor() {
    this._dht = new _DHT({ verify: sodium.crypto_sign_verify_detached })
  }

  /**
   * @param {Uint8Array} key
   * @returns {Promise<{
   *  id: Uint8Array,
   *  k: Uint8Array,
   *  seq: number,
   *  v: Uint8Array,
   *  sig: Uint8Array,
   * }>}
   */
  async get(key) {
    const hash = this._dht._hash(Buffer.from(key, 'hex'))

    return new Promise((resolve, reject) => {
      try {
        this._dht.get(hash, (err, response) => {
          if (err) reject(err)
          else resolve(response)
        })
      } catch (error) {
        reject(error)
      }
    })
  }

  /**
   * @param  {Uint8Array} key
   * @param  {object} request
   * @param  {Uint8Array} request.v
   * @param  {Uint8Array} request.sig
   * @param  {number} request.seq
   *
   * @returns {Promise<{
   *  hash: Uint8Array
   * }>}
   */
  async put(key, request) {
    return new Promise((resolve, reject) => {
      const opts = {
        k: key,
        ...request,
      }
      this._dht.put(opts, (err, hash) => {
        if (err) reject(err)
        else resolve({ hash })
      })
    })
  }

  destroy() {
    return this._dht.destroy()
  }
}

export default DHT
