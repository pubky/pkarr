import _DHT from 'bittorrent-dht'
import sodium from 'sodium-universal'
import crypto from 'crypto'
import bencode from 'bencode'

const verify = sodium.crypto_sign_verify_detached

const DEFAULT_BOOTSTRAP = [
  { host: 'router.bittorrent.com', port: 6881 },
  { host: 'router.utorrent.com', port: 6881 },
  { host: 'dht.transmissionbt.com', port: 6881 },
  // Running a reliable DHT node that responds to requests from behind NAT? please open an issue.
  { host: 'router.nuh.dev', port: 6881 }
]

export class DHT {
  constructor (opts = {}) {
    opts.bootstrap = opts.bootstrap || DEFAULT_BOOTSTRAP

    this._dht = new _DHT(opts)

    this.opening = this._open()
  }

  _open () {
    return new Promise(resolve => this._dht.once('ready', resolve))
  }

  async ready () {
    return this.opening
  }

  /**
   * Reguest a mutable value from the DHT.
   *
   * @param {Uint8Array} key
   * @returns {Promise<{
   *  id: Uint8Array,
   *  k: Uint8Array,
   *  seq: number,
   *  v: Uint8Array,
   *  sig: Uint8Array,
   *  nodes?: Array<{ host: string, port: number }>
   * }>}
   */
  async get (key) {
    await this.ready()

    const target = hash(key)
    const targetHex = target.toString('hex')

    const _dht = this._dht

    let value = _dht._values.get(targetHex) || null
    const nodes = []

    return new Promise((resolve, reject) => {
      if (value) {
        // If value was directly stored in this node in put request
        value = createGetResponse(_dht._rpc.id, value)
        return process.nextTick(done)
      }

      _dht._closest(
        target,
        {
          q: 'get',
          a: {
            id: _dht._rpc.id,
            target
          }
        },
        onreply,
        done
      )

      function done (err) {
        if (err) reject(err)
        else resolve(value && { ...value, nodes })
      }

      function onreply (message, from) {
        const r = message.r
        if (!r.sig || !r.k) return true
        if (!verify(r.sig, encodeSigData(r), r.k)) return true
        if (hash(r.k).equals(target)) {
          if (!value || r.seq >= value.seq) {
            nodes.push({ host: from.host || from.address, port: from.port })
            value = r
          }
        }
        return true
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
   *  hash: Uint8Array,
   *  nodes: Array<{ id: Uint8Array, host: string, port: number }>
   * }>}
   */
  async put (key, request) {
    validate(key, request)
    const target = hash(key)

    await this.ready()
    const _dht = this._dht

    let closestNodes = _dht._tables.get(target.toString('hex'))?.closest(target)

    if (!closestNodes) {
      await new Promise((resolve, reject) => {
        _dht._closest(
          target,
          {
            q: 'get',
            a: {
              id: _dht._rpc.id,
              target
            }
          },
          null,
          (err, n) => {
            if (err) reject(err)
            else resolve(n)
          }
        )
      })

      closestNodes = _dht._tables.get(target.toString('hex'))?.closest(target)
    }

    const message = {
      q: 'put',
      a: {
        id: _dht._rpc.id,
        token: null, // queryAll sets this
        v: request.v,
        k: key,
        seq: request.seq,
        sig: request.sig
      }
    }

    return new Promise((resolve, reject) => {
      _dht._rpc.queryAll(
        closestNodes,
        message,
        null,
        (err, n) => {
          if (err) reject(err)
          else resolve({ target, nodes: closestNodes })
        }
      )
    })
  }

  destroy () {
    try {
      this._dht.destroy()
    } catch { }
  }
}

export default DHT

function validate (key, request) {
  if (request.v === undefined) {
    throw new Error('request.v not given')
  }
  if (request.v.length >= 1000) {
    throw new Error('v must be less than 1000 bytes in put()')
  }
  if (key.length !== 32) {
    throw new Error('key ed25519 public key must be 32 bytes')
  }
  if (!Buffer.isBuffer(request.sig)) {
    throw new Error('request.sig signature is required for mutable put')
  }
  if (request.seq === undefined) {
    throw new Error('request.seq not provided for a mutable update')
  }
  if (typeof request.seq !== 'number') {
    throw new Error('request.seq not an integer')
  }
}

function hash (buf) {
  return crypto.createHash('sha1').update(buf).digest()
}

function createGetResponse (id, value, nodes) {
  return {
    id,
    v: value.v,
    sig: value.sig,
    k: value.k,
    seq: value.seq,
    nodes
  }
}

function encodeSigData (msg) {
  const ref = { seq: msg.seq || 0, v: msg.v }
  if (msg.salt) ref.salt = msg.salt
  return bencode.encode(ref).slice(1, -1)
}
