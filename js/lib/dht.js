import _DHT from 'bittorrent-dht'
import sodium from 'sodium-universal'
import crypto from 'crypto'
import goodbye from 'graceful-goodbye'
import fs from 'fs'
import { homedir } from 'os'
import path from 'path'

import { encodeSigData } from './tools.js'

const verify = sodium.crypto_sign_verify_detached

const DEFAULT_BOOTSTRAP = [
  'router.magnets.im:6881',
  'router.bittorrent.com:6881',
  'router.utorrent.com:6881',
  'dht.transmissionbt.com:6881',
  'router.nuh.dev:6881'
].map(addr => {
  const [host, port] = addr.split(':')
  return { host, port: Number(port) }
})

const DEFAULT_STORAGE_LOCATION = path.join(homedir(), '.config', 'pkarr')

export class DHT {
  /**
   * @param {object} [options]
   * @param {{host:string, port:number}[]} [options.bootstrap] - List of bootstrap nodes. example [{host: "router.utorrent.com", port: 6881}]
   * @param {Storage} [options.storage]
   * @param {string} [options.storageLocation] - location to store bootstrap nodes at
   */
  constructor (options = {}) {
    const _storage = options.storage || new Storage(options.storageLocation)
    options.bootstrap = options.bootstrap || DEFAULT_BOOTSTRAP

    this._dht = new _DHT(options)

    _storage?.loadRoutingTable(this._dht)

    goodbye(() => {
      _storage?.saveRoutingTable(this._dht)
      this.destroy()
    })
  }

  /**
   * Reguest a mutable value from the DHT.
   *
   * @param {Uint8Array} key
   * @param {object} [options]
   * @param {boolean} [options.fullLookup] - If true, will perform a full lookup, otherwise return the first valid result.
   *
   * @returns {Promise<{
   *  id: Uint8Array,
   *  k: Uint8Array,
   *  seq: number,
   *  v: Uint8Array,
   *  sig: Uint8Array,
   *  nodes?: Array<{ host: string, port: number, client?: string }>
   * }>}
   */
  async get (key, options = {}) {
    const target = hash(key)
    const targetHex = target.toString('hex')

    let value = this._dht._values.get(targetHex) || null
    const nodes = []

    return new Promise((resolve, reject) => {
      if (value) {
        // If value was directly stored in this node in put request
        value = createGetResponse(this._dht._rpc.id, value)
        return process.nextTick(done)
      }

      this._dht._closest(
        target,
        {
          q: 'get',
          a: {
            id: this._dht._rpc.id,
            target
          }
        },
        onreply,
        done
      )

      /**
       * @param {Error} [err]
       */
      function done (err) {
        if (err) reject(err)
        else resolve(value && { ...value, nodes })
      }

      /**
       * @param {MutableGetResponse} message
       * @param {Node} from
       */
      function onreply (message, from) {
        const r = message.r
        if (!r.sig || !r.k) return true

        const msg = encodeSigData(r)
        if (!verify(r.sig, msg, r.k)) return true

        if (hash(r.k).equals(target)) {
          if (!value || r.seq >= value.seq) {
            nodes.push({
              host: from.host || from.address,
              port: from.port,
              client: message.v?.toString().slice(0, 2)
            })
            value = r
            if (!options.fullLookup) {
              resolve({ ...value, nodes })
            }
          }
        }
        return true
      }
    })
  }

  /**
   * @param  {object} args
   * @param  {Uint8Array} args.k
   * @param  {Uint8Array} args.v
   * @param  {Uint8Array} args.sig
   * @param  {number} args.seq
   *
   * @returns {Promise<{
   *  target: Uint8Array,
   *  nodes: Array<{ id: Uint8Array, host: string, port: number }>
   * }>}
   */
  async put (args) {
    const key = args.k
    validate(key, args)
    const target = hash(key)

    let closestNodes = this._dht._tables.get(target.toString('hex'))?.closest(target)

    if (!closestNodes) {
      await new Promise((resolve, reject) => {
        this._dht._closest(
          target,
          {
            q: 'get',
            a: {
              id: this._dht._rpc.id,
              target
            }
          },
          null,
          /**
           * @param {Error} [err]
           * @param {number} [n]
           */
          (err, n) => {
            if (err) reject(err)
            else resolve(n)
          }
        )
      })

      closestNodes = this._dht._tables.get(target.toString('hex'))?.closest(target)
    }

    const message = {
      q: 'put',
      a: {
        id: this._dht._rpc.id,
        token: null, // queryAll sets this
        v: args.v,
        k: key,
        seq: args.seq,
        sig: args.sig
      }
    }

    return new Promise((resolve, reject) => {
      this._dht._rpc.queryAll(
        closestNodes,
        message,
        null,
        /**
         * @param {Error} [err]
         * @param {number} [_n]
         */
        (err, _n) => {
          if (err) reject(err)
          else resolve({ target, nodes: closestNodes })
        }
      )
    })
  }

  destroy () {
    return new Promise((resolve, reject) => {
      try {
        this._dht.destroy(resolve)
      } catch (error) {
        reject(error)
      }
    })
  }
}

/**
 * @param {Uint8Array} key
 * @param {object} request
 * @param {number} request.seq
 * @param {Uint8Array} request.v
 * @param {Uint8Array} request.sig
 */
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

  if (!verify(request.sig, encodeSigData(request), key)) {
    throw new Error('invalid signature')
  }
}

/**
 * @param {Uint8Array} input
 */
function hash (input) {
  return crypto.createHash('sha1').update(input).digest()
}

/**
 * @param {Uint8Array} id
 * @param {{v:Uint8Array, sig:Uint8Array, k:Uint8Array, seq:number}} value
 * @param {{host:string, port:number}[]} [nodes]
 */
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

class Storage {
  /**
   * @param {string} location
   */
  constructor (location) {
    this._location = location || DEFAULT_STORAGE_LOCATION

    this._loaded = []
  }

  /**
   * @param {_DHT} dht
   */
  loadRoutingTable (dht) {
    const filepath = path.join(this._location, 'routing-table.json')

    try {
      const data = fs.readFileSync(filepath)
      const string = data.toString()
      const nodes = JSON.parse(string)

      for (const node of nodes) {
        dht.addNode(node)
      }
    } catch (error) {
      if (error.code !== 'ENOENT') throw error
    }
  }

  /**
   * @param {_DHT} dht
   */
  saveRoutingTable (dht) {
    const filePath = path.join(this._location, 'routing-table.json')

    const nodes = dht.toJSON().nodes
    const json = JSON.stringify(nodes)

    try {
      fs.writeFileSync(filePath, json)
    } catch (error) {
      if (error.code !== 'ENOENT') throw error
      fs.mkdirSync(this._location)
      fs.writeFileSync(filePath, json)
    }
  }
}

export default DHT

/**
 * @typedef {{host:string, port:number, address?: string}} Node
 * @typedef {{v: string, r: any}} GenericResponse
 * @typedef {{sig:Uint8Array, k:Uint8Array, seq:number, v: Uint8Array}} PutRequest
 * @typedef {GenericResponse & {r: PutRequest}} MutableGetResponse
 */
