import dgram from 'dgram'
import bencode from 'bencode'
import randombytes from 'randombytes'

const bootstrap = [
  { host: 'router.bittorrent.com', port: 6881 },
  { host: 'router.utorrent.com', port: 6881 },
  { host: 'dht.transmissionbt.com', port: 6881 },
  { host: '87.98.162.88', port: 6881 }
]

class Client {
  constructor() {
    this._id = randombytes(20)

    this.socket = dgram.createSocket('udp4')
    this.socket.on('message', this._onmessage.bind(this))

    this.opened = this._open()
  }

  ready() {
    return this.opened
  }

  /**
   * Bind the client to a port
   *
   * @param {number} port
   */
  _bind(port = 6881) {
    return new Promise((resolve) => {
      this.socket.on('listening', onlistening)
      this.socket.bind(port)

      function onlistening(...args) {
        console.log("listening", ...args, this.address())
        resolve()
      }
    })
  }

  _open() {
    console.log("Opening!")
    return this._bind()
  }

  _onmessage(buf, peerInfo) {
    console.log("GOT MESSAGE", { buf, peerInfo })
  }

  /**
   * @param {Peer} peer
   * @param {Query} query
   */
  async query(peer, query) {
    // TODO: Handle DNS host
    // if (!this.isIP(peer.host)) return this._resolveAndQuery(peer, query, cb)

    const message = {
      t: Buffer.allocUnsafe(2),
      y: 'q',
      q: query.q,
      a: query.a
    }

    // TODO: deal with this fancy stuff later!
    // const req = {
    //   ttl: 4,
    //   peer: peer,
    //   message: message,
    // }
    //
    // if (this._tick === 65535) this._tick = 0
    // var tid = ++this._tick

    // var free = this._ids.indexOf(0)
    // if (free === -1) free = this._ids.push(0) - 1
    // this._ids[free] = tid
    // while (this._reqs.length < free) this._reqs.push(null)
    // this._reqs[free] = req
    //
    // this.inflight++
    // message.t.writeUInt16BE(tid, 0)
    this.send(peer, message)
    // return tid
  }

  /**
   * @param {Peer} peer
   * @param {Message} message
   */
  send(peer, message) {
    const buf = bencode.encode(message)

    return new Promise((resolve, reject) => {
      this.socket.send(
        buf,
        0,
        buf.length,
        peer.port,
        peer.host,
        /** 
         * @param {Error} error
         * @param {number} bytes
         */
        (error, bytes) => {
          if (error) {
            reject(error)
          } else {
            resolve(bytes)
          }
        }
      )
    })
  }

  /**
   * @param {Peer} peer
   */
  ping(peer) {
    return this.query(
      peer,
      {
        q: 'ping',
        a: { id: this._id }
      }
    )
  }

  destroy() {
    this.socket.close()
  }
}

const client = new Client()

client.ping(bootstrap[bootstrap.length - 1])
// client.ping({ host: '212.129.33.59', port: 6881 })
// client.ping({ host: '87.98.162.88', port: 6881 })
// client.ping({ host: '67.215.246.10', port: 6881 })
// client.ping({ host: '82.221.103.244', port: 6881 })

// client.destroy()

/**
 * @typedef {{host:string, port: number}} Peer
 * @typedef {{
 *  t?: Uint8Array,
 *  y: 'q', 
 *  q: 'ping',
 *  a: { id: Uint8Array }
 * }} Query
 * @typedef {Query} Message
 */
