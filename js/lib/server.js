import DHT from 'bittorrent-dht'
import sodium from 'sodium-universal'
import fastify from 'fastify'

export const verify = sodium.crypto_sign_verify_detached

export default class Server {
  /**
   * @param {object} [opts] 
   * @param {*} [opts.dht] optional dht instance for mocking
   */
  constructor(opts = {}) {
    const dht = opts.dht || new DHT({ verify })
    dht.listen(6881)
    this.dht = dht

    this.app = fastify({ logger: !(opts.logger === false) })
    this.app.route({
      method: 'PUT',
      url: '/pkarr/:key',
      schema: {
        params: {
          type: 'object',
          required: ['key'],
          properties: {
            key: { type: 'string', pattern: '^[a-fA-F0-9]{64}$' }
          }
        },
        body: {
          type: 'object',
          required: ['seq', 'sig', 'v'],
          properties: {
            seq: { type: 'number' },
            sig: { type: 'string', pattern: '^[a-fA-F0-9]{128}$' },
            v: { type: "string", contentEncoding: "base64" }
          }
        },
        response: {
          200: {
            type: 'object',
            properties: {
              hash: { type: 'string', pattern: '^[a-fA-F0-9]{128}$' }
            }
          }
        }
      },
      handler: async (request, reply) => {
        const key = Buffer.from(request.params.key, 'hex')
        const opts = {
          k: key,
          seq: request.body.seq,
          v: Buffer.from(request.body.v, 'base64'),
          sign: () => Buffer.from(request.body.sig, 'hex')
        }

        return new Promise((resolve, reject) => {
          this.dht.put(opts, (err, hash) => {
            if (err) reject(err)
            else resolve(hash)
          })
        })
          .then(hash => reply.code(200).send({ hash: hash.toString('hex') }))
          .catch((error) => reply.code(400).send(error))
      }
    })
  }

  static async start(opts = {}) {
    const server = new Server(opts)
    await server.listen(opts.port)
    return server
  }

  listen(port) {
    return new Promise((resolve, reject) => {
      this.app.listen({ port }, (err) => {
        if (err) reject(err)
        resolve()
      })
    })
  }

  destroy() {
    this.dht.destroy()
    this.app.server.close()
  }
}
