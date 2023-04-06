import fastify from 'fastify'
import fastifyCors from '@fastify/cors'
import pinoPretty from 'pino-pretty'
import pino from 'pino'

const logger =
  process.env.NODE_ENV === 'production'
    ? pino()
    : pino(pinoPretty({
      colorize: true,
      minimumLevel: 'info',
      colorizeObjects: true
    }))

/**
 * Pkarr web server
 */
export default class Server {
  /** 
   * @param {import('./dht.js').default} dht
   */
  constructor(dht) {
    /** @type {import('./dht.js').default} */
    this.dht = dht

    this.app = fastify({ logger })
    // Register the fastify-cors plugin
    this.app.register(fastifyCors, {
      // Set your CORS options here
      origin: '*' // Allow any origin to access your API (you can also specify specific domains)
      // Uncomment and configure the options below if needed
      // methods: ['GET', 'POST', 'PUT', 'DELETE'], // Allowed HTTP methods
      // allowedHeaders: ['Content-Type', 'Authorization'], // Allowed headers
      // credentials: true, // Allow cookies to be sent along with the request
    })

    this.listen = this.app.listen.bind(this.app)

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
            v: { type: 'string', contentEncoding: 'base64' }
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

        const req = {
          seq: request.body.seq,
          v: Buffer.from(request.body.v, 'base64'),
          sig: Buffer.from(request.body.sig, 'hex'),
        }

        return this.dht.put(key, req)
          .then(hash => reply.code(200).send({ hash: hash.toString('hex') }))
      }
    })

    this.app.route({
      method: 'GET',
      url: '/pkarr/:key',
      schema: {
        params: {
          type: 'object',
          required: ['key'],
          properties: {
            key: { type: 'string', pattern: '^[a-fA-F0-9]{64}$' }
          }
        },
        response: {
          200: {
            type: 'object',
            properties: {
              seq: { type: 'number' },
              sig: { type: 'string', pattern: '^[a-fA-F0-9]{128}$' },
              v: { type: 'string', contentEncoding: 'base64' }
            },
            required: ['seq', 'sig', 'v']
          }
        }
      },
      handler: async (request, reply) => {
        const response = await this.dht.get(request.params.key)

        if (!response) reply.code(404).send(null)
        else {
          reply.code(200).send({
            seq: response.seq,
            v: response.v.toString('base64'),
            sig: response.sig.toString('hex')
          })
        }
      }
    })
  }

  static async start(opts = {}) {
    const server = new Server(opts.dht)
    await server.listen({ host: '0.0.0.0', ...opts })
    return server
  }

  destroy() {
    this.dht.destroy()
    this.app.server.close()
  }
}
