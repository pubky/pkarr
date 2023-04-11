import fastify from 'fastify'
import fastifyCors from '@fastify/cors'
import pinoPretty from 'pino-pretty'
import pino from 'pino'

const DHT_QUERY_THROTTLE = 6 * 1000 // 6 seconds
const MAX_CACHE_SIZE = 256 * 256

/**
 * Most simple cache I can think of!
 * @param {number} size
 */
class Cache extends Map {
  constructor (size = MAX_CACHE_SIZE) {
    super()
    this._maxSize = size
  }

  set (key, value) {
    this.delete(key)
    super.set(key, value)

    if (this.size > this._maxSize) {
      this.delete(this.keys().next().value)
    }

    return this
  }

  get (key) {
    return super.get(key)
  }
}

const SCHEMAS = {
  RECORD: {
    description: 'Record parameters to be (or as) stored in the DHT',
    type: 'object',
    required: ['seq', 'sig', 'v'],
    properties: {
      v: {
        description: 'Value of the record in base64',
        type: 'string',
        contentEncoding: 'base64'
      },
      seq: {
        description: 'Timestamp of the record',
        type: 'number'
      },
      sig: {
        description: 'Signature of the record value and sequnce number, in hex encoding',
        type: 'string',
        pattern: '^[a-fA-F0-9]{128}$'
      }
    }
  },
  QUERY_METADATA: {
    description: 'Last query to the DHT from which the record was retrieved or stored',
    type: 'object',
    required: ['type', 'nodes', 'time'],
    properties: {
      type: {
        description: 'Type of the query',
        type: 'string',
        enum: ['put', 'get']
      },
      nodes: {
        description: 'Number of responding nodes',
        type: 'number'
      },
      time: {
        description: 'Timestamp of the query in seconds',
        type: 'number'
      }
    }
  }
}

/**
 * Pkarr web server
 */
export default class Server {
  /**
   * @param {object} opts
   * @param {import('./dht.js').default} [opts.dht]
   * @param {Map} [opts.cache] Cache: same interface as Map, but get() can be async
   */
  constructor (opts) {
    /** @type {import('./dht.js').default} */
    this.dht = opts.dht
    this.cache = opts.cache || new Cache()

    const logger = opts.log === false
      ? false
      : opts.production
        ? pino()
        : pino(pinoPretty({
          colorize: true,
          minimumLevel: 'info',
          colorizeObjects: true
        }))

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
          properties: {
            record: SCHEMAS.RECORD,
            query: SCHEMAS.QUERY_METADATA
          }
        }
      },
      handler: async (request, reply) => {
        let cached = await this.cache.get(request.params.key)

        const shouldQuery = !cached ||
          // Too few nodes (See Bep44 #Expiration)
          cached.query.nodes <= 8 ||
          (
            // Newer data than cache
            request.body.v !== cached.record.v && request.body.seq > cached.record.seq &&
            // AND the cache is older than the DHT_QUERY_THROTTLE
            olderThanThrottle()
          )

        if (shouldQuery) {
          const response = await this.dht.put(
            Buffer.from(request.params.key, 'hex'),
            {
              seq: request.body.seq,
              v: Buffer.from(request.body.v, 'base64'),
              sig: Buffer.from(request.body.sig, 'hex')
            }
          )

          cached = {
            record: request.body,
            query: {
              type: 'put',
              nodes: response.nodes.length,
              time: Math.floor(Date.now() / 1000)
            }
          }

          this.cache.set(request.params.key, cached)
        }

        reply.code(200).send(cached)
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
              record: SCHEMAS.RECORD,
              query: SCHEMAS.QUERY_METADATA
            }
          }
        }
      },
      handler: async (request, reply) => {
        let cached = await this.cache.get(request.params.key)

        const shouldQuery = !cached || olderThanThrottle(cached)

        if (shouldQuery) {
          const response = await this.dht.get(Buffer.from(request.params.key, 'hex'))

          if (!response && !cached) {
            reply.code(404).send(null)
            return
          }

          cached = {
            record: {
              seq: response.seq,
              v: response.v.toString('base64'),
              sig: response.sig.toString('hex')
            },
            query: {
              type: 'get',
              nodes: response.nodes.length,
              time: Math.floor(Date.now() / 1000)
            }
          }

          this.cache.set(request.params.key, cached)
        }

        reply.code(200).send(cached)
      }
    })
  }

  static async start (opts) {
    const server = new Server(opts)
    await server.listen({ host: '0.0.0.0', port: 0, ...opts })
    return server
  }

  address () {
    const addr = this.app.server.address()
    addr.toString = () => typeof addr === 'string' ? addr : `http://${addr.address}:${addr.port}`
    return addr
  }

  destroy () {
    this.dht.destroy()
    this.app.server.close()
  }
}

function olderThanThrottle (cached) {
  if (!cached?.query?.time) return true
  const passed = Date.now() - cached.query.time * 1000
  return passed > DHT_QUERY_THROTTLE
}
