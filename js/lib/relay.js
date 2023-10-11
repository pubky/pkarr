import http from 'http'
import z32 from 'z32'

import DHT from './dht.js'
import SignedPacket from './signed_packet.js'

const DEFAULT_PORT = 0

const MAX_BODY_SIZE = 1000 + 64 + 8 // 1000 bytes for value, 64 for signature and 8 for seq

export default class Server {
  /**
   * @param {object} [options]
   * @param {DHT} [options.dht]
   */
  constructor (options) {
    this._server = http.createServer(this._handle.bind(this))

    this._dht = options.dht || new DHT()
  }

  /**
   * @param {object} [options]
   * @param {DHT} [options.dht]
   * @param {number} [options.port]
   */
  static start (options) {
    const server = new Server(options)
    return server.listen(options.port)
  }

  get port () {
    // @ts-ignore
    return this._server.address().port
  }

  get address () {
    return 'http://localhost:' + this.port
  }

  /**
   * Start a web relay listening on the provided port or default port 3000
   *
   * @param {number} [port]
   */
  listen (port) {
    return new Promise(resolve => {
      this._server.listen(port || DEFAULT_PORT, () => {
        resolve(this)
      })
    })
  }

  /**
   * Close the web relay
   */
  async close () {
    await this._dht.destroy()
    return this._server.close()
  }

  /**
   * @param {http.IncomingMessage} req
   * @param {http.ServerResponse} res
   */
  _handle (req, res) {
    const key = parseURL(req.url)
    if (!key) {
      return badRequest(req, res, 'Invalid key')
    }

    // Set CORS headers on all responses
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Methods', 'GET, PUT, OPTIONS')

    switch (req.method) {
      case 'OPTIONS':
        this._OPTIONS(req, res)
        break
      case 'GET':
        this._GET(req, res, key)
        break
      case 'PUT':
        this._PUT(req, res, key)
        break
      default:
        res.writeHead(405, 'Method not allowed')
        res.end()
    }
  }

  /**
   * Respond to preflight requests
   *
   * @param {http.IncomingMessage} _req
   * @param {http.ServerResponse} res
   */
  _OPTIONS (_req, res) {
    res.writeHead(204)
    res.end()
  }

  /**
   * @param {http.IncomingMessage} req
   * @param {http.ServerResponse} res
   * @param {Uint8Array} key
   */
  async _PUT (req, res, key) {
    const requestBodyChunks = []
    let requestBodyLength = 0

    // Handle data events to read the request body.
    req.on('data', (chunk) => {
      // Check if the body size exceeds the maximum allowed size.
      if (requestBodyLength + chunk.length > MAX_BODY_SIZE) {
        res.statusCode = 413 // Payload Too Large
        res.end('Request body size should not exceed ' + MAX_BODY_SIZE + ' bytes. Got ' + requestBodyLength + ' bytes.')
        req.destroy() // Close the request to stop receiving data.
        return
      }

      // Append the chunk to the request body and update the length.
      requestBodyChunks.push(chunk)
      requestBodyLength += chunk.length
    })

    // Handle the end event when the request is complete.
    req.on('end', async () => {
      // Process the request body (requestBody) as needed.
      // For this example, we'll just log it.
      const body = Buffer.concat(requestBodyChunks)

      /** @type {SignedPacket} */
      let signedPacket

      try {
        signedPacket = SignedPacket.fromBytes(key, body)
      } catch (error) {
        return badRequest(req, res, error.message)
      }

      try {
        await this._dht.put(signedPacket.bep44Args())

        success(req, res)
      } catch (error) {
        internalServerError(req, res, error)
      }
    })

    // Handle any errors during the request.
    req.on('error', (error) => {
      internalServerError(req, res, error)
    })
  }

  /**
   * @param {http.IncomingMessage} req
   * @param {http.ServerResponse} res
   * @param {Uint8Array} key
   */
  async _GET (req, res, key) {
    const response = await this._dht.get(key)
    if (!response) {
      res.writeHead(404, 'Not found')
      res.end()
      return
    }

    const signedPacket = SignedPacket.fromBep44Args(response)
    const body = signedPacket.bytes()

    success(req, res, body)
  }
}

/**
         * @param {string} url
         */
function parseURL (url) {
  try {
    const parts = url.split('/')
    const userID = parts[1]

    const publicKey = z32.decode(userID)
    if (publicKey.length === 32) return publicKey
  } catch { }

  return false
}

/**
 * @param {http.IncomingMessage} req
 * @param {http.ServerResponse} res
 * @param {Uint8Array} [body]
 */
function success (req, res, body) {
  console.log('success', req.method, req.url)
  res.writeHead(200, 'OK')
  res.end(body)
}

/**
 * @param {http.IncomingMessage} req
 * @param {http.ServerResponse} res
 * @param {string} message
 */
function badRequest (req, res, message) {
  console.log('bad request', req.method, req.url, message)
  res.writeHead(400, message)
  res.end(message)
}

/**
 * @param {http.IncomingMessage} req
 * @param {http.ServerResponse} res
 * @param {Error} error
 */
function internalServerError (req, res, error) {
  console.log('Internal server error', req.method, req.url, error)
  res.statusCode = 500 // Internal Server Error
  res.writeHead(500, 'Internal Server Error')
  res.end()
}
