import http from 'http'
import DHT from 'bittorrent-dht'
import sodium from 'sodium-universal'

export const verify = sodium.crypto_sign_verify_detached

export default class Server {
  constructor (opts = {}) {
    const dht = new DHT({ verify })
    dht.listen(6881)
    this.dht = dht

    this.server = http.createServer(function (req, res) {
      try {
        const key = boilerplate(req, res)
        if (!key) return

        // Handle GET /pkarr/:key
        if (req.method === 'GET') {
          handleGet(req, res, dht, { key })
        } else if (req.method === 'PUT') {
          handlePut(req, res, dht, { key })
        }
      } catch (error) {
        console.error('Unexpected Error', error)
        res.statusCode = 500
        res.end(JSON.stringify({ error: 'An unexpected error occurred on the server' }));
      }
    })
  }

  static async start (opts) {
    const server = new Server(opts)
    await server.listen()
    return server
  }

  async listen (port = 3000) {
    return new Promise((resolve, reject) => this.server.listen(port, err => err ? reject(err) : resolve()))
  }

  async destroy () {
    this.dht.destroy()
    return new Promise((resolve, reject) => {
      this.server.close(err => {
        if (err) {
          reject(err)
        } else {
          resolve()
        }
      })
    })
  }
}

function handleGet (_, res, dht, { key }) {
  dht.get(key, (err, response) => {
    if (err) {
      res.statusCode = 500
      res.end(JSON.stringify({error: 'Failed to fetch value from DHT'}))
      return
    }

    if (!response) {
      res.statusCode = 404
      res.end(JSON.stringify({ error: 'Value not found in DHT'}))
      return
    }

    res.end(JSON.stringify({
      k: response.k.toString('hex'),
      s: response.seq || 0,
      v: response.v.toString('hex')
    }))
  })
}

function handlePut (req, res, dht, { key }) {
  let body = ''
  req.on('data', chunk => body += chunk)
  req.on('end', async () => {
    const payload = jsonSafeParse(body);

    if (!payload) {
      res.end(JSON.stringify({ error: 'Invalid payload', payload }))
      return 
    }

    const opts = {
      k: key,
      seq: payload.seq || 0,
      v: Buffer.from(payload.v, 'hex'),
      sign: () => Buffer.from(payload.sig, 'hex')
    }

    try { 
      dht.put(opts, (err) => {
        if (err) {
          res.statusCode = 500
          res.end(JSON.stringify({ error: err.message }))
          return
        }
        res.end(JSON.stringify({ key: key.toString('hex') }))
      })
    } catch (error) {
      res.end(JSON.stringify({ error: error.message }))
    }
  })
}

function boilerplate (req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET,PUT')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
  res.setHeader('Content-Type', 'application/json')

  // Preflight request. Reply successfully:
  if(req.method === 'OPTIONS') {
    res.statusCode = 204
    res.end()
    return;
  }

  const path = req.url.split('/')

  if (!path[1] === 'pkarr') {
    res.statusCode = 404
    res.end(JSON.stringify({ error: 'Not Found' }))
    return
  }

  try {
    return Buffer.from(path[2], 'hex')
  } catch (error) {
    if (!res.writableEnded) {
      res.end(JSON.stringify({ error: error.message }))
    }
  }
}

function jsonSafeParse (str) {
  try {
    return JSON.parse(str)
  } catch (error) {
    return null
  }
}
