#!/usr/bin/env node

import Server from './lib/server.js'
import DHT from './lib/dht.js'

const dht = new DHT()

await Server.start({
  dht,
  port: process.env.PORT || 7527,
  production: process.env.NODE_ENV === 'production'
})
