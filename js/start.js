#!/usr/bin/env node

import Server from './lib/server.js'
import DHT from './lib/dht.js'

const dht = new DHT()

const port = process.env.PORT || 7527
await Server.start({ dht, port })
