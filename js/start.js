#!/usr/bin/env node

import Server from './lib/server.js'

const port = process.env.PORT || 7527
await Server.start({ port })
