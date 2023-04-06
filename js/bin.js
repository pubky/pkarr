#!/usr/bin/env node
import z32 from 'z32'

import DHT from './lib/dht.js'

const resolveKey = async (key) => {
  if (!key) {
    console.error('Please provide a key to resolve')
    return
  }

  const dht = new DHT()

  const keyBytes = z32.decode(key.replace('pk:', ''))
  const response = await dht.get(keyBytes)
  console.log(response)

  dht.destroy()
}

const showHelp = () => {
  console.log(`
Usage: pkarr [command] [options]

Commands:
  resolve <key> Resolve resource records for a given key
  help          Show this help message

Examples:
  pkarr resolve pk:54ftp7om3nkc619oaxwbz4mg4btzesnnu1k63pukonzt36xq144y
  pkarr help
`)
}

const command = process.argv[2]

switch (command) {
  case 'resolve':
    resolveKey(process.argv[3])
    break
  case '-h':
  case 'help':
    showHelp()
    break
  default:
    showHelp()
    process.exit(1)
}
