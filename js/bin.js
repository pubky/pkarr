#!/usr/bin/env node
import z32 from 'z32'
import Table from 'cli-table3'
import chalk from 'chalk'
import fs from 'fs'

import DHT from './lib/dht.js'
import * as pkarr from './lib/tools.js'

const resolveKey = async (key) => {
  if (!key) {
    console.error('Please provide a key to resolve')
    return
  }

  console.log(chalk.gray('Resolving ' + key + ' ...'))
  const start = Date.now()
  const dht = new DHT()

  const keyBytes = z32.decode(key.replace('pk:', ''))
  const response = await dht.get(keyBytes)

  if (!response) {
    console.log(chalk.red("Couldn't resolve records"))
    dht.destroy()
    return
  }

  const records = await pkarr.codec.decode(response.v)

  console.log(chalk.green('Resolved Resource Records in ' + (Date.now() - start) / 1000 + ' seconds'))
  const table = new Table({
    head: ['name', 'value']
  })
  records.forEach((record) => table.push(record))
  console.log(table.toString())

  const metadata = new Table({ head: ['metadata', 'value'] })
  metadata.push(['last update', new Date(response.seq * 1000).toLocaleString()])
  metadata.push(['size', response.v.byteLength + '/1000 bytes'])
  metadata.push(['responding nodes ', response.nodes.length])
  console.log(metadata.toString())

  dht.destroy()
}

const publish = async (seedPath, records) => {
  records = records.reduce((acc, item, index) => {
    if (index % 2 === 0) {
      acc.push([item])
    } else {
      acc[acc.length - 1].push(item)
    }
    return acc
  }, [])

  const dht = new DHT()

  const seed = Buffer.from(fs.readFileSync(seedPath, 'utf-8'), 'hex')
  const keyPair = pkarr.generateKeyPair(seed)

  console.log('Publishing records for', 'pk:' + z32.encode(keyPair.publicKey) + '\n')
  const start = Date.now()
  try {
    const request = await pkarr.createPutRequest(keyPair, records)

    await dht.put(keyPair.publicKey, request)
    console.log(chalk.green('Published Resource Records in ' + (Date.now() - start) / 1000 + ' seconds'))
  } catch (error) {
    console.log(chalk.red('Failed to publish. got an error:'))
    console.log(error)
  }

  dht.destroy()
}

const showHelp = () => {
  console.log(`
Usage: pkarr [command] [options]

Commands:
  resolve <key>                                 Resolve resource records for a given key
  publish <seed path (hex string)> [...records] Publish resource records for a given seed
  help                                          Show this help message

Examples:
  pkarr resolve pk:54ftp7om3nkc619oaxwbz4mg4btzesnnu1k63pukonzt36xq144y
  pkarr publish ./seed foo bar answer 42
  pkarr help
`)
}

const command = process.argv[2]

switch (command) {
  case 'resolve':
    resolveKey(process.argv[3])
    break
  case 'publish':
    publish(process.argv[3], process.argv.slice(4))
    break
  case '-h':
  case 'help':
    showHelp()
    break
  default:
    showHelp()
    process.exit(1)
}
