#!/usr/bin/env node
import z32 from 'z32'
import chalk from 'chalk'
import fs from 'fs'

import DHT from './lib/dht.js'
import * as pkarr from './lib/tools.js'

const resolveKey = async (key) => {
  if (!key) {
    console.error(chalk.red('✘') + ' Please provide a key to resolve!\n')
    return
  }

  const dht = new DHT()

  const keyBytes = z32.decode(key.replace('pk:', ''))

  const [success, fail] = loading('Resolving')
  const response = await dht.get(keyBytes)

  if (response) {
    success('Resolved')
    dht.destroy()
  } else {
    fail("couldn't resolve records!")
    console.log("")
    dht.destroy()
    return
  }

  const records = await pkarr.codec.decode(response.v)

  table(records).forEach((row) => {
    console.log(chalk.green('  ❯ ') + row.join(' '))
  })

  console.log('')

  const metadata = [
    ['updated_at', new Date(response.seq * 1000).toLocaleString()],
    ['size', (response.v?.byteLength || 0) + '/1000 bytes'],
    ['nodes', response.nodes.length]
  ]
  table(metadata).forEach((row) => {
    console.log(chalk.dim('  › ' + row.join(': ')))
  })

  console.log('')
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
  try {
    const request = await pkarr.createPutRequest(keyPair, records)

    const start = Date.now()
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

function loading(message) {
  const start = Date.now()
  let dots = 1
  let started = false

  next()
  const interval = setInterval(next, 200)

  function next() {
    if (started) removeLastLine()
    else started = true
    console.log(chalk.dim(' ◌ '), message + '.'.repeat(dots))
    dots += 1
    dots = dots % 4
  }

  function success(message) {
    removeLastLine()
    clearInterval(interval)
    console.log(chalk.green(' ✔ ' + message + seconds()))
  }
  function fail(message) {
    removeLastLine()
    clearInterval(interval)
    clearInterval(interval)
    console.log(chalk.red(' ✘ '), message + seconds())
  }
  function seconds() {
    return ` (${((Date.now() - start) / 1000).toFixed(2)} seconds)`
  }

  return [success, fail]
}

function removeLastLine() {
  process.stdout.write('\u001B[F\u001B[K')
}

function table(records) {
  const pads = {}

  records = records.filter(r => r[0] && r[1])

  records.forEach(record => {
    record.forEach((element, index) => {
      pads[index] = Math.max(pads[index] || 0, element?.toString().length || 0)
    })
  })

  return records.map(r => r.map((c, i) => c.toString().padEnd(pads[i])))
}
