#!/usr/bin/env node
import z32 from 'z32'
import chalk from 'chalk'
import crypto from 'crypto'
import fs from 'fs'
import path, { dirname } from 'path'
import { homedir } from 'os'
import { fileURLToPath } from 'url'

import DHT from './lib/dht.js'
import * as pkarr from './lib/tools.js'
import Republisher from './lib/republisher.js'

const ROOT_DIR = path.join(homedir(), '.pkarr')
const KEEP_ALIVE_PATH = path.join(ROOT_DIR, 'keepalive.json')

const resolveKey = async (key) => {
  if (!key) {
    console.error(chalk.red('✘') + ' Please provide a key to resolve!\n')
    return
  }

  const dht = new DHT()

  const keyBytes = z32.decode(key.replace('pk:', ''))

  try {
    if (keyBytes.byteLength !== 32) throw new Error('Invalid key')
  } catch {
    console.error(chalk.red('✘') + ` Key ${key} is not valid\n`, chalk.dim('keys must be z-base32 encoded 32 bytes'))
    return
  }

  const [success, fail] = loading('Resolving')
  const response = await dht.get(keyBytes)

  if (response) {
    success('Resolved')
    dht.destroy()
  } else {
    fail("couldn't resolve records!")
    console.log('')
    dht.destroy()
    return
  }

  const records = await pkarr.codec.decode(response.v)

  table(records).forEach((row) => {
    console.log(chalk.green('   ❯ ') + row.join(' '))
  })

  console.log('')

  const metadata = [
    ['updated_at', new Date(response.seq * 1000).toLocaleString()],
    ['size', (response.v?.byteLength || 0) + '/1000 bytes'],
    ['from', response.nodes.map(n => n.host + ':' + n.port + (n.client ? ' - ' + n.client : '')).join(', ')]
  ]
  table(metadata).forEach((row) => {
    console.log(chalk.dim('   › ' + row.join(': ')))
  })

  console.log('')
}

const publish = async () => {
  console.log(chalk.dim(' ◌ '), 'Enter a passphrase', chalk.dim('(learn more at https://www.useapassphrase.com/)'))

  const prompt = chalk.dim('    Passphrase: ')
  console.log(prompt)

  const stdin = process.stdin
  stdin.setRawMode(true)

  const passphrase = await new Promise(resolve => {
    let pass = ''

    const listener = (char) => {
      const keyCode = char.toString('utf8').charCodeAt(0)
      removeLastLine()

      if (keyCode === 13) { // Enter key
        stdin.removeListener('data', listener)
        resolve(pass)
      } else if (keyCode === 127 || keyCode === 8) { // Backspace or Delete key
        if (pass.length > 0) {
          pass = pass.slice(0, -1)
        }
      } else if (keyCode === 3) { // Ctrl + c
        process.exit()
      } else {
        pass += char
      }

      console.log(prompt + pass.split('').map(() => '*').join(''))
    }

    stdin.on('data', listener)
  })

  const seed = crypto.createHash('sha256').update(passphrase, 'utf-8').digest()
  const keyPair = pkarr.generateKeyPair(seed)
  const pk = 'pk:' + z32.encode(keyPair.publicKey)

  console.log(chalk.green('    ❯', pk))

  console.log(chalk.dim(' ◌ '), 'Enter records to publish:')
  console.log(chalk.green('    ❯'), chalk.dim('Add record "<name> <value>" or press enter to submit'))

  const records = await new Promise(resolve => {
    const _records = []
    let current = ''

    stdin.on('data', (char) => {
      const keyCode = char.toString('utf8').charCodeAt(0)

      if (keyCode === 13) { // Enter key
        if (current.length > 0) {
          _records.push(current.split(' '))
          current = ''
        } else {
          resolve(_records)
          return
        }
      } else if (keyCode === 127 || keyCode === 8) { // Backspace or Delete key
        removeLastLine()
        if (current.length > 0) {
          current = current.slice(0, -1)
        }
      } else if (keyCode === 3) { // Ctrl + c
        process.exit()
      } else {
        removeLastLine()
        current += char
      }

      console.log(chalk.green('    ❯'), current.length > 0 ? current : chalk.dim('Add record or press enter to submit'))
    })
  })
  stdin.setRawMode(false)
  stdin.pause()

  const dht = new DHT()
  const request = await pkarr.createPutRequest(keyPair, records)
  const [success, fail] = loading('Publishing')

  try {
    await dht.put(keyPair.publicKey, request)
    success('Published')
  } catch (error) {
    fail('Failed to publish. got an error:')
    console.log('    ', error)
  }

  dht.destroy()
}

const keepalive = (command, ...args) => {
  let set = new Set()
  try {
    set = new Set(JSON.parse(fs.readFileSync(KEEP_ALIVE_PATH).toString()))
  } catch {
    try {
      fs.mkdirSync(ROOT_DIR)
    } catch { }
  }

  let keys = set

  // Validate args
  if (['add', 'remove'].includes(command)) {
    if (args.length === 0) {
      console.error(chalk.red('✘') + ' Please provide key(s) to add to the list of keys to keep alive!\n')
      return
    }

    args = args.map(arg => arg.replace('pk:', ''))

    keys = args
  }

  // Validate keys
  for (const key of keys) {
    try {
      const decoded = z32.decode(key)
      if (decoded.byteLength !== 32) throw new Error('Invalid key')
    } catch {
      console.error(chalk.red('✘') + ` Key ${key} is not valid\n`, chalk.dim('keys must be z-base32 encoded 32 bytes'))
      return
    }
  }

  switch (command) {
    case 'add':
      for (const key of args) {
        set.add(key)
      }
      break

    case 'remove':
      for (const key of args) {
        set.delete(key)
      }
      break

    case 'list':
      for (const key of set) {
        console.log(chalk.green('    ❯'), key)
      }
      break

    case undefined:
      if (set.size === 0) {
        console.error(chalk.red('✘') + ' keep alive list is empty!')
        return
      }

      console.log(chalk.green('Starting republisher...'))

      Republisher.start([...set].map(key => ({ key: z32.decode(key) })))
      break

    default:
      console.error(chalk.red('✘') + ` Unknown command "${command}"`)
      break
  }

  if (['add', 'remove'].includes(command)) {
    fs.writeFileSync(KEEP_ALIVE_PATH, JSON.stringify([...set]))
    console.log(chalk.green(' ✔ '), 'Successfully updated list of keys!')
  }
}

const version = () => {
  const __filename = fileURLToPath(import.meta.url)
  const __dirname = dirname(__filename)
  const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, './package.json')).toString())
  console.log(pkg.version)
}

const showHelp = () => {
  console.log(`
Usage: pkarr [command] [options]

Commands:
  resolve <key>                       Resolve resource records for a given key
  publish                             Publish resource records for a given seed
  keepalive [add | remove] [...keys]  Add or remove keys to the list of keys to keepalive
  keepalive list                      List stored keys
  keepalive                           Run the publisher to keep stored keys alive
  help                                Show this help message

Examples:
  pkarr resolve pk:yqrx81zchh6aotjj85s96gdqbmsoprxr3ks6ks6y8eccpj8b7oiy
  pkarr publish
  pkarr keepalive add pk:yqrx81zchh6aotjj85s96gdqbmsoprxr3ks6ks6y8eccpj8b7oiy
  pkarr keepalive remove yqrx81zchh6aotjj85s96gdqbmsoprxr3ks6ks6y8eccpj8b7oiy
  pkarr keepalive list
  pkarr keepalive
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
  case 'keepalive':
    keepalive(...process.argv.slice(3))
    break
  case '-v':
  case 'version':
    version()
    break
  case '-h':
  case 'help':
    showHelp()
    break
  default:
    showHelp()
    process.exit(1)
}

function loading (message) {
  const start = Date.now()
  let dots = 1
  let started = false

  next()
  const interval = setInterval(next, 200)

  function next () {
    if (started) removeLastLine()
    else started = true
    console.log(chalk.dim(' ◌ '), message + '.'.repeat(dots))
    dots += 1
    dots = dots % 4
  }

  function success (message) {
    removeLastLine()
    clearInterval(interval)
    console.log(chalk.green(' ✔ ' + message + seconds()))
  }
  function fail (message) {
    removeLastLine()
    clearInterval(interval)
    clearInterval(interval)
    console.log(chalk.red(' ✘ '), message + seconds())
  }
  function seconds () {
    return ` (${((Date.now() - start) / 1000).toFixed(2)} seconds)`
  }

  return [success, fail]
}

function removeLastLine () {
  process.stdout.write('\u001B[F\u001B[K')
}

function table (records) {
  const pads = {}

  records = records.filter(r => r[0] && r[1])

  records.forEach(record => {
    record.forEach((element, index) => {
      pads[index] = Math.max(pads[index] || 0, element?.toString().length || 0)
    })
  })

  return records.map(r => r.map((c, i) => c.toString().padEnd(pads[i])))
}
