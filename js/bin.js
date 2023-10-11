#!/usr/bin/env node
import z32 from 'z32'
import chalk from 'chalk'
import crypto from 'crypto'
import fs from 'fs'
import path, { dirname } from 'path'
import { homedir } from 'os'
import { fileURLToPath } from 'url'

import DHT from './lib/dht.js'
import Republisher from './lib/republisher.js'
import Server from './lib/relay.js'
import Pkarr from './index.js'
import SignedPacket from './lib/signed_packet.js'

const ROOT_DIR = path.join(homedir(), '.pkarr')
const KEEP_ALIVE_PATH = path.join(ROOT_DIR, 'keepalive.json')

const resolveKey = async (key, fullLookup) => {
  if (!key) {
    console.error(chalk.red('✘') + ' Please provide a key to resolve!\n')
    return
  }

  const [success, fail] = loading('Resolving')

  let response

  try {
    response = await Pkarr.resolve(key, { fullLookup })

    if (response) {
      success('Resolved')
    } else {
      fail("couldn't resolve records!")
      console.log('')
      return
    }

    const { signedPacket, nodes } = response

    table(signedPacket.packet().answers.map(a => [a.name, a.ttl, a.class, a.type, a.data])).forEach((answer) => {
      console.log(chalk.green('   ❯ ') + answer.join(' '))
    })

    console.log('')

    const metadata = [
      ['updated_at', new Date(signedPacket.timestamp() / 1000).toLocaleString()],
      ['size', signedPacket.size() + '/1000 bytes'],
      ['from', nodes.map(n => n.host + ':' + n.port + (n.client ? ' - ' + n.client : '')).join(', ')]
    ]
    table(metadata).forEach((row) => {
      console.log(chalk.dim('   › ' + row.join(': ')))
    })

    console.log('')
  } catch (error) {
    if (error.message === 'Invalid key') {
      fail(` Key ${key} is not valid\n`, chalk.dim('keys must be z-base32 encoded 32 bytes'))
    } else {
      console.error(error)
    }
  }
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
  const keyPair = Pkarr.generateKeyPair(seed)
  const pk = 'pk:' + z32.encode(keyPair.publicKey)

  console.log(chalk.green('    ❯', pk))

  console.log(chalk.dim(' ◌ '), 'Enter records to publish:')
  console.log(chalk.green('    ❯'), chalk.dim('Add record "<name> <ttl> <class> <value>" for example "@ 30 IN 1.1.1.1" or press enter to submit'))

  /** @type {import('dns-packet').Answer[]} */
  const records = await new Promise(resolve => {
    const _records = []
    let current = ''

    stdin.on('data', (char) => {
      const keyCode = char.toString('utf8').charCodeAt(0)

      if (keyCode === 13) { // Enter key
        if (current.length > 0) {
          const [name, ttl, _class, type, data] = current.split(' ')
          _records.push({ name, ttl, _class, type, data })
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
  const [success, fail] = loading('Publishing')

  /** @type {import('dns-packet').Packet} */
  const packet = {
    id: 0,
    type: 'response',
    flags: 0,
    answers: records
  }

  const signedPacket = SignedPacket.fromPacket(keyPair, packet)

  try {
    await Pkarr.publish(signedPacket)
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

      Republisher.start([...set].map(key => ({ k: z32.decode(key) })))
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

/**
 * @param {number} port
 */
const relay = async (port = 6881) => {
  const server = await Server.start({ port })
  console.log('Relay server is listening on address:', server.address)
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
  relay                               Run a relay server
  help                                Show this help message

Options:
  -f, --full-lookup                   Perform a full lookup while resolving a key
  -p, --port                          Port to run the relay server on (default: 6881)

Examples:
  pkarr resolve pk:yqrx81zchh6aotjj85s96gdqbmsoprxr3ks6ks6y8eccpj8b7oiy
  pkarr publish
  pkarr keepalive add pk:yqrx81zchh6aotjj85s96gdqbmsoprxr3ks6ks6y8eccpj8b7oiy
  pkarr keepalive remove yqrx81zchh6aotjj85s96gdqbmsoprxr3ks6ks6y8eccpj8b7oiy
  pkarr keepalive list
  pkarr keepalive
  pkarr relay --port=6881
  pkarr help
`)
}

const command = process.argv[2]

switch (command) {
  case 'resolve':
    resolveKey(process.argv[3], process.argv.find(arg => arg === '-f' || arg === '--full-lookup'))
    break
  case 'publish':
    publish(process.argv[3], process.argv.slice(4))
    break
  case 'keepalive':
    keepalive(...process.argv.slice(3))
    break
  case 'relay':
    relay(process.argv.find(arg => arg.startsWith('-p=') || arg.startsWith('--port'))?.split('=')[1])
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
