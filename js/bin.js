#!/usr/bin/env node
import z32 from 'z32'
import chalk from 'chalk'
import crypto, { generateKeyPair } from 'crypto'

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
    console.log(chalk.green('   ❯ ') + row.join(' '))
  })

  console.log('')

  const metadata = [
    ['updated_at', new Date(response.seq * 1000).toLocaleString()],
    ['size', (response.v?.byteLength || 0) + '/1000 bytes'],
    ['nodes', response.nodes.length]
  ]
  table(metadata).forEach((row) => {
    console.log(chalk.dim('   › ' + row.join(': ')))
  })

  console.log('')
}

const publish = async () => {
  console.log(chalk.dim('◌ '), "Enter a passphrase", chalk.dim("(learn more at https://www.useapassphrase.com/)"))

  const prompt = chalk.dim('   Passphrase: ')
  console.log(prompt)

  const stdin = process.stdin;
  stdin.setRawMode(true);

  const passphrase = await new Promise(resolve => {
    let pass = ''; + '\n'

    const listener = (char) => {
      const keyCode = char.toString('utf8').charCodeAt(0);
      removeLastLine()

      if (keyCode === 13) { // Enter key
        stdin.removeListener('data', listener)
        resolve(pass)
      } else if (keyCode === 127 || keyCode === 8) { // Backspace or Delete key
        if (pass.length > 0) {
          pass = pass.slice(0, -1);
        }
      } else if (keyCode === 3) { // Ctrl + c
        process.exit()
      } else {
        pass += char;
      }

      console.log(prompt + pass.split('').map(() => "*").join(''))
    };

    stdin.on('data', listener)
  })

  const seed = crypto.createHash('sha256').update(passphrase, 'utf-8').digest()
  const keyPair = pkarr.generateKeyPair(seed)
  const pk = 'pk:' + z32.encode(keyPair.publicKey)

  console.log(chalk.green('   ❯', pk))

  console.log(chalk.dim('◌ '), 'Enter records to publish:')
  console.log(chalk.green("   ❯"), chalk.dim('Add record "<name> <value>" or press enter to submit'))

  const records = await new Promise(resolve => {
    const _records = [];
    let current = '';

    stdin.on('data', (char) => {
      const keyCode = char.toString('utf8').charCodeAt(0);

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
          current = current.slice(0, -1);
        }
      } else if (keyCode === 3) { // Ctrl + c
        process.exit()
      } else {
        removeLastLine()
        current += char;
      }

      console.log(chalk.green("   ❯"), current.length > 0 ? current : chalk.dim("Add record or press enter to submit"))
    });
  })
  stdin.setRawMode(false);
  stdin.pause()

  const dht = new DHT()
  const request = await pkarr.createPutRequest(keyPair, records)
  const [success, fail] = loading('Publishing')

  try {
    await dht.put(keyPair.publicKey, request)
    success('Published Resource Records for ' + chalk.green(pk))
  } catch (error) {
    fail('Failed to publish. got an error:')
    console.log(error)
  }

  dht.destroy()
}

const showHelp = () => {
  console.log(`
Usage: pkarr [command] [options]

Commands:
  resolve <key>  Resolve resource records for a given key
  publish        Publish resource records for a given seed
  help           Show this help message

Examples:
  pkarr resolve pk:54ftp7om3nkc619oaxwbz4mg4btzesnnu1k63pukonzt36xq144y
  pkarr publish
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
