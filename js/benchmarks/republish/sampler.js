import fs from 'fs'
import DHT from '../../lib/dht.js'

/**
 * Sample a random user and check if its record is alive,
 * log the success rate every 1000 records
 */

const PATH = './data/results-test.csv'
const INTERVAL = 1000 // Make a request per second
const MAX_SAMPLE_SIZE = 100

const dht = new DHT()

const users = fs.readFileSync('./data/users.csv')
  .toString()
  .split(/\n/)
  .filter(Boolean)
  .map(r => r.split(',')[0])

const batch = new Map()

onInterval()
setInterval(onInterval, INTERVAL)

function onInterval () {
  flushBatchMaybe()
  const key = sample()
  check(key)
}

/**
 * @param {string} key
 */
async function check (key) {
  try {
    const response = await dht.get(Buffer.from(key, 'hex'))

    const nodes = response?.nodes.length || 0

    batch.set(key, nodes)

    log(key.toString('hex'), 'GET', 'nodes', nodes, 'batch', batch.size, 'rate', successRate())
  } catch (error) {
    log('ERROR', error)
  }
}

function successRate () {
  const checkout = [...batch.entries()]
  const resolved = checkout.filter(entry => entry[1] > 0)
  const rate = resolved.length / checkout.length
  return rate.toFixed(2)
}

function flushBatchMaybe () {
  if (batch.size < MAX_SAMPLE_SIZE) return

  const rate = successRate()

  const line = [now(), batch.size, rate].join(',') + '\n'
  log('Appending line', line)

  batch.clear()

  fs.appendFileSync(PATH, line)
}

/**
 * @returns {string}
 */
function sample () {
  const randomIndex = Math.floor(Math.random() * users.length)
  const key = users[randomIndex]

  // already in current batch
  if (batch.has(key.toString('hex'))) return sample()

  return key
}

function log (...args) {
  console.log(now(), ...args)
}

function now () {
  return new Date().toLocaleString('en-GB', { timeZone: 'Asia/Istanbul' }).replace(',', '')
}
