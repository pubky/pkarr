import fs from 'fs'
import DHT from 'pkarr/lib/dht.js'

const REPUBLISH_INTERVAL = 1 * 1000 // 1 request per second
const MAX_BATCH_SIZE = 1000 // log every 1000 requests

const PATH = './data/results.csv'

const dht = new DHT()

const users = fs.readFileSync('./data/users.csv')
  .toString()
  .split(/\n/)
  .filter(Boolean)
  .map(r => r.split(',').map(e => Buffer.from(e, 'hex')))

const v = Buffer.from('000b0c805b5b225f74657374222c227374696c6c20616c697665225d5d03', 'hex');
const seq = 1681656800

const batch = new Map()

onInterval()
setInterval(onInterval, REPUBLISH_INTERVAL);

function onInterval() {
  flushBatchMaybe()
  const [key, sig] = next()
  republish(key, sig)
}

/**
 * @param {Buffer} key
 * @param {Buffer} sig
 */
async function republish(key, sig) {
  try {

    let response = await dht.get(key)

    const nodes = response?.nodes.length || 0

    batch.set(key.toString('hex'), nodes)

    log(key.toString('hex'), "GET", "nodes", nodes, "batch", batch.size)

    if (nodes < 8) {
      response = await dht.put(key, { v, seq, sig })
    }
  } catch (error) {
    log("ERROR", error)
  }
}

function flushBatchMaybe() {
  if (batch.size < MAX_BATCH_SIZE) return

  const checkout = [...batch.entries()]
  batch.clear()

  const resolved = checkout.filter(entry => entry[1] > 0)
  const ratio = resolved.length / checkout.length

  const datetime = new Date().toISOString().replace(/\..*$/g, '').replace(/-/g, '/').split('T').join(' ')

  const line = [datetime, checkout.length, ratio].join(',') + '\n'
  log("Appending line", line)

  fs.appendFileSync(PATH, line)
}

/**
 * @returns {[Buffer, Buffer]}
 */
function next() {
  const randomIndex = Math.floor(Math.random() * users.length);
  const [key, sig] = users[randomIndex]

  // already in current batch
  if (batch.has(key.toString('hex'))) return next()

  return [key, sig]
}

function log(...args) {
  console.log(new Date().toLocaleString(), ...args)
}
