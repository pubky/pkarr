import fs from 'fs'
import DHT from '../../lib/dht.js'

const v = Buffer.from('000b0c805b5b225f74657374222c227374696c6c20616c697665225d5d03', 'hex');
const seq = 1681656800

const users = fs.readFileSync('./data/users.csv')
  .toString()
  .split(/\n/)
  .filter(Boolean)
  .map(r => {
    const [key, sig] = r.split(',').map(e => Buffer.from(e, 'hex'))
    return [key, v, seq, sig]
  })

const REPUBLISH_INTERVAL = 2 // once per two hours
const MAX_CONCURRENCY = users.length / 40 / REPUBLISH_INTERVAL // 40 is an emperical constant! (fancy for trial and error)


const dht = new DHT({ concurrency: MAX_CONCURRENCY })
await dht.ready()
console.log('done')

let count = 0;

republishAll()
setInterval(republishAll, REPUBLISH_INTERVAL * 60 * 60 * 1000)

function republishAll() {
  log("Republishing all")
  count = 0; // reset count
  users.map(u => republish(...u))
}

/*
 * @param {Buffer} key
 * @param {Buffer} sig
 */
async function republish(key, v, seq, sig) {
  try {
    let shouldPut = false;

    {
      let response = await dht.get(key)
      const nodes = response?.nodes.length || 0
      log(key.toString('hex'), "GET", "nodes", nodes)
      shouldPut = nodes < 8
    }

    if (shouldPut) {
      const response = await dht.put(key, { v, seq, sig })
      const nodes = response?.nodes.length || 0
      if (nodes > 1) count += 1

      log(key.toString('hex'), "PUT", "nodes", nodes, "count", count)
    } else {
      count += 1
    }
  } catch (error) {
    log("ERROR", key.toString('hex'), error)
  }
}

function log(...args) {
  console.log(now(), ...args)
}

function now() {
  return new Date().toLocaleString('en-GB', { timeZone: 'Asia/Istanbul' }).replace(',', '')
}
