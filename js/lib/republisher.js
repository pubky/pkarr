import DHT from './dht.js'
import z32 from 'z32'

const REPUBLISH_INTERVAL = 1 * 60 * 60 * 1000 // Republish every hour (according to BEP44 recommendations)
const LOOP_INTERVAL = 360 // Target ~10k per hour!
const MAX_CONCURRENCY = 360 // allow enough concurrency (20x the default)

export class Republisher {
  /**
   * @param {Array<Record>} records
   */
  constructor (records) {
    this.dht = new DHT({ concurrency: MAX_CONCURRENCY })
    this.records = records

    this.sampled = new Set()

    this.requests = 0
    this.start = Date.now()

    this.loopInterval = setInterval(this.tick.bind(this), LOOP_INTERVAL)
    this.republishInterval = setInterval(() => this.sampled.clear(), REPUBLISH_INTERVAL)
  }

  /**
   * @param {Array<Record>} records
   */
  static start (records) {
    return new Republisher(records)
  }

  async tick () {
    const record = this.sample()
    if (!record) return

    const key = record.key
    const z32Key = 'pk:' + z32.encode(key)

    try {
      const resolved = await this.dht.get(key)
        .catch(noop)

      // Assume there are other republisher (according to BEP44 recommendations)
      if ((resolved?.nodes?.length || 0) < 8) {
        // Discovered a more recent version
        if (resolved && resolved.seq > (record.seq || -1)) {
          record.v = resolved.v
          record.seq = resolved.seq
          record.sig = resolved.sig
        }

        if (!record.v || !record.sig) {
          throw new Error("Couldn't resolve a record to republish!")
        }

        await this.dht.put(key, record)
      }

      this.requests += 1
      log(z32Key, 'requests', this.requests, 'rate', this.rate())
    } catch (error) {
      log('ERROR', z32Key, error)
    }
  }

  rate () {
    const elapsed = (Date.now() - this.start) / 3600000
    return Math.floor(this.requests / elapsed)
  }

  /**
   * @returns {Record | undefined}
   */
  sample () {
    const valid = this.records.filter(r => !this.sampled.has(r.key.toString('hex')))

    const randomIndex = Math.floor(Math.random() * valid.length)
    const record = valid[randomIndex]

    if (record) { this.sampled.add(record.key.toString('hex')) }

    return record
  }

  destroy () {
    clearInterval(this.loopInterval)
    clearInterval(this.republishInterval)
    this.dht.destroy()
  }
}

function log (...args) {
  console.log(now(), ...args)
}

function now () {
  return new Date().toLocaleString('en-GB', { timeZone: 'Asia/Istanbul' }).replace(',', '')
}

function noop () { }

export default Republisher

/**
 * @typedef {{key:Uint8Array, v?:Uint8Array, seq?:number, sig?: Uint8Array}} Record
 */
