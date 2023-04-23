import DHT from './dht.js'

const REPUBLISH_INTERVAL = 1 * 60 * 60 * 1000
const MAX_CONCURRENCY = 320
const LOOP_INTERVAL = 360

export class Republisher {
  /**
   * @param {Array<Record>} records
   */
  constructor (records) {
    this.dht = new DHT({ concurrency: MAX_CONCURRENCY })
    this.records = records
    this.requests = 0
    this.start = Date.now()
    this.interval = setInterval(this.tick.bind(this), LOOP_INTERVAL)
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
    const keyhex = key.toString('hex')

    try {
      const resolved = await this.dht.get(key)
        .catch(noop)

      // Assume there are other republisher (according to BEP44 recommendations)
      if ((resolved?.nodes?.length || 0) < 8) {
        // Discovered a more recent version
        if (resolved && resolved.seq > record.seq) {
          record.v = resolved.v
          record.seq = resolved.seq
          record.sig = resolved.sig
        }

        await this.dht.put(key, record)
      }

      record.last = Date.now()

      this.requests += 1
      log(keyhex, 'requests', this.requests, 'rate', this.rate())
    } catch (error) {
      log('ERROR', keyhex, error)
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
    const valid = this.records.filter(r => !r.last || (Date.now() - r.last) > REPUBLISH_INTERVAL)

    const randomIndex = Math.floor(Math.random() * valid.length)
    return valid[randomIndex]
  }

  destroy () {
    clearInterval(this.interval)
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

/**
 * @typedef {{key:Uint8Array, v?:Uint8Array, seq?:number, sig?: Uint8Array, last?: number}} Record
 */
