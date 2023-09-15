import Pkarr from 'pkarr/relayed.js';
import { createMutable } from 'solid-js/store';
import b4a from 'b4a'
import z32 from 'z32'

const DEFAULT_RELAYS = [
  'https://relay.pkarr.org'
]

const store = createMutable({
  seed: window.localStorage.getItem('seed'),
  records: [],
  relays: DEFAULT_RELAYS,
  lastPublished: 'Not published yet',

  keyPair: null,
  pk() {
    if (!this.keyPair) return "...missing publicKey"
    return 'pk:' + z32.encode(this.keyPair.publicKey)
  },

  records: [[]],
  recordsSize: 0,
  publishing: false,

  addRecord() {
    this.records = [...this.records, []]
  },
  updateRecords(records) {
    this.records = records
    const stringified = JSON.stringify(records)

    Pkarr.codec.encode(stringified).then(bytes => {
      this.recordsSize = bytes.byteLength || 0
    })

    localStorage.setItem('records', stringified)
  },

  publish() {
    const keyPair = { ...this.keyPair }
    const records = store.records.map(r => [...r])
    const relays = [...this.relays]

    this.publishing = true

    const start = Date.now()
    Pkarr.publish(keyPair, records, relays)
      .then(published => {
        if (published) {
          this.publishing = false;
          this.lastPublished = new Date().toLocaleString()
          localStorage.setItem('lastPublished', this.lastPublished)

          const time = Date.now() - start
          this.temporaryMessage = "Published ... took " + time + " ms"
          setTimeout(() => this.temporaryMessage = null, 2000)
        } else {
          alert('Error publishing to all relays')
        }
      })
  },

  resolved: [[]],
  resolving: false,
  resolvedSize: 0,
  resolvedLastPublished: 'Not resolved yet...',
  temporaryMessage: null,
  resolve(target) {
    let key = target.replace('pk:', '')
    try {
      key = z32.decode(key)
    } catch (error) {
      console.log("error: can't decode key", key)
      return
    }

    this.resolving = true

    const relays = [...this.relays]

    const start = Date.now()
    Pkarr.resolve(key, relays)
      .then(result => {
        if (result) {
          this.resolved = result.records
          this.resolvedLastPublished = new Date(result.seq * 1000).toLocaleString()
          this.resolving = false;

          Pkarr.codec.encode(result.records).then(bytes => {
            this.resolvedSize = bytes.byteLength || 0
          })

          const time = Date.now() - start
          this.temporaryMessage = "Resolved ... took " + time + " ms"
          setTimeout(() => this.temporaryMessage = null, 2000)
        } else {
          alert('No records founds from any relay')
        }
      })
  },
  updateSettings(seed, relays) {
    if (this.seed !== seed) {
      this.records = [[]]
      localStorage.setItem('records', JSON.stringify(this.records))
    }

    this.seed = seed
    this.keyPair = Pkarr.generateKeyPair(b4a.from(seed, 'hex'))
    this.relays = relays

    localStorage.setItem('seed', seed)
    localStorage.setItem('relays', JSON.stringify(relays))
  },
  resetRelays() {
    this.relays = DEFAULT_RELAYS
  },

  load() {
    {
      // Seeed
      const string = this.seed;
      const seed = string ? b4a.from(string, 'hex') : Pkarr.generateSeed()

      this.seed = string || b4a.toString(seed, 'hex')
      localStorage.setItem('seed', this.seed)

      // keyPair
      this.keyPair = Pkarr.generateKeyPair(seed)
    }
    {
      // Relays
      const relays = localStorage.getItem('relays')
      try {
        this.relays = JSON.parse(relays) || DEFAULT_RELAYS
      } catch { }
    }
    {
      // Records
      const records = localStorage.getItem('records')
      if (records) {
        try {
          this.records = JSON.parse(records)
        } catch { }
      }
    }
    {
      // Last published
      this.lastPublished = localStorage.getItem('lastPublished')
    }
  },
})

store.load()

export default store
