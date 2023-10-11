import { Pkarr, z32, dns, SignedPacket } from 'pkarr';
import { createMutable } from 'solid-js/store';

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
    this.records = [...this.records, {}]
  },
  /**
   * @param {import('dns-packet').Answer[]} records
   */
  updateRecords(records) {
    const packet = {
      id: 0,
      type: 'response',
      answers: records.map(rr => ({ name: "", ttl: 30, class: "IN", type: "TXT", data: "", ...rr }))
    }
    const encodedPacket = dns.encode(packet)

    this.recordsSize = encodedPacket.length || 0

    localStorage.setItem('records', JSON.stringify(records))
  },

  publish() {
    const keyPair = { ...this.keyPair }
    const records = store.records.map(r => ({ ...r }))
    const relays = [...this.relays]

    this.publishing = true

    const packet = {
      id: 0,
      type: "response",
      answers: records.map(rr => ({ name: "", ttl: 30, class: "IN", type: "TXT", data: "", ...rr }))
    }

    const signedPacket = SignedPacket.fromPacket(keyPair, packet)

    const start = Date.now()
    Pkarr.relayPut(relays[0], signedPacket)
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

  resolved: [{}],
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
    Pkarr.relayGet(relays[0], key)
      .then(signedPacket => {
        console.log({ signedPacket })
        if (signedPacket) {
          this.resolved = signedPacket.packet().answers
            .map(rr => {
              let denormalizedName = rr.name.replace(z32.encode(key), '');

              if (denormalizedName.length === 0) {
                denormalizedName = '@'
              }
              if (denormalizedName.endsWith('.')) {
                denormalizedName = denormalizedName.slice(0, -1)
              }

              return {
                ...rr, name: denormalizedName
              }
            })

          this.resolvedLastPublished = new Date(signedPacket.timestamp() / 1000).toLocaleString()
          this.resolving = false;

          this.resolvedSize = signedPacket.size()

          const time = Date.now() - start
          this.temporaryMessage = "Resolved ... took " + time + " ms"
          setTimeout(() => this.temporaryMessage = null, 2000)
        } else {
          alert('No records founds from any relay')
        }
      })
      .catch(error => {
        alert("Error: " + error.message)
      })
  },
  updateSettings(seed, relays) {
    if (this.seed !== seed) {
      this.records = [{}]
      localStorage.setItem('records', JSON.stringify(this.records))
    }

    this.seed = seed
    this.keyPair = Pkarr.generateKeyPair(Buffer.from(seed, 'hex'))
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
      const seed = string ? Buffer.from(string, 'hex') : Pkarr.generateSeed()

      this.seed = string || Buffer.from(seed).toString('hex')
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
          this.updateRecords(this.records)
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
