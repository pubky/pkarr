import pkarr from 'pkarr';
import { createMutable } from 'solid-js/store';
import b4a from 'b4a'
import z32 from 'z32'

const DEFAULT_SERVERS = [
  'pkarr1.nuhvi.com'
]

const store = createMutable({
  seed: window.localStorage.getItem('seed'),
  records: [],
  servers: DEFAULT_SERVERS,
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
  updateRecord(e, rowIndex, inputIndex) {
    this.records[rowIndex][inputIndex] = e.target.value;
    const updated = [...this.records].map(x => [...x])
    const stringified = JSON.stringify(updated)
    pkarr.codec.encode(stringified).then(bytes => {
      this.recordsSize = bytes.byteLength || 0
    })

    localStorage.setItem('records', stringified)
  },

  publish() {
    const keyPair = { ...this.keyPair }
    const records = store.records.map(r => [...r])
    const servers = [...this.servers]

    this.publishing = true

    pkarr.put(keyPair, records, servers)
      .then(result => {
        if (result.ok) {
          this.publishing = false;
          this.lastPublished = new Date(result.request.seq * 1000).toLocaleString()
        } else {
          alert(
            'Error publishing to all servers:\n' +
            result.errors.map(e => e.server + ": " + e.error.message).join('\n')
          )
        }
      })
  },

  load() {
    {
      // Seeed
      const string = this.seed;
      const seed = string ? b4a.from(string, 'hex') : pkarr.randomBytes()

      this.seed = string || b4a.toString(seed, 'hex')
      window.localStorage.setItem('seed', this.seed)

      // keyPair
      this.keyPair = pkarr.generateKeyPair(seed)
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
  },
})

store.load()

export default store
