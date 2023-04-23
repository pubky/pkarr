import fs from 'fs'
import Republisher from '../../lib/republisher.js'

const v = Buffer.from('000b0c805b5b225f74657374222c227374696c6c20616c697665225d5d03', 'hex')
const seq = 1681656800

const records = fs.readFileSync('./data/users.csv')
  .toString()
  .split(/\n/)
  .filter(Boolean)
  .map(r => {
    const [key, sig] = r.split(',').map(e => Buffer.from(e, 'hex'))
    return { key, v, seq, sig }
  })

Republisher.start(records)
