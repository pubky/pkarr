import lz4 from 'lz4'
import brotli from 'brotli-compress'
import assert from 'assert'
import bencode from 'bencode'
import * as ipCodec from '@leichtgewicht/ip-codec'

// https://www.lucidchart.com/techblog/2019/12/06/json-compression-alternative-binary-formats-and-compression-methods/

const TYPICAL = `
@,100,IN,A,104.16.132.229
fedimint.@,97,IN,AAAA,2606::4700::6810::84e5
router.@,97,IN,AAAA,2606:2800:220:1:248:1893:25c8:19
@,97,IN,CNAME,nuh.dev
_atproto.@,300,IN,TXT,"did=did:plc:zki2d3v7nckil3azsodv72sy"
_matrix.@,1080,IN,TXT,"handle=@nuhvi:matrix.org"
_nostr.@,2000,IN,TXT,"nprofile1qqsfxrxw7y3h9hf0zczhelz57rdajse4mz63kn38xu3kkqx2kuv0ekgpzemhxue69uhhyetvv9ujumn0wd68ytnzv9hxgme06qp"
`

const csv = TYPICAL

const records = csvToRecords(csv)

const BASE = csv.length

const results = {}

const jsonEncoded = JSON.stringify(records)

// Brotli JSON
{
  const compressStart = Date.now()
  const compressed = lz4.encode(jsonEncoded)
  const compressEnd = Date.now()

  const decompressStart = Date.now()
  const decompressed = JSON.parse(lz4.decode(compressed).toString())
  const decompressEnd = Date.now()

  assert(decompressed.length === records.length)

  results['json + brotli'] = {
    sizeRatio: sizeRatio(compressed),
    compressTime: compressEnd - compressStart,
    decompressTime: decompressEnd - decompressStart
  }
}

const bencoded = bencode.encode(records)

// LZ4 bencode
{
  const compressStart = Date.now()
  const compressed = lz4.encode(bencoded)
  const compressEnd = Date.now()

  const decompressStart = Date.now()
  const decompressed = bencode.decode(Buffer.from(lz4.decode(compressed))).map(r => r.map(c => {
    try {
      return Buffer.from(c).toString()
    } catch (error) {
      return c
    }
  }))
  const decompressEnd = Date.now()

  assert(decompressed.length === records.length)

  results['bencode + lz4'] = {
    sizeRatio: sizeRatio(compressed),
    compressTime: compressEnd - compressStart,
    decompressTime: decompressEnd - decompressStart
  }
}

// LZ4 CSV
{
  const compressStart = Date.now()
  const compressed = lz4.encode(csv)
  const compressEnd = Date.now()

  const decompressStart = Date.now()
  const decompressed = csvToRecords(lz4.decode(compressed).toString())
  const decompressEnd = Date.now()

  assert(decompressed.length === records.length)

  results['csv + lz4'] = {
    sizeRatio: sizeRatio(compressed),
    compressTime: compressEnd - compressStart,
    decompressTime: decompressEnd - decompressStart
  }
}

// Brotli bencode
{
  const compressStart = Date.now()
  const compressed = await brotli.compress(bencoded)
  const compressEnd = Date.now()

  const decompressStart = Date.now()
  const decompressed = bencode.decode(await brotli.decompress(compressed)).map(r => r.map(c => {
    try {
      return Buffer.from(c).toString()
    } catch (error) {
      return c
    }
  }))
  const decompressEnd = Date.now()

  assert(decompressed.length === records.length)

  results['bencode + brotli'] = {
    sizeRatio: sizeRatio(compressed),
    compressTime: compressEnd - compressStart,
    decompressTime: decompressEnd - decompressStart
  }
}

// Brotli CSV
{
  const compressStart = Date.now()
  const compressed = await brotli.compress(Buffer.from(csv))
  const compressEnd = Date.now()

  const decompressStart = Date.now()
  const decompressed = csvToRecords(Buffer.from((await brotli.decompress(compressed))).toString())
  const decompressEnd = Date.now()

  assert(decompressed.length === records.length)

  results['csv + brotli'] = {
    sizeRatio: sizeRatio(compressed),
    compressTime: compressEnd - compressStart,
    decompressTime: decompressEnd - decompressStart
  }
}

// Brotli + custom
{
  const compressStart = Date.now()
  const compressed = await brotli.compress(encode(records))
  const compressEnd = Date.now()

  const decompressStart = Date.now()
  const decompressed = decode(await brotli.decompress(compressed))
  const decompressEnd = Date.now()

  assert(decompressed.length === records.length)

  results['custom + brotli'] = {
    sizeRatio: sizeRatio(compressed),
    compressTime: compressEnd - compressStart,
    decompressTime: decompressEnd - decompressStart
  }
}

// custom without compression
{
  const compressStart = Date.now()
  const compressed = encode(records)
  const compressEnd = Date.now()

  const decompressStart = Date.now()
  const decompressed = decode(compressed)
  const decompressEnd = Date.now()

  assert(decompressed.length === records.length)

  results['custom + no comp'] = {
    sizeRatio: sizeRatio(compressed),
    compressTime: compressEnd - compressStart,
    decompressTime: decompressEnd - decompressStart
  }
}

function encode (records) {
  const buffer = Buffer.alloc(1000)

  let offset = 0

  for (const record of records) {
    const name = record[0]
    const type = record[1]
    const rdata = record[2]

    const slice = buffer.subarray(offset)

    slice.writeUint16BE((() => {
      switch (type) {
        case 'A':
          return 1
        case 'NS':
          return 2
        case 'CNAME':
          return 5
        case 'MX':
          return 15
        case 'AAAA':
          return 28
        case 'TXT':
          return 16
        default:
          return 255
      }
    })())

    slice[2] = name.length
    slice.write(name, 3)

    const rdataOfsset = 3 + name.length
    offset += rdataOfsset

    switch (type) {
      case 'A':
        slice.set(ipCodec.v4.encode(rdata), rdataOfsset)
        offset += 4
        break
      case 'AAAA':
        slice.set(ipCodec.v6.encode(rdata), rdataOfsset)
        offset += 16
        break
      case 'TXT':
        const txt = Buffer.from(rdata.slice(1, rdata.length - 1))
        slice.set([txt.length], rdataOfsset)
        slice.set(txt, rdataOfsset + 1)
        offset += txt.length + 1
        break
      default:
        const any = Buffer.from(rdata)
        slice.set([any.length], rdataOfsset)
        slice.set(any, rdataOfsset + 1)
        offset += any.length + 1
        break
    }
  }

  return buffer.subarray(0, offset)
}

function decode (buffer) {
  buffer = Buffer.from(buffer)

  const result = []

  let row = {}
  let offset = 0

  while (true) {
    const slice = buffer.subarray(offset)

    row.type = ((type) => {
      switch (type) {
        case 1:
          return 'A'
        case 2:
          return 'NS'
        case 5:
          return 'CNAME'
        case 15:
          return 'MX'
        case 28:
          return 'AAAA'
        case 16:
          return 'TXT'
        default:
          return 'ANY'
      }
    })(slice.readUint16BE())

    row.name = slice.subarray(3, 3 + slice[2]).toString()

    const rdataOffset = 3 + row.name.length
    offset += rdataOffset

    row.data = (() => {
      switch (row.type) {
        case ('A'):
          offset += 4
          return ipCodec.v4.decode(slice.subarray(rdataOffset, rdataOffset + 4))
        case ('AAAA'):
          offset += 16
          return ipCodec.v6.decode(slice.subarray(rdataOffset, rdataOffset + 16))
        default:
          const length = slice[rdataOffset]
          offset += length + 1
          return slice.subarray(rdataOffset + 1, rdataOffset + 1 + length).toString()
      }
    })()

    result.push(row)
    row = {}

    if (offset >= buffer.length) break
  }

  return result
}

function sizeRatio (compressed) {
  return compressed.length / BASE
}

function csvToRecords (csv) {
  const records = csv
    .split(/\n/g).filter(Boolean)
    .map(
      row => {
        let _row = row.split(/,/g).filter(Boolean)
          .map(s => {
            const n = Number(s)
            return Number.isNaN(n) ? s : n
          })

        _row[1] = _row[2] = null
        _row = _row.filter(Boolean)

        return _row
      }
    )

  return records
}

console.log(' ┌--------------------┬------------┬----------┬------------┐')
console.log(' | TYPE               | SIZE RATIO | COMPRESS | DECOMPRESS |')
console.log(' |--------------------|------------|----------|------------|')
for (const [name, row] of Object.entries(results)) {
  console.log(' | ' + name.padEnd(18) + ' | ' + row.sizeRatio.toFixed(2).padEnd(10) + ' | ' + (row.compressTime + ' ms').padEnd(8) + ' | ' + (row.decompressTime + ' ms').padEnd(10) + ' |')
}
console.log(' └--------------------┴------------┴----------┴------------┘')
