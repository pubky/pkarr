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
  const compressed = await brotli.compress(encode(csv))
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

  function encode (string) {
    const result = string.split(/\n/g).filter(Boolean).map(row => {
      const parts = row.split(',')
      const name = parts[0]
      const type = parts[3]
      const data = parts[4]

      return Buffer.from([
        ...(() => {
          switch (type) {
            case 'A':
              return Buffer.from([0, 1])
            case 'NS':
              return Buffer.from([0, 2])
            case 'CNAME':
              return Buffer.from([0, 5])
            case 'MX':
              return Buffer.from([0, 15])
            case 'AAAA':
              return Buffer.from([0, 28])
            case 'TXT':
              return Buffer.from([0, 16])
            default:
              return Buffer.from([0, 255])
          }
        })(),
        ...Buffer.from(name),
        ...Buffer.from(' '),
        ...(() => {
          switch (type) {
            case 'A':
              return ipCodec.v4.encode(data)
            case 'AAAA':
              return ipCodec.v6.encode(data)
            case 'TXT':
              return Buffer.from(data.slice(1, data.length - 2))
            default:
              return Buffer.from(data)
          }
        })(),
        ...Buffer.from('\n')
      ])
    })
    return Buffer.concat(result)
  }

  function decode (buf) {
    const result = []

    let row = []

    for (const x of buf) {
      if (x === Buffer.from('\n')[0]) {
        const type = row[1]
        const rest = Buffer.from(row).subarray(2)
        let name = []

        for (const d of rest) {
          if (d === Buffer.from(' ')[0]) {
            break
          }
          name.push(d)
        }

        name = Buffer.from(name).toString()

        const _row = {
          type: (() => {
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
          })(),
          name: name.replace(/@$/, 'o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy.'),
          data: (() => {
            const _data = rest.subarray(name.length + 1)

            switch (type) {
              case 1:
                return ipCodec.v4.decode(_data)
              case 28:
                return ipCodec.v6.decode(_data)
              default:
                return _data
            }
          })()
        }

        result.push(_row)
        row = []
      } else {
        row.push(x)
      }
    }

    return result
  }
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
