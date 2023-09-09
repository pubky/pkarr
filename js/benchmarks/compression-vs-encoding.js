import lz4 from 'lz4'
import brotli from 'brotli-compress'
import assert from 'assert'
import bencode from 'bencode'
import * as ipCodec from '@leichtgewicht/ip-codec'

// https://www.lucidchart.com/techblog/2019/12/06/json-compression-alternative-binary-formats-and-compression-methods/

const ITERATIONS = 2 // brotli-compress seems to have a memory leak issue so can't make too many iterations

// Fewer records should have lower encoding overhead in JSON and bencode
// IP should have more compact size without compression, while TXT favors brotli
const vectors = [
  {
    name: 'single record - IP',
    records: [
      ['router.@', 'AAAA', '2606:2800:220:1:248:1893:25c8:19']
    ]
  },
  {
    name: 'single record - TXT',
    records: [
      ['_pkarr.@', 'TXT', 'proxies=proxy.nuh.dev,proxy2.pkarr.org;image=https://example.com;stripe-verification=5096d01ff2cf194285dd51cae18f24fa9c26dc928cebac3636d462b4c6925623']
    ]
  },
  {
    name: 'multiple records - TXT - compact text',
    records: [
      ['_matrix.@', 'TXT', 'handle=@nuhvi:matrix.org'],
      ['_nostr.@', 'TXT', 'nprofile1qqsfxrxw7y3h9hf0zczhelz57rdajse4mz63kn38xu3kkqx2kuv0ekgpzemhxue69uhhyetvv9ujumn0wd68ytnzv9hxgme06qp'],
      ['_atproto.@', 'TXT', 'did=did:plc:zki2d3v7nckil3azsodv72sy']
    ]
  },
  {
    name: 'medium - mix of A, AAAA, CNAME and TXT',
    records: [
      ['@', 'A', '104.16.132.229'],
      ['fedimint.@', 'AAAA', '2606::4700::6810::84e5'],
      ['router.@', 'AAAA', '2606:2800:220:1:248:1893:25c8:19'],
      ['@', 'CNAME', 'nuh.dev'],
      ['pkarr.@', 'CNAME', 'pkarr.org'],
      ['_atproto.@', 'TXT', 'did=did:plc:zki2d3v7nckil3azsodv72sy'],
      ['_matrix.@', 'TXT', 'handle=@nuhvi:matrix.org'],
      ['_nostr.@', 'TXT', 'nprofile1qqsfxrxw7y3h9hf0zczhelz57rdajse4mz63kn38xu3kkqx2kuv0ekgpzemhxue69uhhyetvv9ujumn0wd68ytnzv9hxgme06qp']
    ]
  },
  {
    name: 'multiple records - A, AAA',
    records: [
      ['@', 'A', '104.16.132.229'],
      ['fedimint.@', 'AAAA', '2606::4700::6810::84e5'],
      ['router.@', 'AAAA', '2606:2800:220:1:248:1893:25c8:19']
    ]
  },
  {
    name: 'large - text heavy',
    records: [
      ['router.@', 'AAAA', '2606:2800:220:1:248:1893:25c8:19'],
      ['@', 'CNAME', 'https://www.something.com'],
      ['nat_server.@', 'CNAME', 'https://0682-85-103-17-207.ngrok-free.app/'],
      ['_image.@', 'TXT', 'https://metadata.example.domain/7f1887a8bf19b14fc0df6fd9b2acc9af147ea85/41b1a0649752af1b28b3dc29a1556eee781e4a4c3a1f7f53f90fa834de098c4d/image'],
      ['_name.@', 'TXT', 'Foobarstuff'],
      ['_description.@', 'TXT', 'Doing stuff, worked on stuff, shipped stuff, I am really cool!'],
      ['_nodes.@', 'TXT', 'node1=104.16.132.229,node2=104.16.229.229']
    ]
  }
]

for (const { name, records } of vectors) {
  const results = {}
  // The simplest implementation
  const baseSize = encodeCSV(records).length

  await test('csv - no comp', () => {
    const encodeStart = Date.now()
    const encoded = encodeCSV(records)
    const encodeEnd = Date.now()

    const decodeStart = Date.now()
    const decoded = decodeCSV(encoded)
    const decodeEnd = Date.now()

    assert(decoded.length === records.length)

    return {
      size: encoded.length,
      encodeTime: encodeEnd - encodeStart,
      decodeTime: decodeEnd - decodeStart
    }
  })

  await test('bencode - no comp', () => {
    const encodeStart = Date.now()
    const encoded = bencode.encode(records)
    const encodeEnd = Date.now()

    const decodeStart = Date.now()
    const decoded = bencode.decode(encoded)
    const decodeEnd = Date.now()

    assert(decoded.length === records.length)

    return {
      size: encoded.length,
      encodeTime: encodeEnd - encodeStart,
      decodeTime: decodeEnd - decodeStart
    }
  })

  // lz4 + JSON
  await test('json + lz4', () => {
    const encodeStart = Date.now()
    const encoded = lz4.encode(JSON.stringify(records))
    const encodeEnd = Date.now()

    const decodeStart = Date.now()
    const decoded = JSON.parse(lz4.decode(encoded).toString())
    const decodeEnd = Date.now()

    assert(decoded.length === records.length)

    return {
      size: encoded.length,
      encodeTime: encodeEnd - encodeStart,
      decodeTime: decodeEnd - decodeStart
    }
  })

  // LZ4 bencode
  await test('bencode + lz4', () => {
    const encodeStart = Date.now()
    const encoded = lz4.encode(bencode.encode(records))
    const encodeEnd = Date.now()

    const decodeStart = Date.now()
    const decoded = bencode.decode(Buffer.from(lz4.decode(encoded))).map(r => r.map(c => {
      try {
        return Buffer.from(c).toString()
      } catch (error) {
        return c
      }
    }))
    const decodeEnd = Date.now()

    assert(decoded.length === records.length)

    return {
      size: encoded.length,
      encodeTime: encodeEnd - encodeStart,
      decodeTime: decodeEnd - decodeStart
    }
  })

  // // LZ4 CSV
  await test('csv + lz4', () => {
    const encodeStart = Date.now()
    const encoded = lz4.encode(encodeCSV(records))
    const encodeEnd = Date.now()

    const decodeStart = Date.now()
    const decoded = decodeCSV(lz4.decode(encoded).toString())
    const decodeEnd = Date.now()

    assert(decoded.length === records.length)

    return {
      size: encoded.length,
      encodeTime: encodeEnd - encodeStart,
      decodeTime: decodeEnd - decodeStart
    }
  })

  // brotli + JSON
  await test('json + brotli', async () => {
    const encodeStart = Date.now()
    const encoded = await brotli.compress(Buffer.from(JSON.stringify(records)))
    const encodeEnd = Date.now()

    const decodeStart = Date.now()
    const decoded = JSON.parse(Buffer.from(await brotli.decompress(encoded)).toString())
    const decodeEnd = Date.now()

    assert(decoded.length === records.length)

    return {
      size: encoded.length,
      encodeTime: encodeEnd - encodeStart,
      decodeTime: decodeEnd - decodeStart
    }
  })

  // Brotli bencode
  await test('bencode + brotli', async () => {
    const encodeStart = Date.now()
    const encoded = await brotli.compress(bencode.encode(records))
    const encodeEnd = Date.now()

    const decodeStart = Date.now()
    const decoded = bencode.decode(await brotli.decompress(encoded)).map(r => r.map(c => {
      try {
        return Buffer.from(c).toString()
      } catch (error) {
        return c
      }
    }))
    const decodeEnd = Date.now()

    assert(decoded.length === records.length)

    return {
      size: encoded.length,
      encodeTime: encodeEnd - encodeStart,
      decodeTime: decodeEnd - decodeStart
    }
  })

  // // Brotli CSV
  await test('csv + brotli', async () => {
    const encodeStart = Date.now()
    const encoded = await brotli.compress(encodeCSV(records))
    const encodeEnd = Date.now()

    const decodeStart = Date.now()
    const decoded = decodeCSV(await brotli.decompress(encoded))
    const decodeEnd = Date.now()

    assert(decoded.length === records.length)

    return {
      size: encoded.length,
      encodeTime: encodeEnd - encodeStart,
      decodeTime: decodeEnd - decodeStart
    }
  })

  // Brotli + custom
  await test('custom + brotli', async () => {
    const encodeStart = Date.now()
    const encoded = await brotli.compress(encodeCustom(records))
    const encodeEnd = Date.now()

    const decodeStart = Date.now()
    const decoded = decodeCustom(await brotli.decompress(encoded))
    const decodeEnd = Date.now()

    assert(decoded.length === records.length)

    return {
      size: encoded.length,
      encodeTime: encodeEnd - encodeStart,
      decodeTime: decodeEnd - decodeStart
    }
  })

  // custom without compression
  await test('custom + no comp', () => {
    const encodeStart = Date.now()
    const encoded = encodeCustom(records)
    const encodeEnd = Date.now()

    const decodeStart = Date.now()
    const decoded = decodeCustom(encoded)
    const decodeEnd = Date.now()

    assert(decoded.length === records.length)

    return {
      size: encoded.length,
      encodeTime: encodeEnd - encodeStart,
      decodeTime: decodeEnd - decodeStart
    }
  })

  function encodeCustom (records) {
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
        default:
          // TXT or anything else
          const any = Buffer.from(rdata)
          slice.set([any.length], rdataOfsset)
          slice.set(any, rdataOfsset + 1)
          offset += any.length + 1
          break
      }
    }
    return buffer.subarray(0, offset)
  }

  function decodeCustom (buffer) {
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

  function encodeCSV (records) {
    const rows = records.map(r => r.join(','))
    return Buffer.from(rows.join('\n'))
  }
  function decodeCSV (buffer) {
    return Buffer.from(buffer).toString().split(/\n/g).filter(Boolean).map(r => r.split(','))
  }

  async function test (name, cb) {
    const tries = []

    for (let i = 0; i < ITERATIONS; i++) {
      const { size, encodeTime, decodeTime } = await cb()

      tries.push({ saving: size - baseSize, relativeSaving: ((size - baseSize) / baseSize), encodeTime, decodeTime })
    }

    const sum = tries.reduce((acc, cur) => {
      return {
        saving: acc.saving + cur.saving,
        relativeSaving: acc.relativeSaving + cur.relativeSaving,
        encodeTime: acc.encodeTime + cur.encodeTime,
        decodeTime: acc.decodeTime + cur.decodeTime
      }
    }, { saving: 0, relativeSaving: 0, encodeTime: 0, decodeTime: 0 })

    results[name] = {
      saving: sum.saving / tries.length,
      relativeSaving: sum.relativeSaving / tries.length,
      encodeTime: (sum.encodeTime / tries.length).toFixed(2),
      decodeTime: (sum.decodeTime / tries.length).toFixed(2)
    }
  }

  console.log('\n ===', name, '===')
  console.log(' base size:', baseSize)
  console.log(' ┌--------------------┬----------┬----------┬----------┬------------┐')
  console.log(' | TYPE               | Absolute | Relative | ENCODE   | DECODE     |')
  console.log(' |--------------------|----------|----------|----------|------------|')
  for (const [name, row] of Object.entries(results)) {
    const type = name.padEnd(18)
    const absolute = ((row.saving >= 0 ? '+' : '') + row.saving.toFixed(2)).padEnd(8)
    const relative = ((row.relativeSaving >= 0 ? '+' : '') + row.relativeSaving.toFixed(2) + ' %').padEnd(8)
    const encode = (row.encodeTime + ' ms').padEnd(8)
    const decode = (row.decodeTime + ' ms').padEnd(10)

    console.log(' | ' + type + ' | ' + absolute + ' | ' + relative + ' | ' + encode + ' | ' + decode + ' |')
  }
  console.log(' └--------------------┴----------┴----------┴----------┴------------┘')
}
