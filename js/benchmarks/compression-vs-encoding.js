import lz4 from 'lz4'
import brotli from 'brotli-compress'
import assert from 'assert'
import bencode from 'bencode'

// https://www.lucidchart.com/techblog/2019/12/06/json-compression-alternative-binary-formats-and-compression-methods/

const csv = `
cloudflare.com.,300,IN,CAA,0 issuewild "digicert.com; cansignhttpexchanges=yes"
cloudflare.com.,146,IN,A,104.16.132.229
cloudflare.com.,146,IN,A,104.16.133.229
cloudflare.com.,97,IN,AAAA,2606:4700::6810:85e5
cloudflare.com.,97,IN,AAAA,2606:4700::6810:84e5
cloudflare.com.,44965,IN,NS,ns5.cloudflare.com.
cloudflare.com.,44965,IN,NS,ns7.cloudflare.com.
cloudflare.com.,44965,IN,NS,ns6.cloudflare.com.
cloudflare.com.,44965,IN,NS,ns4.cloudflare.com.
cloudflare.com.,44965,IN,NS,ns3.cloudflare.com.
cloudflare.com.,1800,IN,MX,5 mailstream-canary.mxrecord.io.
cloudflare.com.,1800,IN,MX,20 mailstream-central.mxrecord.mx.
cloudflare.com.,1800,IN,MX,10 mailstream-east.mxrecord.io.
cloudflare.com.,1800,IN,MX,10 mailstream-west.mxrecord.io.
cloudflare.com.,300,IN,TXT,"google-site-verification=ZdlQZLBBAPkxeFTCM1rpiB_ibtGff_JF5KllNKwDR9I"
cloudflare.com.,300,IN,TXT,"ZOOM_verify_7LFBvOO9SIigypFG2xRlMA"
cloudflare.com.,300,IN,TXT,"docker-verification=c578e21c-34fb-4474-9b90-d55ee4cba10c"
cloudflare.com.,300,IN,TXT,"atlassian-domain-verification=WxxKyN9aLnjEsoOjUYI6T0bb5vcqmKzaIkC9Rx2QkNb751G3LL/cus8/ZDOgh8xB"
cloudflare.com.,300,IN,TXT,"miro-verification=bdd7dfa0a49adfb43ad6ddfaf797633246c07356"
cloudflare.com.,300,IN,TXT,"google-site-verification=C7thfNeXVahkVhniiqTI1iSVnElKR_kBBtnEHkeGDlo"
cloudflare.com.,300,IN,TXT,"status-page-domain-verification=r14frwljwbxs"
cloudflare.com.,300,IN,TXT,"drift-domain-verification=f037808a26ae8b25bc13b1f1f2b4c3e0f78c03e67f24cefdd4ec520efa8e719f"
cloudflare.com.,300,IN,TXT,"apple-domain-verification=DNnWJoArJobFJKhJ"
cloudflare.com.,300,IN,TXT,"logmein-verification-code=b3433c86-3823-4808-8a7e-58042469f654"
cloudflare.com.,300,IN,TXT,"stripe-verification=5096d01ff2cf194285dd51cae18f24fa9c26dc928cebac3636d462b4c6925623"
cloudflare.com.,300,IN,TXT,"MS=ms70274184"
cloudflare.com.,300,IN,TXT,"facebook-domain-verification=h9mm6zopj6p2po54woa16m5bskm6oo"
cloudflare.com.,300,IN,TXT,"v=spf1 ip4:199.15.212.0/22 ip4:173.245.48.0/20 include:_spf.google.com include:spf1.mcsv.net include:spf.mandrillapp.com include:mail.zendesk.com include:stspg-customer.com include:_spf.salesforce.com -all"
cloudflare.com.,300,IN,TXT,"cisco-ci-domain-verification=27e926884619804ef987ae4aa1c4168f6b152ada84f4c8bfc74eb2bd2912ad72"
cloudflare.com.,300,IN,TXT,"stripe-verification=bf1a94e6b16ace2502a4a7fff574a25c8a45291054960c883c59be39d1788db9"
cloudflare.com.,300,IN,CAA,0 issue "digicert.com; cansignhttpexchanges=yes"
cloudflare.com.,300,IN,CAA,0 issuewild "digicert.com; cansignhttpexchanges=yes"
cloudflare.com.,300,IN,CAA,0 issue "letsencrypt.org"
cloudflare.com.,300,IN,CAA,0 issuewild "comodoca.com"
cloudflare.com.,300,IN,CAA,0 issue "comodoca.com"
cloudflare.com.,300,IN,CAA,0 iodef "mailto:tls-abuse@cloudflare.com"
cloudflare.com.,300,IN,CAA,0 issuewild "letsencrypt.org"
cloudflare.com.,300,IN,CAA,0 issuewild "pki.goog; cansignhttpexchanges=yes"
cloudflare.com.,300,IN,CAA,0 issue "pki.goog; cansignhttpexchanges=yes"
`
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

{
  // Brotli bencode
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

  function encode(string) {
    const result = string.split(/\n/g).filter(Boolean).map(row => {
      const parts = row.split(',')
      const name = parts[0]
      const type = parts[3]
      const data = parts[4]

      return Buffer.from([
        (() => {
          switch (type) {
            case 'A':
              return 1
            case 'NS':
              return 2
            case 'MX':
              return 15
            case 'AAAA':
              return 28
            case 'TXT':
              return 16
            default:
              return 255 // ANY
          }
        })(),
        ...Buffer.from(name),
        ...Buffer.from(' '),
        ...Buffer.from(data),
        ...Buffer.from('\n')
      ])
    })
    return Buffer.concat(result)
  }

  function decode(buf) {
    const result = []

    let row = []

    for (let x of buf) {
      if (x === Buffer.from('\n')[0]) {
        const type = row[0]
        const string = Buffer.from(row).toString().slice(1)
        const name = string.split(' ')[0]

        result.push({
          type: (() => {
            switch (type) {
              case 1:
                return 'A'
              case 2:
                return 'NS'
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
          name,
          data: string.slice(name.length + 1)
        })
        row = []
      } else {
        row.push(x)
      }
    }

    return result
  }
}

function sizeRatio(compressed) {
  return compressed.length / BASE
}

function csvToRecords(csv) {
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
