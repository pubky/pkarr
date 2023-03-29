import cbor from 'cbor'
import lz4 from 'lz4'
import brotli from 'brotli'
import assert from 'assert'

// https://www.lucidchart.com/techblog/2019/12/06/json-compression-alternative-binary-formats-and-compression-methods/

let records = `
cloudflare.com. 300 IN CAA 0 issuewild "digicert.com\; cansignhttpexchanges=yes"
cloudflare.com.		146 IN A 104.16.132.229
cloudflare.com.		146 IN A 104.16.133.229
cloudflare.com.		97 IN AAAA 2606:4700::6810:85e5
cloudflare.com.		97 IN AAAA 2606:4700::6810:84e5
cloudflare.com.		44965 IN NS ns5.cloudflare.com.
cloudflare.com.		44965 IN NS ns7.cloudflare.com.
cloudflare.com.		44965 IN NS ns6.cloudflare.com.
cloudflare.com.		44965 IN NS ns4.cloudflare.com.
cloudflare.com.		44965 IN NS ns3.cloudflare.com.
cloudflare.com.		1800 IN	MX 5 mailstream-canary.mxrecord.io.
cloudflare.com.		1800 IN	MX 20 mailstream-central.mxrecord.mx.
cloudflare.com.		1800 IN	MX 10 mailstream-east.mxrecord.io.
cloudflare.com.		1800 IN	MX 10 mailstream-west.mxrecord.io.
cloudflare.com.		300 IN TXT "google-site-verification=ZdlQZLBBAPkxeFTCM1rpiB_ibtGff_JF5KllNKwDR9I"
cloudflare.com.		300 IN TXT "ZOOM_verify_7LFBvOO9SIigypFG2xRlMA"
cloudflare.com.		300 IN TXT "docker-verification=c578e21c-34fb-4474-9b90-d55ee4cba10c"
cloudflare.com.		300 IN TXT "atlassian-domain-verification=WxxKyN9aLnjEsoOjUYI6T0bb5vcqmKzaIkC9Rx2QkNb751G3LL/cus8/ZDOgh8xB"
cloudflare.com.		300 IN TXT "miro-verification=bdd7dfa0a49adfb43ad6ddfaf797633246c07356"
cloudflare.com.		300 IN TXT "google-site-verification=C7thfNeXVahkVhniiqTI1iSVnElKR_kBBtnEHkeGDlo"
cloudflare.com.		300 IN TXT "status-page-domain-verification=r14frwljwbxs"
cloudflare.com.		300 IN TXT "drift-domain-verification=f037808a26ae8b25bc13b1f1f2b4c3e0f78c03e67f24cefdd4ec520efa8e719f"
cloudflare.com.		300 IN TXT "apple-domain-verification=DNnWJoArJobFJKhJ"
cloudflare.com.		300 IN TXT "logmein-verification-code=b3433c86-3823-4808-8a7e-58042469f654"
cloudflare.com.		300 IN TXT "stripe-verification=5096d01ff2cf194285dd51cae18f24fa9c26dc928cebac3636d462b4c6925623"
cloudflare.com.		300 IN TXT "MS=ms70274184"
cloudflare.com.		300 IN TXT "facebook-domain-verification=h9mm6zopj6p2po54woa16m5bskm6oo"
cloudflare.com.		300 IN TXT "v=spf1 ip4:199.15.212.0/22 ip4:173.245.48.0/20 include:_spf.google.com include:spf1.mcsv.net include:spf.mandrillapp.com include:mail.zendesk.com include:stspg-customer.com include:_spf.salesforce.com -all"
cloudflare.com.		300 IN TXT "cisco-ci-domain-verification=27e926884619804ef987ae4aa1c4168f6b152ada84f4c8bfc74eb2bd2912ad72"
cloudflare.com.		300 IN TXT "stripe-verification=bf1a94e6b16ace2502a4a7fff574a25c8a45291054960c883c59be39d1788db9"
cloudflare.com.		300 IN CAA 0 issue "digicert.com\; cansignhttpexchanges=yes"
cloudflare.com.		300 IN CAA 0 issuewild "digicert.com\; cansignhttpexchanges=yes"
cloudflare.com.		300 IN CAA 0 issue "letsencrypt.org"
cloudflare.com.		300 IN CAA 0 issuewild "comodoca.com"
cloudflare.com.		300 IN CAA 0 issue "comodoca.com"
cloudflare.com.		300 IN CAA 0 iodef "mailto:tls-abuse@cloudflare.com"
cloudflare.com.		300 IN CAA 0 issuewild "letsencrypt.org"
cloudflare.com.		300 IN CAA 0 issuewild "pki.goog\; cansignhttpexchanges=yes"
cloudflare.com.		300 IN CAA 0 issue "pki.goog\; cansignhttpexchanges=yes"
`
  .split(/\n/g).filter(Boolean).map(
    row => row.split(/\s/g).filter(Boolean)
  )

const json = JSON.stringify(records)
console.log("JSON         :", json.length, 1)

console.time('cbor')
const cborEncoded = cbor.encode(records)
console.log("\nCBOR         :", cborEncoded.byteLength, (cborEncoded.byteLength / json.length).toFixed(2))
console.timeEnd('cbor')

console.time('lz4')
const jsonLZ4 = lz4.encode(Buffer.from(json))
console.log("\nJSON + lz4   :", jsonLZ4.byteLength, (jsonLZ4.byteLength / json.length).toFixed(2))
console.timeEnd('lz4')

const cborLZ4 = lz4.encode(cborEncoded)
console.log("\nCBOR + lz4   :", cborLZ4.byteLength, (cborLZ4.byteLength / json.length).toFixed(2))

console.time('json + brotli')
const jsonBrotil = brotli.compress(Buffer.from(json));
console.log("\nJSON + brotli:", jsonBrotil.byteLength, (jsonBrotil.byteLength / json.length).toFixed(2))
console.timeEnd('json + brotli')

console.time('brotli + cbor')
const cborBrtoli = brotli.compress(cbor.encode(records));
console.log("\nCBOR + brotli:", cborBrtoli.byteLength, (cborBrtoli.byteLength / json.length).toFixed(2))
console.timeEnd('brotli + cbor')

// console.time('\ndecode - json + brotli')
// const d1 = JSON.parse(Buffer.from(brotli.decompress(jsonBrotil)))
// assert.deepEqual(d1, records)
// console.timeEnd('\ndecode - json + brotli')

console.time('\ndecode - cbor + brotli')
const d2 = cbor.decode(Buffer.from(brotli.decompress(cborBrtoli)))
assert.deepEqual(d2, records)
console.timeEnd('\ndecode - cbor + brotli')
