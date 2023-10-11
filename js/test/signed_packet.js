import test from 'brittle'

import { SignedPacket, generateKeyPair } from '../index.js'

test('successful put - get', async (t) => {
  const keypair = generateKeyPair(Buffer.alloc(32))

  /** @type {import('../index.js').Packet} */
  const packet = {
    id: 0,
    type: 'response',
    flags: 0,
    answers: [
      { name: '@', type: 'A', data: '1.1.1.1' },
      { name: 'foo', type: 'TXT', ttl: 30, data: 'bar' }
    ]
  }

  const signedPacket = SignedPacket.fromPacket(keypair, packet, { timestamp: 1697016827303000 })

  t.is(signedPacket.packet().answers[0].name, '8pinxxgqs41n4aididenw5apqp1urfmzdztr8jt4abrkdn435ewo', 'Replace @ with the origin')
  t.is(signedPacket.packet().answers[1].name, 'foo.8pinxxgqs41n4aididenw5apqp1urfmzdztr8jt4abrkdn435ewo', 'Add Origin to the end of the name')
  t.is(signedPacket.timestamp(), 1697016827303000)
  t.is(signedPacket.publicKey(), keypair.publicKey)
  t.is(Buffer.from(signedPacket.signature()).toString('hex'), 'c45d1577e6c0d7fe24d1cd6696776a25628b438ffa10b1419dfdede3c51f8577d8447d44df4523fa43f574d6f13a65433fa828eb280065c271fcdcaef5365c01')
  t.is(Buffer.from(signedPacket.bytes()).toString('hex'), 'c45d1577e6c0d7fe24d1cd6696776a25628b438ffa10b1419dfdede3c51f8577d8447d44df4523fa43f574d6f13a65433fa828eb280065c271fcdcaef5365c010006076d852b5458000080000000000200000000343870696e787867717334316e346169646964656e773561707170317572666d7a647a7472386a74346162726b646e34333565776f00000100010000000000040101010103666f6f343870696e787867717334316e346169646964656e773561707170317572666d7a647a7472386a74346162726b646e34333565776f00001000010000001e000403626172')

  t.alike(signedPacket.bep44Args(), {
    k: signedPacket.publicKey(),
    seq: 1697016827303000,
    sig: signedPacket.signature(),
    v: signedPacket.bytes().slice(72)
  })

  t.is(signedPacket.resourceRecords('@')[0], packet.answers[0])
  t.is(signedPacket.resourceRecords('foo')[0], packet.answers[1])

  const fromBytes = SignedPacket.fromBytes(keypair.publicKey, signedPacket.bytes())

  t.alike(Buffer.from(fromBytes.bytes()).toString('hex'), Buffer.from(signedPacket.bytes()).toString('hex'))
  t.is(fromBytes.timestamp(), signedPacket.timestamp())
})
