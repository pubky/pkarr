const { Keypair, SignedPacketBuilder, Utils } = require('../pkarr.js');

/**
 * Spin up a fresh Keypair + Builder
 */
function newFixture() {
  const keypair = new Keypair();
  const builder = SignedPacketBuilder.builder();
  return { keypair, builder };
}

module.exports = {
    newFixture,
    validatePublicKey: Utils.validatePublicKey,
    parseSignedPacket: Utils.parseSignedPacket,
  };