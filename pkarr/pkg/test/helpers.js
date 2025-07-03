const { Keypair, SignedPacketBuilder } = require('../pkarr.js');

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
    sleep
  };