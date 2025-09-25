/**
 * Test Helper Functions
 */

const { Keypair, SignedPacket, Utils } = require('../index.cjs');

// Re-export commonly used functions for tests
const newFixture = () => ({
    builder: SignedPacket.builder(),
    keypair: new Keypair()
});

const validatePublicKey = Utils.validatePublicKey;

module.exports = {
    newFixture,
    validatePublicKey
};