/**
 * Unit Tests for Pkarr WASM
 * 
 * Tests individual components and methods in isolation
 */

const { Client, Keypair, SignedPacket, Utils } = require('../pkarr.js');

async function runUnitTests() {
    console.log('🧪 Running Pkarr WASM Unit Tests...\n');
    console.log('=' .repeat(60));
    console.log('🔬 UNIT TESTS');
    console.log('=' .repeat(60));
    
    let passed = 0;
    let failed = 0;
    
    // Helper function to run a test
    function test(name, testFn) {
        try {
            console.log(`\n🔍 Testing: ${name}`);
            testFn();
            console.log(`✅ PASS: ${name}`);
            passed++;
        } catch (error) {
            console.log(`❌ FAIL: ${name} - ${error.message}`);
            failed++;
        }
    }
    
    // Test 1: Client instantiation
    test("Client instantiation", () => {
        const client = new Client();
        if (!client) throw new Error("Client not created");
    });
    
    // Test 2: Client with custom relays
    test("Client with custom relays", () => {
        const customRelays = ['https://example.com'];
        const client = new Client(customRelays, 5000);
        if (!client) throw new Error("Client with custom config not created");
    });
    
    // Test 3: Default relays static method
    test("Default relays static method", () => {
        const relays = Client.defaultRelays();
        if (!relays || relays.length === 0) throw new Error("No default relays returned");
        if (relays.length !== 2) throw new Error("Expected 2 default relays");
    });
    
    // Test 4: Keypair generation
    test("Keypair generation", () => {
        const keypair = new Keypair();
        if (!keypair) throw new Error("Keypair not created");
    });
    
    // Test 5: Keypair public key string
    test("Keypair public key string", () => {
        const keypair = new Keypair();
        const publicKey = keypair.public_key_string();
        if (!publicKey || typeof publicKey !== 'string') throw new Error("Invalid public key string");
        if (publicKey.length !== 52) throw new Error("Public key string should be 52 characters");
    });
    
    // Test 6: Keypair secret key bytes
    test("Keypair secret key bytes", () => {
        const keypair = new Keypair();
        const secretKey = keypair.secret_key_bytes();
        if (!secretKey || secretKey.length !== 32) throw new Error("Secret key should be 32 bytes");
    });
    
    // Test 7: Keypair public key bytes
    test("Keypair public key bytes", () => {
        const keypair = new Keypair();
        const publicKey = keypair.public_key_bytes();
        if (!publicKey || publicKey.length !== 32) throw new Error("Public key should be 32 bytes");
    });
    
    // Test 8: Keypair from secret key
    test("Keypair from secret key", () => {
        const originalKeypair = new Keypair();
        const secretKey = originalKeypair.secret_key_bytes();
        const restoredKeypair = Keypair.from_secret_key(secretKey);
        
        if (originalKeypair.public_key_string() !== restoredKeypair.public_key_string()) {
            throw new Error("Restored keypair public key doesn't match");
        }
    });
    
    // Test 9: SignedPacket builder creation
    test("SignedPacket builder creation", () => {
        const builder = SignedPacket.builder();
        if (!builder) throw new Error("Builder not created");
    });
    
    // Test 10: Adding TXT record
    test("Adding TXT record", () => {
        const builder = SignedPacket.builder();
        builder.addTxtRecord("test", "value", 3600);
        // Note: recordCount() was removed - we'll verify by building
        const keypair = new Keypair();
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("TXT record not added");
    });
    
    // Test 11: Adding A record
    test("Adding A record", () => {
        const builder = SignedPacket.builder();
        builder.addARecord("www", "192.168.1.1", 3600);
        const keypair = new Keypair();
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("A record not added");
    });
    
    // Test 12: Adding AAAA record
    test("Adding AAAA record", () => {
        const builder = SignedPacket.builder();
        builder.addAAAARecord("www", "2001:db8::1", 3600);
        const keypair = new Keypair();
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("AAAA record not added");
    });
    
    // Test 13: Adding CNAME record
    test("Adding CNAME record", () => {
        const builder = SignedPacket.builder();
        builder.addCnameRecord("alias", "target", 3600);
        const keypair = new Keypair();
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("CNAME record not added");
    });
    
    // Test 14: Multiple records
    test("Multiple records", () => {
        const builder = SignedPacket.builder();
        builder.addTxtRecord("txt", "value", 3600);
        builder.addARecord("www", "192.168.1.1", 3600);
        builder.addAAAARecord("www", "2001:db8::1", 3600);
        const keypair = new Keypair();
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 3) throw new Error("Expected 3 records");
    });
    
    // Test 15: Building and signing packet
    test("Building and signing packet", () => {
        const keypair = new Keypair();
        const builder = SignedPacket.builder();
        builder.addTxtRecord("test", "value", 3600);
        
        const packet = builder.buildAndSign(keypair);
        if (!packet) throw new Error("Packet not created");
        if (packet.publicKeyString !== keypair.public_key_string()) {
            throw new Error("Packet public key doesn't match keypair");
        }
    });
    
    // Test 16: Packet serialization
    test("Packet serialization", () => {
        const keypair = new Keypair();
        const builder = SignedPacket.builder();
        builder.addTxtRecord("test", "value", 3600);
        
        const packet = builder.buildAndSign(keypair);
        const bytes = packet.toBytes();
        if (!bytes || bytes.length === 0) throw new Error("Packet serialization failed");
    });
    
    // Test 17: Public key validation
    test("Public key validation", () => {
        const keypair = new Keypair();
        const publicKey = keypair.public_key_string();
        
        const isValid = Utils.validatePublicKey(publicKey);
        if (!isValid) throw new Error("Valid public key marked as invalid");
        
        const isInvalid = Utils.validatePublicKey("invalid-key");
        if (isInvalid) throw new Error("Invalid public key marked as valid");
    });
    
    // Test 18: Packet parsing
    test("Packet parsing", () => {
        const keypair = new Keypair();
        const builder = SignedPacket.builder();
        builder.addTxtRecord("test", "value", 3600);
        
        const originalPacket = builder.buildAndSign(keypair);
        const bytes = originalPacket.toBytes();
        const parsedPacket = Utils.parseSignedPacket(bytes);
        
        if (parsedPacket.publicKeyString !== originalPacket.publicKeyString) {
            throw new Error("Parsed packet public key doesn't match");
        }
    });
    
    // Test 19: Custom timestamp
    test("Custom timestamp", () => {
        const keypair = new Keypair();
        const builder = SignedPacket.builder();
        const customTime = Date.now();
        
        builder.addTxtRecord("test", "value", 3600);
        builder.setTimestamp(customTime);
        
        const packet = builder.buildAndSign(keypair);
        // The timestamp is stored in microseconds (multiplied by 1000)
        if (packet.timestampMs !== customTime * 1000) {
            throw new Error(`Custom timestamp not set correctly: expected ${customTime * 1000}, got ${packet.timestampMs}`);
        }
    });
    
    // Test 20: Error handling for invalid IPv4 address
    test("Error handling for invalid IPv4 address", () => {
        const builder = SignedPacket.builder();
        try {
            builder.addARecord("www", "invalid-ip", 3600);
            throw new Error("Should have thrown error for invalid IPv4");
        } catch (error) {
            if (error.message === "Should have thrown error for invalid IPv4") {
                throw error;
            }
            // Expected error - test passed
        }
    });
    
    // Test 21: Error handling for invalid IPv6 address
    test("Error handling for invalid IPv6 address", () => {
        const builder = SignedPacket.builder();
        try {
            builder.addAAAARecord("www", "invalid-ipv6", 3600);
            throw new Error("Should have thrown error for invalid IPv6");
        } catch (error) {
            if (error.message === "Should have thrown error for invalid IPv6") {
                throw error;
            }
            // Expected error - test passed
        }
    });
    
    // Test 22: SignedPacket static builder method
    test("SignedPacket static builder method", () => {
        const builder = SignedPacket.builder();
        if (!builder) throw new Error("Static builder method failed");
        
        builder.addTxtRecord("test", "value", 3600);
        const keypair = new Keypair();
        const packet = builder.buildAndSign(keypair);
        if (!packet) throw new Error("Builder from static method failed");
    });
    
    console.log('\n' + '=' .repeat(60));
    console.log('📊 UNIT TEST RESULTS');
    console.log('=' .repeat(60));
    console.log(`Total tests: ${passed + failed}`);
    console.log(`✅ Passed: ${passed}`);
    console.log(`❌ Failed: ${failed}`);
    console.log(`Success rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);
    
    if (failed > 0) {
        throw new Error(`${failed} unit tests failed`);
    }
    
    console.log('🎉 All unit tests passed!');
}

// Export for use in test runner
module.exports = { runUnitTests };

// Run if called directly
if (require.main === module) {
    runUnitTests().catch(error => {
        console.error('❌ Unit tests failed:', error.message);
        process.exit(1);
    });
} 