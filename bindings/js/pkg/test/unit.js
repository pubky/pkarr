/**
 * Unit Tests for Pkarr WASM
 * 
 * Tests individual components and methods in isolation
 */

const { Client, Keypair, SignedPacket } = require('../pkarr.js');
const { newFixture, validatePublicKey, parseSignedPacket } = require('./helpers.js');

async function runUnitTests() {
    console.log('Running Unit Tests...');
    
    let failed = 0;
    
    // Helper function to run a test
    function test(name, testFn) {
        try {
            //console.log(`\t-${name}`);
            testFn();
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
        const {builder, keypair} = newFixture();
        builder.addTxtRecord("test", "value", 3600);
        
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("TXT record not added");
    });
    
    // Test 11: Adding A record
    test("Adding A record", () => {
        const {builder, keypair} = newFixture();
        builder.addARecord("www", "192.168.1.1", 3600);
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("A record not added");
    });
    
    // Test 12: Adding AAAA record
    test("Adding AAAA record", () => {
        const {builder, keypair} = newFixture();
        builder.addAAAARecord("www", "2001:db8::1", 3600);
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("AAAA record not added");
    });
    
    // Test 13: Adding CNAME record
    test("Adding CNAME record", () => {
        const {builder, keypair} = newFixture();
        builder.addCnameRecord("alias", "target", 3600);
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("CNAME record not added");
    });
    
    // Test 14: Adding HTTPS record
    test("Adding HTTPS record", () => {
        const {builder, keypair} = newFixture();
        builder.addHttpsRecord("_443._tcp", 1, "primary.example.com", 3600);
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("HTTPS record not added");
        
        // Verify the record structure
        const record = packet.records[0];
        if (record.rdata.type !== "HTTPS") throw new Error("Wrong record type");
        if (record.rdata.priority !== 1) throw new Error("Wrong HTTPS priority");
        if (record.rdata.target !== "primary.example.com") throw new Error("Wrong HTTPS target");
    });
    
    // Test 15: Adding SVCB record
    test("Adding SVCB record", () => {
        const {builder, keypair} = newFixture();
        builder.addSvcbRecord("_api._tcp", 10, "api-server.example.com", 3600);
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("SVCB record not added");
        
        // Verify the record structure
        const record = packet.records[0];
        if (record.rdata.type !== "SVCB") throw new Error("Wrong record type");
        if (record.rdata.priority !== 10) throw new Error("Wrong SVCB priority");
        if (record.rdata.target !== "api-server.example.com") throw new Error("Wrong SVCB target");
    });
    
    // Test 16: Adding NS record
    test("Adding NS record", () => {
        const {builder, keypair} = newFixture();
        builder.addNsRecord("subdomain", "ns1.example.com", 86400);
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("NS record not added");
        
        // Verify the record structure
        const record = packet.records[0];
        if (record.rdata.type !== "NS") throw new Error("Wrong record type");
        if (record.rdata.nsdname !== "ns1.example.com") throw new Error("Wrong NS nameserver");
    });
    
    // Test 17: Multiple records (all 7 types)
    test("Multiple records (all 7 types)", () => {
        const {builder, keypair} = newFixture();
        builder.addTxtRecord("txt", "value", 3600);
        builder.addARecord("www", "192.168.1.1", 3600);
        builder.addAAAARecord("www", "2001:db8::1", 3600);
        builder.addCnameRecord("alias", "target", 3600);
        builder.addHttpsRecord("_443._tcp", 1, "primary.example.com", 3600);
        builder.addSvcbRecord("_api._tcp", 10, "api-server.example.com", 3600);
        builder.addNsRecord("subdomain", "ns1.example.com", 86400);
        
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 7) throw new Error("Expected 7 records");
        
        // Verify we have all record types
        const types = packet.records.map(r => r.rdata.type).sort();
        const expectedTypes = ["A", "AAAA", "CNAME", "HTTPS", "NS", "SVCB", "TXT"];
        if (JSON.stringify(types) !== JSON.stringify(expectedTypes)) {
            throw new Error(`Expected types ${expectedTypes.join(',')}, got ${types.join(',')}`);
        }
    });
    
    // Test 18: Building and signing packet
    test("Building and signing packet", () => {
        const {builder, keypair} = newFixture();
        builder.addTxtRecord("test", "value", 3600);
        
        const packet = builder.buildAndSign(keypair);
        if (!packet) throw new Error("Packet not created");
        if (packet.publicKeyString !== keypair.public_key_string()) {
            throw new Error("Packet public key doesn't match keypair");
        }
    });
    
    // Test 19: Packet serialization
    test("Packet serialization", () => {
        const {builder, keypair} = newFixture();
        builder.addTxtRecord("test", "value", 3600);
        
        const packet = builder.buildAndSign(keypair);
        const bytes = packet.toBytes();
        if (!bytes || bytes.length === 0) throw new Error("Packet serialization failed");
    });
    
    // Test 20: Public key validation
    test("Public key validation", () => {
        const keypair = new Keypair();
        const publicKey = keypair.public_key_string();
        
        const isValid = validatePublicKey(publicKey);
        if (!isValid) throw new Error("Valid public key marked as invalid");
        
        const isInvalid = validatePublicKey("invalid-key");
        if (isInvalid) throw new Error("Invalid public key marked as valid");
    });
    
    // Test 21: Packet parsing
    test("Packet parsing", () => {
        const {builder, keypair} = newFixture();
        builder.addTxtRecord("test", "value", 3600);
        
        const originalPacket = builder.buildAndSign(keypair);
        const bytes = originalPacket.toBytes();
        const parsedPacket = parseSignedPacket(bytes);
        
        if (parsedPacket.publicKeyString !== originalPacket.publicKeyString) {
            throw new Error("Parsed packet public key doesn't match");
        }
    });
    
    // Test 22: Custom timestamp
    test("Custom timestamp", () => {
        const {builder, keypair} = newFixture();
        const customTime = Date.now();
        
        builder.addTxtRecord("test", "value", 3600);
        builder.setTimestamp(customTime);
        
        const packet = builder.buildAndSign(keypair);
        // The timestamp is stored in microseconds (multiplied by 1000)
        if (packet.timestampMs !== customTime * 1000) {
            throw new Error(`Custom timestamp not set correctly: expected ${customTime * 1000}, got ${packet.timestampMs}`);
        }
    });
    
    // Test 23: Error handling for invalid IPv4 address
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
    
    // Test 24: Error handling for invalid IPv6 address
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

    // Test 25: HTTPS record with priority 0 and target "."
    test("HTTPS record with priority 0 and target '.'", () => {
        const {builder, keypair} = newFixture();
        builder.addHttpsRecord("_443._tcp", 0, ".", 3600);
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("HTTPS record not added");
        
        const record = packet.records[0];
        if (record.rdata.priority !== 0) throw new Error("Wrong priority for alias mode");
        // The "." target gets normalized to an empty string in HTTPS/SVCB records
        if (record.rdata.target !== "") {
            throw new Error(`Wrong target for alias mode: expected empty string, got '${record.rdata.target}'`);
        }
    });
    
    // Test 26: SVCB record with high priority
    test("SVCB record with high priority", () => {
        const {builder, keypair} = newFixture();
        builder.addSvcbRecord("_service._tcp", 65535, "backup.example.com", 3600);
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("SVCB record not added");
        
        const record = packet.records[0];
        if (record.rdata.priority !== 65535) throw new Error("Wrong maximum priority");
    });
    
    // Test 27: NS record validation
    test("NS record validation", () => {
        const {builder, keypair} = newFixture();
        builder.addNsRecord("zone", "ns.example.com", 86400);
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 1) throw new Error("NS record not added");
        
        const record = packet.records[0];
        if (record.name !== `zone.${keypair.public_key_string()}`) throw new Error("NS record name not properly normalized");
        if (record.ttl !== 86400) throw new Error("Wrong TTL for NS record");
    });
    
    // Test 28: Multiple HTTPS records with different priorities
    test("Multiple HTTPS records with different priorities", () => {
        const {builder, keypair} = newFixture();
        builder.addHttpsRecord("_443._tcp", 1, "primary.example.com", 3600);
        builder.addHttpsRecord("_443._tcp", 2, "secondary.example.com", 3600);
        builder.addHttpsRecord("_443._tcp", 0, ".", 3600); // Alias mode
        
        const packet = builder.buildAndSign(keypair);
        if (packet.records.length !== 3) throw new Error("Expected 3 HTTPS records");
        
        // Check that all are HTTPS records
        const httpsRecords = packet.records.filter(r => r.rdata.type === "HTTPS");
        if (httpsRecords.length !== 3) throw new Error("Expected 3 HTTPS records");
        
        // Check priorities
        const priorities = httpsRecords.map(r => r.rdata.priority).sort((a, b) => a - b);
        if (JSON.stringify(priorities) !== JSON.stringify([0, 1, 2])) {
            throw new Error("Wrong priorities for HTTPS records");
        }
    });
    
    // Test 29: SignedPacket static builder method
    test("SignedPacket static builder method", () => {
        const {builder, keypair} = newFixture();
        if (!builder) throw new Error("Static builder method failed");
        
        builder.addTxtRecord("test", "value", 3600);
        const packet = builder.buildAndSign(keypair);
        if (!packet) throw new Error("Builder from static method failed");
    });

    if (failed > 0) {
        throw new Error(`${failed} unit tests failed`);
    }
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