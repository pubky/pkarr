/**
 * Unit Tests for Pkarr WASM
 * 
 * Tests individual components and methods in isolation
 */

const { Client, WasmKeypair, SignedPacket, WasmUtils } = require('../pkarr.js');

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
        const keypair = new WasmKeypair();
        if (!keypair) throw new Error("Keypair not created");
    });
    
    // Test 5: Keypair public key string
    test("Keypair public key string", () => {
        const keypair = new WasmKeypair();
        const publicKey = keypair.public_key_string();
        if (!publicKey || typeof publicKey !== 'string') throw new Error("Invalid public key string");
        if (publicKey.length !== 52) throw new Error("Public key string should be 52 characters");
    });
    
    // Test 6: Keypair secret key bytes
    test("Keypair secret key bytes", () => {
        const keypair = new WasmKeypair();
        const secretKey = keypair.secret_key_bytes();
        if (!secretKey || secretKey.length !== 32) throw new Error("Secret key should be 32 bytes");
    });
    
    // Test 7: Keypair public key bytes
    test("Keypair public key bytes", () => {
        const keypair = new WasmKeypair();
        const publicKey = keypair.public_key_bytes();
        if (!publicKey || publicKey.length !== 32) throw new Error("Public key should be 32 bytes");
    });
    
    // Test 8: Keypair from secret key
    test("Keypair from secret key", () => {
        const originalKeypair = new WasmKeypair();
        const secretKey = originalKeypair.secret_key_bytes();
        const restoredKeypair = WasmKeypair.from_secret_key(secretKey);
        
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
        if (builder.recordCount() !== 1) throw new Error("TXT record not added");
    });
    
    // Test 11: Adding A record
    test("Adding A record", () => {
        const builder = SignedPacket.builder();
        builder.addARecord("www", "192.168.1.1", 3600);
        if (builder.recordCount() !== 1) throw new Error("A record not added");
    });
    
    // Test 12: Adding AAAA record
    test("Adding AAAA record", () => {
        const builder = SignedPacket.builder();
        builder.addAAAARecord("www", "2001:db8::1", 3600);
        if (builder.recordCount() !== 1) throw new Error("AAAA record not added");
    });
    
    // Test 13: Adding CNAME record
    test("Adding CNAME record", () => {
        const builder = SignedPacket.builder();
        builder.addCnameRecord("alias", "target", 3600);
        if (builder.recordCount() !== 1) throw new Error("CNAME record not added");
    });
    
    // Test 14: Multiple records
    test("Multiple records", () => {
        const builder = SignedPacket.builder();
        builder.addTxtRecord("txt", "value", 3600);
        builder.addARecord("www", "192.168.1.1", 3600);
        builder.addAAAARecord("www", "2001:db8::1", 3600);
        if (builder.recordCount() !== 3) throw new Error("Expected 3 records");
    });
    
    // Test 15: Building and signing packet
    test("Building and signing packet", () => {
        const keypair = new WasmKeypair();
        const builder = SignedPacket.builder();
        builder.addTxtRecord("test", "value", 3600);
        
        const packet = builder.buildAndSign(keypair);
        if (!packet) throw new Error("Packet not created");
        if (packet.public_key_string !== keypair.public_key_string()) {
            throw new Error("Packet public key doesn't match keypair");
        }
    });
    
    // Test 16: Packet serialization
    test("Packet serialization", () => {
        const keypair = new WasmKeypair();
        const builder = SignedPacket.builder();
        builder.addTxtRecord("test", "value", 3600);
        
        const packet = builder.buildAndSign(keypair);
        const bytes = packet.to_bytes();
        if (!bytes || bytes.length === 0) throw new Error("Packet serialization failed");
    });
    
    // Test 17: Public key validation
    test("Public key validation", () => {
        const keypair = new WasmKeypair();
        const publicKey = keypair.public_key_string();
        
        const isValid = WasmUtils.validate_public_key(publicKey);
        if (!isValid) throw new Error("Valid public key marked as invalid");
        
        const isInvalid = WasmUtils.validate_public_key("invalid-key");
        if (isInvalid) throw new Error("Invalid public key marked as valid");
    });
    
    // Test 18: Packet parsing
    test("Packet parsing", () => {
        const keypair = new WasmKeypair();
        const builder = SignedPacket.builder();
        builder.addTxtRecord("test", "value", 3600);
        
        const originalPacket = builder.buildAndSign(keypair);
        const bytes = originalPacket.to_bytes();
        const parsedPacket = WasmUtils.parse_signed_packet(bytes);
        
        if (parsedPacket.public_key_string !== originalPacket.public_key_string) {
            throw new Error("Parsed packet public key doesn't match");
        }
    });
    
    // Test 19: Custom timestamp
    test("Custom timestamp", () => {
        const keypair = new WasmKeypair();
        const builder = SignedPacket.builder();
        const customTime = Date.now();
        
        builder.addTxtRecord("test", "value", 3600);
        builder.setTimestamp(customTime);
        
        const packet = builder.buildAndSign(keypair);
        // The timestamp is stored in microseconds (multiplied by 1000)
        if (packet.timestamp_ms !== customTime * 1000) {
            throw new Error(`Custom timestamp not set correctly: expected ${customTime * 1000}, got ${packet.timestamp_ms}`);
        }
    });
    
    // Test 20: Error handling for invalid inputs
    test("Error handling for invalid inputs", () => {
        try {
            WasmKeypair.from_secret_key(new Uint8Array(16)); // Wrong size
            throw new Error("Should have thrown error for invalid secret key size");
        } catch (error) {
            // WASM errors might not have detailed messages, just check that it throws
            if (error.message === "Should have thrown error for invalid secret key size") {
                throw error; // Re-throw if it's our test error
            }
            // If it's any other error, that's acceptable - it means validation worked
        }
    });
    
    // Test 21: SignedPacket properties compatibility
    test("SignedPacket properties compatibility", () => {
        const keypair = new WasmKeypair();
        const builder = SignedPacket.builder();
        builder.addTxtRecord("test", "value", 3600);
        builder.addARecord("www", "192.168.1.1", 3600);
        
        const packet = builder.buildAndSign(keypair);
        
        // Check all expected properties exist
        const expectedProperties = [
            'public_key_string',
            'timestamp_ms', 
            'records',
            'to_bytes'
        ];
        
        expectedProperties.forEach(prop => {
            if (!(prop in packet)) {
                throw new Error(`Missing expected property: ${prop}`);
            }
        });
        
        // Check property types
        if (typeof packet.public_key_string !== 'string') {
            throw new Error("public_key_string should be a string");
        }
        
        if (typeof packet.timestamp_ms !== 'number') {
            throw new Error("timestamp_ms should be a number");
        }
        
        if (!Array.isArray(packet.records)) {
            throw new Error("records should be an array");
        }
        
        if (typeof packet.to_bytes !== 'function') {
            throw new Error("to_bytes should be a function");
        }
        
        // Check that records have expected structure
        if (packet.records.length !== 2) {
            throw new Error("Expected 2 records");
        }
        
        const record = packet.records[0];
        const expectedRecordProperties = ['name', 'ttl', 'rdata'];
        
        expectedRecordProperties.forEach(prop => {
            if (!(prop in record)) {
                throw new Error(`Missing expected record property: ${prop}`);
            }
        });
        
        // Check rdata structure
        if (typeof record.rdata !== 'object' || !record.rdata) {
            throw new Error("rdata should be an object");
        }
        
        if (!('type' in record.rdata)) {
            throw new Error("rdata should have a type property");
        }
        
        if (typeof record.rdata.type !== 'string') {
            throw new Error("rdata.type should be a string");
        }
        
        // Check that name is a string and includes the public key
        if (typeof record.name !== 'string') {
            throw new Error("record.name should be a string");
        }
        
        if (!record.name.includes(packet.public_key_string.toLowerCase())) {
            throw new Error("record.name should include the public key");
        }
        
        // Check TTL is a number
        if (typeof record.ttl !== 'number') {
            throw new Error("record.ttl should be a number");
        }
        
        // Verify public key format (should be 52 character base32)
        if (packet.public_key_string.length !== 52) {
            throw new Error("public_key_string should be 52 characters");
        }
        
        // Verify timestamp is reasonable (should be in microseconds)
        if (packet.timestamp_ms < 1000000000000) { // Should be > year 2001 in microseconds
            throw new Error("timestamp_ms appears to be in wrong format");
        }
        
        // Verify to_bytes returns Uint8Array
        const bytes = packet.to_bytes();
        if (!(bytes instanceof Uint8Array)) {
            throw new Error("to_bytes should return Uint8Array");
        }
        
        if (bytes.length === 0) {
            throw new Error("to_bytes should return non-empty array");
        }
    });
    
    // Test 22: rdata properties for all DNS record types
    test("rdata properties for all DNS record types", () => {
        const keypair = new WasmKeypair();
        const builder = SignedPacket.builder();
        
        // Add all supported record types
        builder.addTxtRecord("txt", "test-value", 3600);
        builder.addARecord("www", "192.168.1.1", 3600);
        builder.addAAAARecord("ipv6", "2001:db8::1", 3600);
        builder.addCnameRecord("alias", "target.example.com", 3600);
        
        const packet = builder.buildAndSign(keypair);
        
        if (packet.records.length !== 4) {
            throw new Error("Expected 4 records for all DNS types");
        }
        
        // Test TXT record rdata
        const txtRecord = packet.records[0];
        if (txtRecord.rdata.type !== 'TXT') {
            throw new Error("TXT record should have type 'TXT'");
        }
        if (!('value' in txtRecord.rdata)) {
            throw new Error("TXT record rdata should have 'value' property");
        }
        if (typeof txtRecord.rdata.value !== 'string') {
            throw new Error("TXT record value should be a string");
        }
        if (!txtRecord.rdata.value.includes('test-value')) {
            throw new Error("TXT record value should contain the input text");
        }
        
        // Test A record rdata
        const aRecord = packet.records[1];
        if (aRecord.rdata.type !== 'A') {
            throw new Error("A record should have type 'A'");
        }
        if (!('address' in aRecord.rdata)) {
            throw new Error("A record rdata should have 'address' property");
        }
        if (aRecord.rdata.address !== '192.168.1.1') {
            throw new Error("A record address should match input");
        }
        
        // Test AAAA record rdata
        const aaaaRecord = packet.records[2];
        if (aaaaRecord.rdata.type !== 'AAAA') {
            throw new Error("AAAA record should have type 'AAAA'");
        }
        if (!('address' in aaaaRecord.rdata)) {
            throw new Error("AAAA record rdata should have 'address' property");
        }
        if (aaaaRecord.rdata.address !== '2001:db8::1') {
            throw new Error("AAAA record address should match input");
        }
        
        // Test CNAME record rdata
        const cnameRecord = packet.records[3];
        if (cnameRecord.rdata.type !== 'CNAME') {
            throw new Error("CNAME record should have type 'CNAME'");
        }
        if (!('target' in cnameRecord.rdata)) {
            throw new Error("CNAME record rdata should have 'target' property");
        }
        if (cnameRecord.rdata.target !== 'target.example.com') {
            throw new Error("CNAME record target should match input");
        }
        
        // Verify all records have exactly the expected properties
        const expectedProperties = {
            'TXT': ['type', 'value'],
            'A': ['type', 'address'],
            'AAAA': ['type', 'address'],
            'CNAME': ['type', 'target']
        };
        
        packet.records.forEach((record, index) => {
            const recordType = record.rdata.type;
            const actualProps = Object.keys(record.rdata).sort();
            const expectedProps = expectedProperties[recordType].sort();
            
            if (JSON.stringify(actualProps) !== JSON.stringify(expectedProps)) {
                throw new Error(`${recordType} record has wrong properties. Expected: [${expectedProps.join(', ')}], Got: [${actualProps.join(', ')}]`);
            }
        });
        
        // Verify all rdata objects have the common 'type' property
        packet.records.forEach((record, index) => {
            if (!record.rdata.type) {
                throw new Error(`Record ${index} rdata missing 'type' property`);
            }
            if (typeof record.rdata.type !== 'string') {
                throw new Error(`Record ${index} rdata 'type' should be a string`);
            }
        });
    });
    
    console.log('\n' + '=' .repeat(60));
    console.log('📊 UNIT TEST RESULTS');
    console.log('=' .repeat(60));
    console.log(`✅ Passed: ${passed}`);
    console.log(`❌ Failed: ${failed}`);
    console.log(`📈 Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);
    console.log(`📋 Total Tests: ${passed + failed}`);
    
    if (failed > 0) {
        console.log('\n❌ Some unit tests failed!');
        process.exit(1);
    } else {
        console.log('\n🎉 All unit tests passed!');
    }
}

// Run tests if this file is executed directly
if (require.main === module) {
    runUnitTests().catch(error => {
        console.error('❌ Unit test suite failed:', error);
        process.exit(1);
    });
}

module.exports = { runUnitTests }; 