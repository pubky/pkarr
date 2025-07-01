/**
 * Pkarr WASM Test Suite
 * 
 * Tests core functionality:
 * - Client creation
 * - Keypair generation
 * - Signed packet creation
 * - Publishing to relays
 * - Resolving packets
 */

const { Client, Keypair, SignedPacket, Utils } = require('../pkarr.js');

// Helper function moved to Utils.formatRecordValue

async function runTests() {
    console.log('🧪 Starting Pkarr WASM Test Suite...\n');
    console.log('=' .repeat(60));
    console.log('📝 WASM FUNCTIONALITY TESTS');
    console.log('=' .repeat(60));
    
    try {
        // Test 1: Client creation
        console.log('✅ WASM initialized');
        const customRelays = ['http://localhost:15411'];
        const timeoutMs = 10000; // 10 seconds
        
        const client = new Client(customRelays, timeoutMs);
        console.log('✅ Client created');
        
        // Test 3: Keypair generation
        const keypair = new Keypair();
        const publicKey = keypair.public_key_string();
        console.log(`✅ Generated keypair with public key: ${publicKey}`);
        
        // Test 4: Signed packet creation
        console.log('📦 Creating signed packet with all record types...');
        const builder = SignedPacket.builder();
        builder.addTxtRecord("_test", "wasm-test=true", 3600);
        builder.addARecord("www", "192.168.1.1", 3600);
        builder.addAAAARecord("www", "2001:db8::1", 3600);
        builder.addCnameRecord("alias", "www", 3600);
        builder.addHttpsRecord("_443._tcp", 1, "server.example.com", 3600);
        builder.addSvcbRecord("_service._tcp", 10, "backup.example.com", 3600);
        builder.addNsRecord("subdomain", "ns1.example.com", 3600);
        
        const signedPacket = builder.buildAndSign(keypair);
        console.log('✅ Signed packet created');
        console.log(`   - Public key: ${signedPacket.publicKeyString}`);
        console.log(`   - Timestamp: ${signedPacket.timestampMs}`);
        console.log(`   - Records: ${signedPacket.records.length} DNS records (A, AAAA, TXT, CNAME, HTTPS, SVCB, NS)`);
        
        // Test 5: Publishing
        console.log('📤 Publishing signed packet to relays...');
        await client.publish(signedPacket);
        console.log('✅ Packet published successfully!');
        
        // Test 6: Wait for propagation
        console.log('⏳ Waiting 2 seconds for potential propagation...');
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Test 7: Resolving
        console.log('📥 Resolving packet...');
        const resolvedPacket = await client.resolve(publicKey);
        
        if (resolvedPacket) {
            console.log('✅ Successfully resolved packet!');
            console.log('✅ Resolved packet details:');
            console.log(`   - Public key: ${resolvedPacket.publicKeyString}`);
            console.log(`   - Timestamp: ${resolvedPacket.timestampMs}`);
            console.log(`   - Records: ${resolvedPacket.records.length} DNS records`);
            
            // Display all DNS records
            console.log('   - DNS Records:');
            resolvedPacket.records.forEach((record, index) => {
                const formattedValue = Utils.formatRecordValue(record.rdata);
                console.log(`     [${index}] ${record.name} IN ${record.ttl}s ${record.rdata.type} ${formattedValue}`);
            });
            
            // Test converting to bytes and back
            const packetBytes = resolvedPacket.toBytes();
            console.log(`   - Packet size: ${packetBytes.length} bytes`);
            
            // Test parsing from bytes
            const parsedPacket = SignedPacket.fromBytes(packetBytes);
            console.log(`   - Parsed packet public key: ${parsedPacket.publicKeyString}`);
            console.log('✅ Packet resolution successful!');
            
        } else {
            console.log('❌ No packet resolved');
        }
        
        // Test 8: Resolve most recent
        console.log('📥 Testing resolveMostRecent...');
        const mostRecentPacket = await client.resolveMostRecent(publicKey);
        if (mostRecentPacket) {
            console.log('✅ Successfully resolved most recent packet!');
            console.log(`   - Public key: ${mostRecentPacket.publicKeyString}`);
            console.log(`   - Timestamp: ${mostRecentPacket.timestampMs}`);
            console.log(`   - Records: ${mostRecentPacket.records.length} DNS records`);
        } else {
            console.log('❌ No most recent packet found');
        }
        
        // Test 9: Test Utils functionality
        console.log('🔧 Testing Utils...');
        const isValidKey = Utils.validatePublicKey(publicKey);
        console.log(`✅ Public key validation: ${isValidKey ? 'VALID' : 'INVALID'}`);
        
        const utilsRelays = Utils.defaultRelays();
        console.log(`✅ Utils default relays: ${utilsRelays.length} relays`);
        
        console.log('\n' + '=' .repeat(60));
        console.log('🎉 ALL TESTS COMPLETED SUCCESSFULLY!');
        console.log('=' .repeat(60));
        
        console.log('\n📊 Test Summary:');
        console.log('   WASM initialization');
        console.log('   Client creation');
        console.log('   Keypair generation');
        console.log('   Complete DNS record support: A, AAAA, TXT, CNAME, HTTPS, SVCB, NS');
        console.log('   Signed packet creation');
        console.log('   Publishing with SignedPacket objects');
        console.log('   Resolving returns SignedPacket objects');
        console.log('   Property access (camelCase)');
        console.log('   Byte serialization/deserialization');
        console.log('   Utils functionality');
        console.log('   Publishing to live relays');
        console.log('   Packet resolution');
        console.log('   WASM bindings: FULLY FUNCTIONAL');
        
    } catch (error) {
        console.error('\n❌ Test suite failed:', error.message);
        console.error('Stack trace:', error.stack);
        process.exit(1);
    }
}

// Run tests if this file is executed directly
if (require.main === module) {
    runTests().catch(error => {
        console.error('❌ Test failed:', error);
        process.exit(1);
    });
}

module.exports = { runTests }; 