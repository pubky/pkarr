const { Client, Keypair, SignedPacket, Utils } = require('./index.js');

/**
 * Comprehensive Pkarr WASM Example
 * 
 * This example demonstrates both basic and advanced usage patterns
 * of the Pkarr WASM library in a single comprehensive walkthrough.
 */

// Helper function for delays
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function runExample() {
    console.log('🚀 Pkarr WASM Comprehensive Example\n');
    
    try {
        // === SECTION 1: Basic Client Setup ===
        console.log('📡 SECTION 1: Client Setup');

        const client = new Client();
        console.log('✅ Client created');
        console.log();
        
        // Client custom configuration
        // const customRelays = ['http://localhost:15411'];
        // const timeoutMs = 10000;
        // const advancedClient = new Client(customRelays, timeoutMs);
        
        console.log('🔑 SECTION 2: Keypair Management');
        
        const keypair = new Keypair();
        const publicKey = keypair.public_key_string();
        console.log(`✅ Generated keypair: ${publicKey}`);
        
        // Note: You can also create from existing secret key:
        // const existingSecretKey = new Uint8Array(32); // Your 32-byte secret key
        // const keypairFromSecret = Keypair.from_secret_key(existingSecretKey);
        console.log();
        
        console.log('📦 SECTION 3: DNS Packet Creation');
        
        const builder = SignedPacket.builder();
        
        try {
            // TXT record for service discovery
            builder.addTxtRecord("_service", "v=1;type=web;secure=true", 3600);
            
            // A record for IPv4 address
            builder.addARecord("www", "192.168.1.100", 3600);
            
            // AAAA record for IPv6 address
            builder.addAAAARecord("www", "2001:db8::1", 3600);
            
            // CNAME record for aliasing
            builder.addCnameRecord("blog", "www", 3600);
            
            // HTTPS service record
            builder.addHttpsRecord("_443._tcp", 1, "primary.example.com", 3600);
            
            // SVCB service binding record
            builder.addSvcbRecord("_api._tcp", 10, "api-primary.example.com", 3600);
            
            // NS record for subdomain delegation
            builder.addNsRecord("subdomain", "ns1.example.com", 86400);
            
        } catch (error) {
            console.log(`❌ Builder validation error: ${error.message}`);
            throw error;
        }
        
        const signedPacket = builder.buildAndSign(keypair);
        console.log('✅ DNS packet created');
        console.log();
        
        console.log('📤 SECTION 4: Publishing');
        
        // Basic publishing
        console.log('   📤 Publishing packet...');
        await client.publish(signedPacket);
        console.log('   ✅ Successful!');
        console.log();
        
        console.log('📥 SECTION 5: Resolution Strategies');
        
        // Wait for potential propagation
        console.log('   ⏳ Waiting for propagation...');
        await sleep(2000);
        
        // Basic resolution
        console.log('   📥 Resolving packet...');
        const resolvedPacket = await client.resolve(publicKey);
        if (resolvedPacket) {
            console.log('   ✅ Successful!');
            console.log(`      Timestamp: ${new Date(resolvedPacket.timestampMs / 1000).toISOString()}`);
            console.log(`      Records: ${resolvedPacket.records.length}`);
        } else {
            console.log('   ❌ Resolve failed');
        }
        console.log();
        
        // === SECTION 6: Compare-and-Swap Publishing ===
        console.log('🔄 SECTION 6: Compare-and-Swap Publishing');
        
        // Create an updated packet with fewer records
        builder.clear(); // Reset the builder
        builder.addTxtRecord("_service", "v=2;type=web;updated=true", 3600);
        builder.addARecord("www", "192.168.1.200", 3600); // Updated IP
        
        const updatedPacket = builder.buildAndSign(keypair);
        
        // Demonstrate CAS publishing
        console.log('   🔍 CAS Debug Information:');
        if (signedPacket) {
            console.log(`      Current timestamp: ${new Date(signedPacket.timestampMs / 1000).toISOString()}`);
            console.log(`      Update timestamp: ${new Date(updatedPacket.timestampMs / 1000).toISOString()}`);
            
            const casTimestamp = signedPacket.timestampMs / 1000;
            try {
                await client.publish(updatedPacket, casTimestamp);
                console.log('   ✅ Compare-and-swap publish successful!');
            } catch (error) {
                console.log(`   ❌ Compare-and-swap failed: ${error}`);
            }
        } else {
            console.log('   ❌ Cannot perform CAS - no previous packet found');
        }
        console.log();
        
        console.log('🛠️  SECTION 7: Utility Functions');
        
        // Public key validation
        const isValidKey = Utils.validatePublicKey(publicKey);
        console.log(`   ✅ Public key validation: ${isValidKey ? 'VALID' : 'INVALID'}`);
        
        // Default relays
        const defaultRelays = Utils.defaultRelays();
        console.log(`   ✅ Default relays: ${defaultRelays.length} relays`);
        
        // Packet serialization and parsing
        const packetBytes = signedPacket.uncompressedBytes();
        console.log(`   📦 Packet size - uncompressed: ${packetBytes.length} bytes`);
        const compressedPacketBytes = signedPacket.compressedBytes();
        console.log(`   📦 Packet size - compressed: ${compressedPacketBytes.length} bytes`);
        
        try {
            const parsedPacket = SignedPacket.fromBytes(packetBytes);
            console.log('   ✅ Packet parsing successful');
            console.log(`      Parsed public key: ${parsedPacket.publicKeyString}`);
            console.log(`      Parsed timestamp: ${new Date(parsedPacket.timestampMs / 1000).toISOString()}`);
        } catch (error) {
            console.log(`   ❌ Packet parsing failed: ${error.message}`);
        }
        
        // Alternative parsing method
        try {
            const parsedPacket2 = SignedPacket.fromBytes(packetBytes);
            console.log('   ✅ Alternative parsing successful');
            console.log(`      Public key match: ${parsedPacket2.publicKeyString === publicKey}`);
        } catch (error) {
            console.log(`   ❌ Alternative parsing failed: ${error.message}`);
        }
        console.log();
        
    } catch (error) {
        console.error('❌ Example failed:', error);
        process.exit(1);
    }
}

// Run the example if this file is executed directly
if (require.main === module) {
    runExample().catch(error => {
        console.error('❌ Example failed:', error);
        process.exit(1);
    });
}

module.exports = { runExample }; 