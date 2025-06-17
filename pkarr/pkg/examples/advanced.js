/**
 * Advanced Pkarr WASM Example
 * 
 * This example demonstrates advanced usage of pkarr WASM bindings:
 * - Custom relay configuration
 * - Multiple DNS record types
 * - Error handling and retry logic
 * - Compare-and-swap publishing
 * - Resolving most recent packets
 * - Working with existing keypairs
 */

const { Client, WasmKeypair, SignedPacket, WasmUtils } = require('../pkarr.js');

async function advancedExample() {
    console.log('🚀 Pkarr WASM Advanced Example\n');
    
    try {
        // Step 1: Create a client with custom configuration
        console.log('📡 Creating client with custom configuration...');
        
        // You can specify custom relays and timeout
        const customRelays = [
            'https://pkarr.pubky.app',
            'https://pkarr.pubky.org'
        ];
        const timeoutMs = 10000; // 10 seconds
        
        const client = new Client(customRelays, timeoutMs);
        console.log('✅ Client created with custom relays');
        console.log(`   Timeout: ${timeoutMs}ms`);
        console.log(`   Relays: ${customRelays.join(', ')}`);
        console.log();
        
        // Step 2: Create a keypair from existing secret key (or generate new one)
        console.log('🔑 Working with keypairs...');
        
        // Generate a new keypair
        const keypair = new WasmKeypair();
        const publicKey = keypair.public_key_string();
        console.log(`✅ Generated new keypair: ${publicKey}`);
        
        // You can also create from existing secret key bytes:
        // const existingSecretKey = new Uint8Array(32); // Your 32-byte secret key
        // const keypairFromSecret = WasmKeypair.from_secret_key(existingSecretKey);
        
        console.log();
        
        // Step 3: Create a comprehensive DNS packet with multiple record types
        console.log('📦 Creating comprehensive DNS packet...');
        const builder = SignedPacket.builder();
        
        // Add various DNS record types
        builder.addTxtRecord("_service", "v=1;type=web", 3600);
        builder.addTxtRecord("description", "My decentralized app", 7200);
        builder.addTxtRecord("contact", "admin@example.com", 3600);
        
        // IPv4 addresses
        builder.addARecord("www", "192.168.1.100", 3600);
        builder.addARecord("api", "192.168.1.101", 3600);
        
        // IPv6 addresses
        builder.addAAAARecord("www", "2001:db8::1", 3600);
        builder.addAAAARecord("api", "2001:db8::2", 3600);
        
        // CNAME records
        builder.addCnameRecord("blog", "www", 3600);
        builder.addCnameRecord("docs", "www", 3600);
        
        // Set custom timestamp (optional - defaults to current time)
        const customTimestamp = Date.now();
        builder.setTimestamp(customTimestamp);
        
        console.log(`📝 Created packet with ${builder.recordCount()} DNS records`);
        
        // Sign the packet
        const signedPacket = builder.buildAndSign(keypair);
        console.log('✅ Packet signed');
        console.log(`   Public key: ${signedPacket.public_key_string}`);
        console.log(`   Timestamp: ${signedPacket.timestamp_ms}`);
        console.log(`   Records: ${signedPacket.records.length}`);
        console.log();
        
        // Step 4: Publish with error handling and retry logic
        console.log('📤 Publishing with retry logic...');
        let publishSuccess = false;
        let attempts = 0;
        const maxAttempts = 3;
        
        while (!publishSuccess && attempts < maxAttempts) {
            attempts++;
            try {
                console.log(`   Attempt ${attempts}/${maxAttempts}...`);
                await client.publish(signedPacket);
                publishSuccess = true;
                console.log('✅ Packet published successfully!');
            } catch (error) {
                console.log(`   ❌ Attempt ${attempts} failed: ${error.message}`);
                if (attempts < maxAttempts) {
                    console.log('   ⏳ Waiting 2 seconds before retry...');
                    await new Promise(resolve => setTimeout(resolve, 2000));
                }
            }
        }
        
        if (!publishSuccess) {
            throw new Error(`Failed to publish after ${maxAttempts} attempts`);
        }
        console.log();
        
        // Step 5: Demonstrate resolving most recent packet
        console.log('📥 Resolving most recent packet...');
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for propagation
        
        const mostRecentPacket = await client.resolveMostRecent(publicKey);
        
        if (mostRecentPacket) {
            console.log('✅ Most recent packet resolved!');
            console.log(`   Timestamp: ${mostRecentPacket.timestamp_ms}`);
            console.log(`   Records: ${mostRecentPacket.records.length}`);
            
            // Compare timestamps
            if (mostRecentPacket.timestamp_ms === signedPacket.timestamp_ms) {
                console.log('🎉 Timestamps match - this is our packet!');
            } else {
                console.log('📅 Different timestamp - there might be a newer packet');
            }
        } else {
            console.log('❌ No packet found');
        }
        console.log();
        
        // Step 6: Demonstrate compare-and-swap publishing
        console.log('🔄 Demonstrating compare-and-swap publishing...');
        
        // Create an updated packet
        const updatedBuilder = SignedPacket.builder();
        updatedBuilder.addTxtRecord("_service", "v=2;type=web;updated=true", 3600);
        updatedBuilder.addTxtRecord("last-update", new Date().toISOString(), 3600);
        updatedBuilder.addARecord("www", "192.168.1.200", 3600); // Updated IP
        
        const updatedPacket = updatedBuilder.buildAndSign(keypair);
        
        // Use the previous packet's timestamp as CAS
        const casTimestamp = mostRecentPacket ? mostRecentPacket.timestamp_ms : null;
        
        try {
            await client.publish(updatedPacket, casTimestamp);
            console.log('✅ Compare-and-swap publish successful!');
        } catch (error) {
            console.log(`❌ Compare-and-swap failed: ${error.message}`);
            console.log('   This might happen if another update occurred concurrently');
        }
        console.log();
        
        // Step 7: Utility functions demonstration
        console.log('🛠️  Demonstrating utility functions...');
        
        // Validate public key
        const isValidKey = WasmUtils.validate_public_key(publicKey);
        console.log(`✅ Public key validation: ${isValidKey ? 'VALID' : 'INVALID'}`);
        
        // Get packet bytes and parse back
        const packetBytes = signedPacket.to_bytes();
        console.log(`📦 Packet size: ${packetBytes.length} bytes`);
        
        try {
            const parsedPacket = WasmUtils.parse_signed_packet(packetBytes);
            console.log('✅ Packet parsing successful');
            console.log(`   Parsed public key: ${parsedPacket.public_key_string}`);
            console.log(`   Parsed timestamp: ${parsedPacket.timestamp_ms}`);
        } catch (error) {
            console.log(`❌ Packet parsing failed: ${error.message}`);
        }
        
        console.log('\n🎉 Advanced example completed successfully!');
        console.log('\n📋 Summary of demonstrated features:');
        console.log('   ✅ Custom relay configuration');
        console.log('   ✅ Multiple DNS record types (TXT, A, AAAA, CNAME)');
        console.log('   ✅ Error handling and retry logic');
        console.log('   ✅ Compare-and-swap publishing');
        console.log('   ✅ Most recent packet resolution');
        console.log('   ✅ Utility functions');
        console.log('   ✅ Packet serialization and parsing');
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error('Stack trace:', error.stack);
        process.exit(1);
    }
}

// Run the example if this file is executed directly
if (require.main === module) {
    advancedExample().catch(console.error);
}

module.exports = { advancedExample }; 