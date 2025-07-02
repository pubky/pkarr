/**
 * Advanced Pkarr WASM Example
 * 
 * This example demonstrates advanced usage of pkarr WASM bindings:
 * - Custom relay configuration
 * - All 7 DNS record types (A, AAAA, TXT, CNAME, HTTPS, SVCB, NS)  
 * - Error handling and retry logic
 * - Compare-and-swap publishing
 * - Resolving most recent packets
 * - Working with existing keypairs
 */

const { Client, Keypair, SignedPacket, Utils } = require('../pkarr.js');

async function advancedExample() {
    console.log('🚀 Pkarr WASM Advanced Example\n');
    
    try {
        // Step 1: Create a client with custom configuration
        console.log('📡 Creating client with custom configuration...');
        
        // You can specify custom relays and timeout
        const customRelays = ['http://localhost:15411'];
        const timeoutMs = 10000; // 10 seconds
        
        const client = new Client(customRelays, timeoutMs);
        console.log('✅ Client created with custom relays');
        console.log(`   Timeout: ${timeoutMs}ms`);
        console.log(`   Relays: ${customRelays.join(', ')}`);
        console.log();
        
        // Step 2: Create a keypair from existing secret key (or generate new one)
        console.log('🔑 Working with keypairs...');
        
        // Generate a new keypair
        const keypair = new Keypair();
        const publicKey = keypair.public_key_string();
        console.log(`✅ Generated new keypair: ${publicKey}`);
        
        // You can also create from existing secret key bytes:
        // const existingSecretKey = new Uint8Array(32); // Your 32-byte secret key
        // const keypairFromSecret = Keypair.from_secret_key(existingSecretKey);
        
        console.log();
        
        // Step 3: Create a comprehensive DNS packet with multiple record types
        console.log('📦 Creating comprehensive DNS packet...');
        const builder = SignedPacket.builder();
        
        // Add various DNS record types (now with better error handling)
        try {
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
            
            // HTTPS service records (modern service discovery)
            builder.addHttpsRecord("_443._tcp", 1, "primary.example.com", 3600);
            builder.addHttpsRecord("_443._tcp", 2, "backup.example.com", 7200);
            
            // SVCB service binding records
            builder.addSvcbRecord("_api._tcp", 10, "api-primary.example.com", 3600);
            builder.addSvcbRecord("_api._tcp", 20, "api-backup.example.com", 3600);
            
            // NS records for subdomain delegation
            builder.addNsRecord("subdomain", "ns1.example.com", 86400);
            builder.addNsRecord("subdomain", "ns2.example.com", 86400);
        } catch (error) {
            console.log(`❌ Builder validation caught error: ${error.message}`);
            throw error;
        }
        
        // Set custom timestamp (optional - defaults to current time)
        const customTimestamp = Date.now();
        console.log(`   📅 Setting custom timestamp: ${new Date(customTimestamp).toISOString()}`);
        builder.setTimestamp(customTimestamp);
        
        // Sign the packet
        const signedPacket = builder.buildAndSign(keypair);
        console.log('✅ Packet signed');
        console.log(`   Public key: ${signedPacket.publicKeyString}`);
        console.log(`   Timestamp: ${new Date(signedPacket.timestampMs / 1000).toISOString()} (${signedPacket.timestampMs})`);
        console.log(`   Records: ${signedPacket.records.length}`);
        
        // Display all records in a nice format
        console.log('   📋 DNS Records:');
        signedPacket.records.forEach((record, index) => {
            const formattedValue = Utils.formatRecordValue(record.rdata);
            console.log(`      [${index}] ${record.name} IN ${record.ttl}s ${record.rdata.type} ${formattedValue}`);
        });
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
            console.log(`   Timestamp: ${new Date(mostRecentPacket.timestampMs / 1000).toISOString()} (${mostRecentPacket.timestampMs})`);
            console.log(`   Records: ${mostRecentPacket.records.length}`);
            
            // Compare timestamps
            if (mostRecentPacket.timestampMs === signedPacket.timestampMs) {
                console.log('🎉 Timestamps match - this is our packet!');
            } else {
                console.log('📅 Different timestamp - there might be a newer packet');
                console.log(`   📊 Original: ${new Date(signedPacket.timestampMs / 1000).toISOString()}`);
                console.log(`   📊 Resolved: ${new Date(mostRecentPacket.timestampMs / 1000).toISOString()}`);
            }
        } else {
            console.log('❌ No packet found');
        }
        console.log();
        
        // Step 6: Demonstrate compare-and-swap publishing
        console.log('🔄 Demonstrating compare-and-swap publishing...');
        
        // Create an updated packet
        const updatedBuilder = SignedPacket.builder();
        updatedBuilder.addTxtRecord("_service", "v=3;type=web;updated=true", 3600);
        updatedBuilder.addTxtRecord("last-update", new Date().toISOString(), 3600);
        updatedBuilder.addARecord("www", "192.168.1.200", 3600); // Updated IP
        updatedBuilder.addHttpsRecord("_443._tcp", 1, "new-primary.example.com", 3600); // Updated service
        
        const updatedPacket = updatedBuilder.buildAndSign(keypair);
        
        // Debug: Show what we're working with
        console.log('🔍 CAS Debug Information:');
        console.log(`   Local packet timestamp: ${new Date(signedPacket.timestampMs / 1000).toISOString()} (${signedPacket.timestampMs})`);
        if (mostRecentPacket) {
            console.log(`   Most recent timestamp: ${new Date(signedPacket.timestampMs / 1000).toISOString()} (${signedPacket.timestampMs})`);
            console.log(`   Timestamps match: ${mostRecentPacket.timestampMs === signedPacket.timestampMs ? 'YES' : 'NO'}`);
        }
        
        // Use the previous packet's timestamp as CAS
        // timestamp is in microseconds, but CAS expects milliseconds
        const casTimestamp = mostRecentPacket ? mostRecentPacket.timestampMs / 1000 : signedPacket.timestampMs / 1000;
        console.log(`   Updated packet Timestamp: ${new Date(updatedPacket.timestampMs / 1000).toISOString()} (${updatedPacket.timestampMs})`);
        
        try {
            await client.publish(updatedPacket, casTimestamp);
            console.log('✅ Compare-and-swap publish successful!');
        } catch (error) {
            const errorMessage = error?.message || error?.toString() || 'Unknown error';
            console.log(`❌ Compare-and-swap failed: ${errorMessage}`);
        }
        console.log();
        
        // Step 7: Utility functions demonstration
        console.log('🛠️  Demonstrating utility functions...');
        
        // Validate public key
        const isValidKey = Utils.validatePublicKey(publicKey);
        console.log(`✅ Public key validation: ${isValidKey ? 'VALID' : 'INVALID'}`);
        
        // Get packet bytes and parse back
        const packetBytes = signedPacket.toBytes();
        console.log(`📦 Packet size: ${packetBytes.length} bytes`);
        
        try {
            const parsedPacket = Utils.parseSignedPacket(packetBytes);
            console.log('✅ Packet parsing successful');
            console.log(`   Parsed public key: ${parsedPacket.publicKeyString}`);
            console.log(`   Parsed timestamp: ${new Date(parsedPacket.timestampMs / 1000).toISOString()} (${parsedPacket.timestampMs})`);
        } catch (error) {
            console.log(`❌ Packet parsing failed: ${error.message}`);
        }
        
        console.log('\n🎉 Advanced example completed successfully!');
        console.log('\n📋 Summary of demonstrated features:');
        console.log('   Custom relay configuration');
        console.log('   All 7 DNS record types (TXT, A, AAAA, CNAME, HTTPS, SVCB, NS)');
        console.log('   Enhanced builder with input validation');
        console.log('   Error handling and retry logic');
        console.log('   Compare-and-swap publishing');
        console.log('   Most recent packet resolution');
        console.log('   Utility functions');
        console.log('   Packet serialization and parsing');
        
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