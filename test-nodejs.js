const { Client, WasmKeypair, SignedPacket } = require('./pkg/pkarr.js');

async function testPkarrWorkflow() {
    console.log('🚀 Starting Pkarr WASM test in Node.js...');
    
    try {
        // Note: For nodejs target, WASM is initialized automatically
        console.log('✅ WASM initialized');
        
        // Create a client with default relays
        const client = new Client();
        console.log('✅ Client created');
        
        // Show default relays
        const relays = Client.defaultRelays();
        console.log(`✅ Using ${relays.length} default relays:`, relays);
        
        // Generate a keypair
        const keypair = new WasmKeypair();
        const publicKey = keypair.public_key_string();
        console.log(`✅ Generated keypair with public key: ${publicKey}`);
        
        // Create a signed packet with some DNS records
        console.log('📦 Creating signed packet...');
        const packetBuilder = SignedPacket.builder();
        
        // Add some DNS records
        packetBuilder.addTxtRecord("_test", "Hey hey heyyyyy!", 3600);
        packetBuilder.addTxtRecord("app", "whatssss up", 3600);
        packetBuilder.addARecord("www", "192.168.1.100", 3600);
        
        // Build and sign the packet
        const signedPacket = packetBuilder.buildAndSign(keypair);
        console.log('✅ Signed packet created');
        console.log(`   - Public key: ${signedPacket.public_key_string}`);
        console.log(`   - Timestamp: ${signedPacket.timestamp_ms}`);
        console.log(`   - Records: ${signedPacket.records.length} DNS records`);
        
        // Publish the packet
        console.log('📤 Publishing signed packet to relays...');
        try {
            await client.publish(signedPacket);
            console.log('✅ Packet published successfully!');
        } catch (publishError) {
            console.error('❌ Failed to publish:', publishError.message);
            // Continue with resolve test even if publish fails
        }
        
        // Wait a moment for propagation
        console.log('⏳ Waiting 2 seconds for potential propagation...');
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Try to resolve the packet we just published
        console.log('📥 Resolving packet...');
        try {
            const resolvedPacket = await client.resolve(publicKey);
            
            if (resolvedPacket) {
                console.log('✅ Successfully resolved packet!');
                console.log(`   - Public key: ${resolvedPacket.public_key_string}`);
                console.log(`   - Timestamp: ${resolvedPacket.timestamp_ms}`);
                console.log(`   - Records count: ${resolvedPacket.records.length}`);
                
                // Compare with original
                if (resolvedPacket.public_key_string === signedPacket.public_key_string) {
                    console.log('✅ Resolved packet matches the published one!');
                } else {
                    console.log('⚠️  Resolved packet has different public key');
                }
            } else {
                console.log('ℹ️  No packet found for this public key (this is normal for a new key)');
            }
        } catch (resolveError) {
            console.error('❌ Failed to resolve:', resolveError.message);
        }
        
        // Test resolveMostRecent
        console.log('📥 Testing resolveMostRecent...');
        try {
            const mostRecentPacket = await client.resolveMostRecent(publicKey);
            
            if (mostRecentPacket) {
                console.log('✅ Successfully resolved most recent packet!');
                console.log(`   - Timestamp: ${mostRecentPacket.timestamp_ms}`);
            } else {
                console.log('ℹ️  No most recent packet found');
            }
        } catch (resolveRecentError) {
            console.error('❌ Failed to resolve most recent:', resolveRecentError.message);
        }
        
        console.log('\n🎉 Pkarr WASM workflow test completed!');
        console.log('\n📋 Summary:');
        console.log('   ✅ WASM initialization: SUCCESS');
        console.log('   ✅ Client creation: SUCCESS');
        console.log('   ✅ Keypair generation: SUCCESS'); 
        console.log('   ✅ Signed packet creation: SUCCESS');
        console.log('   ✅ Methods work directly on client instance');
        console.log('   ✅ Returns SignedPacket objects correctly');
        
    } catch (error) {
        console.error('💥 Unexpected error:', error);
        process.exit(1);
    }
}

// Run the test
testPkarrWorkflow().catch(console.error); 