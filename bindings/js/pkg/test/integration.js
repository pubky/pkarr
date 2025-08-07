/**
 * Integration Tests for Pkarr WASM
 * 
 * Tests complete workflows with live network operations
 */

const { Client, Keypair, SignedPacket } = require('../index.js');
const { newFixture } = require('./helpers.js');

async function runIntegrationTests() {
    console.log('Running Integration Tests...');
    
    try {
        // Test 1: Basic publish and resolve workflow
        //console.log('\t- Basic publish and resolve workflow');
        
        const localRelay = ['http://0.0.0.0:15411'];
        const timeoutMs = 10000; // 10 seconds
        const client = new Client(localRelay, timeoutMs);

        const {builder, keypair} = newFixture();
        const publicKey = keypair.public_key_string();
        
        // Create a packet
        builder.addTxtRecord("_test", "integration-test=true", 3600);
        builder.addARecord("www", "192.168.1.100", 3600);
        const packet = builder.buildAndSign(keypair);

        await client.publish(packet);
        
        // Wait for propagation
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Resolve the packet
        const resolvedPacket = await client.resolve(publicKey);
        
        if (!resolvedPacket) {
            throw new Error('Failed to resolve published packet');
        }
        
        if (resolvedPacket.publicKeyString !== publicKey) {
            throw new Error('Resolved packet has wrong public key');
        }
        
        if (resolvedPacket.records.length !== 2) {
            throw new Error('Resolved packet has wrong number of records');
        }
        
        // Test 2: Multiple record types
        //console.log('\t- Multiple DNS record types');

        const {builder: builder2, keypair: keypair2} = newFixture();
        const publicKey2 = keypair2.public_key_string();
        
        builder2.addTxtRecord("_service", "type=web;version=1", 3600);
        builder2.addTxtRecord("description", "Integration test service", 7200);
        builder2.addARecord("www", "192.168.1.1", 3600);
        builder2.addARecord("api", "192.168.1.2", 3600);
        builder2.addAAAARecord("www", "2001:db8::1", 3600);
        builder2.addAAAARecord("api", "2001:db8::2", 3600);
        builder2.addCnameRecord("blog", "www", 3600);
        builder2.addHttpsRecord("_443._tcp", 1, "primary.example.com", 3600);
        builder2.addSvcbRecord("_api._tcp", 10, "api-server.example.com", 3600);
        builder2.addNsRecord("subdomain", "ns1.example.com", 86400);
        
        const packet2 = builder2.buildAndSign(keypair2);
        await client.publish(packet2);
        
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const resolvedPacket2 = await client.resolve(publicKey2);
        if (!resolvedPacket2) {
            throw new Error('Complex packet resolution failed');
        }
        if (resolvedPacket2.records.length !== 10) {
            throw new Error(`Complex packet should have 10 records, got ${resolvedPacket2.records.length}`);
        }
        
        // Verify we have all record types
        const recordTypes = resolvedPacket2.records.map(r => r.rdata.type).sort();
        const expectedTypes = ["A", "A", "AAAA", "AAAA", "CNAME", "HTTPS", "NS", "SVCB", "TXT", "TXT"];
        if (JSON.stringify(recordTypes) !== JSON.stringify(expectedTypes)) {
            throw new Error(`Expected record types ${expectedTypes.join(',')}, got ${recordTypes.join(',')}`);
        }
        
        // Test 3: Custom relay configuration
        //console.log('\t- Custom relay configuration');
        const customRelays = ['http://0.0.0.0:15411'];
        const customClient = new Client(customRelays, 10000);
        
        const {builder: builder3, keypair: keypair3} = newFixture();
        builder3.addTxtRecord("_custom", "relay-test=true", 3600);
        const packet3 = builder3.buildAndSign(keypair3);
        
        await customClient.publish(packet3);
        
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const resolvedPacket3 = await customClient.resolve(keypair3.public_key_string());
        if (!resolvedPacket3) {
            throw new Error('Custom relay resolution failed');
        }
        
        // Test 4: resolveMostRecent functionality
        //console.log('\t- resolveMostRecent functionality');
        const mostRecentPacket = await client.resolveMostRecent(publicKey);
        
        if (!mostRecentPacket) {
            throw new Error('resolveMostRecent failed');
        }
        
        if (mostRecentPacket.publicKeyString !== publicKey) {
            throw new Error('resolveMostRecent returned wrong packet');
        }
        
        // Test 5: Packet update workflow
        //console.log('\t- Packet update workflow');
        const updateBuilder = SignedPacket.builder();
        updateBuilder.addTxtRecord("_test", "integration-test=updated", 3600);
        updateBuilder.addTxtRecord("updated", new Date().toISOString(), 3600);
        updateBuilder.addARecord("www", "192.168.1.200", 3600); // Updated IP
        
        const updatePacket = updateBuilder.buildAndSign(keypair);
        
        await client.publish(updatePacket);
        
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const latestPacket = await client.resolveMostRecent(publicKey);
        if (!latestPacket) {
            throw new Error('Failed to resolve updated packet');
        }
        
        if (latestPacket.timestampMs <= packet.timestampMs) {
            throw new Error('Updated packet should have newer timestamp');
        }
        
        // Test 6: Error handling for non-existent keys
        //console.log('\t- Error handling for non-existent keys');
        const nonExistentKey = new Keypair().public_key_string();
        const nonExistentResult = await client.resolve(nonExistentKey);
        
        if (nonExistentResult !== undefined && nonExistentResult !== null) {
            throw new Error(`Expected null or undefined for non-existent key, got: ${nonExistentResult}`);
        }
    } catch (error) {
        console.error('\n❌ Integration test failed:', error.message);
        throw error;
    }
}

// Export for use in test runner
module.exports = { runIntegrationTests };

// Run if called directly
if (require.main === module) {
    runIntegrationTests().catch(error => {
        console.error('❌ Integration tests failed:', error);
        process.exit(1);
    });
} 