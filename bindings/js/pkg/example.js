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
    console.log('Pkarr JS Bindings Example\n');
    
    try {
        // === SECTION 1: Basic Client Setup ===
        console.log('⚙️ Client Setup');
        const client = new Client();
        console.log('Client created successfully');
        console.log();
        
        // Client custom configuration example:
        // const customRelays = ['http://localhost:15411'];
        // const timeoutMs = 10000;
        // const advancedClient = new Client(customRelays, timeoutMs);
        
        console.log('🔑 Keypair Management');
        const keypair = new Keypair();
        const publicKey = keypair.public_key_string();
        console.log(`Generated keypair: ${publicKey}`);
        console.log();
        
        console.log('📦 DNS Packet Creation');
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
            
            // HTTPS service record without parameters
            builder.addHttpsRecord("_443._tcp", 1, "primary.example.com", 3600);

            // HTTPS record with all supported parameter types
            builder.addHttpsRecord("api", 6, "server.example.com", 3600, {
                alpn: "http/1.1",
                port: 8443,
                ipv4hint: "192.168.1.100",
                ipv6hint: "2001:db8::1",
            });
            
            // SVCB records
            builder.addSvcbRecord("_api._tcp", 2, "api-master.example.com", 3600);
            builder.addSvcbRecord("_api._tcp", 8, "api-slave.example.com", 8888, {
                alpn: ["h3"],
                port: 6888,
                ipv4hint: "192.188.88.8",
                ipv6hint: "2001:db8::1"
            });
            
            // NS record for subdomain delegation
            builder.addNsRecord("subdomain", "ns1.example.com", 86400);
            
        } catch (error) {
            console.log(`Error: ${error}`);
            throw error;
        }
        
        const signedPacket = builder.buildAndSign(keypair);
        console.log('DNS packet created successfully');
        console.log();
        
        console.log('📡 Publishing');
        console.log('Publishing packet...');
        await client.publish(signedPacket);
        console.log('Publish successful');
        console.log();
        
        console.log('🌐 Resolution');
        console.log('Waiting for propagation...');
        await sleep(2000);
        
        console.log('Resolving packet...');
        const resolvedPacket = await client.resolve(publicKey);
        if (resolvedPacket) {
            console.log('Resolution successful');
            console.log(`Timestamp: ${new Date(resolvedPacket.timestampMs / 1000).toISOString()}`);
            console.log(`Records: ${resolvedPacket.records.length}`);
        } else {
            console.log('Resolution failed');
        }
        console.log();
        
        // TODO: There is something wrong with the CAS publishing
        // console.log('🔄 Compare-and-Swap Publishing');
        
        // // Create an updated packet
        // builder.clear();
        // builder.addTxtRecord("_service", "v=2;type=web;updated=true", 3600);
        // builder.addARecord("www", "192.168.1.200", 3600);
        
        // const updatedPacket = builder.buildAndSign(keypair);
        
        // if (signedPacket) {
        //     console.log(`Current timestamp: ${new Date(signedPacket.timestampMs / 1000).toISOString()}`);
        //     console.log(`Update timestamp: ${new Date(updatedPacket.timestampMs / 1000).toISOString()}`);
            
        //     const casTimestamp = signedPacket.timestampMs / 1000;
        //     try {
        //         await client.publish(updatedPacket, casTimestamp);
        //         console.log('Compare-and-swap publish successful');
        //     } catch (error) {
        //         console.log(`Compare-and-swap failed: ${error}`);
        //     }
        // }
        // console.log();
        
        console.log('🧰 Utility Functions');
        
        // Public key validation
        const isValidKey = Utils.validatePublicKey(publicKey);
        console.log(`Public key validation: ${isValidKey ? 'valid' : 'invalid'}`);
        
        // Default relays
        const defaultRelays = Utils.defaultRelays();
        console.log(`Default relays available: ${defaultRelays.length}`);
        
        // Packet operations
        const packetBytes = signedPacket.bytes();
        const compressedPacketBytes = signedPacket.compressedBytes();
        console.log(`Packet size - uncompressed: ${packetBytes.length} bytes`);
        console.log(`Packet size - compressed: ${compressedPacketBytes.length} bytes`);
        
        try {
            const parsedPacket = SignedPacket.fromBytes(packetBytes);
            console.log('Packet parsing successful');
            console.log(`Parsed public key: ${parsedPacket.publicKeyString}`);
            console.log(`Parsed timestamp: ${new Date(parsedPacket.timestampMs / 1000).toISOString()}`);

            // Record value formatting
            console.log('Record Value Formatting:');
            for (const record of parsedPacket.records) {
                const formattedValue = Utils.formatRecordValue(record.rdata);
                console.log(`\t${record.name} (${record.rdata.type}): ${formattedValue}`);
            }

        } catch (error) {
            console.log(`Packet parsing failed: ${error}`);
        }
        
        try {
            const parsedPacket2 = SignedPacket.fromBytes(packetBytes);
            let sameKey = parsedPacket2.publicKeyString === publicKey;
        } catch (error) {
            console.log(`Alternative parsing failed: ${error}`);
        }
        console.log();
        
    } catch (error) {
        console.error('Example failed:', error);
        process.exit(1);
    }
}

// Run the example if this file is executed directly
if (require.main === module) {
    runExample().catch(error => {
        console.error('Example failed:', error);
        process.exit(1);
    });
}

module.exports = { runExample }; 