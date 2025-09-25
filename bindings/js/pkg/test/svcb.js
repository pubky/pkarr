/**
 * SVCB/HTTPS Record Validation Tests
 * 
 * Tests validation of SVCB and HTTPS record parameters
 */

const { SignedPacket, Keypair } = require('../index.cjs');

async function runSvcbTests() {
    console.log('Running SVCB/HTTPS Tests...');
    
    const keypair = new Keypair();

    // Test 1: Valid ALPN protocols should work
    try {
        const builder1 = SignedPacket.builder();
        builder1.addHttpsRecord("test", 1, "example.com", 3600, {
            alpn: "h2,h3"  // Valid protocols
        });
        builder1.buildAndSign(keypair);
    } catch (error) {
        console.log(`Test 1: Valid ALPN protocols`);
        throw error;
    }

    // Test 2: Invalid ALPN protocol should throw error
    try {
        const builder2 = SignedPacket.builder();
        builder2.addHttpsRecord("test", 1, "example.com", 3600, {
            alpn: "h4"  // Invalid protocol
        });
        builder2.buildAndSign(keypair);
        throw new Error("Invalid ALPN 'h4' was incorrectly accepted");
    } catch (error) {
        if (!error.toString().includes("Invalid ALPN protocol")) {
            console.log(`Test 2: Invalid ALPN protocol`);
            throw error;
        }
    }

    // Test 3: Mixed valid and invalid protocols
    try {
        const builder3 = SignedPacket.builder();
        builder3.addHttpsRecord("test", 1, "example.com", 3600, {
            alpn: ["h2", "h4", "h3"]  // h4 is invalid
        });
        builder3.buildAndSign(keypair);
        throw new Error("Mixed valid/invalid ALPN was incorrectly accepted");
    } catch (error) {
        if (!error.toString().includes("Invalid ALPN protocol")) {
            console.log(`Test 3: Mixed valid/invalid protocols`);
            throw error;
        }
    }

    // Test 4: All valid protocols in array format
    try {
        const builder4 = SignedPacket.builder();
        builder4.addHttpsRecord("test", 1, "example.com", 3600, {
            alpn: ["h2", "h3", "http/1.1"]  // All valid protocols
        });
        builder4.buildAndSign(keypair);
    } catch (error) {
        console.log(`Test 4: Valid ALPN array`);
        throw error;
    }

    // Test 5: Valid port numbers should work
    try {
        const builder5 = SignedPacket.builder();
        builder5.addHttpsRecord("test", 1, "example.com", 3600, {
            port: 8080  // Valid port number
        });
        builder5.buildAndSign(keypair);
    } catch (error) {
        console.log(`Test 5: Valid port numbers`);
        throw error;
    }

    // Test 6: Edge case port numbers
    try {
        const builder6a = SignedPacket.builder();
        builder6a.addHttpsRecord("test", 1, "example.com", 3600, {
            port: 1  // Minimum valid port
        });
        builder6a.buildAndSign(keypair);

        const builder6b = SignedPacket.builder();
        builder6b.addHttpsRecord("test", 1, "example.com", 3600, {
            port: 65535  // Maximum valid port
        });
        builder6b.buildAndSign(keypair);
    } catch (error) {
        console.log(`Test 6: Edge case port numbers`);
        throw error;
    }

    // Test 7: Invalid port numbers
    try {
        const builder7 = SignedPacket.builder();
        builder7.addHttpsRecord("test", 1, "example.com", 3600, {
            port: -1  // Negative port
        });
        builder7.buildAndSign(keypair);
        throw new Error("Negative port -1 was incorrectly accepted");
    } catch (error) {
        if (!error.toString().includes("Port must be")) {
            console.log(`Test 7: Invalid port numbers`);
            throw error;
        }
    }

    // Test 8: Port as 2-byte Uint8Array
    try {
        const builder8 = SignedPacket.builder();
        builder8.addHttpsRecord("test", 1, "example.com", 3600, {
            port: new Uint8Array([0x1F, 0x90])  // Port 8080 as bytes
        });
        builder8.buildAndSign(keypair);
    } catch (error) {
        console.log(`Test 8: Port as 2-byte Uint8Array`);
        throw error;
    }

    // Test 9: Invalid Uint8Array length
    try {
        const builder9 = SignedPacket.builder();
        builder9.addHttpsRecord("test", 1, "example.com", 3600, {
            port: new Uint8Array([0x1F, 0x90, 0x00])  // 3 bytes instead of 2
        });
        builder9.buildAndSign(keypair);
        throw new Error("3-byte Uint8Array was incorrectly accepted");
    } catch (error) {
        if (!error.toString().includes("Port parameter as bytes must be exactly 2 bytes")) {
            console.log(`Test 9: Invalid Uint8Array length`);
            throw error;
        }
    }

    // Test 10: Invalid port types
    try {
        const builder10 = SignedPacket.builder();
        builder10.addHttpsRecord("test", 1, "example.com", 3600, {
            port: "8080"  // String instead of number
        });
        builder10.buildAndSign(keypair);
        throw new Error("Port as string was incorrectly accepted");
    } catch (error) {
        if (!error.toString().includes("Port parameter must be")) {
            console.log(`Test 10: Invalid port types`);
            throw error;
        }
    }

    // Test 11: IPv4hint validation
    try {
        const builder11 = SignedPacket.builder();
        builder11.addHttpsRecord("test", 1, "example.com", 3600, {
            ipv4hint: "192.168.1.1"  // Single IPv4 address
        });
        builder11.buildAndSign(keypair);
    } catch (error) {
        console.log(`Test 11: IPv4hint validation`);
        throw error;
    }

    // Test 12: IPv4hint array validation
    try {
        const builder12 = SignedPacket.builder();
        builder12.addHttpsRecord("test", 1, "example.com", 3600, {
            ipv4hint: ["192.168.1.1", "10.0.0.1", "203.0.113.1"]  // Multiple IPv4 addresses
        });
        builder12.buildAndSign(keypair);   
    } catch (error) {
        console.log(`Test 12: IPv4hint array validation`);
        throw error;
    }

    // Test 13: IPv4hint raw bytes validation
    try {
        const builder13 = SignedPacket.builder();
        builder13.addHttpsRecord("test", 1, "example.com", 3600, {
            ipv4hint: new Uint8Array([192, 168, 1, 1, 10, 0, 0, 1])  // Two IPv4 addresses as bytes
        });
        builder13.buildAndSign(keypair);
    } catch (error) {
        console.log(`Test 13: IPv4hint raw bytes validation`);
        throw error;
    }

    // Test 14: Invalid IPv4hint format
    try {
        const builder14 = SignedPacket.builder();
        builder14.addHttpsRecord("test", 1, "example.com", 3600, {
            ipv4hint: "999.999.999.999"  // Invalid IPv4 address
        });
        builder14.buildAndSign(keypair);
        throw new Error("Invalid IPv4 address was incorrectly accepted");
    } catch (error) {
        if (!error.toString().includes("Invalid IPv4hint")) {
            console.log(`Test 14: Invalid IPv4hint format`);
            throw error;
        }
    }

    // Test 15: IPv4hint invalid byte length
    try {
        const builder15 = SignedPacket.builder();
        builder15.addHttpsRecord("test", 1, "example.com", 3600, {
            ipv4hint: new Uint8Array([192, 168, 1])  // 3 bytes instead of 4
        });
        builder15.buildAndSign(keypair);
        throw new Error("Invalid IPv4hint byte length was incorrectly accepted");
    } catch (error) {
        if (!error.toString().includes("IPv4hint raw bytes must be multiple of 4 bytes")) {
            console.log(`Test 15: IPv4hint invalid byte length`);
            throw error;
        }
    }

    // Test 16: IPv6hint validation
    try {
        const builder16 = SignedPacket.builder();
        builder16.addHttpsRecord("test", 1, "example.com", 3600, {
            ipv6hint: "2001:db8::1"  // Single IPv6 address
        });
        builder16.buildAndSign(keypair);
    } catch (error) {
        console.log(`Test 16: IPv6hint validation`);
        throw error;
    }

    // Test 17: IPv6hint array validation
    try {
        const builder17 = SignedPacket.builder();
        builder17.addHttpsRecord("test", 1, "example.com", 3600, {
            ipv6hint: ["2001:db8::1", "::1", "fe80::1"]  // Multiple IPv6 addresses
        });
        builder17.buildAndSign(keypair);
    } catch (error) {
        console.log(`Test 17: IPv6hint array validation`);
        throw error;
    }

    // Test 18: IPv6hint raw bytes validation
    try {
        const builder18 = SignedPacket.builder();
        builder18.addHttpsRecord("test", 1, "example.com", 3600, {
            ipv6hint: new Uint8Array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])  // IPv6 loopback as bytes
        });
        builder18.buildAndSign(keypair);
    } catch (error) {
        console.log(`Test 18: IPv6hint raw bytes validation`);
        throw error;
    }

    // Test 19: Invalid IPv6hint format
    try {
        const builder19 = SignedPacket.builder();
        builder19.addHttpsRecord("test", 1, "example.com", 3600, {
            ipv6hint: "invalid::address::format"  // Invalid IPv6 address
        });
        builder19.buildAndSign(keypair);
        throw new Error("Invalid IPv6 address was incorrectly accepted");
    } catch (error) {
        if (!error.toString().includes("Invalid IPv6hint")) {
            console.log(`Test 19: Invalid IPv6hint format`);
            throw error;
        }
    }

    // Test 20: IPv6hint invalid byte length
    try {
        const builder20 = SignedPacket.builder();
        builder20.addHttpsRecord("test", 1, "example.com", 3600, {
            ipv6hint: new Uint8Array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])  // 15 bytes instead of 16
        });
        builder20.buildAndSign(keypair);
        throw new Error("Invalid IPv6hint byte length was incorrectly accepted");
    } catch (error) {
        if (!error.toString().includes("IPv6hint raw bytes must be multiple of 16 bytes")) {
            console.log(`Test 20: IPv6hint invalid byte length`);
            throw error;
        }
    }

    // Test 21: Multiple SVCB parameters
    try {
        const builder21 = SignedPacket.builder();
        builder21.addHttpsRecord("test", 1, "example.com", 3600, {
            alpn: ["h2", "h3"],
            port: 8443,
            ipv4hint: ["192.168.1.1", "10.0.0.1"],
            ipv6hint: "2001:db8::1"
        });
        builder21.buildAndSign(keypair);
    } catch (error) {
        console.log(`Test 21: Multiple SVCB parameters`);
        throw error;
    }
}

module.exports = { runSvcbTests }; 

// Run if called directly
if (require.main === module) {
    runSvcbTests().catch(error => {
        console.error('❌ SVCB tests failed:', error);
        process.exit(1);
    });
} 