const { SignedPacket, Keypair } = require('./index.js');

console.log('SVCB Validation Test...\n');

const keypair = new Keypair();

// Test 1: Valid ALPN protocols should work
console.log('Test 1: Valid ALPN protocols');
try {
    const builder1 = SignedPacket.builder();
    builder1.addHttpsRecord("test", 1, "example.com", 3600, {
        alpn: "h2,h4"  // Valid protocols
    });
    const packet1 = builder1.buildAndSign(keypair);
    console.log('✅ Valid ALPN "h2,h3" accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

// Test 2: Invalid ALPN protocol should throw error
console.log('\nTest 2: Invalid ALPN protocol');
try {
    const builder2 = SignedPacket.builder();
    builder2.addHttpsRecord("test", 1, "example.com", 3600, {
        alpn: "h4"  // Invalid protocol
    });
    const packet2 = builder2.buildAndSign(keypair);
    console.log('❌ Invalid ALPN "h4" was incorrectly accepted!');
} catch (error) {
    console.log(`✅ Invalid ALPN "h4" correctly rejected: ${error}`);
}

// Test 3: Mixed valid and invalid protocols
console.log('\nTest 3: Mixed valid and invalid protocols');
try {
    const builder3 = SignedPacket.builder();
    builder3.addHttpsRecord("test", 1, "example.com", 3600, {
        alpn: ["h2", "h1.1", "h3"]  // h1.1 is invalid
    });
    const packet3 = builder3.buildAndSign(keypair);
    console.log('❌ Mixed valid/invalid ALPN was incorrectly accepted!');
} catch (error) {
    console.log(`✅ Mixed valid/invalid ALPN correctly rejected: ${error}`);
}

// Test 4: All valid protocols in array format
console.log('\nTest 4: Valid ALPN protocols in array format');
try {
    const builder4 = SignedPacket.builder();
    builder4.addHttpsRecord("test", 1, "example.com", 3600, {
        alpn: ["h2", "h3", "http/1.1"]  // All valid protocols
    });
    const packet4 = builder4.buildAndSign(keypair);
    console.log('✅ Valid ALPN array ["h2", "h3", "http/1.1"] accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

// Test 5: Valid port numbers should work
console.log('\nTest 5: Valid port numbers');
try {
    const builder5 = SignedPacket.builder();
    builder5.addHttpsRecord("test", 1, "example.com", 3600, {
        port: 8080  // Valid port number
    });
    const packet5 = builder5.buildAndSign(keypair);
    console.log('✅ Valid port 8080 accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

// Test 6: Edge case port numbers
console.log('\nTest 6: Edge case port numbers');
try {
    const builder6a = SignedPacket.builder();
    builder6a.addHttpsRecord("test", 1, "example.com", 3600, {
        port: 1  // Minimum valid port
    });
    const packet6a = builder6a.buildAndSign(keypair);
    console.log('✅ Minimum port 1 accepted');

    const builder6b = SignedPacket.builder();
    builder6b.addHttpsRecord("test", 1, "example.com", 3600, {
        port: 65535  // Maximum valid port
    });
    const packet6b = builder6b.buildAndSign(keypair);
    console.log('✅ Maximum port 65535 accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

// Test 7: Invalid port numbers
console.log('\nTest 7: Invalid port numbers');
try {
    const builder7 = SignedPacket.builder();
    builder7.addHttpsRecord("test", 1, "example.com", 3600, {
        port: -1  // Negative port
    });
    const packet7 = builder7.buildAndSign(keypair);
    console.log('❌ Negative port -1 was incorrectly accepted!');
} catch (error) {
    console.log(`✅ Negative port -1 correctly rejected: ${error}`);
}

// Test 8: Port as 2-byte Uint8Array (big-endian)
console.log('\nTest 8: Port as 2-byte Uint8Array');
try {
    const builder8 = SignedPacket.builder();
    // Port 8080 = 0x1F90 in big-endian: [0x1F, 0x90]
    builder8.addHttpsRecord("test", 1, "example.com", 3600, {
        port: new Uint8Array([0x1F, 0x90])  // Port 8080 as bytes
    });
    const packet8 = builder8.buildAndSign(keypair);
    console.log('✅ Port as 2-byte Uint8Array [0x1F, 0x90] (8080) accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

// Test 9: Invalid Uint8Array length
console.log('\nTest 9: Invalid Uint8Array length');
try {
    const builder9 = SignedPacket.builder();
    builder9.addHttpsRecord("test", 1, "example.com", 3600, {
        port: new Uint8Array([0x1F, 0x90, 0x00])  // 3 bytes instead of 2
    });
    const packet9 = builder9.buildAndSign(keypair);
    console.log('❌ 3-byte Uint8Array was incorrectly accepted!');
} catch (error) {
    console.log(`✅ 3-byte Uint8Array correctly rejected: ${error}`);
}

// Test 10: Invalid port types
console.log('\nTest 10: Invalid port types');
try {
    const builder10 = SignedPacket.builder();
    builder10.addHttpsRecord("test", 1, "example.com", 3600, {
        port: "8080"
    });
    const packet10 = builder10.buildAndSign(keypair);
    console.log('❌ Port as string "8080" was incorrectly accepted!');
} catch (error) {
    console.log(`✅ Port as string "8080" correctly rejected: ${error}`);
}

console.log('\nTest 11: IPv4hint validation');
try {
    const builder11 = SignedPacket.builder();
    builder11.addHttpsRecord("test", 1, "example.com", 3600, {
        ipv4hint: "192.168.1.1"  // Single IPv4 address
    });
    const packet11 = builder11.buildAndSign(keypair);
    console.log('✅ Single IPv4hint "192.168.1.1" accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

console.log('\nTest 12: IPv4hint array validation');
try {
    const builder12 = SignedPacket.builder();
    builder12.addHttpsRecord("test", 1, "example.com", 3600, {
        ipv4hint: ["192.168.1.1", "10.0.0.1", "203.0.113.1"]  // Multiple IPv4 addresses
    });
    const packet12 = builder12.buildAndSign(keypair);
    console.log('✅ IPv4hint array ["192.168.1.1", "10.0.0.1", "203.0.113.1"] accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

console.log('\nTest 13: IPv4hint raw bytes validation');
try {
    const builder13 = SignedPacket.builder();
    // 192.168.1.1 = [192, 168, 1, 1] and 10.0.0.1 = [10, 0, 0, 1]
    builder13.addHttpsRecord("test", 1, "example.com", 3600, {
        ipv4hint: new Uint8Array([192, 168, 1, 1, 10, 0, 0, 1])  // Two IPv4 addresses as bytes
    });
    const packet13 = builder13.buildAndSign(keypair);
    console.log('✅ IPv4hint as raw bytes accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

console.log('\nTest 14: Invalid IPv4hint format');
try {
    const builder14 = SignedPacket.builder();
    builder14.addHttpsRecord("test", 1, "example.com", 3600, {
        ipv4hint: "999.999.999.999"  // Invalid IPv4 address
    });
    const packet14 = builder14.buildAndSign(keypair);
    console.log('❌ Invalid IPv4 address was incorrectly accepted!');
} catch (error) {
    console.log('✅ Invalid IPv4 address correctly rejected:', error);
}

console.log('\nTest 15: IPv4hint invalid byte length');
try {
    const builder15 = SignedPacket.builder();
    builder15.addHttpsRecord("test", 1, "example.com", 3600, {
        ipv4hint: new Uint8Array([192, 168, 1])  // 3 bytes instead of 4
    });
    const packet15 = builder15.buildAndSign(keypair);
    console.log('❌ Invalid IPv4hint byte length was incorrectly accepted!');
} catch (error) {
    console.log('✅ Invalid IPv4hint byte length correctly rejected:', error);
}

console.log('\nTest 16: IPv6hint validation');
try {
    const builder16 = SignedPacket.builder();
    builder16.addHttpsRecord("test", 1, "example.com", 3600, {
        ipv6hint: "2001:db8::1"  // Single IPv6 address
    });
    const packet16 = builder16.buildAndSign(keypair);
    console.log('✅ Single IPv6hint "2001:db8::1" accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

console.log('\nTest 17: IPv6hint array validation');
try {
    const builder17 = SignedPacket.builder();
    builder17.addHttpsRecord("test", 1, "example.com", 3600, {
        ipv6hint: ["2001:db8::1", "::1", "fe80::1"]  // Multiple IPv6 addresses
    });
    const packet17 = builder17.buildAndSign(keypair);
    console.log('✅ IPv6hint array ["2001:db8::1", "::1", "fe80::1"] accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

console.log('\nTest 18: IPv6hint raw bytes validation');
try {
    const builder18 = SignedPacket.builder();
    // ::1 (loopback) = 16 bytes: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    builder18.addHttpsRecord("test", 1, "example.com", 3600, {
        ipv6hint: new Uint8Array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])  // IPv6 loopback as bytes
    });
    const packet18 = builder18.buildAndSign(keypair);
    console.log('✅ IPv6hint as raw bytes accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

console.log('\nTest 19: Invalid IPv6hint format');
try {
    const builder19 = SignedPacket.builder();
    builder19.addHttpsRecord("test", 1, "example.com", 3600, {
        ipv6hint: "invalid::address::format"  // Invalid IPv6 address
    });
    const packet19 = builder19.buildAndSign(keypair);
    console.log('❌ Invalid IPv6 address was incorrectly accepted!');
} catch (error) {
    console.log('✅ Invalid IPv6 address correctly rejected:', error);
}

console.log('\nTest 20: IPv6hint invalid byte length');
try {
    const builder20 = SignedPacket.builder();
    builder20.addHttpsRecord("test", 1, "example.com", 3600, {
        ipv6hint: new Uint8Array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])  // 15 bytes instead of 16
    });
    const packet20 = builder20.buildAndSign(keypair);
    console.log('❌ Invalid IPv6hint byte length was incorrectly accepted!');
} catch (error) {
    console.log('✅ Invalid IPv6hint byte length correctly rejected:', error);
}

console.log('\nTest 21: Multiple SVCB parameters');
try {
    const builder21 = SignedPacket.builder();
    builder21.addHttpsRecord("test", 1, "example.com", 3600, {
        alpn: ["h2", "h3"],
        port: 8443,
        ipv4hint: ["192.168.1.1", "10.0.0.1"],
        ipv6hint: "2001:db8::1"
    });
    const packet21 = builder21.buildAndSign(keypair);
    console.log('✅ Multiple SVCB parameters accepted');
} catch (error) {
    console.log(`❌ Unexpected error: ${error}`);
}

console.log('\n🎉 All SVCB Parameter Tests Complete!');