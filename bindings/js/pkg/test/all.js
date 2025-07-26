/**
 * Comprehensive Test Runner for Pkarr WASM
 * 
 * Runs all test suites in order with comprehensive reporting
 */

const { runUnitTests } = require('./unit.js');
const { runIntegrationTests } = require('./integration.js');
const { runEdgeCasesTests } = require('./edge-cases.js');
const { runSvcbTests } = require('./svcb.js');

async function runAllTests() {
    await runUnitTests();
    await runSvcbTests();
    await runEdgeCasesTests();
    await runIntegrationTests();
}

// Run all tests if this file is executed directly
if (require.main === module) {
    runAllTests().catch(error => {
        console.error('Test runner crashed:', error);
        process.exit(1);
    });
}

module.exports = { runAllTests }; 