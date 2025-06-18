/**
 * Comprehensive Test Runner for Pkarr WASM
 * 
 * Runs all test suites in order with comprehensive reporting
 */

const { runUnitTests } = require('./unit.js');
const { runIntegrationTests } = require('./integration.js');
const { runPerformanceTests } = require('./performance.js');
const { runEdgeCasesTests } = require('./edge-cases.js');

async function runAllTests() {
    console.log('🧪 Pkarr WASM - Comprehensive Test Suite');
    console.log('=' .repeat(80));
    console.log('🚀 Starting complete test execution...\n');
    
    const startTime = Date.now();
    const results = {
        unit: { status: 'pending', duration: 0, error: null },
        integration: { status: 'pending', duration: 0, error: null },
        performance: { status: 'pending', duration: 0, error: null },
        edgeCases: { status: 'pending', duration: 0, error: null }
    };
    
    // Helper function to run a test suite
    async function runTestSuite(name, testFunction) {
        console.log(`\n${'='.repeat(80)}`);
        console.log(`🏃 Running ${name.toUpperCase()} Tests...`);
        console.log(`${'='.repeat(80)}`);
        
        const suiteStart = Date.now();
        
        try {
            await testFunction();
            const duration = Date.now() - suiteStart;
            results[name].status = 'passed';
            results[name].duration = duration;
            
            console.log(`\n✅ ${name.toUpperCase()} Tests: PASSED (${duration}ms)`);
            return true;
        } catch (error) {
            const duration = Date.now() - suiteStart;
            results[name].status = 'failed';
            results[name].duration = duration;
            results[name].error = error.message;
            
            console.log(`\n❌ ${name.toUpperCase()} Tests: FAILED (${duration}ms)`);
            console.log(`   Error: ${error.message}`);
            return false;
        }
    }
    
    let totalPassed = 0;
    let totalFailed = 0;
    
    // Run Unit Tests
    if (await runTestSuite('unit', runUnitTests)) {
        totalPassed++;
    } else {
        totalFailed++;
    }
    
    // Run Integration Tests (only if unit tests passed)
    if (results.unit.status === 'passed') {
        if (await runTestSuite('integration', runIntegrationTests)) {
            totalPassed++;
        } else {
            totalFailed++;
        }
    } else {
        console.log('\n⚠️  Skipping Integration Tests due to Unit Test failures');
        results.integration.status = 'skipped';
        totalFailed++;
    }
    
    // Run Performance Tests (independent of other tests)
    if (await runTestSuite('performance', runPerformanceTests)) {
        totalPassed++;
    } else {
        totalFailed++;
    }
    
    // Run Edge Cases Tests (only if unit tests passed)
    if (results.unit.status === 'passed') {
        if (await runTestSuite('edgeCases', runEdgeCasesTests)) {
            totalPassed++;
        } else {
            totalFailed++;
        }
    } else {
        console.log('\n⚠️  Skipping Edge Cases Tests due to Unit Test failures');
        results.edgeCases.status = 'skipped';
        totalFailed++;
    }
    
    const totalTime = Date.now() - startTime;
    
    // Generate comprehensive report
    console.log('\n' + '=' .repeat(80));
    console.log('📊 COMPREHENSIVE TEST REPORT');
    console.log('=' .repeat(80));
    
    console.log('\n🔍 Test Suite Results:');
    console.log('-' .repeat(50));
    
    Object.entries(results).forEach(([name, result]) => {
        const statusIcon = {
            'passed': '✅',
            'failed': '❌',
            'skipped': '⚠️',
            'pending': '⏳'
        }[result.status];
        
        const statusText = result.status.toUpperCase().padEnd(8);
        const durationText = result.duration > 0 ? `${result.duration}ms` : 'N/A';
        
        console.log(`   ${statusIcon} ${name.padEnd(12)} ${statusText} ${durationText.padStart(8)}`);
        
        if (result.error) {
            console.log(`      └─ Error: ${result.error}`);
        }
    });
    
    console.log('\n📈 Summary Statistics:');
    console.log('-' .repeat(50));
    console.log(`   Total Test Suites: ${totalPassed + totalFailed}`);
    console.log(`   Passed: ${totalPassed}`);
    console.log(`   Failed: ${totalFailed}`);
    console.log(`   Success Rate: ${((totalPassed / (totalPassed + totalFailed)) * 100).toFixed(1)}%`);
    console.log(`   Total Duration: ${totalTime}ms (${(totalTime / 1000).toFixed(1)}s)`);
    
    // Detailed breakdown
    console.log('\n🎯 Test Coverage Analysis:');
    console.log('-' .repeat(50));
    
    const coverageAreas = [
        { name: 'Core Functionality', status: results.unit.status },
        { name: 'Network Operations', status: results.integration.status },
        { name: 'Performance Benchmarks', status: results.performance.status },
        { name: 'Edge Cases & Error Handling', status: results.edgeCases.status }
    ];
    
    coverageAreas.forEach(area => {
        const icon = area.status === 'passed' ? '✅' : area.status === 'skipped' ? '⚠️' : '❌';
        console.log(`   ${icon} ${area.name}`);
    });
    
    // Recommendations
    console.log('\n💡 Recommendations:');
    console.log('-' .repeat(50));
    
    if (results.unit.status === 'failed') {
        console.log('   🔴 CRITICAL: Fix unit test failures before proceeding');
        console.log('   🔧 Unit tests validate core WASM functionality');
    }
    
    if (results.integration.status === 'failed') {
        console.log('   🟡 IMPORTANT: Network integration issues detected');
        console.log('   🌐 Check relay connectivity and network configuration');
    }
    
    if (results.performance.status === 'failed') {
        console.log('   🟠 WARNING: Performance benchmarks failed');
        console.log('   ⚡ Review performance characteristics and optimization opportunities');
    }
    
    if (results.edgeCases.status === 'failed') {
        console.log('   🟡 IMPORTANT: Edge case handling needs improvement');
        console.log('   🛡️ Enhance error handling and input validation');
    }
    
    if (totalFailed === 0) {
        console.log('   🎉 EXCELLENT: All test suites passed!');
        console.log('   ✨ The WASM implementation is ready for production use');
        console.log('   🚀 Consider adding more specific test cases for your use case');
    }
    
    // Environment information
    console.log('\n🔧 Test Environment:');
    console.log('-' .repeat(50));
    console.log(`   Node.js Version: ${process.version}`);
    console.log(`   Platform: ${process.platform} ${process.arch}`);
    console.log(`   Memory Usage: ${(process.memoryUsage().heapUsed / 1024 / 1024).toFixed(2)} MB`);
    console.log(`   Test Execution Date: ${new Date().toISOString()}`);
    
    // Create JSON report
    const jsonReport = {
        timestamp: new Date().toISOString(),
        totalDuration: totalTime,
        environment: {
            nodeVersion: process.version,
            platform: process.platform,
            arch: process.arch
        },
        summary: {
            totalSuites: totalPassed + totalFailed,
            passed: totalPassed,
            failed: totalFailed,
            successRate: (totalPassed / (totalPassed + totalFailed)) * 100
        },
        suites: results
    };
    
    // Output JSON report
    console.log('\n📄 JSON Test Report:');
    console.log(JSON.stringify(jsonReport, null, 2));
    
    console.log('\n' + '=' .repeat(80));
    
    if (totalFailed > 0) {
        console.log('❌ TEST SUITE FAILED - Some tests did not pass');
        console.log('🔧 Please review the failures above and fix the issues');
        process.exit(1);
    } else {
        console.log('🎉 ALL TESTS PASSED - WASM implementation is working correctly!');
        console.log('✨ Your Pkarr WASM bindings are ready for use');
        process.exit(0);
    }
}

// Run all tests if this file is executed directly
if (require.main === module) {
    runAllTests().catch(error => {
        console.error('\n💥 Test runner crashed:', error);
        console.error('Stack trace:', error.stack);
        process.exit(1);
    });
}

module.exports = { runAllTests }; 