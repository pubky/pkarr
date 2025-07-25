// Node.js CommonJS header for Pkarr WASM bindings
// This header is prepended to the generated JavaScript to ensure proper CommonJS compatibility

const __dirname = typeof __dirname !== 'undefined' ? __dirname : (function() {
    if (typeof module !== 'undefined' && module.exports) {
        // Running in Node.js
        const path = require('path');
        return path.dirname(__filename);
    }
    return '';
})();

const imports = {}; 