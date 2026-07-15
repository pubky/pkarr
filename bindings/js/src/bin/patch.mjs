// This script is used to generate isomorphic code for web and nodejs
//
// Based on hacks from [this issue](https://github.com/rustwasm/wasm-pack/issues/1334)

import { readFile, writeFile, rename } from "node:fs/promises";
import { fileURLToPath } from 'node:url';
import path, { dirname } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const cargoTomlContent = await readFile(path.join(__dirname, "../../Cargo.toml"), "utf8");
const cargoPackageName = /\[package\]\nname = "(.*?)"/.exec(cargoTomlContent)[1]
const name = cargoPackageName.replace(/-/g, '_')

const content = await readFile(path.join(__dirname, `../../pkg/nodejs/${name}.js`), "utf8");
const exportedNames = Array.from(
  content.matchAll(/^exports\.([A-Za-z_$][\w$]*) = \1;$/gm),
  match => match[1],
).filter(exportName => exportName !== "__wasm");
const wasmBytesVariable = content.includes("const wasmBytes =") ? "wasmBytes" : "bytes";

const patched = content
  // Named exports are appended below in ESM syntax.
  .replace(/^exports\.[A-Za-z_$][\w$]* = [A-Za-z_$][\w$]*;\n?/gm, "")
  // inline wasm bytes
  .replace(
    /\nconst (?:path|wasmPath).*\nconst (?:bytes|wasmBytes).*\n/,
    `
var __toBinary = /* @__PURE__ */ (() => {
  var table = new Uint8Array(128);
  for (var i = 0; i < 64; i++)
    table[i < 26 ? i + 65 : i < 52 ? i + 71 : i < 62 ? i - 4 : i * 4 - 205] = i;
  return (base64) => {
    var n = base64.length, bytes = new Uint8Array((n - (base64[n - 1] == "=") - (base64[n - 2] == "=")) * 3 / 4 | 0);
    for (var i2 = 0, j = 0; i2 < n; ) {
      var c0 = table[base64.charCodeAt(i2++)], c1 = table[base64.charCodeAt(i2++)];
      var c2 = table[base64.charCodeAt(i2++)], c3 = table[base64.charCodeAt(i2++)];
      bytes[j++] = c0 << 2 | c1 >> 4;
      bytes[j++] = c1 << 4 | c2 >> 2;
      bytes[j++] = c2 << 6 | c3;
    }
    return bytes;
  };
})();

const ${wasmBytesVariable} = __toBinary(${JSON.stringify(await readFile(path.join(__dirname, `../../pkg/nodejs/${name}_bg.wasm`), "base64"))
    });
`,
  );

if (exportedNames.length === 0) {
  throw new Error("No wasm-bindgen exports found");
}

const exports = `
const imports = { ${exportedNames.join(", ")} };
export { ${exportedNames.join(", ")} };
export default imports;
globalThis['pubky'] = imports;
`;

await writeFile(path.join(__dirname, `../../pkg/index.js`), patched + exports);

// Move outside of nodejs
await Promise.all([".js", ".d.ts", "_bg.wasm"].map(suffix =>
  rename(
    path.join(__dirname, `../../pkg/nodejs/${name}${suffix}`),
    path.join(__dirname, `../../pkg/${suffix === '.js' ? "index.cjs" : (name + suffix)}`),
  ))
)
