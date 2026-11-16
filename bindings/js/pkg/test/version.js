const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "pkarr-version-"));
try {
  fs.mkdirSync(path.join(workspace, "pkarr/src"), { recursive: true });
  fs.mkdirSync(path.join(workspace, "bindings/js/src/bin"), { recursive: true });
  fs.mkdirSync(path.join(workspace, "bindings/js/pkg"), { recursive: true });
  fs.writeFileSync(path.join(workspace, "Cargo.toml"),
    '[workspace]\nmembers = ["pkarr"]\nresolver = "2"\n' +
    '[workspace.package]\nversion = "10.0.0-beta.1"\n');
  fs.writeFileSync(path.join(workspace, "pkarr/Cargo.toml"),
    '[package]\nname = "pkarr"\nversion.workspace = true\nedition = "2021"\n');
  fs.writeFileSync(path.join(workspace, "pkarr/src/lib.rs"), "");
  const manifest = path.join(workspace, "bindings/js/pkg/package.json");
  fs.writeFileSync(manifest, '{"name":"example","version":"1.0.0"}\n');
  const script = path.join(workspace, "bindings/js/src/bin/sync-version.mjs");
  fs.copyFileSync(path.join(__dirname, "../../src/bin/sync-version.mjs"), script);

  execFileSync(process.execPath, [script]);
  const pkg = JSON.parse(fs.readFileSync(manifest, "utf8"));
  assert.equal(pkg.version, "10.0.0-beta.1");
  assert.equal(pkg.name, "example");
} finally {
  fs.rmSync(workspace, { recursive: true, force: true });
}
