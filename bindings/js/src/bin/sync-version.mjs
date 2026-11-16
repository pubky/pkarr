import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const workspaceManifest = fileURLToPath(new URL("../../../../Cargo.toml", import.meta.url));
const packageManifest = fileURLToPath(new URL("../../pkg/package.json", import.meta.url));

const metadata = JSON.parse(execFileSync("cargo", [
  "metadata", "--no-deps", "--format-version", "1", "--manifest-path", workspaceManifest,
], { encoding: "utf8" }));
const pkarr = metadata.packages.find(pkg => pkg.name === "pkarr");
if (!pkarr) throw new Error("pkarr is missing from Cargo metadata");

const contents = await readFile(packageManifest, "utf8");
const packageJson = JSON.parse(contents);
if (packageJson.version !== pkarr.version) {
  packageJson.version = pkarr.version;
  await writeFile(packageManifest, `${JSON.stringify(packageJson, null, 2)}\n`);
  console.log(`Synchronized npm package version to ${pkarr.version}`);
}
