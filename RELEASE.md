# Release Guide

This document describes the manual release process for the pkarr
repository. All packages follow [Semantic Versioning][semver].

[semver]: https://semver.org/

## Packages

| Package              | Location                     | Registry   |
| -------------------- | ---------------------------- | ---------- |
| `pkarr`              | `pkarr/Cargo.toml`          | crates.io  |
| `pkarr-relay`        | `relay/Cargo.toml`          | crates.io  |
| `@synonymdev/pkarr`  | `bindings/js/pkg/package.json` | npm     |

`pkarr` is the primary crate. The relay and JS bindings track it
loosely and are versioned independently, but any release that
changes `pkarr` should also check whether the dependents need
a bump.

## Semantic Versioning

Every version bump must follow semver strictly:

- **Major** -- breaking API changes.
- **Minor** -- new functionality, backwards-compatible.
- **Patch** -- bug fixes, no API changes.

Pre-release versions (e.g. `5.1.0-rc.1`, `0.1.4-rc.3`) are
allowed for early testing on npm or crates.io.

## Files to Update

### If changed

1. **`pkarr/Cargo.toml`** -- bump `version`.

### When the relay is included in the release

If pkarr is bumped the relay should also be bumped.

2. **`relay/Cargo.toml`** -- bump the crate `version`.
3. **`relay/Cargo.toml`** -- update the `pkarr` dependency version
   to match the new `pkarr` version (if pkarr was bumped).

### When the JS bindings are included in the release

4. **`bindings/js/pkg/package.json`** -- bump `version`.

### Lock file

5. **`Cargo.lock`** -- run `cargo check` (or `cargo build`) after
   editing any `Cargo.toml` so the lock file reflects the new
   versions. Commit the updated lock file in the same PR.

## Release Process

### 1. Dry run

After bumping versions, verify that everything compiles and
packages correctly **before** opening the PR.

```sh
# Rust crates
cargo publish -p pkarr --dry-run
cargo publish -p pkarr-relay --dry-run

# JS bindings (if included in the release)
cd bindings/js/pkg && npm run build
```

Fix any errors before proceeding.

### 2. Open a version-bump PR

Create a branch (e.g. `chore/v5.1.0`) and commit the version
changes listed above. The PR title should follow the pattern:

```
chore: release v5.1.0
```

Include a summary of what changed since the last release in the
PR description.

### 3. Review and merge

Get the PR reviewed and merge it into `main`.

### 4. Create a GitHub release (and tag)

Create the release on the GitHub website -- this also creates the
git tag in one step:

1. Go to **Releases > Draft a new release**.
2. Click **Choose a tag**, type `v5.1.0`, and select
   **Create new tag: v5.1.0 on publish**.
3. Set the target branch to `main`.
4. Set the release title to `v5.1.0`.
5. Click **Generate release notes** to auto-populate the
   description from merged PRs.
6. Edit the generated notes if needed, then **Publish release**.

There is no changelog file; the GitHub release is the canonical
record of what changed.

### 5. Publish packages

Publish **in order** -- `pkarr` first, since the relay depends
on it.

#### a) pkarr (crates.io)

```sh
cargo publish -p pkarr
```

#### b) pkarr-relay (crates.io)

```sh
cargo publish -p pkarr-relay
```

#### c) @synonymdev/pkarr (npm)

Build the WASM bundle first, then publish. From
`bindings/js/pkg/`:

```sh
npm run build
npm publish --access public
```

The build step requires `wasm-pack` and `node` to be installed.
It compiles the Rust WASM target and generates the isomorphic
JS/CJS wrappers.

For pre-release versions, add a dist-tag:

```sh
npm publish --access public --tag rc
```

## Checklist

```
[ ] Version bumped in pkarr/Cargo.toml
[ ] Version bumped in relay/Cargo.toml (if applicable)
[ ] Relay's pkarr dependency version updated (if pkarr bumped)
[ ] Version bumped in bindings/js/pkg/package.json (if applicable)
[ ] Cargo.lock updated (cargo check)
[ ] Dry run passed (cargo publish --dry-run, npm build)
[ ] PR opened, reviewed, and merged
[ ] GitHub release created (tag + release notes)
[ ] pkarr published to crates.io
[ ] pkarr-relay published to crates.io (if applicable)
[ ] @synonymdev/pkarr published to npm (if applicable)
```
