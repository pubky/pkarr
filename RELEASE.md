# Release Guide

This document describes the release process for the pkarr
repository. All packages follow [Semantic Versioning][semver].

[semver]: https://semver.org/

## Packages

| Package              | Location                     | Registry   |
| -------------------- | ---------------------------- | ---------- |
| `pkarr`              | `pkarr/Cargo.toml`          | crates.io  |
| `pkarr-relay`        | `relay/Cargo.toml`          | crates.io  |
| `@synonymdev/pkarr`  | `bindings/js/pkg/package.json` | npm     |

`pkarr` is the primary crate. A single GitHub release publishes all three
packages. The npm package uses the same version as `pkarr`, while
`pkarr-relay` is versioned independently. A major relay release can ship
alongside a patch release of `pkarr`.

## Semantic Versioning

Every version bump must follow semver strictly:

- **Major** -- breaking API changes.
- **Minor** -- new functionality, backwards-compatible.
- **Patch** -- bug fixes, no API changes.

Pre-release versions (e.g. `5.1.0-rc.1`, `0.1.4-rc.3`) are
allowed for early testing on npm or crates.io.

## Files to Update

1. **`pkarr/Cargo.toml`** -- bump `version`. The GitHub tag must match this
   version.
2. **`relay/Cargo.toml`** -- bump its independent `version` and update its
   `pkarr` dependency requirement to the new `pkarr` version.
3. **`bindings/js/pkg/package.json`** -- set `version` to the new `pkarr`
   version.

### Lock file

4. **`Cargo.lock`** -- run `cargo check` (or `cargo build`) after
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

# JS bindings
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

Create the release on the GitHub website -- this also creates the git tag in
one step. The tag matches the `pkarr` and npm versions. For example, `v8.0.2`
can publish `pkarr` 8.0.2, `pkarr-relay` 3.0.0, and `@synonymdev/pkarr` 8.0.2.

1. Go to **Releases > Draft a new release**.
2. Click **Choose a tag**, type `v8.0.2`, and select
   **Create new tag: v8.0.2 on publish**.
3. Set the target branch to `main`.
4. Set the release title to the tag name.
5. Click **Publish release**. The release workflow generates the changelog
   from Git history and adds it to the release description.

There is no changelog file; the GitHub release is the canonical
record of what changed. Publishing a `v*` tag triggers the release workflow,
which validates the tag, publishes all three packages, and attaches relay
binary archives for Linux amd64/arm64, Windows amd64, and macOS amd64/arm64
to the GitHub release. The release notes list each package version.

### 5. Publish packages

The release workflow publishes `pkarr` before `pkarr-relay` so the relay's
dependency is available on crates.io. It also builds and publishes the npm
package with provenance. Stable npm versions use the `latest` dist-tag;
prereleases use `next`. The relay Docker image uses the GitHub release tag; the
release notes identify the relay crate version.

Before the first automated release, configure trusted publishing for the `pkarr`
and `pkarr-relay` crates on crates.io and for `@synonymdev/pkarr` on npm. Each
publisher must trust `.github/workflows/release.yml` in this repository.

The macOS builds use the private GHCR images listed in `Cross.toml`, shared
with `pubky-homeserver`. Grant `pubky/pkarr` Actions access to both packages
in their GitHub package settings. For local builds, authenticate to `ghcr.io`
with an account that can pull these images before running
`.scripts/build-relay-artifacts.sh`.

## Checklist

```
[ ] Version bumped in pkarr/Cargo.toml
[ ] Version bumped in relay/Cargo.toml
[ ] Relay's pkarr dependency requirement updated to the new pkarr version
[ ] bindings/js/pkg/package.json version matches pkarr
[ ] Cargo.lock updated (cargo check)
[ ] Dry run passed (cargo publish --dry-run, npm build)
[ ] PR opened, reviewed, and merged
[ ] GitHub release created (tag + release notes)
[ ] Release workflow published both crates to crates.io
[ ] Release workflow published the npm package
[ ] Relay binary archives attached to the GitHub release
[ ] Relay Docker image published with the GitHub release tag
```
