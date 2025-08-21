# Contributing to Pkarr

Thanks for taking the time to contribute! Pkarr is a Rust implementation with JavaScript/WASM bindings that publishes and resolves signed DNS records over Mainline DHT.

- Repo: https://github.com/pubky/pkarr
- License: MIT (inbound = outbound)
- Scope: the Rust crate(s), the relay, JS/WASM bindings, examples, and docs/specs under `/design`.

---

## Ways to contribute

- **Bug reports** — minimal repro steps, expected vs actual, OS/Rust/Node versions, logs.
- **Feature requests** — clearly state the problem, not just the solution. Add rough API sketch if applicable.
- **Docs** — clarify README/examples/specs; fix typos; add usage notes.
- **Code** — focused pull requests that improve reliability, performance, or ergonomics without bloating complexity.

Use **Issues** for bugs/features and **Discussions** for open questions or design exploration.

---

## Ground rules

- Be concise, technical, and respectful. No bikeshedding or personal attacks.
- Prefer simple, auditable solutions over clever complexity.
- Changes must keep Pkarr’s core constraints intact (small signed packet ≤ 1000 bytes; heavy caching; minimal DHT chatter).

---

## Development setup

### Prereqs
- **Rust**: stable toolchain (`rustup`).
- **Node + npm**: for JS/WASM bindings.
- **wasm-pack**: `cargo install wasm-pack` (if touching bindings).
- Optional: **Nix** and/or **Docker** if you prefer reproducible envs or container builds.

### Build & test (Rust)
```bash
# format & lint
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# run tests
cargo test --all --all-features

# dependency policy (only if you changed dependencies)
cargo install cargo-deny --locked
cargo deny check
```

### JS/WASM bindings
```bash
# build WASM package (adjust path if needed)
wasm-pack build bindings/js --release

# optional: run JS tests if present
cd bindings/js
npm ci
npm test
```

### Relay (local)
```bash
# build relay binary
cargo build -p pkarr-relay

# or via Docker (example)
docker build -t pkarr .
docker run --rm -p 6881:6881 -v $(pwd)/config.toml:/config.toml -v $(pwd)/.pkarr_cache:/cache pkarr   pkarr-relay --config=/config.toml
```

---

## Pull request checklist

Before you open a PR:

- [ ] The change set is **small and focused**. Separate orthogonal changes.
- [ ] `cargo fmt` and `cargo clippy` pass with no warnings.
- [ ] `cargo test --all --all-features` passes locally.
- [ ] If deps changed: `cargo deny check` passes.
- [ ] Docs updated (README/examples/specs under `/design`) if behavior or API changes.
- [ ] Added tests (unit/integration) where it makes sense.
- [ ] No `unwrap()`/`expect()` in library code paths; handle errors with `Result` and meaningful error types.

**Commit messages**: follow a Conventional-Commits-style prefix when possible (`feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `chore:`). It helps with release notes and history triage.

**Branch naming**: `feat/...`, `fix/...`, or `chore/...`.

---

## API & spec changes

- Any change that affects the on-wire format, verification rules, or relay behavior requires:
  1. An **Issue** describing the change and trade-offs.
  2. Updates to the relevant docs in `/design` (and examples).
  3. Tests demonstrating compatibility or clearly marking breaking behavior.

We use **SemVer**:
- Breaking changes → major bump
- Additive, backward-compatible features → minor
- Fixes/internal refactors with no API change → patch

---

## Performance & security

- Keep packets small; avoid increasing DHT load.
- Cache aggressively where appropriate; don’t build chatty loops.
- Never commit secrets or real private keys. Test keys must be ephemeral.
- Be conservative with unsafe code (ideally none). If unavoidable, document why and add tests.

---

## Reporting vulnerabilities

Please **do not open a public issue** for security reports.
Use GitHub’s *“Report a vulnerability”* (Security Advisories) to contact maintainers privately, or email the maintainers listed in `Cargo.toml`/`README` if needed. We’ll coordinate a fix and a responsible disclosure.

---

## Code style notes

- Rust: `rustfmt` defaults; small modules; explicit error types; docs with `///`; runnable examples preferred.
- JS/WASM: keep the public surface minimal and typed; don’t leak Rust internals into the JS API; keep build scripts reproducible.

---

## CI

All PRs must pass CI (format/lint/tests). Maintainers may push small fixups to your branch (e.g., formatting).

---

## Licensing

By contributing, you agree your contributions are licensed under the repository’s MIT license.

---

## Good first issues

Look for the **good first issue** / **help wanted** labels. If something looks approachable but isn’t labeled, ask in the Issue and we’ll guide you.

---

## Maintainers

Project direction and final reviews are handled by the Pubky team. We’re friendly but opinionated about simplicity and maintaining a credible exit.
