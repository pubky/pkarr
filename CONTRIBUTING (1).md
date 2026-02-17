# Contributing to Pkarr

Thanks for taking the time to contribute! Pkarr is a Rust implementation with JavaScript/WASM bindings that publishes and resolves signed DNS-like records over the Mainline DHT.

- Repo: https://github.com/pubky/pkarr
- License: MIT (inbound = outbound)
- Scope: the Rust crates, relay, JS/WASM bindings, examples, and docs/specs under `/design`.

---

## TL;DR workflow
1) Open/confirm an **Issue** for anything non-trivial.
2) Create a small, focused **PR** from a feature branch.
3) Make CI happy: **fmt + clippy + tests** (and fuzz/benches if touched).
4) Update docs/examples when behavior or APIs change.
5) Expect **review focused on simplicity, safety, and wire-compatibility**.

---

## Ways to contribute
- **Bug reports** — minimal repro, expected vs actual, OS/Rust/Node versions, logs.
- **Features** — state the *problem* first; include an API sketch if relevant.
- **Docs** — clarify README/examples/specs; fix typos; add usage notes.
- **Code** — targeted PRs that improve reliability, performance, or ergonomics without adding unnecessary complexity.

Use **Issues** for bugs/features and **Discussions** for open questions or design exploration.

---

## Ground rules
- Be concise, technical, and respectful. No bikeshedding or personal attacks.
- Prefer simple, auditable solutions over clever complexity.
- Keep Pkarr’s core constraints intact: **small signed packet (≤ ~1000 bytes), heavy caching, minimal DHT chatter**.
- **No panics** (`unwrap`/`expect`) in library paths. Propagate errors with `Result` and meaningful error types.
- New `unsafe` code requires a strong justification, comments, and tests. Avoid where possible.

---

## Compatibility & versioning
- **SemVer** for all published packages (Rust crate and NPM bindings).
- **On‑wire format is sacred.** Any change to encoding, verification, or relay behavior must:
  1. Start with an Issue describing the change and trade‑offs.
  2. Include `/design` docs and migration notes.
  3. Provide tests that prove compatibility or clearly mark breaking behavior.
- **MSRV**: we target the current stable Rust toolchain. We don’t guarantee older compilers.
- **Bindings policy**: JS/WASM public surface should be minimal and stable; avoid leaking Rust internals. Keep NPM major/minor in step with crate whenever APIs change.

---

## Development setup

### Prereqs
- **Rust**: stable toolchain (`rustup`).
- **Node + npm**: for JS/WASM bindings.
- **wasm-pack**: `cargo install wasm-pack` (if touching bindings).
- Optional: **Nix** and/or **Docker** for reproducible envs or container builds.

### Build & test (Rust)
```bash
# format & lint
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# run tests
cargo test --all --all-features

# dependency policy (if you changed dependencies)
cargo install cargo-deny --locked
cargo deny check
```

### Fuzzing (recommended for parsers/verification)
If you modify packet encoding/decoding or signature verification, add or update fuzz targets:
```bash
cargo install cargo-fuzz --locked
cargo fuzz run parse_record
cargo fuzz run verify_signature
```
Keep fuzz harnesses fast and deterministic; minimize seed corpus when possible.

### Benchmarks (for perf‑sensitive code)
Use `criterion` benches where relevant, and include before/after numbers in the PR description. Avoid micro‑optimizations that harm clarity unless the win is material.

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
- [ ] Small, focused change set (split orthogonal changes).
- [ ] `cargo fmt` and `cargo clippy` pass with no warnings.
- [ ] `cargo test --all --all-features` is green locally.
- [ ] If deps changed: `cargo deny check` passes.
- [ ] Added/updated tests (unit/integration; fuzz/bench if relevant).
- [ ] Updated docs (README/examples/`/design`) for any behavior or API changes.
- [ ] No `unwrap`/`expect` in library code.
- [ ] PR description includes rationale, alternatives considered, and any perf data if applicable.

**Commit messages**: use Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `chore:`).  
**Branch names**: `feat/...`, `fix/...`, `chore/...`.

---

## Code style
- Rust: `rustfmt` defaults; small modules; explicit error types; `///` docs with runnable examples when possible.
- JS/WASM: minimal, typed public surface; reproducible builds; avoid exposing Rust-specific types directly.

---

## Security
Please **do not open a public issue** for security reports. Use GitHub’s **“Report a vulnerability”** to contact maintainers privately, or email the maintainers listed in `Cargo.toml`/`README`. We’ll coordinate a fix and responsible disclosure.

---

## CI
All PRs must pass CI (format/lint/tests). Maintainers may push small fixups to your branch (e.g., formatting).

---

## Licensing
By contributing, you agree your contributions are licensed under the repository’s MIT license (inbound = outbound).

---

## Good first issues
Look for the **good first issue** / **help wanted** labels. If something looks approachable but isn’t labeled, ask in the Issue and we’ll guide you.

---

## Maintainers
Project direction and final reviews are handled by the Pubky team. We’re friendly but opinionated about simplicity and maintaining a credible exit.
