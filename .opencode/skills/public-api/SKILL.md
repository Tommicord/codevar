---
name: Public API
description: Use when changing a public function signature, struct field, enum variant, or trait in the Codevar repository, or when reviewing whether a change breaks callers. Codevar exposes over 4000 public functions across 53 crates, so a signature change silently breaks dependents that are no longer in the working tree.
---

# Public API

Codevar exposes **4,095 public functions and 1,061 public types** across 53
crates. Because the workspace is split into many small crates, a change to one
crate's public API breaks its dependents at compile time — but only the ones
that happen to be built in the same run.

## Before changing a public item

1. **Find the callers.** Do not guess:
   ```bash
   .opencode/skills/crate-context/scripts/crate-info.sh codevar-tls
   ```
   The "workspace dependents" list tells you which crates must be rebuilt.

2. **Check whether the change can be additive.** Prefer this order:
   - Add a new item, deprecate the old one. (best)
   - Add a new item, keep the old one delegating to it.
   - Change the signature. (last resort)

3. **Consider the semver impact.** This is a `0.x`-era project on a
   `1.0.0` workspace version, but the crates are consumed internally and the
   API is large. A breaking change to a widely used item — `codevar-time-util`
   has 418 public functions, `codevar-wredit` 392, `codevar-tls` 296 — is a
   large, review-heavy diff. Say so before making it.

## After changing a public item

Rebuild every dependent crate, not just the one you edited:

```bash
crates=$(git diff --name-only --diff-filter=ACMR HEAD | cut -d/ -f2 | sort -u | sed 's/^/-p /')
cargo check $crates
cargo clippy $crates --all-targets -- -D warnings
cargo test  $crates
```

`cargo check -p codevar-tls` already builds the workspace crates it depends
on, but it does **not** build crates that depend on `codevar-tls`. Those are
the ones your change breaks. Always include them explicitly.

## Detecting API drift

`cargo-public-api` produces a machine-readable snapshot of a crate's public
surface. Comparing snapshots tells you whether a change altered the contract
without reading a single implementation line.

```bash
# Install once
cargo install cargo-public-api

# Snapshot the current API
cargo public-api --manifest-path crates/codevar-tls/Cargo.toml

# Diff against a stored snapshot
cargo public-api diff --manifest-path crates/codevar-tls/Cargo.toml
```

If the tool is not installed, fall back to a manual review of the public
items in the diff:

```bash
git diff -U0 HEAD -- crates/codevar-tls/src \
  | grep -E '^\+.*\bpub\b' | grep -vE '^\+\s*//'
```

That output is the public surface delta. Every line in it is a promise you are
making or breaking.

## Reviewing a public API change

For each changed public item, the reviewer must be able to answer:

- [ ] What breaks for an existing caller, and is the breakage visible at
      compile time or only at runtime?
- [ ] Is the new signature still `no_std` compatible?
- [ ] Does the doc comment state the new contract, including the new
      failure modes?
- [ ] Is the change consistent with the other items in the same module?
- [ ] Does the change belong in this crate, or does it need a new one?

## Report format

```
Public API change in codevar-tls:

  Session::handshake(&mut self, cfg: &Config) -> Result<Session, TlsError>
    -> Session::handshake(&mut self, cfg: &Config, now: Timestamp) -> Result<...>

  Breaking. `now` is required to expire session tickets without calling the
  clock, which the bare-metal target cannot do. Affected crates:
  codevar-ocl-tc, codevar-wsocket. Both updated in the same branch.
```

State plainly whether the change is breaking. Never let the reviewer
discover it.
