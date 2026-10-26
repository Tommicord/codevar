---
name: Verify
description: Use before declaring a task complete, before every commit, or when a cargo check, clippy, or test run fails. Runs the smallest check that proves the work - scoped per-crate check, clippy and test for the crates touched by the diff - then the full workspace check once at the end.
---

# Verify

Codevar has many crates with thousands of lines of code that cannot be readed at once. A full-workspace `cargo clippy`
takes minutes; a per-crate one takes seconds. Verification must be **fast enough to run after every edit** and
**complete enough to trust at the end**.

The rule: **only CI decides whether the work is complete.** Never report a task as done because the code "looks right".

## Tier 1 - scoped checks (after every edit, before every commit)

```bash
.opencode/skills/verify/scripts/affected-crates.sh
```

The script determines which crates the working tree touched and runs
`cargo check`, `cargo clippy --all-targets -- -D warnings`, and `cargo test`
for exactly those crates. Options:

```bash
affected-crates.sh --list           # print the affected crates and exit
affected-crates.sh --check          # cargo check only
affected-crates.sh --clippy         # cargo clippy only
affected-crates.sh --test           # cargo test only
affected-crates.sh --workspace      # full workspace instead of scoped
```

To scope manually:

```bash
crates=$(git diff --name-only --diff-filter=ACMR HEAD | cut -d/ -f2 | sort -u | sed 's/^/-p /')
cargo clippy $crates --all-targets -- -D warnings
```

## Tier 2 - full workspace (once, before the task is done)

```bash
cargo fmt --all -- --check
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

CI runs exactly these four (`.github/workflows/rust.yml`). If the workspace is already red for reasons unrelated to your
change, say so explicitly in your final report — do not silently ignore a failure, and do not "fix" unrelated code to
make it green. That is a separate task and a separate commit.

## Tier 3 - targeted checks (when the change warrants them)

| Change            | Extra check                                                                  |
|-------------------|------------------------------------------------------------------------------|
| `unsafe` code     | `cargo +nightly miri test -p <crate>` — see the `unsafe-audit` skill         |
| Parser or decoder | Property/fuzz test the input boundary; feed truncated and malformed input    |
| Cryptography      | Constant-time review; test against known-answer vectors                      |
| Performance       | `cargo bench -p <crate>` before and after; report the numbers                |
| `no_std` crate    | `cargo build -p <crate> --no-default-features` and confirm no `std` crept in |
| Public API change | `cargo public-api diff` — see the `public-api` skill                         |

## Reading failures

1. **Read the first error, not the last.** Cargo stops after the first failing crate; later errors are usually
   consequences.
2. Distinguish *your* failure from *pre-existing* failure. Check with
   `git stash` + rerun, or reproduce on `HEAD` before you touched it.
3. Never work around a failure by weakening a lint, adding `#[allow(...)]`
   without a reason comment, deleting a test, or loosening a safety check. Each of those is a silent correctness
   regression.

## Reporting

State the evidence, not the intention:

```
Verified: cargo clippy -p codevar-tls --all-targets -- -D warnings (clean),
cargo test -p codevar-tls (34 passed, 0 failed).
```

Not:

```
Verified: the code compiles and looks correct.
```

If a check could not be run, say which one and why.
