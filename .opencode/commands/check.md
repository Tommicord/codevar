---
description: Run scoped cargo check, clippy, and test for the crates affected by the current changes
agent: build
---

Run the verification loop for the current work.

Affected crates:

!`bash .opencode/skills/verify/scripts/affected-crates.sh --list`

Load the `verify` skill and follow it.

## Steps

1. Run the scoped checks for the affected crates:
   ```
   .opencode/skills/verify/scripts/affected-crates.sh
   ```

2. If something fails, report the **first** failure in full — file, line, and
   the complete diagnostic. Later errors in a cargo run are usually
   consequences of the first.

3. Classify each failure as caused by the change or pre-existing. To check
   whether a failure is pre-existing, stash the working tree, rerun, and
   restore it:
   ```
   git stash push --include-untracked
   <the failing command>
   git stash pop
   ```
   Never "fix" a pre-existing failure as part of an unrelated task.

4. Never work around a failure by adding `#[allow(...)]` without a reason,
   deleting a test, weakening a lint, or loosening a safety check.

5. Report the evidence:
   ```
   Scope: codevar-tls
     cargo check  : PASS
     cargo clippy : PASS
     cargo test   : PASS (34 tests, 0 failures)
   ```

Extra checks to run when the change warrants them:

| Change         | Check                                                             |
|----------------|-------------------------------------------------------------------|
| `unsafe` code  | `cargo +nightly miri test -p <crate>`                             |
| `no_std` crate | `cargo build -p <crate> --no-default-features`                    |
| Performance    | `cargo bench -p <crate>`                                          |
| Public API     | `cargo public-api diff --manifest-path crates/<crate>/Cargo.toml` |

If an extra check cannot run, say which one and why.
