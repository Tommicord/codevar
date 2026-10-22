---
name: Crate Context
description: Use before editing any crate in the Codevar repository, or when asked what a crate does, what depends on it, or what its public API looks like. Loads only the context needed for the task - crate purpose, public API surface, dependencies, invariants, unsafe hotspots - instead of reading the whole codebase.
---

# Crate Context

Codevar has so many lines of code, **Never read the whole repository and
never paste a whole crate into context.** Load the minimum context that lets
you make the edit safely, then edit.

## Step 1 - locate the crate

```bash
.opencode/skills/crate-context/scripts/crate-info.sh codevar-tls
```

The script prints, for one crate:

- purpose (the `description` from `Cargo.toml`)
- line counts per file, largest first
- dependencies (workspace crates only) and workspace dependents
- public items (`pub fn`, `pub struct`, `pub enum`, `pub trait`, `pub const`)
- unsafe hotspots: count of `unsafe` blocks and `unsafe fn` per file
- test files and `#[test]` count

Run it without an argument to list every crate with its size and dependency
depth.

## Step 2 - read the code, in this order

1. `Cargo.toml` of the crate — features, dependencies, lints.
2. `src/lib.rs` — the module tree and re-exports. This is the crate's map.
3. The one or two files you will actually change.
4. The doc comments on the public items you touch — they are the contract.

Do not read a file you are not changing. If you need to know "who calls this",
use the `explore` subagent instead of grepping the workspace by hand:

```
Use the explore subagent to find every call site of `Session::handshake` in the
workspace and report the file paths and line numbers.
```

## Step 3 - know the invariants

Read the safety contract of every `unsafe` item you touch. If the crate has no
documented invariant for an `unsafe` block you are modifying, that gap is part
of your change — see the `unsafe-audit` skill.

Watch for these repo-wide rules while reading:

- No `.unwrap()`, `.expect()`, `panic!`, `assert!`, `unreachable!`,
  `println!`, `eprintln!` in production code. The workspace denies
  `unwrap_used` and `expect_used` via clippy; the rest is enforced by review.
- Errors propagate with `?`; fallbacks use `unwrap_or` / `unwrap_or_else`.
- Logging goes through `codevar_logger` macros, never `println!`.
- Crates are `no_std` unless they explicitly opt out. Do not pull in `std`.
- New public items need doc comments. New files need the Apache 2.0 header.

## Step 4 - check the blast radius

Before changing a `pub` item, find out who depends on the crate and who calls
the item. A signature change to a `pub fn` in `codevar-tls` affects every
dependent crate. The `public-api` skill covers that case in detail.

## Output

Before editing, be able to state in one or two sentences:

- which crate owns the change and which file inside it,
- what the current behavior is,
- what the new behavior will be,
- which public items (if any) change signature,
- how you will verify it.
