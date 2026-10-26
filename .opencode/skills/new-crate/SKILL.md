---
name: New Crate
description: Use when adding a new crate to the Codevar workspace, or when asked to scaffold a module, library, or package. Produces a crate that matches the workspace conventions - lint inheritance, no_std, license header, workspace dependency style, and a registered workspace member.
---

# New Crate

New crates are how Codevar stays reviewable: each crate is an isolation
boundary that lets an agent work on one concern without touching the other 52.

**Only one agent may add a crate at a time.** Concurrent additions collide in
`Cargo.lock`. If another agent is mid-change in the workspace `Cargo.toml`,
wait for them or serialize the change.

## Checklist

- [ ] `crates/codevar-<name>/Cargo.toml` created
- [ ] `crates/codevar-<name>/src/lib.rs` created with the Apache 2.0 header
- [ ] Registered in the workspace `members` list of the root `Cargo.toml`
- [ ] Registered in the workspace `[workspace.dependencies]` table of the root `Cargo.toml`
- [ ] Every dependency declared as `workspace = true`, not a bare version
- [ ] `[lints] workspace = true` present
- [ ] `no_std` unless the crate genuinely needs `std`
- [ ] A `description` in `Cargo.toml`
- [ ] Public items documented
- [ ] Scoped checks pass: `cargo clippy -p codevar-<name> --all-targets -- -D warnings`

## Cargo.toml template

```toml
[package]
name = "codevar-<name>"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
description = "<one line: what this crate is for>"

[lib]
name = "codevar_<name>"

[dependencies]
codevar-logger = { workspace = true }

[lints]
workspace = true
```

Add the same crate to the root `Cargo.toml` in two places:

```toml
[workspace]
members = [
    ...
    "crates/codevar-<name>",
]

[workspace.dependencies]
codevar-<name> = { path = "crates/codevar-<name>" }
```

## lib.rs template

```rust
//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! <What this crate provides.>

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc; // only if the crate needs the allocator
```

- `#![cfg_attr(not(test), no_std)]` keeps the crate usable on the bare-metal
  and WASM targets. Add `extern crate alloc;` only when the crate actually
  allocates.
- `#![forbid(unsafe_code)]` is the default for new crates. Remove it only
  when there is a concrete reason, and say so in the crate docs.
- `#![warn(missing_docs)]` makes CI complain about undocumented public items.

## Positioning in the dependency graph

Codevar's crates form 13 dependency levels. A new crate must not depend on
anything that would create a cycle, and must not become a dependency hub.

Rules:

- A low-level crate (allocator, codec, parser) depends on `codevar-logger` at
  most. It must not depend on `codevar-base`, `codevar-io`, or anything
  higher up.
- A high-level crate may depend on low-level crates.
- Never create a crate that everything depends on unless it is a genuine
  primitive. `codevar-base` is the current hub; do not add a second one.

Check the placement before you commit:

```bash
.opencode/skills/crate-context/scripts/crate-info.sh codevar-<name>
```

The dependents list must be short and sensible.

## Verify

```bash
cargo check -p codevar-<name>
cargo clippy -p codevar-<name> --all-targets -- -D warnings
cargo test  -p codevar-<name>
cargo build -p codevar-<name> --no-default-features   # confirm the no_std claim
```

## Commit

The crate skeleton and its first implementation are separate commits:

```
codevar-<name>: scaffold the crate

Registers the crate in the workspace and adds the standard lib.rs with
the license header, no_std, and forbid(unsafe_code). No behavior yet.

Refs: #<issue>
```

```
codevar-<name>: implement <the first real thing>
```

Keep each within the commit budget: 800 added lines, 3 files, one crate.
