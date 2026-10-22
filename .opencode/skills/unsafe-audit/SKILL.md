---
name: Unsafe Audit
description: Use when adding, modifying, or reviewing unsafe code in the Codevar repository, when a SAFETY comment is missing or wrong, or when auditing a crate's unsafe surface. Codevar has over 2600 unsafe blocks and 300 unsafe fns, concentrated in a few crates - this skill covers the safety contract, the documentation format, and the Miri check.
---

# Unsafe Audit

Codevar contains **thousands of `unsafe` blocks and `unsafe fn`** across the workspace.

Unsafe code is where an agent's mistakes become silent memory corruption instead of a compile error. Treat it as the
highest-risk surface in the repo.

## The safety contract

Every `unsafe` item must carry a `# Safety` section (for `unsafe fn`) or a
`// SAFETY:` comment (for an `unsafe` block) that states the **preconditions the caller must uphold**. A comment that
restates the code is worthless.

```rust
/// # Safety
///
/// `ptr` must point to `count` initialized, properly aligned `T` values that
/// are valid for reads of `size_of::<T>() * count` bytes. The caller must
/// ensure the region is not concurrently mutated for the duration of the call.
#[inline]
unsafe fn read_array<T>(ptr: *const T, count: usize) -> Vec<T> { /* ... */ }
```

```rust
// SAFETY: `buf` was resized to `len` above and `dst` is derived from the
// same allocation, so the two regions cannot overlap. The pointer stays
// valid for the duration of the copy because `buf` is borrowed immutably.
unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
```

A good safety comment names: the pointer's provenance, the required alignment, the length bound, the aliasing rule, the
lifetime, and the threading model — whichever of those apply.

## Rules

- **Prefer safe code (if know unsafe will don't give any benefits of performance).** If a safe alternative exists, use
  it. `unsafe` is not a performance shortcut you may take on faith; it requires a benchmark.
- **Keep unsafe blocks minimal.** One operation per block, with the comment attached to that operation.
- **No `unsafe fn` in a public API unless the crate is a low-level one.**
  A public `unsafe fn` exports an obligation to every caller. Prefer
  `unsafe fn` private to the module plus a safe public wrapper that establishes the invariants.
- **Document the invariants, not the implementation.**
- **`panic!` and `abort!` are forbidden** in production code, including in
  `unsafe` blocks. An `unsafe` block that can panic must not be relied upon for memory safety.
- **Never add `#[allow(clippy::undocumented_unsafe_blocks)]`** to silence the missing-safety-comment lint. Add the
  comment.

## Verification

```bash
# Scoped build and tests for the crate
cargo clippy -p <crate> --all-targets -- -D warnings
cargo test -p <crate>

# UB detection - the only tool that actually checks this
cargo +nightly miri test -p <crate>

# Lint for missing SAFETY comments (needs nightly)
cargo +nightly clippy -p <crate> -- -W clippy::undocumented_unsafe_blocks
```

Miri is slow. Run it on the crate you touched, not the workspace. If the crate needs an unsupported feature (e.g. FFI to
a C library), Miri will not run — say so in your report rather than skipping it silently.

## Auditing an existing crate

1. List the unsafe surface:
   ```bash
   grep -rn 'unsafe' crates/<crate>/src --include='*.rs' | head -50
   ```
2. For each `unsafe fn`, check that it has a `# Safety` section.
3. For each `unsafe` block, check that it has a `// SAFETY:` comment.
4. Check the comment against the actual code — stale comments are worse than missing ones, because the next agent trusts
   them.
5. Report the gaps. A missing `SAFETY:` comment is a documentation task and a great candidate for a **mechanical,
   per-file commit** that stays well inside the commit budget.

## Report format

```
codevar-tlsf-alloc: audit of the free-list unsafe surface

Blocks audited   : 126
With SAFETY note : 31
Missing          : 95
Stale (present but does not match the code): 4

The 4 stale comments are the real problem: they assert the allocator
never frees a block twice, but two of them guard the size-class path
that was rewritten in #881.
```
