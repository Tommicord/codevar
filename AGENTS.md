# AGENTS.md

Instructions for AI coding agents working in the Codevar repository.

## Project overview

Codevar is a high-performance code editor targeting for WASM (web) with plans for native desktop and mobile
applications. The core library is written in Rust and the project is in early development.

### Key Features

- **WebAssembly Target**: Optimized for browser-based deployment with wasm-pack
- **Vulkan Graphics**: High-performance rendering using Vulkan graphics API
- **Collaborative Editing**: Real-time collaborative editing with CRDT-like conflict resolution
- **Run everywhere goal** Made for all operating systems including less common
- **AI Integration**: Designed for future integration with AI agents like Claude Code
- **High Performance**: SIMD optimizations and GPU compute shader support for parallel algorithms

## Environment

- **Rust**: 1.93.0 (pinned in `rust-toolchain.toml`)
- **Edition**: 2024
- **License**: Apache-2.0 — preserve the copyright header when creating new source files

## Commands

Run from the repository root:

```bash
# Build
cargo build --workspace

# Run all tests
cargo test --workspace

# Format (must pass in CI)
cargo fmt --all

# Lint (must pass in CI; warnings are errors)
cargo clippy --workspace --all-targets -- -D warnings
```

CI (`.github/workflows/rust.yml`) runs build, test, `cargo fmt --check`, and clippy on every push/PR to `main`.

### Scoped commands (preferred while working)

The workspace has 53 crates and ~215k lines of code. A full-workspace check costs
minutes; a per-crate check costs seconds. Scope every command to the crates you
actually touched, and run the full workspace check once at the end of the task.

```bash
# Which crates did the working tree touch?
git diff --name-only --diff-filter=ACMR HEAD | cut -d/ -f2 | sort -u

# Build + lint + test only those crates
cargo check  -p codevar-tls -p codevar-io
cargo clippy -p codevar-tls -p codevar-io --all-targets -- -D warnings
cargo test   -p codevar-tls -p codevar-io
```

### Diff size check (run before every commit)

```bash
git diff --cached --shortstat   # "3 files changed, 120 insertions(+)"
```

If the staged diff exceeds the commit budget in **Commit discipline** below, split
it before committing. Never accumulate a large uncommitted change and commit it
once at the end.

## AI agent operating rules

These rules exist because this repository is developed primarily with AI agents.
They are not stylistic preferences — violating them makes the codebase unmaintainable
for a single human reviewer.

### Scope discipline

- **One task = one crate or one coherent concern.** Do not refactor unrelated code.
- **Do not touch files outside the scope of the task.** If a change requires touching
  another crate's public API, stop and report it instead of silently widening the diff.
- **Read before writing.** Match existing patterns in the surrounding code; do not
  invent new abstractions or module prefixes.

### Commit discipline

The single hardest constraint in this repository: **a commit must be small enough for a
human to review**. The repository historically averaged ~3,700 added lines per commit,
which is roughly 10–19 hours of expert review each — impossible for one person, and
impossible to bisect when a regression lands. That is now forbidden.

**Hard limits per commit:**

| Limit | Value |
| --- | --- |
| Added lines | **≤ 800** (excluding `Cargo.lock`) |
| Files changed | **≤ 3**, or ≤ 20 for a mechanical sweep that stays under 800 lines |
| Crates touched | **1** (a second crate requires a separate commit) |
| Purpose | **One** — never mix a refactor with a behavior change |

**When to commit:**

- Commit as soon as the current step compiles and its scoped tests pass. Do **not**
  save everything up for one large commit at the end of a task.
- Commit every ~400 lines of new work, or when a coherent subtask completes —
  whichever comes first.
- If the working tree approaches the limit, **stop implementing and commit the green
  state**, then continue. A green checkpoint always beats a bigger uncommitted pile.
- If a step cannot be made to compile, the step is too large. Reduce its size.

**Never:**

- Never leave more than one logical step uncommitted.
- Never commit code that does not compile or whose scoped tests fail.
- Never amend, rebase, or squash a commit that has already been pushed. Add a new commit.
- Never use `WIP`, `fix stuff`, `updates`, or a bullet dump of filenames as a message.

**Commit message format:**

```
<scope>: <imperative summary, ≤ 72 characters>

<why the change was needed and what it does, wrapped at 72 characters, ≤ 6 lines>
```

- `<scope>` is the crate or module name (`codevar-tls`, `ocl-ir`, `logger`).
- The summary states the *effect*, not the activity: `codevar-tls: reject
  renegotiation after handshake failure`, not `update tls code`.
- The body explains **why**, and states what a reviewer should check.
- Reference issues as `Refs: #123` on the last line.

Examples:

```
codevar-tls: reject renegotiation after handshake failure

RFC 8446 forbids renegotiation after a completed handshake. Accepting it
let a peer reset the cipher suite mid-session, which invalidated the
session ticket binding.

Refs: #412
```

```
wredit: split the rope table into a fixed-size array

The rope table was a Vec that reallocated on every insert, which showed up
as 8% of frame time in the scroll benchmark. A fixed array keeps the
allocation off the hot path.
```

### Context loading

The codebase is ~293k lines across 53 crates. **Never try to read the whole repository
and never paste a whole crate into context.** Instead:

1. Identify the crate that owns the change and read only that crate.
2. Load the crate's public API surface, dependencies, and invariants before editing
   (see the `crate-context` skill).
3. Use the `explore` subagent for questions of the form "where is X used?", "what
   depends on Y?" instead of grepping by hand across the workspace.

### Verification

- Run the **scoped** checks for the crates you touched before every commit.
- Run the **full** workspace checks (`build`, `test`, `fmt --check`, `clippy`) once
  before declaring a task complete.
- Never report a task as complete on the strength of "it looks right". Only CI
  decides whether the work is complete.

## OpenCode

This repository ships OpenCode configuration under `.opencode/`. It is the reference
tooling for AI agents working here; other harnesses should follow the same rules.

| Path | Purpose |
| --- | --- |
| `.opencode/opencode.jsonc` | Project config: agents, permissions, instructions, worktree layout |
| `.opencode/skills/<id>/SKILL.md` | Task skills, loaded on demand with the `skill` tool |
| `.opencode/agents/*.md` | Subagents (reviewer, verifier) |
| `.opencode/commands/*.md` | Slash commands (`/commit`, `/check`, `/review`, `/crate`) |

**Skills.** A skill is a directory under `.opencode/skills/` containing a `SKILL.md`.
The skill ID is the directory name (lowercase, kebab-case). Skills are advertised to
the agent by their `description`; load one explicitly with the `skill` tool by ID, or
by mentioning `@<skill-id>` in a prompt.

| Skill ID | Load it when |
| --- | --- |
| `commit-discipline` | Committing work, or when the diff is growing past ~800 lines |
| `crate-context` | Before editing any crate, to load its API surface, deps, and invariants |
| `verify` | Before declaring a task complete, or when a check fails |
| `unsafe-audit` | Adding, changing, or reviewing `unsafe` code |
| `new-crate` | Scaffolding a new crate in the workspace |
| `public-api` | Changing a public API, or reviewing whether a change breaks callers |

**Commands.** Run from the TUI: `/commit` (checkpoint the current work within the
commit budget), `/check` (scoped verification for the affected crates), `/review`
(read-only review of the current diff), `/crate <name>` (brief for a crate).

**Worktrees.** When several agents work in parallel, each agent must use its own
worktree so changes cannot collide:

```text
worktree_create(name: "feature-tls-reneg")   # creates .lane/trees/<name>
session_move(directory: ...)                  # switch this session into it
worktree_remove(directory: ...)              # clean up after the merge
worktree_list()                              # see what exists
```

Only one agent may edit the workspace `Cargo.toml` member list at a time; concurrent
crate additions conflict in `Cargo.lock`. Use the `new-crate` skill's skeleton-first
workflow instead of several agents creating crates simultaneously.

## Coding conventions

### Style

- Follow `rustfmt` settings in `.rustfmt.toml` (edition 2024).
- Use `` on hot-path small functions, matching existing code.
- `unsafe` is allowed at the crate level; document invariants when adding unsafe blocks.

### File headers

New Rust files must include the Apache 2.0 copyright header used elsewhere:

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
```

### Documentation

- Public items should have doc comments
- Match the existing style: type-level docs with field descriptions for `#[repr(C)]` structs.

## Testing

- Keep unit tests close to the code when they only exercise one module.
- Run `cargo test --workspace` before finishing work.
- Only add tests that cover meaningful behavior; avoid trivial assertions.

## Guardrails

### Core Development Principles

- **Minimize scope** — fix the requested problem; do not refactor unrelated code.
- **Match existing patterns** — read surrounding code before adding new abstractions.
- **Commit incrementally** — follow **Commit discipline** above. Commits are expected as
  part of the work, not only at the end of it. Never push or open a PR unless asked.
- **Do not add Markdown docs** (README, guides, etc.) unless requested. `AGENTS.md` and
  the agent tooling under `.opencode/` are exempt.
- **Performance First** — every change should consider performance implications.
- **Safety First** — leverage Rust's type system for zero-cost safety.

### Error Handling Requirements

- **FORBIDDEN use of `.unwrap()` or `.expect()` in production code** — instead, manage errors with Result or Option and
  handle them appropriately, that functions are only acceptable on testing, But not in production-code or doc examples
- Use `?` operator for error propagation in functions returning `CompressorResult`
- Use `.unwrap_or()`, `.unwrap_or_default()`, or `.unwrap_or_else()` for fallback values
- `.unwrap()` and `.expect()` are ONLY permitted in unit tests with explicit justification
- **FORBIDDEN use of `panic!`, `abort()`, and other panicking methods in production code**
- **FORBIDDEN use of `assert!`, `assert_eq!`, `assert_ne!` in production code**
- Use `debug_assert!` only in debug builds for invariant checking
- Always handle errors comprehensively with appropriate error types

### Logging and Output Requirements

- **FORBIDDEN use of `println!` or `eprintln!` for production logging**
- Configure appropriate log levels for different environments
- Structure log messages with context and relevant data
- Avoid excessive logging in hot paths

### Memory and Performance

- Avoid unnecessary cloning and copying
- Prefer stack allocation over heap allocation when possible
- Use `&[T]` and `&str` for read-only data views
- Consider memory layout and cache efficiency
- Profile performance changes before merging
- Use `#[inline]` on hot-path small functions
- Avoid storing more than one `bool` inside structs, Use enums with custom BitOr overloads instead

### CPU Utilization and Optimization

Codevar runs on one core on many of its targets, so keeping that core busy —
and knowing *why* it is not — matters as much as any GPU optimization. Use
`codevar-inssched` for both levels of CPU scheduling rather than hand-rolling a
queue or a priority loop.

**Scheduling multiple tasks on one core (program level).** `codevar-inssched`
exposes a program-level `Kernel` that multiplexes many tasks on a single core
the way an operating system does on a uniprocessor: it hands the core out in
time slices, lets a `Timer` interrupt a slice, saves and restores a task's CPU
context on a switch, and chooses the next task with a `SchedulePolicy`.

- Reach for `codevar_inssched::policy::RoundRobin` for fair, equal-share work,
  `Priority` when critical system work must preempt background work, and
  `ShortestJobFirst` when average waiting time is the metric that matters.
- Model each unit of work as a `Task` with a realistic `burst`; derive that
  burst from real code with `Task::burst_from_block` so the numbers track the
  instruction schedule instead of guesses.
- Call `Kernel::block` when a task waits on I/O instead of spinning. A busy
  wait burns the slice the blocked task would have released.
- Pick a `Config::quantum` that is a few times the context-switch cost: too
  small and switch overhead dominates, too large and the core stops being fair.
- Never busy-wait between tasks; if nothing is runnable, let the kernel idle or
  block rather than spin.

**Ordering instructions inside a task (instruction level).** Order each task's
straight-line block before running it, so latency is hidden instead of stalled:

- Use `codevar_inssched::sched::Latency` on stalled in-order cores (Cortex-M,
  ARM7, RV32 embedded) to hoist work past load and multiply latency.
- Use `RegisterPressure` on wide out-of-order cores (x86-64, aarch64), where
  live-range pressure — not latency — is the binding constraint.
- Use `SourceOrder` when scheduling is not known to help; it repairs stalls
  without reordering.

**Measure, do not guess.** `Kernel::run` returns `Stats`; read
`cpu_utilization`, `idle_cycles`, `overhead_cycles`, `context_switches`, and
`average_waiting` to find the bottleneck before tuning a quantum or a policy.
High `idle_cycles` with runnable work means a scheduling bug, high
`overhead_cycles` means the quantum is too small, and high `average_waiting`
means the policy is starving a task. Profile CPU changes the same way GPU
changes are profiled.

### Optimizing binary sizes

- Use #![cfg_attr (not (test), no_std)] when adding new modules for reducing binary sizes

### Unsafe Code Guidelines

- **Use unsafe only when absolutely necessary** — for SIMD optimizations, memory management, or FFI
- **Document all invariants clearly** for each unsafe block
- Provide comprehensive safety documentation
- Isolate unsafe code in well-defined modules
- Review unsafe code thoroughly before merging
- Prefer safe alternatives when available

### Testing Requirements

- Write meaningful tests that cover actual behavior
- Avoid trivial assertions that don't add value
- Use property-based testing for data processing algorithms
- Test error paths and edge cases
- Maintain high test coverage for critical paths
- Run `cargo test --workspace` before finishing work

### Documentation Requirements

- Public APIs must have doc comments
- Document all unsafe blocks with safety invariants
- Provide examples for complex algorithms
- Document performance characteristics for public APIs
- Include panics/safety sections where relevant
- Match existing documentation style

### Module Organization

- Use existing re-exports from public APIs
- Avoid inventing new module prefixes
- Follow the existing module structure
- Keep related functionality together
- Use proper visibility modifiers

### GPU Compute Shader Development

When adding CUDA or Vulkan compute shader support:

#### CUDA Development

- Design algorithms for massive parallelism (thousands of threads)
- Minimize thread divergence within warps
- Use shared memory for frequently accessed data
- Coalesce global memory access patterns
- Avoid atomic operations when possible
- Design for optimal memory bandwidth utilization
- **FORBIDDEN deeply nested loops** in kernel code
- **Minimize branch divergence** within warps
- **Use appropriate block sizes** (typically 128-512 threads)
- Profile and optimize based on actual hardware metrics

#### Cross-Platform Compute

- Abstract compute operations behind Rust interfaces
- Support fallback to CPU implementations when GPU unavailable
- Design algorithms that work efficiently on both CPU and GPU
- Provide consistent behavior across different backends
- Implement comprehensive error handling for GPU initialization

### Security Considerations

- Validate all external inputs
- Use bounded integer operations to prevent overflow
- Be careful with pointer arithmetic in unsafe code
- Consider side-channel attacks in cryptographic code
- Validate array bounds before access
- Use well-vetted cryptographic libraries
- Avoid implementing custom cryptography
- Constant-time operations for secret data

### Code Review Checklist

Before considering code complete, verify:

- [ ] No `.unwrap()` or `.expect()` in production code
- [ ] No `panic!`, `abort()`, or panicking methods in production code
- [ ] No `println!` or `eprintln!` — use `codevar_logger` workspace crate instead
- [ ] All unsafe code has proper documentation
- [ ] Public APIs have comprehensive documentation
- [ ] Error handling is comprehensive and proper
- [ ] Performance characteristics are considered
- [ ] Code follows existing patterns and conventions
- [ ] Tests cover meaningful behavior
- [ ] No unnecessary dependencies added
- [ ] SIMD and parallel code is efficient
- [ ] Memory access patterns are optimized
- [ ] Code compiles with `cargo build --workspace`
- [ ] Tests pass with `cargo test --workspace`
- [ ] Code formatting passes with `cargo fmt --all`
- [ ] Clippy passes with `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Commits are within budget: ≤ 800 added lines, ≤ 3 files, 1 crate per commit
- [ ] Commit messages follow `<scope>: <imperative summary>` with a body explaining why
- [ ] Work was committed incrementally, not accumulated into one large commit

### Unsafe Code

```rust
// Correct: unsafe with proper documentation
/// # Safety
///
/// This function is safe to call when:
/// - `ptr` is properly aligned for T
/// - `ptr` points to initialized memory
/// - The memory at `ptr` is valid for reads of `size * std::mem::size_of::<T>()` bytes
#[inline]
unsafe fn read_array<T>(ptr: *const T, size: usize) -> Vec<T> {
    let mut result = Vec::with_capacity(size);
    std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), size);
    result.set_len(size);
    result
}
```

