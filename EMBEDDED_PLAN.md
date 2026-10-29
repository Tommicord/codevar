# Codevar Embedded Runtime Plan — Remove Dynamic Container Growth Risks

## Goal

Replace `Vec::new()` (and other dynamically-growing containers that can crash on a
giant loop via `.push`) with capacity-aware allocation (`Vec::with_capacity`,
`try_reserve`) or stack-allocated slices + streaming APIs, so the embedded runtime
never loads an entire input into memory and never grows without a bound.

## Crates in scope (embedded-suitable)

These crates are candidates for the embedded runtime and must have their
production `Vec::new` usages audited/replaced:

- `codevar-wredit` (core editor engine — `no_std` + `alloc`)
- `codevar-io` (buffered I/O / cursor / split / lines)
- `codevar-base` (core primitives)
- `codevar-tls` (TLS — embedded-suitable)
- `codevar-logger`
- `codevar-packlow`, `codevar-shake`, `codevar-mergen`
- `codevar-tlsf-alloc`
- `codevar-wsocket`
- `codevar-cli-parse`, `codevar-pathbuf`
- `codevar-inssched` (instruction scheduler — already in progress)
- `codevar-truetype-*` (font rasterization pipeline)
- `codevar-ocl-*` / `codevar-oclc` (OpenCL codegen — already <8 KB RAM, still audit)

## Crates OUT of scope (desktop-only — do NOT touch)

- `codevar-kuitui-*` (UI toolkit)
- `codevar-wayland`, `codevar-xdg-portal`, `codevar-dbus`, `codevar-launcher`
- `codevar-gpu-vk`, `codevar-gpu-wg` (desktop graphics)

## Replacement strategy

| Situation | Replacement |
|-----------|-------------|
| `Vec::new()` then bounded `.push` with known max | `Vec::with_capacity(max)` |
| `Vec::new()` then unbounded `.push` in a loop | Stream in chunks; keep a small fixed cache (8/16/24 bytes); expose a "request a memory region" API instead of returning an owned `Vec` |
| `Vec::new()` used as a scratch buffer reused across calls | `Vec::with_capacity(initial)` + `try_reserve` + clear-and-reuse |
| Fixed tiny working set | Stack array `[u8; N]` (N = 8/16/24/32) |
| Test-only `Vec::new` | Leave as-is (tests may use `.unwrap()`) |

## Memory region / streaming API

Callers that previously did `let data = read_all()?` (loading everything) should
instead request a region:

```rust
/// Read at most `cap` bytes into `buf`, returning how many were consumed.
fn read_region(&mut self, buf: &mut [u8]) -> CompressorResult<usize>;
```

The engine keeps a small cache (8/16/24 bytes) of the currently loaded values
and refills it on demand, never holding the whole input.

## Code quality rules (CODE_QUALITY.md)

- No `.unwrap()` / `.expect()` / `panic!` / `assert!` / `unreachable!` in production
- No `println!` / `eprintln!` — use `codevar_logger` macros
- Errors via `CompressorResult` / `?`; fallbacks via `unwrap_or` / `unwrap_or_else`
- `unsafe` blocks need documented safety invariants
- Public items need doc comments; new files need Apache 2.0 header
- `#![cfg_attr(not(test), no_std)]` crates use `alloc::vec::Vec`

## Verification

- `cargo check -p <crate>`
- `cargo clippy -p <crate> --all-targets -- -D warnings`
- `cargo test -p <crate>`
- `cargo fmt` for touched files
- Benchmarks must not regress significantly

## Commit discipline

- One crate (or one coherent concern) per commit
- ≤ 800 added lines, ≤ 3 files per commit
- Message format: `<scope>: <imperative summary>`
