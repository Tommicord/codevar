# Contributing to Codevar

First off, thank you for considering contributing to Codevar! 🎉

Codevar is a high-performance code editor written in Rust, targeting WebAssembly
(the web) with plans for native desktop and mobile applications. This document
provides guidelines and instructions for contributing.

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Conventions](#coding-conventions)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Security Vulnerabilities](#security-vulnerabilities)

## Getting Started

- Fork the repository on GitHub.
- Clone your fork locally:
  ```bash
  git clone https://github.com/<your-username>/codevar.git
  cd codevar
  ```
- Add the upstream repository as a remote:
  ```bash
  git remote add upstream https://github.com/Tommicord/codevar.git
  ```
- Create a branch for your changes:
  ```bash
  git checkout -b feature/my-feature
  ```

## Development Setup

### Prerequisites

- **Rust**: Version pinned in `rust-toolchain.toml` (currently 1.93.0,
  edition 2024). Install via [rustup](https://rustup.rs/).
- **wasm-pack**: Required for building the WebAssembly target:
  ```bash
  cargo install wasm-pack
  ```

### Building and Testing

Run these from the repository root:

```bash
# Build the entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Format the code (required to pass CI)
cargo fmt --all

# Lint (required to pass CI; warnings are treated as errors)
cargo clippy --workspace --all-targets -- -D warnings
```

CI runs the build, tests, `cargo fmt --check`, and clippy on every push and pull
request to `main`. Make sure all of these pass locally before submitting your
PR.

## How to Contribute

### Improving Documentation

Documentation improvements (README, doc comments, examples) are always
welcome. Look for files with `TODO` comments or issues labeled `documentation`.

### Reporting Bugs

Before creating a bug report, please [search existing issues](https://github.com/Tommicord/codevar/issues)
to see if the problem has already been reported. If it has, add a comment to
the existing issue instead of opening a new one.

When filing a bug report, use the **Bug Report** issue template and include:

- A clear and descriptive title
- Steps to reproduce the problem
- The expected and actual behavior
- Your environment (OS, Rust version, browser if applicable)
- Relevant logs, error messages, or screenshots

### Suggesting Features

Feature suggestions are welcome! Use the **Feature Request** issue template and
describe:

- The problem your suggestion solves
- Your proposed solution
- Any alternatives you have considered
- Additional context (mockups, references, etc.)

Note that Codevar is in early development, so some features may be out of scope
for now. Check the [README](README.md) for the current project vision.

### Reporting Security Vulnerabilities

**Do not report security vulnerabilities through public GitHub issues.**
Please see our [Security Policy](SECURITY.md) for private reporting instructions.

## Pull Request Process

1. **Update documentation** if your change affects the public API or behavior.
2. **Add tests** for new features and bug fixes, when applicable.
3. **Follow the coding conventions** described below.
4. **Keep your PR focused** — submit one fix or feature per PR. Unrelated
   changes should go in separate PRs.
5. **Write a clear PR description** explaining the problem, the solution, and
   any trade-offs. Reference any related issues (e.g. `Fixes #123`).
6. **Be responsive to feedback** — maintainers may request changes before
   merging.
7. Ensure CI is green. The branch must pass:
   - `cargo build --workspace`
   - `cargo test --workspace`
   - `cargo fmt --all --check`
   - `cargo clippy --workspace --all-targets -- -D warnings`

Maintainers will squash-merge PRs once they are approved.

## Coding Conventions

The full guidelines live in [AGENTS.md](AGENTS.md). The highlights:

### Style

- Follow the `rustfmt` settings in `.rustfmt.toml` (edition 2024).
- Use `#[inline]` on hot-path small functions.
- Match the existing code style; read surrounding code before adding new
  abstractions.
- Preserve the Apache 2.0 copyright header in existing files. New Rust files
  must include the license header (see [AGENTS.md](AGENTS.md#file-headers)).

### Error Handling

- **Never use `.unwrap()` or `.expect()` in production code** — use `Result`,
  `Option`, and the `?` operator instead. These are only acceptable in unit
  tests.
- **Never use `panic!`, `abort()`, `assert!`, `assert_eq!`, or `assert_ne!`**
  in production code. `debug_assert!` is allowed in debug builds for invariant
  checks.
- Use `.unwrap_or()`, `.unwrap_or_default()`, or `.unwrap_or_else()` for
  fallback values.

### Logging

- **Never use `println!` or `eprintln!`** for production logging — use the
  `codevar_logger` workspace crate instead.
- Structure log messages with context and relevant data.
- Avoid excessive logging on hot paths.

### Memory and Performance

- Avoid unnecessary cloning and copying.
- Prefer stack allocation over heap allocation when possible.
- Use `&[T]` and `&str` for read-only data views.
- Performance is a first-class concern for this project — profile before and
  after significant changes.

### Unsafe Code

- Use `unsafe` only when absolutely necessary (SIMD, memory management, FFI).
- Document all invariants for each unsafe block with a `# Safety` section.
- Prefer safe alternatives when available.

### New Modules

- Use `#![cfg_attr(not(test), no_std)]` when adding new modules to help
  reduce binary size.

## Testing

- Write meaningful tests that cover actual behavior; avoid trivial assertions.
- Test error paths and edge cases.
- Use property-based testing for data processing algorithms.
- Keep unit tests close to the code when they only exercise one module.
- Run `cargo test --workspace` before submitting your PR.

## Documentation

- Public APIs must have doc comments.
- Document performance characteristics for public APIs.
- Document all unsafe blocks with their safety invariants.
- Provide examples for complex algorithms.
- Include `Panics` and `Safety` sections where relevant.

## Review Checklist

Before requesting a review, verify:

- [ ] No `.unwrap()` or `.expect()` in production code
- [ ] No `panic!`, `abort()`, or panicking macros in production code
- [ ] No `println!` or `eprintln!` — uses `codevar_logger` instead
- [ ] All unsafe code is properly documented
- [ ] Public APIs have comprehensive documentation
- [ ] Tests cover meaningful behavior
- [ ] Code builds: `cargo build --workspace`
- [ ] Tests pass: `cargo test --workspace`
- [ ] Formatting passes: `cargo fmt --all`
- [ ] Clippy passes: `cargo clippy --workspace --all-targets -- -D warnings`

## Questions?

Feel free to open a GitHub issue with the `question` label, or reach out to the
maintainers. We are happy to help!

Thank you for contributing! 🚀
