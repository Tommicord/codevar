# AGENTS.md

Instructions for AI coding agents working in the Codevar repository.

## Project overview

Codevar is a high-performance code editor targeting for WASM (web) with plans for native desktop and mobile applications. The core library is written in Rust and the project is in early development. 

### Key Features

- **WebAssembly Target**: Optimized for browser-based deployment with wasm-pack
- **Vulkan Graphics**: High-performance rendering using Vulkan graphics API
- **Collaborative Editing**: Real-time collaborative editing with CRDT-like conflict resolution
- **Cross-Platform**: Support for web, desktop, and mobile platforms
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

# Benchmarks (UTF-8/16/32 encoding)
cargo bench -p codevar-core
```

CI (`.github/workflows/rust.yml`) runs build, test, `cargo fmt --check`, and clippy on every push/PR to `main`.

```

## Coding conventions

### Style

- Follow `rustfmt` settings in `.rustfmt.toml` (90-column width, 4-space indent, edition 2024).
- Clippy is enabled with `clippy::all` and `clippy::pedantic` at the crate level.
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

- Public items should have doc comments (`#![warn(missing_docs)]` is enabled).
- Match the existing style: type-level docs with field descriptions for `#[repr(C)]` structs.

## Testing

- Add integration tests in `crates/codevar-core/tests/` for behavior that spans modules.
- Keep unit tests close to the code when they only exercise one module.
- Run `cargo test --workspace` before finishing work.
- Only add tests that cover meaningful behavior; avoid trivial assertions.

## Guardrails

### Core Development Principles

- **Minimize scope** — fix the requested problem; do not refactor unrelated code.
- **Match existing patterns** — read surrounding code before adding new abstractions.
- **Do not commit** unless explicitly asked.
- **Do not add Markdown docs** (README, AGENTS.md, etc.) unless requested.
- **Performance First** — every change should consider performance implications.
- **Safety First** — leverage Rust's type system for zero-cost safety.

### Error Handling Requirements

- **FORBIDDEN use of `.unwrap()` or `.expect()` in production code** — instead, manage errors with Result or Option and handle them appropriately
- Use `?` operator for error propagation in functions returning `Result`
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
- Design algorithms for CPU and GPU parallel execution

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

- Public APIs must have doc comments (`#![warn(missing_docs)]` is enabled)
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
- [ ] No `println!` or `eprintln!` — use `log` crate instead
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

## Examples

### Module Usage

```rust
// Correct: import from the public API
use codevar_wredit::BaseWritable;

// Correct: generic params match existing tests
let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
```

```rust
// Avoid: inventing new module prefixes or bypassing re-exports
use codevar_wredit::writable_base::BaseWritable; // use edit::BaseWritable instead
```

### Error Handling

```rust
// Correct: proper error handling
fn process_data(input: &str) -> Result<ProcessedData, ProcessingError> {
    let parsed = parse_input(input).map_err(ProcessingError::ParseError)?;
    let validated = validate_data(&parsed).map_err(ProcessingError::ValidationError)?;
    Ok(ProcessedData::new(validated))
}

// Correct: using fallback values
let value = some_option.unwrap_or(0);
let value = some_option.unwrap_or_else(|| compute_default());
```

```rust
// Forbidden: unwrap in production code
let value = some_option.unwrap();
let result = some_result.expect("This should never fail");
```

### Logging

```rust
// Correct: using log crate
use log::{error, warn, info, debug, trace};

error!("Failed to process request: {}", error);
warn!("Cache miss for key: {}", key);
info!("User logged in: user_id={}", user_id);
debug!("Processing block: block_id={}, size={}", block_id, size);
```

```rust
// Forbidden: println! in production code
println!("Processing data: {}", data);
eprintln!("Error occurred: {}", error);
```

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

### GPU Compute Shader Example

```rust
// Correct: cross-platform compute abstraction
pub trait ComputeBackend {
    fn process_blocks(&self, input_a: &[u8], input_b: &[u8]) -> Result<Vec<u8>, ComputeError>;
    fn is_available(&self) -> bool;
}

pub struct ComputeManager {
    backend: Box<dyn ComputeBackend>,
}

impl ComputeManager {
    pub fn new() -> Result<Self, ComputeError> {
        let backend = if VulkanBackend::is_available() {
            Box::new(VulkanBackend::new()?) as Box<dyn ComputeBackend>
        } else if CudaBackend::is_available() {
            Box::new(CudaBackend::new()?) as Box<dyn ComputeBackend>
        } else {
            Box::new(CpuBackend::new()) as Box<dyn ComputeBackend>
        };
        
        Ok(Self { backend })
    }
}
```
