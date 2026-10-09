# CODE_QUALITY.md

Strict quality standards for high-performance Rust code in the Codevar project.

## Core Principles

- **Performance First**: Every line of code should be written with performance in mind
- **Memory Safety**: Leverage Rust's ownership system for zero-cost safety guarantees
- **Parallelism Ready**: Design algorithms for CPU and GPU parallel execution
- **Error Resilient**: Comprehensive error handling without panics
- **Professional Standards**: Production-grade logging, testing, and documentation

## Rust Code Quality Standards

### Error Handling

#### Strict Requirements

- **NEVER use `.unwrap()` or `.expect()` in production code**
- **ALWAYS handle errors using `CompressorResult`, `Option`, or appropriate error handling methods**
- Use `?` operator for error propagation in functions returning `CompressorResult`
- Use `.unwrap_or()`, `.unwrap_or_default()`, or `.unwrap_or_else()` for fallback values
- Only `.unwrap()` and `.expect()` are permitted in unit tests with explicit justification

#### Examples

```rust
// ❌ FORBIDDEN in production code
let value = some_option.unwrap();
let result = some_result.expect("This should never fail");

// ✅ CORRECT error handling
let value = some_option.unwrap_or(0);
let value = some_option.unwrap_or_else( | | compute_default());
let result = some_result?;

// ✅ CORRECT comprehensive error handling
fn process_data(input: &str) -> Result<ProcessedData, ProcessingError> {
    let parsed = parse_input(input).map_err(ProcessingError::ParseError)?;
    let validated = validate_data(&parsed).map_err(ProcessingError::ValidationError)?;
    Ok(ProcessedData::new(validated))
}
```

### Panic Prevention

#### Strict Requirements

- **NEVER use `panic!`, `abort()`, or other panicking methods in production code**
- **LIMIT the use of `assert!` and another asserting macros in production code
- **NEVER use `unreachable!()` in production code**
- Use `CompressorResult` and `Option` for all error conditions
- Use `debug_assert!` only in debug builds for invariant checking
- Only panicking methods are permitted in unit tests

#### Examples

```rust
// ❌ FORBIDDEN in production code
panic!("This should never happen");
assert!(condition, "Invariant violated");
unreachable!();

// ✅ CORRECT error handling
if ! condition {
return Err(Error::InvariantViolation);
}

// ✅ CORRECT debug assertions (debug builds only)
debug_assert!(condition, "Debug invariant check");
```

### Logging and Output

#### Strict Requirements

- **NEVER use `println!` or `eprintln!` for production logging**
- **Use the `codevar_logger` crate macros for logging: `log_error!`, `log_warn!`, `log_info!`, `log_debug!`, `log_irr!`**
- Configure appropriate log levels for different environments
- Structure log messages with context and relevant data
- Avoid excessive logging in hot paths

#### Examples

```rust
// ❌ FORBIDDEN in production code
println!("Processing data: {}", data);
eprintln!("Error occurred: {}", error);

// ✅ CORRECT logging
use codevar_logger::{log_error, log_warn, log_info, log_debug, log_irr};

log_error!("Failed to process request: {}", error);
log_warn!("Cache miss for key: {}", key);
log_info!("User logged in: user_id={}", user_id);
log_debug!("Processing block: block_id={}, size={}", block_id, size);
log_irr!("CRITICAL error happened: Out of memory");
```

### Memory Management

#### Strict Requirements

- Prefer stack allocation over heap allocation when possible
- Use `&[T]` and `&str` for read-only data views
- Use `Cow<'_, T>` for conditional ownership
- Avoid unnecessary `clone()` operations
- Use `Arc` sparingly and only when shared ownership is required
- Use `Box` for large data structures or trait objects
- Implement `Drop` carefully to avoid performance issues

#### Examples

```rust
// ❌ AVOID unnecessary cloning
fn process(data: Vec<u8>) -> Vec<u8> {
    let cloned = data.clone(); // Unnecessary clone
    // ... process cloned
    cloned
}

// ✅ CORRECT borrowing
fn process(data: &[u8]) -> Vec<u8> {
    // Process without cloning
    data.iter().map(|&b| b * 2).collect()
}

// ✅ CORRECT Arc usage for shared ownership
use std::sync::Arc;

struct SharedConfig {
    data: Arc<Vec<u8>>,
}
```

### Unsafe Code Guidelines

#### Strict Requirements

- **Use unsafe only when absolutely necessary**
- **Document all invariants clearly**
- **Provide safety documentation for each unsafe block**
- Prefer safe alternatives when available
- Isolate unsafe code in well-defined modules
- Review unsafe code thoroughly before merging

#### Examples

```rust
// ✅ CORRECT unsafe usage with documentation
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

### Performance Optimization

#### SIMD and Vectorization

- Use `#[inline]` on hot-path small functions
- Use `#[target_feature]` for architecture-specific optimizations
- Prefer explicit SIMD intrinsics over relying on compiler auto-vectorization
- Ensure proper memory alignment for SIMD operations
- Profile SIMD code to ensure performance benefits

#### Examples

```rust
// ✅ CORRECT SIMD usage
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256i, _mm256_add_epi32, _mm256_loadu_si256};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn process_avx2(data: &[i32]) -> Vec<i32> {
    let simd_len = data.len() & !7;
    let mut result = Vec::with_capacity(data.len());

    for i in (0..simd_len).step_by(8) {
        let vec_a = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
        let vec_b = _mm256_loadu_si256(data.as_ptr().add(i + 8) as *const __m256i);
        let added = _mm256_add_epi32(vec_a, vec_b);
        _mm256_storeu_si256(result.as_mut_ptr().add(i) as *mut __m256i, added);
    }

    result.set_len(data.len());
    result
}
```

#### Memory Layout and Cache Efficiency

- Use structs with field ordering that minimizes padding
- Consider `#[repr(C)]` for FFI compatibility
- Use arrays instead of vectors when size is known
- Design data structures for cache-friendly access patterns
- Use `#[cold]` on error paths that are rarely executed

### Concurrency and Parallelism

#### Strict Requirements

- Prefer message passing over shared memory
- Use `Arc<Mutex<T>>` sparingly and prefer `RwLock` for read-heavy workloads
- Design algorithms for lock-free execution when possible
- Use appropriate atomic operations for shared state
- Consider async/await for I/O-bound operations
- Be aware of priority inversion and deadlock scenarios
- Optimize code for memory usage (the Heap is only 16384 bytes, So prefer streaming data in chunks)

#### Performance Guidelines

- **Avoid deeply nested loops** in kernel code
- **Minimize branch divergence** within warps
- **Use shared memory** for frequently accessed data
- **Ensure memory coalescing** for global memory access
- **Limit kernel complexity** to avoid register pressure
- **Use appropriate block sizes** (typically 128-512 threads)
- **Profile and optimize** based on actual hardware metrics

### Cross-Platform Compute Guidelines

#### API Design

- Abstract compute operations behind Rust interfaces
- Support fallback to CPU implementations when GPU unavailable
- Design algorithms that work efficiently on both CPU and GPU
- Provide consistent behavior across different backends
- Implement comprehensive error handling for GPU initialization

## Parallel Algorithm Design

### Mergen Algorithm Guidelines

#### Block-Based Processing

- Process data in fixed-size blocks (8×8 or 16×16 for optimal GPU scheduling)
- Ensure memory alignment for optimal performance
- Minimize data dependencies between blocks
- Use efficient data structures for parallel access

### Conflict Resolution for Parallel Execution

#### Deterministic Results

- Ensure deterministic ordering of operations
- Use hash-based ordering for consistent results
- Design conflict resolution strategies that work in parallel
- Avoid race conditions in shared data structures
- Implement proper synchronization when needed

## Documentation Standards

### Code Documentation

- **Public APIs must have documentation** (`#![warn(missing_docs)]` is enabled)
- **Document all unsafe blocks** with safety invariants
- **Provide examples** for complex algorithms
- **Document performance characteristics** for public APIs
- **Include panics/safety sections** where relevant

## Testing Standards

### Unit Testing

- Write focused unit tests for individual functions
- Test edge cases and error conditions
- Use property-based testing for data processing algorithms
- Maintain high test coverage for critical paths
- Tests should be fast and deterministic

### Integration Testing

- Test interactions between modules
- Test concurrent execution scenarios
- Test error propagation across module boundaries
- Include performance regression tests
- Test on different platforms when applicable

## Code Review Standards

### Review Checklist

- [ ] No `.unwrap()` or `.expect()` in production code
- [ ] No `panic!`, `abort()`, or panicking methods in production code
- [ ] No `println!` or `eprintln!` - use `log` crate instead
- [ ] All unsafe code has proper documentation
- [ ] Public APIs have comprehensive documentation
- [ ] Error handling is comprehensive and proper
- [ ] Performance characteristics are considered
- [ ] Code follows existing patterns and conventions
- [ ] Tests cover meaningful behavior
- [ ] No unnecessary dependencies added

### Performance Review

- Profile hot paths and optimize bottlenecks
- Consider memory allocation patterns
- Review SIMD and parallel code for efficiency
- Check for unnecessary copies or clones
- Verify that abstractions don't introduce overhead

## Security Considerations

### Memory Safety

- Validate all external inputs
- Use bounded integer operations to prevent overflow
- Be careful with pointer arithmetic in unsafe code
- Consider side-channel attacks in cryptographic code
- Validate array bounds before access

### Cryptographic Operations

- Use well-vetted cryptographic libraries
- Avoid implementing custom cryptography
- Constant-time operations for secret data
- Proper key management and disposal
- Secure random number generation

# Using libraries

- Avoid adding external dependencies that can break the code quality rules
- Try to use smaller libraries instead of bigger ones
- The library code MUST be exhaustively reviewed to ensure safety in production before adding to the project

## Continuous Integration

### Quality Gates

- All tests must pass (`cargo test --workspace`)
- Code must be formatted (`cargo fmt --all`)
- Clippy must pass with no warnings (`cargo clippy --workspace --all-targets -- -D warnings`)
- Benchmarks must not regress significantly
- Documentation builds without warnings

### Performance Benchmarks

- Run benchmarks before and after performance changes
- Track performance metrics over time
- Set up automated performance regression detection
- Profile bottlenecks and optimize systematically

## Conclusion

These quality standards ensure that Codevar maintains high-performance, safe, and maintainable code. Adherence to these
standards is mandatory for all contributions to the codebase. When in doubt, err on the side of safety, performance, and
clarity.