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
- **ALWAYS handle errors using `Result`, `Option`, or appropriate error handling methods**
- Use `?` operator for error propagation in functions returning `Result`
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
- Use `Result` and `Option` for all error conditions
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
- **ALWAYS use the `log` crate macros: `error!`, `warn!`, `info!`, `debug!`, `trace!`**
- Configure appropriate log levels for different environments
- Structure log messages with context and relevant data
- Avoid excessive logging in hot paths

#### Examples

```rust
// ❌ FORBIDDEN in production code
println!("Processing data: {}", data);
eprintln!("Error occurred: {}", error);

// ✅ CORRECT logging
use log::{error, warn, info, debug, trace};

error!("Failed to process request: {}", error);
warn!("Cache miss for key: {}", key);
info!("User logged in: user_id={}", user_id);
debug!("Processing block: block_id={}, size={}", block_id, size);
trace!("Detailed state: state={:?}", state);
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

#### Examples

```rust
// ✅ CORRECT concurrent design
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ConcurrentCounter {
    value: Arc<AtomicUsize>,
}

impl ConcurrentCounter {
    fn increment(&self) -> usize {
        self.value.fetch_add(1, Ordering::SeqCst)
    }
}
```

## GPU Compute Shader Development Guidelines

### CUDA Development Standards

#### Architecture Requirements

- Design algorithms for massive parallelism (thousands of threads)
- Minimize thread divergence within warps
- Use shared memory for frequently accessed data
- Coalesce global memory access patterns
- Avoid atomic operations when possible
- Design for optimal memory bandwidth utilization

#### Code Quality Standards

```cuda
// ✅ CORRECT CUDA kernel design
__global__ void merge_blocks(
    const uint8_t* __restrict__ data_a,
    const uint8_t* __restrict__ data_b,
    uint8_t* __restrict__ result,
    const size_t block_size
) {
    const size_t tid = threadIdx.x;
    const size_t bid = blockIdx.x;
    const size_t global_id = bid * blockDim.x + tid;
    
    // Shared memory for cache efficiency
    __shared__ uint8_t shared_a[256];
    __shared__ uint8_t shared_b[256];
    
    // Coalesced memory access
    if (tid < block_size && global_id < block_size) {
        shared_a[tid] = data_a[global_id];
        shared_b[tid] = data_b[global_id];
    }
    
    __syncthreads();
    
    // Simple merge logic - avoid nested loops
    if (tid < block_size && global_id < block_size) {
        result[global_id] = (shared_a[tid] <= shared_b[tid]) ? 
                            shared_a[tid] : shared_b[tid];
    }
}
```

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

#### Examples

```rust
// ✅ CORRECT cross-platform compute design
pub trait ComputeBackend {
    fn process_blocks(&self, input_a: &[u8], input_b: &[u8]) -> Result<Vec<u8>, ComputeError>;
    fn is_available(&self) -> bool;
    fn device_info(&self) -> DeviceInfo;
}

pub struct ComputeManager {
    backend: Box<dyn ComputeBackend>,
}

impl ComputeManager {
    pub fn new() -> Result<Self, ComputeError> {
        // Try GPU backends first, fall back to CPU
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

## Parallel Algorithm Design

### Mergen Algorithm Guidelines

#### Block-Based Processing

- Process data in fixed-size blocks (8×8 or 16×16 for optimal GPU scheduling)
- Design algorithms for SIMD vectorization
- Ensure memory alignment for optimal performance
- Minimize data dependencies between blocks
- Use efficient data structures for parallel access

#### Examples

```rust
// ✅ CORRECT block-based parallel processing
pub const BLOCK_SIZE: usize = 16;

pub fn process_parallel(data: &[u8]) -> Vec<u8> {
    let block_count = (data.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    (0..block_count)
        .into_par_iter()
        .map(|block_idx| {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(data.len());
            process_block(&data[start..end])
        })
        .flatten()
        .collect()
}
```

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

#### Examples

```rust
// ✅ CORRECT documentation
/// Merges two blocks of data using binary conflict resolution.
///
/// This function performs a deterministic merge of two data blocks using
/// hash-based ordering to ensure consistent results across different executions.
///
/// # Arguments
///
/// * `block_a` - First data block to merge
/// * `block_b` - Second data block to merge
///
/// # Returns
///
/// A vector containing the merged result in deterministic order.
///
/// # Performance
///
/// This function uses SIMD instructions when available and has O(n) complexity
/// where n is the combined size of both blocks.
///
/// # Safety
///
/// This function uses unsafe SIMD operations but maintains the following invariants:
/// - All memory accesses are within bounds
/// - SIMD operations are only performed on supported architectures
/// - Memory is properly aligned for SIMD operations
pub fn merge_blocks(block_a: &[u8], block_b: &[u8]) -> Vec<u8> {
    // Implementation
}
```

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

#### Examples

```rust
// ✅ CORRECT testing approach
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_merge_deterministic() {
        let result1 = merge_blocks(b"hello", b"world");
        let result2 = merge_blocks(b"hello", b"world");
        assert_eq!(result1, result2);
    }

    proptest! {
        #[test]
        fn test_merge_properties(a in any::<Vec<u8>>(), b in any::<Vec<u8>>()) {
            let result = merge_blocks(&a, &b);
            assert_eq!(result.len(), a.len() + b.len());
        }
    }
}
```

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