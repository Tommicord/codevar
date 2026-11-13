---
name: OpenCL Kernels
description: Use when writing, reviewing, or maintaining OpenCL kernels for codevar-oclc (crates/oclc/kernels/). Defines the supported syntax, constraints, naming rules, and optimization recommendations the compiler pipeline and build script enforce.

---

# OpenCL Kernels — Codevar dialect

This skill governs how to write `*.rs` OpenCL kernel sources under `crates/oclc/kernels/` that the `codevar-oclc` compiler accepts, compiles to SPIR-V (and optionally native machine code), and embeds via `build.rs`. The reference kernel is `crates/oclc/kernels/truetype/raster.rs`; `crates/oclc/kernels/bench/vector_add.rs` and `crates/oclc/kernels/bench/saxpy.rs` are the minimal benchmarks.

## Where kernels live

- Directory: `crates/oclc/kernels/` (overridable with `CV_KERNEL_DIR`).
- Extension: `.rs` (matched case-insensitively). The `build.rs` walks the tree recursively; every directory becomes a `kernels` submodule and every `.rs` file becomes a set of statics inside it.
- Module layout: `kernels/truetype/raster.rs` → `kernels::truetype::RASTER_SPIRV`, `kernels::truetype::RASTER_SOURCE`, `kernels::truetype::RASTER_ENTRY_POINTS`.
- Static prefix: file stem upper-cased, non-alphanumerics → `_`, leading digit prefixed with `_` (e.g. `block.rs` → `BLOCK`).
- Artifact stem: module path + stem joined with `_` (`wredit/block.rs` → `wredit_block.rs`), so same-stem files in different directories never collide.

## Supported syntax

Sources are Rust-flavored OpenCL dialect, parsed by `codevar-ocl-parse` and analyzed by `codevar-ocl-sar`. The AST supports:

- **Items**: `fn` definitions (with optional generics `<'a, T>`), `struct` declarations (named fields or unit `;`), `type` aliases.
- **Function**: `#[kernel] fn name(params) -> Ret { body }`. Kernels must return `void` and take only plain parameters (pointers, scalars, vectors). No generics/lifetimes on kernel entry points in practice.
- **Statements**: `let pat: T = init;` (type and init optional), expression statements, empty `;`.
- **Patterns**: `ident`, `mut ident`, `_` wildcard.
- **Types**: `bool`, `char`/`i8`, `short`/`i16`, `int`/`i32`, `long`/`i64`, `uchar`/`u8`, `ushort`/`u16`, `uint`/`u32`, `ulong`/`u64`, `half`/`f16`, `float`/`f32`, `double`/`f64`; vectors `int4`, `float3`, … (lanes 2, 3, 4, 8, 16 only); `*const T`, `*mut T`; `[T]`, `[T; N]`; tuples `(A, B)`; structs.
- **Expressions**: literals, `true`/`false`, paths, unary (`-`, `!`, `*`, `&`, `&mut`), binary (`+ - * / % << >> & ^ | == != < <= > > && ||`), assign (`=` and compound `+=`, etc.), call, index `a[i]`, field `a.b` / `a.0`, `a?`, cast `a as T`, ranges `a..b`, blocks, `if/else`, `while`, `loop`, `for pat in iter`, `return`, `break`, `continue`, tuples, arrays.
- **Kernels**: every `#[kernel]` function becomes an entry point; the analyzer collects `DeclKind::Function { kernel: true }`.

## Built-in functions (query `codevar_ocl_sar::lookup_builtin_fn`)

Work-item queries (`int` dim → `size_t` result):
`get_global_id`, `get_global_size`, `get_global_offset`, `get_global_linear_id`, `get_local_id`, `get_local_size`, `get_local_linear_id`, `get_enqueued_local_size`, `get_group_id`, `get_num_groups`.

Fences (`barrier`, `mem_fence`, `read_mem_fence`, `write_mem_fence` take one `u32` fence flag; `sub_group_barrier` takes none).

Integer helpers (arity shown): `abs`1, `sign`1, `popcount`1, `min`2, `max`2, `clamp`3 — result keeps shape/element type of the first argument; unsuffixed literals default to `int`.

Boolean reductions (arity 1): `all`, `any` — `boolN` collapses to `bool`.

Float math (result keeps shape of first arg; ints promote to `float`): `sqrt`, `rsqrt`, `cbrt`, `pow`, `fmod`, `atan2`, `fmin`, `fmax`, `step`, `mix`, `smoothstep`, `fma` (arity 2/3 as listed); `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `floor`, `ceil`, `round`, `trunc`, `fabs`, `degrees`, `radians` (arity 1).

`printf` is **not** available; do not call it.

## Constraints the pipeline enforces

1. **UTF-8 only.** Non-UTF-8 files fail the build (`BuildFailure::NotUtf8`).
2. **Every `#[kernel]` returns `void`.** Kernels returning anything else are an analysis error.
3. **Bounds-guard responsibility.** The kernel does not bounds-check buffers; the host must prove every address it passes to a work item lies inside the allocation. `raster.rs` guards only `px < width && py < height` and leaves `origin + py * pitch + px` to the caller.
4. **No synchronization inside kernels.** Kernels perform no inter-work-item synchronization; a plain 1-D or 2-D dispatch suffices. `barrier`/`mem_fence` exist in the builtin table but the raster/vector_add/saxpy kernels do not use them — work items are independent (embarrassingly parallel).
5. **No hidden globals.** Work items communicate only through kernel arguments (pointers and scalars). There is no module-level mutable state visible to a kernel.
6. **Address space / pointer discipline.** Arguments use `*const T` / `*mut T`. A `*mut` widens to `*const` (coercion), never the reverse. Read-only inputs should be `*const T`; outputs `*mut T`.
7. **No `.unwrap()` / `.expect()` / `panic!` / `println!` in kernel source logic** — these are host-side Rust restrictions; kernel code is a dialect, but avoid host-only constructs. Tests may use `unwrap` only on inputs they construct.
8. **One file, one or more entry points.** A file declaring zero `#[kernel]` functions is embedded anyway with a warning; prefer at least one.
9. **Naming collisions.** Static prefixes are non-injective (punctuation folds to `_`), so `block.rs` in two directories collide at `WEDIT::BLOCK`; rename one file. Artifact stems likewise collide; `derive_artifact_stem` folds directory separators to `_`.
10. **No `.cl` extension.** The build script only matches `.rs`. Do not name kernel sources `.cl`.

## Optimization recommendations

Memory:
- Prefer contiguous, tightly packed buffers (`origin = 0, pitch = width`). `raster.rs` accepts signed `pitch` for bottom-up vs top-down, but a tight buffer is the fastest path.
- One output element per work item, no shared writes → no synchronization, no host reduction (`vector_add`, `saxpy`).
- Coalesce global loads/stores: work item `gid` touches only `a[gid]`/`b[gid]`/`out[gid]`.
- Avoid per-work-item indirect loads through non-contiguous pointers; index linearly.

Arithmetic:
- Use vector types (`float4`, `int4`) where the data layout allows lane-parallel work; builtins keep shape (`sqrt(float4)` type-checks by shape).
- Fuse multiply-add with `fma(a, b, c)` instead of `a * b + c` when the target supports it.
- Prefer `min`/`max`/`clamp` builtins over hand-written branches for element-wise selection.
- Use `mix(a, b, t)` for linear interpolation and `step(edge, x)` / `smoothstep` for transitions instead of `if` chains.

Control flow:
- Hoist loop-invariant computations out of `while`/`for` bodies (`raster.rs` recomputes `sample_x`/`sample_y` inside the edge loop — fine for correctness, but move invariant math out if it does not depend on `e`).
- Minimize divergence within a work-group: work items taking different branches serialize. Prefer predication (`mix`, `step`) over `if` when the condition varies per lane.
- Keep loop bounds small and provable; the edge loop in `raster.rs` walks all edges per sample — consider spatial partitioning (edge buckets) for large glyphs.

Launch:
- Launch a global size covering at least the output domain; extra work items are discarded by the bounds guard.
- 1-D or 2-D dispatch is enough (no sync needed). Match dimensionality to the data: `raster.rs` uses `(px, py)` 2-D; `vector_add`/`saxpy` use 1-D.
- Native machine code is only built for the host architecture at the profile's opt level (release → O3, else O0); SPIR-V is the portable path. A cross build embeds SPIR-V only.

## File header

Every new kernel file must carry the Apache 2.0 copyright header:

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

## Reference example

`crates/oclc/kernels/truetype/raster.rs` shows the full contract: a `#[kernel]` whose work item owns one output pixel, samples a 2x2 sub-pixel grid, runs PNPOLY crossing tests over `edges`, merges four coverage hits into one `uchar`, and writes `coverage[origin + py * pitch + px]`. It documents its launch contract, bounds guard, fill rule, and signed-pitch convention in the module doc comment — every kernel should document those same facts.

## Verification

Before touching a kernel, run the scoped checks:

```bash
cargo check  -p codevar-oclc
cargo clippy -p codevar-oclc --all-targets -- -D warnings
cargo test   -p codevar-oclc
```

The `native_kernels.rs` integration test recompiles every embedded kernel to native machine code and asserts position-independence (no absolute addresses) and a valid entry offset — a new kernel must pass it. `tests.rs` in the crate exercises the `--emit spirv` path end to end.

## Never

- Never add a `#[kernel]` that returns a non-`void` type.
- Never rely on kernel-side synchronization or shared mutable state.
- Never name a kernel source `.cl` (only `.rs` is picked up).
- Never use `printf` or host-only Rust constructs in kernel bodies.
- Never skip the bounds-guards documentation: record the launch domain, the guard, and who proves buffer membership.
- Never commit a kernel whose scoped `codevar-oclc` checks fail.
