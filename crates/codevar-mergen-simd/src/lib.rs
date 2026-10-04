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

//! # Mergen SIMD primitives
//!
//! Deterministic, constant-time SIMD kernels and primitives used by the Mergen
//! conflict-free merge algorithm.
//!
//! The crate provides three groups of building blocks:
//!
//! - **Tier detection** ([`SimdTier`]): runtime detection of the best available
//!   SIMD tier (scalar, SSSE3, SSE4.1, AVX2, AVX512VL, NEON, WASM SIMD128).
//! - **Constant-time helpers**: branchless select/compare primitives used to
//!   keep merge evaluation independent of secret key material.
//! - **Merge kernels**: sorting-key merge, partition point and first-mismatch
//!   kernels where every backend (scalar and SIMD) is bit-identical.
//!
//! ## Determinism
//!
//! All backends produce bit-identical results for the same inputs; the detected
//! tier only affects performance, never output.
//!
//! ## GPU mapping
//!
//! Kernels operate on fixed-size blocks (8 or 16 keys) with no data-dependent
//! memory access so the same logic maps to GPU thread blocks later; only the
//! CPU + SIMD backends are implemented for now.

#![cfg_attr(not(test), no_std)]
