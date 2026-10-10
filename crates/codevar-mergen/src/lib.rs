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

//! # Mergen
//!
//! Conflict-free merge algorithm for collaborative editing: two
//! streams of changes are interleaved deterministically by their
//! SHAKE256 change keys, with constant-time ranking so evaluation
//! does not leak the content being merged.
//!
//! The crate currently provides the SIMD rank-merge kernels the
//! algorithm is built on ([`merge_rank`], [`count_less`],
//! [`count_le`], [`first_mismatch_u8`]); the merge pipeline
//! itself is under construction.
//!
//! ## Determinism
//!
//! Every backend (scalar, SSE, AVX, NEON, WASM SIMD128) produces
//! bit-identical results; backend selection only affects performance.
//!
//! ## Examples
//!
//! ```
//! let a = [1u64, 3, 5];
//! let b = [2u64, 3, 4];
//! let mut ranks_a = [0u32; 3];
//! let mut ranks_b = [0u32; 3];
//! codevar_mergen::merge_rank(&a, &b, &mut ranks_a, &mut ranks_b);
//! assert_eq!(ranks_a, [0, 2, 5]);
//! assert_eq!(ranks_b, [1, 3, 4]);
//! ```

//! TODO: Implement mergen algorithm
