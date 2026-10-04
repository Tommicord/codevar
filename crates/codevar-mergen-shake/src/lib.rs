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

//! # Mergen SHAKE256
//!
//! FIPS 202 SHAKE256 extendable-output function used by the Mergen algorithm
//! for deterministic content digests.
//!
//! ## Features
//!
//! - One-shot ([`shake256`]) and streaming ([`Shake256`]) APIs
//! - Compile-time evaluation via [`shake256_const`] (`const fn`)
//! - Multi-buffer batch hashing ([`shake256_x2`], [`shake256_x4`], [`shake256_x8`])
//!   following the XKCP times-N layout
//! - Backend dispatch: scalar, SSSE3/SSE4.1, AVX2, AVX512VL (`vpternlogq`,
//!   `vprolq`/`vprolvq`), NEON, WASM SIMD128
//!
//! ## Determinism
//!
//! Every backend computes bit-identical output for the same input; backend
//! selection only affects performance.
//!
//! ## Constant-time behavior
//!
//! The Keccak-f[1600] permutation has no data-dependent branches, memory
//! indexing, or rotation amounts, so all functions are constant-time with
//! respect to message contents.

#![cfg_attr(not(test), no_std)]
