// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

//! WEZC: a context-mixing codec for the elements a `Writable` holds in RAM.
//!
//! The Wredit engine stores a document as an array of packed 32-bit
//! elements, `(element type << 28) | codepoint`, so a character costs four
//! bytes of RAM by construction. This crate compresses that array - not
//! UTF-8, not a rope node - and is deliberately independent of the engine:
//! the engine is the caller, so a dependency in the other direction would
//! cycle, and the element layout is a fixed ABI both sides already agree
//! on.
//!
//! # Stages
//!
//! | stage | module | what it does |
//! |-------|--------|--------------|
//! | `plane` | plane split | splits the element array into byte planes: the low byte (the text itself), the codepoint bits 8..23 (non-zero only for wide characters, so it is stored as a compacted escape list) and the element-type byte. The planes are what the model sees; measured on engine data, modelling the interleaved elements instead costs a context-mixing model 60-119% more output bytes. |
//! | `model` | context mixing | orders 1-6 plus word and sparse contexts, each a bit-history state whose probability comes from a per-context state map, a match model, and a two-layer logistic mixer with an SSE stage. |
//! | `coder` | arithmetic coding | the carryless 32-bit binary arithmetic coder that turns the mixed probability into bits. |
//!
//! `frame` carries the container: magic, the model parameters the decoder
//! needs, and the lengths. `codec` drives the three stages: one bit loop with
//! an encoder on one side and a decoder on the other, so the two cannot drift
//! apart, plus the optional dictionary block that primes the model without
//! entering the stream.
//!
//! The encoder and the decoder drive one shared coding loop, so a model
//! change cannot make the two drift apart: whatever the encoder computes for
//! a bit, the decoder computes identically before it knows the bit.
//!
//! # Memory
//!
//! The model is sized by a single `mem_bits` parameter: the context tables
//! and the match table each hold `1 << mem_bits` entries. The default is a
//! desktop budget; a document is only worth compressing when it is larger
//! than the model it needs, and the model reports its own budget so a caller
//! can decide.
//!
//! # Throughput
//!
//! Context mixing is serial by construction - the coder feeds back into the
//! model every bit - so the vectorizable work is in stage 1 (plane split and
//! compaction) and in the mixer and state-map arithmetic, not in the coder.
//! Kernels are selected at run time through [`codevar_base::cpuid`] and every
//! path produces bit-identical output, which is what lets a stream encoded
//! on one machine decode on another.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod codec;
pub mod coder;
pub mod element;
pub mod error;
pub mod frame;
pub mod model;
pub mod plane;
pub mod tables;

mod simd;
mod tables_data;
