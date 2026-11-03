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

//! PTBP — the Parallel Text Buffer Protocol.
//!
//! PTBP is the device-side representation of a Wredit document: one
//! flat image of little-endian 32-bit words at fixed offsets, sized so
//! that a kernel can address every field it needs with a word index
//! and nothing else. The protocol exists because the kernel dialect
//! this repository compiles to has no struct parameters, no `__local`
//! memory, no atomics and no work-group scan: the only way to move a
//! document through it is to make the whole state addressable and the
//! work purely data-parallel.
//!
//! # Image
//!
//! ```text
//! word 0 ┌──────────────────────────────────────────────────────┐
//!        │ control      batch state: lengths, counts, flags      │
//!        ├──────────────────────────────────────────────────────┤
//!        │ ops          the pending operation queue              │
//!        ├──────────────────────────────────────────────────────┤
//!        │ log          retained records, newest last (undo)     │
//!        ├──────────────────────────────────────────────────────┤
//!        │ boundaries   source→destination discontinuity table   │
//!        ├──────────────────────────────────────────────────────┤
//!        │ cursors      caret and selection, one row per cursor  │
//!        ├──────────────────────────────────────────────────────┤
//!        │ gaps         staging gap descriptors                  │
//!        ├──────────────────────────────────────────────────────┤
//!        │ payload      inserted and captured text runs          │
//!        ├──────────────────────────────────────────────────────┤
//!        │ text A       document plane                           │
//!        ├──────────────────────────────────────────────────────┤
//!        │ text B       document plane                           │
//!        └──────────────────────────────────────────────────────┘
//! ```
//!
//! The planes tile the image with no gaps ([`PLANES`]), so the
//! buffer is allocated once and every dispatch passes one base
//! pointer.
//!
//! # What a kernel may assume
//!
//! * The control plane carries [`MAGIC`] and [`VERSION`], and a
//!   kernel returns without touching a text plane when it does not.
//! * Every element is one word: the packed `char | type` element of
//!   [`crate::edit::base`] is stored as-is, so uploading a document
//!   costs a copy and no translation.
//! * Plane offsets are word indices in the same image the kernel
//!   receives, which is why [`record_word`] is the only addressing a
//!   kernel needs to re-declare.

/// Plane tiling, capacities and record addressing.
pub mod geometry;

pub use geometry::{
    BOUNDARY_CAPACITY, BOUNDARY_WORDS, CONTROL_WORDS, CURSOR_CAPACITY, CURSOR_WORDS, GAP_CAPACITY, GAP_WORDS,
    IMAGE_WORDS, LOG_CAPACITY, MAGIC, OP_CAPACITY, OP_WORDS, PAYLOAD_CAPACITY, PLANES, Plane, TEXT_A_OFFSET,
    TEXT_B_OFFSET, TEXT_CAPACITY, VERSION, WORD_BYTES, record_word,
};
