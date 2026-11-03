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
//! pointer. [`src_plane`] and [`dst_plane`] read the parity bit in
//! the control plane and return two *different* text planes: an apply
//! pass always writes the next document beside the one it reads, so
//! no kernel ever reads and writes the same word and a batch needs no
//! in-place hazard handling.
//!
//! # Editing as an index map
//!
//! A batch of edits is reduced to a sorted table of discontinuities:
//! each [`BoundaryRow`] names the first source index where a signed
//! jump starts, and the prefix sum of the jumps is the destination
//! offset from that index onward. Shifting a million elements then
//! costs one binary search and one store per element with no
//! cross-element dependency, which is the whole reason the
//! representation is used: the O(n) bulk shift is the part of text
//! editing that actually benefits from a device, and it becomes
//! data-parallel instead of a sequential memmove.
//!
//! The prefix sums are filled on the device by a Hillis-Steele scan
//! over global memory ([`CTRL_SCAN_STRIDE`] selects the pass, and
//! each pass is a separate dispatch because the dialect has no barrier
//! that survives across work items). Ordering work items without
//! synchronization is done by *dispatch chaining*: the queue order
//! between two dispatches is the only synchronization a kernel may
//! rely on, so a batch is planned as a fixed sequence of passes and
//! never as one kernel that races itself.
//!
//! # Undo and redo
//!
//! Applied records are retained in [`Plane::Log`] and the text a
//! batch removed can be captured into the payload plane
//! ([`FLAG_CAPTURE`]) before it goes away, which makes the inverse of
//! a batch expressible in the same representation: replay the
//! retained boundary table with the jump signs reversed
//! ([`FLAG_INVERSE`]) and the payloads swapped. Undo and redo are
//! therefore ordinary apply passes over retained state, not a second
//! engine.
//!
//! # What a kernel may assume
//!
//! * The control plane carries [`MAGIC`] and [`VERSION`], and a
//!   kernel returns without touching a text plane when it does not
//!   (see [`is_conforming`]).
//! * Counts live in the control plane, never in a launch argument, so
//!   a kernel that indexes past them is a bug the bounds guard
//!   rejects.
//! * Every element is one word: the packed `char | type` element of
//!   [`crate::edit::base`] is stored as-is, so uploading a document
//!   costs a copy and no translation.
//! * Plane offsets are word indices in the same image the kernel
//!   receives, which is why [`record_word`] is the only addressing a
//!   kernel needs to re-declare.
//! * The geometry above is only half the ABI: the operation tags that
//!   fill the queue are defined beside it, and both halves are guarded
//!   by the same host/kernel drift test.

/// Plane tiling, capacities and record addressing.
pub mod geometry;

/// Record field offsets, flags and the boundary row.
pub mod fields;

/// Little-endian word and run access, plus image initialisation.
pub mod access;

pub use access::{
    control_at, get_word, image_words, init, is_conforming, put_word, read_run, set_control, write_run,
};
pub use fields::{
    BND_CUM, BND_DELTA, BND_PAYLOAD, BND_POS, BoundaryRow, CTRL_FIELDS, CTRL_RESERVED, CURSOR_FIELDS,
    CURSOR_LIVE, CURSOR_SELECTED, FLAG_BITS, FLAG_CAPTURE, FLAG_FLUSH_GAPS, FLAG_INVERSE, FLAG_LOG,
    FLAG_PARITY, FLAG_REMAP, GAP_FIELDS, GAP_SHARED, NO_PAYLOAD, dst_plane, signed, src_plane,
};
pub use geometry::{
    BOUNDARY_CAPACITY, BOUNDARY_WORDS, CONTROL_WORDS, CURSOR_CAPACITY, CURSOR_WORDS, GAP_CAPACITY, GAP_WORDS,
    IMAGE_WORDS, LOG_CAPACITY, MAGIC, OP_CAPACITY, OP_WORDS, PAYLOAD_CAPACITY, PLANES, Plane, TEXT_A_OFFSET,
    TEXT_B_OFFSET, TEXT_CAPACITY, VERSION, WORD_BYTES, record_word,
};
