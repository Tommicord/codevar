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

//! The HOT path: device-backed editing over the PTBP image.
//!
//! This is the module [`crate::error`] promises when it says a failing
//! or absent device is not an error condition: a batch built here is
//! dispatched to the GPU, and the same batch run through the scalar
//! model in this module is what the CPU fallback executes. The two
//! paths are not separate implementations of editing — the scalar
//! model *is* the specification the kernels are tested against, so a
//! kernel that disagrees with it fails a test instead of corrupting a
//! document in the field.
//!
//! # What the path buys
//!
//! The engine edits a gap buffer: every insertion or removal moves the
//! text behind it, so a queue of edits costs one memmove per edit on
//! the CPU. The same queue folded through [`sequentially_batch`] costs
//! one write per element *total*, because the edits reduce to a
//! boundary table and the bulk shift becomes a per-element lookup the
//! device runs in parallel. Everything in this module serves that
//! reduction: positions stay in the coordinate system the queue
//! declares them in, and the table is the only state a kernel reads.
//!
//! # Coordinates
//!
//! Three index spaces are in play and mixing them is the easy way to
//! corrupt a batch:
//!
//! * **source** — offsets into the text plane the pass reads;
//! * **destination** — offsets into the text plane the pass writes,
//!   which is where a boundary row's prefix sum lands an element;
//! * **current** — the document as the *previous* ops of the same
//!   batch already edited it, which is the space the operation queue
//!   states its positions in.
//!
//! [`sequentially_batch`] is what carries a position from current to
//! source, and only the source-space table is ever handed to a kernel.

/// Folding a sequential operation queue into one boundary table.
pub mod sequentially_batch;

/// Staging a batch into one device image and running the pass it
/// describes.
pub mod buffer;

/// Shifting cursors and gap bases along a boundary table.
pub mod shift;

pub use sequentially_batch::{
    Batch, MAX_PIECES, MAX_ROWS, Piece, PieceKind, apply_rows, dst_of, is_drop, is_insert,
};
pub use shift::{caret_at, cum_before, cum_through, drop_anchor, dst_len, gap_base_at, inserts_at};
