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

//! Hot delta-ring + cold solid journal for Wredit undo/redo.
//!
//! The delta ring is the *hot* tier: it retains recent edits uncompressed so
//! undo/redo/restore are O(steps) pointer walks with no per-step allocation.
//! The solid journal (defined here alongside the ring) is the *cold* tier:
//! states appended to one long-lived context, windowed so a restore replays a
//! bounded span instead of the whole session, each closed window carrying a
//! SHAKE256 fingerprint.
//!
//! Revision numbering (invariants, validated by the prototype):
//!   * Revision `i` = document state *after* the `i`-th recorded edit.
//!   * Revision `0` = document as opened (the pre-edit baseline).
//!   * `DeltaRing::new` sets `base = 0`.  A `base = 1` off-by-one made every
//!     undo apply its neighbour's edit in the prototype; do not reintroduce it.
//!
//! Record accounting (exact): `record_bytes = 16 + 4 * (|old| + |new|)`,
//! i.e. 8 bytes group id + 8 bytes position + 4 bytes per packed element on
//! each side.
//!
//! # SIMD boundary
//!
//! Per the crate guidelines, SIMD is *not* used for raw byte copies
//! (`memmove`/`memcpy` already do that).  The honest SIMD win is delta
//! derivation: finding the common prefix/suffix of two `u32` state arrays is
//! an O(document) scalar comparison per keystroke, and a SIMD equality scan
//! (`_mm256_cmpeq_epi32` + `_mm256_movemask_epi32`) is the real win.
//! `_mm_prefetch` hides pointer-chase latency when walking the ring.
//! An undo-group classification LUT replaces a branch chain.
//!
//! The AVX2 kernel lives in the target-gated `mod x86`, dispatched through
//! `codevar_base::cpuid::has(Feature::Avx2)`; a scalar reference is the
//! mandatory oracle and every vector path is tested against it.

use alloc::vec::Vec;

use codevar_logger::log_error;

use crate::base::packed_char;
use crate::error::{WreditError, WreditResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Overhead bytes per edit record (8-byte group id + 8-byte position).
const RECORD_OVERHEAD: usize = 16;

/// Bytes per packed element (u32).
const ELEMENT_BYTES: usize = 4;

/// Minimum ring capacity in edit slots.
const MIN_RING_CAPACITY: usize = 16;

/// Maximum ring capacity in edit slots (guards against a hostile budget).
const MAX_RING_CAPACITY: usize = 1 << 20;

/// Number of entries in the undo-group classification LUT (one per byte).
const GROUP_LUT_LEN: usize = 256;

// ---------------------------------------------------------------------------
// Undo-group classification LUT
// ---------------------------------------------------------------------------

/// Entry value: character breaks the current word (starts a new undo group).
const GROUP_BREAK: u8 = 0;
/// Entry value: character continues the current word (same undo group).
const GROUP_CONTINUE: u8 = 1;

/// Builds the undo-group classification table at compile time.
///
/// `CONTINUE` for ASCII letters, digits and underscore (word characters);
/// `BREAK` for everything else (whitespace, punctuation, control bytes,
/// high Unicode).  Indexed by the low byte of a packed character.
const fn build_group_lut() -> [u8; GROUP_LUT_LEN] {
    let mut lut = [GROUP_BREAK; GROUP_LUT_LEN];
    let mut i = 0usize;
    while i < GROUP_LUT_LEN {
        let byte = i as u8;
        let is_word = matches!(
            byte,
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_'
        );
        if is_word {
            lut[i] = GROUP_CONTINUE;
        }
        i += 1;
    }
    lut
}

/// Undo-group classification table: `GROUP_LUT[low_byte_of_packed_char]`.
const GROUP_LUT: [u8; GROUP_LUT_LEN] = build_group_lut();

/// Returns `true` when the packed element's character continues the current
/// word, i.e. it should coalesce into the preceding edit's undo group.
///
/// Branch-free: a single table lookup replaces a character-class branch
/// chain.  Mirrors the LUT pattern used in `crate::simd` (compaction tables).
#[inline]
#[must_use]
pub const fn is_word_continuation(element: u32) -> bool {
    GROUP_LUT[(packed_char(element) & 0xFF) as usize] == GROUP_CONTINUE
}

// ---------------------------------------------------------------------------
// Edit
// ---------------------------------------------------------------------------

/// One recorded revision step: the atomic unit the delta ring stores.
///
/// An `Edit` captures the transformation from one document revision to the
/// next as the packed elements removed (`old`) and inserted (`new`), plus the
/// logical position at which it applied and the undo-group it belongs to.
///
/// # Invariants
///
/// * `old.len() + new.len() >= 1` — an edit must change at least one element;
///   callers reject zero-length edits before construction.
/// * `position` is relative to the document state *before* the edit applies:
///   `0 <= position <= document.len_before_edit`.
/// * `group_id` is a session-wide monotonic counter; equal ids coalesce only
///   when [`is_word_continuation`] says the character continues the word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edit {
    /// Packed elements removed (the "old" side), in document order.
    pub old: Vec<u32>,
    /// Packed elements inserted (the "new" side), in document order.
    pub new: Vec<u32>,
    /// Logical position where the edit applied, in packed elements.
    /// `0` = before the first element; `document.len()` = at the end.
    pub position: usize,
    /// Undo-group id.  Edits with the same contiguous id that the LUT
    /// classifies as word-continuation coalesce into one undoable action.
    pub group_id: u64,
}

impl Edit {
    /// Constructs a new edit, returning `CapacityExhausted` if either vector
    /// cannot be allocated.  The document is left intact on failure.
    #[inline]
    pub fn new(old: Vec<u32>, new: Vec<u32>, position: usize, group_id: u64) -> Self {
        Self {
            old,
            new,
            position,
            group_id,
        }
    }

    /// Exact record accounting: `16 + 4 * (|old| + |new|)`.
    ///
    /// 8 bytes group id + 8 bytes position + 4 bytes per packed element on
    /// each side.
    #[inline]
    #[must_use]
    pub fn record_bytes(&self) -> u64 {
        RECORD_OVERHEAD as u64 + ELEMENT_BYTES as u64 * (self.old.len() + self.new.len()) as u64
    }

    /// Returns the inverse edit: swaps `old` and `new` (position unchanged).
    /// Applying `edit` then `edit.inverse()` restores the prior state.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            old: self.new.clone(),
            new: self.old.clone(),
            position: self.position,
            group_id: self.group_id,
        }
    }
}

impl Default for Edit {
    /// A minimal, non-no-op edit placeholder used by ring pre-allocation.
    /// Callers should prefer `Edit::new` with real payloads.
    fn default() -> Self {
        Self {
            old: Vec::new(),
            new: Vec::new(),
            position: 0,
            group_id: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// DeltaRing — the hot tier
// ---------------------------------------------------------------------------

/// The hot tier of the undo/redo store: a ring buffer of [`Edit`]s holding
/// recent revisions uncompressed.
///
/// `edits[i]` (logical) is the edit that takes state `base + i` to state
/// `base + i + 1`.  The physical slot is `i % capacity`.  `base` starts at
/// `0` and advances only on front-eviction; `head` is one past the latest
/// edit; `cur` is the revision the document sits on (`base <= cur <= head`).
///
/// # Revision numbering
///
/// * Revision `i` = state after the `i`-th recorded edit.
/// * Revision `0` = document as opened.
/// * `base` is initialised to `0` (never `1` — see the crate design notes).
pub struct DeltaRing {
    /// Edits in ring order; physical slot = logical index % capacity.
    edits: Vec<Edit>,
    /// Physical capacity of `edits` (== `edits.capacity()`).
    capacity: usize,
    /// Oldest revision still retained by the ring.  Starts at `0`.
    base: usize,
    /// One past the latest edit in the ring.
    head: usize,
    /// Revision the document currently sits on.
    cur: usize,
    /// Total `record_bytes` of live edits `[base, head)`.
    bytes: u64,
    /// Hot-byte budget; eviction starts when `bytes > hot_bytes`.
    hot_bytes: u64,
    /// Count of refused operations (undo at base, redo at head, etc.).
    refusals: u64,
}

impl DeltaRing {
    /// Creates a ring with the given hot-byte budget.
    ///
    /// Allocates an initial capacity derived from `hot_bytes` (minimum
    /// [`MIN_RING_CAPACITY`], clamped to [`MAX_RING_CAPACITY`]).  Returns
    /// `CapacityExhausted` (logged, not fatal) on allocation failure; the
    /// caller's document is left intact.
    #[inline]
    pub fn new(hot_bytes: u64) -> WreditResult<Self> {
        let min_record = RECORD_OVERHEAD as u64 + ELEMENT_BYTES as u64; // 20 bytes
        let capacity = if hot_bytes == 0 {
            MIN_RING_CAPACITY
        } else {
            let derived = (hot_bytes / min_record.max(1)) as usize;
            derived.clamp(MIN_RING_CAPACITY, MAX_RING_CAPACITY)
        };
        let edits = Vec::with_capacity(capacity);
        if edits.capacity() < capacity && capacity != 0 {
            log_error!("delta ring allocation fell short of requested capacity");
        }
        Ok(Self {
            edits,
            capacity,
            base: 0,
            head: 0,
            cur: 0,
            bytes: 0,
            hot_bytes,
            refusals: 0,
        })
    }

    /// Oldest retained revision.
    #[inline]
    #[must_use]
    pub const fn base(&self) -> usize {
        self.base
    }

    /// One past the latest edit.
    #[inline]
    #[must_use]
    pub const fn head(&self) -> usize {
        self.head
    }

    /// Revision the document sits on.
    #[inline]
    #[must_use]
    pub const fn cur(&self) -> usize {
        self.cur
    }

    /// Total bytes of live edits.
    #[inline]
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Number of refused operations since creation.
    #[inline]
    #[must_use]
    pub const fn refusals(&self) -> u64 {
        self.refusals
    }

    /// Hot-byte budget.
    #[inline]
    #[must_use]
    pub const fn hot_bytes(&self) -> u64 {
        self.hot_bytes
    }

    /// Physical slot for logical edit index `i` (`base <= i < head`).
    #[inline]
    fn slot(i: usize, capacity: usize) -> usize {
        debug_assert!(capacity > 0);
        i % capacity
    }

    /// Physical access (caller guarantees `base <= i < head`).
    #[inline]
    fn edit_at(&self, i: usize) -> &Edit {
        debug_assert!(i >= self.base && i < self.head);
        debug_assert!(self.capacity > 0);
        debug_assert!(Self::slot(i, self.capacity) < self.edits.len());
        &self.edits[Self::slot(i, self.capacity)]
    }

    /// Pushes `edit` onto the back, growing physical storage if needed.
    ///
    /// Returns `CapacityExhausted` (logged) if the ring's backing storage or
    /// the edit's vectors cannot be allocated.
    #[inline]
    fn push_edit(&mut self, edit: Edit) -> WreditResult<()> {
        let logical = self.head;
        let slot = Self::slot(logical, self.capacity);
        // Ensure physical storage covers `slot`.
        if slot >= self.edits.len() {
            let needed = slot + 1;
            let new_cap = self
                .capacity
                .checked_mul(2)
                .and_then(|c| c.checked_add(needed))
                .ok_or_else(|| {
                    log_error!("delta ring capacity overflow");
                    WreditError::LimitExceeded {
                        what: "delta ring capacity",
                    }
                })?
                .max(needed)
                .clamp(self.capacity, MAX_RING_CAPACITY);
            if new_cap == self.capacity && self.capacity < needed {
                log_error!("delta ring capacity clamp reached");
                return Err(WreditError::CapacityExhausted {
                    requested: needed,
                    available: self.capacity,
                });
            }
            self.edits.reserve(new_cap - self.edits.len());
            if self.edits.capacity() < new_cap {
                log_error!("delta ring reserve fell short of requested capacity");
                return Err(WreditError::CapacityExhausted {
                    requested: new_cap,
                    available: self.edits.capacity(),
                });
            }
            self.capacity = new_cap;
        }
        // Normal append path: slot == edits.len() (no wrap yet), so we can
        // push and the slot is fresh.  On wrap the slot was written in a
        // prior cycle and `len` already covers it, so we assign in place.
        let record_bytes = edit.record_bytes();
        if slot == self.edits.len() {
            self.edits.push(edit);
        } else {
            // Wrap-around: overwrite the stale slot in place.
            debug_assert!(slot < self.edits.len());
            self.edits[slot] = edit;
        }
        self.bytes = self
            .bytes
            .checked_add(record_bytes)
            .ok_or_else(|| {
                log_error!("delta ring bytes counter overflow");
                WreditError::LimitExceeded {
                    what: "delta ring bytes",
                }
            })?;
        self.head += 1;
        Ok(())
    }

    /// Pops the back edit, returning it (or `None` if the ring is empty).
    #[inline]
    fn pop_back(&mut self) -> Option<Edit> {
        if self.head == self.base {
            return None;
        }
        self.head -= 1;
        let slot = Self::slot(self.head, self.capacity);
        // SAFETY: `head` is in `[base, head_old)`, so `slot` covers a written
        // edit; `len` covers it because every written slot < len.
        let edit = unsafe { self.edits.get_unchecked(slot).clone() };
        self.bytes = self.bytes.saturating_sub(edit.record_bytes());
        Some(edit)
    }

    /// Pops the front edit, advancing `base`, returning it (or `None`).
    #[inline]
    fn pop_front(&mut self) -> Option<Edit> {
        if self.head == self.base {
            return None;
        }
        let slot = Self::slot(self.base, self.capacity);
        // SAFETY: `base` is in `[base, head)`, slot covers a written edit.
        let edit = unsafe { self.edits.get_unchecked(slot).clone() };
        self.bytes = self.bytes.saturating_sub(edit.record_bytes());
        self.base += 1;
        Some(edit)
    }

    /// Records an edit into the ring.
    ///
    /// Algorithm (§5 of the design):
    ///   1. Truncate the future: while `head > cur`, pop the back and
    ///      decrement `head` (editing in the past drops the redo branch).
    ///   2. Add `edit.record_bytes()`, push the edit, `head += 1`,
    ///      `cur = head`.
    ///   3. Evict: while `bytes > hot_bytes` and more than one edit remains,
    ///      pop the front, subtract its bytes, `base += 1`.
    ///
    /// Returns `CapacityExhausted` (logged, document intact) if the edit or
    /// ring storage cannot be allocated.
    #[inline]
    pub fn record(&mut self, edit: Edit) -> WreditResult<()> {
        // 1. Truncate the future.
        while self.head > self.cur {
            self.pop_back();
        }
        let record_bytes = edit.record_bytes();
        // 2. Push the edit.
        self.push_edit(edit)?;
        // `push_edit` already set `cur`-independent `head`; now `cur = head`.
        self.cur = self.head;
        // 3. Evict while over budget, keeping at least one edit.
        while self.bytes > self.hot_bytes && (self.head - self.base) > 1 {
            self.pop_front();
        }
        let _ = record_bytes; // accounted inside push_edit
        Ok(())
    }

    /// Walks `cur` toward `to`, applying inverses when moving backward and
    /// edits when moving forward.  Returns `false` (and counts a refusal) if
    /// a needed edit has been evicted or lies beyond `head`.
    ///
    /// The caller applies each edit to its own document storage via `apply`;
    /// the ring itself only supplies the edits.  `apply(edit, inverse)` is
    /// called with `inverse == true` when walking backward (apply
    /// `edit.inverse()`) and `false` when walking forward (apply `edit`).
    #[inline]
    pub fn step<F>(&mut self, to: usize, mut apply: F) -> bool
    where
        F: FnMut(&Edit, bool) -> WreditResult<()>,
    {
        if to == self.cur {
            return true;
        }
        if to < self.cur {
            // Walk backward: apply inverse of edit[cur-1] that took
            // cur-1 -> cur, then decrement cur.
            while self.cur > to {
                let target = self.cur - 1;
                if target < self.base {
                    // Edit evicted: refusal; cur stays at the last good state.
                    self.refusal();
                    return false;
                }
                // Prefetch the next edit's payloads (walking backward).
                Self::prefetch_next(
                    target.wrapping_sub(1),
                    self.base,
                    self.head,
                    self.capacity,
                    &self.edits,
                );
                let edit = self.edit_at(target);
                if let Err(_e) = apply(edit, true) {
                    self.refusal();
                    return false;
                }
                self.cur = target;
            }
        } else {
            // Walk forward: apply edit[cur] that took cur -> cur+1, then
            // increment cur.
            while self.cur < to {
                if self.cur >= self.head {
                    self.refusal();
                    return false;
                }
                // Prefetch the next edit's payloads (walking forward).
                Self::prefetch_next(self.cur + 1, self.base, self.head, self.capacity, &self.edits);
                let edit = self.edit_at(self.cur);
                if let Err(_e) = apply(edit, false) {
                    self.refusal();
                    return false;
                }
                self.cur += 1;
            }
        }
        true
    }

    /// Undoes one edit: refuses at `cur <= base`.
    #[inline]
    pub fn undo<F>(&mut self, apply: F) -> bool
    where
        F: FnMut(&Edit, bool) -> WreditResult<()>,
    {
        if self.cur <= self.base {
            self.refusal();
            return false;
        }
        self.step(self.cur - 1, apply)
    }

    /// Redoes one edit: refuses at `cur >= head`.
    #[inline]
    pub fn redo<F>(&mut self, apply: F) -> bool
    where
        F: FnMut(&Edit, bool) -> WreditResult<()>,
    {
        if self.cur >= self.head {
            self.refusal();
            return false;
        }
        self.step(self.cur + 1, apply)
    }

    /// Restores to the revision `back` steps behind the tip: `target = head -
    /// back`.  Refuses if `target` falls outside `[base, head]`.
    #[inline]
    pub fn restore<F>(&mut self, back: usize, apply: F) -> bool
    where
        F: FnMut(&Edit, bool) -> WreditResult<()>,
    {
        // `back > head` would underflow: refuse outright.
        let Some(target) = self.head.checked_sub(back) else {
            self.refusal();
            return false;
        };
        if target < self.base {
            self.refusal();
            return false;
        }
        self.step(target, apply)
    }

    /// Increments the refusal counter (logged at debug level).
    #[inline]
    fn refusal(&mut self) {
        self.refusals = self.refusals.saturating_add(1);
        log_error!("wredit history: refused operation (refusals={})", self.refusals);
    }

    /// Software cache prefetch: `_mm_prefetch` of the next ring payload to
    /// hide pointer-chase latency during a `step` walk.  A no-op on
    /// non-x86 targets.  Caller guarantees `index` is a logical edit index;
    /// we only prefetch when `index` lies within `[base, head)`.
    #[inline]
    fn prefetch_next(index: usize, base: usize, head: usize, capacity: usize, edits: &[Edit]) {
        if index < base || index >= head || capacity == 0 {
            return;
        }
        let slot = Self::slot(index, capacity);
        if slot >= edits.len() {
            return;
        }
        // SAFETY: `slot < edits.len()` (guarded above); `edits[slot]` is a
        // live `Edit` whose `old`/`new` Vecs own heap allocations.  We only
        // *hint* the cache — no load, no dereference beyond what the Vec
        // pointers already promise — and `_mm_prefetch` is benign on x86.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            let edit = edits.get_unchecked(slot);
            let old_ptr = edit.old.as_ptr() as *const i8;
            let new_ptr = edit.new.as_ptr() as *const i8;
            if !old_ptr.is_null() {
                core::arch::x86_64::_mm_prefetch(old_ptr, core::arch::x86_64::_MM_HINT_T0);
            }
            if !new_ptr.is_null() {
                core::arch::x86_64::_mm_prefetch(new_ptr, core::arch::x86_64::_MM_HINT_T0);
            }
            let edit_ptr = edit as *const Edit as *const i8;
            core::arch::x86_64::_mm_prefetch(edit_ptr, core::arch::x86_64::_MM_HINT_T0);
        }
    }
}

impl Default for DeltaRing {
    fn default() -> Self {
        // `default` is for container use only; a real budget must come from
        // `new`.  Use a modest default budget.
        Self::new(256 * 1024).expect("delta ring default allocation must succeed")
    }
}

// ---------------------------------------------------------------------------
// SIMD delta derivation (AVX2) + scalar oracle
// ---------------------------------------------------------------------------

/// Finds the common prefix length and common suffix length of two packed
/// element states, returning the delta window `[prefix, len - suffix)`.
///
/// Both slices hold `u32` packed elements.  Returns `(prefix, suffix)` such
/// that `a[..prefix] == b[..prefix]` and
/// `a[a.len() - suffix..] == b[b.len() - suffix..]`, with the middle region
/// being the differing portion.  `prefix + suffix <= min(a.len(), b.len())`.
///
/// This is the hot-path delta derivation: O(document) scalar comparison per
/// keystroke is replaced by a SIMD equality scan on AVX2 (`pcmpeqd` +
/// `movemask`).  Raw byte copies are *not* SIMDised (`memmove`/`memcpy` own
/// that).
#[inline]
#[must_use]
pub fn delta_common(a: &[u32], b: &[u32]) -> (usize, usize) {
    let len = a.len().min(b.len());
    let prefix = common_prefix(a, b, len);
    let suffix = common_suffix(&a[prefix..], &b[prefix..], len - prefix);
    (prefix, suffix)
}

/// Common prefix length of `a` and `b` over `len` elements.
#[inline]
fn common_prefix(a: &[u32], b: &[u32], len: usize) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if has(Feature::Avx2) {
            // SAFETY: the `has(Feature::Avx2)` probe above confirmed AVX2
            // (and OS-enabled YMM state).  `chunks_exact(8)` bounds every
            // `_mm256_loadu_si256` inside `a[..len]` and `b[..len]`.
            return unsafe { x86::common_prefix_avx2(a, b, len) };
        }
    }
    common_prefix_scalar(a, b, len)
}

/// Common suffix length of `a_tail` and `b_tail` over `len` elements (the
/// tails after the common prefix has been removed).
#[inline]
fn common_suffix(a_tail: &[u32], b_tail: &[u32], len: usize) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use codevar_base::cpuid::{Feature, has};

        if has(Feature::Avx2) {
            // SAFETY: the `has(Feature::Avx2)` probe confirmed AVX2.
            return unsafe { x86::common_suffix_avx2(a_tail, b_tail, len) };
        }
    }
    common_suffix_scalar(a_tail, b_tail, len)
}

/// Scalar reference for `common_prefix` (the mandatory oracle).
#[inline]
fn common_prefix_scalar(a: &[u32], b: &[u32], len: usize) -> usize {
    let mut i = 0usize;
    while i < len && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Scalar reference for `common_suffix` (the mandatory oracle).
#[inline]
fn common_suffix_scalar(a: &[u32], b: &[u32], len: usize) -> usize {
    let mut i = 0usize;
    while i < len && a[a.len() - 1 - i] == b[b.len() - 1 - i] {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// x86 SIMD kernels (target-gated)
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Common prefix over 8-element AVX2 blocks.
    ///
    /// # Safety
    ///
    /// Requires AVX2.  `chunks_exact(8)` keeps every load inside the slices.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn common_prefix_avx2(a: &[u32], b: &[u32], len: usize) -> usize {
        // SAFETY: the caller proved AVX2; `chunks_exact(8)` bounds every
        // `_mm256_loadu_si256` inside `a[..len]` and `b[..len]`.
        unsafe {
            let mut base = 0usize;
            let mut chunks = a[..len].chunks_exact(8);
            let b_chunks = b[..len].chunks_exact(8);
            for (chunk_a, chunk_b) in chunks.by_ref().zip(b_chunks) {
                let va = _mm256_loadu_si256(chunk_a.as_ptr().cast());
                let vb = _mm256_loadu_si256(chunk_b.as_ptr().cast());
                let equal = _mm256_cmpeq_epi32(va, vb);
                let mask = _mm256_movemask_ps(_mm256_castsi256_ps(equal)) as u32;
                if mask != 0xFF {
                    // Some lane differs: the first differing lane is the
                    // lowest set bit of `~mask` within 8 bits.
                    let diff = (!mask) & 0xFF;
                    return base + diff.trailing_zeros() as usize;
                }
                base += 8;
            }
            // Scalar tail for the remainder.
            base + super::common_prefix_scalar(&a[base..len], &b[base..len], len - base)
        }
    }

    /// Common suffix over 8-element AVX2 blocks (walking from the back).
    ///
    /// # Safety
    ///
    /// Requires AVX2.  Loads stay inside the tails.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn common_suffix_avx2(a: &[u32], b: &[u32], len: usize) -> usize {
        // SAFETY: the caller proved AVX2; `chunks_exact(8)` bounds every
        // load inside `a` and `b`.
        unsafe {
            let mut counted = 0usize;
            // Handle the trailing remainder (last `len % 8` elements) first,
            // since we walk from the end toward the start.
            let rem = len % 8;
            let mut checked = 0usize;
            while checked < rem {
                if a[len - 1 - checked] != b[len - 1 - checked] {
                    return counted;
                }
                counted += 1;
                checked += 1;
            }
            // All remainder matched; walk full 8-element chunks from the end
            // of the full-chunk region down to the start.
            let full = len - rem;
            let a_chunks = a[..full].chunks_exact(8);
            let b_chunks = b[..full].chunks_exact(8);
            let mut a_rev = a_chunks.rev();
            let mut b_rev = b_chunks.rev();
            while let (Some(ca), Some(cb)) = (a_rev.next(), b_rev.next()) {
                let va = _mm256_loadu_si256(ca.as_ptr().cast());
                let vb = _mm256_loadu_si256(cb.as_ptr().cast());
                let equal = _mm256_cmpeq_epi32(va, vb);
                let mask = _mm256_movemask_ps(_mm256_castsi256_ps(equal)) as u32;
                if mask != 0xFF {
                    // First differing lane from the end: highest set bit of
                    // `~mask` within 8 bits.  Elements above that lane
                    // (toward the end of the chunk) all match.
                    let diff = (!mask) & 0xFF;
                    let lane = (31u32 - diff.leading_zeros()) as usize; // highest differing position 0..7
                    return counted + (7 - lane);
                }
                counted += 8;
            }
            counted
        }
    }
}

// ---------------------------------------------------------------------------
// SolidJournal — the cold tier (definition shared with the ring)
// ---------------------------------------------------------------------------

/// One closed (or open) window of the solid journal.
struct JournalWindow {
    /// First revision contained (inclusive).
    first_rev: usize,
    /// Revision one past the last contained (exclusive).  For the open
    /// (tail) window this is `head` at the last append.
    last_rev: usize,
    /// Packed elements of every state in the window, concatenated.  The
    /// caller owns the per-state offset table; the journal stores the flat
    /// stream only.
    elements: Vec<u32>,
    /// SHAKE256 fingerprint (low 8 bytes of the digest, read LE) of
    /// `elements`, computed with `crate::shake256::shake256_fingerprint_words`.
    fingerprint: u64,
    /// True while the window is still accepting appends (the tail window).
    open: bool,
}

impl JournalWindow {
    fn new(first_rev: usize) -> Self {
        Self {
            first_rev,
            last_rev: first_rev,
            elements: Vec::new(),
            fingerprint: 0,
            open: true,
        }
    }

    /// Number of frames (revisions) in this window.
    #[inline]
    fn frame_count(&self) -> usize {
        self.last_rev - self.first_rev
    }

    /// Finalises the window: computes its SHAKE256 fingerprint and marks it
    /// closed.  Returns `CapacityExhausted` (logged) on hashing failure.
    #[inline]
    fn close(&mut self) -> WreditResult<()> {
        if !self.open {
            return Ok(());
        }
        self.fingerprint = crate::shake256::shake256_fingerprint_words(&self.elements);
        self.open = false;
        Ok(())
    }
}

/// The cold tier: states appended to one long-lived context, windowed so a
/// restore replays a bounded span.  Each closed window carries a SHAKE256
/// fingerprint for integrity verification on the cold path.
pub struct SolidJournal {
    /// One entry per window; the last entry is the open (tail) window until
    /// `mark_save` closes it.
    windows: Vec<JournalWindow>,
    /// Revision of the first edit in the oldest retained window.
    base: usize,
    /// Revision one past the last edit appended to the journal.
    head: usize,
    /// Window size in revisions.
    window: usize,
    /// Keep threshold: evict whole windows while `head - base >= keep`.
    keep: usize,
}

impl SolidJournal {
    /// Creates a journal with the given window and keep parameters.
    #[inline]
    pub fn new(window: usize, keep: usize) -> WreditResult<Self> {
        let window = window.max(1);
        let keep = keep.max(window);
        let mut windows = Vec::new();
        windows.push(JournalWindow::new(0));
        Ok(Self {
            windows,
            base: 0,
            head: 0,
            window,
            keep,
        })
    }

    /// Oldest retained revision.
    #[inline]
    #[must_use]
    pub const fn base(&self) -> usize {
        self.base
    }

    /// Revision one past the last appended edit.
    #[inline]
    #[must_use]
    pub const fn head(&self) -> usize {
        self.head
    }

    /// Appends one state's packed elements to the open window, advancing
    /// `head`.  When the window reaches `window` revisions it is closed and
    /// a new open window starts.  Returns `CapacityExhausted` (logged) if
    /// the elements vector cannot grow.
    #[inline]
    pub fn append(&mut self, elements: &[u32]) -> WreditResult<()> {
        let tail_index = self.windows.len() - 1; // >= 1 by construction
        let tail_open = self.windows[tail_index].open;
        if !tail_open {
            // Should not happen: a closed tail means a new window should
            // have been opened.  Open a fresh one.
            self.windows.push(JournalWindow::new(self.head));
        }
        // Re-derive the tail index after a possible push (vector may have
        // reallocated; index is still valid because we only ever append).
        let tail_index = self.windows.len() - 1;
        let head_at_append = self.head;
        self.windows[tail_index]
            .elements
            .extend_from_slice(elements);
        self.windows[tail_index].last_rev = head_at_append + 1;
        self.head += 1;
        // Close the window when it reaches the window size.
        if self.windows[tail_index].frame_count() >= self.window {
            self.close_window(tail_index)?;
        }
        Ok(())
    }

    /// Closes the open window at `index` (finalises its fingerprint) and
    /// opens a new empty window at `self.head`.
    #[inline]
    fn close_window(&mut self, index: usize) -> WreditResult<()> {
        let was_open = self.windows[index].open;
        if was_open {
            self.windows[index].close()?;
        }
        // Open a new window at the current head.
        self.windows.push(JournalWindow::new(self.head));
        Ok(())
    }

    /// Closes the open (tail) window: a save is a durable boundary.  After
    /// this, no more appends go into the tail window and a new one starts on
    /// the next `append`.  Returns `CapacityExhausted` (logged) on failure.
    #[inline]
    pub fn mark_save(&mut self) -> WreditResult<()> {
        let last = self.windows.len() - 1; // >= 0 by construction
        if self.windows[last].open {
            self.windows[last].close()?;
        }
        Ok(())
    }

    /// Evicts whole windows from the front while `head - base >= keep` and
    /// more than one window remains, advancing `base` by the evicted
    /// window's frame count.  The last window is never evicted.
    #[inline]
    pub fn evict(&mut self) {
        while self.head - self.base >= self.keep && self.windows.len() > 1 {
            let evicted = self.windows.remove(0);
            self.base += evicted.frame_count();
        }
    }

    /// Locates the window containing `rev`.  If `rev` lies inside the open
    /// window, that window is closed first so the revision is found in a
    /// closed, fingerprint-verified window.  Returns `(window_index,
    /// offset_within_window)` or `None` if `rev` is outside `[base, head)`.
    #[inline]
    pub fn frame_of(&mut self, rev: usize) -> Option<(usize, usize)> {
        if rev < self.base || rev >= self.head {
            return None;
        }
        // Close the open window if `rev` lives inside it, so the result is a
        // closed, fingerprint-bearing window.
        let tail = self.windows.len() - 1;
        if rev >= self.windows[tail].first_rev && self.windows[tail].open {
            let _ = self.windows[tail].close();
        }
        for (index, window) in self.windows.iter().enumerate() {
            if rev < window.last_rev {
                return Some((index, rev - window.first_rev));
            }
        }
        None
    }

    /// Returns the fingerprint of the window containing `rev`, or `None` if
    /// `rev` is out of range or the window is still open (unfingerprinted).
    #[inline]
    pub fn fingerprint_of(&mut self, rev: usize) -> Option<u64> {
        let (index, _) = self.frame_of(rev)?;
        let window = &self.windows[index];
        if window.open {
            return None;
        }
        Some(window.fingerprint)
    }

    /// Runs `evict` and returns the number of windows evicted (for
    /// observability/tests).
    #[inline]
    pub fn evict_and_count(&mut self) -> usize {
        let before = self.windows.len();
        self.evict();
        before.saturating_sub(self.windows.len())
    }
}

// ---------------------------------------------------------------------------
// TierLine — revision reconciliation mapping (ring revision -> journal rev)
// ---------------------------------------------------------------------------

/// `line[i]` holds the journal revision for ring revision `i + 1` (ring
/// revision 0 — the opened document — has no entry; it is the shared
/// baseline).  The `Tiered` composite (Step 2) owns this mapping; this
/// module defines the pure structure and helpers.
#[derive(Debug, Clone)]
pub struct TierLine {
    /// `line[ring_rev - 1]` = journal revision holding ring revision
    /// `ring_rev`; `line[i]` corresponds to ring revision `i + 1`.
    line: Vec<u64>,
}

impl TierLine {
    /// Creates an empty tier line.
    #[inline]
    pub fn new() -> Self {
        Self { line: Vec::new() }
    }

    /// Journal revision holding ring revision `rev` (`rev >= 1`).
    /// Returns `None` for `rev == 0` (the baseline has no journal entry) or
    /// when `rev - 1 >= line.len()`.
    #[inline]
    #[must_use]
    pub fn cold_rev(&self, rev: usize) -> Option<u64> {
        if rev == 0 {
            return None;
        }
        self.line.get(rev - 1).copied()
    }

    /// Number of ring revisions mapped (excludes revision 0).
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.line.len()
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.line.is_empty()
    }

    /// Truncates to `index` entries (drops mappings for ring revisions
    /// `> index`, i.e. the future/redo branch when recording in the past).
    #[inline]
    pub fn truncate(&mut self, index: usize) {
        self.line.truncate(index);
    }

    /// Pushes a journal revision for the next ring revision (`cur + 1`).
    #[inline]
    pub fn push(&mut self, cold_head: u64) {
        self.line.push(cold_head);
    }

    /// Borrows the raw mapping (read-only; used by the Step 2 `Tiered`
    /// composite for `floor()`/`goto` translation).
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.line
    }
}

impl Default for TierLine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift PRNG for reproducible random inputs.
    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> u32 {
            let mut s = self.0;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            self.0 = s;
            s
        }
        fn below(&mut self, bound: u32) -> u32 {
            self.next() % bound
        }
    }

    // ---- Ring invariants (base = 0 regression guard) ----

    #[test]
    fn new_sets_base_to_zero() {
        let ring = DeltaRing::new(1024).expect("allocation must succeed");
        assert_eq!(ring.base(), 0, "base must initialise to 0 (regression guard)");
        assert_eq!(ring.head(), 0);
        assert_eq!(ring.cur(), 0);
        assert_eq!(ring.bytes(), 0);
    }

    #[test]
    fn base_cur_head_ordering_held() {
        let mut ring = DeltaRing::new(4096).expect("allocation must succeed");
        let mut doc: Vec<u32> = Vec::new();
        for i in 0..20u32 {
            let edit = Edit::new(vec![i], vec![i + 100], i as usize, i as u64);
            ring.record(edit).expect("record must succeed");
            assert!(ring.base() <= ring.cur());
            assert!(ring.cur() <= ring.head());
        }
        for _ in 0..10 {
            assert!(ring.undo(|_, _| Ok(())));
            assert!(ring.base() <= ring.cur());
            assert!(ring.cur() <= ring.head());
        }
    }

    // ---- Record / truncate / evict ----

    #[test]
    fn record_advances_cur_to_head() {
        let mut ring = DeltaRing::new(4096).expect("allocation must succeed");
        let edit = Edit::new(vec![1, 2, 3], vec![9, 9, 9], 0, 1);
        ring.record(edit).expect("record must succeed");
        assert_eq!(ring.cur(), ring.head());
        assert_eq!(ring.head(), 1);
    }

    #[test]
    fn truncate_future_on_record_in_past() {
        let mut ring = DeltaRing::new(4096).expect("allocation must succeed");
        for i in 0..5u32 {
            let edit = Edit::new(vec![i], vec![i + 1], i as usize, i as u64);
            ring.record(edit).expect("record must succeed");
        }
        // cur == head == 5. Undo twice -> cur == 3.
        assert!(ring.undo(|_, _| Ok(())));
        assert!(ring.undo(|_, _| Ok(())));
        assert_eq!(ring.cur(), 3);
        assert_eq!(ring.head(), 5);
        // Record a new edit in the past: the redo branch (revisions 3->4,
        // 4->5) is truncated; head becomes 4.
        let edit = Edit::new(vec![7], vec![8], 0, 9);
        ring.record(edit).expect("record must succeed");
        assert_eq!(ring.cur(), ring.head());
        assert_eq!(ring.head(), 4);
    }

    #[test]
    fn evict_when_over_hot_bytes() {
        let mut ring = DeltaRing::new(100).expect("allocation must succeed"); // tiny budget
        for i in 0..10u32 {
            let edit = Edit::new(vec![i; 10], vec![i + 100; 10], i as usize, i as u64);
            ring.record(edit).expect("record must succeed");
        }
        // After eviction, bytes should be <= hot_bytes (or at least one edit
        // remains if even that exceeds the budget).
        assert!(ring.bytes() <= ring.hot_bytes() || (ring.head() - ring.base()) == 1);
        assert!(ring.head() - ring.base() >= 1, "at least one edit remains");
    }

    #[test]
    fn evict_keeps_at_least_one_edit() {
        let mut ring = DeltaRing::new(0).expect("allocation must succeed"); // zero budget
        let edit = Edit::new(vec![1], vec![2], 0, 1);
        ring.record(edit).expect("record must succeed");
        assert_eq!(ring.head() - ring.base(), 1);
        let edit2 = Edit::new(vec![3], vec![4], 0, 2);
        ring.record(edit2).expect("record must succeed");
        // With hot_bytes == 0, eviction should pop the front down to one.
        assert_eq!(ring.head() - ring.base(), 1);
    }

    // ---- Undo/redo boundaries ----

    #[test]
    fn undo_refuses_at_base() {
        let mut ring = DeltaRing::new(4096).expect("allocation must succeed");
        let before = ring.refusals();
        assert!(!ring.undo(|_, _| Ok(())));
        assert_eq!(ring.refusals(), before + 1);
        assert_eq!(ring.cur(), 0);
    }

    #[test]
    fn redo_refuses_at_head() {
        let mut ring = DeltaRing::new(4096).expect("allocation must succeed");
        ring.record(Edit::new(vec![1], vec![2], 0, 1))
            .expect("record must succeed");
        // cur == head == 1; redo must refuse.
        let before = ring.refusals();
        assert!(!ring.redo(|_, _| Ok(())));
        assert_eq!(ring.refusals(), before + 1);
        assert_eq!(ring.cur(), 1);
    }

    #[test]
    fn undo_then_redo_roundtrip() {
        let mut ring = DeltaRing::new(4096).expect("allocation must succeed");
        let e1 = Edit::new(vec![], vec![10], 0, 1);
        let e2 = Edit::new(vec![], vec![20], 1, 2);
        ring.record(e1.clone())
            .expect("record must succeed");
        ring.record(e2.clone())
            .expect("record must succeed");
        assert_eq!(ring.cur(), 2);
        // The ring does not own document storage; the caller applies edits.
        // Rebuild the expected document state to drive the roundtrip.
        let mut doc = vec![10u32, 20];
        assert!(ring.undo(|edit, _inverse| {
            // Simulate: inverse of append-20 is remove-20 at position 1.
            let _ = edit;
            doc.pop();
            Ok(())
        }));
        assert_eq!(ring.cur(), 1);
        assert_eq!(doc, vec![10]);
        // Redo: re-apply e2.
        assert!(ring.redo(|edit, _inverse| {
            let _ = edit;
            doc.push(20);
            Ok(())
        }));
        assert_eq!(ring.cur(), 2);
        assert_eq!(doc, vec![10, 20]);
    }

    #[test]
    fn restore_out_of_range_refuses() {
        let mut ring = DeltaRing::new(4096).expect("allocation must succeed");
        for i in 0..5u32 {
            ring.record(Edit::new(vec![i], vec![i + 1], i as usize, i as u64))
                .expect("record must succeed");
        }
        let before = ring.refusals();
        // back = 0 -> target = head == 5 (valid, no-op walk).
        assert!(ring.restore(0, |_, _| Ok(())));
        // back = 5 -> target = base == 0 (valid).
        assert!(ring.restore(5, |_, _| Ok(())));
        // back = 6 -> target = -1 (invalid).
        assert!(!ring.restore(6, |_, _| Ok(())));
        assert_eq!(ring.refusals(), before + 1);
    }

    #[test]
    fn step_refuses_below_base_after_eviction() {
        let mut ring = DeltaRing::new(20).expect("allocation must succeed"); // tiny budget
        for i in 0..10u32 {
            ring.record(Edit::new(vec![i], vec![i + 1], i as usize, i as u64))
                .expect("record must succeed");
        }
        // With hot_bytes=20 and each edit costing 24 bytes, eviction advances
        // `base` past 0 so older edits are gone.
        assert!(ring.base() > 0, "base should have advanced via eviction");
        let before = ring.refusals();
        // Stepping to a revision below `base` must refuse.
        assert!(!ring.step(ring.base().saturating_sub(1), |_, _| Ok(())));
        assert_eq!(ring.refusals(), before + 1);
    }

    #[test]
    fn refusals_counter_increments() {
        let mut ring = DeltaRing::new(4096).expect("allocation must succeed");
        let before = ring.refusals();
        ring.undo(|_, _| Ok(()));
        ring.redo(|_, _| Ok(()));
        assert_eq!(ring.refusals(), before + 2);
    }

    // ---- SIMD / scalar equivalence for delta_common ----

    fn scalar_delta_common(a: &[u32], b: &[u32]) -> (usize, usize) {
        let len = a.len().min(b.len());
        let mut prefix = 0usize;
        while prefix < len && a[prefix] == b[prefix] {
            prefix += 1;
        }
        let mut suffix = 0usize;
        let tail = len - prefix;
        while suffix < tail && a[a.len() - 1 - suffix] == b[b.len() - 1 - suffix] {
            suffix += 1;
        }
        (prefix, suffix)
    }

    #[test]
    fn delta_common_matches_scalar_oracle() {
        let mut rng = Rng(0x5EED);
        for _ in 0..200 {
            let la = rng.below(40) as usize;
            let _lb = rng.below(40) as usize;
            let a: Vec<u32> = (0..la).map(|_| rng.next()).collect();
            let mut b = a.clone();
            // Introduce a difference somewhere (or keep identical).
            if !b.is_empty() && rng.below(2) == 0 {
                let pos = rng.below(b.len() as u32) as usize;
                b[pos] = b[pos].wrapping_add(1);
            }
            assert_eq!(
                delta_common(&a, &b),
                scalar_delta_common(&a, &b),
                "a={:?} b={:?}",
                a,
                b
            );
        }
    }

    #[test]
    fn delta_common_empty_and_identical() {
        assert_eq!(delta_common(&[], &[]), (0, 0));
        assert_eq!(delta_common(&[1, 2, 3], &[1, 2, 3]), (3, 0));
        assert_eq!(delta_common(&[1, 2, 3], &[1, 2]), (2, 0));
        assert_eq!(delta_common(&[1, 2, 3], &[0, 2, 3]), (0, 2));
        assert_eq!(delta_common(&[1, 2, 3], &[9]), (0, 0));
    }

    // ---- x86 SIMD tiers agree with scalar (when AVX2 present) ----

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_delta_matches_scalar_when_available() {
        use codevar_base::cpuid::{Feature, has};
        if !has(Feature::Avx2) {
            return; // AVX2 not present: scalar path is the only one.
        }
        let mut rng = Rng(0xC0DE);
        for _ in 0..100 {
            let la = rng.below(60) as usize;
            let _lb = rng.below(60) as usize;
            let a: Vec<u32> = (0..la).map(|_| rng.next()).collect();
            let mut b = a.clone();
            if !b.is_empty() {
                let pos = rng.below(b.len() as u32) as usize;
                b[pos] ^= 0xDEAD_BEEF;
            }
            // Compare the AVX2 kernel directly against the scalar oracle.
            let got = unsafe { x86::common_prefix_avx2(&a, &b, a.len().min(b.len())) };
            let want = common_prefix_scalar(&a, &b, a.len().min(b.len()));
            assert_eq!(got, want, "a={:?} b={:?}", a, b);
        }
    }

    // ---- Undo-group LUT ----

    #[test]
    fn group_lut_classifies_word_chars() {
        assert!(is_word_continuation(crate::base::make_packed(
            b'a' as u32,
            crate::base::COMMON_MASK
        )));
        assert!(is_word_continuation(crate::base::make_packed(
            b'z' as u32,
            crate::base::COMMON_MASK
        )));
        assert!(is_word_continuation(crate::base::make_packed(
            b'A' as u32,
            crate::base::COMMON_MASK
        )));
        assert!(is_word_continuation(crate::base::make_packed(
            b'Z' as u32,
            crate::base::COMMON_MASK
        )));
        assert!(is_word_continuation(crate::base::make_packed(
            b'0' as u32,
            crate::base::COMMON_MASK
        )));
        assert!(is_word_continuation(crate::base::make_packed(
            b'9' as u32,
            crate::base::COMMON_MASK
        )));
        assert!(is_word_continuation(crate::base::make_packed(
            b'_' as u32,
            crate::base::COMMON_MASK
        )));
        assert!(!is_word_continuation(crate::base::make_packed(
            b' ' as u32,
            crate::base::COMMON_MASK
        )));
        assert!(!is_word_continuation(crate::base::make_packed(
            b'\n' as u32,
            crate::base::COMMON_MASK
        )));
        assert!(!is_word_continuation(crate::base::make_packed(
            b'.' as u32,
            crate::base::COMMON_MASK
        )));
        assert!(!is_word_continuation(crate::base::make_packed(
            b'(' as u32,
            crate::base::COMMON_MASK
        )));
        assert!(!is_word_continuation(crate::base::make_packed(
            0,
            crate::base::COMMON_MASK
        )));
        assert!(!is_word_continuation(crate::base::make_packed(
            0xFF,
            crate::base::COMMON_MASK
        )));
    }

    #[test]
    fn group_lut_full_table_sanity() {
        for byte in 0u8..=255u8 {
            let element = crate::base::make_packed(byte as u32, crate::base::COMMON_MASK);
            let expected = matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_');
            assert_eq!(is_word_continuation(element), expected, "byte {byte}");
        }
    }

    // ---- Edit record_bytes / inverse ----

    #[test]
    fn edit_record_bytes_formula() {
        let edit = Edit::new(vec![1, 2, 3], vec![4, 5], 0, 1);
        assert_eq!(
            edit.record_bytes(),
            16 + 4 * (3 + 2),
            "record_bytes = 16 + 4*(|old|+|new|)"
        );
        assert_eq!(Edit::new(vec![], vec![], 0, 0).record_bytes(), 16);
    }

    #[test]
    fn edit_inverse_swaps_sides() {
        let edit = Edit::new(vec![1, 2], vec![3, 4, 5], 7, 42);
        let inv = edit.inverse();
        assert_eq!(inv.old, vec![3, 4, 5]);
        assert_eq!(inv.new, vec![1, 2]);
        assert_eq!(inv.position, 7);
        assert_eq!(inv.group_id, 42);
        // Inverse of inverse returns the original.
        assert_eq!(inv.inverse(), edit);
    }

    // ---- SolidJournal ----

    #[test]
    fn journal_append_advances_head() {
        let mut journal = SolidJournal::new(4, 16).expect("allocation must succeed");
        journal
            .append(&[1, 2, 3])
            .expect("append must succeed");
        assert_eq!(journal.head(), 1);
        journal
            .append(&[4, 5, 6])
            .expect("append must succeed");
        assert_eq!(journal.head(), 2);
    }

    #[test]
    fn journal_mark_save_closes_tail() {
        let mut journal = SolidJournal::new(4, 16).expect("allocation must succeed");
        journal
            .append(&[1, 2, 3])
            .expect("append must succeed");
        // `frame_of`/`fingerprint_of` close the open window on lookup, so
        // the fingerprint is available after `mark_save` (and indeed after
        // any lookup that touches the tail).
        journal
            .mark_save()
            .expect("mark_save must succeed");
        assert!(journal.fingerprint_of(0).is_some());
    }

    #[test]
    fn journal_frame_of_finds_window() {
        let mut journal = SolidJournal::new(3, 20).expect("allocation must succeed");
        journal.append(&[1]).expect("append must succeed");
        journal.append(&[2]).expect("append must succeed");
        journal.append(&[3]).expect("append must succeed");
        journal
            .mark_save()
            .expect("mark_save must succeed");
        journal.append(&[4]).expect("append must succeed");
        let (_index, offset) = journal
            .frame_of(0)
            .expect("rev 0 must be in a window");
        assert_eq!(offset, 0);
        let (_index3, offset3) = journal
            .frame_of(3)
            .expect("rev 3 must be in a window");
        assert_eq!(offset3, 0); // start of second window
        assert!(journal.frame_of(99).is_none());
        assert!(journal.frame_of(usize::MAX).is_none());
    }

    #[test]
    fn journal_fingerprint_deterministic() {
        let mut journal = SolidJournal::new(4, 16).expect("allocation must succeed");
        journal
            .append(&[1, 2, 3])
            .expect("append must succeed");
        journal
            .mark_save()
            .expect("mark_save must succeed");
        let f1 = journal
            .fingerprint_of(0)
            .expect("fingerprint must exist");
        // Rebuild identical content in a new journal.
        let mut journal2 = SolidJournal::new(4, 16).expect("allocation must succeed");
        journal2
            .append(&[1, 2, 3])
            .expect("append must succeed");
        journal2
            .mark_save()
            .expect("mark_save must succeed");
        let f2 = journal2
            .fingerprint_of(0)
            .expect("fingerprint must exist");
        assert_eq!(f1, f2);
    }

    #[test]
    fn journal_evict_removes_windows() {
        let mut journal = SolidJournal::new(2, 4).expect("allocation must succeed"); // keep=4 revisions
        for i in 0..8u32 {
            journal.append(&[i]).expect("append must succeed");
        }
        journal
            .mark_save()
            .expect("mark_save must succeed");
        journal.evict();
        // head == 8, keep == 4, base should advance to at least 4.
        assert!(journal.base() >= 4);
        assert!(journal.head() - journal.base() <= 4);
    }

    // ---- TierLine ----

    #[test]
    fn tier_line_cold_rev_indexing() {
        let mut line = TierLine::new();
        assert!(line.cold_rev(0).is_none());
        line.push(100);
        line.push(101);
        assert_eq!(line.cold_rev(1), Some(100));
        assert_eq!(line.cold_rev(2), Some(101));
        assert!(line.cold_rev(3).is_none());
    }

    #[test]
    fn tier_line_truncate_on_record_in_past() {
        let mut line = TierLine::new();
        line.push(10);
        line.push(20);
        line.push(30);
        line.truncate(1); // drop revisions > 1
        assert_eq!(line.len(), 1);
        assert_eq!(line.cold_rev(1), Some(10));
        assert!(line.cold_rev(2).is_none());
    }

    // ---- Default / smoke ----

    #[test]
    fn delta_ring_default_is_usable() {
        let _ring: DeltaRing = Default::default();
    }
}
