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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! # TLSF heap allocator
//!
//! A production-grade **Two-Level Segregated Fit (TLSF)** memory allocator
//! for `no_std` and `std` targets, designed to back Rust's global heap.
//!
//! ## Highlights
//!
//! - **O(1) worst case** allocation, deallocation and in-place resize,
//!   including coalescing of both neighbours.
//! - **SIMD-accelerated**: the first-level class bitmap is scanned with
//!   AVX2/NEON kernels dispatched at run time through
//!   `codevar_base::basic_cpuid`; `alloc_zeroed` fills with SIMD stores.
//! - **`no_std` + `const`-constructible**: a heap can live in a `static`
//!   and be armed at boot with [`Tlsf::add_region`]; the control structure
//!   is ~8 KiB and never allocates.
//! - **Global-allocator ready**: [`LockedTlsf`] implements
//!   [`GlobalAlloc`](core::alloc::GlobalAlloc) (including `realloc` and
//!   `alloc_zeroed` overrides) behind a `spin::Mutex`, and **aborts the
//!   process on OOM** via
//!   [`handle_alloc_error`](alloc::alloc::handle_alloc_error).
//! - **Self-growing (Unix)**: [`LockedGrowableTlsf`] is a drop-in
//!   [`LockedTlsf`] replacement that needs no manual region setup — when
//!   the free lists cannot satisfy a request it maps a fresh chunk of
//!   virtual pages (`mmap`, never `malloc`, so it is safe to use as the
//!   global allocator itself) and grows geometrically.
//! - **Hardware-derived alignment**: the minimum block alignment
//!   ([`ALIGNMENT`]) is derived from `size_of::<usize>()`, and every
//!   allocation honours the exact `Layout::align()` of the request —
//!   nothing is hardcoded.
//! - **Robust**: checked arithmetic on every size computation, boundary
//!   tags with per-block free/prev-free flags, a sentinel header at each
//!   region end, and no panicking paths in production code.
//!
//! ## Usage as the global allocator
//!
//! ```no_run
//! use codevar_tlsf_alloc::LockedTlsf;
//! use core::alloc::GlobalAlloc;
//!
//! #[global_allocator]
//! static HEAP: LockedTlsf = LockedTlsf::new();
//!
//! # fn boot() {
//! let mut region = [0u64; 4096];
//! let ptr = region.as_mut_ptr().cast::<u8>();
//! let len = core::mem::size_of_val(&region);
//! // SAFETY: `region` is live, writable and owned by this boot path.
//! if unsafe { HEAP.lock().add_region(ptr, len) }.is_err() {
//!     // Region too small — real firmware would halt or fall back.
//! }
//! // From here on, Rust's global allocations are served by the TLSF heap.
//! let v = vec![1u8; 33];
//! assert_eq!(v.len(), 33);
//! # }
//! ```
//!
//! ## Performance characteristics
//!
//! | Operation                        | Cost                             |
//! |----------------------------------|----------------------------------|
//! | `allocate` / `deallocate`        | O(1), bounded list hops          |
//! | in-place `realloc` (grow/shrink) | O(1)                             |
//! | neighbour coalescing             | O(1) (boundary tags)             |
//! | first-level bitmap scan          | 1–2 SIMD words (AVX2/NEON)       |
//! | control structure                | ~8 KiB, static, no allocations   |
//!
//! Worst-case execution time is bounded by the size-class search plus at
//! most 16 list hops, independent of heap size or fragmentation —
//! suitable for real-time and interrupt-context heaps (given a lock-free
//! or externally locked use of [`Tlsf`]).
//!
//! ## Memory overhead
//!
//! Every block costs a 2-word header (previous size, size+flags); free
//! blocks additionally overlay two link words, so the minimum block is
//! `MINSIZE` = 4 words (32 bytes on 64-bit). Each added region wastes one
//! sentinel header (2 words) plus at most `2 * ALIGNMENT - 1` bytes of
//! boundary rounding.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod simd;

use alloc::alloc::handle_alloc_error;
use core::alloc::{GlobalAlloc, Layout};
use core::error::Error;
use core::fmt;
use core::mem::size_of;
use core::ops::Deref;
use core::ptr::{self, NonNull};
use spin::Mutex;

/// Pointer-width word size in bytes.
const WORD: usize = size_of::<usize>();

/// Minimum alignment (in bytes) of every block handed out by the heap.
///
/// Derived from the hardware pointer width — `2 × size_of::<usize>()` —
/// which is the classical TLSF granularity: the smallest power of two that
/// keeps the 2-word block header naturally aligned on every architecture
/// Rust targets. Individual allocations may request (and always receive)
/// any larger power-of-two alignment via `Layout::align()`.
pub const ALIGNMENT: usize = WORD << 1;

/// Size in bytes of the always-present block header.
const HEADER_SIZE: usize = WORD << 1;

/// Flag: this block is currently free (on a free list).
const FLAG_FREE: usize = 1 << 0;

/// Flag: the immediately preceding block is free.
const FLAG_PREV_FREE: usize = 1 << 1;

/// Mask extracting the block size from a `size_flags` word.
const SIZE_MASK: usize = !(FLAG_FREE | FLAG_PREV_FREE);

/// Minimum block size: header (2 words) + free-list links (2 words).
const MINSIZE: usize = 4 * WORD;

/// `log2` of the number of second-level classes per first-level class.
const SL_SHIFT: usize = 4;

/// Number of second-level classes per first-level class.
const SL_COUNT: usize = 1 << SL_SHIFT;

/// Sizes below this limit live in first-level class 0 (linear sub-classes).
const SMALL_SIZE_LIMIT: usize = SL_COUNT * ALIGNMENT;

/// Number of first-level size classes (`61` on 64-bit, `29` on 32-bit).
const FL_COUNT: usize = usize::BITS as usize - SL_SHIFT + 1;

/// Number of machine words backing the first-level bitmap.
const FL_BITMAP_WORDS: usize = FL_COUNT.div_ceil(64);

/// Total number of size-class free lists.
const FREE_LIST_COUNT: usize = FL_COUNT * SL_COUNT;

/// Sentinel used instead of a null pointer in free-list links.
const NULL: usize = 0;

/// On-block header stored at the start of every block.
///
/// Layout relative to the block address (offsets in words):
///
/// ```text
/// 0: prev_size  — size of the previous block, valid iff FLAG_PREV_FREE
/// 1: size_flags — size of this block | FLAG_FREE | FLAG_PREV_FREE
/// 2: fd         — next free block, valid only while FLAG_FREE
/// 3: bk         — previous free block, valid only while FLAG_FREE
/// ```
#[repr(C)]
struct BlockHeader {
    prev_size: usize,
    size_flags: usize,
    fd: usize,
    bk: usize,
}

/// Reads the `size_flags` word of the block at `addr`.
///
/// # Safety
///
/// `addr` must be the start of a block belonging to this heap.
#[inline]
unsafe fn read_size_flags(addr: usize) -> usize {
    // SAFETY: the caller guarantees `addr` is a block start; the field is
    // always written when the block is created, and `addr` is aligned to
    // `ALIGNMENT`, which satisfies the header's alignment requirement.
    unsafe { ptr::addr_of!((*(addr as *const BlockHeader)).size_flags).read() }
}

/// Writes the `size_flags` word of the block at `addr`.
///
/// # Safety
///
/// `addr` must be the start of a block belonging to this heap.
#[inline]
unsafe fn write_size_flags(addr: usize, size_flags: usize) {
    // SAFETY: the caller guarantees `addr` is a block start aligned for
    // the header; the write covers exactly the `size_flags` field.
    unsafe {
        ptr::addr_of_mut!((*(addr as *mut BlockHeader)).size_flags).write(size_flags);
    }
}

/// Reads the `prev_size` word of the block at `addr`.
///
/// # Safety
///
/// `addr` must be the start of a block whose `FLAG_PREV_FREE` flag is set
/// (that flag is what makes the word valid).
#[inline]
unsafe fn read_prev_size(addr: usize) -> usize {
    // SAFETY: the caller guarantees the flag is set, which means a free
    // neighbour wrote this word when it was linked.
    unsafe { ptr::addr_of!((*(addr as *const BlockHeader)).prev_size).read() }
}

/// Writes the `prev_size` word of the block at `addr`.
///
/// # Safety
///
/// `addr` must be the start of a block belonging to this heap.
#[inline]
unsafe fn write_prev_size(addr: usize, prev_size: usize) {
    // SAFETY: the caller guarantees `addr` is a block start aligned for
    // the header; the write covers exactly the `prev_size` field.
    unsafe {
        ptr::addr_of_mut!((*(addr as *mut BlockHeader)).prev_size).write(prev_size);
    }
}

/// Reads the free-list links `(fd, bk)` of the free block at `addr`.
///
/// # Safety
///
/// `addr` must be the start of a **free** block of this heap (only free
/// blocks have their link words initialized).
#[inline]
unsafe fn read_links(addr: usize) -> (usize, usize) {
    // SAFETY: the caller guarantees the block is free, so `fd`/`bk` were
    // written by `push_list` and remain valid while the block is listed.
    unsafe {
        let base = addr as *const BlockHeader;
        (ptr::addr_of!((*base).fd).read(), ptr::addr_of!((*base).bk).read())
    }
}

/// Writes the free-list links of the block at `addr`.
///
/// # Safety
///
/// `addr` must be the start of a block of this heap that is being placed
/// on (or relinked within) a free list.
#[inline]
unsafe fn write_links(addr: usize, fd: usize, bk: usize) {
    // SAFETY: the caller guarantees `addr` is a block start aligned for
    // the header; the writes cover exactly the `fd`/`bk` fields.
    unsafe {
        let base = addr as *mut BlockHeader;
        ptr::addr_of_mut!((*base).fd).write(fd);
        ptr::addr_of_mut!((*base).bk).write(bk);
    }
}

/// Computes the free-list index of size class `(fl, sl)`.
#[inline]
const fn list_index(fl: usize, sl: usize) -> usize {
    fl * SL_COUNT + sl
}

/// Rounds `value` up to the next multiple of the power of two `align`.
#[inline]
fn checked_align_up(value: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    value
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
}

/// Maps a block size to its first- and second-level size class.
///
/// The size must be at least [`MINSIZE`]. Sizes handed to this function by
/// [`mapping_search`] are deliberately *not* `ALIGNMENT`-multiples: the
/// rounding step only needs the arithmetic below, which is exact for any
/// positive size, and every block actually stored in a class is a multiple
/// of [`ALIGNMENT`].
///
/// Class `(0, sl)` covers the linear range `[sl·ALIGNMENT, (sl+1)·ALIGNMENT)`;
/// class `(fl, sl)` for `fl ≥ 1` covers
/// `[(16+sl)·2^(fl-1), (17+sl)·2^(fl-1))`. The mapping is total for every
/// `usize` size: the largest class index is `FL_COUNT - 1`.
#[inline]
fn mapping(size: usize) -> (usize, usize) {
    debug_assert!(size >= MINSIZE);
    if size < SMALL_SIZE_LIMIT {
        (0, size / ALIGNMENT)
    } else {
        let fl = (usize::BITS - 1 - size.leading_zeros()) as usize;
        let sl = (size >> (fl - SL_SHIFT)) & (SL_COUNT - 1);
        (fl - SL_SHIFT + 1, sl)
    }
}

/// Maps a request size to the smallest size class whose lower bound is
/// greater than or equal to the request, so that any block popped from
/// that class (or a higher one) is guaranteed to be large enough.
///
/// This is the standard TLSF "round up before mapping" step.
#[inline]
fn mapping_search(size: usize) -> (usize, usize) {
    if size >= SMALL_SIZE_LIMIT {
        let fl = (usize::BITS - 1 - size.leading_zeros()) as usize;
        let round = (1usize << (fl - SL_SHIFT + 1)) - 1;
        mapping(size.saturating_add(round))
    } else {
        mapping(size)
    }
}

/// Computes the lower bound, in bytes, of size class `(fl, sl)`.
///
/// The bound is the smallest block size that maps to the class, and is
/// always a multiple of [`ALIGNMENT`]. Returns `None` on overflow (only
/// reachable for classes near the top of the size space).
#[inline]
fn class_lower_bound(fl: usize, sl: usize) -> Option<usize> {
    debug_assert!(fl < FL_COUNT && sl < SL_COUNT);
    if fl == 0 {
        Some(sl * ALIGNMENT)
    } else {
        (SL_COUNT + sl).checked_mul(1usize << (fl - 1))
    }
}

/// Failure modes of [`Tlsf::add_region`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddRegionError {
    /// `start + size` overflows the address space.
    Overflow,
    /// The aligned region cannot hold one minimum block plus the
    /// end-of-region sentinel header.
    TooSmall,
}

impl fmt::Display for AddRegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => f.write_str("region start + size overflows the address space"),
            Self::TooSmall => f.write_str("region too small for a block and a sentinel header"),
        }
    }
}

impl Error for AddRegionError {}

/// A snapshot of heap usage, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Total bytes managed by the heap (sum of all added regions).
    pub total_bytes: usize,
    /// Bytes currently handed out to live allocations (including their
    /// block headers).
    pub used_bytes: usize,
}

impl Stats {
    /// Bytes currently on the free lists.
    #[inline]
    pub fn free_bytes(&self) -> usize {
        self.total_bytes.saturating_sub(self.used_bytes)
    }
}

/// A Two-Level Segregated Fit heap.
///
/// The heap starts empty; memory is contributed with [`Tlsf::add_region`]
/// (any number of disjoint regions). All methods take `&mut self`, so use
/// it single-threaded or wrap it in a lock — [`LockedTlsf`] does exactly
/// that and adds the [`GlobalAlloc`](core::alloc::GlobalAlloc) impl.
///
/// `Tlsf` contains no heap pointers into Rust-managed memory and no
/// allocations of its own; it is `Send` and can live in a `static` via
/// [`Tlsf::new`] (a `const fn`).
pub struct Tlsf {
    /// Bit `fl` is set iff first-level class `fl` has a non-empty
    /// second-level class. Scanned with the SIMD kernels in [`simd`].
    fl_bitmap: [u64; FL_BITMAP_WORDS],
    /// Bit `sl` of `sl_bitmaps[fl]` is set iff class `(fl, sl)` is
    /// non-empty.
    sl_bitmaps: [u16; FL_COUNT],
    /// Head pointer (block address, `NULL` when empty) of every
    /// size-class free list, indexed by [`list_index`].
    lists: [usize; FREE_LIST_COUNT],
    /// Sum of the usable spans of all added regions.
    total_bytes: usize,
    /// Sum of the sizes of all currently allocated blocks.
    used_bytes: usize,
}

impl Tlsf {
    /// Creates an empty heap in constant time.
    ///
    /// The result can be stored in a `static`; arm it at boot with
    /// [`Tlsf::add_region`].
    pub const fn new() -> Self {
        Self {
            fl_bitmap: [0; FL_BITMAP_WORDS],
            sl_bitmaps: [0; FL_COUNT],
            lists: [NULL; FREE_LIST_COUNT],
            total_bytes: 0,
            used_bytes: 0,
        }
    }

    /// Returns a snapshot of the heap usage.
    #[inline]
    pub fn stats(&self) -> Stats {
        Stats {
            total_bytes: self.total_bytes,
            used_bytes: self.used_bytes,
        }
    }

    /// Adds the memory range `[start, start + size)` to the heap.
    ///
    /// The range is rounded inward to an [`ALIGNMENT`] boundary; a
    /// sentinel header is placed at its end so the last block never
    /// coalesces out of the region. Regions may be added at any time and
    /// need not be power-of-two sized.
    ///
    /// # Safety
    ///
    /// - The range must be valid, writable memory that outlives the heap.
    /// - It must not be used by any other allocator or alias any memory
    ///   reachable through this heap, including regions added earlier.
    /// - No part of the range may be accessed through `start`/`size`
    ///   (or any other pointer) for as long as the heap manages it,
    ///   except via this heap's own allocations.
    pub unsafe fn add_region(&mut self, start: *mut u8, size: usize) -> Result<(), AddRegionError> {
        let start_addr = start as usize;
        let end = start_addr
            .checked_add(size)
            .ok_or(AddRegionError::Overflow)?;
        let aligned_start = checked_align_up(start_addr, ALIGNMENT).ok_or(AddRegionError::Overflow)?;
        let aligned_end = end & !(ALIGNMENT - 1);
        if aligned_end < aligned_start {
            return Err(AddRegionError::TooSmall);
        }
        let span = aligned_end - aligned_start;
        // Reserve the sentinel header; the rest becomes one free block.
        if span < MINSIZE + HEADER_SIZE {
            return Err(AddRegionError::TooSmall);
        }
        let block_size = span - HEADER_SIZE;
        let sentinel = aligned_start + block_size;
        // SAFETY: `[aligned_start, aligned_end)` is the caller-guaranteed
        // region, rounded inward; the sentinel header sits inside it.
        unsafe {
            write_prev_size(sentinel, 0);
            // Sentinel: marked allocated so no coalescing walks past it.
            // Its size is never followed because FLAG_FREE stays clear.
            write_size_flags(sentinel, HEADER_SIZE);
            self.insert_block(aligned_start, block_size, 0);
        }
        self.total_bytes += block_size;
        Ok(())
    }

    /// Allocates a block suitable for `size` bytes with alignment `align`.
    ///
    /// A `size` of zero allocates a minimum block. `align` must be a power
    /// of two; alignments larger than [`ALIGNMENT`] are satisfied by
    /// over-allocating and carving an aligned window out of the block,
    /// returning any margins to the free lists.
    ///
    /// Returns `None` on out of memory or an invalid `align`.
    ///
    /// # Performance
    ///
    /// O(1): one bitmap probe, one list pop and at most two list inserts
    /// (large-alignment carve path).
    #[must_use]
    pub fn allocate(&mut self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if align == 0 || !align.is_power_of_two() {
            return None;
        }
        let inner = Self::block_size_for(size)?;
        if align <= ALIGNMENT {
            let addr = self.pop_fit(inner)?;
            // SAFETY: `addr` is a free block of this heap, popped above.
            let actual = unsafe { self.finish_alloc(addr, inner) };
            self.used_bytes += actual;
            // SAFETY: `addr + HEADER_SIZE` is inside the region and non-null.
            Some(unsafe { NonNull::new_unchecked((addr + HEADER_SIZE) as *mut u8) })
        } else {
            self.carve_aligned(align, inner)
        }
    }

    /// Frees the block previously returned by [`Tlsf::allocate`] at
    /// `ptr`, coalescing it with any free neighbours.
    ///
    /// # Safety
    ///
    /// `ptr` must originate from [`Tlsf::allocate`] on this very heap and
    /// must not have been deallocated already.
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>) {
        let addr = ptr.as_ptr() as usize - HEADER_SIZE;
        // SAFETY: the caller guarantees `ptr` is a live allocation of this
        // heap, so `addr` is one of its block starts.
        unsafe { self.free_block(addr) };
    }

    /// Returns the number of free blocks currently on the free lists.
    ///
    /// Walks every size class, so this is O(number of free blocks) —
    /// intended for diagnostics and tests, not hot paths.
    pub fn free_block_count(&self) -> usize {
        let mut count = 0;
        for fl in 0..FL_COUNT {
            for sl in 0..SL_COUNT {
                let mut cursor = self.lists[list_index(fl, sl)];
                while cursor != NULL {
                    count += 1;
                    // SAFETY: every block on a free list is a free block
                    // of this heap whose links were written by `push_list`.
                    cursor = unsafe { read_links(cursor) }.0;
                }
            }
        }
        count
    }

    /// Computes the allocated block size (header included) for a user
    /// request of `size` bytes, or `None` on overflow.
    #[inline]
    fn block_size_for(size: usize) -> Option<usize> {
        let with_header = size.checked_add(HEADER_SIZE)?;
        let aligned = checked_align_up(with_header, ALIGNMENT)?;
        Some(aligned.max(MINSIZE))
    }

    /// Allocates with `align > ALIGNMENT` by over-allocating and carving
    /// an aligned window out of the block.
    ///
    /// `inner` must be the result of [`Tlsf::block_size_for`] for the
    /// original user request.
    fn carve_aligned(&mut self, align: usize, inner: usize) -> Option<NonNull<u8>> {
        // Extra slack: up to `align` bytes of front margin plus `MINSIZE`
        // so both the front and back remainders can always be split off
        // (or absorbed) legally.
        let outer = inner.checked_add(align)?.checked_add(MINSIZE)?;
        let outer = checked_align_up(outer, ALIGNMENT)?;
        let addr = self.pop_fit(outer)?;
        // SAFETY: `addr` is a free block of this heap, popped above.
        let (total, prev_free) = unsafe {
            let sf = read_size_flags(addr);
            (sf & SIZE_MASK, sf & FLAG_PREV_FREE)
        };
        debug_assert!(total >= outer);
        let base = addr + HEADER_SIZE;
        let Some(mut user) = base
            .checked_add(align - 1)
            .map(|v| v & !(align - 1))
        else {
            // Roll the untouched block back onto its free list.
            // SAFETY: the block is free and was only popped, not modified.
            unsafe { self.push_list(addr, total) };
            return None;
        };
        let mut front = user - HEADER_SIZE - addr;
        if front != 0 && front < MINSIZE {
            // Shift the window forward by a full `align` (≥ 2·ALIGNMENT =
            // MINSIZE) so the front margin becomes a legal free block.
            user += align;
            front += align;
        }
        debug_assert!(front + inner <= total);
        let mid = user - HEADER_SIZE;
        // `mid + inner + tail == addr + total`: `front` already accounts
        // for the header bytes between `addr` and `mid`, so subtracting
        // HEADER_SIZE again would leave an untracked gap at the end of
        // the popped block and corrupt the following header.
        let tail = total - front - inner;
        let allocated = if tail >= MINSIZE {
            let tail_addr = mid + inner;
            // SAFETY: `[tail_addr, tail_addr + tail)` is the back margin
            // of the popped block; its predecessor becomes allocated.
            unsafe { self.insert_block(tail_addr, tail, 0) };
            inner
        } else {
            // Absorb a tail too small to form a legal free block.
            // SAFETY: `addr + total` is the block following the popped
            // block; its predecessor is now allocated, so clear the flag.
            unsafe {
                let after = addr + total;
                let asf = read_size_flags(after);
                write_size_flags(after, asf & !FLAG_PREV_FREE);
            }
            inner + tail
        };
        // SAFETY: `mid` is the carved block header position; when there is
        // no front margin its predecessor is the original one, otherwise
        // `insert_block` below ORs FLAG_PREV_FREE in for us.
        unsafe { write_size_flags(mid, allocated | if front == 0 { prev_free } else { 0 }) };
        if front > 0 {
            // SAFETY: the front margin is a multiple of ALIGNMENT and at
            // least MINSIZE; its predecessor is the original one.
            unsafe { self.insert_block(addr, front, prev_free) };
        }
        self.used_bytes += allocated;
        // SAFETY: `user` is inside the region, non-null and aligned.
        Some(unsafe { NonNull::new_unchecked(user as *mut u8) })
    }

    /// Completes an allocation popped for a block of at least `min_size`
    /// bytes, splitting (or absorbing) the excess, and returns the actual
    /// allocated block size.
    ///
    /// # Safety
    ///
    /// `addr` must be the start of a free block of this heap, of at least
    /// `min_size` bytes, that has already been removed from the free lists.
    unsafe fn finish_alloc(&mut self, addr: usize, min_size: usize) -> usize {
        // SAFETY: the block is free (so its header is fully written).
        let sf = unsafe { read_size_flags(addr) };
        let total = sf & SIZE_MASK;
        let prev_free = sf & FLAG_PREV_FREE;
        debug_assert!(total >= min_size);
        let remainder = total - min_size;
        if remainder >= MINSIZE {
            let tail = addr + min_size;
            // SAFETY: the back margin is a legal free block whose
            // predecessor (the block handed to the user) is allocated.
            unsafe { self.insert_block(tail, remainder, 0) };
            // SAFETY: shrinking the block; the predecessor is unchanged.
            unsafe { write_size_flags(addr, min_size | prev_free) };
            min_size
        } else {
            // Absorb the unusable tail into the allocation.
            // SAFETY: the following block's predecessor is now allocated.
            unsafe {
                let after = addr + total;
                let asf = read_size_flags(after);
                write_size_flags(after, asf & !FLAG_PREV_FREE);
            }
            // SAFETY: only the size grows; flags stay as they are.
            unsafe { write_size_flags(addr, total | prev_free) };
            total
        }
    }

    /// Frees the block at `addr`, coalescing with both neighbours.
    ///
    /// # Safety
    ///
    /// `addr` must be the start of an allocated block of this heap that is
    /// not already free.
    unsafe fn free_block(&mut self, mut addr: usize) {
        // SAFETY: the block is allocated, so its header is fully written.
        let sf = unsafe { read_size_flags(addr) };
        debug_assert!(sf & FLAG_FREE == 0, "double free");
        let mut size = sf & SIZE_MASK;
        let mut prev_free = sf & FLAG_PREV_FREE;
        self.used_bytes -= size;

        // Coalesce with the free block before this one.
        if prev_free != 0 {
            // SAFETY: FLAG_PREV_FREE guarantees `prev_size` is valid.
            let prev_size = unsafe { read_prev_size(addr) };
            let prev = addr - prev_size;
            // SAFETY: `prev` is a free block of this heap.
            unsafe {
                let psf = read_size_flags(prev);
                self.remove_from_list(prev, prev_size);
                prev_free = psf & FLAG_PREV_FREE;
            }
            size += prev_size;
            addr = prev;
        }

        // Coalesce with free blocks after this one.
        loop {
            let next = addr + size;
            // SAFETY: `next` is a block of this heap (bounded by the
            // region sentinel, which is never free).
            let nsf = unsafe { read_size_flags(next) };
            if nsf & FLAG_FREE == 0 {
                break;
            }
            let next_size = nsf & SIZE_MASK;
            // SAFETY: `next` is a free block of this heap.
            unsafe { self.remove_from_list(next, next_size) };
            size += next_size;
        }

        // SAFETY: the merged span is a legal free block not overlapping
        // any other free block (both neighbours were absorbed).
        unsafe { self.insert_block(addr, size, prev_free) };
    }

    /// Tries to resize the block at `addr` to hold `new_size` bytes with
    /// alignment `align`, in place. Returns the new block size.
    ///
    /// # Safety
    ///
    /// `addr` must be the start of an allocated block of this heap.
    unsafe fn resize_in_place(&mut self, addr: usize, new_size: usize, align: usize) -> Option<usize> {
        // In-place resize never moves the block; if the user asked for a
        // stronger alignment than the block address provides, fall back to
        // allocate-copy-free.
        if align > ALIGNMENT && (addr + HEADER_SIZE) & (align - 1) != 0 {
            return None;
        }
        let needed = Self::block_size_for(new_size)?;
        // SAFETY: the block is allocated, so its header is fully written.
        let sf = unsafe { read_size_flags(addr) };
        let current = sf & SIZE_MASK;
        let prev_free = sf & FLAG_PREV_FREE;

        if needed <= current {
            let remainder = current - needed;
            if remainder >= MINSIZE {
                let tail = addr + needed;
                // SAFETY: the back margin is a legal free block; the
                // block handed back to the user is already allocated.
                unsafe {
                    self.insert_block(tail, remainder, 0);
                    write_size_flags(addr, needed | prev_free);
                }
                return Some(needed);
            }
            return Some(current);
        }

        // Grow by absorbing the following free block, if large enough.
        let next = addr + current;
        // SAFETY: `next` is a block of this heap, bounded by the sentinel.
        let nsf = unsafe { read_size_flags(next) };
        if nsf & FLAG_FREE == 0 {
            return None;
        }
        let next_size = nsf & SIZE_MASK;
        let total = current + next_size;
        if total < needed {
            return None;
        }
        // SAFETY: `next` is a free block of this heap.
        unsafe { self.remove_from_list(next, next_size) };
        let remainder = total - needed;
        if remainder >= MINSIZE {
            let tail = addr + needed;
            // SAFETY: the back margin is a legal free block.
            unsafe {
                self.insert_block(tail, remainder, 0);
                write_size_flags(addr, needed | prev_free);
            }
            Some(needed)
        } else {
            // SAFETY: the block after the absorbed span now follows an
            // allocated block, so its predecessor flag must be cleared.
            unsafe {
                write_size_flags(addr, total | prev_free);
                let after = addr + total;
                let asf = read_size_flags(after);
                write_size_flags(after, asf & !FLAG_PREV_FREE);
            }
            Some(total)
        }
    }

    /// Pops a block of at least `min_size` bytes from the free lists.
    fn pop_fit(&mut self, min_size: usize) -> Option<usize> {
        let (fl, sl) = mapping_search(min_size);
        // Same first-level class, same or higher second-level class.
        debug_assert!(fl < FL_COUNT && sl < SL_COUNT);
        let candidates = self.sl_bitmaps[fl] & !((1u16 << sl) - 1);
        if candidates != 0 {
            let next_sl = candidates.trailing_zeros() as usize;
            return self.pop_list(fl, next_sl);
        }
        // Otherwise the next non-empty first-level class (SIMD bitmap scan).
        let index = simd::find_first_set_from(&self.fl_bitmap, (fl + 1) as u32)?;
        let next_fl = index as usize;
        debug_assert!(next_fl < FL_COUNT);
        let next_sl = self.sl_bitmaps[next_fl].trailing_zeros() as usize;
        self.pop_list(next_fl, next_sl)
    }

    /// Pushes the free block at `addr` of `size` bytes onto its class list.
    ///
    /// # Safety
    ///
    /// `addr` must be the start of a free block of this heap, `size` its
    /// size (a multiple of [`ALIGNMENT`], at least [`MINSIZE`]), and the
    /// block must not be on any free list.
    unsafe fn push_list(&mut self, addr: usize, size: usize) {
        let (fl, sl) = mapping(size);
        debug_assert!(fl < FL_COUNT && sl < SL_COUNT);
        let index = list_index(fl, sl);
        let head = self.lists[index];
        // SAFETY: `addr` is a free block of this heap; its link fields are
        // about to be initialized for list membership.
        unsafe { write_links(addr, head, NULL) };
        if head != NULL {
            // SAFETY: `head` is the free-list head, so it is a free block
            // with valid links.
            let head_fd = unsafe { read_links(head).0 };
            unsafe { write_links(head, head_fd, addr) };
        }
        self.lists[index] = addr;
        self.sl_bitmaps[fl] |= 1u16 << sl;
        self.fl_bitmap[fl / 64] |= 1u64 << (fl % 64);
    }

    /// Pops the head of the free list of class `(fl, sl)`.
    fn pop_list(&mut self, fl: usize, sl: usize) -> Option<usize> {
        debug_assert!(fl < FL_COUNT && sl < SL_COUNT);
        let index = list_index(fl, sl);
        let head = self.lists[index];
        if head == NULL {
            return None;
        }
        // SAFETY: `head` is a free block of this heap on this list, so its
        // links are valid.
        let next = unsafe { read_links(head).0 };
        self.lists[index] = next;
        if next != NULL {
            // SAFETY: `next` is now the list head (a free block).
            let next_fd = unsafe { read_links(next).0 };
            unsafe { write_links(next, next_fd, NULL) };
        } else {
            self.clear_class_bits(fl, sl);
        }
        Some(head)
    }

    /// Removes the free block at `addr` of `size` bytes from its class list.
    ///
    /// # Safety
    ///
    /// `addr` must currently be on the free list of the class matching
    /// `size`.
    unsafe fn remove_from_list(&mut self, addr: usize, size: usize) {
        let (fl, sl) = mapping(size);
        debug_assert!(fl < FL_COUNT && sl < SL_COUNT);
        // SAFETY: the block is on a free list, so its links are valid.
        let (fd, bk) = unsafe { read_links(addr) };
        if bk != NULL {
            // SAFETY: `bk` is a free block on the same list; preserve    // its own back pointer while rerouting its forward pointer.
            let bk_bk = unsafe { read_links(bk).1 };
            unsafe { write_links(bk, fd, bk_bk) };
        } else {
            self.lists[list_index(fl, sl)] = fd;
        }
        if fd != NULL {
            // SAFETY: `fd` is a free block on the same list; preserve
            // its own forward pointer while rerouting its back pointer.
            let fd_fd = unsafe { read_links(fd).0 };
            unsafe { write_links(fd, fd_fd, bk) };
        }
        if self.lists[list_index(fl, sl)] == NULL {
            self.clear_class_bits(fl, sl);
        }
    }

    /// Inserts the free block at `addr` of `size` bytes, linking it and
    /// updating the following block's boundary tags.
    ///
    /// # Safety
    ///
    /// `addr` must be the start of a free block of this heap, `size` its
    /// size (a multiple of [`ALIGNMENT`], at least [`MINSIZE`]), and
    /// `prev_free` the value of its own `FLAG_PREV_FREE` bit. The span
    /// must not overlap any other free block.
    unsafe fn insert_block(&mut self, addr: usize, size: usize, prev_free: usize) {
        debug_assert!(size >= MINSIZE && size.is_multiple_of(ALIGNMENT));
        debug_assert!(prev_free == 0 || prev_free == FLAG_PREV_FREE);
        // SAFETY: `addr` is a free block start; write the header, then
        // link it. The next block's tag is updated last.
        unsafe {
            write_size_flags(addr, size | FLAG_FREE | prev_free);
            self.push_list(addr, size);
            let next = addr + size;
            write_prev_size(next, size);
            let nsf = read_size_flags(next);
            write_size_flags(next, nsf | FLAG_PREV_FREE);
        }
    }

    /// Clears the summary bits of class `(fl, sl)` when it empties.
    #[inline]
    fn clear_class_bits(&mut self, fl: usize, sl: usize) {
        self.sl_bitmaps[fl] &= !(1u16 << sl);
        if self.sl_bitmaps[fl] == 0 {
            self.fl_bitmap[fl / 64] &= !(1u64 << (fl % 64));
        }
    }
}

impl Default for Tlsf {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Tlsf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tlsf")
            .field("stats", &self.stats())
            .finish()
    }
}

/// A [`Tlsf`] heap guarded by a `spin::Mutex`, implementing
/// [`GlobalAlloc`](core::alloc::GlobalAlloc).
///
/// Declare it as the process heap:
///
/// ```no_run
/// use codevar_tlsf_alloc::LockedTlsf;
///
/// #[global_allocator]
/// static HEAP: LockedTlsf = LockedTlsf::new();
/// ```
///
/// then add memory regions at boot:
///
/// ```ignore
/// // SAFETY: the region is valid, writable and exclusively ours.
/// unsafe { HEAP.lock().add_region(ptr, len) }?;
/// ```
///
/// # OOM behaviour
///
/// `alloc`, `alloc_zeroed` and `realloc` **abort the process** (through
/// [`handle_alloc_error`](alloc::alloc::handle_alloc_error)) when the heap
/// cannot satisfy a request, as required for the Codevar runtime; they
/// never return null.
pub struct LockedTlsf(Mutex<Tlsf>);

impl LockedTlsf {
    /// Creates an empty locked heap in constant time.
    pub const fn new() -> Self {
        Self(Mutex::new(Tlsf::new()))
    }
}

impl Default for LockedTlsf {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for LockedTlsf {
    type Target = Mutex<Tlsf>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Debug for LockedTlsf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LockedTlsf")
            .finish_non_exhaustive()
    }
}

unsafe impl GlobalAlloc for LockedTlsf {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match self
            .0
            .lock()
            .allocate(layout.size(), layout.align())
        {
            Some(ptr) => ptr.as_ptr(),
            // Aborts: the OOM policy of this heap.
            None => handle_alloc_error(layout),
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        match NonNull::new(ptr) {
            // SAFETY: the `GlobalAlloc` contract guarantees `ptr` is a
            // live allocation of this heap.
            Some(ptr) => unsafe { self.0.lock().deallocate(ptr) },
            // The contract forbids null; stay panic-free in release.
            None => debug_assert!(!ptr.is_null(), "dealloc called with null pointer"),
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = match self
            .0
            .lock()
            .allocate(layout.size(), layout.align())
        {
            Some(ptr) => ptr,
            None => handle_alloc_error(layout),
        };
        // SAFETY: the block is exclusively ours and `layout.size()` bytes
        // of it are allocated and writable; the SIMD fill honours that.
        unsafe { simd::fill_bytes(ptr.as_ptr(), layout.size(), 0) };
        ptr.as_ptr()
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let mut heap = self.0.lock();
        let addr = ptr as usize - HEADER_SIZE;
        // SAFETY: `ptr` is a live allocation of this heap (GlobalAlloc
        // contract), so `addr` is one of its block starts.
        let old_block = unsafe { read_size_flags(addr) } & SIZE_MASK;
        // SAFETY: same as above.
        if let Some(new_block) = unsafe { heap.resize_in_place(addr, new_size, layout.align()) } {
            heap.used_bytes = heap.used_bytes - old_block + new_block;
            return ptr;
        }
        // Fallback: allocate-copy-dealloc, still under the same lock.
        let new_layout = match Layout::from_size_align(new_size, layout.align()) {
            Ok(layout) => layout,
            Err(_) => handle_alloc_error(layout),
        };
        let new_ptr = match heap.allocate(new_layout.size(), new_layout.align()) {
            Some(ptr) => ptr.as_ptr(),
            None => handle_alloc_error(new_layout),
        };
        let old_usable = old_block - HEADER_SIZE;
        let copy = if old_usable < new_size {
            old_usable
        } else {
            new_size
        };
        // SAFETY: both blocks are valid and distinct allocations of this
        // heap, so the regions cannot overlap.
        unsafe { ptr::copy_nonoverlapping(ptr, new_ptr, copy) };
        // SAFETY: `ptr` is the live allocation we just copied from.
        unsafe { heap.deallocate(NonNull::new_unchecked(ptr)) };
        new_ptr
    }
}

/// Anonymous mapping layer for the growable heap (Unix only).
#[cfg(unix)]
mod os {
    use core::ptr;
    use core::ptr::NonNull;

    /// Queries the system page size, falling back to 4 KiB on error.
    pub(crate) fn page_size() -> usize {
        // SAFETY: `sysconf` with `_SC_PAGESIZE` only reads kernel state and
        // never touches memory through our allocator.
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if size > 0 { size as usize } else { 4096 }
    }

    /// Maps `len` anonymous, writable bytes of virtual memory.
    ///
    /// The pointer is page-aligned (so it always satisfies [`super::ALIGNMENT`]).
    /// `len` must be a non-zero multiple of the system page size.
    pub(crate) fn map_pages(len: usize) -> Option<NonNull<u8>> {
        debug_assert!(len != 0 && len.is_multiple_of(page_size()));
        // SAFETY: a null address lets the kernel choose where to map; the
        // anonymous private mapping needs no fd or offset, and the region
        // is never inherited by `exec` children.
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            None
        } else {
            NonNull::new(ptr.cast::<u8>())
        }
    }

    /// Unmaps the `len` bytes previously obtained with [`map_pages`].
    ///
    /// # Safety
    ///
    /// `[ptr, ptr + len)` must be an anonymous mapping of exactly `len`
    /// bytes obtained from [`map_pages`], and no access to it may happen
    /// after this call.
    pub(crate) unsafe fn unmap_pages(ptr: *mut u8, len: usize) {
        // SAFETY: the caller guarantees the range is a live mapping owned
        // by the growable heap; failures only leak, never corrupt.
        let _ = unsafe { libc::munmap(ptr.cast(), len) };
    }
}

/// Number of OS-backed regions a [`GrowableTlsf`] can track for unmapping.
#[cfg(unix)]
const MAX_MAPPED_REGIONS: usize = 0x100;

/// Size (in bytes) of the first chunk mapped on demand.
#[cfg(unix)]
const INITIAL_CHUNK: usize = 0x2000;

/// Upper bound (in bytes) on the geometric growth of a single chunk.
#[cfg(unix)]
const MAX_CHUNK: usize = 0x8000;

/// A chunk of virtual memory owned (and later unmapped) by a [`GrowableTlsf`].
#[cfg(unix)]
#[derive(Clone, Copy)]
struct MappedRegion {
    /// Base address returned by the OS mapping layer.
    ptr: *mut u8,
    /// Exact length, in bytes, of the mapping.
    len: usize,
}

/// A [`Tlsf`] heap that grows itself by mapping virtual pages (Unix only).
///
/// Unlike [`Tlsf`], which only manages memory explicitly contributed with
/// [`Tlsf::add_region`], a `GrowableTlsf` obtains its backing memory from
/// the operating system on demand: when no free block can satisfy a
/// request, it maps a fresh anonymous chunk of virtual pages and hands it
/// to the inner [`Tlsf`]. The mapping layer uses `mmap`/`munmap` directly
/// — never `malloc` — so the heap is safe to install as Rust's global
/// allocator without recursion.
///
/// Growth is geometric: the first chunk is [`INITIAL_CHUNK`] bytes, each
/// subsequent chunk doubles up to [`MAX_CHUNK`], and a chunk is always at
/// least large enough for the request that triggered it. Regions
/// contributed manually with [`GrowableTlsf::add_region`] are *not*
/// unmapped when the heap is dropped; only OS-backed chunks are.
///
/// Dropping the heap unmaps every OS-backed chunk, invalidating all live
/// allocations — drop only when nothing allocated from it is still in use
/// (a `static` heap is never dropped, so this only matters in tests).
#[cfg(unix)]
pub struct GrowableTlsf {
    /// Inner two-level segregated fit heap over all contributed regions.
    tlsf: Tlsf,
    /// OS-backed chunks, tracked so [`Drop`] can unmap them.
    regions: [MappedRegion; MAX_MAPPED_REGIONS],
    /// Number of valid entries in `regions`.
    region_count: usize,
    /// Size (in bytes) of the next chunk to map; doubles each growth.
    next_chunk: usize,
    /// Cached system page size (`0` until first queried).
    page_size: usize,
}

#[cfg(unix)]
impl GrowableTlsf {
    /// Creates an empty growable heap in constant time.
    ///
    /// No memory is mapped until the first allocation that cannot be
    /// served from the (initially empty) free lists.
    pub const fn new() -> Self {
        Self {
            tlsf: Tlsf::new(),
            regions: [MappedRegion {
                ptr: ptr::null_mut(),
                len: 0,
            }; MAX_MAPPED_REGIONS],
            region_count: 0,
            next_chunk: INITIAL_CHUNK,
            page_size: 0,
        }
    }

    /// Returns a snapshot of the heap usage (all regions, OS-backed or not).
    #[inline]
    pub fn stats(&self) -> Stats {
        self.tlsf.stats()
    }

    /// Returns the total number of bytes currently mapped by the OS
    /// (excluding regions added manually with [`GrowableTlsf::add_region`]).
    pub fn mapped_bytes(&self) -> usize {
        self.regions[..self.region_count]
            .iter()
            .map(|region| region.len)
            .sum()
    }

    /// Returns the number of OS-backed chunks currently mapped.
    #[inline]
    pub fn region_count(&self) -> usize {
        self.region_count
    }

    /// Adds caller-provided memory to the heap, exactly like
    /// [`Tlsf::add_region`].
    ///
    /// Such regions are never unmapped when the heap is dropped — only
    /// chunks the heap mapped itself are.
    ///
    /// # Safety
    ///
    /// Same contract as [`Tlsf::add_region`].
    pub unsafe fn add_region(&mut self, start: *mut u8, size: usize) -> Result<(), AddRegionError> {
        // SAFETY: the caller upholds `Tlsf::add_region`'s contract, which
        // is exactly what this delegation requires.
        unsafe { self.tlsf.add_region(start, size) }
    }

    /// Allocates a block for `size` bytes with alignment `align`,
    /// mapping a fresh chunk of virtual pages when the free lists cannot
    /// satisfy the request.
    ///
    /// Returns `None` only on an invalid `align`, on arithmetic overflow,
    /// on OS mapping failure, or when [`MAX_MAPPED_REGIONS`] chunks are
    /// already tracked (the region table is exhausted).
    #[must_use]
    pub fn allocate(&mut self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if align == 0 || !align.is_power_of_two() {
            return None;
        }
        if let Some(ptr) = self.tlsf.allocate(size, align) {
            return Some(ptr);
        }
        // No free block fits: size a new OS chunk that can, then retry.
        let min_free = Self::min_free_block(size, align)?;
        self.grow(min_free)?;
        self.tlsf.allocate(size, align)
    }

    /// Frees the block previously returned by [`GrowableTlsf::allocate`].
    ///
    /// # Safety
    ///
    /// Same contract as [`Tlsf::deallocate`].
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>) {
        // SAFETY: the caller upholds `Tlsf::deallocate`'s contract.
        unsafe { self.tlsf.deallocate(ptr) }
    }

    /// Minimum free-block size that can satisfy a request of `size` bytes
    /// with alignment `align` (mirrors the sizing rules of
    /// [`Tlsf::allocate`], including the large-alignment carve path).
    fn min_free_block(size: usize, align: usize) -> Option<usize> {
        let inner = Tlsf::block_size_for(size)?;
        if align <= ALIGNMENT {
            Some(inner)
        } else {
            // `Tlsf::carve_aligned` pops a block of at least
            // `inner + align + MINSIZE` bytes.
            inner.checked_add(align)?.checked_add(MINSIZE)
        }
    }

    /// Maps a fresh chunk of virtual memory large enough to hold a free
    /// block of `min_free` bytes and hands it to the inner [`Tlsf`].
    fn grow(&mut self, min_free: usize) -> Option<()> {
        if self.region_count >= MAX_MAPPED_REGIONS {
            return None;
        }
        let page = {
            if self.page_size == 0 {
                let size = os::page_size();
                self.page_size = size;
                size
            } else {
                self.page_size
            }
        };
        // The free block a fresh chunk contributes must not merely hold
        // `min_free` bytes: TLSF's first-fit search rounds the request up
        // to a size class, so a block smaller than that class's lower
        // bound is skipped even though it could hold the allocation. Size
        // the chunk to the class lower bound instead, plus the sentinel
        // header and a minimum block of slack, at least `next_chunk`
        // (geometric growth), and page-aligned.
        let (fl, sl) = mapping_search(min_free);
        let fit = class_lower_bound(fl, sl)
            .unwrap_or(usize::MAX)
            .max(min_free);
        let want = fit
            .saturating_add(HEADER_SIZE)
            .saturating_add(MINSIZE)
            .max(self.next_chunk);
        let want = want.checked_add(page - 1)? & !(page - 1);
        let base = os::map_pages(want)?;
        // SAFETY: the mapping is valid, writable, exclusively ours, and
        // `want` bytes long — exactly `Tlsf::add_region`'s contract.
        unsafe {
            if self.tlsf.add_region(base.as_ptr(), want).is_err() {
                os::unmap_pages(base.as_ptr(), want);
                return None;
            }
        }
        self.regions[self.region_count] = MappedRegion {
            ptr: base.as_ptr(),
            len: want,
        };
        self.region_count += 1;
        // Double the next chunk, capped at `MAX_CHUNK` but never below a
        // page (in case the page size exceeds the cap).
        let doubled = self
            .next_chunk
            .saturating_mul(2)
            .max(self.next_chunk);
        self.next_chunk = doubled.min(MAX_CHUNK.max(page));
        Some(())
    }
}

#[cfg(unix)]
impl Default for GrowableTlsf {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: the raw pointers in `regions` refer to anonymous mappings owned
// exclusively by this heap; moving the heap moves only the bookkeeping,
// never the mappings themselves. All access to the mappings is mediated
// by `&mut self` (or, in [`LockedGrowableTlsf`], by its mutex), so the
// usual aliasing rules are upheld across threads.
#[cfg(unix)]
unsafe impl Send for GrowableTlsf {}

#[cfg(unix)]
impl fmt::Debug for GrowableTlsf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GrowableTlsf")
            .field("stats", &self.stats())
            .field("mapped_bytes", &self.mapped_bytes())
            .finish()
    }
}

#[cfg(unix)]
impl Drop for GrowableTlsf {
    fn drop(&mut self) {
        for region in &self.regions[..self.region_count] {
            // SAFETY: every tracked region was mapped by `os::map_pages`,
            // is page-aligned and of exactly `len` bytes, and `Drop` runs
            // only when the heap itself is destroyed (a `static` heap is
            // never dropped), so no allocation can outlive the mapping.
            unsafe { os::unmap_pages(region.ptr, region.len) };
        }
    }
}

/// A [`GrowableTlsf`] heap guarded by a `spin::Mutex`, implementing
/// [`GlobalAlloc`](core::alloc::GlobalAlloc).
///
/// A drop-in replacement for [`LockedTlsf`] that needs no manual region
/// setup: declare it as the process heap and allocations are served
/// immediately, growing the heap by mapping virtual pages on demand
/// (Unix only — the mapping layer needs `mmap` from libc):
///
/// ```no_run
/// use codevar_tlsf_alloc::LockedGrowableTlsf;
///
/// #[global_allocator]
/// static HEAP: LockedGrowableTlsf = LockedGrowableTlsf::new();
///
/// # fn boot() {
/// // Grows past any fixed size: the heap maps fresh pages as needed.
/// let v = vec![1u8; 1 << 20];
/// assert_eq!(v.len(), 1 << 20);
/// # }
/// ```
///
/// # OOM behaviour
///
/// Same as [`LockedTlsf`]: `alloc`, `alloc_zeroed` and `realloc`
/// **abort the process** (through
/// [`handle_alloc_error`](alloc::alloc::handle_alloc_error)) when the heap
/// cannot satisfy a request — an OS mapping failure or an exhausted
/// [`MAX_MAPPED_REGIONS`] table included — and never return null.
#[cfg(unix)]
pub struct LockedGrowableTlsf(Mutex<GrowableTlsf>);

#[cfg(unix)]
impl LockedGrowableTlsf {
    /// Creates an empty locked growable heap in constant time.
    pub const fn new() -> Self {
        Self(Mutex::new(GrowableTlsf::new()))
    }
}

#[cfg(unix)]
impl Default for LockedGrowableTlsf {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(unix)]
impl Deref for LockedGrowableTlsf {
    type Target = Mutex<GrowableTlsf>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(unix)]
impl fmt::Debug for LockedGrowableTlsf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LockedGrowableTlsf")
            .finish_non_exhaustive()
    }
}

#[cfg(unix)]
unsafe impl GlobalAlloc for LockedGrowableTlsf {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match self
            .0
            .lock()
            .allocate(layout.size(), layout.align())
        {
            Some(ptr) => ptr.as_ptr(),
            // Aborts: the OOM policy of this heap.
            None => handle_alloc_error(layout),
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        match NonNull::new(ptr) {
            // SAFETY: the `GlobalAlloc` contract guarantees `ptr` is a
            // live allocation of this heap.
            Some(ptr) => unsafe { self.0.lock().deallocate(ptr) },
            // The contract forbids null; stay panic-free in release.
            None => debug_assert!(!ptr.is_null(), "dealloc called with null pointer"),
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = match self
            .0
            .lock()
            .allocate(layout.size(), layout.align())
        {
            Some(ptr) => ptr,
            None => handle_alloc_error(layout),
        };
        // SAFETY: the block is exclusively ours and `layout.size()` bytes
        // of it are allocated and writable; the SIMD fill honours that.
        unsafe { simd::fill_bytes(ptr.as_ptr(), layout.size(), 0) };
        ptr.as_ptr()
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let mut heap = self.0.lock();
        let addr = ptr as usize - HEADER_SIZE;
        // SAFETY: `ptr` is a live allocation of this heap (GlobalAlloc
        // contract), so `addr` is one of its block starts.
        let old_block = unsafe { read_size_flags(addr) } & SIZE_MASK;
        // SAFETY: same as above.
        if let Some(new_block) = unsafe {
            heap.tlsf
                .resize_in_place(addr, new_size, layout.align())
        } {
            heap.tlsf.used_bytes = heap.tlsf.used_bytes - old_block + new_block;
            return ptr;
        }
        // Fallback: allocate-copy-dealloc, still under the same lock.
        // `heap.allocate` grows the heap by mapping pages if needed.
        let new_layout = match Layout::from_size_align(new_size, layout.align()) {
            Ok(layout) => layout,
            Err(_) => handle_alloc_error(layout),
        };
        let new_ptr = match heap.allocate(new_layout.size(), new_layout.align()) {
            Some(ptr) => ptr.as_ptr(),
            None => handle_alloc_error(new_layout),
        };
        let old_usable = old_block - HEADER_SIZE;
        let copy = if old_usable < new_size {
            old_usable
        } else {
            new_size
        };
        // SAFETY: both blocks are valid and distinct allocations of this
        // heap, so the regions cannot overlap.
        unsafe { ptr::copy_nonoverlapping(ptr, new_ptr, copy) };
        // SAFETY: `ptr` is the live allocation we just copied from.
        unsafe { heap.deallocate(NonNull::new_unchecked(ptr)) };
        new_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A heap armed over a keep-alive backing buffer.
    ///
    /// The buffer is returned alongside the heap so its lifetime covers every
    /// allocation made from that heap.
    fn heap_with(bytes: usize) -> (Tlsf, Vec<u8>) {
        let mut region = vec![0u8; bytes];
        let mut heap = Tlsf::new();
        // SAFETY: `region` is live, writable and exclusively owned by the
        // returned tuple (the heap borrows it implicitly via contract).
        unsafe { heap.add_region(region.as_mut_ptr(), region.len()) }
            .expect("test region must fit a block and a sentinel");
        (heap, region)
    }

    /// Deterministic xorshift64 PRNG.
    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// [`Tlsf::block_size_for`] exposed for test assertions.
    fn block_size_for_test(size: usize) -> usize {
        Tlsf::block_size_for(size).expect("size fits")
    }

    #[test]
    fn mapping_covers_size_ranges_contiguously() {
        // Linear (small) classes.
        assert_eq!(mapping(MINSIZE), (0, MINSIZE / ALIGNMENT));
        assert_eq!(mapping(SMALL_SIZE_LIMIT - ALIGNMENT), (0, SL_COUNT - 1));
        // First large class starts exactly where the small ones end.
        assert_eq!(mapping(SMALL_SIZE_LIMIT), (5, 0));
        // Granularity doubles per first-level class; 2^k maps to (k-3, 0).
        // Below SMALL_SIZE_LIMIT (2^8) the linear small classes apply instead.
        for k in (usize::BITS as usize - 3)..(usize::BITS as usize) {
            let size = 1usize << k;
            assert_eq!(mapping(size), (k - SL_SHIFT + 1, 0), "size={size}");
        }
        // mapping_search never maps below the request's own class.
        for size in (MINSIZE..4096).step_by(ALIGNMENT) {
            let (fl_a, sl_a) = mapping_search(size);
            let (fl_b, sl_b) = mapping(size);
            assert!(
                (fl_a, sl_a) >= (fl_b, sl_b),
                "search class below block class for size={size}"
            );
        }
    }

    #[test]
    fn allocate_deallocate_roundtrip() {
        let (mut heap, _keep) = heap_with(64 * 1024);
        let ptr = heap.allocate(100, ALIGNMENT).expect("alloc");
        assert_eq!(ptr.as_ptr() as usize % ALIGNMENT, 0);
        assert!(heap.stats().used_bytes >= 100 + HEADER_SIZE);
        // SAFETY: 100 fresh bytes are writable.
        unsafe { core::ptr::write_bytes(ptr.as_ptr(), 0x5A, 100) };
        // SAFETY: `ptr` came from this heap and is still live.
        unsafe { heap.deallocate(ptr) };
        assert_eq!(heap.stats().used_bytes, 0);
        assert_eq!(heap.free_block_count(), 1);
    }

    #[test]
    fn zero_sized_allocation_succeeds() {
        let (mut heap, _keep) = heap_with(4096);
        let ptr = heap
            .allocate(0, ALIGNMENT)
            .expect("zero-size alloc");
        assert_eq!(ptr.as_ptr() as usize % ALIGNMENT, 0);
        // SAFETY: `ptr` came from this heap and is still live.
        unsafe { heap.deallocate(ptr) };
    }

    #[test]
    fn honors_arbitrary_power_of_two_alignments() {
        let (mut heap, _keep) = heap_with(128 * 1024);
        for &align in &[2 * ALIGNMENT, 32usize, 64, 128, 256, 512, 4096] {
            for &size in &[1usize, 17, 100, 1000, 8000] {
                let ptr = heap.allocate(size, align).expect("alloc");
                assert_eq!(ptr.as_ptr() as usize % align, 0, "size={size} align={align}");
                // SAFETY: `size` fresh bytes are writable.
                unsafe { core::ptr::write_bytes(ptr.as_ptr(), 0xA5, size) };
                // SAFETY: `ptr` came from this heap and is still live.
                unsafe { heap.deallocate(ptr) };
            }
        }
        assert_eq!(heap.stats().used_bytes, 0);
    }

    #[test]
    fn invalid_align_is_rejected() {
        let (mut heap, _keep) = heap_with(4096);
        assert!(heap.allocate(16, 0).is_none());
        assert!(heap.allocate(16, 3).is_none());
        assert!(heap.allocate(16, 24).is_none());
    }

    #[test]
    fn oom_returns_none_and_heap_stays_usable() {
        let (mut heap, _keep) = heap_with(4096);
        let total = heap.stats().total_bytes;
        assert!(heap.allocate(total, ALIGNMENT).is_none());
        // The failed request must not have corrupted anything.
        let ptr = heap
            .allocate(64, ALIGNMENT)
            .expect("small alloc after OOM");
        // SAFETY: `ptr` came from this heap and is still live.
        unsafe { heap.deallocate(ptr) };
    }

    #[test]
    fn tiny_region_is_rejected() {
        let mut heap = Tlsf::new();
        let mut buf = [0u8; 8];
        // SAFETY: the buffer is live and writable.
        let result = unsafe { heap.add_region(buf.as_mut_ptr(), buf.len()) };
        assert_eq!(result, Err(AddRegionError::TooSmall));
        // A non-null dangling base plus `usize::MAX` overflows the address
        // space, so `add_region` rejects it before touching any memory.
        let ptr = core::ptr::dangling::<u8>() as *mut u8;
        // SAFETY: the overflow check fails before the region is accessed.
        let result = unsafe { heap.add_region(ptr, usize::MAX) };
        assert_eq!(result, Err(AddRegionError::Overflow));
    }

    #[test]
    fn multiple_regions_are_independent_and_mergeable() {
        let mut heap = Tlsf::new();
        let mut first = vec![0u8; 8 * 1024];
        let mut second = vec![0u8; 8 * 1024];
        // SAFETY: both buffers are live, writable and disjoint.
        unsafe {
            heap.add_region(first.as_mut_ptr(), first.len())
                .expect("first");
            heap.add_region(second.as_mut_ptr(), second.len())
                .expect("second");
        }
        assert_eq!(heap.stats().total_bytes, heap.stats().free_bytes());
        let a = heap.allocate(3000, ALIGNMENT).expect("a");
        let b = heap.allocate(3000, ALIGNMENT).expect("b");
        // SAFETY: both are live allocations of this heap.
        unsafe {
            heap.deallocate(a);
            heap.deallocate(b);
        }
        // Two regions cannot merge into one block, but each must be whole.
        assert_eq!(heap.stats().used_bytes, 0);
        assert!(heap.free_block_count() >= 2);
        // Two regions cannot merge into one block: an allocation larger than
        // a single region must fail even though the heap as a whole is big
        // enough, while a quarter-heap request (fits in either region) works.
        let quarter = heap.stats().total_bytes / 4;
        let big = heap
            .allocate(quarter, ALIGNMENT)
            .expect("quarter-heap alloc fits in one region");
        // SAFETY: `big` is live.
        unsafe { heap.deallocate(big) };
        let spanning = heap.stats().total_bytes - MINSIZE;
        assert!(
            heap.allocate(spanning, ALIGNMENT).is_none(),
            "one allocation must never span two regions"
        );
    }

    #[test]
    fn coalescing_recovers_the_whole_region() {
        let (mut heap, _keep) = heap_with(64 * 1024);
        let a = heap.allocate(1000, ALIGNMENT).expect("a");
        let b = heap.allocate(1000, ALIGNMENT).expect("b");
        let c = heap.allocate(1000, ALIGNMENT).expect("c");
        // Free out of order to exercise both coalescing directions.
        // SAFETY: all three are live allocations of this heap.
        unsafe {
            heap.deallocate(b);
            heap.deallocate(a);
            heap.deallocate(c);
        }
        assert_eq!(heap.stats().used_bytes, 0);
        assert_eq!(
            heap.free_block_count(),
            1,
            "three adjacent frees must merge into one block"
        );
        // A quarter-heap request can only succeed if the merge really happened.
        let big = heap
            .allocate(heap.stats().total_bytes / 4, ALIGNMENT)
            .expect("large alloc after full coalesce");
        // SAFETY: `big` is live.
        unsafe { heap.deallocate(big) };
    }

    #[test]
    fn realloc_in_place_shrink_and_grow() {
        let (mut heap, _keep) = heap_with(64 * 1024);
        let ptr = heap.allocate(4096, ALIGNMENT).expect("alloc");
        let addr = ptr.as_ptr() as usize - HEADER_SIZE;
        // SAFETY: `ptr` is a live allocation of this heap.
        unsafe {
            // Shrink: the tail must be split back into the free lists.
            assert_eq!(
                heap.resize_in_place(addr, 64, ALIGNMENT),
                Some(block_size_for_test(64))
            );
            // Grow back by absorbing the free tail.
            assert_eq!(
                heap.resize_in_place(addr, 4096, ALIGNMENT),
                Some(block_size_for_test(4096))
            );
            heap.deallocate(ptr);
        }
        assert_eq!(heap.stats().used_bytes, 0);
        assert_eq!(heap.free_block_count(), 1);
    }

    #[test]
    fn global_alloc_trait_roundtrip_and_zeroed() {
        use core::alloc::GlobalAlloc;

        let locked = LockedTlsf::new();
        let mut region = vec![0u8; 64 * 1024];
        // SAFETY: the buffer is live, writable and exclusively ours.
        unsafe {
            locked
                .lock()
                .add_region(region.as_mut_ptr(), region.len())
                .expect("region");
        }
        let layout = Layout::from_size_align(128, 64).expect("layout");
        // SAFETY: `layout` is valid and the heap is armed.
        let ptr = unsafe { locked.alloc(layout) };
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 64, 0);
        // SAFETY: writing is fine, then free through the same trait.
        unsafe {
            core::ptr::write_bytes(ptr, 0xEE, 128);
            locked.dealloc(ptr, layout);
        }
        // SAFETY: the zeroed path must return all-zero memory.
        let zptr = unsafe { locked.alloc_zeroed(layout) };
        assert!(!zptr.is_null());
        for i in 0..128 {
            // SAFETY: `zptr` is live for `layout.size()` bytes.
            assert_eq!(unsafe { *zptr.add(i) }, 0, "byte {i} not zeroed");
        }
        // SAFETY: free the zeroed allocation.
        unsafe { locked.dealloc(zptr, layout) };

        // Realloc through the trait: grow and shrink, contents preserved.
        let big = Layout::from_size_align(1024, ALIGNMENT).expect("layout");
        // SAFETY: `big` is a valid layout.
        let bptr = unsafe { locked.alloc(big) };
        // SAFETY: `bptr` is live for 1024 bytes.
        unsafe { core::ptr::write_bytes(bptr, 0x3C, 1024) };
        let grown = Layout::from_size_align(2048, ALIGNMENT).expect("layout");
        // SAFETY: all inputs honour the GlobalAlloc contract.
        let gptr = unsafe { locked.realloc(bptr, big, grown.size()) };
        // SAFETY: the first 1024 bytes must survive the grow.
        unsafe {
            for i in 0..1024 {
                assert_eq!(*gptr.add(i), 0x3C, "realloc lost byte {i}");
            }
            locked.dealloc(gptr, grown);
        }
        drop(region);
    }

    #[test]
    fn stress_randomized_alloc_free_stays_consistent() {
        const REGION: usize = 256 * 1024;
        let (mut heap, _keep) = heap_with(REGION);
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut live: Vec<(NonNull<u8>, usize, u8)> = Vec::new();
        let aligns = [ALIGNMENT, 2 * ALIGNMENT, 64usize, 256];

        for step in 0..4000 {
            let want_alloc = live.len() < 48 || (xorshift64(&mut rng) & 1) == 0;
            if want_alloc {
                let size = (xorshift64(&mut rng) as usize % 4096) + 1;
                let align = aligns[(xorshift64(&mut rng) as usize) % aligns.len()];
                if let Some(ptr) = heap.allocate(size, align) {
                    assert_eq!(
                        ptr.as_ptr() as usize % align,
                        0,
                        "step={step} size={size} align={align}"
                    );
                    let tag = (xorshift64(&mut rng) as u8) | 1;
                    // SAFETY: `size` fresh bytes are writable.
                    unsafe { core::ptr::write_bytes(ptr.as_ptr(), tag, size) };
                    live.push((ptr, size, tag));
                }
            } else {
                let index = (xorshift64(&mut rng) as usize) % live.len();
                let (ptr, size, tag) = live.swap_remove(index);
                // SAFETY: every byte of the allocation must still carry its
                // tag; a mismatch means a neighbouring block wrote into it
                // (an allocator bug).
                let mut off = 0;
                while off < size {
                    // SAFETY: `ptr` is live and `off < size`.
                    let byte = unsafe { *ptr.as_ptr().add(off) };
                    assert_eq!(byte, tag, "corruption at offset {off}, step={step}");
                    off += 1;
                }
                // SAFETY: `ptr` is a live allocation of this heap.
                unsafe { heap.deallocate(ptr) };
            }
            assert!(
                heap.stats().used_bytes <= heap.stats().total_bytes,
                "used exceeds total at step {step}"
            );
        }

        // Drain everything; the heap must fully coalesce back down.
        for (ptr, _, _) in live {
            // SAFETY: every entry is a live allocation of this heap.
            unsafe { heap.deallocate(ptr) };
        }
        assert_eq!(heap.stats().used_bytes, 0);
        assert_eq!(
            heap.free_block_count(),
            1,
            "heap must coalesce into a single block after the stress test"
        );
        let big = heap
            .allocate(heap.stats().total_bytes / 4, ALIGNMENT)
            .expect("post-stress large alloc");
        // SAFETY: `big` is live.
        unsafe { heap.deallocate(big) };
    }

    #[test]
    fn stats_and_debug_reflect_state() {
        let (mut heap, _keep) = heap_with(16 * 1024);
        let stats = heap.stats();
        assert_eq!(stats.used_bytes, 0);
        assert_eq!(stats.free_bytes(), stats.total_bytes);
        let ptr = heap.allocate(512, ALIGNMENT).expect("alloc");
        assert!(heap.stats().used_bytes >= 512 + HEADER_SIZE);
        let text = format!("{heap:?}");
        assert!(text.contains("Tlsf"), "debug output: {text}");
        // SAFETY: `ptr` is live.
        unsafe { heap.deallocate(ptr) };
    }

    /// [`Tlsf::remove_from_list`] must preserve each neighbor's *own* link
    /// while rerouting it around the removed block.
    ///
    /// Preserving the wrong field (the neighbor's opposite link) re-links
    /// the list to the removed block itself, and a later pop resurrects it
    /// as a "free" block over live user data — heap corruption that showed
    /// up as a SIGSEGV in `codevar-oclc`.
    #[test]
    fn remove_from_list_preserves_neighbor_links() {
        let (mut heap, _keep) = heap_with(64 * 1024);
        let mut blocks = Vec::new();
        for _ in 0..7 {
            blocks.push(
                heap.allocate(16, ALIGNMENT)
                    .expect("32-byte block"),
            );
        }
        // Free every other block so live neighbours hold the three free
        // blocks apart (no coalescing) and they all land in one class.
        // SAFETY: all seven are live allocations of this heap.
        unsafe {
            heap.deallocate(blocks[1]);
            heap.deallocate(blocks[3]);
            heap.deallocate(blocks[5]);
        }
        let head = blocks[5].as_ptr() as usize - HEADER_SIZE;
        let middle = blocks[3].as_ptr() as usize - HEADER_SIZE;
        let tail = blocks[1].as_ptr() as usize - HEADER_SIZE;
        let index = list_index(0, 2);
        assert_eq!(heap.lists[index], head, "last freed block is the head");

        // Removing the middle block must reroute head/tail around it
        // without clobbering either neighbor's own back/forward link.
        // SAFETY: `middle` is a free block of this heap on that list.
        unsafe { heap.remove_from_list(middle, 32) };
        // SAFETY: `head` and `tail` are free blocks of this heap on that
        // list, so their link words are valid.
        assert_eq!(
            unsafe { read_links(head) },
            (tail, NULL),
            "head links after middle removal"
        );
        assert_eq!(
            unsafe { read_links(tail) },
            (NULL, head),
            "tail links after middle removal"
        );
        assert_eq!(heap.lists[index], head, "head unchanged by middle removal");

        // Removing the tail leaves the head as a singleton.
        // SAFETY: `tail` is a free block of this heap on that list.
        unsafe { heap.remove_from_list(tail, 32) };
        // SAFETY: `head` is a free block of this heap on that list, so its
        // link words are valid.
        assert_eq!(
            unsafe { read_links(head) },
            (NULL, NULL),
            "head links after tail removal"
        );
        assert_eq!(heap.lists[index], head);

        // Removing the last block empties the class.
        // SAFETY: `head` is a free block of this heap on that list.
        unsafe { heap.remove_from_list(head, 32) };
        assert_eq!(heap.lists[index], NULL, "class empty after last removal");
    }

    #[cfg(unix)]
    mod growable {
        use super::*;

        #[test]
        fn starts_empty_and_grows_on_demand() {
            let mut heap = GrowableTlsf::new();
            assert_eq!(heap.stats().total_bytes, 0, "nothing mapped at start");
            let ptr = heap
                .allocate(4096, ALIGNMENT)
                .expect("first allocation grows");
            assert!(heap.region_count() >= 1, "a chunk was mapped");
            assert!(heap.mapped_bytes() >= heap.stats().total_bytes);
            // SAFETY: 4096 fresh bytes are writable.
            unsafe { ptr.as_ptr().write_bytes(0xAB, 4096) };
            // SAFETY: live allocation of this heap.
            unsafe { heap.deallocate(ptr) };
        }

        #[test]
        fn allocation_larger_than_small_static_region_succeeds() {
            // 1 MiB far exceeds any fixed-size boot region the bins used to
            // install; the growable heap must satisfy it via mapping.
            let mut heap = GrowableTlsf::new();
            let ptr = heap.allocate(1 << 20, 16).expect("grow to fit");
            // SAFETY: the block is live and exclusively ours.
            unsafe { ptr.as_ptr().write_bytes(0xCD, 1 << 20) };
            // SAFETY: live allocation of this heap.
            unsafe { heap.deallocate(ptr) };
        }

        #[test]
        fn reuse_after_free_does_not_map_again() {
            let mut heap = GrowableTlsf::new();
            let first = heap
                .allocate(32 * 1024, ALIGNMENT)
                .expect("alloc");
            let mapped = heap.mapped_bytes();
            // SAFETY: live allocation of this heap.
            unsafe { heap.deallocate(first) };
            let second = heap
                .allocate(32 * 1024, ALIGNMENT)
                .expect("reuse");
            assert_eq!(
                heap.mapped_bytes(),
                mapped,
                "reuse of freed memory must not map a new chunk"
            );
            // SAFETY: live allocation of this heap.
            unsafe { heap.deallocate(second) };
        }

        #[test]
        fn live_set_spans_multiple_chunks() {
            let mut heap = GrowableTlsf::new();
            let mut blocks = Vec::new();
            for _ in 0..8 {
                // SAFETY: each allocation is live until the cleanup below.
                blocks.push(
                    heap.allocate(128 * 1024, ALIGNMENT)
                        .expect("grow"),
                );
            }
            assert!(
                heap.region_count() >= 2,
                "1 MiB live set must span several chunks"
            );
            let stats = heap.stats();
            assert!(stats.used_bytes >= 8 * 128 * 1024);
            assert!(heap.mapped_bytes() >= stats.total_bytes);
            for block in blocks {
                // SAFETY: every block is a live allocation of this heap.
                unsafe { heap.deallocate(block) };
            }
        }

        #[test]
        fn large_alignment_grows_enough_to_carve() {
            let mut heap = GrowableTlsf::new();
            let ptr = heap
                .allocate(64, 4096)
                .expect("carve-aligned allocation");
            assert_eq!(ptr.as_ptr() as usize % 4096, 0);
            // SAFETY: live allocation of this heap.
            unsafe { heap.deallocate(ptr) };
        }

        #[test]
        fn invalid_align_never_maps() {
            let mut heap = GrowableTlsf::new();
            assert!(heap.allocate(64, 0).is_none());
            assert!(heap.allocate(64, 3).is_none(), "non-power-of-two align");
            assert_eq!(heap.region_count(), 0, "no chunk mapped for bad align");
        }

        #[test]
        fn manual_regions_are_not_unmapped_on_drop() {
            let mut region = vec![0u8; 16 * 1024];
            {
                let mut heap = GrowableTlsf::new();
                // SAFETY: `region` outlives the heap block below.
                unsafe { heap.add_region(region.as_mut_ptr(), region.len()) }.expect("region fits");
                let ptr = heap.allocate(128, ALIGNMENT).expect("alloc");
                // SAFETY: live allocation of this heap.
                unsafe { heap.deallocate(ptr) };
                assert_eq!(heap.mapped_bytes(), 0, "manual region is not OS-backed");
            }
            // The manual region must still be valid memory after the heap's
            // Drop ran (drop only unmaps OS-backed chunks).
            region[0] = 1;
            assert_eq!(region[0], 1);
        }
    }
}
