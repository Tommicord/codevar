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
//! ```
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
            // SAFETY: `bk` is a free block on the same list.
            let bk_fd = unsafe { read_links(bk).0 };
            unsafe { write_links(bk, fd, bk_fd) };
        } else {
            self.lists[list_index(fl, sl)] = fd;
        }
        if fd != NULL {
            // SAFETY: `fd` is a free block on the same list.
            let fd_bk = unsafe { read_links(fd).1 };
            unsafe { write_links(fd, fd_bk, bk) };
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
/// ```
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

#[cfg(test)]
mod tests;
