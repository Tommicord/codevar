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

//! Unit tests for the TLSF heap. Only panicking assertions are allowed
//! here (see CODE_QUALITY.md: `.unwrap()`/`assert!` are test-only).

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
