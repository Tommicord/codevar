//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

use super::wredit_observer::{
    UserActionEvent, UserActionObserver, UserActionObserverRegistry, UserActionType,
};
use super::{Cursor, EventBridge, SharedGap, Writable, WritableAlignedPtr};
use crate::base::Allocator;
use crate::edit::wredit_history::History;
use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::sync::Arc;

/// `static NAME_LEN: usize = 255`
///
/// Defines the maximum length for a writable name (e.g., file name).
/// This constant limits the size of the name array to prevent excessive
/// memory usage and ensure consistent sizing across all writable instances.
static NAME_LEN: usize = 255;

/// `static MAX_CURSOR_COUNT: usize = 512`
///
/// Defines the maximum number of cursors that can be simultaneously
/// active in a writable data structure. This limit prevents excessive
/// memory consumption and ensures predictable performance when managing
/// multiple cursor positions in the same buffer.
static MAX_CURSOR_COUNT: usize = 512;

/// `static INIT_RAW_BUFFER_COUNT: usize = 8192`
///
/// Defines the initial capacity of the raw data buffer when a writable
/// is created without an explicit capacity specification. This value
/// provides a reasonable default size that balances memory usage with
/// the need to accommodate typical file sizes without frequent reallocations.
static INIT_RAW_BUFFER_COUNT: usize = 8192;

/// Text encoding constant: UTF-8 (8-bit Unicode Transformation Format)
///
/// Variable-length encoding that uses 1-4 bytes per character. The most
/// common encoding for web and modern applications, providing ASCII compatibility
/// and efficient storage for Latin scripts.
pub const ENCODING_UTF8: u16 = 0;

/// Text encoding constant: UTF-16 (16-bit Unicode Transformation Format)
///
/// Variable-length encoding that uses 2 or 4 bytes per character. Commonly used
/// in Windows APIs and Java strings. Supports all Unicode characters with
/// surrogate pairs for characters outside the Basic Multilingual Plane.
pub const ENCODING_UTF16: u16 = 1;

/// Text encoding constant: UTF-16LE (UTF-16 Little Endian)
///
/// Little-endian variant of UTF-16, where the least significant byte comes first.
/// The default UTF-16 encoding on Windows and x86/x64 systems.
pub const ENCODING_UTF16LE: u16 = 2;

/// Text encoding constant: UTF-16BE (UTF-16 Big Endian)
///
/// Big-endian variant of UTF-16, where the most significant byte comes first.
/// Commonly used on big-endian systems and in network protocols.
pub const ENCODING_UTF16BE: u16 = 3;

/// Text encoding constant: UTF-32 (32-bit Unicode Transformation Format)
///
/// Fixed-length encoding that uses exactly 4 bytes per character. Provides
/// simple random access to characters but uses more memory. Supports all
/// Unicode code points directly without surrogate pairs.
pub const ENCODING_UTF32: u16 = 4;

/// Text encoding constant: UTF-32LE (UTF-32 Little Endian)
///
/// Little-endian variant of UTF-32, where the least significant byte comes first.
/// The default UTF-32 encoding on Windows and x86/x64 systems.
pub const ENCODING_UTF32LE: u16 = 5;

/// Text encoding constant: UTF-32BE (UTF-32 Big Endian)
///
/// Big-endian variant of UTF-32, where the most significant byte comes first.
/// Commonly used on big-endian systems and in network protocols.
pub const ENCODING_UTF32BE: u16 = 6;

/// Text encoding constant: ISO-8859-1 (Latin-1)
///
/// Single-byte encoding that covers Western European languages. Compatible with
/// the first 256 Unicode code points. Does not support characters outside the
/// Latin-1 range.
pub const ENCODING_ISO_8859_1: u16 = 7;

/// Text encoding constant: ISO-8859-15 (Latin-9)
///
/// Single-byte encoding similar to ISO-8859-1 but with the Euro sign (€) and
/// other characters replacing less common symbols. Covers Western European
/// languages with improved currency support.
pub const ENCODING_ISO_8859_15: u16 = 8;

/// Text encoding constant: Windows-1252 (ANSI)
///
/// Single-byte encoding used by Windows in Western European locales. Similar to
/// ISO-8859-1 but with additional characters in the 0x80-0x9F range. The default
/// encoding for many legacy Windows applications.
pub const ENCODING_WINDO1252: u16 = 9;

/// Text encoding constant: ASCII (7-bit American Standard Code for Information Interchange)
///
/// 7-bit encoding that only supports English characters (0-127). A subset of UTF-8
/// and many other encodings. Cannot represent non-English characters or symbols.
pub const ENCODING_ASCII: u16 = 10;

/// Text custom encoding
pub const ENCODING_CUSTOM: u16 = 11;

/// `pub struct WritableGapPtr<T, const GAP_SIZE: usize>`
///
/// Gap pointer for writable data structures with circular buffer semantics.
///
/// This struct manages a gap buffer that enables efficient insertions and
/// deletions in writable data structures. The gap buffer uses circular buffer
/// semantics, where the gap pointer wraps around when it reaches the end of
/// the buffer. This design allows for O(1) insertions at the current cursor
/// position without needing to shift existing data.
///
/// The gap buffer works by maintaining a "gap" of unused space in the buffer.
/// When inserting data, it's written into the gap, and the gap pointer advances.
/// When the gap is full, it must be flushed to the main buffer, which may
/// trigger a shift operation to make room for the new data.
///
/// # Type Parameters
///
/// * `T` - The type of elements stored in the gap buffer (typically u32 for packed data)
/// * `GAP_SIZE` - The size of the gap buffer (must be a power of 2 for efficient masking)
///
/// # Fields
///
/// * `gap_ptr` - Current position within the gap buffer. This pointer increments
///   as data is written and wraps around when it reaches GAP_SIZE. The circular
///   nature is achieved using bitwise AND with (GAP_SIZE - 1) for indexing.
/// * `buff` - Optional pointer to the allocated gap buffer memory. This pointer
///   is null until `allocate()` is called, and is freed automatically when the
///   struct is dropped. The buffer stores packed values that combine type and
///   character information.

#[repr(C)]
pub struct WritableGapPtr<T, const GAP_SIZE: usize> {
    /// Current gap pointer position. This tracks where the next write operation
    /// will occur within the circular gap buffer. The pointer wraps around
    /// using modulo arithmetic (implemented via bitwise AND for power-of-2 sizes).
    gap_ptr: usize,
    /// Gap buffer array pointer. This points to the allocated memory that stores
    /// the gap buffer contents. It's wrapped in Option to handle the unallocated
    /// state, and NonNull provides null-safety guarantees. The buffer stores
    /// packed 32-bit values containing both type and character information.
    buff: Option<NonNull<T>>,
}

impl<T, const GAP_SIZE: usize> WritableGapPtr<T, GAP_SIZE> {
    /// Creates a new gap pointer

    pub fn new() -> Self {
        Self {
            gap_ptr: 0,
            buff: None,
        }
    }

    /// Allocates the gap buffer
    ///
    /// # Safety
    ///
    /// The caller must ensure the allocation succeeds

    pub unsafe fn allocate(&mut self) {
        let layout = Layout::array::<T>(GAP_SIZE).unwrap_or_else(|_| {
            // Fallback to a safe default layout if the requested one is invalid
            Layout::new::<T>()
        });
        let ptr = alloc::alloc(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        self.buff = NonNull::new(ptr as *mut T);
    }

    /// Frees the gap buffer
    ///
    /// # Safety
    ///
    /// The buffer must have been allocated

    pub unsafe fn free(&mut self) {
        if let Some(buff) = self.buff {
            let layout = Layout::array::<T>(GAP_SIZE).unwrap_or_else(|_| {
                // Fallback to a safe default layout if the requested one is invalid
                Layout::new::<T>()
            });
            alloc::dealloc(buff.as_ptr() as *mut u8, layout);
            self.buff = None;
        }
    }

    /// Returns the gap pointer position

    pub fn gap_ptr(&self) -> usize {
        self.gap_ptr
    }

    /// Sets the gap pointer position

    pub fn set_gap_ptr(&mut self, ptr: usize) {
        self.gap_ptr = ptr;
    }

    /// Returns the gap buffer pointer

    pub fn buff(&self) -> *const T {
        self.buff.map_or(std::ptr::null(), |b| b.as_ptr())
    }

    /// Returns the mutable gap buffer pointer

    pub fn buff_mut(&mut self) -> *mut T {
        self.buff.map_or(std::ptr::null_mut(), |b| b.as_ptr())
    }
}

impl<T, const GAP_SIZE: usize> Drop for WritableGapPtr<T, GAP_SIZE> {
    fn drop(&mut self) {
        unsafe {
            self.free();
        }
    }
}

/// `pub struct WritableGap<T, const GAP_SIZE: usize>`
///
/// Gap structure representing a contiguous gap region in the main buffer.
///
/// This struct defines a gap within the main writable buffer, delimited by
/// start and end pointers. The gap represents a region of unused space that
/// can be used for efficient insertions without shifting existing data. The
/// gap is associated with a circular gap buffer that temporarily stores
/// insertions until the gap is flushed.
///
/// The gap bounds (start and end) define the region in the main buffer where
/// the gap is located. When the gap buffer fills up, its contents are flushed
/// to the main buffer at the gap's location, and the gap may be moved to a
/// new position to accommodate further insertions.
///
/// # Type Parameters
///
/// * `T` - The type of elements in the main buffer (typically u32 for packed data)
/// * `GAP_SIZE` - The size of the associated circular gap buffer
///
/// # Fields
///
/// * `gap_start` - Pointer to the beginning of the gap region in the main buffer.
///   This marks where the gap starts and is used to calculate the gap's position
///   relative to the buffer start. It's a mutable pointer to allow gap movement.
/// * `gap_end` - Pointer to the end of the gap region in the main buffer. This
///   marks where the gap ends and the valid data resumes. The difference between
///   gap_end and gap_start determines the gap's current size.
/// * `gap_ptr` - The circular gap buffer that temporarily stores insertions before
///   they are flushed to the main buffer. This buffer uses circular semantics for
///   efficient write operations and is managed independently of the main buffer.
/// * `gap_siz` - The size of the gap buffer. This is a constant value determined
///   at compile time and must be a power of 2 for efficient circular indexing.
///   It defines the capacity of the temporary storage before a flush is required.
#[repr(C)]
pub struct WritableGap<T, const GAP_SIZE: usize> {
    /// Pointer to the start of the gap region in the main buffer. This marks
    /// the beginning of unused space where new data can be inserted without
    /// shifting existing elements. The pointer is mutable to allow the gap to
    /// be moved to different positions in the buffer as needed.
    gap_start: *mut T,
    /// Pointer to the end of the gap region in the main buffer. This marks
    /// where the gap ends and valid data resumes. The gap size is calculated
    /// as (gap_end - gap_start), and this boundary is used during gap flush
    /// operations to determine where to write the accumulated data.
    gap_end: *mut T,
    /// The circular gap buffer that temporarily stores insertions. This buffer
    /// uses circular semantics with wrap-around indexing, allowing O(1) writes
    /// at the current position. When this buffer fills, its contents are flushed
    /// to the main buffer at the gap's location.
    gap_ptr: WritableGapPtr<T, GAP_SIZE>,
    /// The size of the gap buffer. This is a compile-time constant that
    /// determines the capacity of the temporary storage. Larger values reduce
    /// the frequency of flush operations but increase memory usage. Must be a
    /// power of 2 for efficient circular indexing using bitwise operations.
    gap_size: usize,
}

impl<T, const GAP_SIZE: usize> WritableGap<T, GAP_SIZE> {
    /// Creates a new gap

    pub fn new() -> Self {
        Self {
            gap_start: std::ptr::null_mut(),
            gap_end: std::ptr::null_mut(),
            gap_ptr: WritableGapPtr::new(),
            gap_size: GAP_SIZE,
        }
    }

    /// Returns the gap start pointer

    pub fn gap_start(&self) -> *mut T {
        self.gap_start
    }

    /// Returns the gap end pointer

    pub fn gap_end(&self) -> *mut T {
        self.gap_end
    }

    /// Sets the gap bounds
    ///
    /// # Safety
    ///
    /// The pointers must be valid and within the allocated buffer

    pub fn set_gap_bounds(&mut self, start: *mut T, end: *mut T) {
        self.gap_start = start;
        self.gap_end = end;
    }

    /// Returns the gap pointer

    pub fn gap_ptr(&self) -> &WritableGapPtr<T, GAP_SIZE> {
        &self.gap_ptr
    }

    /// Returns the mutable gap pointer

    pub fn gap_ptr_mut(&mut self) -> &mut WritableGapPtr<T, GAP_SIZE> {
        &mut self.gap_ptr
    }

    /// Returns the gap size

    pub fn gap_size(&self) -> usize {
        self.gap_size
    }
}

impl<T, const GAP_SIZE: usize> Drop for WritableGap<T, GAP_SIZE> {
    fn drop(&mut self) {
        // Nullify pointers to prevent dangling pointer access during teardown
        self.gap_start = std::ptr::null_mut();
        self.gap_end = std::ptr::null_mut();
    }
}

/// `pub struct WritableGaps<T, const GAP_SIZE: usize>`
///
/// Gap manager for multi-cursor writable data structures.
///
/// This struct manages multiple gap buffers using a vector for indexed access
/// and an arena allocator for efficient memory management. In multi-cursor
/// scenarios, each cursor may have its own gap buffer, and cursors that are
/// close to each other may share a gap buffer to reduce memory overhead and
/// improve cache locality.
///
/// The arena allocator provides O(1) allocation with minimal memory fragmentation,
/// and all gaps are stored contiguously in the vector, which improves cache
/// performance when iterating over multiple gaps. The manager maintains the
/// gap count and provides indexed access to individual gaps.
///
/// # Type Parameters
///
/// * `T` - The type of elements in the gap buffers (typically u32 for packed data)
/// * `GAP_SIZE` - The size of each individual gap buffer
///
/// # Fields
///
/// * `gaps` - Vector of gap structures providing indexed access to individual gaps.
///   This vector stores all active gaps contiguously in memory, which improves
///   cache performance during multi-gap operations. The gaps can be accessed
///   directly by index for O(1) lookup time.
/// * `arena` - The arena allocator that manages memory for gap structures.
///   This allocator provides efficient allocation with minimal fragmentation.
///   While gaps are stored in the vector for indexed access, the arena can be
///   used for additional allocations if needed.
#[repr(C)]
pub struct WritableGaps<T, const GAP_SIZE: usize> {
    /// Vector of gap structures stored in the arena. This provides indexed
    /// access to individual gaps while maintaining cache locality through
    /// contiguous storage. The gaps are allocated from the arena for efficient
    /// memory management.
    gaps: Vec<WritableGap<T, GAP_SIZE>>,
    /// Arena allocator for managing gap structure memory. This allocator provides
    /// O(1) allocation with minimal memory fragmentation. All gap structures
    /// are stored contiguously within the arena, which improves cache performance
    /// when iterating over multiple gaps during multi-cursor operations.
    arena: Allocator,
}

impl<T, const GAP_SIZE: usize> WritableGaps<T, GAP_SIZE> {
    /// Creates a new gap manager

    pub fn new() -> Self {
        let arena = Allocator::new(1024);
        let gap = WritableGap::new();

        Self {
            gaps: vec![gap],
            arena,
        }
    }

    /// Returns the gap at the specified index
    ///
    /// # Safety
    ///
    /// The index must be within bounds

    pub unsafe fn get(&self, index: usize) -> Option<&WritableGap<T, GAP_SIZE>> {
        self.gaps.get(index)
    }

    /// Returns the mutable gap at the specified index
    ///
    /// # Safety
    ///
    /// The index must be within bounds

    pub unsafe fn get_mut(
        &mut self,
        index: usize,
    ) -> Option<&mut WritableGap<T, GAP_SIZE>> {
        self.gaps.get_mut(index)
    }

    /// Returns the number of gaps

    pub fn gap_count(&self) -> usize {
        self.gaps.len()
    }

    /// Returns the arena allocator

    pub fn arena(&self) -> &Allocator {
        &self.arena
    }

    /// Returns the mutable arena allocator

    pub fn arena_mut(&mut self) -> &mut Allocator {
        &mut self.arena
    }
}

impl<T, const GAP_SIZE: usize> Drop for WritableGaps<T, GAP_SIZE> {
    fn drop(&mut self) {
        // Clear gaps vector to ensure WritableGap instances are dropped
        // before the arena is dropped. This prevents dangling pointers
        // from WritableGap's gap_start/gap_end fields.
        self.gaps.clear();
    }
}

/// `pub enum WritableEventType`
///
/// Enumeration of event types emitted by writable data structures.
///
/// This enum defines all possible event types that can be emitted through
/// the event bridge system. Events are used to notify external systems
/// (such as UI components, logging systems, or synchronization mechanisms)
/// about changes to the writable buffer, cursor positions, or internal state.
///
/// Each event type carries specific information relevant to that type of
/// event, such as cursor positions for cursor events or error messages for
/// error events. The event system is designed to be decoupled from the
/// writable implementation through dependency injection.
///
/// # Variants
///
/// * `BufferChanged` - Emitted when the contents of the buffer have been
///   modified, typically after a flush operation or direct modification.
///   Carries a message describing the nature of the change.
/// * `CursorMoved` - Emitted when a cursor's position has changed. Includes
///   the cursor ID and the new row/column position.
/// * `CursorSelectionChanged` - Emitted when a cursor's selection range has
///   changed. Includes cursor ID, position, and selection start/end points.
/// * `CursorAdded` - Emitted when a new cursor is added to the writable.
///   Includes the ID of the newly added cursor.
/// * `CursorRemoved` - Emitted when a cursor is removed from the writable.
///   Includes the ID of the removed cursor.
/// * `GapBlockShifted` - Emitted when a gap block is shifted within the buffer.
///   Includes gap ID, offset, and size information for the shifted block.
/// * `Error` - Emitted when an error occurs during writable operations.
///   Carries an error message describing the failure.
#[repr(C)]
#[derive(Clone, Debug)]
pub enum WritableEventType {
    /// Emitted when the buffer contents have been modified. This typically
    /// occurs after a gap flush operation or when the buffer is directly
    /// modified. The event carries a message describing what changed.
    BufferChanged,
    /// Emitted when a cursor's position changes. This includes navigation
    /// operations, insertions that move the cursor, and explicit cursor
    /// positioning. The event includes the cursor ID and new row/column.
    CursorMoved,
    /// Emitted when a cursor's selection range changes. This occurs when
    /// the user selects or deselects text. The event includes cursor ID,
    /// position, and the selection start and end indices.
    CursorSelectionChanged,
    /// Emitted when a new cursor is added to the writable. This occurs
    /// when `add_cursor()` is called. The event includes the ID of the
    /// newly created cursor.
    CursorAdded,
    /// Emitted when a cursor is removed from the writable. This occurs
    /// when `remove_cursor()` is called. The event includes the ID of the
    /// removed cursor.
    CursorRemoved,
    /// Emitted when a gap block is shifted within the buffer. This occurs
    /// during gap buffer management when the gap needs to be moved to
    /// accommodate insertions. The event includes gap ID, offset, and size.
    GapBlockShifted,
    /// Emitted when an error occurs during writable operations. This can
    /// include allocation failures, invalid cursor operations, or other
    /// runtime errors. The event carries an error message describing the failure.
    Error,
}

/// `pub struct WritableEvent`
///
/// Event structure containing detailed information about writable events.
///
/// This struct carries all relevant information for events emitted by writable
/// data structures. Different event types use different subsets of the
/// available fields, with unused fields typically set to zero or empty
/// values. The event is designed to be passed through the event bridge
/// system to external observers.
///
/// The event structure is cloneable and debuggable to facilitate logging
/// and debugging of event flows. Events are typically created with the
/// `new()` constructor and then populated with relevant fields before
/// being emitted through the event bridge.
///
/// # Fields
///
/// * `event_type` - The type of event being emitted. This determines
///   which other fields are relevant and should be populated.
/// * `cursor_id` - The ID of the cursor associated with this event. Used
///   for cursor-related events (moved, added, removed, selection changed).
/// * `row` - The row position (line number) for cursor-related events.
///   Zero-indexed from the beginning of the buffer.
/// * `col` - The column position within the row for cursor-related events.
///   Zero-indexed from the beginning of the line.
/// * `start` - The start index of a selection for selection-related events.
///   Represents the beginning of the selected text range.
/// * `end` - The end index of a selection for selection-related events.
///   Represents the end of the selected text range.
/// * `offset` - The offset within the buffer for gap block events. Indicates
///   where the gap block was shifted from or to.
/// * `size` - The size of the gap block for gap block events. Indicates
///   how many elements were affected by the shift operation.
/// * `message` - A descriptive message for error and buffer changed events.
///   Contains human-readable information about what occurred.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct WritableEvent {
    /// The type of event being emitted. This field determines which other
    /// fields in the struct are relevant and should be populated by the
    /// event emitter. For example, cursor-related events should populate
    /// cursor_id, row, and col, while error events should populate message.
    pub event_type: WritableEventType,
    /// The ID of the cursor associated with this event. This field is used
    /// for cursor-related events (moved, added, removed, selection changed)
    /// to identify which cursor the event refers to. For non-cursor events,
    /// this field is typically set to 0.
    pub cursor_id: usize,
    /// The row position (line number) for cursor-related events. This is
    /// zero-indexed from the beginning of the buffer and represents the
    /// vertical position of the cursor. For non-cursor events, this field
    /// is typically set to 0.
    pub row: usize,
    /// The column position within the row for cursor-related events. This
    /// is zero-indexed from the beginning of the line and represents the
    /// horizontal position of the cursor. For non-cursor events, this field
    /// is typically set to 0.
    pub col: usize,
    /// The start index of a selection for selection-related events. This
    /// represents the beginning of the selected text range in the buffer.
    /// For non-selection events, this field is typically set to 0.
    pub start: usize,
    /// The end index of a selection for selection-related events. This
    /// represents the end of the selected text range in the buffer. The
    /// selection includes all characters from start to end (exclusive).
    /// For non-selection events, this field is typically set to 0.
    pub end: usize,
    /// The offset within the buffer for gap block events. This indicates
    /// where in the buffer the gap block was shifted from or to. It's
    /// used to track the movement of gap blocks during buffer operations.
    /// For non-gap events, this field is typically set to 0.
    pub offset: usize,
    /// The size of the gap block for gap block events. This indicates how
    /// many elements were affected by the gap block shift operation. It's
    /// used to track the extent of gap movements during buffer operations.
    /// For non-gap events, this field is typically set to 0.
    pub size: usize,
    /// A descriptive message for error and buffer changed events. This
    /// contains human-readable information about what occurred, such as
    /// error descriptions or details about buffer modifications. For events
    /// that don't require a message, this field is typically empty.
    pub message: String,
}

impl WritableEvent {
    /// Creates a new writable event

    pub fn new(event_type: WritableEventType) -> Self {
        Self {
            event_type,
            cursor_id: 0,
            row: 0,
            col: 0,
            start: 0,
            end: 0,
            offset: 0,
            size: 0,
            message: String::new(),
        }
    }
}

/// `pub struct BaseWritable<Raw, Buf, const GAP_SIZE: WritableGapSize>`
///
/// The codevar-core writable data structure with multi-cursor gap buffer support.
///
/// This struct implements a gap buffer data structure adapted for multi-cursor
/// scenarios. Unlike traditional gap buffers which maintain a single gap, this
/// implementation supports multiple gaps - one per cursor, with shared gaps
/// created when cursors are in proximity. This design enables efficient
/// concurrent editing from multiple cursor positions without excessive memory
/// overhead.
///
/// The gap buffer optimization allows O(1) insertions at cursor positions by
/// maintaining unused space (gaps) where insertions occur. When a gap fills up,
/// its contents are flushed to the main buffer, and the gap may be moved to
/// a new position. This avoids the O(n) cost of shifting all subsequent elements
/// that would occur in a standard array-based buffer.
///
/// The struct uses unsafe cells for interior mutability, allowing multiple
/// cursors to modify the shared buffer without requiring mutable references.
/// Event emission is handled through dependency injection via the event bridge
/// trait, decoupling the writable implementation from specific event systems.
///
/// # Type Parameters
///
/// * `Raw` - The raw data type stored in the buffer (typically u32 for packed data)
/// * `Buf` - The buffer type for output operations (e.g., u8 for UTF-8 output)
/// * `GAP_SIZE` - The gap size enum determining gap buffer capacity
///
/// # Fields
///
/// * `raw` - The aligned pointer to the raw data array. This stores the main
///   buffer contents as packed 32-bit values combining type and character
///   information. The alignment (typically 64 bytes) ensures optimal cache
///   performance on modern architectures.
/// * `buffer` - Optional pointer to the output buffer. This buffer is used
///   during flush operations to hold the decoded character data before it's
///   written to external systems. It's allocated on-demand and freed when
///   the writable is dropped.
/// * `gaps` - The gap manager containing all active gap buffers. This is
///   wrapped in UnsafeCell for interior mutability, allowing gaps to be
///   modified through immutable references to the writable. The manager uses
///   an arena allocator for efficient gap allocation.
/// * `cursors` - The vector of active cursors. Each cursor maintains its
///   position, bidirectional index, and dimension information. Cursors are
///   stored as options to handle removal without shifting indices. The vector
///   is wrapped in UnsafeCell for interior mutability.
/// * `active_cursor_id` - The ID of the currently active cursor. This determines
///   which cursor is used for operations that don't specify a cursor ID.
///   Wrapped in UnsafeCell for interior mutability.
/// * `event_bridge` - The event bridge for emitting events. This uses dependency
///   injection to decouple the writable from specific event systems. Different
///   implementations (channel-based, null, custom) can be injected as needed.
///   Wrapped in UnsafeCell for interior mutability.
/// * `name` - The name of the writable (e.g., file name). This is stored as
///   a fixed-size byte array for C compatibility and is null-terminated.
///   The maximum length is defined by NAME_LEN.
/// * `flags` - Bit flags for writable state and configuration. These flags
///   can be used to enable/disable features or track state information.
/// * `raw_length` - The current length of valid data in the raw buffer. This
///   tracks how many elements are actually in use, as opposed to the allocated
///   capacity. It's used during flush operations and bounds checking.
/// * `space_y` - The row count (vertical space) in the writable. This tracks
///   how many newline characters have been encountered, used for cursor
///   positioning and display calculations.
#[repr(C)]
pub struct BaseWritable<Raw, Buf, const GAP_SIZE: usize> {
    /// Aligned pointer to the raw data array. This stores the main buffer
    /// contents as packed 32-bit values that combine type and character
    /// information. The alignment (typically 64 bytes) ensures optimal
    /// cache line utilization on modern architectures and enables SIMD
    /// optimizations. The pointer manages its own allocation and growth.
    /// Wrapped in UnsafeCell for interior mutability.
    raw: UnsafeCell<WritableAlignedPtr<Raw, 64>>,
    /// Optional pointer to the output buffer. This buffer is allocated on-demand
    /// during flush operations to hold decoded character data before it's
    /// written to external systems. The buffer type determines the encoding
    /// (e.g., u8 for UTF-8, u16 for UTF-16). It's freed automatically when
    /// the writable is dropped.
    buffer: Option<NonNull<Buf>>,
    /// The gap manager containing all active gap buffers. This manager uses
    /// an arena allocator for efficient gap allocation and provides indexed
    /// access to individual gaps. Wrapped in UnsafeCell for interior
    /// mutability, allowing gap modifications through immutable references.
    gaps: UnsafeCell<WritableGaps<Raw, GAP_SIZE>>,
    /// Shared gaps for multi-cursor scenarios. When two or more cursors are
    /// within GAP_SIZE range of each other, they share a single gap buffer
    /// with suballocated regions for each cursor. This vector stores all
    /// currently active shared gaps. Wrapped in UnsafeCell for interior mutability.
    shared_gaps: UnsafeCell<Vec<SharedGap<Raw, GAP_SIZE>>>,
    /// Maps cursor IDs to their assigned gap index. This tracks which gap
    /// (regular or shared) each cursor belongs to. A value of usize::MAX
    /// indicates the cursor is not currently assigned to any gap.
    /// Wrapped in UnsafeCell for interior mutability.
    cursor_to_gap: UnsafeCell<Vec<usize>>,
    /// The vector of active cursors. Each cursor maintains its position,
    /// bidirectional index, and dimension information for navigation and
    /// selection. Cursors are stored as options to handle removal without
    /// shifting subsequent indices. Wrapped in UnsafeCell for interior mutability.
    cursors: UnsafeCell<Vec<Option<Cursor<Raw, Buf, GAP_SIZE>>>>,
    /// The ID of the currently active cursor. This determines which cursor
    /// is used for operations that don't specify a cursor ID. The active
    /// cursor can be changed via set_active_cursor(). Wrapped in UnsafeCell
    /// for interior mutability.
    active_cursor_id: UnsafeCell<usize>,
    /// The event bridge for emitting events. This uses dependency injection
    /// to decouple the writable from specific event systems. Different
    /// implementations can be injected (ChannelEventBridge for real events,
    /// NullEventBridge for testing). Wrapped in UnsafeCell for interior mutability.
    event_bridge: UnsafeCell<Option<Arc<dyn EventBridge>>>,
    /// The user action observer registry for tracking user actions. This enables
    /// collaborative features by capturing and propagating user edits to observers.
    /// Wrapped in UnsafeCell for interior mutability.
    action_observer_registry: UnsafeCell<UserActionObserverRegistry>,
    /// The history tracking for this writable. This maintains a record of
    /// all changes made to the buffer, enabling undo/redo functionality and
    /// change tracking. Wrapped in UnsafeCell for interior mutability.
    history: UnsafeCell<History>,
    /// The name of the writable (e.g., file name). Stored as a fixed-size
    /// byte array for C compatibility and null-terminated. The maximum
    /// length is defined by NAME_LEN (255 bytes). Used for identification
    /// and debugging purposes.
    name: [u8; NAME_LEN],
    /// Bit flags for writable state and configuration. These flags can
    /// enable/disable features or track state information. Individual bits
    /// can represent settings like read-only mode, auto-flush enabled, etc.
    flags: u32,
    /// The current length of valid data in the raw buffer. This tracks how
    /// many elements are actually in use, as opposed to the allocated capacity.
    /// Used during flush operations, bounds checking, and to determine when
    /// reallocation is needed.
    raw_length: usize,
    /// The row count (vertical space) in the writable. This tracks how many
    /// newline characters have been encountered during insertions. Used for
    /// cursor positioning, display calculations, and determining line numbers.
    vspace: usize,
    /// The text encoding used for the buffer content. This field specifies
    /// which character encoding is used when reading from or writing to the
    /// buffer. Common values include ENCODING_UTF8, ENCODING_UTF16, ENCODING_UTF32,
    /// ENCODING_ISO_8859_1, ENCODING_WINDO1252, and ENCODING_ASCII. The encoding
    /// affects how bytes are interpreted as characters during serialization,
    /// deserialization, and I/O operations. Default is UTF-8 for maximum compatibility.
    encoding: u16,
}

impl<Raw, Buf, const GAP_SIZE: usize> BaseWritable<Raw, Buf, GAP_SIZE> {
    /// Creates a new base writable

    pub fn new() -> Self {
        let mut raw = WritableAlignedPtr::<Raw, 64>::new();
        let gap_size = INIT_RAW_BUFFER_COUNT + GAP_SIZE;
        unsafe {
            raw.malloc(gap_size);
            std::ptr::write_bytes(raw.as_mut_ptr(), 0, gap_size);
        }
        let mut gaps = WritableGaps::new();
        unsafe {
            if let Some(gap) = gaps.get_mut(0) {
                gap.set_gap_bounds(raw.as_mut_ptr(), raw.as_mut_ptr().add(gap_size));
                gap.gap_ptr_mut().allocate();
            }
        }
        let mut initial_cursor = Cursor::new();
        unsafe {
            initial_cursor.set_raw(raw.as_ptr(), gap_size);
        }
        Self {
            raw: UnsafeCell::new(raw),
            buffer: None,
            gaps: UnsafeCell::new(gaps),
            shared_gaps: UnsafeCell::new(Vec::new()),
            cursor_to_gap: UnsafeCell::new(vec![usize::MAX]),
            cursors: UnsafeCell::new(vec![Some(initial_cursor)]),
            active_cursor_id: UnsafeCell::new(0),
            event_bridge: UnsafeCell::new(None),
            action_observer_registry: UnsafeCell::new(UserActionObserverRegistry::new()),
            history: UnsafeCell::new(History::new()),
            name: [0; NAME_LEN],
            flags: 0,
            raw_length: 0,
            vspace: 0,
            encoding: ENCODING_UTF8,
        }
    }

    /// Creates a new base writable with the specified name and capacity
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the writable
    /// * `capacity` - The initial capacity

    pub fn with_capacity(name: &str, capacity: usize) -> Self {
        let n_capacity = if capacity == 0 {
            INIT_RAW_BUFFER_COUNT
        } else {
            capacity
        };
        let gap_size = n_capacity + GAP_SIZE;

        let mut raw = WritableAlignedPtr::<Raw, 64>::new();

        unsafe {
            raw.malloc(gap_size);
            std::ptr::write_bytes(raw.as_mut_ptr(), 0, gap_size);
        }

        let mut name_bytes = [0u8; NAME_LEN];
        let name_len = name.len().min(NAME_LEN - 1);
        name_bytes[..name_len].copy_from_slice(name.as_bytes());

        let mut gaps = WritableGaps::new();
        unsafe {
            if let Some(gap) = gaps.get_mut(0) {
                gap.set_gap_bounds(raw.as_mut_ptr(), raw.as_mut_ptr().add(gap_size));
                gap.gap_ptr_mut().allocate();
            }
        }

        let mut initial_cursor = Cursor::new();
        unsafe {
            initial_cursor.set_raw(raw.as_ptr(), gap_size);
        }

        Self {
            raw: UnsafeCell::new(raw),
            buffer: None,
            gaps: UnsafeCell::new(gaps),
            shared_gaps: UnsafeCell::new(Vec::new()),
            cursor_to_gap: UnsafeCell::new(vec![usize::MAX]),
            cursors: UnsafeCell::new(vec![Some(initial_cursor)]),
            active_cursor_id: UnsafeCell::new(0),
            event_bridge: UnsafeCell::new(None),
            action_observer_registry: UnsafeCell::new(UserActionObserverRegistry::new()),
            history: UnsafeCell::new(History::new()),
            name: name_bytes,
            flags: 0,
            raw_length: 0,
            vspace: 0,
            encoding: ENCODING_UTF8,
        }
    }

    /// Adds a new cursor

    pub fn append_cursor(&self) -> usize {
        unsafe {
            let cursors = &mut *self.cursors.get();
            let cursor_to_gap = &mut *self.cursor_to_gap.get();
            if cursors.len() >= MAX_CURSOR_COUNT {
                return cursors.len() - 1;
            }
            let active_id = *self.active_cursor_id.get();
            let mut new_cursor =
                if active_id < cursors.len() && cursors[active_id].is_some() {
                    if let Some(cursor_ref) = &cursors[active_id].as_ref() {
                        Cursor::clone(cursor_ref)
                    } else {
                        Cursor::new()
                    }
                } else {
                    Cursor::new()
                };

            let raw_len = self.cursor_raw_length();
            new_cursor.set_raw(self.raw_ptr(), raw_len);

            cursors.push(Some(new_cursor));
            cursor_to_gap.push(usize::MAX); // Initially unassigned to any gap
            cursors.len() - 1
        }
    }

    /// Removes a cursor

    pub fn delete_cursor(&self, cursor_id: usize) -> bool {
        unsafe {
            let cursors = &mut *self.cursors.get();
            let cursor_to_gap = &mut *self.cursor_to_gap.get();
            if cursor_id >= cursors.len() || cursor_id == 0 {
                return false;
            }
            // Remove cursor from any shared gap it belongs to
            let gap_index = cursor_to_gap[cursor_id];
            if gap_index != usize::MAX {
                let shared_gaps = &mut *self.shared_gaps.get();
                if let Some(shared_gap) = shared_gaps.get(gap_index) {
                    shared_gap.remove_cursor_region(cursor_id);
                }
            }
            cursors[cursor_id] = None;
            cursor_to_gap[cursor_id] = usize::MAX;

            let active_id = *self.active_cursor_id.get();
            if active_id == cursor_id {
                *self.active_cursor_id.get() = 0;
            }
            true
        }
    }

    /// Returns the cursor with the specified ID

    pub fn cursor_from_id(
        &self,
        cursor_id: usize,
    ) -> Option<&Cursor<Raw, Buf, GAP_SIZE>> {
        unsafe {
            let cursors = &*self.cursors.get();
            if cursor_id >= cursors.len() {
                None
            } else {
                cursors[cursor_id].as_ref()
            }
        }
    }

    pub fn v_space(&self) -> usize {
        self.vspace
    }

    pub fn v_space_mut(&mut self) -> &mut usize {
        &mut self.vspace
    }

    /// Returns the current text encoding.
    ///
    /// # Returns
    ///
    /// The encoding value (e.g., ENCODING_UTF8, ENCODING_UTF16, etc.)
    pub fn encoding(&self) -> u16 {
        self.encoding
    }

    /// Sets the text encoding.
    ///
    /// # Arguments
    ///
    /// * `encoding` - The new encoding value (e.g., ENCODING_UTF8, ENCODING_UTF16, etc.)
    pub fn set_encoding(&mut self, encoding: u16) {
        self.encoding = encoding;
    }

    pub fn shared_gaps(&self) -> &UnsafeCell<Vec<SharedGap<Raw, GAP_SIZE>>> {
        &self.shared_gaps
    }

    /// Returns the mutable cursor with the specified ID
    pub unsafe fn cursor_from_id_mut(
        &self,
        cursor_id: usize,
    ) -> Option<&mut Cursor<Raw, Buf, GAP_SIZE>> {
        let cursors = &mut *self.cursors.get();
        if cursor_id >= cursors.len() {
            None
        } else {
            cursors[cursor_id].as_mut()
        }
    }

    /// Returns the active cursor ID

    pub fn active_cursor_id(&self) -> usize {
        unsafe { *self.active_cursor_id.get() }
    }

    /// Sets the active cursor ID

    pub fn set_active_cursor(&self, cursor_id: usize) {
        unsafe {
            let cursors = &*self.cursors.get();
            if cursor_id < cursors.len() {
                *self.active_cursor_id.get() = cursor_id;
            }
        }
    }

    /// Returns the active cursor

    pub fn active_cursor(&self) -> Option<&Cursor<Raw, Buf, GAP_SIZE>> {
        unsafe { self.cursor_from_id(self.active_cursor_id()) }
    }

    /// Returns the raw pointer

    pub fn raw_ptr(&self) -> *const Raw {
        unsafe { (*self.raw.get()).as_ptr() }
    }

    /// Returns the mutable raw pointer

    pub unsafe fn raw_mut_ptr(&self) -> *mut Raw {
        (*self.raw.get()).as_mut_ptr()
    }

    /// Returns the raw length

    pub fn raw_length(&self) -> usize {
        self.raw_length
    }

    /// Eagerly copies `bytes` into the raw buffer and updates the length.
    ///
    /// Bytes beyond the allocated raw region are ignored.
    pub fn load_bytes(&mut self, bytes: &[u8]) {
        let max_len = unsafe { (*self.raw.get()).count() };
        let copy_len = bytes.len().min(max_len);
        if copy_len == 0 {
            self.raw_length = 0;
            return;
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr().cast::<Raw>(),
                self.raw_mut_ptr(),
                copy_len,
            );
        }
        self.raw_length = copy_len;
    }

    /// Returns the raw buffer length available to cursors.

    fn cursor_raw_length(&self) -> usize {
        self.raw_length
    }

    /// Sets the event bridge (dependency injection)
    ///
    /// # Arguments
    ///
    /// * `bridge` - The event bridge to use

    pub fn set_event_bridge(&self, bridge: Arc<dyn EventBridge>) {
        unsafe {
            *self.event_bridge.get() = Some(bridge);
        }
    }

    /// Returns the event bridge

    pub fn event_bridge(&self) -> Option<Arc<dyn EventBridge>> {
        unsafe { (*self.event_bridge.get()).as_ref().cloned() }
    }

    /// Registers a user action observer.
    ///
    /// # Arguments
    ///
    /// * `observer` - The observer to register
    pub fn register_action_observer(&self, observer: Arc<dyn UserActionObserver>) {
        unsafe {
            (*self.action_observer_registry.get()).register(observer);
        }
    }

    /// Notifies all observers of a user action.
    ///
    /// # Arguments
    ///
    /// * `event` - The user action event
    pub fn notify_action_observers(&self, event: &UserActionEvent) {
        unsafe {
            (*self.action_observer_registry.get()).notify(event);
        }
    }

    /// Returns the action observer registry.
    pub fn action_observer_registry(&self) -> &UnsafeCell<UserActionObserverRegistry> {
        &self.action_observer_registry
    }

    /// Checks if two cursors are within GAP_SIZE range of each other.
    ///
    /// # Arguments
    ///
    /// * `cursor_id1` - The first cursor ID
    /// * `cursor_id2` - The second cursor ID
    ///
    /// # Returns
    ///
    /// `true` if the cursors are within GAP_SIZE range, `false` otherwise

    pub unsafe fn proximity(&self, cursor_id1: usize, cursor_id2: usize) -> bool {
        if let (Some(c1), Some(c2)) = (
            self.cursor_from_id(cursor_id1),
            self.cursor_from_id(cursor_id2),
        ) {
            let pos1 = c1.array_pos();
            let pos2 = c2.array_pos();
            pos1.abs_diff(pos2) <= GAP_SIZE
        } else {
            false
        }
    }

    pub fn name(&self) -> &[u8] {
        &self.name
    }

    pub fn name_mut(&mut self) -> &mut [u8] {
        self.name.as_mut()
    }

    pub fn flags(&self) -> u32 {
        self.flags
    }

    pub fn flags_mut(&mut self) -> &mut u32 {
        &mut self.flags
    }

    /// Finds or creates a shared gap for the specified cursor.
    ///
    /// This method checks if the cursor is in proximity with any other cursor
    /// that already belongs to a shared gap. If so, it adds the cursor to that
    /// shared gap. Otherwise, it checks if the cursor is in proximity with any
    /// other cursor and creates a new shared gap if needed.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID to find or create a shared gap for
    ///
    /// # Safety
    ///
    /// Initializes a shared gap's circular buffer and main-buffer bounds.

    unsafe fn init_shared_gap(
        shared_gap: &mut SharedGap<Raw, GAP_SIZE>,
        raw_ptr: *mut Raw,
    ) {
        if raw_ptr.is_null() {
            return;
        }
        if shared_gap.gap().gap_ptr().buff().is_null() {
            shared_gap.gap_mut().gap_ptr_mut().allocate();
        }
        if shared_gap.gap().gap_start().is_null() {
            shared_gap
                .gap_mut()
                .set_gap_bounds(raw_ptr, raw_ptr.add(GAP_SIZE));
        }
    }

    /// Finds or creates a shared gap for the specified cursor.

    pub unsafe fn new_shared_gap(&self, cursor_id: usize) {
        let cursor_to_gap = &mut *self.cursor_to_gap.get();
        let shared_gaps = &mut *self.shared_gaps.get();
        if cursor_to_gap[cursor_id] != usize::MAX {
            return;
        }
        let raw_ptr = self.raw_mut_ptr();
        for (other_id, &gap_idx) in cursor_to_gap.iter().enumerate() {
            if other_id != cursor_id
                && other_id > 0
                && gap_idx != usize::MAX
                && self.proximity(cursor_id, other_id)
            {
                // Add cursor to existing shared gap
                if let Some(shared_gap) = shared_gaps.get_mut(gap_idx) {
                    Self::init_shared_gap(shared_gap, raw_ptr);
                    shared_gap.add_cursor_region(cursor_id);
                    cursor_to_gap[cursor_id] = gap_idx;
                    return;
                }
            }
        }
        // Check if cursor is in proximity with any other cursor (create new shared gap)
        for other_id in 0..cursor_to_gap.len() {
            if other_id != cursor_id
                && other_id > 0
                && self.proximity(cursor_id, other_id)
            {
                let new_shared_gap = SharedGap::new();
                let gap_idx = shared_gaps.len();
                shared_gaps.push(new_shared_gap);

                if let Some(shared_gap) = shared_gaps.get_mut(gap_idx) {
                    Self::init_shared_gap(shared_gap, raw_ptr);
                    shared_gap.add_cursor_region(cursor_id);
                    shared_gap.add_cursor_region(other_id);
                    cursor_to_gap[cursor_id] = gap_idx;
                    cursor_to_gap[other_id] = gap_idx;
                }
                return;
            }
        }
    }

    /// Returns the shared gap index for the specified cursor.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID
    ///
    /// # Returns
    ///
    /// The shared gap index, or usize::MAX if the cursor is not in a shared gap

    pub unsafe fn shared_gap_index(&self, cursor_id: usize) -> Option<usize> {
        if cursor_id >= self.cursor_count() {
            return None;
        }
        let cursor_to_gap = &*self.cursor_to_gap.get();
        Some(cursor_to_gap[cursor_id])
    }

    /// Checks if a cursor is in a shared gap.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID
    ///
    /// # Returns
    ///
    /// `true` if the cursor is in a shared gap, `false` otherwise

    pub unsafe fn is_cursor_in_shared_gap(&self, cursor_id: usize) -> bool {
        if let Some(val) = self.shared_gap_index(cursor_id) {
            val != usize::MAX
        } else {
            false
        }
    }

    /// Returns a mutable reference to the raw buffer length.

    pub(crate) fn raw_length_mut(&mut self) -> &mut usize {
        &mut self.raw_length
    }

    /// Returns a mutable reference to the optional merge output buffer.

    pub(crate) fn buffer_mut(&mut self) -> &mut Option<NonNull<Buf>> {
        &mut self.buffer
    }

    /// Returns mutable access to the gap manager.
    ///
    /// # Safety
    ///
    /// Caller must not create conflicting aliases to gap data.

    pub(crate) unsafe fn gaps_mut(&self) -> &mut WritableGaps<Raw, GAP_SIZE> {
        &mut *self.gaps.get()
    }

    /// Returns read-only access to the gap manager
    ///
    /// # Safety
    ///
    /// Caller must not create conflicting aliases to gap data.

    pub(crate) unsafe fn gaps(&self) -> &WritableGaps<Raw, GAP_SIZE> {
        &*self.gaps.get()
    }

    /// Returns mutable access to the shared gaps vector.
    ///
    /// # Safety
    ///
    /// Caller must not create conflicting aliases to shared gap data.

    pub(crate) unsafe fn shared_gaps_mut(&self) -> &mut Vec<SharedGap<Raw, GAP_SIZE>> {
        &mut *self.shared_gaps.get()
    }

    /// Returns the cursor-to-gap mapping table.

    pub(crate) unsafe fn cursor_to_gap(&self) -> &Vec<usize> {
        &*self.cursor_to_gap.get()
    }

    /// Reanchors the primary gap after a direct raw-buffer edit.
    pub(crate) unsafe fn reset_gap_at(&mut self, position: usize) {
        let raw = self.raw_mut_ptr();
        if raw.is_null() {
            return;
        }
        let capacity = (*self.raw.get()).allocated();
        if position.saturating_add(GAP_SIZE) > capacity {
            return;
        }

        if let Some(gap) = self.gaps_mut().get_mut(0) {
            gap.set_gap_bounds(raw.add(position), raw.add(position + GAP_SIZE));
            gap.gap_ptr_mut().set_gap_ptr(0);
            let buffer = gap.gap_ptr_mut().buff_mut();
            if !buffer.is_null() {
                std::ptr::write_bytes(buffer, 0, GAP_SIZE);
            }
        }
        self.shared_gaps_mut().clear();
        for gap in &mut *self.cursor_to_gap.get() {
            *gap = usize::MAX;
        }
    }

    /// Adjusts cursor positions after removing a raw-buffer range.
    pub(crate) unsafe fn adjust_cursors_after_delete(&self, start: usize, end: usize) {
        let removed = end.saturating_sub(start);
        if removed == 0 {
            return;
        }
        for cursor in (&mut *self.cursors.get()).iter_mut().flatten() {
            let position = cursor.array_pos();
            let adjusted = if position >= end {
                position - removed
            } else if position > start {
                start
            } else {
                position
            };
            cursor.set_array_pos(adjusted);
        }
    }

    /// Returns the number of registered cursors.

    pub(crate) unsafe fn cursor_count(&self) -> usize {
        (*self.cursors.get()).len()
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> Writable<Raw, Buf, GAP_SIZE>
    for BaseWritable<Raw, Buf, GAP_SIZE>
{
    fn new() -> Self {
        Self::new()
    }

    fn with_capacity(name: &str, capacity: usize) -> Self {
        Self::with_capacity(name, capacity)
    }

    fn append_cursor(&self) -> usize {
        self.append_cursor()
    }

    fn delete_cursor(&self, cursor_id: usize) -> bool {
        self.delete_cursor(cursor_id)
    }

    fn cursor_from_id(&self, cursor_id: usize) -> Option<&Cursor<Raw, Buf, GAP_SIZE>> {
        self.cursor_from_id(cursor_id)
    }

    unsafe fn cursor_from_id_mut(
        &self,
        cursor_id: usize,
    ) -> Option<&mut Cursor<Raw, Buf, GAP_SIZE>> {
        self.cursor_from_id_mut(cursor_id)
    }

    fn v_space(&self) -> usize {
        self.v_space()
    }

    fn shared_gaps(&self) -> &UnsafeCell<Vec<SharedGap<Raw, GAP_SIZE>>> {
        self.shared_gaps()
    }

    fn active_cursor_id(&self) -> usize {
        self.active_cursor_id()
    }

    fn set_active_cursor(&self, cursor_id: usize) {
        self.set_active_cursor(cursor_id)
    }

    fn active_cursor(&self) -> Option<&Cursor<Raw, Buf, GAP_SIZE>> {
        self.active_cursor()
    }

    fn raw_ptr(&self) -> *const Raw {
        self.raw_ptr()
    }

    unsafe fn raw_mut_ptr(&self) -> *mut Raw {
        self.raw_mut_ptr()
    }

    fn raw_length(&self) -> usize {
        self.raw_length()
    }

    fn set_event_bridge(&self, bridge: Arc<dyn EventBridge>) {
        self.set_event_bridge(bridge)
    }

    fn event_bridge(&self) -> Option<Arc<dyn EventBridge>> {
        self.event_bridge()
    }

    fn name(&self) -> &[u8] {
        self.name()
    }

    fn name_mut(&mut self) -> &mut [u8] {
        self.name_mut()
    }

    fn flags(&self) -> u32 {
        self.flags()
    }

    fn flags_mut(&mut self) -> &mut u32 {
        self.flags_mut()
    }

    unsafe fn proximity(&self, cursor_id1: usize, cursor_id2: usize) -> bool {
        self.proximity(cursor_id1, cursor_id2)
    }

    unsafe fn new_shared_gap(&self, cursor_id: usize) {
        self.new_shared_gap(cursor_id)
    }

    unsafe fn shared_gap_index(&self, cursor_id: usize) -> usize {
        if let Some(val) = self.shared_gap_index(cursor_id) {
            val
        } else {
            usize::MAX
        }
    }

    unsafe fn is_cursor_in_shared_gap(&self, cursor_id: usize) -> bool {
        self.is_cursor_in_shared_gap(cursor_id)
    }

    fn history(&self) -> &UnsafeCell<History> {
        &self.history
    }

    fn history_current_index(&self) -> usize {
        unsafe { (*self.history.get()).current_index() }
    }

    fn set_history_current_index(&self, index: usize) {
        unsafe {
            (*self.history.get()).set_current_index(index);
        }
    }

    unsafe fn record_change(&self) {
        let history = &mut *self.history.get();
        // Update timestamp and hash when recording a change
        let timestamp = history.gen_history_timestamp();
        let hash = history.gen_history_hash();
        history.set_timestamp(timestamp);
        history.set_hash(hash);
    }

    fn register_action_observer(&self, observer: Arc<dyn UserActionObserver>) {
        self.register_action_observer(observer);
    }

    fn notify_action_observers(&self, event: &UserActionEvent) {
        self.notify_action_observers(event);
    }

    unsafe fn select_all(&mut self) {
        if let Some(cursor) = self.cursor_from_id_mut(0) {
            cursor.move_to(0, 0);
            cursor.set_selection_anchor(0);
            let content_len = self.raw_length();
            cursor.set_selection_end(content_len);
        }
    }

    unsafe fn copy_selection(&mut self) {
        let _ = self;
    }

    unsafe fn paste_from_clipboard(&mut self) {
        let _ = self;
    }

    unsafe fn cut_selection(&mut self) {
        let _ = self;
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> Drop for BaseWritable<Raw, Buf, GAP_SIZE> {
    fn drop(&mut self) {
        unsafe {
            if let Some(buffer) = self.buffer {
                let layout = Layout::array::<Buf>(self.raw_length).unwrap_or_else(|_| {
                    // Fallback to a safe default layout if the requested one is invalid
                    Layout::new::<Buf>()
                });
                alloc::dealloc(buffer.as_ptr() as *mut u8, layout);
            }
        }
    }
}

/// `pub struct StreamWritable<Raw, Buf, const GAP_SIZE: WritableGapSize>`
///
/// Stream-oriented writable with character-level manipulation and gap buffer support.
///
/// This struct extends BaseWritable with stream-specific operations for
/// character-level manipulation. It provides methods for putting characters,
/// managing gap buffers, and flushing data to output buffers. The stream
/// writable is designed for scenarios where data is written character by
/// character (e.g., text input, streaming data processing) rather than
/// bulk operations.
///
/// The stream writable implements bit packing to combine type and character
/// information into a single 32-bit value. This packing enables efficient
/// storage and allows the gap buffer to distinguish between different types
/// of data (mergeable, common, etc.) during flush operations. The packing
/// uses the upper 4 bits for type and the lower 24 bits for character data.
///
/// # Type Parameters
///
/// * `Raw` - The raw data type (typically u32 for packed data)
/// * `Buf` - The buffer type for output (e.g., u8 for UTF-8)
/// * `GAP_SIZE` - The gap size enum determining gap buffer capacity
///
/// # Fields
///
/// * `base` - The underlying BaseWritable that provides codevar-core functionality
///   including raw buffer management, cursor management, and gap management.
///   The stream writable delegates most operations to this base instance.
/// * `raw_t` - PhantomData marker for the Raw type parameter. This is used
///   to make the generic type parameter part of the struct without storing
///   an actual value. It enables the struct to be generic over Raw while
///   maintaining zero-cost abstraction.
#[repr(C)]
pub struct StreamWritable<Raw, Buf, const GAP_SIZE: usize> {
    /// The underlying BaseWritable instance that provides codevar-core functionality.
    /// This includes raw buffer management (allocation, growth, access),
    /// cursor management (add, remove, access), gap management (gap
    /// allocation, bounds setting), and event emission. The stream writable
    /// extends this with character-level operations and gap buffer flushing.
    base: BaseWritable<Raw, Buf, GAP_SIZE>,
    /// PhantomData marker for the Raw type parameter. This Rust construct
    /// allows the struct to be generic over Raw without actually storing
    /// a value of type Raw. It's used here to enable type-level operations
    /// (like make_packed, get_type, get_char) while maintaining zero-cost
    /// abstraction - the PhantomData compiles away to nothing.
    raw_: std::marker::PhantomData<Raw>,
}

impl<Raw, Buf, const GAP_SIZE: usize> StreamWritable<Raw, Buf, GAP_SIZE> {
    /// Bit shift for type field in packed value
    pub const SHIFT_TYPE: u32 = 28;
    /// Bit shift for character field in packed value
    pub const SHIFT_CHAR: u32 = 0;
    /// Mask for type field
    pub const TYPE_MASK: u32 = 0xF;
    /// Mask for character field
    pub const CHAR_MASK: u32 = 0xFFFFFF;
    /// ID for mergeable type
    pub const T_MERGE: u32 = 0xD;
    /// ID for common type
    pub const T_COMMON: u32 = 0xE;

    /// Creates a new stream writable

    pub fn new() -> Self {
        Self {
            base: BaseWritable::new(),
            raw_: std::marker::PhantomData,
        }
    }

    pub fn base(&self) -> &BaseWritable<Raw, Buf, { GAP_SIZE }> {
        &self.base
    }
    pub fn base_mut(&mut self) -> &mut BaseWritable<Raw, Buf, GAP_SIZE> {
        &mut self.base
    }

    /// Creates a new stream writable with the specified name and capacity
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the writable
    /// * `capacity` - The initial capacity

    pub fn with_capacity(name: &str, capacity: usize) -> Self {
        Self {
            base: BaseWritable::with_capacity(name, capacity),
            raw_: std::marker::PhantomData,
        }
    }

    /// Creates a packed value from type and character
    ///
    /// # Arguments
    ///
    /// * `type_` - The type field (4 bits)
    /// * `c` - The character field (24 bits)
    ///
    /// # Returns
    ///
    /// The packed 32-bit value

    pub fn make_packed(type_: u32, c: u32) -> u32 {
        ((type_ & 0xF) << Self::SHIFT_TYPE) | ((c & 0xFFFFFF) << Self::SHIFT_CHAR)
    }

    /// Extracts the type field from a packed value
    ///
    /// # Arguments
    ///
    /// * `packed` - The packed value
    ///
    /// # Returns
    ///
    /// The type field

    pub fn packed(packed: u32) -> u32 {
        (packed >> Self::SHIFT_TYPE) & Self::TYPE_MASK
    }

    /// Extracts the character field from a packed value
    ///
    /// # Arguments
    ///
    /// * `packed` - The packed value
    ///
    /// # Returns
    ///
    /// The character field

    pub fn char(packed: u32) -> u32 {
        (packed & Self::CHAR_MASK) >> Self::SHIFT_CHAR
    }

    /// Flushes the gap buffer
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The cursor ID
    ///
    /// # Safety
    ///
    /// The cursor must be valid

    pub unsafe fn flush_gap(&mut self, cursor_id: usize) {
        super::wredit_flush::flush_gap(&mut self.base, cursor_id);
    }

    /// Flushes all gap buffers (regular and shared) into the raw buffer.
    ///
    /// # Safety
    ///
    /// All cursors must remain valid for the duration of the flush.

    pub unsafe fn flush_all(&mut self) -> usize {
        super::wredit_flush::flush_all(&mut self.base)
    }
}

impl<Raw, Buf, const GAP_SIZE: usize> Writable<Raw, Buf, GAP_SIZE>
    for StreamWritable<Raw, Buf, GAP_SIZE>
{
    fn new() -> Self {
        Self::new()
    }

    fn with_capacity(name: &str, capacity: usize) -> Self {
        Self::with_capacity(name, capacity)
    }

    fn append_cursor(&self) -> usize {
        self.base.append_cursor()
    }

    fn delete_cursor(&self, cursor_id: usize) -> bool {
        self.base.delete_cursor(cursor_id)
    }

    fn cursor_from_id(&self, cursor_id: usize) -> Option<&Cursor<Raw, Buf, GAP_SIZE>> {
        self.base.cursor_from_id(cursor_id)
    }

    unsafe fn cursor_from_id_mut(
        &self,
        cursor_id: usize,
    ) -> Option<&mut Cursor<Raw, Buf, GAP_SIZE>> {
        self.base.cursor_from_id_mut(cursor_id)
    }

    fn v_space(&self) -> usize {
        self.base.v_space()
    }

    fn shared_gaps(&self) -> &UnsafeCell<Vec<SharedGap<Raw, GAP_SIZE>>> {
        self.base.shared_gaps()
    }

    fn active_cursor_id(&self) -> usize {
        self.base.active_cursor_id()
    }

    fn set_active_cursor(&self, cursor_id: usize) {
        self.base.set_active_cursor(cursor_id)
    }

    fn active_cursor(&self) -> Option<&Cursor<Raw, Buf, GAP_SIZE>> {
        self.base.active_cursor()
    }

    fn raw_ptr(&self) -> *const Raw {
        self.base.raw_ptr()
    }

    unsafe fn raw_mut_ptr(&self) -> *mut Raw {
        self.base.raw_mut_ptr()
    }

    fn raw_length(&self) -> usize {
        self.base.raw_length()
    }

    fn set_event_bridge(&self, bridge: Arc<dyn EventBridge>) {
        self.base.set_event_bridge(bridge)
    }

    fn event_bridge(&self) -> Option<Arc<dyn EventBridge>> {
        self.base.event_bridge()
    }

    fn name(&self) -> &[u8] {
        self.base.name()
    }

    fn name_mut(&mut self) -> &mut [u8] {
        self.base.name_mut()
    }

    fn flags(&self) -> u32 {
        self.base.flags()
    }

    fn flags_mut(&mut self) -> &mut u32 {
        self.base.flags_mut()
    }

    unsafe fn proximity(&self, cursor_id1: usize, cursor_id2: usize) -> bool {
        self.base.proximity(cursor_id1, cursor_id2)
    }

    unsafe fn new_shared_gap(&self, cursor_id: usize) {
        self.base.new_shared_gap(cursor_id)
    }

    unsafe fn shared_gap_index(&self, cursor_id: usize) -> usize {
        if let Some(val) = self.base.shared_gap_index(cursor_id) {
            val
        } else {
            usize::MAX
        }
    }

    unsafe fn is_cursor_in_shared_gap(&self, cursor_id: usize) -> bool {
        self.base.is_cursor_in_shared_gap(cursor_id)
    }

    fn history(&self) -> &UnsafeCell<History> {
        self.base.history()
    }

    fn history_current_index(&self) -> usize {
        self.base.history_current_index()
    }

    fn set_history_current_index(&self, index: usize) {
        self.base.set_history_current_index(index)
    }

    unsafe fn record_change(&self) {
        self.base.record_change()
    }

    fn register_action_observer(&self, observer: Arc<dyn UserActionObserver>) {
        self.base.register_action_observer(observer);
    }

    fn notify_action_observers(&self, event: &UserActionEvent) {
        self.base.notify_action_observers(event);
    }

    unsafe fn select_all(&mut self) {
        // Select all text by setting cursor to start and creating selection to end
        if let Some(cursor) = self.base.cursor_from_id_mut(0) {
            cursor.move_to(0, 0);
            // Set selection to cover entire content
            cursor.set_selection_anchor(0);
            let content_len = self.base.raw_length();
            cursor.set_selection_end(content_len);
        }
    }

    unsafe fn copy_selection(&mut self) {
        // Copy selection to clipboard (placeholder implementation)
        // This would typically interact with system clipboard APIs
    }

    unsafe fn paste_from_clipboard(&mut self) {
        // Paste from clipboard (placeholder implementation)
        // This would typically interact with system clipboard APIs
    }

    unsafe fn cut_selection(&mut self) {
        // Cut selection to clipboard (placeholder implementation)
        // This would typically interact with system clipboard APIs
    }
}

#[cfg(test)]
mod tests {
    use crate::edit::{BaseWritable, StreamWritable, TextEditablePut};

    fn creation() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        assert_eq!(writable.raw_length(), 0);
        assert_eq!(writable.v_space(), 0);
    }

    fn with_capacity() {
        let writable: BaseWritable<u32, u8, 4096> =
            BaseWritable::with_capacity("test", 1024);
        assert_eq!(writable.raw_length(), 0);
    }

    fn add_cursor() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let cursor_id = writable.append_cursor();
        assert!(cursor_id > 0);

        let cursor = writable.cursor_from_id(cursor_id);
        assert!(cursor.is_some());
    }

    fn add_multiple_cursors() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let id1 = writable.append_cursor();
        let id2 = writable.append_cursor();
        let id3 = writable.append_cursor();

        assert!(id1 < id2);
        assert!(id2 < id3);
        assert!(writable.cursor_from_id(id1).is_some());
        assert!(writable.cursor_from_id(id2).is_some());
        assert!(writable.cursor_from_id(id3).is_some());
    }

    fn remove_cursor() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let cursor_id = writable.append_cursor();

        let removed = writable.delete_cursor(cursor_id);
        assert!(removed);

        let cursor = writable.cursor_from_id(cursor_id);
        assert!(cursor.is_none());
    }

    fn remove_nonexistent_cursor() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let removed = writable.delete_cursor(999);
        assert!(!removed);
    }

    fn remove_cursor_zero() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let removed = writable.delete_cursor(0);
        assert!(!removed); // Cannot remove cursor 0
    }

    fn get_cursor() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let cursor_id = writable.append_cursor();

        let cursor = writable.cursor_from_id(cursor_id);
        assert!(cursor.is_some());
    }

    fn get_invalid_cursor() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let cursor = writable.cursor_from_id(999);
        assert!(cursor.is_none());
    }

    fn get_active_cursor_id() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let active_id = writable.active_cursor_id();
        assert_eq!(active_id, 0);
    }

    fn set_active_cursor() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let cursor_id = writable.append_cursor();

        writable.set_active_cursor(cursor_id);
        assert_eq!(writable.active_cursor_id(), cursor_id);
    }

    fn set_active_cursor_invalid() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        writable.set_active_cursor(999);
        // Should not panic, just set to 0
        assert_eq!(writable.active_cursor_id(), 0);
    }

    fn name() {
        let writable: BaseWritable<u32, u8, 4096> =
            BaseWritable::with_capacity("file", 1024);

        let name = writable.name();
        assert!(name.starts_with(b"file"));
    }

    fn flags() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        assert_eq!(writable.flags(), 0);
    }

    fn space_y() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        assert_eq!(writable.v_space(), 0);
    }

    fn raw_length() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        assert_eq!(writable.raw_length(), 0);
    }

    fn cursor_to_gap_initialization() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();

        unsafe {
            let gap_idx = writable.shared_gap_index(0).unwrap_or(usize::MAX);
            assert_eq!(gap_idx, usize::MAX); // Initially unassigned
        }
    }

    fn cursor_to_gap_after_add() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let cursor_id = writable.append_cursor();

        unsafe {
            let gap_idx = writable.shared_gap_index(cursor_id).unwrap_or(usize::MAX);
            assert_eq!(gap_idx, usize::MAX); // Still unassigned until shared gap is created
        }
    }

    fn is_cursor_in_shared_gap() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let cursor_id = writable.append_cursor();

        unsafe {
            assert!(!writable.is_cursor_in_shared_gap(cursor_id));
        }
    }

    fn cursors_in_proximity_same() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let cursor_id = writable.append_cursor();

        unsafe {
            // Same cursor should be considered in proximity
            assert!(writable.proximity(cursor_id, cursor_id));
        }
    }

    fn cursors_in_proximity_different() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let id1 = writable.append_cursor();
        let id2 = writable.append_cursor();

        unsafe {
            // Different cursors at same position should be in proximity
            assert!(writable.proximity(id1, id2));
        }
    }

    fn cursors_in_proximity_invalid() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();

        unsafe {
            assert!(!writable.proximity(0, 999));
            assert!(!writable.proximity(999, 0));
        }
    }

    fn find_or_create_shared_gap_no_proximity() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let cursor_id = writable.append_cursor();

        unsafe {
            writable.new_shared_gap(cursor_id);
            // Should not create shared gap if no other cursor in proximity
            assert!(!writable.is_cursor_in_shared_gap(cursor_id));
        }
    }

    fn max_cursor_count() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();

        // Add many cursors (should be limited by MAX_CURSOR_COUNT)
        for _ in 0..600 {
            writable.append_cursor();
        }

        // Should not panic, just stop adding after max
        let cursor = writable.cursor_from_id(511);
        assert!(cursor.is_some());
    }

    fn get_cursor_mut() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let cursor_id = writable.append_cursor();

        unsafe {
            let cursor = writable.cursor_from_id_mut(cursor_id);
            assert!(cursor.is_some());
        }
    }

    fn get_cursor_mut_invalid() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();

        unsafe {
            let cursor = writable.cursor_from_id_mut(999);
            assert!(cursor.is_none());
        }
    }

    fn event_bridge_none() {
        let writable: BaseWritable<u32, u8, 4096> = BaseWritable::new();
        let bridge = writable.event_bridge();
        assert!(bridge.is_none());
    }

    fn stream_writable_creation() {
        let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();
        assert_eq!(writable.base().raw_length(), 0);
    }

    fn stream_writable_with_capacity() {
        let writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);
        assert_eq!(writable.base().raw_length(), 0);
    }

    fn stream_writable_make_packed() {
        let packed = StreamWritable::<u32, u8, 4096>::make_packed(0xD, 0x41);
        assert_eq!(packed, (0xD << 28) | 0x41);
    }

    fn stream_writable_get_type() {
        let packed = (0xD << 28) | 0x41;
        let type_field = StreamWritable::<u32, u8, 4096>::packed(packed);
        assert_eq!(type_field, 0xD);
    }

    fn stream_writable_get_char() {
        let packed = (0xD << 28) | 0x41;
        let char_field = StreamWritable::<u32, u8, 4096>::char(packed);
        assert_eq!(char_field, 0x41);
    }

    fn stream_writable_put_character() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, false, 0); // Put 'A'
        }
    }

    fn stream_writable_put_newline() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x0A, false, 0); // Put newline
            assert_eq!(writable.base().v_space(), 1);
        }
    }

    fn stream_writable_put_multiple_characters() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            for i in 0..10 {
                writable.put(0x41 + (i as u32), false, 0);
            }
        }
    }

    fn stream_writable_put_with_points_new() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, true, 0); // Put with points_new = true
            let cursor = writable.base().cursor_from_id(0);
            assert!(cursor.is_some());
        }
    }

    fn stream_writable_put_invalid_cursor() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x41, false, 999); // Invalid cursor ID
            // Should not panic
        }
    }

    fn stream_writable_base_access() {
        let writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();
        let base = writable.base();
        assert_eq!(base.raw_length(), 0);
    }

    fn stream_writable_base_mut_access() {
        let mut writable: StreamWritable<u32, u8, 4096> = StreamWritable::new();
        let base = writable.base_mut();
        assert_eq!(base.raw_length(), 0);
    }

    fn stream_writable_packed_value_roundtrip() {
        let type_field = 0xD;
        let char_field = 0x123456;

        let packed = StreamWritable::<u32, u8, 4096>::make_packed(type_field, char_field);
        let extracted_type = StreamWritable::<u32, u8, 4096>::packed(packed);
        let extracted_char = StreamWritable::<u32, u8, 4096>::char(packed);

        assert_eq!(extracted_type, type_field);
        assert_eq!(extracted_char, char_field);
    }

    fn stream_writable_type_mask() {
        let packed = StreamWritable::<u32, u8, 4096>::make_packed(0xF, 0);
        let type_field = StreamWritable::<u32, u8, 4096>::packed(packed);
        assert_eq!(type_field, 0xF);
    }

    fn stream_writable_char_mask() {
        let packed = StreamWritable::<u32, u8, 4096>::make_packed(0, 0xFFFFFF);
        let char_field = StreamWritable::<u32, u8, 4096>::char(packed);
        assert_eq!(char_field, 0xFFFFFF);
    }

    fn stream_writable_id_merge() {
        let id_merge = StreamWritable::<u32, u8, 4096>::T_MERGE;
        assert_eq!(id_merge, 0xD);
    }

    fn stream_writable_id_common() {
        let id_common = StreamWritable::<u32, u8, 4096>::T_COMMON;
        assert_eq!(id_common, 0xE);
    }

    fn stream_writable_put_shared_not_in_shared_gap() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            // Cursor not in shared gap, should use regular gap handling
            writable.put(0x41, false, 0);
            assert!(!writable.base().is_cursor_in_shared_gap(0));
        }
    }

    fn stream_writable_multiple_newlines() {
        let mut writable: StreamWritable<u32, u8, 4096> =
            StreamWritable::with_capacity("test", 1024);

        unsafe {
            writable.put(0x0A, false, 0);
            writable.put(0x0A, false, 0);
            writable.put(0x0A, false, 0);
            assert_eq!(writable.base().v_space(), 3);
        }
    }

    fn stream_writable_shift_type() {
        let shift_type = StreamWritable::<u32, u8, 4096>::SHIFT_TYPE;
        assert_eq!(shift_type, 28);
    }

    fn stream_writable_shift_char() {
        let shift_char = StreamWritable::<u32, u8, 4096>::SHIFT_CHAR;
        assert_eq!(shift_char, 0);
    }

    fn stream_writable_type_mask_constant() {
        let type_mask = StreamWritable::<u32, u8, 4096>::TYPE_MASK;
        assert_eq!(type_mask, 0xF);
    }

    fn stream_writable_char_mask_constant() {
        let char_mask = StreamWritable::<u32, u8, 4096>::CHAR_MASK;
        assert_eq!(char_mask, 0xFFFFFF);
    }
}
