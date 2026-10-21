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

//! The kernel invocation ABI.
//!
//! One contract is shared by the two sides of a kernel launch: the
//! driver, which owns the work-item state and the argument block, and
//! the native code generator, which emits machine code that reads them.
//! The contract lives in its own crate so neither side can drift: the
//! driver includes this definition, and the code generator addresses the
//! very same fields by the offsets published here.
//!
//! # The call
//!
//! A kernel is an `unsafe extern "C" fn(*mut WorkItem)`.  The pointer
//! names a per-work-item state block that lives on the launching core's
//! stack for the duration of the call.  The generated code reads the
//! work-item identifiers from it and reaches the kernel's arguments
//! through the block's `arg_base` field.
//!
//! # The argument block
//!
//! The block holds one fixed-size slot per kernel argument:
//! [`ARG_SLOT`] bytes each, [`MAX_ARGS`] slots in total.  A buffer
//! argument is stored as a native pointer in the first word of its
//! slot; a scalar argument is stored in the first bytes of its slot in
//! its natural representation.  The slot size is fixed so an argument
//! can be addressed with a constant offset from the block base, which
//! is what lets the generated code read its arguments without a
//! descriptor table.
//!
//! # Example
//!
//! ```
//! use codevar_ocl_abi::{ARG_SLOT, MAX_ARGS, MAX_PARAMS, WorkItem};
//!
//! assert_eq!(MAX_ARGS * ARG_SLOT, MAX_PARAMS);
//! assert_eq!(core::mem::size_of::<WorkItem>(), 72 + core::mem::size_of::<*mut u8>());
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

use core::ffi::c_void;

/// The invocation type of a kernel — the single ABI every device back
/// end calls into.
///
/// # Safety
///
/// Implementations receive a fully initialized [`WorkItem`] whose
/// `arg_base` points at a marshalled argument block with at least the
/// slots the kernel declares; they must not escape the work item or its
/// argument block beyond the call.
pub type KernelFn = unsafe extern "C" fn(*mut WorkItem);

/// Maximum number of kernel arguments the driver can marshal.
pub const MAX_ARGS: usize = 16;

/// Size in bytes of the whole argument block.
pub const MAX_PARAMS: usize = 256;

/// Size in bytes of one argument slot.
pub const ARG_SLOT: usize = MAX_PARAMS / MAX_ARGS;

/// Per-work-item state handed to a kernel.
///
/// `#[repr(C)]`, documented field by field because generated code and
/// hand-written assembly kernels address these offsets directly —
/// changing the order breaks them.
#[repr(C)]
#[derive(Debug)]
pub struct WorkItem {
    /// Global id of this work-item, offset 0.
    pub global_id: [u32; 3],
    /// Local (in-group) id, offset 12.
    pub local_id: [u32; 3],
    /// Group id of the containing work-group, offset 24.
    pub group_id: [u32; 3],
    /// Global offset of the launch, offset 36.
    pub global_offset: [u32; 3],
    /// Number of groups in each dimension, offset 48.
    pub num_groups: [u32; 3],
    /// Local size of the launch, offset 60.
    pub local_size: [u32; 3],
    /// Argument block base, offset 72 (native pointer width).
    pub arg_base: *mut u8,
}

impl WorkItem {
    /// Pointer to argument slot `index`.
    ///
    /// Returns null for `index >= MAX_ARGS` so a kernel probing an
    /// unused slot fails closed instead of stepping off the block.
    #[must_use]
    #[inline]
    pub fn slot(&self, index: usize) -> *mut u8 {
        if index >= MAX_ARGS || self.arg_base.is_null() {
            return core::ptr::null_mut();
        }
        // The caller (the claim loop) guarantees `arg_base` is a live
        // `MAX_PARAMS`-byte block, and `index` is in range.
        unsafe { self.arg_base.add(index * ARG_SLOT) }
    }

    /// Reads the native pointer stored in argument slot `index`.
    ///
    /// # Safety
    ///
    /// Slot `index` must hold a buffer argument marshalled by the
    /// driver (its first `size_of::<usize>()` bytes are a valid
    /// pointer).
    #[must_use]
    #[inline]
    pub unsafe fn arg_ptr(&self, index: usize) -> *mut u8 {
        let slot = self.slot(index);
        if slot.is_null() {
            return core::ptr::null_mut();
        }
        // SAFETY: slot is inside the driver-maintained block; pointers are
        // written as native `usize` in the first word of the slot.
        let mut bytes = [0u8; core::mem::size_of::<usize>()];
        unsafe {
            core::ptr::copy_nonoverlapping(slot, bytes.as_mut_ptr(), core::mem::size_of::<usize>());
        }
        usize::from_ne_bytes(bytes) as *mut u8
    }

    /// Reads the `u32` scalar in argument slot `index` (zero when the
    /// slot is shorter than four bytes).
    ///
    /// # Safety
    ///
    /// Slot `index` must belong to this work-item's argument block.
    #[must_use]
    #[inline]
    pub unsafe fn arg_u32(&self, index: usize) -> u32 {
        let slot = self.slot(index);
        if slot.is_null() {
            return 0;
        }
        // SAFETY: `slot` is in the live argument block; a short read is
        // impossible because the block is a flat `[u8]` of `MAX_PARAMS`.
        unsafe {
            let mut bytes = [0u8; 4];
            core::ptr::copy_nonoverlapping(slot, bytes.as_mut_ptr(), 4);
            u32::from_ne_bytes(bytes)
        }
    }
}

/// A field of the per-work-item state a kernel can read.
///
/// The variants name the fields of [`WorkItem`] in the order the
/// driver lays them out.  The code generator maps a work-item query
/// onto one of these variants and then onto the byte offset published
/// by [`WorkItemField::offset`], so the generated code and the driver
/// agree on every access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkItemField {
    /// The global id of the work-item.
    GlobalId,
    /// The in-group id of the work-item.
    LocalId,
    /// The id of the containing work-group.
    GroupId,
    /// The global offset of the launch.
    GlobalOffset,
    /// The number of work-groups in the launch.
    NumGroups,
    /// The local size of the launch.
    LocalSize,
}

impl WorkItemField {
    /// Every field, in layout order.
    pub const ALL: [Self; 6] = [
        Self::GlobalId,
        Self::LocalId,
        Self::GroupId,
        Self::GlobalOffset,
        Self::NumGroups,
        Self::LocalSize,
    ];

    /// Byte offset of the field inside [`WorkItem`].
    #[must_use]
    #[inline]
    pub const fn offset(self) -> u32 {
        match self {
            Self::GlobalId => 0,
            Self::LocalId => 12,
            Self::GroupId => 24,
            Self::GlobalOffset => 36,
            Self::NumGroups => 48,
            Self::LocalSize => 60,
        }
    }

    /// Number of components the field holds.
    #[must_use]
    #[inline]
    pub const fn components(self) -> u32 {
        3
    }

    /// Size in bytes of one component of the field.
    #[must_use]
    #[inline]
    pub const fn component_size(self) -> u32 {
        4
    }
}

/// Byte offset of the argument-block pointer inside [`WorkItem`].
///
/// The pointer follows the six id vectors, so the offset is
/// `6 * 3 * 4 = 72` on every target whose pointer is at least four
/// bytes wide.
pub const ARG_BASE_OFFSET: u32 = 72;

/// A null work-item pointer, used by the driver to mark an unbound
/// kernel.
pub const NULL_WORK_ITEM: *mut c_void = core::ptr::null_mut();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_tile_the_argument_block() {
        assert_eq!(MAX_ARGS * ARG_SLOT, MAX_PARAMS);
    }

    #[test]
    fn fields_are_ordered_by_offset() {
        let mut previous = 0;
        for field in WorkItemField::ALL {
            assert!(field.offset() >= previous);
            previous = field.offset();
        }
        assert_eq!(previous, 60);
    }

    #[test]
    fn work_item_layout_matches_the_published_offsets() {
        let item = WorkItem {
            global_id: [0; 3],
            local_id: [0; 3],
            group_id: [0; 3],
            global_offset: [0; 3],
            num_groups: [0; 3],
            local_size: [0; 3],
            arg_base: core::ptr::null_mut(),
        };
        let base = &item as *const WorkItem as usize;
        let global = &item.global_id as *const [u32; 3] as usize;
        assert_eq!((global - base) as u32, WorkItemField::GlobalId.offset());
        let local = &item.local_id as *const [u32; 3] as usize;
        assert_eq!((local - base) as u32, WorkItemField::LocalId.offset());
        let groups = &item.num_groups as *const [u32; 3] as usize;
        assert_eq!((groups - base) as u32, WorkItemField::NumGroups.offset());
        let size = &item.local_size as *const [u32; 3] as usize;
        assert_eq!((size - base) as u32, WorkItemField::LocalSize.offset());
        let args = &item.arg_base as *const *mut u8 as usize;
        assert_eq!((args - base) as u32, ARG_BASE_OFFSET);
    }
}
