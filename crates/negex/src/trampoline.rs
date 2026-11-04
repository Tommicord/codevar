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
//! permissions and limitations under the license.

//! A custom trampoline for executing Negex compiled machine code.
//!
//! Mirrors [`codevar-ocl-bare::Trampoline`]: wraps position-independent
//! machine code in executable memory and provides a uniform call that
//! follows the Negex ABI (a single `*mut MatchItem` argument).

use core::ffi::c_void;
use core::fmt;
use core::ptr::NonNull;

use codevar_negex_abi::MatchItem;

/// Why a [`Trampoline`] could not be created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrampolineError {
    /// `entry_offset` is past the end of `bytes` at construction time.
    BadEntryOffset {
        /// The offset the caller asked for.
        offset: u32,
        /// The length of the bytes the caller supplied.
        length: usize,
    },
    /// Allocating executable memory failed (hosted Unix only).
    AllocationFailed,
    /// The requested code did not fit the static pool (bare-metal only).
    TooLarge,
    /// No native trampoline backend is wired up for this target.
    Unsupported,
}

impl fmt::Display for TrampolineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadEntryOffset { offset, length } => {
                write!(f, "entry offset {offset} is past the {length}-byte code")
            }
            Self::AllocationFailed => f.write_str("executable memory allocation failed"),
            Self::TooLarge => f.write_str("machine code is too large for the static pool"),
            Self::Unsupported => f.write_str("no native trampoline backend for this target"),
        }
    }
}

/// One callable native machine-code blob, sitting in executable memory.
///
/// Constructed from a byte slice of position-independent code and a
/// `u32` entry-point offset. The entry point follows the Negex ABI: a
/// single `*mut MatchItem` argument in the first parameter register.
pub struct Trampoline {
    entry: usize,
    length: usize,
    #[cfg(unix)]
    region: Option<(NonNull<c_void>, usize)>,
    #[cfg(target_os = "none")]
    pool_slot: Option<u16>,
}

impl fmt::Debug for Trampoline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dbg = f.debug_struct("Trampoline");
        dbg.field("entry", &format_args!("{:#x}", self.entry))
            .field("length", &self.length);
        dbg.finish()
    }
}

impl Trampoline {
    /// Loads `bytes` into executable memory and prepares the entry point
    /// at `bytes[entry_offset..]`.
    ///
    /// # Errors
    ///
    /// * [`TrampolineError::BadEntryOffset`] -- `entry_offset` is past
    ///   the end of `bytes`.
    /// * [`TrampolineError::AllocationFailed`] -- the hosted `mmap`
    ///   call returned `MAP_FAILED`.
    /// * [`TrampolineError::TooLarge`] -- bare-metal pool overflow or
    ///   oversized blob.
    /// * [`TrampolineError::Unsupported`] -- no backend for this target.
    pub fn new(bytes: &[u8], entry_offset: u32) -> Result<Self, TrampolineError> {
        let length = bytes.len();
        if entry_offset as usize >= length {
            return Err(TrampolineError::BadEntryOffset {
                offset: entry_offset,
                length,
            });
        }

        #[cfg(all(unix, target_arch = "x86_64"))]
        {
            new_unix_x86_64(bytes, entry_offset)
        }
        #[cfg(all(unix, target_arch = "aarch64"))]
        {
            new_unix_aarch64(bytes, entry_offset)
        }
        #[cfg(all(target_os = "none", target_arch = "arm"))]
        {
            new_bare_arm(bytes, entry_offset)
        }
        #[cfg(not(any(
            all(unix, target_arch = "x86_64"),
            all(unix, target_arch = "aarch64"),
            all(target_os = "none", target_arch = "arm"),
        )))]
        {
            let _ = (bytes, entry_offset);
            Err(TrampolineError::Unsupported)
        }
    }

    /// Dispatches the match with `item` as the per-match state.
    ///
    /// # Safety
    ///
    /// * `item` must be a valid pointer to a fully initialized
    ///   [`MatchItem`].
    /// * The blob this trampoline was built from must follow the Negex
    ///   ABI (its entry point consumes the `*mut MatchItem` and nothing
    ///   else).
    pub fn call(&self, item: *mut MatchItem) {
        let entry = self.entry;
        #[cfg(all(unix, target_arch = "x86_64"))]
        {
            unsafe {
                core::arch::asm!(
                    "call rax",
                    in("rax") entry,
                    in("rdi") item,
                    clobber_abi("sysv64"),
                );
            }
        }
        #[cfg(all(unix, target_arch = "aarch64"))]
        {
            unsafe {
                core::arch::asm!(
                    "blr x9",
                    in("x9") entry,
                    in("x0") item,
                    clobber_abi("systemv"),
                );
            }
        }
        #[cfg(all(target_os = "none", target_arch = "arm"))]
        {
            unsafe {
                core::arch::asm!(
                    "push {lr}",
                    "mov r0, {item}",
                    "blx {entry}",
                    "pop {pc}",
                    item = in(reg) item,
                    entry = in(reg) entry,
                    out("r1") _,
                    out("r2") _,
                    out("r3") _,
                    out("r12") _,
                    options(noreturn),
                );
            }
        }
        #[cfg(not(any(
            all(unix, target_arch = "x86_64"),
            all(unix, target_arch = "aarch64"),
            all(target_os = "none", target_arch = "arm"),
        )))]
        {
            let _ = item;
        }
    }
}

#[cfg(unix)]
impl Drop for Trampoline {
    fn drop(&mut self) {
        if let Some((region, length)) = self.region {
            unsafe {
                libc::munmap(region.as_ptr().cast(), length);
            }
        }
    }
}

#[cfg(target_os = "none")]
impl Drop for Trampoline {
    fn drop(&mut self) {
        if let Some(slot) = self.pool_slot {
            unsafe {
                crate::trampoline::pool::release_slot(slot);
            }
        }
    }
}

#[cfg(all(unix, target_arch = "x86_64"))]
fn new_unix_x86_64(bytes: &[u8], entry_offset: u32) -> Result<Trampoline, TrampolineError> {
    let length = bytes.len();
    let ptr = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    if ptr.cast::<c_void>() == libc::MAP_FAILED {
        return Err(TrampolineError::AllocationFailed);
    }
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), length);
    }
    let entry = ptr as usize + entry_offset as usize;
    let region = NonNull::new(ptr.cast::<c_void>()).ok_or(TrampolineError::AllocationFailed)?;
    Ok(Trampoline {
        entry,
        length,
        #[cfg(unix)]
        region: Some((region, length)),
        #[cfg(target_os = "none")]
        pool_slot: None,
    })
}

#[cfg(all(unix, target_arch = "aarch64"))]
fn new_unix_aarch64(bytes: &[u8], entry_offset: u32) -> Result<Trampoline, TrampolineError> {
    let length = bytes.len();
    let ptr = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    if ptr.cast::<c_void>() == libc::MAP_FAILED {
        return Err(TrampolineError::AllocationFailed);
    }
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), length);
    }
    let entry = ptr as usize + entry_offset as usize;
    let region = NonNull::new(ptr.cast::<c_void>()).ok_or(TrampolineError::AllocationFailed)?;
    Ok(Trampoline {
        entry,
        length,
        #[cfg(unix)]
        region: Some((region, length)),
        #[cfg(target_os = "none")]
        pool_slot: None,
    })
}

#[cfg(all(target_os = "none", target_arch = "arm"))]
fn new_bare_arm(bytes: &[u8], entry_offset: u32) -> Result<Trampoline, TrampolineError> {
    let length = bytes.len();
    let slot = unsafe { crate::trampoline::pool::acquire_slot(length)? };
    let base = unsafe { crate::trampoline::pool::slot_address(slot) };
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), base.cast::<u8>(), length);
    }
    let entry = (base as usize + entry_offset as usize) | 1;
    Ok(Trampoline {
        entry,
        length,
        #[cfg(unix)]
        region: None,
        #[cfg(target_os = "none")]
        pool_slot: Some(slot),
    })
}

/// The bare-metal RAM pool of [`Trampoline`] slots.
#[cfg(all(target_os = "none", target_arch = "arm"))]
mod pool {
    use core::ffi::c_void;
    use core::sync::atomic::{AtomicU16, Ordering};

    use super::TrampolineError;

    pub(super) const SLOT_SIZE: usize = 4096;
    pub(super) const SLOT_COUNT: usize = 16;

    #[repr(C, align(4))]
    struct Slot([u8; SLOT_SIZE]);

    static POOL: [Slot; SLOT_COUNT] = [Slot([0; SLOT_SIZE]); SLOT_COUNT];

    static NEXT_SLOT: AtomicU16 = AtomicU16::new(0);

    static SLOTS_USED: [AtomicU16; SLOT_COUNT] = [
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
        AtomicU16::new(0),
    ];

    pub(super) unsafe fn acquire_slot(length: usize) -> Result<u16, TrampolineError> {
        if length > SLOT_SIZE {
            return Err(TrampolineError::TooLarge);
        }
        let slot = NEXT_SLOT.fetch_add(1, Ordering::AcqRel);
        if slot as usize >= SLOT_COUNT {
            return Err(TrampolineError::TooLarge);
        }
        SLOTS_USED[slot as usize].store(1, Ordering::Release);
        Ok(slot)
    }

    pub(super) unsafe fn release_slot(slot: u16) {
        SLOTS_USED[slot as usize].store(0, Ordering::Release);
    }

    pub(super) unsafe fn slot_address(slot: u16) -> *mut c_void {
        (&raw const POOL[slot as usize])
            .cast_mut()
            .cast::<c_void>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_entry_past_end() {
        let error = Trampoline::new(&[], 0).expect_err("empty code");
        assert!(matches!(error, TrampolineError::BadEntryOffset { .. }));
    }

    #[cfg(not(any(
        all(unix, target_arch = "x86_64"),
        all(unix, target_arch = "aarch64"),
        all(target_os = "none", target_arch = "arm"),
    )))]
    #[test]
    fn unsupported_targets_return_unsupported() {
        let result = Trampoline::new(&[0xC3], 0);
        assert!(matches!(result, Err(TrampolineError::Unsupported)));
    }
}
