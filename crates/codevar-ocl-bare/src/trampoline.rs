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

//! Native machine code execution: jump to a `.bin` blob produced by
//! `codevar-oclc`.
//!
//! A [`Trampoline`] wraps position-independent machine code in
//! executable memory and provides a uniform [`Trampoline::call`] that
//! follows the kernel ABI of `codevar-ocl-abi`: one `*mut WorkItem`
//! argument in the register the target uses for the first parameter.
//!
//! This is the **fast path** for kernels the driver can find a `.bin`
//! for. Kernels whose target back end is missing, or whose `.bin` is
//! empty, fall through to the SPIR-V interpreter -- the **fallback
//! path** that [`crate::spirv`] implements -- so the driver always
//! executes a kernel it was asked to launch.
//!
//! # Backends
//!
//! | Target                            | Memory                              |
//! |-----------------------------------|-------------------------------------|
//! | `unix` (Linux/macOS, x86-64/aarch64) | `mmap`'d anonymous RWX page       |
//! | `thumbv7m-none-eabi`              | slot of a static RAM pool (Cortex-M3 has no W^X) |
//! | any other                         | none -- [`Trampoline::new`] returns [`TrampolineError::Unsupported`] |
//!
//! A `Trampoline` owns its executable region and may be moved into
//! another thread; the region is read-only after construction, so
//! `Send + Sync` are unconditionally safe.

#![allow(unsafe_code)]

use core::ffi::c_void;
use core::fmt;
use core::ptr::NonNull;

use crate::exec::{KernelFn, WorkItem};

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
/// Constructed from a byte slice of position-independent code (the
/// output of `codevar-ocl-tc`) and a `u32` entry-point offset. The
/// `Drop` implementation releases the executable region.
///
/// The struct is deliberately not `Copy`: the region it owns is a
/// system-level allocation, and silently cloning it would either leak
/// the resource or require an `Arc`, which would defeat the point of
/// the type -- a thin, zero-overhead wrapper around an executable
/// address.
pub struct Trampoline {
    /// Raw address of the entry point: the kernel ABI says the first
    /// argument is in the first parameter register of the target
    /// (`rdi` on x86-64, `x0` on aarch64, `r0` on Thumb-2); the
    /// `Trampoline::call` inline asm sets it.
    entry: usize,
    /// Length of the loaded code in bytes -- kept so a future
    /// `release`/`rebind` API can sanity-check the offset, and so
    /// debugging is easier.
    length: usize,
    /// Hosted Unix: the mmap region (host pointer, length). `None` on
    /// targets whose backend does not allocate.
    #[cfg(unix)]
    region: Option<(NonNull<c_void>, usize)>,
    /// Bare-metal: the slot index in the static RAM pool. `None` on
    /// targets whose backend does not use the pool.
    #[cfg(target_os = "none")]
    pool_slot: Option<u16>,
}

impl fmt::Debug for Trampoline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dbg = f.debug_struct("Trampoline");
        dbg.field("entry", &format_args!("{:#x}", self.entry))
            .field("length", &self.length);
        #[cfg(unix)]
        if let Some((region, length)) = self.region {
            dbg.field(
                "region",
                &format_args!("{:#x} ({} bytes)", region.as_ptr() as usize, length),
            );
        }
        #[cfg(target_os = "none")]
        if let Some(slot) = self.pool_slot {
            dbg.field("pool_slot", &slot);
        }
        dbg.finish()
    }
}

impl Trampoline {
    /// Loads `bytes` into executable memory and prepares the entry
    /// point at `bytes[entry_offset..]`.
    ///
    /// The entry point follows the kernel ABI of `codevar-ocl-abi`:
    /// a single `*mut WorkItem` argument in the first parameter
    /// register. The blob is *position-independent* -- the host target
    /// has no base address to fix up -- so the same bytes can be loaded
    /// into any executable region of the right alignment.
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
    #[allow(clippy::needless_return)]
    pub fn new(bytes: &[u8], entry_offset: u32) -> Result<Self, TrampolineError> {
        let length = bytes.len();
        if entry_offset as usize >= length {
            return Err(TrampolineError::BadEntryOffset {
                offset: entry_offset,
                length,
            });
        }
        // Backend dispatch. The cfg branches are mutually exclusive
        // (unix and target_os = "none" cannot both match), and the
        // final `not(any(...))` arm catches every other target.
        #[cfg(all(unix, target_arch = "x86_64"))]
        {
            return new_unix_x86_64(bytes, entry_offset);
        }
        #[cfg(all(unix, target_arch = "aarch64"))]
        {
            return new_unix_aarch64(bytes, entry_offset);
        }
        #[cfg(all(target_os = "none", target_arch = "arm"))]
        {
            return new_bare_arm(bytes, entry_offset);
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

    /// Dispatches the kernel with `item` as the per-work-item state.
    ///
    /// The function is **safe** to call: every invariant the
    /// trampoline relies on (the entry point is `PROT_EXEC` memory,
    /// the `Trampoline` owns that memory, the kernel ABI matches
    /// `codevar-ocl-abi`) is established by [`Trampoline::new`] and
    /// the kernel is *read-only* after that. The caller still has
    /// the usual obligation to pass a live `*mut WorkItem` — see the
    /// **Safety** section below.
    ///
    /// # Safety
    ///
    /// * `item` must be a valid pointer to a fully initialized
    ///   [`WorkItem`] whose `arg_base` points at a marshalled argument
    ///   block the kernel can read.
    /// * The blob this trampoline was built from must follow the
    ///   kernel ABI of `codevar-ocl-abi` (its entry point consumes the
    ///   `*mut WorkItem` and nothing else).
    /// * The kernel must not outlive its trampoline or `item`.
    pub fn call(&self, item: *mut WorkItem) {
        // SAFETY: see the contract on the public function. The
        // `as` casts widen `entry` (a `usize` that already carries
        // any Thumb bit) to the inline asm's `in(reg)` operand type.
        let entry = self.entry;
        #[cfg(all(unix, target_arch = "x86_64"))]
        {
            // SAFETY: `entry` is a valid PROT_EXEC function address;
            // `item` is the caller's valid `*mut WorkItem`. The
            // `sysv64` ABI clobber covers every caller-saved
            // register the kernel is allowed to trash.
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
            // SAFETY: same -- `entry` is a valid PROT_EXEC function
            // address; the `systemv` ABI clobber covers x0-x18 plus
            // the caller's v0-v7 vectors, all of which the kernel is
            // allowed to trash.
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
            // SAFETY: `entry` is a Thumb function address (LSB set on
            // every trampoline this crate produces), `item` is the
            // kernel's first argument in `r0` per AAPCS. `lr` push/pop
            // around `blx` preserves the caller's return address
            // across the kernel's own `bx lr` / `pop {pc}` epilogue.
            unsafe {
                core::arch::asm!(
                    "push {{lr}}",
                    "mov r0, {item}",
                    "blx {entry}",
                    "pop {{pc}}",
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

    /// Returns the entry point as a [`KernelFn`] function pointer.
    ///
    /// Available only on hosted Unix, where the mmap'd region sits at
    /// a stable address that can be transmuted directly. On bare metal
    /// the trampoline is reachable only through [`Trampoline::call`].
    ///
    /// This is what lets the existing claim loop dispatch a kernel
    /// through a trampoline without an extra wrapper on the hot path:
    /// `Launch::kernel` is set to the trampoline's entry point and
    /// the loop's `unsafe { (launch.kernel)(&mut item) }` reaches the
    /// machine code directly.
    #[cfg(unix)]
    #[must_use]
    pub fn kernel_fn(&self) -> KernelFn {
        // SAFETY: `entry` is the address of valid PROT_EXEC code that
        // follows the kernel ABI (`unsafe extern "C" fn(*mut
        // WorkItem)`); a transmute from `usize` to that function
        // pointer type is therefore sound.
        unsafe { core::mem::transmute::<usize, KernelFn>(self.entry) }
    }
}

#[cfg(unix)]
impl Drop for Trampoline {
    fn drop(&mut self) {
        if let Some((region, length)) = self.region {
            // SAFETY: `region` was returned by `mmap` for `length`
            // bytes and the `Trampoline` owns it; `Drop` runs exactly
            // once because the type is not `Clone`.
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
            // SAFETY: `slot` was reserved by `acquire_slot` at
            // construction and the `Trampoline` owns the reservation
            // until `Drop`. The pool is single-threaded per kernel,
            // matching the rest of the driver.
            unsafe {
                pool::release_slot(slot);
            }
        }
    }
}

/// Allocates an anonymous RWX page and copies `bytes` into it on
/// `unix` + `x86_64`.
///
/// Returns the populated [`Trampoline`] so the caller (the generic
/// `Trampoline::new`) can stay target-agnostic.
#[cfg(all(unix, target_arch = "x86_64"))]
fn new_unix_x86_64(bytes: &[u8], entry_offset: u32) -> Result<Trampoline, TrampolineError> {
    let length = bytes.len();
    // SAFETY: `mmap` with `MAP_ANONYMOUS | MAP_PRIVATE`, no `fd`, and
    // `offset == 0` is the standard anonymous-mapping call; the kernel
    // either returns a valid pointer or `MAP_FAILED`.
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
    // SAFETY: `mmap` returned a valid pointer to `length` writable
    // bytes; we copy exactly that many bytes from the caller's slice.
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

/// Same as [`new_unix_x86_64`], but for `unix` + `aarch64`. Kept
/// separate so the inline asm above can name the right register.
#[cfg(all(unix, target_arch = "aarch64"))]
fn new_unix_aarch64(bytes: &[u8], entry_offset: u32) -> Result<Trampoline, TrampolineError> {
    let length = bytes.len();
    // SAFETY: see `new_unix_x86_64`.
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
    // SAFETY: see `new_unix_x86_64`.
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

/// Bare-metal ARM (Thumb-2) trampoline: copies `bytes` into the next
/// free slot of a static RAM pool, sets the entry's Thumb bit.
///
/// The pool is sized by [`pool::SLOT_COUNT`] x [`pool::SLOT_SIZE`],
/// fixed at link time. Each successful [`Trampoline::new`] reserves
/// one slot until its `Drop` runs; an over-sized blob returns
/// [`TrampolineError::TooLarge`], a full pool returns the same
/// variant (both are "we cannot hold this code right now").
#[cfg(all(target_os = "none", target_arch = "arm"))]
fn new_bare_arm(bytes: &[u8], entry_offset: u32) -> Result<Trampoline, TrampolineError> {
    let length = bytes.len();
    // SAFETY: `acquire_slot` reserves the slot atomically and
    // `slot_address` returns its base. The reserved slot stays live
    // until `Drop` calls `release_slot`.
    let slot = unsafe { pool::acquire_slot(length)? };
    let base = unsafe { pool::slot_address(slot) };
    // SAFETY: `slot` is reserved for at least `length` bytes; `bytes`
    // has exactly `length` readable bytes.
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), base.cast::<u8>(), length);
    }
    // Cortex-M3 (and every Thumb-only core) requires the LSB of a
    // function address to be `1` to switch the core into Thumb mode;
    // `blx {entry}` reads that bit and switches.
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
///
/// Cortex-M3 has no W^X enforcement, so RAM is always executable;
/// this module just needs an aligned, fixed region. Slot allocation
/// is a single atomic increment of a "next free" counter; release
/// resets the slot's "in use" flag so the counter can wrap back to
/// released slots. The counter is monotonic -- it does not reuse
/// slots -- which is fine for the embedded profile where a board
/// holds a handful of kernels for its lifetime.
///
/// Splitting it from the rest of the trampoline code keeps the
/// `Drop` impl above readable and the slot bookkeeping testable in
/// isolation.
#[cfg(all(target_os = "none", target_arch = "arm"))]
mod pool {
    use core::ffi::c_void;
    use core::sync::atomic::{AtomicU16, Ordering};

    use super::TrampolineError;

    /// Bytes per slot: 4 KiB matches the Cortex-M3 flash page and
    /// comfortably holds any single kernel `codevar-oclc` produces
    /// for the embedded subset.
    pub(super) const SLOT_SIZE: usize = 4096;
    /// Maximum number of simultaneously resident trampolines: 16
    /// gives the pool 64 KiB, the same order as the TLSF heap a
    /// bare-metal bring-up typically configures.
    pub(super) const SLOT_COUNT: usize = 16;

    /// One slot: aligned to 4 bytes so Thumb-2 instructions (which
    /// must be 2-byte aligned, with many needing 4) never cross a
    /// slot boundary they did not ask for.
    #[repr(C, align(4))]
    struct Slot([u8; SLOT_SIZE]);

    /// The pool itself. `static` so the linker places it in RAM and
    /// the address is stable across calls.
    static POOL: [Slot; SLOT_COUNT] = [Slot([0; SLOT_SIZE]); SLOT_COUNT];

    /// Monotonic "next free" slot index. Once the pool is exhausted,
    /// new trampolines fail with [`TrampolineError::TooLarge`].
    static NEXT_SLOT: AtomicU16 = AtomicU16::new(0);

    /// Per-slot in-use flag. The release path sets it to zero; the
    /// counter still monotonically increments, but no caller ever
    /// touches a released slot.
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

    /// Reserves the next free slot and returns its index.
    ///
    /// # Errors
    ///
    /// * [`TrampolineError::TooLarge`] when `length > SLOT_SIZE` or
    ///   every slot is already reserved.
    ///
    /// # Safety
    ///
    /// Must run in a context where the trampoline's single-host
    /// rule holds: one host thread constructs the trampolines of a
    /// given program at a time. The pool counter is atomic, so the
    /// rule is about kernel ownership, not slot allocation.
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

    /// Releases `slot`, marking it free for any future inspection
    /// (the counter does not wrap, so the slot is not reused).
    ///
    /// # Safety
    ///
    /// `slot` must have been returned by [`acquire_slot`] and not
    /// already released.
    pub(super) unsafe fn release_slot(slot: u16) {
        SLOTS_USED[slot as usize].store(0, Ordering::Release);
    }

    /// Returns the address of `slot` as a `*mut c_void`.
    ///
    /// # Safety
    ///
    /// `slot` must have been returned by [`acquire_slot`] and not
    /// yet released.
    pub(super) unsafe fn slot_address(slot: u16) -> *mut c_void {
        // SAFETY: `slot` is in range (`acquire_slot` checked) and
        // `POOL` outlives every trampoline that holds a slot.
        unsafe {
            (&raw const POOL[slot as usize])
                .cast_mut()
                .cast::<c_void>()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial host-target native function: writes `42` into
    /// `item.global_id[0]` and returns. Used to exercise the
    /// `Trampoline::call` and `kernel_fn` paths on x86-64.
    #[cfg(all(unix, target_arch = "x86_64"))]
    const WRITE_42: &[u8] = &[
        0xC7, 0x07, 0x2A, 0x00, 0x00, 0x00, // mov dword ptr [rdi], 42
        0xC3, // ret
    ];

    /// Same kernel, aarch64 host: `mov w8, #42; str w8, [x0]; ret`.
    #[cfg(all(unix, target_arch = "aarch64"))]
    const WRITE_42: &[u8] = &[
        0x00, 0x00, 0x80, 0x52, // mov w8, #0x2a
        0x08, 0x00, 0x00, 0xB9, // str w8, [x0]
        0xC0, 0x03, 0x5F, 0xD6, // ret
    ];

    /// Bare-metal Thumb-2: the same kernel, addressing `r0` and
    /// returning via `bx lr`.
    #[cfg(all(target_os = "none", target_arch = "arm"))]
    const WRITE_42: &[u8] = &[
        0x2A, 0x20, // mov r0, #42
        0x70, 0x47, // bx lr
    ];

    fn blank_item() -> WorkItem {
        WorkItem {
            global_id: [0; 3],
            local_id: [0; 3],
            group_id: [0; 3],
            global_offset: [0; 3],
            num_groups: [0; 3],
            local_size: [0; 3],
            arg_base: core::ptr::null_mut(),
        }
    }

    #[test]
    fn new_rejects_an_entry_past_the_end() {
        let error = Trampoline::new(&[], 0).expect_err("empty code, offset 0");
        assert!(matches!(
            error,
            TrampolineError::BadEntryOffset { offset: 0, length: 0 }
        ));
        let error = Trampoline::new(&[0x90], 4).expect_err("1-byte code, offset 4");
        assert!(matches!(
            error,
            TrampolineError::BadEntryOffset { offset: 4, length: 1 }
        ));
    }

    /// On targets without a backend, [`Trampoline::new`] returns
    /// [`TrampolineError::Unsupported`].
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

    /// The host backend can load and call a hand-written kernel.
    #[cfg(any(
        all(unix, target_arch = "x86_64"),
        all(unix, target_arch = "aarch64"),
        all(target_os = "none", target_arch = "arm"),
    ))]
    #[test]
    fn host_can_load_and_call_a_kernel() {
        let trampoline = Trampoline::new(WRITE_42, 0).expect("load the kernel");
        let mut item = blank_item();
        trampoline.call(&mut item);
        assert_eq!(item.global_id[0], 42);
        assert_eq!(item.global_id[1], 0);
    }

    /// `kernel_fn` on hosted Unix lets the existing claim loop call
    /// the trampoline through a function pointer.
    #[cfg(unix)]
    #[test]
    fn kernel_fn_is_a_function_pointer_on_hosted() {
        let trampoline = Trampoline::new(WRITE_42, 0).expect("load the kernel");
        let kernel = trampoline.kernel_fn();
        let mut item = blank_item();
        // SAFETY: see `host_can_load_and_call_a_kernel`.
        unsafe { kernel(&mut item) };
        assert_eq!(item.global_id[0], 42);
    }

    /// The entry offset selects the entry point inside the blob.
    #[cfg(any(
        all(unix, target_arch = "x86_64"),
        all(unix, target_arch = "aarch64"),
        all(target_os = "none", target_arch = "arm"),
    ))]
    #[test]
    fn entry_offset_skips_prefix_bytes() {
        let mut code = vec![0x90; 4]; // nop sled
        code.extend_from_slice(WRITE_42);
        let trampoline = Trampoline::new(&code, 4).expect("load with offset");
        let mut item = blank_item();
        trampoline.call(&mut item);
        assert_eq!(item.global_id[0], 42);
    }
}
