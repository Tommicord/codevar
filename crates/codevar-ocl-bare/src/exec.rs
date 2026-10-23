//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Kernel invocation ABI: work-item context, argument marshaling and the
//! work-group claim loop every device back end shares.
//!
//! # Kernel ABI
//!
//! A kernel is an [`KernelFn`] — `unsafe extern "C" fn(*mut WorkItem)`.
//! The [`WorkItem`] is `#[repr(C)]` and starts with six `[u32; 3]` id
//! vectors (72 bytes), so a hand-written Thumb kernel that indexes the
//! structure directly agrees with Rust on every field offset. Arguments are
//! reached through `arg_base`, which points at the launch's argument block:
//! one 16-byte slot per argument, buffers as native pointers, scalars in the
//! first bytes of their slot.
//!
//! The contract itself belongs to `codevar-ocl-abi` — the crate the
//! compiler side and this driver share, so generated code and the driver
//! cannot drift apart. This module re-exports it ([`WorkItem`],
//! [`KernelFn`], [`MAX_ARGS`], [`MAX_PARAMS`], [`ARG_SLOT`]) and adds the
//! parts that are the driver's own business: the [`ArgBlock`] the launch
//! marshals into, the [`Launch`] descriptor and the claim loop.
//!
//! # Claim loop
//!
//! Work-groups are distributed by an atomic claim counter: each core
//! (submitting core, SMP workers, the remote side of an AMP ring, the
//! service of an FPGA descriptor queue) fetches the next group index,
//! decodes it into 3-D group coordinates, runs every work-item of the group
//! and posts progress. The loop needs no central scheduler and no locks,
//! which is why the same code runs on one core (QEMU's `mps2-an385`) and on
//! a real multi-core part without a `cfg`.

use core::ffi::c_void;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::error::{Error, Result};
use crate::trampoline::Trampoline;

/// The kernel ABI, owned by `codevar-ocl-abi` and shared with the
/// compiler side.
///
/// The re-export keeps the driver's own paths — `codevar_ocl_bare::exec::WorkItem`
/// and friends — working while there is exactly one definition of the
/// contract in the workspace.
pub use codevar_ocl_abi::{ARG_SLOT, KernelFn, MAX_ARGS, MAX_PARAMS, WorkItem};

/// Handle to the SPIR-V entry point a program-backed kernel executes.
///
/// A null handle ([`SpirvBinding::NONE`]) marks a native (registry or
/// accelerator-opcode) kernel, whose entry point is `Launch::kernel`; a
/// non-null handle points at a program-owned entry descriptor inside the
/// `Box<SpirvModule>` the program keeps alive for as long as any of its
/// kernels exists (a kernel retains its program).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpirvBinding(*const c_void);

impl SpirvBinding {
    /// The empty binding of a native kernel.
    pub const NONE: Self = Self(core::ptr::null());

    /// Wraps an entry-point descriptor of a boxed [`crate::spirv::SpirvModule`].
    #[must_use]
    #[inline]
    pub(crate) const fn new(pointer: *const c_void) -> Self {
        Self(pointer)
    }

    /// Whether this launch runs a native kernel (no SPIR-V binding).
    #[must_use]
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0.is_null()
    }

    /// Whether this launch runs through the SPIR-V interpreter.
    #[must_use]
    #[inline]
    pub const fn is_some(self) -> bool {
        !self.0.is_null()
    }

    /// The raw entry-point pointer (null for native kernels).
    #[must_use]
    #[inline]
    pub(crate) const fn as_ptr(self) -> *const c_void {
        self.0
    }
}

// SAFETY: the pointer targets a program-owned entry descriptor that stays
// valid for as long as any kernel of that program can be enqueued (each
// kernel retains its program), and it is only ever dereferenced under that
// lifetime on cores that the driver's submit serialization orders.
unsafe impl Send for SpirvBinding {}
unsafe impl Sync for SpirvBinding {}

/// The marshaled argument block of a launch: one [`ARG_SLOT`]-byte slot
/// per argument, buffer arguments as native pointers, scalars copied
/// verbatim into the front of their slot.
///
/// `#[repr(C)]` (a single array field) so descriptor payloads shared with
/// an engine built from the same crate see a fixed layout.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ArgBlock {
    bytes: [u8; MAX_PARAMS],
}

impl core::fmt::Debug for ArgBlock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ArgBlock").finish_non_exhaustive()
    }
}

impl ArgBlock {
    /// An all-zero block; unset slots read as null pointers / zero scalars.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bytes: [0; MAX_PARAMS],
        }
    }

    /// Stores a buffer pointer in slot `index`.
    pub fn set_ptr(&mut self, index: usize, ptr: *mut c_void) {
        if index >= MAX_ARGS {
            return;
        }
        let start = index * ARG_SLOT;
        let raw = ptr as usize;
        let bytes = raw.to_ne_bytes();
        // Bounds checked against the fixed block above.
        self.bytes[start..start + bytes.len()].copy_from_slice(&bytes);
    }

    /// Stores a scalar of `len <= ARG_SLOT` bytes in slot `index`.
    ///
    /// Returns [`Error::TooManyArgs`] when the value does not fit, so an
    /// oversized `clSetKernelArg` can never silently truncate.
    pub fn set_scalar(&mut self, index: usize, value: &[u8]) -> Result<()> {
        if index >= MAX_ARGS {
            return Err(Error::TooManyArgs);
        }
        if value.len() > ARG_SLOT {
            return Err(Error::TooManyArgs);
        }
        let start = index * ARG_SLOT;
        self.bytes[start..start + value.len()].copy_from_slice(value);
        Ok(())
    }

    /// The raw bytes of slot `index`, for copying into a shared ring.
    #[must_use]
    #[inline]
    pub fn slot(&self, index: usize) -> &[u8; ARG_SLOT] {
        let start = index * ARG_SLOT;
        // `index < MAX_ARGS` keeps `start + ARG_SLOT <= MAX_PARAMS`.
        match self.bytes.get(start..start + ARG_SLOT) {
            Some(slice) => slice.try_into().unwrap_or(&[0; ARG_SLOT]),
            None => &[0; ARG_SLOT],
        }
    }

    /// The whole block, for publication into an AMP descriptor.
    #[must_use]
    #[inline]
    pub(crate) fn as_bytes(&self) -> &[u8; MAX_PARAMS] {
        &self.bytes
    }

    /// Rebuilds a block from bytes read out of a shared descriptor.
    #[must_use]
    #[inline]
    pub(crate) fn from_bytes(bytes: &[u8; MAX_PARAMS]) -> Self {
        Self { bytes: *bytes }
    }

    /// Pointer to the first byte, the value stored in `WorkItem::arg_base`.
    #[must_use]
    #[inline]
    pub fn as_ptr(&mut self) -> *mut u8 {
        self.bytes.as_mut_ptr()
    }
}

impl Default for ArgBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// A validated NDRange launch, ready for any device back end.
#[derive(Clone, Copy, Debug)]
pub struct Launch<'a> {
    /// Kernel entry point (native kernels; never called when `binding` is
    /// set — the SPIR-V interpreter runs the entry point instead).
    pub kernel: KernelFn,
    /// Optional native trampoline: when set, `run_group` dispatches through
    /// it instead of `kernel`. The trampoline is borrowed from the owning
    /// `Program` and lives at least as long as this launch.
    pub native: Option<&'a Trampoline>,
    /// The kernel name — what the accelerator's opcode dispatch looks up.
    /// Descriptor rings do not carry it (only the host-side dispatch reads
    /// it, before publishing), so a rebuilt descriptor launch reports `""`.
    pub name: &'a str,
    /// SPIR-V entry point for program-backed kernels, or
    /// [`SpirvBinding::NONE`] for native ones.
    pub binding: SpirvBinding,
    /// Marshalled arguments.
    pub args: &'a ArgBlock,
    /// Work dimension, `1..=3`.
    pub dims: u32,
    /// Global offset per axis (zeros when the caller passed none).
    pub offset: [u32; 3],
    /// Global work size per axis (axes `>= dims` are 1).
    pub global: [u32; 3],
    /// Local work size per axis (axes `>= dims` are 1).
    pub local: [u32; 3],
}

impl<'a> Launch<'a> {
    /// Number of work-groups per axis (axes `>= dims` are 1).
    #[must_use]
    #[inline]
    pub fn groups(&self) -> [u32; 3] {
        [
            self.global[0] / self.local[0],
            self.global[1] / self.local[1],
            self.global[2] / self.local[2],
        ]
    }

    /// Total number of work-groups.
    #[must_use]
    #[inline]
    pub fn group_count(&self) -> u32 {
        let g = self.groups();
        g[0].saturating_mul(g[1]).saturating_mul(g[2])
    }

    /// Total number of work-items.
    #[must_use]
    #[inline]
    pub fn item_count(&self) -> u64 {
        u64::from(self.global[0]) * u64::from(self.global[1]) * u64::from(self.global[2])
    }

    /// Local size made explicit on all three axes (axes `>= dims` are 1).
    #[must_use]
    #[inline]
    pub fn local_full(&self) -> [u32; 3] {
        let mut local = self.local;
        let mut axis = 0;
        while axis < 3 {
            if axis >= self.dims as usize {
                local[axis] = 1;
            }
            axis += 1;
        }
        local
    }
}

/// Validates the raw sizes of an NDRange against a device's limits.
///
/// Checks the rules the OpenCL entry point must report individually —
/// dimension range, zero global sizes, divisibility of global by local,
/// and the device work-group cap — and returns the normalized
/// (offset, global, local) triple on success.
///
/// # Errors
///
/// * [`Error::InvalidLaunch`] — dimension not in `1..=3`.
/// * [`Error::InvalidArgument`] — a zero global size, a zero/oversized
///   local size, global not divisible by local, or the product of the local
///   sizes exceeding `max_work_group_size`.
pub fn validate_sizes(
    dims: u32,
    offset: Option<&[usize]>,
    global: &[usize],
    local: Option<&[usize]>,
    max_work_group_size: u32,
) -> Result<([u32; 3], [u32; 3], [u32; 3])> {
    if !(1..=3).contains(&dims) {
        return Err(Error::InvalidLaunch);
    }
    let axis = dims as usize;
    let mut off = [0u32; 3];
    let mut glob = [1u32; 3];
    let mut loc = [1u32; 3];
    if let Some(offset) = offset {
        for (i, slot) in off.iter_mut().enumerate().take(axis) {
            *slot = u32::try_from(*offset.get(i).unwrap_or(&0)).map_err(|_| Error::InvalidArgument)?;
        }
    }
    for (i, slot) in glob.iter_mut().enumerate().take(axis) {
        let value = *global.get(i).ok_or(Error::InvalidArgument)?;
        if value == 0 {
            return Err(Error::InvalidArgument);
        }
        *slot = u32::try_from(value).map_err(|_| Error::InvalidArgument)?;
    }
    if let Some(local) = local {
        let mut product: u64 = 1;
        for (i, slot) in loc.iter_mut().enumerate().take(axis) {
            let value = *local.get(i).ok_or(Error::InvalidArgument)?;
            if value == 0 {
                return Err(Error::InvalidArgument);
            }
            let value = u32::try_from(value).map_err(|_| Error::InvalidArgument)?;
            if glob[i] % value != 0 {
                return Err(Error::InvalidArgument);
            }
            product = product.saturating_mul(u64::from(value));
            *slot = value;
        }
        if product > u64::from(max_work_group_size) {
            return Err(Error::InvalidArgument);
        }
    }
    Ok((off, glob, loc))
}

/// Calls the kernel for one work-item `item`, preferring the native
/// trampoline when one is loaded and falling back to `launch.kernel`.
///
/// # Safety
///
/// `item` must be a valid pointer to a fully initialized [`WorkItem`]
/// whose `arg_base` points at a marshalled argument block the kernel can
/// read.
unsafe fn call_kernel(launch: &Launch<'_>, item: &mut WorkItem) {
    if let Some(trampoline) = launch.native {
        // SAFETY: the trampoline's entry point is valid executable memory
        // following the kernel ABI; `item` is a complete WorkItem.
        trampoline.call(item as *mut WorkItem);
    } else {
        // SAFETY: `launch.kernel` is valid for the arguments in
        // `launch.args`, and `item` is fully initialized.
        unsafe { (launch.kernel)(item) };
    }
}

/// Executes every work-item of work-group `index` of `launch` on the
/// calling core.
///
/// The row-major index is decoded into 3-D group coordinates (`x` fastest,
/// matching the fill order a debugger sees in the group ids), then the
/// group's work-items run in local-id order. This is the single place the
/// work-group geometry is interpreted; both [`run_claimed`] and [`run_all`]
/// funnel through it.
///
/// # Errors
///
/// Propagates a per-work-group execution failure (today only the SPIR-V
/// interpreter can produce one — resource exhaustion in a work-item frame
/// or a kernel memory error).
///
/// # Safety
///
/// `launch.kernel` must be valid for the arguments in `launch.args`, and
/// `index` must be below `launch.group_count()`.
#[inline]
unsafe fn run_group(launch: &Launch<'_>, index: u32) -> Result<()> {
    let groups = launch.groups();
    let local = launch.local_full();
    let x = index % groups[0];
    let y = (index / groups[0]) % groups[1];
    let z = index / (groups[0] * groups[1]);
    if launch.binding.is_some() {
        return crate::spirv::execute_group(launch, [x, y, z]);
    }
    let mut item = WorkItem {
        global_id: [0; 3],
        local_id: [0; 3],
        group_id: [x, y, z],
        global_offset: launch.offset,
        num_groups: groups,
        local_size: local,
        arg_base: launch.args.bytes.as_ptr().cast_mut(),
    };
    let mut zl = 0u32;
    while zl < local[2] {
        let mut yl = 0u32;
        while yl < local[1] {
            let mut xl = 0u32;
            while xl < local[0] {
                item.local_id = [xl, yl, zl];
                item.global_id = [
                    launch.offset[0] + x * local[0] + xl,
                    launch.offset[1] + y * local[1] + yl,
                    launch.offset[2] + z * local[2] + zl,
                ];
                // SAFETY: `item` is a fully initialized WorkItem; the kernel
                // (trampoline or `launch.kernel`) is valid for this launch.
                unsafe { call_kernel(launch, &mut item) };
                xl += 1;
            }
            yl += 1;
        }
        zl += 1;
    }
    Ok(())
}

/// Runs work-groups of `launch` until the claim counter is exhausted,
/// posting each completed group to `done`.
///
/// This is *the* execution primitive: the submitting core runs it directly,
/// SMP workers run it against the shared job slot, the AMP remote runs it
/// over a descriptor, and the accelerator service runs it over an opcode
/// handler. One core alone is the degenerate case (it claims every group).
///
/// `total` must equal `launch.group_count()`; `claim` and `done` are reset
/// by the caller before the launch is published.
///
/// A work-group that fails does not stop the loop: the failure is recorded
/// as the returned error, the group still counts in `done`, and claiming
/// continues, so the `claim`/`done` protocol stays balanced for every
/// other claimant and the quiescence wait of the publisher cannot hang.
///
/// Returns the number of groups this caller executed.
///
/// # Errors
///
/// The first per-group execution failure observed by this caller.
///
/// # Safety
///
/// `launch.kernel` must be a valid kernel for the arguments in
/// `launch.args`, and no other caller may run a *different* launch against
/// the same `claim`/`done` pair concurrently.
pub unsafe fn run_claimed(launch: &Launch<'_>, claim: &AtomicU32, done: &AtomicU32) -> Result<u32> {
    let total = launch.group_count();
    let mut executed = 0u32;
    let mut first_error = None;
    loop {
        let index = claim.fetch_add(1, Ordering::Relaxed);
        if index >= total {
            break;
        }
        // SAFETY: the index comes from the claim counter below `total`.
        if let Err(error) = unsafe { run_group(launch, index) }
            && first_error.is_none()
        {
            first_error = Some(error);
        }
        executed += 1;
        // Release so the waiter that observes `done == total` also observes
        // every store the kernel made — including for a failed group, whose
        // completion the publisher still waits for.
        done.fetch_add(1, Ordering::Release);
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(executed),
    }
}

/// Executes every work-group of `launch` on the calling core, bypassing the
/// claim counter — the path a single-executor remote (AMP peer, accelerator
/// service) takes when it owns the whole NDRange.
///
/// # Errors
///
/// The first per-group execution failure; remaining groups are not run.
///
/// # Safety
///
/// Same kernel/argument contract as [`run_claimed`].
pub unsafe fn run_all(launch: &Launch<'_>) -> Result<()> {
    let total = launch.group_count();
    let mut index = 0u32;
    while index < total {
        // SAFETY: `index` walks `0..total` in lockstep with the count above.
        unsafe { run_group(launch, index)? };
        index += 1;
    }
    Ok(())
}

/// A reference kernel used by tests and the demo: `out[i] = a[i] + b[i]`.
///
/// Declared at module level so unit tests of the claim loop (and the
/// driver integration tests) do not need a program object.
///
/// # Safety
///
/// Reads the three buffer pointers of slots 0..=2 and the `u32` count of
/// slot 3; the caller must have marshaled them with matching sizes.
#[cfg(test)]
pub(crate) unsafe extern "C" fn vector_add(item: *mut WorkItem) {
    // SAFETY: the claim loop passes a live stack work-item.
    let item = unsafe { &mut *item };
    // SAFETY: slots 0..=2 are `f32` buffers of at least `count` elements and
    // slot 3 holds the element count, as marshaled by the demo/tests.
    unsafe {
        let a = item.arg_ptr(0).cast::<f32>();
        let b = item.arg_ptr(1).cast::<f32>();
        let out = item.arg_ptr(2).cast::<f32>();
        let count = item.arg_u32(3);
        if a.is_null() || b.is_null() || out.is_null() {
            return;
        }
        let global = item.global_id;
        if global[1] != 0 || global[2] != 0 || global[0] >= count {
            return;
        }
        let i = global[0] as usize;
        *out.add(i) = *a.add(i) + *b.add(i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn record_ids(item: *mut WorkItem) {
        // SAFETY: live item from the claim loop; the test's shared log is
        // only touched from the single-threaded test body.
        let item = unsafe { &*item };
        // SAFETY: slot 0 holds the log pointer the test marshaled.
        let dst = unsafe { item.arg_ptr(0).cast::<u32>() };
        if dst.is_null() {
            return;
        }
        // Row-major over the global extent with one `[gx, gy, gz]` triple
        // per work-item — the layout the assertions below read back.
        let width = item.local_size[0] * item.num_groups[0];
        let base = ((item.global_id[1] * width + item.global_id[0]) * 3) as usize;
        // SAFETY: `dst` points at the test's `16 * 3` (or `8 * 3`) log and
        // the id is inside the launched 4×4 (or 8×1) range.
        unsafe {
            *dst.add(base) = item.global_id[0];
            *dst.add(base + 1) = item.global_id[1];
            *dst.add(base + 2) = item.global_id[2];
        }
    }

    #[test]
    fn claim_loop_covers_every_work_item_once() {
        let mut args = ArgBlock::new();
        let mut log = [99u32; 16 * 3];
        args.set_ptr(0, log.as_mut_ptr().cast());
        let launch = Launch {
            kernel: record_ids,
            name: "record_ids",
            binding: SpirvBinding::NONE,
            native: None,
            args: &args,
            dims: 2,
            offset: [0, 0, 0],
            global: [4, 4, 1],
            local: [2, 2, 1],
        };
        let claim = AtomicU32::new(0);
        let done = AtomicU32::new(0);
        // SAFETY: kernel/args match this launch.
        let ran = unsafe { run_claimed(&launch, &claim, &done) }.expect("runs");
        assert_eq!(ran, 4);
        assert_eq!(done.load(Ordering::Acquire), 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = ((y * 4 + x) * 3) as usize;
                assert_eq!(log[i], x);
                assert_eq!(log[i + 1], y);
                assert_eq!(log[i + 2], 0);
            }
        }
    }

    #[test]
    fn run_all_matches_claimed() {
        let mut args = ArgBlock::new();
        let mut log = [7u32; 8 * 3];
        args.set_ptr(0, log.as_mut_ptr().cast());
        let launch = Launch {
            kernel: record_ids,
            name: "record_ids",
            binding: SpirvBinding::NONE,
            native: None,
            args: &args,
            dims: 1,
            offset: [0, 0, 0],
            global: [8, 1, 1],
            local: [4, 1, 1],
        };
        // SAFETY: kernel/args match this launch.
        unsafe { run_all(&launch) }.expect("runs");
        for i in 0..8u32 {
            assert_eq!(log[(i * 3) as usize], i);
        }
    }

    #[test]
    fn validate_sizes_rejects_bad_geometries() {
        assert!(validate_sizes(0, None, &[1], None, 64).is_err());
        assert!(validate_sizes(1, None, &[0], None, 64).is_err());
        // No local size: any non-zero global size is valid.
        assert!(validate_sizes(1, None, &[6], None, 64).is_ok());
        // Global not divisible by local.
        assert!(validate_sizes(1, None, &[6], Some(&[4]), 64).is_err());
        // Product of the local sizes beyond the device cap.
        assert!(validate_sizes(1, None, &[128], Some(&[64]), 63).is_err());
        assert!(validate_sizes(1, None, &[64], Some(&[16]), 64).is_ok());
    }

    #[test]
    fn scalar_and_pointer_slots_round_trip() {
        let mut block = ArgBlock::new();
        block.set_ptr(3, 0x1234 as *mut c_void);
        block
            .set_scalar(4, &[1, 2, 3, 4])
            .expect("scalar fits");
        assert!(block.set_scalar(5, &[0; ARG_SLOT + 1]).is_err());
        let item = WorkItem {
            global_id: [0; 3],
            local_id: [0; 3],
            group_id: [0; 3],
            global_offset: [0; 3],
            num_groups: [0; 3],
            local_size: [0; 3],
            arg_base: block.bytes.as_ptr().cast_mut(),
        };
        // SAFETY: slots hold the values written above.
        unsafe {
            assert_eq!(item.arg_ptr(3) as usize, 0x1234);
            assert_eq!(core::slice::from_raw_parts(item.slot(4), 4), &[1, 2, 3, 4]);
            assert!(item.arg_ptr(15).is_null());
            assert!(item.slot(MAX_ARGS).is_null());
        }
    }

    #[test]
    fn vector_add_writes_expected_sums() {
        let mut args = ArgBlock::new();
        let mut a = [1.0f32; 8];
        let mut b = [2.0f32; 8];
        let mut out = [0.0f32; 8];
        args.set_ptr(0, a.as_mut_ptr().cast());
        args.set_ptr(1, b.as_mut_ptr().cast());
        args.set_ptr(2, out.as_mut_ptr().cast());
        args.set_scalar(3, &8u32.to_ne_bytes())
            .expect("scalar fits");
        let launch = Launch {
            kernel: vector_add,
            name: "vector_add",
            binding: SpirvBinding::NONE,
            native: None,
            args: &args,
            dims: 1,
            offset: [0, 0, 0],
            global: [8, 1, 1],
            local: [4, 1, 1],
        };
        let claim = AtomicU32::new(0);
        let done = AtomicU32::new(0);
        // SAFETY: kernel/args match this launch.
        unsafe { run_claimed(&launch, &claim, &done) }.expect("runs");
        assert!(out.iter().all(|v| *v == 3.0));
        a[0] = 10.0;
        b[0] = 5.0;
        // The kernel reads these through the aliased pointers above, so
        // name the values here to keep the intent visible to the compiler.
        assert_eq!((a[0], b[0]), (10.0, 5.0));
        // SAFETY: same launch, fresh counters.
        let claim = AtomicU32::new(0);
        let done = AtomicU32::new(0);
        unsafe { run_claimed(&launch, &claim, &done) }.expect("runs");
        assert_eq!(out[0], 15.0);
    }
}
