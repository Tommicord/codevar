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

//! Accelerator device: an MMIO register block plus a descriptor ring, in
//! the style of the almaif OpenCL driver.
//!
//! Where the SMP device distributes work through a shared claim counter and
//! the AMP device through a peer-owned ring, this device talks to a *third*
//! shape of execution unit: an FPGA soft core (or a DSP, or any engine with
//! a register file) that the host commands through a small
//! `#[repr(C)]` register block — [`AccelRegs`] — and a ring of descriptors —
//! [`AccelRing`].
//!
//! # Protocol
//!
//! 1. The host resolves the kernel's **opcode** in its handler table (kernel
//!    name → `opcode` → [`AccelHandler`]), finds an `EMPTY` ring slot,
//!    writes the [`AccelDesc`], executes `dsb`, stores `PENDING` and
//!    advances `tail` — exactly the AMP publish half.
//! 2. The host then rings the doorbell: `control` is set to `START` and
//!    `sev` is issued. On a real board that store is the interrupt that
//!    starts the soft core; in [`ServiceMode::Cooperative`] it is only a
//!    hint, because [`Device::progress`] will run [`AccelDevice::service`]
//!    itself.
//! 3. The service side (accelerator firmware, or the host in cooperative
//!    mode) consumes `head..tail`, mirrors `RUNNING`/the opcode into the
//!    register block — what a debugger reading the MMIO window sees — and
//!    invokes the handler for the descriptor's opcode. The standard
//!    handler, [`default_accel_handler`], is just [`crate::exec::run_all`],
//!    but a board may register opcodes that touch real hardware instead.
//! 4. The completion is posted to the slot (`DONE`/`ERROR` plus the `CL_*`
//!    result code), `head` advances, the register block mirrors the result
//!    and the interrupt status is raised.
//! 5. [`Device::progress`] on the host reaps: it completes the command's
//!    event, releases the command reference and returns the slot to
//!    `EMPTY`.
//!
//! The register block is plain atomics so the same code runs against a
//! simulated engine in QEMU and (with the block placed at the peripheral
//! base address and the peer running [`AccelDevice::service`]) against real
//! hardware: everything is `#[repr(C)]`, word-granular and ordered by the
//! same `dsb`/`sev` discipline as the other two back ends.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};

use crate::device::{Device, DeviceInfo, ServiceMode};
use crate::error::{Error, Result};
use crate::exec::{ArgBlock, KernelFn, Launch, SpirvBinding, run_all};
use crate::objects::{Event, release_ref};
use crate::sys;

/// Number of descriptors in an [`AccelRing`] — the engine's in-flight depth.
pub const ACCEL_SLOTS: usize = 4;

/// Slot states of the ring.
const EMPTY: u32 = 0;
/// Published by the host, not yet executed by the engine.
const PENDING: u32 = 1;
/// Executed successfully; waiting for the host to reap it.
const DONE: u32 = 2;
/// Execution failed; waiting for the host to reap it.
const ERROR: u32 = 3;

/// `control` bit: ring the engine's doorbell (write-1-to-start).
const CTRL_START: u32 = 1 << 0;

/// `status` values mirrored in the register block.
const STATUS_IDLE: u32 = 0;
/// The engine accepted a descriptor and is executing it.
const STATUS_RUNNING: u32 = 1;
/// The engine finished its queue and the last descriptor succeeded.
const STATUS_DONE: u32 = 2;
/// The engine finished its queue and the last descriptor failed.
const STATUS_ERROR: u32 = 3;

/// `irq_status` bit: a descriptor completed (write-1-to-clear).
const IRQ_DONE: u32 = 1 << 0;

/// The MMIO register block of the accelerator.
///
/// `#[repr(C)]` and word-granular so a board can place it at the engine's
/// peripheral base address; hosted builds use the very same layout in
/// ordinary memory. Field access goes through atomics, which on ARM
/// Cortex-M peripheral space compiles to plain `ldr`/`str` — the ordering
/// discipline lives in the `dsb` calls around the protocol, not in the
/// individual accesses.
#[repr(C)]
pub struct AccelRegs {
    control: AtomicU32,
    status: AtomicU32,
    opcode: AtomicU32,
    irq_status: AtomicU32,
}

impl AccelRegs {
    /// A register block in its reset state (`IDLE`, no pending interrupt).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            control: AtomicU32::new(0),
            status: AtomicU32::new(STATUS_IDLE),
            opcode: AtomicU32::new(0),
            irq_status: AtomicU32::new(0),
        }
    }

    /// The `control` word (only the doorbell bit is defined).
    #[must_use]
    #[inline]
    pub fn control(&self) -> u32 {
        self.control.load(Ordering::Acquire)
    }

    /// The `status` word — one of the driver's `STATUS_*` values.
    #[must_use]
    #[inline]
    pub fn status(&self) -> u32 {
        self.status.load(Ordering::Acquire)
    }

    /// The opcode the engine is (or was last) executing.
    #[must_use]
    #[inline]
    pub fn opcode(&self) -> u32 {
        self.opcode.load(Ordering::Acquire)
    }

    /// The interrupt-status word (`IRQ_DONE` until cleared).
    #[must_use]
    #[inline]
    pub fn irq_status(&self) -> u32 {
        self.irq_status.load(Ordering::Acquire)
    }

    /// Raises the doorbell the engine waits on (host → engine).
    fn ring_doorbell(&self) {
        self.control.store(CTRL_START, Ordering::Release);
    }

    /// Clears the doorbell after the engine picked the work up.
    fn ack_doorbell(&self) {
        self.control.store(0, Ordering::Release);
    }

    /// Mirrors an execution state into the visible register window.
    #[inline]
    fn set_status(&self, status: u32) {
        self.status.store(status, Ordering::Release);
    }

    /// Mirrors the running opcode into the visible register window.
    #[inline]
    fn set_opcode(&self, opcode: u32) {
        self.opcode.store(opcode, Ordering::Release);
    }

    /// Raises the completion interrupt (engine → host).
    #[inline]
    fn raise_irq(&self) {
        self.irq_status
            .fetch_or(IRQ_DONE, Ordering::Release);
    }

    /// Clears the completion interrupt (host, when it reaps).
    #[inline]
    fn clear_irq(&self) {
        self.irq_status
            .fetch_and(!IRQ_DONE, Ordering::Release);
    }
}

impl Default for AccelRegs {
    fn default() -> Self {
        Self::new()
    }
}

/// One descriptor of an [`AccelRing`] — what the host publishes and the
/// engine consumes. Payload layout is fixed (`#[repr(C)]`) so both sides
/// agree without negotiation (a remote image compiled from the same crate
/// sees the same struct).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct AccelDesc {
    /// Handler-table index the engine dispatches on (`0` is reserved for
    /// the idle descriptor, so real opcodes start at 1).
    pub opcode: u32,
    /// Work dimension of the launch, `1..=3`.
    pub dims: u32,
    /// Entry point of the launch (the engine indexes its own handler table
    /// with `opcode`; the field is carried for diagnostics and for engines
    /// that dispatch by address instead).
    pub kernel: KernelFn,
    /// Global offset per axis.
    pub offset: [u32; 3],
    /// Global work size per axis.
    pub global: [u32; 3],
    /// Local work size per axis.
    pub local: [u32; 3],
    /// Marshaled argument block of the launch.
    pub args: ArgBlock,
}

impl AccelDesc {
    /// A descriptor that parks the engine (opcode 0, single no-op item).
    const fn dormant() -> Self {
        Self {
            opcode: 0,
            dims: 1,
            kernel: idle_kernel,
            offset: [0; 3],
            global: [1; 3],
            local: [1; 3],
            args: ArgBlock::new(),
        }
    }

    /// The [`Launch`] this descriptor describes.
    fn launch(&self) -> Launch<'_> {
        // Name and binding stay unset: the host-side opcode dispatch that
        // reads the name runs before publishing, and SPIR-V kernels never
        // enter the ring (`supports` rejects them).
        Launch {
            kernel: self.kernel,
            name: "",
            binding: SpirvBinding::NONE,
            native: None,
            args: &self.args,
            dims: self.dims,
            offset: self.offset,
            global: self.global,
            local: self.local,
        }
    }
}

/// Never-executed kernel stored in a dormant descriptor — a placeholder so
/// the static initial state holds a valid function pointer (a null function
/// pointer would be undefined to even materialize).
extern "C" fn idle_kernel(_item: *mut crate::exec::WorkItem) {}

/// One ring slot: status, completion result and the descriptor cell.
struct AccelSlot {
    status: AtomicU32,
    result: AtomicI32,
    desc: UnsafeCell<AccelDesc>,
}

impl AccelSlot {
    const fn new() -> Self {
        Self {
            status: AtomicU32::new(EMPTY),
            result: AtomicI32::new(sys::SUCCESS),
            desc: UnsafeCell::new(AccelDesc::dormant()),
        }
    }
}

// SAFETY: the descriptor is written by the host only while the status is
// `EMPTY` (exclusive), published with a release store of `PENDING`, read by
// the engine only after acquiring `PENDING`, and not rewritten until the
// host has reaped `DONE`/`ERROR` back to `EMPTY`. `result` is written by
// the engine before its `DONE`/`ERROR` release store and read by the host
// only after acquiring that store.
unsafe impl Sync for AccelSlot {}

/// The shared descriptor ring between host and accelerator.
pub struct AccelRing {
    /// Consumed up to here (engine-owned).
    head: AtomicU32,
    /// Published up to here (host-owned).
    tail: AtomicU32,
    slots: [AccelSlot; ACCEL_SLOTS],
}

// SAFETY: the indices are atomics ordered by the protocol on `AccelSlot`;
// the slots delegate to that same invariant.
unsafe impl Sync for AccelRing {}

impl AccelRing {
    /// An empty ring — typically a `static` so host and engine name the
    /// same address: `static RING: AccelRing = AccelRing::new();`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            slots: [const { AccelSlot::new() }; ACCEL_SLOTS],
        }
    }
}

impl Default for AccelRing {
    fn default() -> Self {
        Self::new()
    }
}

/// The execution entry of one opcode: what the service side runs for a
/// descriptor.
///
/// Handlers receive a fully formed [`Launch`] (arguments marshaled,
/// sizes validated by the enqueue path) and must complete it before
/// returning. [`default_accel_handler`] is the generic whole-NDRange
/// executor; board-specific opcodes may instead program DMA engines or
/// custom logic and then wait for it.
pub type AccelHandler = fn(&Launch<'_>) -> Result<()>;

/// The standard opcode handler: execute the whole NDRange on the calling
/// core through [`run_all`].
///
/// # Errors
///
/// The first per-work-group execution failure (a SPIR-V frame resource
/// error; native kernels cannot fail), reported through the descriptor's
/// result word.
pub fn default_accel_handler(launch: &Launch<'_>) -> Result<()> {
    // SAFETY: the enqueue path validated the launch and the argument block
    // against the kernel definition before publication.
    unsafe { run_all(launch) }
}

/// An accelerator device bound to an [`AccelRegs`] block and an
/// [`AccelRing`].
///
/// In [`ServiceMode::Cooperative`] (QEMU, host tests) the driver services
/// the ring itself from [`Device::progress`] — the engine firmware's job
/// done inline, with every descriptor byte and register transition
/// identical to the real thing. In [`ServiceMode::Remote`] the host only
/// rings the doorbell and reaps; the engine is expected to run
/// [`AccelDevice::service`] from its interrupt handler or main loop.
pub struct AccelDevice {
    info: DeviceInfo,
    regs: &'static AccelRegs,
    ring: &'static AccelRing,
    handlers: &'static [(&'static str, AccelHandler)],
    mode: ServiceMode,
    submit: spin::Mutex<()>,
    inflight: spin::Mutex<Vec<(u32, &'static Event)>>,
}

impl AccelDevice {
    /// Creates a device over caller-owned register and ring blocks — the
    /// blocks a board's linker script placed at the engine's addresses.
    ///
    /// `handlers` maps kernel names to (opcode, handler): the position in
    /// the table plus one *is* the opcode the descriptor carries, so two
    /// tables built in the same order on host and engine agree.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] when the in-flight list cannot allocate.
    pub fn with_parts(
        info: DeviceInfo,
        regs: &'static AccelRegs,
        ring: &'static AccelRing,
        handlers: &'static [(&'static str, AccelHandler)],
        mode: ServiceMode,
    ) -> Result<&'static Self> {
        let mut inflight = Vec::new();
        inflight
            .try_reserve(ACCEL_SLOTS)
            .map_err(|_| Error::OutOfMemory)?;
        Ok(Box::leak(Box::new(Self {
            info,
            regs,
            ring,
            handlers,
            mode,
            submit: spin::Mutex::new(()),
            inflight: spin::Mutex::new(inflight),
        })))
    }

    /// Creates a device with driver-allocated register and ring blocks, for
    /// images where the engine is simulated (or located out of band).
    ///
    /// # Errors
    ///
    /// See [`AccelDevice::with_parts`].
    pub fn simulated(
        info: DeviceInfo,
        handlers: &'static [(&'static str, AccelHandler)],
        mode: ServiceMode,
    ) -> Result<&'static Self> {
        let regs: &'static AccelRegs = Box::leak(Box::new(AccelRegs::new()));
        let ring: &'static AccelRing = Box::leak(Box::new(AccelRing::new()));
        Self::with_parts(info, regs, ring, handlers, mode)
    }

    /// The register block this device commands.
    #[must_use]
    pub fn regs(&self) -> &'static AccelRegs {
        self.regs
    }

    /// The descriptor ring this device publishes to.
    #[must_use]
    pub fn ring(&self) -> &'static AccelRing {
        self.ring
    }

    /// The handler table (name → handler) this device dispatches on.
    #[must_use]
    pub fn handlers(&self) -> &[(&'static str, AccelHandler)] {
        self.handlers
    }

    /// The opcode registered for `name`, if the device supports it.
    #[must_use]
    pub fn opcode_of(&self, name: &str) -> Option<u32> {
        self.handlers
            .iter()
            .position(|(kernel, _)| *kernel == name)
            .map(|position| position as u32 + 1)
    }

    /// The engine's service loop: consume every `PENDING` slot, dispatch
    /// its opcode and post completion.
    ///
    /// Accelerator firmware calls this from its doorbell interrupt (or its
    /// main loop around a `wfe`); cooperative mode calls it from
    /// [`Device::progress`].
    pub fn service(&self) {
        self.regs.ack_doorbell();
        loop {
            let head = self.ring.head.load(Ordering::Acquire);
            let tail = self.ring.tail.load(Ordering::Acquire);
            if head == tail {
                self.regs.set_status(STATUS_IDLE);
                break;
            }
            let index = (head % ACCEL_SLOTS as u32) as usize;
            let slot = &self.ring.slots[index];
            if slot.status.load(Ordering::Acquire) != PENDING {
                break;
            }
            // SAFETY: we acquired `PENDING`, so the descriptor is immutable
            // until we store `DONE`; copying it ends that borrow before any
            // host rewrite can be ordered after our `head` advance.
            let desc = unsafe { *slot.desc.get() };
            self.regs.set_status(STATUS_RUNNING);
            self.regs.set_opcode(desc.opcode);
            let launch = desc.launch();
            let result = self.invoke(desc.opcode, &launch);
            codevar_target::dsb();
            match result {
                Ok(()) => {
                    slot.result.store(sys::SUCCESS, Ordering::Release);
                    slot.status.store(DONE, Ordering::Release);
                    self.regs.set_status(STATUS_DONE);
                }
                Err(error) => {
                    slot.result
                        .store(error.to_cl(), Ordering::Release);
                    slot.status.store(ERROR, Ordering::Release);
                    self.regs.set_status(STATUS_ERROR);
                }
            }
            self.ring
                .head
                .store(head.wrapping_add(1), Ordering::Release);
            self.regs.raise_irq();
            codevar_target::dsb();
            codevar_target::sev();
        }
    }

    /// Dispatches the handler of `opcode` (`0` is the dormant no-op).
    fn invoke(&self, opcode: u32, launch: &Launch<'_>) -> Result<()> {
        match opcode
            .checked_sub(1)
            .and_then(|index| self.handlers.get(index as usize))
        {
            Some((_, handler)) => handler(launch),
            None => Ok(()),
        }
    }

    /// Publishes one launch; the slot index joins the in-flight list.
    fn dispatch(&'static self, launch: &Launch<'_>, event: &'static Event) -> Result<()> {
        event.set_status(sys::SUBMITTED);
        let opcode = self
            .opcode_of(launch.name)
            .ok_or(Error::UnsupportedOnDevice);
        let opcode = match opcode {
            Ok(opcode) => opcode,
            Err(error) => {
                release_ref(event);
                return Err(error);
            }
        };
        let _serial = self.submit.lock();
        // Reserve before publishing, past the publish point no failure is
        // survivable (the descriptor is already visible to the engine), and
        // the submit lock keeps the reservation from being consumed by a
        // concurrent dispatch.
        if self.inflight.lock().try_reserve(1).is_err() {
            release_ref(event);
            return Err(Error::OutOfMemory);
        }
        let published = self.publish(opcode, launch, event);
        let index = match published {
            Ok(index) => index,
            Err(error) => {
                release_ref(event);
                return Err(error);
            }
        };
        event.set_status(sys::RUNNING);
        // The reservation above makes this push infallible.
        self.inflight.lock().push((index, event));
        // A cooperative engine may have finished it already; the eager reap
        // is best effort — failures surface through later `progress` calls,
        // and the command stays owned by the reaper either way.
        let _ = self.progress();
        Ok(())
    }

    /// Writes the descriptor, rings the doorbell and returns the slot.
    fn publish(&self, opcode: u32, launch: &Launch<'_>, _event: &'static Event) -> Result<u32> {
        let mut attempts = 0u32;
        loop {
            let tail = self.ring.tail.load(Ordering::Acquire);
            let slot_index = tail % ACCEL_SLOTS as u32;
            if self.ring.slots[slot_index as usize]
                .status
                .load(Ordering::Acquire)
                == EMPTY
            {
                // SAFETY: `EMPTY` means the host has exclusive use of the
                // slot (see `AccelSlot`'s Sync contract).
                let desc = unsafe { &mut *self.ring.slots[slot_index as usize].desc.get() };
                desc.opcode = opcode;
                desc.dims = launch.dims;
                desc.kernel = launch.kernel;
                desc.offset = launch.offset;
                desc.global = launch.global;
                desc.local = launch.local;
                desc.args = *launch.args;
                codevar_target::dsb();
                self.ring.slots[slot_index as usize]
                    .status
                    .store(PENDING, Ordering::Release);
                self.ring
                    .tail
                    .store(tail.wrapping_add(1), Ordering::Release);
                codevar_target::dsb();
                // The doorbell write *is* the engine's interrupt on a real
                // board; `sev` wakes a peer core running the service loop.
                self.regs.ring_doorbell();
                codevar_target::sev();
                return Ok(slot_index);
            }
            attempts += 1;
            if attempts > ACCEL_SLOTS as u32 * 4 {
                return Err(Error::OutOfResources);
            }
            // A failing progress here leaves the event unpublished;
            // `dispatch` releases it when this error surfaces.
            self.progress()?;
            codevar_target::wfe();
        }
    }
}

impl Device for AccelDevice {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn submit(&self, launch: &Launch<'_>, event: &'static Event) -> Result<()> {
        // `dispatch` needs `&'static self`; the handle API only produces
        // leaked devices, so recover the static lifetime from the address.
        // SAFETY: every `AccelDevice` reachable through the platform was
        // leaked by its constructor and therefore lives for `'static`.
        let this = unsafe {
            core::ptr::from_ref(self)
                .cast::<AccelDevice>()
                .as_ref()
        }
        .ok_or(Error::InvalidDevice)?;
        this.dispatch(launch, event)
    }

    fn progress(&self) -> Result<()> {
        if self.mode == ServiceMode::Cooperative {
            self.service();
        }
        let mut inflight = self.inflight.lock();
        let mut i = 0;
        while i < inflight.len() {
            let (index, event) = inflight[i];
            let slot = &self.ring.slots[index as usize];
            let status = slot.status.load(Ordering::Acquire);
            if status == DONE || status == ERROR {
                let result = slot.result.load(Ordering::Acquire);
                event.finish(if status == DONE {
                    sys::COMPLETE
                } else if result < 0 {
                    result
                } else {
                    sys::OUT_OF_RESOURCES
                });
                // Free the slot for the next publish before releasing the
                // event (which may be the last reference and free it).
                slot.status.store(EMPTY, Ordering::Release);
                self.regs.clear_irq();
                release_ref(event);
                inflight.swap_remove(i);
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    fn inflight(&self) -> usize {
        self.inflight.lock().len()
    }

    fn supports(&self, def: &crate::registry::KernelDef) -> bool {
        // SPIR-V kernels carry a binding pointer the register-block
        // protocol has no field for — they run on the CPU/RING devices.
        def.binding.is_none() && self.opcode_of(&def.name).is_some()
    }
}
