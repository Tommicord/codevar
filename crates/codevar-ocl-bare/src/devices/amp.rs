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

//! AMP device: a shared-memory descriptor ring to a peer processor.
//!
//! Asymmetric multiprocessing — a host core talking to a DSP, a radio
//! core or a second M-class image — shares RAM but not caches worth
//! mentioning, so the transport is the classic OpenAMP-style ring: one
//! producer, one consumer, atomic indices, `dsb` before the doorbell.
//!
//! # Ring protocol
//!
//! * The host finds a slot whose `status` is `EMPTY`, fills the payload,
//!   executes `dsb`, stores `PENDING` (release), advances `tail` and
//!   `sev`s the peer.
//! * [`AmpDevice::service`] — the peer's firmware loop, or the host in
//!   [`ServiceMode::Cooperative`] — consumes `head..tail`, runs the whole
//!   NDRange locally through [`crate::exec::run_all`], stores `DONE` and
//!   advances `head` with a `dsb`-ordered write.
//! * [`Device::progress`] reaps finished slots: it completes the command's
//!   event, releases it and returns the slot to `EMPTY`.
//!
//! The payload only moves through atomics' release/acquire pairs; between
//! a `PENDING` store and its observation the payload bytes are immutable,
//! which is what makes the plain-field copy legal on both sides.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};

use crate::device::{Device, DeviceInfo};
use crate::error::{Error, Result};
use crate::exec::{ArgBlock, KernelFn, Launch, MAX_PARAMS, SpirvBinding, run_all};
use crate::objects::{Event, release_ref};
use crate::sys;

/// Number of descriptors in an [`AmpRing`] — the queue's in-flight depth.
pub const AMP_SLOTS: usize = 8;

/// Slot states of the ring.
const EMPTY: u32 = 0;
/// Published by the host, not yet executed by the peer.
const PENDING: u32 = 1;
/// Executed successfully; waiting for the host to reap it.
const DONE: u32 = 2;
/// Execution failed; waiting for the host to reap it.
const ERROR: u32 = 3;

/// The immutable payload of one ring slot (`#[repr(C)]`: fixed layout for
/// a remote image built from the same crate).
#[derive(Clone, Copy)]
#[repr(C)]
struct AmpPayload {
    kernel: KernelFn,
    binding: SpirvBinding,
    dims: u32,
    offset: [u32; 3],
    global: [u32; 3],
    local: [u32; 3],
    args: [u8; MAX_PARAMS],
}

/// One slot: a status word, the execution result and its payload cell.
struct AmpSlot {
    status: AtomicU32,
    result: AtomicI32,
    payload: UnsafeCell<AmpPayload>,
}

impl AmpSlot {
    const fn new() -> Self {
        Self {
            status: AtomicU32::new(EMPTY),
            result: AtomicI32::new(sys::SUCCESS),
            payload: UnsafeCell::new(AmpPayload {
                kernel: idle_kernel,
                binding: SpirvBinding::NONE,
                dims: 1,
                offset: [0; 3],
                global: [1; 3],
                local: [1; 3],
                args: [0; MAX_PARAMS],
            }),
        }
    }
}

// SAFETY: the payload is written by the host only while `status` is
// `EMPTY` (exclusive), published with a release store of `PENDING`, read
// by the peer only after acquiring `PENDING`, and not rewritten until the
// host has reaped `DONE`/`ERROR` back to `EMPTY`. `result` is written by
// the peer before its `DONE`/`ERROR` release store and read by the host
// only after acquiring that store.
unsafe impl Sync for AmpSlot {}

/// Placeholder kernel for a never-published slot (a null function pointer
/// would be undefined to materialize).
extern "C" fn idle_kernel(_item: *mut crate::exec::WorkItem) {}

/// The shared descriptor ring between host and peer.
pub struct AmpRing {
    /// Consumed up to here (peer-owned).
    head: AtomicU32,
    /// Published up to here (host-owned).
    tail: AtomicU32,
    slots: [AmpSlot; AMP_SLOTS],
}

// SAFETY: indices are atomics with the protocol above; slots delegate to
// `AmpSlot`'s own invariant.
unsafe impl Sync for AmpRing {}

impl AmpRing {
    /// An empty ring — typically a `static` so host and peer name the same
    /// address: `static RING: AmpRing = AmpRing::new();`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            slots: [const { AmpSlot::new() }; AMP_SLOTS],
        }
    }
}

impl Default for AmpRing {
    fn default() -> Self {
        Self::new()
    }
}

/// An AMP device bound to an [`AmpRing`].
///
/// In [`ServiceMode::Cooperative`] (QEMU's single core, host tests) the
/// driver services the ring itself while waiting — the peer firmware's job
/// done inline, with every descriptor byte and status transition identical
/// to the real thing. In [`ServiceMode::Remote`] the host only polls; the
/// peer is expected to run [`AmpDevice::service`] on its own core.
pub struct AmpDevice {
    info: DeviceInfo,
    ring: &'static AmpRing,
    mode: crate::device::ServiceMode,
    submit: spin::Mutex<()>,
    inflight: spin::Mutex<Vec<(u32, &'static Event)>>,
}

impl AmpDevice {
    /// Creates a device over a caller-owned ring — the ring a remote core
    /// was pointed at (its address is the out-of-band detail of the board
    /// bring-up, e.g. a linker section both images share).
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] when the in-flight list cannot allocate.
    pub fn with_ring(
        info: DeviceInfo,
        ring: &'static AmpRing,
        mode: crate::device::ServiceMode,
    ) -> Result<&'static Self> {
        let mut inflight = Vec::new();
        inflight
            .try_reserve(AMP_SLOTS)
            .map_err(|_| Error::OutOfMemory)?;
        Ok(Box::leak(Box::new(Self {
            info,
            ring,
            mode,
            submit: spin::Mutex::new(()),
            inflight: spin::Mutex::new(inflight),
        })))
    }

    /// Creates a device over a driver-allocated ring, for images where the
    /// peer is located out of band (or for cooperative simulation).
    ///
    /// # Errors
    ///
    /// See [`AmpDevice::with_ring`].
    pub fn simulated(info: DeviceInfo, mode: crate::device::ServiceMode) -> Result<&'static Self> {
        let ring: &'static AmpRing = Box::leak(Box::new(AmpRing::new()));
        Self::with_ring(info, ring, mode)
    }

    /// The ring this device publishes to.
    #[must_use]
    pub fn ring(&self) -> &'static AmpRing {
        self.ring
    }

    /// The peer's service loop: consume every `PENDING` slot, run it, and
    /// post completion. Remote firmware calls this forever (typically
    /// around a `wfe`); cooperative mode calls it from
    /// [`Device::progress`].
    pub fn service(&self) {
        loop {
            let head = self.ring.head.load(Ordering::Acquire);
            let tail = self.ring.tail.load(Ordering::Acquire);
            if head == tail {
                break;
            }
            let index = (head % AMP_SLOTS as u32) as usize;
            let slot = &self.ring.slots[index];
            if slot.status.load(Ordering::Acquire) != PENDING {
                break;
            }
            // SAFETY: we acquired `PENDING`, so the payload is immutable
            // until we store `DONE`; copying it ends that borrow before
            // any host rewrite can be ordered after our `head` advance.
            let payload = unsafe { *slot.payload.get() };
            let args = ArgBlock::from_bytes(&payload.args);
            let launch = Launch {
                kernel: payload.kernel,
                // No claim-loop participant reads the name; the host-side
                // opcode dispatch that does never cross the ring.
                name: "",
                binding: payload.binding,
                native: None,
                args: &args,
                dims: payload.dims,
                offset: payload.offset,
                global: payload.global,
                local: payload.local,
            };
            // SAFETY: kernel and arguments came from a completed publish.
            let result = unsafe { run_all(&launch) };
            codevar_target::dsb();
            match result {
                Ok(()) => {
                    slot.result.store(sys::SUCCESS, Ordering::Release);
                    slot.status.store(DONE, Ordering::Release);
                }
                Err(error) => {
                    slot.result
                        .store(error.to_cl(), Ordering::Release);
                    slot.status.store(ERROR, Ordering::Release);
                }
            }
            self.ring
                .head
                .store(head.wrapping_add(1), Ordering::Release);
            codevar_target::dsb();
            codevar_target::sev();
        }
    }

    /// Publishes one launch; the slot index joins the in-flight list.
    ///
    /// Every failure before the descriptor is published releases the
    /// command's event reference (the `Device::submit` failure rule); past
    /// the publish point no failure is possible, so the event always
    /// reaches the reaper.
    fn dispatch(&'static self, launch: &Launch<'_>, event: &'static Event) -> Result<()> {
        event.set_status(sys::SUBMITTED);
        let _serial = self.submit.lock();
        // Reserve the in-flight slot before publishing — the push after the
        // publish cannot fail, and the submit lock keeps the reservation
        // from being consumed by a concurrent dispatch.
        if self.inflight.lock().try_reserve(1).is_err() {
            release_ref(event);
            return Err(Error::OutOfMemory);
        }
        let mut attempts = 0u32;
        let index: u32 = loop {
            let tail = self.ring.tail.load(Ordering::Acquire);
            let slot_index = tail % AMP_SLOTS as u32;
            if self.ring.slots[slot_index as usize]
                .status
                .load(Ordering::Acquire)
                == EMPTY
            {
                // SAFETY: `EMPTY` means the host has exclusive use of the
                // slot (see `AmpSlot`'s Sync contract).
                let payload = unsafe { &mut *self.ring.slots[slot_index as usize].payload.get() };
                payload.kernel = launch.kernel;
                payload.binding = launch.binding;
                payload.dims = launch.dims;
                payload.offset = launch.offset;
                payload.global = launch.global;
                payload.local = launch.local;
                payload.args = *launch.args.as_bytes();
                codevar_target::dsb();
                self.ring.slots[slot_index as usize]
                    .status
                    .store(PENDING, Ordering::Release);
                self.ring
                    .tail
                    .store(tail.wrapping_add(1), Ordering::Release);
                codevar_target::dsb();
                codevar_target::sev();
                break slot_index;
            }
            attempts += 1;
            if attempts > AMP_SLOTS as u32 * 4 {
                release_ref(event);
                return Err(Error::OutOfResources);
            }
            if let Err(error) = self.progress() {
                release_ref(event);
                return Err(error);
            }
            codevar_target::wfe();
        };
        event.set_status(sys::RUNNING);
        // The reservation above makes this push infallible.
        self.inflight.lock().push((index, event));
        // A cooperative peer may have finished it already; the eager reap
        // is best effort, failures surface through later `progress`
        // calls, and the command stays owned by the reaper either way.
        let _ = self.progress();
        Ok(())
    }
}

impl Device for AmpDevice {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn submit(&self, launch: &Launch<'_>, event: &'static Event) -> Result<()> {
        // SAFETY: platform devices are leaked by their constructors.
        let this = unsafe {
            core::ptr::from_ref(self)
                .cast::<AmpDevice>()
                .as_ref()
        }
        .ok_or(Error::InvalidDevice)?;
        this.dispatch(launch, event)
    }

    fn progress(&self) -> Result<()> {
        if self.mode == crate::device::ServiceMode::Cooperative {
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
}
