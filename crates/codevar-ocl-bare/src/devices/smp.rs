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

//! SMP CPU device: the machine's own cores, dispatched through a shared
//! atomic claim counter and `sev`/`wfe` wakeups.
//!
//! # Protocol
//!
//! One [`JobSlot`] per device holds the published launch. A submit:
//!
//! 1. takes the device's submit mutex (queues serialize on one device);
//! 2. writes the descriptor, resets `claim`/`done`, and bumps `epoch`
//!    with release ordering — the descriptor is now visible;
//! 3. `sev`s the workers;
//! 4. runs the claim loop on the submitting core (so a single-core image
//!    does all the work itself);
//! 5. waits for `done == total`, then for every *registered* worker to
//!    acknowledge the epoch.
//!
//! A worker (`worker_loop`, entered by each secondary core at boot) sleeps
//! in `wfe` until `epoch` moves, copies the descriptor, claims groups until
//! the counter is exhausted, and only then acknowledges the epoch. The
//! acknowledgement is the quiescence proof: the next descriptor may be
//! written only after every worker has finished reading the previous one,
//! which makes the descriptor copy race-free without any lock on the read
//! side. Lost wakeups cannot happen — ARM's event register stays set from
//! the `sev` until the `wfe` consumes it, and hosted tests poll instead of
//! sleeping.
//!
//! Registration (`started`) is part of the same handshake: a core that has
//! not entered `worker_loop` yet is skipped by the acknowledgement wait and
//! catches up (acknowledging whatever epoch is current) when it does.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::device::{Device, DeviceInfo};
use crate::error::{Error, Result};
use crate::exec::{run_claimed, ArgBlock, KernelFn, Launch, SpirvBinding};
use crate::objects::{release_ref, Event};
use crate::sys;

/// The published launch shared by all cores of an [`SmpDevice`].
#[derive(Clone)]
struct JobDesc {
    kernel: KernelFn,
    binding: SpirvBinding,
    args: ArgBlock,
    dims: u32,
    offset: [u32; 3],
    global: [u32; 3],
    local: [u32; 3],
}

impl JobDesc {
    /// A descriptor no worker will act on (`epoch` 0 is never published).
    const fn dormant() -> Self {
        Self {
            kernel: idle_kernel,
            binding: SpirvBinding::NONE,
            args: ArgBlock::new(),
            dims: 1,
            offset: [0; 3],
            global: [1; 3],
            local: [1; 3],
        }
    }

    /// A [`Launch`] borrowing this descriptor.
    fn launch(&self) -> Launch<'_> {
        Launch {
            kernel: self.kernel,
            // The name is only read by the accelerator's host-side opcode
            // dispatch, which sees the original launch before publishing;
            // no claim-loop participant ever consults it.
            name: "",
            binding: self.binding,
            args: &self.args,
            dims: self.dims,
            offset: self.offset,
            global: self.global,
            local: self.local,
        }
    }
}

/// Shared job state; see the module docs for the ordering protocol.
struct JobSlot {
    desc: UnsafeCell<JobDesc>,
    epoch: AtomicU32,
    claim: AtomicU32,
    done: AtomicU32,
    error: spin::Mutex<Option<Error>>,
}

// SAFETY: `desc` is written only by a submitter between epochs, after the
// previous epoch's `done`/ack waits proved every reader retired; readers
// observe it only through an acquire load of `epoch`, which is stored
// (release) after the write. `claim`/`done` are atomics and `error` a
// driver-internal mutex.
unsafe impl Sync for JobSlot {}

impl JobSlot {
    const fn new() -> Self {
        Self {
            desc: UnsafeCell::new(JobDesc::dormant()),
            epoch: AtomicU32::new(0),
            claim: AtomicU32::new(0),
            done: AtomicU32::new(0),
            error: spin::Mutex::new(None),
        }
    }

    /// Records `error` as this launch's failure — first one wins.
    fn record(&self, error: Error) {
        let mut slot = self.error.lock();
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    /// Takes the recorded failure, if any, at launch retirement.
    fn take_error(&self) -> Option<Error> {
        self.error.lock().take()
    }
}

/// Never-invoked kernel stored in the dormant descriptor — a placeholder
/// so the static initial state holds a valid function pointer (a null
/// function pointer would be undefined to even materialize).
extern "C" fn idle_kernel(_item: *mut crate::exec::WorkItem) {}

/// A CPU device over `cores` cores of one SMP cluster.
///
/// The submitting core always participates, so `cores == 1` (QEMU's
/// `mps2-an385`) is the degenerate case: the same claim loop runs, just
/// with only one claimant.
pub struct SmpDevice {
    info: DeviceInfo,
    workers: u32,
    slot: JobSlot,
    submit: spin::Mutex<()>,
    started: Vec<AtomicBool>,
    acks: Vec<AtomicU32>,
}

impl SmpDevice {
    /// Creates the device and leaks it, giving it the stable address a
    /// `cl_device_id` needs. `cores` is the compute-unit count the board
    /// reports; cores beyond the first are secondary cores that must enter
    /// [`SmpDevice::worker_loop`] with their index before the first
    /// enqueue (see [`SmpDevice::wait_workers`]).
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] when the registration vectors cannot be
    /// allocated.
    pub fn new(info: DeviceInfo, cores: u32) -> Result<&'static Self> {
        let cores = cores.max(1);
        let workers = (cores - 1) as usize;
        let mut started = Vec::new();
        started
            .try_reserve_exact(workers)
            .map_err(|_| Error::OutOfMemory)?;
        started.resize_with(workers, || AtomicBool::new(false));
        let mut acks = Vec::new();
        acks
            .try_reserve_exact(workers)
            .map_err(|_| Error::OutOfMemory)?;
        acks.resize_with(workers, || AtomicU32::new(0));
        Ok(Box::leak(Box::new(Self {
            info,
            workers: cores - 1,
            slot: JobSlot::new(),
            submit: spin::Mutex::new(()),
            started,
            acks,
        })))
    }

    /// Total cores of the cluster (`CL_DEVICE_MAX_COMPUTE_UNITS`).
    #[must_use]
    pub const fn cores(&self) -> u32 {
        self.workers + 1
    }

    /// Blocks until every secondary core has registered — called after
    /// starting the workers so the first enqueue cannot race their boot.
    pub fn wait_workers(&'static self) {
        codevar_target::wait_until(|| self.started.iter().all(|flag| flag.load(Ordering::Acquire)));
    }

    /// The per-core worker body: register, then loop forever claiming
    /// work-groups of whatever epoch is current.
    ///
    /// Each secondary core must call this exactly once with its unique
    /// index in `0..workers` and never return (the boot stub jumps here).
    /// On a hosted test, spawn one thread per index.
    ///
    /// # Never returns
    ///
    /// An out-of-range index (no such worker, or `cores == 1`) parks the
    /// caller in a `wfe` loop rather than risking a claim-counter race —
    /// with a `debug_assert` in development builds to catch the boot error.
    pub fn worker_loop(&'static self, index: u32) -> ! {
        let valid = index < self.workers;
        debug_assert!(valid, "worker index {index} out of range");
        if !valid {
            loop {
                codevar_target::wfe();
            }
        }
        let index = index as usize;
        // Register: acknowledge the current epoch so a submitter already
        // past this worker's slot is never blocked, then mark started
        // (release) so scanners that see `started` also see the ack.
        let current = self.slot.epoch.load(Ordering::Acquire);
        self.acks[index].store(current, Ordering::Release);
        self.started[index].store(true, Ordering::Release);

        let mut my_epoch = current;
        loop {
            let epoch = self.slot.epoch.load(Ordering::Acquire);
            if epoch == my_epoch {
                codevar_target::wfe();
                continue;
            }
            // Copy the descriptor out: the next writer is fenced behind
            // this worker's acknowledgement, so the copy is stable.
            // SAFETY: protocol documented on `JobSlot` — the acquire load
            // of `epoch` pairs with the submitter's release store, which
            // follows the descriptor write.
            let desc = unsafe { &*self.slot.desc.get() }.clone();
            let launch = desc.launch();
            // SAFETY: the kernel and argument block come from a complete,
            // exclusively-published descriptor.
            if let Err(error) = unsafe { run_claimed(&launch, &self.slot.claim, &self.slot.done) } {
                self.slot.record(error);
            }
            my_epoch = epoch;
            self.acks[index].store(epoch, Ordering::Release);
        }
    }

    /// Publishes and runs one launch to completion.
    ///
    /// Completes `event` and releases the command's reference before
    /// returning — the synchronous half of the [`Device`] contract. A
    /// launch whose execution fails releases that reference *without*
    /// completing the event (the `Device::submit` failure rule) and
    /// returns the error; the API layer completes it when it holds the
    /// caller's reference.
    fn execute(&'static self, launch: &Launch<'_>, event: &'static Event) -> Result<()> {
        event.set_status(sys::SUBMITTED);
        let _serial = self.submit.lock();
        let total = launch.group_count();
        *self.slot.error.lock() = None;
        // Descriptor write, counters and epoch bump under the submit lock;
        // release ordering publishes the whole group to the workers.
        // SAFETY: exclusive access — the submit lock serializes writers,
        // and the previous epoch's waits proved no reader is live.
        unsafe {
            let desc = &mut *self.slot.desc.get();
            desc.kernel = launch.kernel;
            desc.binding = launch.binding;
            desc.args = *launch.args;
            desc.dims = launch.dims;
            desc.offset = launch.offset;
            desc.global = launch.global;
            desc.local = launch.local;
        }
        self.slot.claim.store(0, Ordering::Release);
        self.slot.done.store(0, Ordering::Release);
        let epoch = self.slot.epoch.fetch_add(1, Ordering::Release) + 1;
        codevar_target::dsb();
        codevar_target::sev();

        event.set_status(sys::RUNNING);
        // The submitting core claims alongside the workers; a group failure
        // is recorded but does not stop the claim loop, so `done` still
        // reaches `total` for the wait below.
        // SAFETY: the descriptor above is this launch and is stable until
        // the acknowledgement wait below returns.
        if let Err(error) = unsafe { run_claimed(launch, &self.slot.claim, &self.slot.done) } {
            self.slot.record(error);
        }
        codevar_target::wait_until(|| self.slot.done.load(Ordering::Acquire) >= total);
        // Quiescence: every registered worker has retired this epoch, so
        // the next submit may overwrite the descriptor.
        codevar_target::wait_until(|| {
            self.started
                .iter()
                .zip(self.acks.iter())
                .all(|(started, ack)| !started.load(Ordering::Acquire) || ack.load(Ordering::Acquire) >= epoch)
        });
        if let Some(error) = self.slot.take_error() {
            release_ref(event);
            return Err(error);
        }
        event.finish(sys::COMPLETE);
        release_ref(event);
        Ok(())
    }
}

impl Device for SmpDevice {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn submit(&self, launch: &Launch<'_>, event: &'static Event) -> Result<()> {
        // `execute` needs `&'static self`; the handle API only produces
        // leaked devices, so recover the static lifetime from the address.
        // SAFETY: every `SmpDevice` reachable through the platform was
        // leaked by `SmpDevice::new` and therefore lives for `'static`.
        let this = unsafe { core::ptr::from_ref(self).cast::<SmpDevice>().as_ref() }
            .ok_or(Error::InvalidDevice)?;
        this.execute(launch, event)
    }

    fn progress(&self) -> Result<()> {
        // Synchronous back end: nothing can be in flight when `submit`
        // has returned.
        Ok(())
    }

    fn inflight(&self) -> usize {
        0
    }
}
