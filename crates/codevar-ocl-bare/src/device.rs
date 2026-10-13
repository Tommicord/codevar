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

//! The device abstraction: one trait, three very different back ends.
//!
//! [`Device`] is the seam the whole driver hangs from — PoCL's device ops
//! table, shrunk to what an embedded profile needs. `clGetDeviceInfo`
//! reads [`DeviceInfo`]; `clEnqueueNDRangeKernel` marshals the launch once
//! and calls [`Device::submit`]; event waiting and `clFinish` call
//! [`Device::progress`]/[`Device::wait_all`]. Everything else (buffers,
//! copies, validation) never touches a device, because on a single-address-
//! space machine a buffer copy is a `memcpy`.
//!
//! Implementations live in [`crate::devices`]:
//!
//! * [`crate::SmpDevice`] — CPU cores of one SMP cluster, executing inline
//!   through the shared claim counter.
//! * [`crate::AmpDevice`] — a descriptor ring to an AMP peer (DSP, second
//!   M-class core, or a simulated one).
//! * [`crate::AccelDevice`] — an FPGA/accelerator with an MMIO register
//!   block, in the style of the almaif OpenCL driver.

use crate::error::Result;
use crate::exec::Launch;
use crate::objects::Event;
use crate::registry::KernelDef;

/// How a device's completions are driven while the host waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceMode {
    /// The remote agent (AMP peer, FPGA core, its own core) completes
    /// descriptors on its own; the host only polls status words and sleeps
    /// in `wfe`. This is the real-hardware mode.
    Remote,
    /// No second agent exists (QEMU's single Cortex-M3, a host test): the
    /// host services the ring itself whenever it waits for completion —
    /// exactly what the remote's firmware would have done. The descriptor
    /// protocol is identical; only who steps through it differs.
    Cooperative,
}

/// Static description of a device, backing every `clGetDeviceInfo` query.
///
/// Construct with [`DeviceInfo::new`], which fills the portability-derived
/// defaults; the fields are public for the values a board description wants
/// to pin down explicitly.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// `CL_DEVICE_NAME`.
    pub name: &'static str,
    /// `CL_DEVICE_VENDOR`.
    pub vendor: &'static str,
    /// `CL_DEVICE_TYPE` bitfield.
    pub device_type: u64,
    /// `CL_DEVICE_VENDOR_ID` (an assigned PCI-SIG ID; 0 for unassigned).
    pub vendor_id: u32,
    /// `CL_DEVICE_MAX_COMPUTE_UNITS` — cores of the SMP cluster, 1 for the
    /// ring devices (their remote side is reported separately by the
    /// board's own documentation).
    pub compute_units: u32,
    /// `CL_DEVICE_MAX_WORK_GROUP_SIZE`.
    pub max_work_group_size: u32,
    /// `CL_DEVICE_MAX_WORK_ITEM_SIZES`.
    pub max_work_item_sizes: [u32; 3],
    /// `CL_DEVICE_ADDRESS_BITS`.
    pub address_bits: u32,
    /// `CL_DEVICE_MAX_MEM_ALLOC_SIZE`.
    pub max_mem_alloc_size: u64,
    /// `CL_DEVICE_GLOBAL_MEM_SIZE`.
    pub global_mem_size: u64,
    /// `CL_DEVICE_LOCAL_MEM_SIZE` (on-chip scratch; the CPU device reports
    /// its tightly-coupled RAM, ring devices their descriptor SRAM).
    pub local_mem_size: u64,
    /// `CL_DEVICE_MAX_PARAMETER_SIZE` — the whole marshaled argument
    /// block the device accepts (256 for CPU/AMP, one descriptor's worth
    /// for the accelerator).
    pub max_parameter_size: u32,
    /// `CL_DEVICE_EXTENSIONS`.
    pub extensions: &'static str,
    /// `CL_DRIVER_VERSION`.
    pub driver_version: &'static str,
    /// `CL_DEVICE_VERSION`.
    pub version: &'static str,
    /// `CL_DEVICE_PROFILE` — always `EMBEDDED_PROFILE` for this driver.
    pub profile: &'static str,
}

impl DeviceInfo {
    /// Builds a device description with portability-derived defaults:
    /// address width from the target pointer size, max allocation from a
    /// quarter of global memory, three work-item dimensions bounded by the
    /// work-group cap, and the crate's version/profile strings.
    #[must_use]
    pub const fn new(
        name: &'static str,
        vendor: &'static str,
        device_type: u64,
        compute_units: u32,
        max_work_group_size: u32,
        global_mem_size: u64,
        local_mem_size: u64,
        max_parameter_size: u32,
    ) -> Self {
        Self {
            name,
            vendor,
            device_type,
            vendor_id: 0,
            compute_units,
            max_work_group_size,
            max_work_item_sizes: [max_work_group_size; 3],
            address_bits: (core::mem::size_of::<usize>() * 8) as u32,
            max_mem_alloc_size: global_mem_size / 4,
            global_mem_size,
            local_mem_size,
            max_parameter_size,
            extensions: "",
            driver_version: env!("CARGO_PKG_VERSION"),
            version: "OpenCL 1.2 Codevar-bare",
            profile: "EMBEDDED_PROFILE",
        }
    }
}

/// A device back end of the platform.
///
/// # Implementor contract
///
/// [`Device::submit`] owns one reference of `event` (the command's
/// reference): it must either complete the event and release that reference
/// before returning (synchronous back ends — SMP), or keep it in flight and
/// release it when [`Device::progress`] observes completion (async back
/// ends — AMP, accelerator). On failure it must release the reference and
/// return the error without touching the event again.
pub trait Device: Sync {
    /// The static description backing `clGetDeviceInfo`.
    fn info(&self) -> &DeviceInfo;

    /// Executes (or dispatches) one NDRange launch and transitions `event`
    /// through `CL_SUBMITTED`/`CL_RUNNING` toward `CL_COMPLETE`.
    ///
    /// # Errors
    ///
    /// * [`crate::Error::KernelNotFound`] — the kernel cannot run on this
    ///   device (e.g. an accelerator opcode has no handler).
    /// * [`crate::Error::OutOfResources`] — the descriptor ring is full
    ///   after waiting.
    fn submit(&self, launch: &Launch<'_>, event: &'static Event) -> Result<()>;

    /// Polls asynchronous completions once: services the device if it is
    /// in [`ServiceMode::Cooperative`], then reaps every event whose
    /// descriptor finished, completing and releasing it.
    ///
    /// Called from `clFlush`, `clFinish`, `clWaitForEvents` and event
    /// waiting loops; must be cheap and never block.
    fn progress(&self) -> Result<()>;

    /// Number of commands still in flight on this device.
    fn inflight(&self) -> usize;

    /// Blocks until every in-flight command has completed, polling
    /// [`Device::progress`] and sleeping in between.
    ///
    /// The default implementation is the portable wait loop; devices with
    /// a cheaper "is anything left" signal may override it.
    fn wait_all(&self) -> Result<()> {
        while self.inflight() > 0 {
            self.progress()?;
            codevar_target::wfe();
        }
        Ok(())
    }

    /// Whether this device can execute `def` (the accelerator checks it
    /// against its opcode table; CPU devices accept every registered
    /// kernel). Used to report `CL_INVALID_KERNEL` at enqueue time.
    fn supports(&self, _def: &KernelDef) -> bool {
        true
    }
}
