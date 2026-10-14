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

//! Bare-metal OpenCL embedded-profile driver.
//!
//! Codevar's host runtime (`codevar-ocl`) discovers GPUs through the ICD
//! loader, which needs `dlopen`, a dynamic linker and a JSON manifest. An
//! embedded device has none of that: the firmware image is one flat binary
//! and every byte of the driver must be in it. This crate is the inversion of
//! the ICD model — the `cl*` entry points are ordinary `#[unsafe(no_mangle)]
//! extern "C"` functions compiled into the image, applications link against
//! them statically, and the *devices* are pluggable back ends instead:
//!
//! | Device            | Back end                                          |
//! |-------------------|---------------------------------------------------|
//! | CPU cores (SMP)   | atomic work-group claim loop, WFE/SEV wakeups      |
//! | Remote cores (AMP)| shared-memory descriptor ring + SEV doorbell        |
//! | FPGA / accelerator| MMIO register block + descriptor ring (almaif-style)|
//!
//! All three implement one [`device::Device`] trait, so
//! `clEnqueueNDRangeKernel` marshals the launch exactly once and hands it to
//! whichever back end the queue was created with.
//!
//! # Embedded profile
//!
//! The implemented subset targets the OpenCL **embedded profile**: buffers
//! only (no images/samplers/pipes/SVM), an in-order queue per context, no
//! online compiler — [`api::clCreateProgramWithSource`] returns
//! [`sys::COMPILER_NOT_AVAILABLE`] and programs bind to kernels registered
//! statically in the image through [`api::clCreateProgramWithBuiltInKernels`]
//! (or [`api::clCreateProgramWithBinary`], whose "binary" is the registry key
//! the compiler would have produced). Out-of-order queues and event profiling
//! are rejected with [`sys::INVALID_QUEUE_PROPERTIES`].
//!
//! # Example
//!
//! ```rust,ignore
//! codevar_ocl_bare::init(&codevar_ocl_bare::PlatformConfig {
//!     name: "my-board",
//!     registry: &KERNELS,          // &'static KernelModule list
//!     devices: &[&SMP, &AMP, &FPGA],
//! })?;
//! ```
//!
//! After [`init`], C code (or Rust) calls the `cl*` functions of [`api`]
//! through whatever binding layer it has; nothing is looked up at runtime.
//!
//! There is no MMU: buffer memory, kernel argument memory and the descriptor
//! rings all live in the single flat address space, so a "device" reads the
//! same bytes the host wrote — the copy-free path embedded OpenCL exists for.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod api;
mod device;
mod devices;
mod error;
mod exec;
mod objects;
mod platform;
mod registry;
mod spirv;
pub mod sys;

pub use device::{Device, DeviceInfo, ServiceMode};
pub use devices::{
    AccelDesc, AccelDevice, AccelHandler, AccelRegs, AccelRing, AmpDevice, AmpRing, SmpDevice,
    default_accel_handler,
};
pub use error::{Error, Result};
pub use exec::{KernelFn, MAX_ARGS, MAX_PARAMS, SpirvBinding, WorkItem};
pub use objects::{Context, Event, Kernel, Mem, Program, Queue};
pub use platform::{Platform, PlatformConfig, init, platform};
pub use registry::{ArgKind, KernelDef, KernelModule, Registry};
