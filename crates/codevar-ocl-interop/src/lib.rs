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

//! Buffer and semaphore sharing between an OpenCL and a Vulkan device.
//!
//! Both APIs can work on the same physical GPU, and each can hand memory or
//! a synchronization object to the other without a copy. `codevar-ocl` and
//! the Vulkan loader expose the two halves of that handshake; this crate
//! joins them:
//!
//! * one device has to be the same device — the pairing check compares the
//!   OpenCL `CL_DEVICE_UUID_KHR` / Windows LUID against the Vulkan physical
//!   device identity, because a handle exported by one GPU cannot be
//!   imported by another;
//! * memory crosses as an external handle (`dma-buf`, `OPAQUE_FD`, the
//!   Win32 handle family) and is imported as a buffer of the receiving API,
//!   which then has to take and hand back ownership around each command;
//! * synchronization crosses the same way, as a semaphore handle
//!   (`OPAQUE_FD`, `SYNC_FD`, a Win32 handle or a D3D12 fence) that one API
//!   signals and the other waits on.
//!
//! The crate creates no instance and no device. Every entry point takes the
//! live Vulkan and OpenCL objects from the caller, so a renderer keeps
//! ownership of its own driver handles and this crate only ever borrows
//! them for the duration of one exchange.
//!
//! # Unsafe
//!
//! Sharing a handle means calling vendor entry points through raw function
//! pointers on both sides, so `unsafe` is required here rather than
//! avoided. Every block documents what makes the call sound: handles that
//! belong to the same driver and to the same device, buffers that outlive
//! the call, and out-pointers that are valid for the width the loader reads.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

mod error;
mod handle;
mod pair;
mod session;
mod vk;

pub use error::{Error, Result};
pub use handle::{MemoryHandle, MemoryKind, SemaphoreHandle, SemaphoreKind};
pub use pair::Identity;
pub use session::{Share, SharedBuffer};
pub use vk::Vulkan;
