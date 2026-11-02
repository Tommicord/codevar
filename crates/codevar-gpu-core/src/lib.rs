//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The backend-neutral GPU platform of Codevar.
//!
//! This crate owns the vocabulary every graphics backend agrees on and
//! the traits a backend implements for its platform-dependent parts. It
//! contains no graphics API: Vulkan lives in `codevar-gpu-vk` and
//! WebGPU in `codevar-gpu-wg`, and both implement the traits declared
//! here.
//!
//! # The platform model
//!
//! The model has four levels, each one owned by a backend:
//!
//! 1. [`GpuPlatform`] — an API instance that can create a device.
//! 2. [`GpuDevice`] — a logical device that can create a surface.
//! 3. [`GpuSurface`] — the target frames are rendered into.
//! 4. [`GpuRenderer`] — the frame lifecycle of a backend renderer.
//!
//! A caller that is generic over the backend only depends on this
//! crate; the backend crate is selected once, where the application is
//! assembled. The display list itself is already backend-neutral and
//! comes from `codevar-kuitui`; this crate does not need it.
//!
//! # Why the traits are not boxed
//!
//! Backends own their device and surface types, and a Vulkan surface
//! borrows the device that created it. The traits therefore use
//! associated types and generic lifetimes rather than trait objects:
//! the selection of a backend happens at compile time, and no virtual
//! call sits on the frame path.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod backend;
pub mod handle;
pub mod traits;

pub use backend::{
    AlphaMode, Backend, CompletionKind, FrameCompletion, GpuCapabilities, GpuFailure, GpuFeature,
    GpuFeatures, GpuLimits, PresentMode, SurfaceConfig, SurfaceFormat,
};
pub use handle::OwnedFd;
pub use traits::{GpuDevice, GpuFrame, GpuPlatform, GpuRenderer, GpuSurface};
