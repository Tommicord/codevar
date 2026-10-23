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

//! The WebGPU backend and compositor of Codevar.
//!
//! This crate mirrors [`codevar-gpu-vk`] primitive for primitive — the
//! pipeline context, the offscreen colour targets, the pipe model, the
//! fragment-shader mix pass and the renderer subsystem — but drives the
//! browser graphics API through the raw, bounds-check-free `wgpu-hal`
//! layer instead of `ash`.
//!
//! The traits themselves live in `codevar-gpu-core`; a caller assembled
//! for the web only depends on that crate and this backend.
//!
//! # Target
//!
//! The backend is compiled on `wasm32` only. Every other target builds
//! [`platform`]'s stub, which reports [`Backend::WebGpu`] but refuses to
//! create a device: the browser is the only host that can open a WebGPU
//! surface.
//!
//! [`codevar-gpu-vk`]: https://docs.rs/codevar-gpu-vk

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod base;
pub mod offscreen;
pub mod pipe;
pub mod pipe_ctx;
pub mod pipeline;
pub mod platform;
pub mod renderer;
pub mod types;
