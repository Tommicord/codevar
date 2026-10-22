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

//! The WebGPU backend of the Kuitui user interface.
//!
//! This crate turns the display list of `codevar-kuitui` into GPU
//! work with WebGPU. It is the counterpart of the Vulkan backend that
//! lives in `codevar-gpu-core`: both consume the very same display
//! list, so a user interface looks the same on the desktop and in a
//! browser.
//!
//! # Why WebGPU
//!
//! The desktop of Codevar renders with Vulkan, which is not available
//! in a browser. WebGPU is the only graphics API a browser offers, so
//! the web build of the editor uses this backend. The Vulkan backend
//! is not replaced: it stays the backend of the desktop build.
//!
//! # The shape of a frame
//!
//! 1. The application builds a tree of widgets and lays it out. The
//!    tree is recorded into a [`codevar_kuitui::DrawList`].
//! 2. [`WgRenderer::render`] walks the display list and fills the
//!    vertex and index buffers of the frame.
//! 3. [`WgRenderer::present`] submits the frame and, in a browser,
//!    hands it to the canvas.
//!
//! The crate uses `std`, because `wgpu` does; the rest of Codevar
//! stays `no_std`.

#![warn(missing_docs)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod config;
pub mod error;
pub mod mesh;
pub mod renderer;
pub mod text;

pub use config::WgConfig;
pub use error::WgError;
pub use mesh::{Mesh, PrimitiveKind, Vertex};
pub use renderer::WgRenderer;
pub use text::{Glyph, GlyphAtlas, GlyphKey};
