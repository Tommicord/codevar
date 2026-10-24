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

//! Text rendering for Kuitui: a demand-driven glyph atlas shared by every
//! backend, the indexed glyph mesh the backends draw, and one adapter per
//! platform.
//!
//! The crate splits the text stack along the line the backends already
//! draw:
//!
//! * [`atlas`] owns the CPU-side cache — coverage pages, shelf packing,
//!   LRU eviction, pinning and dirty rectangles. It is platform-neutral.
//! * the mesh builder turns a shaped run into indexed quads (four
//!   vertices and six indices per glyph) so the GPU reuses two vertices
//!   per glyph instead of describing two triangles.
//! * the dirty signal tells a backend whether the text changed at all,
//!   so an idle frame re-uploads no coverage and rebuilds no mesh.
//! * the platform modules own the GPU objects and the upload/draw calls:
//!   Vulkan in [`vulkan`] (native) and WebGPU in [`webgpu`] (wasm32).
//!
//! The atlas never reads data back from the GPU. The CPU is the source of
//! truth for coverage; the GPU receives dirty rectangles through a
//! host-visible staging buffer and keeps its textures between frames.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod atlas;
pub mod dirty;
pub mod mesh;
/// Native Vulkan adapter: uploads atlas coverage and draws the indexed runs.
///
/// The module only builds on targets that have Vulkan (everything except
/// wasm32, which uses [`webgpu`] instead).
#[cfg(not(target_arch = "wasm32"))]
pub mod vulkan;
/// Browser WebGPU adapter: stages coverage and geometry through copies and
/// draws the indexed runs.
///
/// The module only builds on wasm32, which has a WebGPU (and its WebGL2
/// fallback) surface; every other target uses [`vulkan`] instead.
#[cfg(target_arch = "wasm32")]
pub mod webgpu;

pub use atlas::{
    AtlasConfig, AtlasStats, DirtyRect, GlyphKey, GlyphSlot, TextAtlas, TextError, TextFrame, TextVertex,
};
pub use dirty::{DirtySignal, coalesce};
pub use mesh::{GlyphQuad, INDICES_PER_QUAD, Index, IndexedMesh, MeshBuilder, MeshRun, VERTICES_PER_QUAD};
#[cfg(not(target_arch = "wasm32"))]
pub use vulkan::TextPipe;
#[cfg(target_arch = "wasm32")]
pub use webgpu::TextPipe;
