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

//! The configuration of the WebGPU backend.

/// The configuration of the [`WgRenderer`](crate::WgRenderer).
///
/// The configuration says which canvas the frames are presented to and
/// which adapter the renderer asks for. Every field has a default, so
/// the common case is `WgConfig::default()`.
#[derive(Debug, Clone, PartialEq)]
pub struct WgConfig {
    /// The identifier of the canvas the frames are presented to.
    ///
    /// On the web this is the `id` of the `<canvas>` element. On the
    /// desktop it is ignored.
    pub canvas_id: String,
    /// The width of the surface, in pixels.
    pub width: u32,
    /// The height of the surface, in pixels.
    pub height: u32,
    /// Whether the alpha channel of the surface is used.
    pub alpha_mode: WgAlphaMode,
    /// The name of the adapter the renderer asks for.
    ///
    /// An empty name asks for any adapter that can render.
    pub adapter_name: String,
}

impl WgConfig {
    /// Creates a configuration for a surface of `width` by `height`
    /// pixels.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            canvas_id: String::from("codevar"),
            width,
            height,
            alpha_mode: WgAlphaMode::Opaque,
            adapter_name: String::new(),
        }
    }

    /// The configuration that presents to the canvas called `canvas_id`.
    #[must_use]
    pub fn with_canvas_id(mut self, canvas_id: impl Into<String>) -> Self {
        self.canvas_id = canvas_id.into();
        self
    }
}

impl Default for WgConfig {
    fn default() -> Self {
        Self::new(1280, 720)
    }
}

/// How the alpha channel of the surface is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WgAlphaMode {
    /// The surface is opaque; the alpha channel is ignored.
    #[default]
    Opaque,
    /// The surface is transparent; the alpha channel of the frame is
    /// kept.
    Premultiplied,
}
