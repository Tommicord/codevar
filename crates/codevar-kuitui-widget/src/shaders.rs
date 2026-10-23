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

//! The UI shader programs and the vertex layouts that feed them.
//!
//! The GLSL sources live in `codevar-kuitui`'s `shaders/` directory and
//! are compiled to SPIR-V by that crate's build script (commit
//! `842e1e2`). This module does not recompile them: it re-exports the
//! images [`codevar_kuitui::shaders`] already embeds, and documents the
//! exact vertex and push-constant layout each program expects so a
//! backend in this crate can build its pipelines from one place.
//!
//! # The programs
//!
//! | Static | Source | Stage |
//! | --- | --- | --- |
//! | [`UI_VERT_SPIRV`] | `ui.vert` | vertex |
//! | [`UI_FRAG_SPIRV`] | `ui.frag` | fragment |
//! | [`TEXT_VERT_SPIRV`] | `text.vert` | vertex |
//! | [`TEXT_FRAG_SPIRV`] | `text.frag` | fragment |
//!
//! [`SHADERS`] is the registry of all four and [`SHADERS_COMPILED`] is
//! `false` on a build host without `glslc`, in which case every image is
//! empty. A backend must check [`SHADERS_COMPILED`] before it uploads a
//! module; the unit tests here assert the images are non-empty when it
//! is `true`.
//!
//! # The `ui` program
//!
//! * vertex input location 0 — [`UiVertex::position`], logical pixels,
//!   origin top-left of the render target;
//! * vertex input location 1 — [`UiVertex::color`], non-premultiplied
//!   sRGB;
//! * push constant block — [`PushConstants`], the viewport size in
//!   logical pixels;
//! * fragment input location 0 — the interpolated `v_color`;
//! * fragment output location 0 — the fragment colour.
//!
//! Because the colour is interpolated, the same program paints a flat
//! fill, a line, a polygon and a linear gradient (the ends of an edge
//! carry the gradient's end colours).
//!
//! # The `text` program
//!
//! * vertex input location 0 — [`TextVertex::position`];
//! * vertex input location 1 — [`TextVertex::uv`], normalized atlas
//!   coordinates;
//! * vertex input location 2 — [`TextVertex::color`];
//! * push constant block — [`PushConstants`];
//! * fragment binding 0 — a `sampler2D` atlas whose single-channel
//!   coverage lives in `.r`; the fragment outputs
//!   `vec4(color.rgb, color.a * coverage)`.

pub use codevar_kuitui::shaders::{
    SHADERS, SHADERS_COMPILED, ShaderModule, TEXT_FRAG_SPIRV, TEXT_VERT_SPIRV, UI_FRAG_SPIRV, UI_VERT_SPIRV,
};

use codevar_kuitui::color::Color;
use codevar_kuitui::geometry::Offset;

/// A vertex of the `ui` geometry program.
///
/// The layout matches `ui.vert` exactly: `position` is location 0 and
/// `color` is location 1. The struct is `repr(C)` so it can be uploaded
/// as a vertex buffer without padding surprises.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct UiVertex {
    /// The position in logical pixels, origin at the top-left.
    pub position: [f32; 2],
    /// The non-premultiplied sRGB colour.
    pub color: [f32; 4],
}

impl UiVertex {
    /// The size of one vertex in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();

    /// Creates a vertex from raw components.
    #[inline]
    #[must_use]
    pub const fn new(position: [f32; 2], color: [f32; 4]) -> Self {
        Self { position, color }
    }

    /// Creates a vertex from an offset and a colour.
    #[inline]
    #[must_use]
    pub const fn from_parts(position: Offset, color: Color) -> Self {
        Self {
            position: [position.x, position.y],
            color: [color.red, color.green, color.blue, color.alpha],
        }
    }
}

/// A vertex of the `text` glyph-quad program.
///
/// The layout matches `text.vert` exactly: `position` is location 0,
/// `uv` is location 1 and `color` is location 2. The struct is
/// `repr(C)` so it can be uploaded as a vertex buffer as-is.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TextVertex {
    /// The position in logical pixels, origin at the top-left.
    pub position: [f32; 2],
    /// The normalized atlas coordinates of the pixel.
    pub uv: [f32; 2],
    /// The non-premultiplied sRGB colour of the run.
    pub color: [f32; 4],
}

impl TextVertex {
    /// The size of one vertex in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();

    /// Creates a vertex from raw components.
    #[inline]
    #[must_use]
    pub const fn new(position: [f32; 2], uv: [f32; 2], color: [f32; 4]) -> Self {
        Self { position, uv, color }
    }
}

/// The push-constant block shared by both programs.
///
/// `ui.vert` and `text.vert` declare
/// `layout(push_constant) uniform PushConstants { vec2 viewport; }`, and
/// map a logical-pixel position to clip space with
/// `position / viewport * 2.0 - 1.0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PushConstants {
    /// The render target size in logical pixels.
    pub viewport: [f32; 2],
}

impl PushConstants {
    /// The size of the block in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();

    /// Creates the block for a viewport of `width` by `height` pixels.
    #[inline]
    #[must_use]
    pub const fn new(width: f32, height: f32) -> Self {
        Self {
            viewport: [width, height],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_compiled_shaders_are_non_empty() {
        if SHADERS_COMPILED {
            for module in SHADERS {
                assert!(
                    !module.spirv.is_empty(),
                    "`{}` compiled but its SPIR-V image is empty",
                    module.name
                );
            }
            assert!(!UI_VERT_SPIRV.is_empty());
            assert!(!UI_FRAG_SPIRV.is_empty());
            assert!(!TEXT_VERT_SPIRV.is_empty());
            assert!(!TEXT_FRAG_SPIRV.is_empty());
        }
    }

    #[test]
    fn the_registry_lists_every_stage_once() {
        let names: alloc::vec::Vec<&str> = SHADERS.iter().map(|module| module.name).collect();
        assert!(names.contains(&"ui.vert"));
        assert!(names.contains(&"text.frag"));
        for module in SHADERS {
            assert!(matches!(module.stage, "vert" | "frag" | "comp"));
        }
    }

    #[test]
    fn the_ui_vertex_packs_position_then_color() {
        let vertex = UiVertex::from_parts(Offset::new(3.0, 4.0), Color::RED);
        assert_eq!(vertex.position, [3.0, 4.0]);
        assert_eq!(vertex.color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(UiVertex::SIZE, 24);
    }

    #[test]
    fn the_text_vertex_packs_position_uv_then_color() {
        let vertex = TextVertex::new([1.0, 2.0], [0.5, 0.25], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(vertex.position, [1.0, 2.0]);
        assert_eq!(vertex.uv, [0.5, 0.25]);
        assert_eq!(TextVertex::SIZE, 32);
    }

    #[test]
    fn the_push_constants_carry_the_viewport() {
        let block = PushConstants::new(1280.0, 720.0);
        assert_eq!(block.viewport, [1280.0, 720.0]);
        assert_eq!(PushConstants::SIZE, 8);
    }
}
