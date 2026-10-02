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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing
//! permissions and limitations under the License.

//! # Codevar smooth renderer
//!
//! Placeholder module documentation, replaced by the full port.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

use codevar_truetype_core::{GlyphFormat, ModuleClass, ModuleKind, Raster, RasterFuncs, RendererClass};

/// Scratch static used to prove that a self-referential renderer class
/// compiles (FreeType 2.6 `FT_DEFINE_RENDERER` embeds `FT_Module_ClassRec`
/// whose `kind` points back at the enclosing `FT_Renderer_ClassRec`).
static SMOOTH_RENDERER_CLASS: RendererClass = RendererClass {
    root: ModuleClass {
        kind: ModuleKind::Renderer(&SMOOTH_RENDERER_CLASS),
        module_flags: 2,
        module_size: 96,
        module_name: "smooth",
        module_version: 0x10000,
        module_requires: 0x20000,
        module_interface: None,
        module_init: None,
        module_done: None,
        get_interface: None,
    },
    glyph_format: GlyphFormat::Outline,
    render_glyph: None,
    transform_glyph: None,
    get_glyph_cbox: None,
    set_mode: None,
    raster_class: None,
};

/// Scratch function proving `RasterFuncs` can reference our raster factory.
static SCRATCH_RASTER_FUNCS: RasterFuncs = RasterFuncs {
    glyph_format: GlyphFormat::Outline,
    new: |_memory| {
        struct Scratch;
        impl Raster for Scratch {
            fn reset(&mut self, _pool: Option<&mut [u8]>, _pool_size: usize) {}
            fn set_mode(
                &mut self,
                _mode: u64,
                _value: &mut dyn core::any::Any,
            ) -> codevar_truetype_core::TtResult<()> {
                Ok(())
            }
            fn render(
                &mut self,
                _params: &mut codevar_truetype_core::RasterParams<'_>,
            ) -> codevar_truetype_core::TtResult<()> {
                Ok(())
            }
        }
        Ok(alloc::boxed::Box::new(Scratch))
    },
};
