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

//! TrueType parsing, hinting and glyph rasterization engine.
//!
//! The crate turns bytes of a TrueType (`.ttf`/`.otf`-sfnt) font into
//! anti-aliased glyph bitmaps, an optional subpixel (LCD) variant of
//! those bitmaps, GPU-ready vertex data, and a compressed glyph atlas
//! backed by the [`codevar_fcware`] multi-codec frame format.
//!
//! # Pipeline
//!
//! ```text
//! FontFile::parse ──► Cmap::glyph_index ──► glyf::load_glyph
//!        │                                          │
//!        │             interpreter::run (hinting) ◄─┘
//!        │                                          │
//!        │             raster::rasterize ──► subpixel::filter
//!        │                          │
//!        ▼                          ▼
//!   shaders::* (GLSL)         FontAtlas (R8 + FcWare)
//! ```
//!
//! | Module | Responsibility |
//! |--------|----------------|
//! | [`font_file`] | sfnt container, table directory, `head`/`hhea`/`maxp`/`hmtx`/`loca`/`name`/`os2`/`gasp`/`kern`/`cvt`/`fpgm`/`prep` |
//! | [`cmap`] | character-to-glyph lookup (formats 0, 2, 4, 6, 12, 13, 14) |
//! | [`glyf`] | simple and composite glyph outline extraction, phantom points |
//! | [`interpreter`] | TrueType bytecode virtual machine (full opcode set) |
//! | [`raster`] | analytic scanline coverage rasterizer (non-zero winding, exact area AA) |
//! | [`subpixel`] | LCD subpixel rendering with FIR filters and gamma LUTs |
//! | [`simd`] | runtime-dispatched SSE2/SSSE3/AVX2/AVX-512/NEON kernels |
//! | [`atlas`] | shelf packing, atlas building, FcWare compression |
//! | [`shaders`] | GLSL 450 vertex/fragment sources and vertex layouts |
//!
//! # Design notes
//!
//! - `no_std` + `alloc`; the only frame-format dependency is
//!   [`codevar_fcware`].
//! - No top-level renderer is provided yet: [`codevar-ui-core`] owns the
//!   Vulkan pipeline, and this crate exposes everything it needs
//!   (bitmaps, vertices, shader source, compressed atlases).
//! - Every hot loop has a vector kernel selected at run time through
//!   [`codevar_base::basic_cpuid`], following the dispatch style of
//!   [`codevar_base::basic_base64`].

#![cfg_attr(not(test), no_std)]
extern crate alloc;

// TODO(module): pub mod atlas;
pub mod cmap;
pub mod font_file;
pub mod glyf;
#[cfg(test)]
mod test_support;
// TODO(module): pub mod interpreter;
// TODO(module): pub mod raster;
// TODO(module): pub mod shaders;
pub mod simd;
// TODO(module): pub mod subpixel;

use codevar_fcware::CompressorError;
use core::fmt;

/// Result alias used across the Glyphar engine.
///
/// Every fallible engine API returns this alias so callers only ever
/// deal with [`GlypharError`].
pub type GlypharResult<T> = core::result::Result<T, GlypharError>;

/// Errors produced by the Glyphar engine.
///
/// The enum is intentionally `Eq` so tests and callers can match on
/// exact failure modes without string parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlypharError {
    /// The input buffer ended before a structure was fully read.
    ///
    /// `context` names the structure that ran out of bytes (for
    /// example `"cmap format 4 endCode"`).
    Truncated {
        /// Name of the structure that was being read.
        context: &'static str,
    },
    /// A well-sized structure held semantically invalid content.
    Malformed {
        /// Name of the structure that failed validation.
        context: &'static str,
        /// Human-readable detail about the violation.
        detail: &'static str,
    },
    /// The font uses a feature this engine deliberately does not implement.
    Unsupported {
        /// Name of the unsupported feature (for example `"CFF outlines"`).
        feature: &'static str,
    },
    /// An index was outside the legal range for its container.
    OutOfRange {
        /// What was indexed (for example `"glyph id"`).
        what: &'static str,
        /// The rejected index.
        index: u32,
        /// The length of the container.
        len: u32,
    },
    /// A TrueType instruction popped from an empty interpreter stack.
    StackUnderflow {
        /// Number of elements the instruction required.
        needed: u32,
        /// Name of the instruction that underflowed.
        instruction: &'static str,
    },
    /// The interpreter stack exceeded the `maxStackElements` budget.
    StackOverflow {
        /// Configured stack budget in elements.
        limit: u32,
    },
    /// An unknown or reserved opcode was executed.
    InvalidOpcode {
        /// The offending opcode byte.
        opcode: u8,
    },
    /// A jump target landed outside the current code range.
    InvalidJump {
        /// Byte offset the interpreter tried to jump to.
        target: u32,
        /// Length of the code range.
        len: u32,
    },
    /// The instruction budget was exhausted (runaway font program).
    InstructionLimit {
        /// Number of instructions executed before the stop.
        executed: u64,
    },
    /// A hinting setup value was out of range.
    InvalidHinting {
        /// Name of the offending parameter.
        parameter: &'static str,
    },
    /// Rasterization could not proceed with the given geometry.
    Raster {
        /// Description of the rasterization failure.
        reason: &'static str,
    },
    /// The glyph atlas has no room for another shelf/row.
    AtlasFull,
    /// A bitmap or atlas dimension was zero or exceeded limits.
    InvalidDimension {
        /// Name of the dimension that was rejected.
        what: &'static str,
        /// The offending value.
        value: u32,
    },
    /// An FcWare compression or decompression call failed.
    Compress(CompressorError),
}

impl fmt::Display for GlypharError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { context } => write!(f, "truncated font data while reading {context}"),
            Self::Malformed { context, detail } => write!(f, "malformed {context}: {detail}"),
            Self::Unsupported { feature } => write!(f, "unsupported font feature: {feature}"),
            Self::OutOfRange { what, index, len } => {
                write!(f, "{what} index {index} out of range for length {len}")
            }
            Self::StackUnderflow { needed, instruction } => {
                write!(
                    f,
                    "{instruction} needed {needed} stack element(s) but the stack was shorter"
                )
            }
            Self::StackOverflow { limit } => write!(f, "interpreter stack overflow (limit {limit})"),
            Self::InvalidOpcode { opcode } => write!(f, "invalid or reserved opcode 0x{opcode:02X}"),
            Self::InvalidJump { target, len } => {
                write!(f, "jump target {target} outside code range of length {len}")
            }
            Self::InstructionLimit { executed } => {
                write!(f, "instruction budget exhausted after {executed} steps")
            }
            Self::InvalidHinting { parameter } => write!(f, "invalid hinting parameter: {parameter}"),
            Self::Raster { reason } => write!(f, "rasterization failed: {reason}"),
            Self::AtlasFull => f.write_str("glyph atlas is full"),
            Self::InvalidDimension { what, value } => write!(f, "invalid {what}: {value}"),
            Self::Compress(err) => write!(f, "atlas compression failed: {err}"),
        }
    }
}

impl core::error::Error for GlypharError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Compress(err) => Some(err),
            _ => None,
        }
    }
}

impl From<CompressorError> for GlypharError {
    #[inline]
    fn from(err: CompressorError) -> Self {
        Self::Compress(err)
    }
}
