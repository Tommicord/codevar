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

//! `COLR`: color glyph layers and paint graphs.
//!
//! A faithful port of the table validation and the v0 layer iterator of
//! FreeType 2.13's `src/sfnt/ttcolr.c`
//! (`tt_face_load_colr`, `find_base_glyph_record`,
//! `tt_face_get_colr_layer`, `tt_face_get_colr_glyph_paint`).  The table is validated once in
//! [`ColrTable::parse`]; after that [`ColrTable::layers`] walks the
//! base glyph's layer records exactly like `FT_Get_Color_Glyph_Layer`, and [`ColrTable::paint`] evaluates
//! the v1 paint graph exactly like `tt_face_get_colr_glyph_paint`.
//!
//! ## Scope
//!
//! * `COLR` version 0 layers are rendered via [`ColrTable::layers`]
//!   (see [`crate::render`]).
//! * `COLR` version 1 paint graphs are evaluated via
//!   [`ColrTable::paint`] (see [`crate::render::ColorFont::render_paint`]).
//!   All paint formats from the OpenType specification are supported:
//!   solid, linear/radial/sweep gradients, glyph references, transform,
//!   translate, scale, rotate, skew, composite, and color lines.
//!   Clip boxes and item variation stores are validated but not yet
//!   applied during rendering.
//!
//! ## COLR v1 Paint Formats
//!
//! This module implements all paint formats from the OpenType COLR v1
//! specification:
//!
//! | Format | Variant | Description |
//! |--------|---------|-------------|
//! | 1 | [`PaintFormat::ColrLayers`] | Layer list with `SrcOver` compositing |
//! | 2 | [`PaintFormat::Solid`] | Solid color fill |
//! | 4 | [`PaintFormat::LinearGradient`] | Linear gradient |
//! | 6 | [`PaintFormat::RadialGradient`] | Radial gradient |
//! | 8 | [`PaintFormat::SweepGradient`] | Sweep (conic) gradient |
//! | 10 | [`PaintFormat::Glyph`] | Glyph outline filled with paint |
//! | 11 | [`PaintFormat::ColrGlyph`] | Reference to another base glyph's paint |
//! | 12 | [`PaintFormat::Transform`] | Affine transform of child paint |
//! | 14 | [`PaintFormat::Translate`] | Translation of child paint |
//! | 16 | [`PaintFormat::Scale`] | Scale (uniform/non-uniform, centered) of child paint |
//! | 24 | [`PaintFormat::Rotate`] | Rotation (centered) of child paint |
//! | 28 | [`PaintFormat::Skew`] | Skew (centered) of child paint |
//! | 32 | [`PaintFormat::Composite`] | Porter-Duff compositing of two paints |

use codevar_truetype_core::{TtError, TtResult};

use crate::cursor::{Reader, read_u16_at, read_u32_at};

/// `BASE_GLYPH_SIZE`: one v0 `BaseGlyphRecord` (gid, firstLayerIndex,
/// numLayers) is six bytes.
const BASE_GLYPH_SIZE: usize = 6;
/// `LAYER_SIZE`: one v0 `LayerRecord` (gid, paletteIndex) is four
/// bytes.
const LAYER_SIZE: usize = 4;
/// `BASE_GLYPH_PAINT_RECORD_SIZE`: one v1 `BaseGlyphV1Record` (gid,
/// paint offset) is six bytes.
const BASE_GLYPH_PAINT_RECORD_SIZE: usize = 6;
/// `LAYER_V1_LIST_PAINT_OFFSET_SIZE`: one v1 `LayerList` paint offset
/// is four bytes.
const LAYER_V1_LIST_PAINT_OFFSET_SIZE: usize = 4;
/// `COLRV0_HEADER_SIZE`: version, numBaseGlyphRecords,
/// baseGlyphRecordsOffset, layerRecordsOffset, numLayerRecords.
const COLRV0_HEADER_SIZE: usize = 14;
/// `COLRV1_HEADER_SIZE`: the v0 header plus five v1 `Offset32`
/// fields (baseGlyphs, layerList, clipList, varIdxMap, varStore).
const COLRV1_HEADER_SIZE: usize = 34;

/// The `0xFFFF` palette index meaning "the text foreground color"
/// (`FT_COLR_PAINT_FOREGROUND`).
pub const COLOR_FOREGROUND: u16 = 0xFFFF;

/// Paint format identifiers from the OpenType COLR v1 specification.
/// Values match the spec format numbers; internal variable formats
/// (13, 15, 17-23, 25-27, 29-31, 33) are accepted during parsing and
/// resolved to their public equivalents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PaintFormat {
    /// `PaintColrLayers` (format 1): layer list with `SrcOver` compositing.
    ColrLayers = 1,
    /// `PaintSolid` (format 2): solid color fill.
    Solid = 2,
    /// `PaintLinearGradient` (format 4): linear gradient.
    LinearGradient = 4,
    /// `PaintRadialGradient` (format 6): radial gradient.
    RadialGradient = 6,
    /// `PaintSweepGradient` (format 8): sweep (conic) gradient.
    SweepGradient = 8,
    /// `PaintGlyph` (format 10): glyph outline filled with paint.
    Glyph = 10,
    /// `PaintColorGlyph` (format 11): reference to another base glyph's paint.
    ColrGlyph = 11,
    /// `PaintTransform` (format 12): affine transform of child paint.
    Transform = 12,
    /// `PaintTranslate` (format 14): translation of child paint.
    Translate = 14,
    /// `PaintScale` (format 16): scale (uniform/non-uniform, centered) of child paint.
    Scale = 16,
    /// `PaintRotate` (format 24): rotation (centered) of child paint.
    Rotate = 24,
    /// `PaintSkew` (format 28): skew (centered) of child paint.
    Skew = 28,
    /// `PaintComposite` (format 32): Porter-Duff compositing of two paints.
    Composite = 32,
}

impl PaintFormat {
    /// Returns `true` if this format is a valid COLR v1 paint format.
    #[inline]
    pub const fn is_valid(self) -> bool {
        matches!(
            self,
            Self::ColrLayers
                | Self::Solid
                | Self::LinearGradient
                | Self::RadialGradient
                | Self::SweepGradient
                | Self::Glyph
                | Self::ColrGlyph
                | Self::Transform
                | Self::Translate
                | Self::Scale
                | Self::Rotate
                | Self::Skew
                | Self::Composite
        )
    }

    /// Creates a `PaintFormat` from a raw byte, returning `None` for
    /// unknown or unsupported formats.
    ///
    /// Accepts both public format numbers and FreeType's internal
    /// variable format numbers (which are resolved to public formats).
    #[inline]
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::ColrLayers),
            2 => Some(Self::Solid),
            4 => Some(Self::LinearGradient),
            6 => Some(Self::RadialGradient),
            8 => Some(Self::SweepGradient),
            10 => Some(Self::Glyph),
            11 => Some(Self::ColrGlyph),
            12 => Some(Self::Transform),
            14 => Some(Self::Translate),
            16 => Some(Self::Scale),
            17 => Some(Self::Scale), // internal: PaintScaleUniform
            18 => Some(Self::Scale), // internal: PaintScaleCenter
            19 => Some(Self::Scale), // internal: PaintScaleCenter (var)
            20 => Some(Self::Scale), // internal: PaintScaleUniform
            21 => Some(Self::Scale), // internal: PaintScaleUniform (var)
            22 => Some(Self::Scale), // internal: PaintScaleUniformCenter
            23 => Some(Self::Scale), // internal: PaintScaleUniformCenter (var)
            24 => Some(Self::Rotate),
            25 => Some(Self::Rotate), // internal: PaintRotate (var)
            26 => Some(Self::Rotate), // internal: PaintRotateCenter
            27 => Some(Self::Rotate), // internal: PaintRotateCenter (var)
            28 => Some(Self::Skew),
            29 => Some(Self::Skew), // internal: PaintSkew (var)
            30 => Some(Self::Skew), // internal: PaintSkewCenter
            31 => Some(Self::Skew), // internal: PaintSkewCenter (var)
            32 => Some(Self::Composite),
            33 => Some(Self::Glyph), // internal: PaintGlyph2 (Offset32 glyph ID)
            _ => None,
        }
    }

    /// Returns `true` if this format is a uniform scale (scale_x == scale_y).
    /// Used internally to distinguish `PaintScaleUniform` from `PaintScale`.
    #[inline]
    pub const fn is_uniform_scale(self) -> bool {
        matches!(self, Self::Scale)
    }

    /// Returns `true` if this format has a center point.
    /// Used internally to distinguish centered variants.
    #[inline]
    pub const fn has_center(self) -> bool {
        false
    }
}

/// Variant flags for scale/rotate/skew paints that have center coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformVariant {
    /// No center, non-uniform (format 16 Scale).
    None,
    /// Uniform scale (scale_x == scale_y), no center (format 17/20).
    Uniform,
    /// With center coordinates (format 18/19/22/23 for scale,
    /// format 26/27 for rotate, format 30/31 for skew).
    Center,
    /// Uniform with center (format 22/23 for scale).
    UniformCenter,
}

/// Gradient extend mode (`FT_PaintExtend`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PaintExtend {
    /// Clamp to the edge color (`FT_COLR_PAINT_EXTEND_PAD`).
    Pad = 0,
    /// Repeat the gradient (`FT_COLR_PAINT_EXTEND_REPEAT`).
    Repeat = 1,
    /// Reflect the gradient (`FT_COLR_PAINT_EXTEND_REFLECT`).
    Reflect = 2,
}

impl PaintExtend {
    /// Creates a `PaintExtend` from a raw byte, returning `None` for
    /// invalid values.
    #[inline]
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Pad),
            1 => Some(Self::Repeat),
            2 => Some(Self::Reflect),
            _ => None,
        }
    }
}

/// Composite mode for `PaintComposite` (`FT_Composite_Mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CompositeMode {
    /// `FT_COLR_COMPOSITE_CLEAR` (0): clear the destination.
    Clear = 0,
    /// `FT_COLR_COMPOSITE_SRC` (1): source only.
    Src = 1,
    /// `FT_COLR_COMPOSITE_DEST` (2): destination only.
    Dest = 2,
    /// `FT_COLR_COMPOSITE_SRC_OVER` (3): source over destination.
    SrcOver = 3,
    /// `FT_COLR_COMPOSITE_DEST_OVER` (4): destination over source.
    DestOver = 4,
    /// `FT_COLR_COMPOSITE_SRC_IN` (5): source in destination.
    SrcIn = 5,
    /// `FT_COLR_COMPOSITE_DEST_IN` (6): destination in source.
    DestIn = 6,
    /// `FT_COLR_COMPOSITE_SRC_OUT` (7): source out destination.
    SrcOut = 7,
    /// `FT_COLR_COMPOSITE_DEST_OUT` (8): destination out source.
    DestOut = 8,
    /// `FT_COLR_COMPOSITE_SRC_ATOP` (9): source atop destination.
    SrcAtop = 9,
    /// `FT_COLR_COMPOSITE_DEST_ATOP` (10): destination atop source.
    DestAtop = 10,
    /// `FT_COLR_COMPOSITE_XOR` (11): XOR of source and destination.
    Xor = 11,
    /// `FT_COLR_COMPOSITE_PLUS` (12): source plus destination.
    Plus = 12,
    /// `FT_COLR_COMPOSITE_SCREEN` (13): screen blend.
    Screen = 13,
    /// `FT_COLR_COMPOSITE_OVERLAY` (14): overlay blend.
    Overlay = 14,
    /// `FT_COLR_COMPOSITE_DARKEN` (15): darken blend.
    Darken = 15,
    /// `FT_COLR_COMPOSITE_LIGHTEN` (16): lighten blend.
    Lighten = 16,
    /// `FT_COLR_COMPOSITE_COLOR_DODGE` (17): color dodge blend.
    ColorDodge = 17,
    /// `FT_COLR_COMPOSITE_COLOR_BURN` (18): color burn blend.
    ColorBurn = 18,
    /// `FT_COLR_COMPOSITE_HARD_LIGHT` (19): hard light blend.
    HardLight = 19,
    /// `FT_COLR_COMPOSITE_SOFT_LIGHT` (20): soft light blend.
    SoftLight = 20,
    /// `FT_COLR_COMPOSITE_DIFFERENCE` (21): difference blend.
    Difference = 21,
    /// `FT_COLR_COMPOSITE_EXCLUSION` (22): exclusion blend.
    Exclusion = 22,
    /// `FT_COLR_COMPOSITE_MULTIPLY` (23): multiply blend.
    Multiply = 23,
    /// `FT_COLR_COMPOSITE_HSL_HUE` (24): HSL hue blend.
    HslHue = 24,
    /// `FT_COLR_COMPOSITE_HSL_SATURATION` (25): HSL saturation blend.
    HslSaturation = 25,
    /// `FT_COLR_COMPOSITE_HSL_COLOR` (26): HSL color blend.
    HslColor = 26,
    /// `FT_COLR_COMPOSITE_HSL_LUMINOSITY` (27): HSL luminosity blend.
    HslLuminosity = 27,
}

impl CompositeMode {
    /// Creates a `CompositeMode` from a raw byte, returning `None` for
    /// invalid values.
    #[inline]
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Clear),
            1 => Some(Self::Src),
            2 => Some(Self::Dest),
            3 => Some(Self::SrcOver),
            4 => Some(Self::DestOver),
            5 => Some(Self::SrcIn),
            6 => Some(Self::DestIn),
            7 => Some(Self::SrcOut),
            8 => Some(Self::DestOut),
            9 => Some(Self::SrcAtop),
            10 => Some(Self::DestAtop),
            11 => Some(Self::Xor),
            12 => Some(Self::Plus),
            13 => Some(Self::Screen),
            14 => Some(Self::Overlay),
            15 => Some(Self::Darken),
            16 => Some(Self::Lighten),
            17 => Some(Self::ColorDodge),
            18 => Some(Self::ColorBurn),
            19 => Some(Self::HardLight),
            20 => Some(Self::SoftLight),
            21 => Some(Self::Difference),
            22 => Some(Self::Exclusion),
            23 => Some(Self::Multiply),
            24 => Some(Self::HslHue),
            25 => Some(Self::HslSaturation),
            26 => Some(Self::HslColor),
            27 => Some(Self::HslLuminosity),
            _ => None,
        }
    }
}

/// Color index with alpha multiplier (`FT_ColorIndex`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorIndex {
    /// Palette index (0xFFFF = foreground color).
    pub palette_index: u16,
    /// Alpha multiplier in F2Dot14 format (16-bit signed fixed-point).
    pub alpha: i16,
}

/// Color stop for gradients (`FT_ColorStop`).
#[derive(Clone, Copy, Debug)]
pub struct ColorStop {
    /// Stop offset along the gradient (16.16 fixed-point).
    pub stop_offset: i32,
    /// Color and alpha for this stop.
    pub color: ColorIndex,
}

/// Color line (gradient stop list) (`FT_ColorLine`).
#[derive(Clone, Debug)]
pub struct ColorLine {
    /// Extend mode at boundaries.
    pub extend: PaintExtend,
    /// The color stops.
    pub stops: alloc::vec::Vec<ColorStop>,
}

/// Opaque reference to a paint table (`FT_OpaquePaint`).
///
/// This mirrors FreeType's opaque paint handle: it holds an offset into
/// the COLR table's paint data. The actual paint is resolved on demand
/// via [`ColrTable::paint`].
#[derive(Clone, Copy, Debug)]
pub struct OpaquePaint {
    /// Offset from the start of the paint data region.
    offset: u32,
    /// Whether to insert the root transform when resolving.
    insert_root_transform: bool,
}

impl OpaquePaint {
    /// Creates a new opaque paint reference.
    #[inline]
    pub const fn new(offset: u32, insert_root_transform: bool) -> Self {
        Self {
            offset,
            insert_root_transform,
        }
    }

    /// The paint offset.
    #[inline]
    pub const fn offset(&self) -> u32 {
        self.offset
    }

    /// Whether to insert the root transform.
    #[inline]
    pub const fn insert_root_transform(&self) -> bool {
        self.insert_root_transform
    }
}

/// A parsed paint table (`FT_COLR_Paint`).
///
/// This is the main paint representation, mirroring FreeType's
/// `FT_COLR_Paint` union. Each variant contains the fully resolved
/// paint data for that format.
#[derive(Clone, Debug)]
pub enum Paint<'a> {
    /// `PaintColrLayers`: a list of layers composited with `SrcOver`.
    ColrLayers(ColrLayersPaint<'a>),
    /// `PaintSolid`: a solid color fill.
    Solid(SolidPaint),
    /// `PaintLinearGradient`: a linear gradient.
    LinearGradient(LinearGradientPaint),
    /// `PaintRadialGradient`: a radial gradient.
    RadialGradient(RadialGradientPaint),
    /// `PaintSweepGradient`: a sweep (conic) gradient.
    SweepGradient(SweepGradientPaint),
    /// `PaintGlyph`: a glyph outline filled with paint.
    Glyph(GlyphPaint),
    /// `PaintColorGlyph`: reference to another base glyph's paint.
    ColrGlyph(ColrGlyphPaint),
    /// `PaintTransform`: affine transform of child paint.
    Transform(TransformPaint),
    /// `PaintTranslate`: translation of child paint.
    Translate(TranslatePaint),
    /// `PaintScale`: scale of child paint.
    Scale(ScalePaint),
    /// `PaintRotate`: rotation of child paint.
    Rotate(RotatePaint),
    /// `PaintSkew`: skew of child paint.
    Skew(SkewPaint),
    /// `PaintComposite`: Porter-Duff compositing of two paints.
    Composite(CompositePaint),
}

/// `PaintColrLayers` (format 1).
#[derive(Clone, Debug)]
pub struct ColrLayersPaint<'a> {
    /// Layer iterator for the layer list.
    pub layer_iter: LayerV1Iter<'a>,
}

/// `PaintSolid` (format 2).
#[derive(Clone, Copy, Debug)]
pub struct SolidPaint {
    /// Color index and alpha.
    pub color: ColorIndex,
}

/// `PaintLinearGradient` (format 4).
#[derive(Clone, Debug)]
pub struct LinearGradientPaint {
    /// Gradient color line.
    pub color_line: ColorLine,
    /// Start point p0 (16.16 fixed-point).
    pub p0: (i32, i32),
    /// End point p1 (16.16 fixed-point).
    pub p1: (i32, i32),
    /// Rotation point p2 (16.16 fixed-point).
    pub p2: (i32, i32),
}

/// `PaintRadialGradient` (format 6).
#[derive(Clone, Debug)]
pub struct RadialGradientPaint {
    /// Gradient color line.
    pub color_line: ColorLine,
    /// Start circle center c0 (16.16 fixed-point).
    pub c0: (i32, i32),
    /// Start circle radius r0 (16.16 fixed-point, non-negative).
    pub r0: i32,
    /// End circle center c1 (16.16 fixed-point).
    pub c1: (i32, i32),
    /// End circle radius r1 (16.16 fixed-point, non-negative).
    pub r1: i32,
}

/// `PaintSweepGradient` (format 8).
#[derive(Clone, Debug)]
pub struct SweepGradientPaint {
    /// Gradient color line.
    pub color_line: ColorLine,
    /// Center point (16.16 fixed-point).
    pub center: (i32, i32),
    /// Start angle in degrees/180 (16.16 fixed-point).
    pub start_angle: i32,
    /// End angle in degrees/180 (16.16 fixed-point).
    pub end_angle: i32,
}

/// `PaintGlyph` (format 10).
#[derive(Clone, Debug)]
pub struct GlyphPaint {
    /// Child paint for filling the glyph.
    pub paint: OpaquePaint,
    /// Glyph ID from the 'glyf' table.
    pub glyph_id: u16,
}

/// `PaintColorGlyph` (format 11).
#[derive(Clone, Copy, Debug)]
pub struct ColrGlyphPaint {
    /// Base glyph ID to recurse into.
    pub glyph_id: u16,
}

/// `PaintTransform` (format 12).
#[derive(Clone, Debug)]
pub struct TransformPaint {
    /// Child paint to transform.
    pub paint: OpaquePaint,
    /// 2x3 affine matrix in 16.16 fixed-point.
    /// [xx, xy, dx]
    /// [yx, yy, dy]
    pub affine: [i32; 6],
}

/// `PaintTranslate` (format 14).
#[derive(Clone, Copy, Debug)]
pub struct TranslatePaint {
    /// Child paint to translate.
    pub paint: OpaquePaint,
    /// Translation in x (16.16 fixed-point).
    pub dx: i32,
    /// Translation in y (16.16 fixed-point).
    pub dy: i32,
}

/// `PaintScale` (format 16) - covers all Scale variants.
#[derive(Clone, Copy, Debug)]
pub struct ScalePaint {
    /// Child paint to scale.
    pub paint: OpaquePaint,
    /// Scale factor in x (16.16 fixed-point).
    pub scale_x: i32,
    /// Scale factor in y (16.16 fixed-point).
    pub scale_y: i32,
    /// Center x coordinate (16.16 fixed-point).
    pub center_x: i32,
    /// Center y coordinate (16.16 fixed-point).
    pub center_y: i32,
}

/// `PaintRotate` (format 24) - covers Rotate and RotateCenter.
#[derive(Clone, Copy, Debug)]
pub struct RotatePaint {
    /// Child paint to rotate.
    pub paint: OpaquePaint,
    /// Rotation angle in degrees/180 (16.16 fixed-point).
    pub angle: i32,
    /// Center x coordinate (16.16 fixed-point).
    pub center_x: i32,
    /// Center y coordinate (16.16 fixed-point).
    pub center_y: i32,
}

/// `PaintSkew` (format 28) - covers Skew and SkewCenter.
#[derive(Clone, Copy, Debug)]
pub struct SkewPaint {
    /// Child paint to skew.
    pub paint: OpaquePaint,
    /// X skew angle in degrees/180 (16.16 fixed-point).
    pub x_skew_angle: i32,
    /// Y skew angle in degrees/180 (16.16 fixed-point).
    pub y_skew_angle: i32,
    /// Center x coordinate (16.16 fixed-point).
    pub center_x: i32,
    /// Center y coordinate (16.16 fixed-point).
    pub center_y: i32,
}

/// `PaintComposite` (format 32).
#[derive(Clone, Debug)]
pub struct CompositePaint {
    /// Source paint.
    pub source: OpaquePaint,
    /// Composite mode.
    pub mode: CompositeMode,
    /// Backdrop paint.
    pub backdrop: OpaquePaint,
}

/// Layer iterator for v1 `PaintColrLayers` (`FT_LayerIterator` for v1).
#[derive(Clone, Debug)]
pub struct LayerV1Iter<'a> {
    /// The full COLR table data.
    data: &'a [u8],
    /// Number of layers in this list.
    num_layers: u16,
    /// Current layer index.
    current: u16,
    /// Base offset of this layer list's paint offsets within the table.
    list_base: usize,
}

/// Color stop iterator for gradients (`FT_ColorStopIterator`).
#[derive(Clone, Debug)]
pub struct ColorStopIter<'a> {
    /// The color stops to iterate over.
    stops: &'a [ColorStop],
    /// Current stop index.
    current: u16,
    /// Whether to read variable color stops (ignored, no var support).
    _read_variable: bool,
}

/// Root transform mode for paint retrieval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootTransform {
    /// Include the root transform (size + user transform).
    Include,
    /// Do not include the root transform.
    Exclude,
}

/// A parsed `COLR` table, borrowing the font bytes.
#[derive(Clone, Debug)]
pub struct ColrTable<'a> {
    /// The full `COLR` table data (needed for absolute paint offset resolution).
    data: &'a [u8],
    /// The `COLR` version (0 or 1).
    version: u16,
    /// The number of v0 base glyph records.
    num_base_glyphs: u16,
    /// The number of v0 layer records.
    num_layers: u16,
    /// The v0 `BaseGlyphRecords` array (`6 * num_base_glyphs` bytes).
    base_glyphs: &'a [u8],
    /// The v0 `LayerRecords` array (`4 * num_layers` bytes).
    layers: &'a [u8],
    /// `true` when the (validated) v1 header advertises a non-empty
    /// `BaseGlyphV1List`.
    has_base_glyphs_v1: bool,
    /// v1: number of base glyph paint records.
    num_base_glyphs_v1: u32,
    /// v1: start offset of BaseGlyphV1List within the table.
    base_glyphs_v1_offset: usize,
    /// v1: number of layer list entries.
    num_layers_v1: u32,
    /// v1: start offset of LayerV1List within the table.
    layers_v1_offset: usize,
    /// v1: clip list data offset (optional).
    clip_list_offset: usize,
    /// v1: start offset of paint tables within the table.
    paints_start_v1: usize,
}

impl<'a> ColrTable<'a> {
    /// Parses a `COLR` table, mirroring `tt_face_load_colr`
    /// validation for validation (without the GX variation store,
    /// which this port does not model).
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — the table is shorter than its
    ///   header, the version is above 1, an array offset is out of
    ///   range, or an array count overruns the table.
    pub fn parse(data: &'a [u8]) -> TtResult<Self> {
        if data.len() < COLRV0_HEADER_SIZE {
            return Err(TtError::INVALID_TABLE);
        }
        let mut reader = Reader::new(data);
        let version = reader.u16()?;
        if version != 0 && version != 1 {
            return Err(TtError::INVALID_TABLE);
        }
        let num_base_glyphs = reader.u16()?;
        let base_glyph_offset = reader.u32()? as usize;
        let layer_offset = reader.u32()? as usize;
        let num_layers = reader.u16()?;

        let base_glyphs = Self::array(data, base_glyph_offset, num_base_glyphs, BASE_GLYPH_SIZE)?;
        let layers = Self::array(data, layer_offset, num_layers, LAYER_SIZE)?;

        let mut has_base_glyphs_v1 = false;
        let mut base_glyphs_v1_offset = 0usize;
        let mut num_base_glyphs_v1 = 0u32;
        let mut layers_v1_offset = 0usize;
        let mut num_layers_v1 = 0u32;
        let mut clip_list_offset = 0usize;
        let mut paints_start_v1 = 0usize;

        if version == 1 {
            if data.len() < COLRV1_HEADER_SIZE {
                return Err(TtError::INVALID_TABLE);
            }
            // BaseGlyphV1List: an Offset32 whose target must hold a
            // u32 count inside the table (`table_size - 4 <= offset`
            // is FreeType's rejection test).
            base_glyphs_v1_offset = read_u32_at(data, 14)? as usize;
            if data.len() - 4 <= base_glyphs_v1_offset {
                return Err(TtError::INVALID_TABLE);
            }
            let count = read_u32_at(data, base_glyphs_v1_offset)?;
            if (data.len() - base_glyphs_v1_offset) / BASE_GLYPH_PAINT_RECORD_SIZE < count as usize {
                return Err(TtError::INVALID_TABLE);
            }
            num_base_glyphs_v1 = count;
            has_base_glyphs_v1 = count != 0;

            // LayerList (optional): u32 count followed by Offset32s.
            layers_v1_offset = read_u32_at(data, 18)? as usize;
            if data.len() <= layers_v1_offset {
                return Err(TtError::INVALID_TABLE);
            }
            if layers_v1_offset != 0 {
                if data.len() - 4 <= layers_v1_offset {
                    return Err(TtError::INVALID_TABLE);
                }
                let count = read_u32_at(data, layers_v1_offset)?;
                if (data.len() - layers_v1_offset) / LAYER_V1_LIST_PAINT_OFFSET_SIZE < count as usize {
                    return Err(TtError::INVALID_TABLE);
                }
                num_layers_v1 = count;
            }

            // ClipList (optional).
            clip_list_offset = read_u32_at(data, 22)? as usize;
            if data.len() <= clip_list_offset {
                return Err(TtError::INVALID_TABLE);
            }

            // VarIdxMap and VarStore offsets (ignored, no variation support).
            let _var_idx_map_offset = read_u32_at(data, 26)?;
            let _var_store_offset = read_u32_at(data, 30)?;

            // Compute paints_start_v1: minimum of end of BaseGlyphV1List
            // and end of LayerV1List (if present), matching FreeType.
            let base_glyphs_v1_end =
                base_glyphs_v1_offset + 4 + (num_base_glyphs_v1 as usize) * BASE_GLYPH_PAINT_RECORD_SIZE;
            if num_layers_v1 > 0 {
                let layers_v1_end =
                    layers_v1_offset + 4 + (num_layers_v1 as usize) * LAYER_V1_LIST_PAINT_OFFSET_SIZE;
                paints_start_v1 = base_glyphs_v1_end.min(layers_v1_end);
            } else {
                paints_start_v1 = base_glyphs_v1_end;
            }
        }

        Ok(ColrTable {
            data,
            version,
            num_base_glyphs,
            num_layers,
            base_glyphs,
            layers,
            has_base_glyphs_v1,
            num_base_glyphs_v1,
            base_glyphs_v1_offset,
            num_layers_v1,
            layers_v1_offset,
            clip_list_offset,
            paints_start_v1,
        })
    }

    /// Validates one record array: `offset + count * size` must stay
    /// inside `data`, mirroring FreeType's
    /// `table_size <= offset || (table_size - offset) / size < count`
    /// checks.
    fn array(data: &'a [u8], offset: usize, count: u16, size: usize) -> TtResult<&'a [u8]> {
        if offset > data.len() {
            return Err(TtError::INVALID_TABLE);
        }
        let bytes = count as usize;
        if data.len() - offset < bytes.saturating_mul(size) {
            return Err(TtError::INVALID_TABLE);
        }
        let length = bytes
            .checked_mul(size)
            .ok_or(TtError::INVALID_TABLE)?;
        data.get(offset..offset + length)
            .ok_or(TtError::INVALID_TABLE)
    }

    /// The `COLR` version (0 or 1).
    #[inline]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// The number of v0 base glyph records.
    #[inline]
    pub const fn num_base_glyphs(&self) -> u16 {
        self.num_base_glyphs
    }

    /// The number of v0 layer records.
    #[inline]
    pub const fn num_layers(&self) -> u16 {
        self.num_layers
    }

    /// `true` when the table also carries a validated (non-empty)
    /// v1 `BaseGlyphV1List`.  Paint rendering is not implemented;
    /// see the module documentation.
    #[inline]
    pub const fn has_v1_paints(&self) -> bool {
        self.has_base_glyphs_v1
    }

    /// Binary-searches the v0 base glyph record for `gindex`
    /// (`find_base_glyph_record`), returning
    /// `(first_layer_index, num_layers)`.
    ///
    /// Returns `None` when the glyph has no v0 color layers.
    pub fn base_glyph(&self, gindex: u32) -> Option<(u16, u16)> {
        if gindex > u16::MAX as u32 {
            return None;
        }
        let gid = gindex as u16;
        let mut min = 0usize;
        let mut max = self.num_base_glyphs as usize;
        while min < max {
            let mid = min + (max - min) / 2;
            let record = self.base_glyphs.get(mid * BASE_GLYPH_SIZE..)?;
            let record_gid = u16::from_be_bytes(record[0..2].try_into().unwrap_or([0, 0]));
            if record_gid < gid {
                min = mid + 1;
            } else if record_gid > gid {
                max = mid;
            } else {
                let first = u16::from_be_bytes(record[2..4].try_into().unwrap_or([0, 0]));
                let num = u16::from_be_bytes(record[4..6].try_into().unwrap_or([0, 0]));
                return Some((first, num));
            }
        }
        None
    }

    /// Layer record `index` of the v0 layer array, as
    /// `(glyph_index, palette_index)`.
    ///
    /// Returns `None` when the record is outside the validated array
    /// (FreeType stops the layer iteration in that case).
    pub fn layer(&self, index: u16) -> Option<(u16, u16)> {
        let record = self.layers.get((index as usize) * LAYER_SIZE..)?;
        let glyph = u16::from_be_bytes(record[0..2].try_into().unwrap_or([0, 0]));
        let color = u16::from_be_bytes(record[2..4].try_into().unwrap_or([0, 0]));
        Some((glyph, color))
    }

    /// Starts a v0 layer iteration over `gindex`, or returns `None`
    /// when the glyph has no layers (`FT_Get_Color_Glyph_Layer`'s
    /// first call).
    ///
    /// The iterator stops early when a record points at a glyph index
    /// `>= num_glyphs` or at a palette index `>= num_palette_entries`
    /// (other than [`COLOR_FOREGROUND`]), reproducing
    /// `tt_face_get_colr_layer`'s per-record validation.
    pub fn layers(&self, gindex: u32, num_glyphs: u16, num_palette_entries: u16) -> Option<LayerIter<'a>> {
        let (first, num) = self.base_glyph(gindex)?;
        if num == 0 {
            return None;
        }
        // FreeType rejects layer ranges that run past the table; this
        // port checks them against the validated layer array instead
        // (stricter, same effect for well-formed tables).
        let end = (first as u32).checked_add(num as u32)?;
        if end > self.num_layers as u32 {
            return None;
        }
        Some(LayerIter {
            layers: self.layers,
            next: first,
            remaining: num,
            num_glyphs,
            num_palette_entries,
        })
    }

    // ===== COLR v1 Paint Accessors =====

    /// Returns the number of v1 base glyph paint records.
    #[inline]
    pub const fn num_base_glyphs_v1(&self) -> u32 {
        self.num_base_glyphs_v1
    }

    /// Returns the number of v1 layer list entries.
    #[inline]
    pub const fn num_layers_v1(&self) -> u32 {
        self.num_layers_v1
    }

    /// Returns the v1 clip list data offset, or 0 if absent.
    #[inline]
    pub const fn clip_list_offset(&self) -> usize {
        self.clip_list_offset
    }

    /// Finds the v1 base glyph paint record for `gindex`
    /// (`find_base_glyph_v1_record`), returning the paint offset.
    ///
    /// Returns `None` when the glyph has no v1 paint.
    pub fn base_glyph_v1(&self, gindex: u32) -> Option<u32> {
        if gindex > u16::MAX as u32 || self.num_base_glyphs_v1 == 0 {
            return None;
        }
        let gid = gindex as u16;
        let mut min = 0usize;
        let mut max = self.num_base_glyphs_v1 as usize;
        // BaseGlyphV1List: first 4 bytes are count, then records.
        while min < max {
            let mid = min + (max - min) / 2;
            let record_offset = self.base_glyphs_v1_offset + 4 + mid * BASE_GLYPH_PAINT_RECORD_SIZE;
            let record = self
                .data
                .get(record_offset..record_offset + BASE_GLYPH_PAINT_RECORD_SIZE)?;
            let record_gid = u16::from_be_bytes(record[0..2].try_into().unwrap_or([0, 0]));
            if record_gid < gid {
                min = mid + 1;
            } else if record_gid > gid {
                max = mid;
            } else {
                let paint_offset = u32::from_be_bytes(record[2..6].try_into().unwrap_or([0, 0, 0, 0]));
                // Paint offset is relative to the start of BaseGlyphV1List.
                // Convert to absolute offset into the COLR table.
                return Some(self.base_glyphs_v1_offset as u32 + paint_offset);
            }
        }
        None
    }

    /// Starts a v1 layer iteration for a `PaintColrLayers` paint
    /// (`tt_face_get_paint_layers`).
    pub fn layer_v1_iter(&self, layer_list_offset: u32) -> Option<LayerV1Iter<'a>> {
        if layer_list_offset == 0 {
            return None;
        }
        // layer_list_offset is relative to start of LayerV1List.
        let list_base = layer_list_offset as usize;
        if list_base + 4 > self.data.len() {
            return None;
        }
        let num_layers = u32::from_be_bytes(
            self.data[list_base..list_base + 4]
                .try_into()
                .unwrap_or([0, 0, 0, 0]),
        ) as u16;
        if num_layers == 0 {
            return None;
        }
        // Paint offsets follow the count (4 bytes each).
        let paint_offsets_start = list_base + 4;
        if paint_offsets_start + (num_layers as usize) * 4 > self.data.len() {
            return None;
        }
        Some(LayerV1Iter {
            data: self.data,
            num_layers,
            current: 0,
            list_base: paint_offsets_start,
        })
    }

    /// Resolves a paint table at `offset` from the paint data region
    /// (`tt_face_get_paint` + `read_paint`).
    ///
    /// The `offset` is an absolute offset into `data` (paint table offsets
    /// in the COLR v1 table are absolute within the COLR table).
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_TABLE`] — the paint offset is invalid or
    ///   the paint data is malformed.
    pub fn paint(&self, opaque: OpaquePaint, root_transform: RootTransform) -> TtResult<Paint<'a>> {
        // Handle root transform insertion.
        if opaque.insert_root_transform() && root_transform == RootTransform::Include {
            // The root transform is inserted by the caller by wrapping
            // the returned paint in a TransformPaint.
        }

        let paint_offset = opaque.offset() as usize;
        if paint_offset < self.paints_start_v1 || paint_offset >= self.data.len() {
            return Err(TtError::INVALID_TABLE);
        }
        let paint_data = &self.data[paint_offset..];
        if paint_data.is_empty() {
            return Err(TtError::INVALID_TABLE);
        }
        let format_byte = paint_data[0];
        let mut reader = Reader::new(paint_data);
        Self::read_paint(self, &mut reader, paint_data, opaque, format_byte)
    }

    /// Internal paint parsing logic (mirrors FreeType's `read_paint`).
    fn read_paint(
        &self,
        reader: &mut Reader<'_>,
        paint_base: &'a [u8],
        _opaque: OpaquePaint,
        format_byte: u8,
    ) -> TtResult<Paint<'a>> {
        // Ensure we can read at least the format byte.
        if paint_base.is_empty() {
            return Err(TtError::INVALID_TABLE);
        }
        let format = PaintFormat::from_byte(format_byte).ok_or(TtError::INVALID_TABLE)?;

        // Advance reader past format byte.
        reader.u8()?;

        match format {
            PaintFormat::ColrLayers => Self::read_colr_layers(self, reader),
            PaintFormat::Solid => Self::read_solid(self, reader),
            PaintFormat::LinearGradient => Self::read_linear_gradient(self, reader, paint_base),
            PaintFormat::RadialGradient => Self::read_radial_gradient(self, reader, paint_base),
            PaintFormat::SweepGradient => Self::read_sweep_gradient(self, reader, paint_base),
            PaintFormat::Glyph => Self::read_glyph(self, reader, paint_base, format_byte),
            PaintFormat::ColrGlyph => Self::read_colr_glyph(self, reader),
            PaintFormat::Transform => Self::read_transform(self, reader, paint_base),
            PaintFormat::Translate => Self::read_translate(self, reader, paint_base),
            PaintFormat::Scale => Self::read_scale(self, reader, paint_base, format_byte),
            PaintFormat::Rotate => Self::read_rotate(self, reader, paint_base, format_byte),
            PaintFormat::Skew => Self::read_skew(self, reader, paint_base, format_byte),
            PaintFormat::Composite => Self::read_composite(self, reader, paint_base),
        }
    }

    fn read_colr_layers(&self, reader: &mut Reader<'_>) -> TtResult<Paint<'a>> {
        // numLayers (uint8) + firstLayerIndex (Offset32)
        if reader.remaining() < 5 {
            return Err(TtError::INVALID_TABLE);
        }
        let num_layers = reader.u8()? as u16;
        let first_layer_index = reader.u32()?;

        // Validate against the LayerV1List.
        if self.layers_v1_offset == 0 {
            return Err(TtError::INVALID_TABLE);
        }
        if first_layer_index >= self.num_layers_v1 {
            return Err(TtError::INVALID_TABLE);
        }
        if first_layer_index.saturating_add(num_layers as u32) > self.num_layers_v1 {
            return Err(TtError::INVALID_TABLE);
        }

        // Create layer iterator.
        let list_base = self.layers_v1_offset + 4 + (first_layer_index as usize) * 4;
        let iter = LayerV1Iter {
            data: self.data,
            num_layers,
            current: 0,
            list_base,
        };
        Ok(Paint::ColrLayers(ColrLayersPaint { layer_iter: iter }))
    }

    fn read_solid(&self, reader: &mut Reader<'_>) -> TtResult<Paint<'a>> {
        // paletteIndex (uint16) + alpha (F2Dot14)
        if reader.remaining() < 4 {
            return Err(TtError::INVALID_TABLE);
        }
        let palette_index = reader.u16()?;
        let alpha = reader.i16()?;
        Ok(Paint::Solid(SolidPaint {
            color: ColorIndex { palette_index, alpha },
        }))
    }

    fn read_linear_gradient(&self, reader: &mut Reader<'_>, paint_base: &'a [u8]) -> TtResult<Paint<'a>> {
        // colorLine (Offset24) + p0.x + p0.y + p1.x + p1.y + p2.x + p2.y (each int16)
        if reader.remaining() < 3 + 12 {
            return Err(TtError::INVALID_TABLE);
        }
        let color_line_offset = reader.u24()? as usize;
        let color_line = Self::read_color_line(self, paint_base, color_line_offset, false)?;

        let p0_x = reader.i16()? as i32;
        let p0_y = reader.i16()? as i32;
        let p1_x = reader.i16()? as i32;
        let p1_y = reader.i16()? as i32;
        let p2_x = reader.i16()? as i32;
        let p2_y = reader.i16()? as i32;

        Ok(Paint::LinearGradient(LinearGradientPaint {
            color_line,
            p0: (p0_x << 16, p0_y << 16),
            p1: (p1_x << 16, p1_y << 16),
            p2: (p2_x << 16, p2_y << 16),
        }))
    }

    fn read_radial_gradient(&self, reader: &mut Reader<'_>, paint_base: &'a [u8]) -> TtResult<Paint<'a>> {
        // colorLine (Offset24) + c0.x + c0.y + r0 + c1.x + c1.y + r1 (each int16)
        if reader.remaining() < 3 + 12 {
            return Err(TtError::INVALID_TABLE);
        }
        let color_line_offset = reader.u24()? as usize;
        let color_line = Self::read_color_line(self, paint_base, color_line_offset, false)?;

        let c0_x = reader.i16()? as i32;
        let c0_y = reader.i16()? as i32;
        let r0 = reader.i16()? as i32;
        let c1_x = reader.i16()? as i32;
        let c1_y = reader.i16()? as i32;
        let r1 = reader.i16()? as i32;

        // r0 and r1 are UFWORD in spec; negative becomes max.
        let r0 = if r0 < 0 { i32::MAX } else { r0 << 16 };
        let r1 = if r1 < 0 { i32::MAX } else { r1 << 16 };

        Ok(Paint::RadialGradient(RadialGradientPaint {
            color_line,
            c0: (c0_x << 16, c0_y << 16),
            r0,
            c1: (c1_x << 16, c1_y << 16),
            r1,
        }))
    }

    fn read_sweep_gradient(&self, reader: &mut Reader<'_>, paint_base: &'a [u8]) -> TtResult<Paint<'a>> {
        // colorLine (Offset24) + center.x + center.y + startAngle + endAngle (each int16)
        if reader.remaining() < 3 + 8 {
            return Err(TtError::INVALID_TABLE);
        }
        let color_line_offset = reader.u24()? as usize;
        let color_line = Self::read_color_line(self, paint_base, color_line_offset, false)?;

        let center_x = reader.i16()? as i32;
        let center_y = reader.i16()? as i32;
        let start_angle = reader.i16()? as i32;
        let end_angle = reader.i16()? as i32;

        Ok(Paint::SweepGradient(SweepGradientPaint {
            color_line,
            center: (center_x << 16, center_y << 16),
            start_angle: start_angle << 16,
            end_angle: end_angle << 16,
        }))
    }

    fn read_glyph(
        &self,
        reader: &mut Reader<'_>,
        paint_base: &'a [u8],
        format_byte: u8,
    ) -> TtResult<Paint<'a>> {
        // paint (Offset24) + glyphID (uint16 or uint32 for GLYPH2)
        if reader.remaining() < 3 + 2 {
            return Err(TtError::INVALID_TABLE);
        }
        let paint_offset = reader.u24()?;

        let parent_offset = paint_base.as_ptr() as usize - self.data.as_ptr() as usize;
        let child_offset = parent_offset.wrapping_add(paint_offset as usize) as u32;
        let child_paint = OpaquePaint::new(child_offset, false);

        let glyph_id = if format_byte == 33 {
            // GLYPH2: Offset32 glyph ID
            if reader.remaining() < 4 {
                return Err(TtError::INVALID_TABLE);
            }
            reader.u32()? as u16
        } else {
            reader.u16()?
        };

        Ok(Paint::Glyph(GlyphPaint {
            paint: child_paint,
            glyph_id,
        }))
    }

    fn read_colr_glyph(&self, reader: &mut Reader<'_>) -> TtResult<Paint<'a>> {
        // glyphID (uint16)
        if reader.remaining() < 2 {
            return Err(TtError::INVALID_TABLE);
        }
        let glyph_id = reader.u16()?;
        Ok(Paint::ColrGlyph(ColrGlyphPaint { glyph_id }))
    }

    fn read_transform(&self, reader: &mut Reader<'_>, paint_base: &'a [u8]) -> TtResult<Paint<'a>> {
        // paint (Offset24) + affine (6 x int32)
        if reader.remaining() < 3 + 24 {
            return Err(TtError::INVALID_TABLE);
        }
        let paint_offset = reader.u24()?;

        let parent_offset = paint_base.as_ptr() as usize - self.data.as_ptr() as usize;
        let child_offset = parent_offset.wrapping_add(paint_offset as usize) as u32;
        let child_paint = OpaquePaint::new(child_offset, false);

        let xx = reader.i32()?;
        let yx = reader.i32()?;
        let xy = reader.i32()?;
        let yy = reader.i32()?;
        let dx = reader.i32()?;
        let dy = reader.i32()?;

        Ok(Paint::Transform(TransformPaint {
            paint: child_paint,
            affine: [xx, xy, dx, yx, yy, dy],
        }))
    }

    fn read_translate(&self, reader: &mut Reader<'_>, paint_base: &'a [u8]) -> TtResult<Paint<'a>> {
        // paint (Offset24) + dx + dy (each int16 -> 16.16)
        if reader.remaining() < 3 + 4 {
            return Err(TtError::INVALID_TABLE);
        }
        let paint_offset = reader.u24()?;

        let parent_offset = paint_base.as_ptr() as usize - self.data.as_ptr() as usize;
        let child_offset = parent_offset.wrapping_add(paint_offset as usize) as u32;
        let child_paint = OpaquePaint::new(child_offset, false);

        let dx = (reader.i16()? as i32) << 16;
        let dy = (reader.i16()? as i32) << 16;

        Ok(Paint::Translate(TranslatePaint {
            paint: child_paint,
            dx,
            dy,
        }))
    }

    fn read_scale(
        &self,
        reader: &mut Reader<'_>,
        paint_base: &'a [u8],
        format_byte: u8,
    ) -> TtResult<Paint<'a>> {
        // paint (Offset24) + scaleX (F2Dot14) + [scaleY] + [centerX + centerY]
        if reader.remaining() < 3 + 2 {
            return Err(TtError::INVALID_TABLE);
        }
        let paint_offset = reader.u24()?;

        let parent_offset = paint_base.as_ptr() as usize - self.data.as_ptr() as usize;
        let child_offset = parent_offset.wrapping_add(paint_offset as usize) as u32;
        let child_paint = OpaquePaint::new(child_offset, false);

        let scale_x = (reader.i16()? as i32) << 16;

        // Non-uniform formats: 16 (Scale), 18 (ScaleCenter)
        let non_uniform = matches!(format_byte, 16 | 18);
        let scale_y = if non_uniform {
            if reader.remaining() < 2 {
                return Err(TtError::INVALID_TABLE);
            }
            (reader.i16()? as i32) << 16
        } else {
            scale_x
        };

        // Centered formats: 18 (ScaleCenter), 19 (VarScaleCenter),
        // 22 (ScaleUniformCenter), 23 (VarScaleUniformCenter)
        let has_center = matches!(format_byte, 18 | 19 | 22 | 23);
        let (center_x, center_y) = if has_center {
            if reader.remaining() < 4 {
                return Err(TtError::INVALID_TABLE);
            }
            ((reader.i16()? as i32) << 16, (reader.i16()? as i32) << 16)
        } else {
            (0, 0)
        };

        Ok(Paint::Scale(ScalePaint {
            paint: child_paint,
            scale_x,
            scale_y,
            center_x,
            center_y,
        }))
    }

    fn read_rotate(
        &self,
        reader: &mut Reader<'_>,
        paint_base: &'a [u8],
        format_byte: u8,
    ) -> TtResult<Paint<'a>> {
        // paint (Offset24) + angle (F2Dot14) + [centerX + centerY]
        if reader.remaining() < 3 + 2 {
            return Err(TtError::INVALID_TABLE);
        }
        let paint_offset = reader.u24()?;

        let parent_offset = paint_base.as_ptr() as usize - self.data.as_ptr() as usize;
        let child_offset = parent_offset.wrapping_add(paint_offset as usize) as u32;
        let child_paint = OpaquePaint::new(child_offset, false);

        let angle = (reader.i16()? as i32) << 16;

        // Centered formats: 26 (RotateCenter), 27 (VarRotateCenter)
        let has_center = matches!(format_byte, 26 | 27);
        let (center_x, center_y) = if has_center {
            if reader.remaining() < 4 {
                return Err(TtError::INVALID_TABLE);
            }
            ((reader.i16()? as i32) << 16, (reader.i16()? as i32) << 16)
        } else {
            (0, 0)
        };

        Ok(Paint::Rotate(RotatePaint {
            paint: child_paint,
            angle,
            center_x,
            center_y,
        }))
    }

    fn read_skew(
        &self,
        reader: &mut Reader<'_>,
        paint_base: &'a [u8],
        format_byte: u8,
    ) -> TtResult<Paint<'a>> {
        // paint (Offset24) + xSkewAngle + ySkewAngle (each F2Dot14) + [centerX + centerY]
        if reader.remaining() < 3 + 4 {
            return Err(TtError::INVALID_TABLE);
        }
        let paint_offset = reader.u24()?;

        let parent_offset = paint_base.as_ptr() as usize - self.data.as_ptr() as usize;
        let child_offset = parent_offset.wrapping_add(paint_offset as usize) as u32;
        let child_paint = OpaquePaint::new(child_offset, false);

        let x_skew_angle = (reader.i16()? as i32) << 16;
        let y_skew_angle = (reader.i16()? as i32) << 16;

        // Centered formats: 30 (SkewCenter), 31 (VarSkewCenter)
        let has_center = matches!(format_byte, 30 | 31);
        let (center_x, center_y) = if has_center {
            if reader.remaining() < 4 {
                return Err(TtError::INVALID_TABLE);
            }
            ((reader.i16()? as i32) << 16, (reader.i16()? as i32) << 16)
        } else {
            (0, 0)
        };

        Ok(Paint::Skew(SkewPaint {
            paint: child_paint,
            x_skew_angle,
            y_skew_angle,
            center_x,
            center_y,
        }))
    }

    fn read_composite(&self, reader: &mut Reader<'_>, paint_base: &'a [u8]) -> TtResult<Paint<'a>> {
        // sourcePaint (Offset24) + compositeMode (uint8) + backdropPaint (Offset24)
        if reader.remaining() < 3 + 1 + 3 {
            return Err(TtError::INVALID_TABLE);
        }
        let source_offset = reader.u24()?;
        let mode_byte = reader.u8()?;
        let mode = CompositeMode::from_byte(mode_byte).ok_or(TtError::INVALID_TABLE)?;
        let backdrop_offset = reader.u24()?;

        let parent_offset = paint_base.as_ptr() as usize - self.data.as_ptr() as usize;
        let source_child_offset = parent_offset.wrapping_add(source_offset as usize) as u32;
        let backdrop_child_offset = parent_offset.wrapping_add(backdrop_offset as usize) as u32;

        Ok(Paint::Composite(CompositePaint {
            source: OpaquePaint::new(source_child_offset, false),
            mode,
            backdrop: OpaquePaint::new(backdrop_child_offset, false),
        }))
    }

    /// Reads a color line at the given offset.
    fn read_color_line(
        &self,
        paint_base: &'a [u8],
        offset: usize,
        read_variable: bool,
    ) -> TtResult<ColorLine> {
        if offset >= paint_base.len() {
            return Err(TtError::INVALID_TABLE);
        }
        let data = &paint_base[offset..];
        if data.len() < 3 {
            return Err(TtError::INVALID_TABLE);
        }
        let mut reader = Reader::new(data);
        let extend_byte = reader.u8()?;
        let extend = PaintExtend::from_byte(extend_byte).ok_or(TtError::INVALID_TABLE)?;
        let num_stops = reader.u16()?;

        let entry_size = 6 + if read_variable { 4 } else { 0 };
        if data.len() < 3 + (num_stops as usize) * entry_size {
            return Err(TtError::INVALID_TABLE);
        }

        let mut stops = alloc::vec::Vec::with_capacity(num_stops as usize);
        for _ in 0..num_stops {
            let stop_offset = reader.i16()? as i32;
            let palette_index = reader.u16()?;
            let alpha = reader.i16()?;
            stops.push(ColorStop {
                stop_offset: stop_offset << 16,
                color: ColorIndex { palette_index, alpha },
            });
            if read_variable {
                // Skip varIndexBase (uint32)
                reader.u32()?;
            }
        }

        Ok(ColorLine { extend, stops })
    }

    /// Creates an opaque paint for the root paint of a base glyph
    /// (`tt_face_get_colr_glyph_paint`).
    pub fn root_paint(&self, gindex: u32, root_transform: RootTransform) -> Option<OpaquePaint> {
        let paint_offset = self.base_glyph_v1(gindex)?;
        if paint_offset == 0 {
            return None;
        }
        Some(OpaquePaint::new(
            paint_offset,
            root_transform == RootTransform::Include,
        ))
    }
}

/// The v0 layer iterator returned by [`ColrTable::layers`]
/// (`FT_LayerIterator` + `tt_face_get_colr_layer`).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerIter<'a> {
    /// The validated `LayerRecords` array.
    layers: &'a [u8],
    /// The next layer record index.
    next: u16,
    /// How many records remain.
    remaining: u16,
    /// The face's glyph count (record validation).
    num_glyphs: u16,
    /// The active palette's entry count (record validation).
    num_palette_entries: u16,
}

impl Iterator for LayerIter<'_> {
    /// `(glyph_index, palette_index)`; `palette_index` may be
    /// [`COLOR_FOREGROUND`].
    type Item = (u16, u16);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let index = self.next;
        self.next = self.next.checked_add(1)?;
        let (glyph, color) = layer_record(self.layers, index)?;
        if glyph >= self.num_glyphs || (color != COLOR_FOREGROUND && color >= self.num_palette_entries) {
            // `tt_face_get_colr_layer` aborts the iteration on an
            // out-of-range record.
            return None;
        }
        Some((glyph, color))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for LayerIter<'_> {}

/// Decodes layer record `index` of a raw `LayerRecords` slice.
fn layer_record(layers: &[u8], index: u16) -> Option<(u16, u16)> {
    let record = layers.get((index as usize) * LAYER_SIZE..)?;
    let glyph = read_u16_at(record, 0).ok()?;
    let color = read_u16_at(record, 2).ok()?;
    Some((glyph, color))
}

// ===== COLR v1 Iterators =====

impl<'a> LayerV1Iter<'a> {
    /// Returns the number of remaining layers.
    #[inline]
    pub fn remaining(&self) -> u16 {
        self.num_layers - self.current
    }
}

impl<'a> Iterator for LayerV1Iter<'a> {
    type Item = OpaquePaint;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.num_layers {
            return None;
        }
        let offset = self.list_base + (self.current as usize) * 4;
        let paint_offset = u32::from_be_bytes(
            self.data
                .get(offset..offset + 4)?
                .try_into()
                .unwrap_or([0, 0, 0, 0]),
        );
        self.current += 1;
        Some(OpaquePaint::new(paint_offset, false))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining() as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for LayerV1Iter<'_> {}

impl<'a> ColorStopIter<'a> {
    /// Returns the number of remaining color stops.
    #[inline]
    pub fn remaining(&self) -> u16 {
        self.stops.len() as u16 - self.current
    }
}

impl<'a> Iterator for ColorStopIter<'a> {
    type Item = ColorStop;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.stops.len() as u16 {
            return None;
        }
        let stop = self.stops[self.current as usize];
        self.current += 1;
        Some(stop)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining() as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for ColorStopIter<'_> {}

/// Iterator over color stops for a gradient.
/// This is a simpler alternative to ColorStopIter that works with
/// the already-parsed ColorLine.stops.
pub struct ColorStops<'a> {
    stops: &'a [ColorStop],
    index: usize,
}

impl<'a> ColorStops<'a> {
    /// Creates a new color stop iterator over the given stops.
    #[inline]
    pub fn new(stops: &'a [ColorStop]) -> Self {
        Self { stops, index: 0 }
    }
}

impl<'a> Iterator for ColorStops<'a> {
    type Item = &'a ColorStop;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.stops.len() {
            return None;
        }
        let stop = &self.stops[self.index];
        self.index += 1;
        Some(stop)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.stops.len() - self.index;
        (len, Some(len))
    }
}

impl ExactSizeIterator for ColorStops<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Builds a `COLR` v0 table from base glyph and layer records.
    fn colr_v0(bases: &[(u16, u16, u16)], layers: &[(u16, u16)]) -> Vec<u8> {
        let base_offset = COLRV0_HEADER_SIZE;
        let layer_offset = base_offset + bases.len() * BASE_GLYPH_SIZE;
        let mut out = Vec::new();
        out.extend_from_slice(&0u16.to_be_bytes()); // version
        out.extend_from_slice(&(bases.len() as u16).to_be_bytes());
        out.extend_from_slice(&(base_offset as u32).to_be_bytes());
        out.extend_from_slice(&(layer_offset as u32).to_be_bytes());
        out.extend_from_slice(&(layers.len() as u16).to_be_bytes());
        for &(gid, first, num) in bases {
            out.extend_from_slice(&gid.to_be_bytes());
            out.extend_from_slice(&first.to_be_bytes());
            out.extend_from_slice(&num.to_be_bytes());
        }
        for &(gid, color) in layers {
            out.extend_from_slice(&gid.to_be_bytes());
            out.extend_from_slice(&color.to_be_bytes());
        }
        out
    }

    #[test]
    fn binary_search_finds_base_glyphs_in_glyph_order() {
        let bytes = colr_v0(&[(10, 0, 2), (20, 2, 1)], &[(1, 0), (2, 1), (3, 2)]);
        let colr = ColrTable::parse(&bytes).unwrap();

        assert_eq!(colr.version(), 0);
        assert_eq!(colr.num_base_glyphs(), 2);
        assert_eq!(colr.num_layers(), 3);
        assert_eq!(colr.base_glyph(10), Some((0, 2)));
        assert_eq!(colr.base_glyph(20), Some((2, 1)));
        assert_eq!(colr.base_glyph(11), None);
        assert_eq!(colr.base_glyph(5), None);
        assert_eq!(colr.base_glyph(0xFFFF_FFFF), None);
    }

    #[test]
    fn layer_iteration_walks_records_and_stops_on_bad_records() {
        let bytes = colr_v0(&[(10, 0, 3)], &[(1, 0), (2, 7), (3, 2)]);
        let colr = ColrTable::parse(&bytes).unwrap();

        let layers: Vec<(u16, u16)> = colr.layers(10, 5, 8).unwrap().collect();
        assert_eq!(layers, alloc::vec![(1, 0), (2, 7), (3, 2)]);

        // Palette index past the palette: iteration stops right there.
        let layers: Vec<(u16, u16)> = colr.layers(10, 5, 7).unwrap().collect();
        assert_eq!(layers, alloc::vec![(1, 0)]);

        // Glyph index past the face: same.
        let layers: Vec<(u16, u16)> = colr.layers(10, 3, 8).unwrap().collect();
        assert_eq!(layers, alloc::vec![(1, 0), (2, 7)]);

        // The foreground sentinel is always accepted.
        let bytes = colr_v0(&[(10, 0, 1)], &[(1, COLOR_FOREGROUND)]);
        let colr = ColrTable::parse(&bytes).unwrap();
        assert_eq!(
            colr.layers(10, 5, 1).unwrap().collect::<Vec<_>>(),
            alloc::vec![(1, 0xFFFF)]
        );
    }

    #[test]
    fn layer_ranges_past_the_array_end_are_rejected() {
        let bytes = colr_v0(&[(10, 2, 5)], &[(1, 0), (2, 1)]);
        let colr = ColrTable::parse(&bytes).unwrap();
        assert_eq!(colr.layers(10, 5, 8), None);
    }

    #[test]
    fn zero_layer_records_yield_no_iteration() {
        let bytes = colr_v0(&[(10, 0, 0)], &[]);
        let colr = ColrTable::parse(&bytes).unwrap();
        assert!(colr.base_glyph(10).is_some());
        assert!(colr.layers(10, 5, 8).is_none());
    }

    #[test]
    fn malformed_tables_return_errors_and_never_panic() {
        // Shorter than the v0 header.
        assert!(matches!(ColrTable::parse(&[0; 13]), Err(TtError::INVALID_TABLE)));

        // Unknown version.
        let mut bytes = colr_v0(&[(1, 0, 1)], &[(1, 0)]);
        bytes[0..2].copy_from_slice(&2u16.to_be_bytes());
        assert!(matches!(ColrTable::parse(&bytes), Err(TtError::INVALID_TABLE)));

        // Base glyph offset past the end.
        let mut bytes = colr_v0(&[(1, 0, 1)], &[(1, 0)]);
        bytes[4..8].copy_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
        assert!(matches!(ColrTable::parse(&bytes), Err(TtError::INVALID_TABLE)));

        // Layer count overruns the table.
        let mut bytes = colr_v0(&[(1, 0, 1)], &[(1, 0)]);
        bytes[12..14].copy_from_slice(&0xFFFFu16.to_be_bytes());
        assert!(matches!(ColrTable::parse(&bytes), Err(TtError::INVALID_TABLE)));

        // v1 header cut short.
        let mut bytes = colr_v0(&[(1, 0, 1)], &[(1, 0)]);
        bytes[0..2].copy_from_slice(&1u16.to_be_bytes());
        assert!(matches!(ColrTable::parse(&bytes), Err(TtError::INVALID_TABLE)));
    }

    #[test]
    fn v1_header_is_validated_but_paints_stay_unimplemented() {
        // v1 with a v0 header padded to 34 bytes: every v1 offset is
        // zero, so BaseGlyphV1List points at the table start whose
        // u32 "count" is garbage and overruns the table.
        let mut bytes = colr_v0(&[(1, 0, 1)], &[(1, 0)]);
        bytes[0..2].copy_from_slice(&1u16.to_be_bytes());
        bytes.resize(COLRV1_HEADER_SIZE, 0);
        assert!(matches!(ColrTable::parse(&bytes), Err(TtError::INVALID_TABLE)));

        // A well-formed v1 header with an empty BaseGlyphV1List keeps
        // the v0 layers usable.
        //
        // Layout: v0 header (14) + v1 offsets (10) + base record (6)
        // + layer record (4) + BaseGlyphV1List count (4) + 2 pad bytes.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u16.to_be_bytes()); // version
        bytes.extend_from_slice(&1u16.to_be_bytes()); // numBaseGlyphRecords
        bytes.extend_from_slice(&34u32.to_be_bytes()); // base records at 34
        bytes.extend_from_slice(&40u32.to_be_bytes()); // layer records at 40
        bytes.extend_from_slice(&1u16.to_be_bytes()); // numLayerRecords
        bytes.extend_from_slice(&44u32.to_be_bytes()); // BaseGlyphV1List
        bytes.extend_from_slice(&0u32.to_be_bytes()); // layerList absent
        bytes.extend_from_slice(&0u32.to_be_bytes()); // clipList absent
        bytes.extend_from_slice(&0u32.to_be_bytes()); // varIdxMap absent
        bytes.extend_from_slice(&0u32.to_be_bytes()); // varStore absent
        bytes.extend_from_slice(&10u16.to_be_bytes()); // base glyph 10
        bytes.extend_from_slice(&0u16.to_be_bytes()); // first layer
        bytes.extend_from_slice(&1u16.to_be_bytes()); // layer count
        bytes.extend_from_slice(&1u16.to_be_bytes()); // layer glyph 1
        bytes.extend_from_slice(&0u16.to_be_bytes()); // palette entry 0
        bytes.extend_from_slice(&0u32.to_be_bytes()); // v1 list count = 0
        bytes.extend_from_slice(&[0, 0]); // padding

        let colr = ColrTable::parse(&bytes).unwrap();
        assert_eq!(colr.version(), 1);
        assert!(!colr.has_v1_paints());
        assert_eq!(colr.base_glyph(10), Some((0, 1)));
        assert_eq!(
            colr.layers(10, 5, 8).unwrap().collect::<Vec<_>>(),
            alloc::vec![(1, 0)]
        );
    }

    // ===== COLR v1 Paint Tests =====

    /// Builds a minimal COLR v1 table with a BaseGlyphV1List containing
    /// one glyph with a solid paint.
    ///
    /// Layout:
    ///   v0 header: bytes 0-13 (14 bytes)
    ///   v1 offsets: bytes 14-23 (10 bytes)
    ///   BaseGlyphV1List: count (4 bytes) + record (6 bytes) at bytes 24-33
    ///   Paint data region starts at byte 34 (paints_start_v1)
    fn colr_v1_solid(gid: u16, paint_offset: u32, palette_index: u16, alpha: i16) -> Vec<u8> {
        let mut bytes = Vec::new();

        // v0 header (14 bytes)
        bytes.extend_from_slice(&1u16.to_be_bytes()); // version
        bytes.extend_from_slice(&1u16.to_be_bytes()); // numBaseGlyphRecords
        bytes.extend_from_slice(&0u32.to_be_bytes()); // baseGlyphsOffset (unused for v1)
        bytes.extend_from_slice(&0u32.to_be_bytes()); // layerListOffset
        bytes.extend_from_slice(&0u16.to_be_bytes()); // numLayerRecords
        // v1 offsets (20 bytes, positions 14-33)
        bytes.extend_from_slice(&34u32.to_be_bytes()); // baseGlyphsOffset_v1 = 34
        bytes.extend_from_slice(&0u32.to_be_bytes()); // layerListOffset absent
        bytes.extend_from_slice(&0u32.to_be_bytes()); // clipListOffset absent
        bytes.extend_from_slice(&0u32.to_be_bytes()); // varIdxMapOffset absent
        bytes.extend_from_slice(&0u32.to_be_bytes()); // varStoreOffset absent

        // BaseGlyphV1List at offset 34: count (4) + record (6) = bytes 34-43
        bytes.extend_from_slice(&1u32.to_be_bytes()); // count = 1
        bytes.extend_from_slice(&gid.to_be_bytes()); // glyph ID (2 bytes)
        bytes.extend_from_slice(&paint_offset.to_be_bytes()); // paint offset (4 bytes, ULONG)

        // paints_start_v1 = 34 + 4 + 6 = 44
        let paints_start = 44usize;
        bytes.resize(paints_start, 0);

        // Solid paint at paints_start: format (1) + paletteIndex (2) + alpha (2)
        bytes.push(2u8); // PaintSolid format
        bytes.extend_from_slice(&palette_index.to_be_bytes());
        bytes.extend_from_slice(&alpha.to_be_bytes());

        bytes
    }

    #[test]
    fn v1_base_glyph_paint_record_lookup() {
        // paint_offset = 10 means absolute offset = 34 + 10 = 44 (paints_start_v1)
        let bytes = colr_v1_solid(10, 10, 5, 0x7FFF);
        let colr = ColrTable::parse(&bytes).unwrap();

        assert_eq!(colr.version(), 1);
        assert_eq!(colr.num_base_glyphs_v1(), 1);
        // Paint offset stored in record is 10, absolute offset = 34 + 10 = 44
        assert_eq!(colr.base_glyph_v1(10), Some(44));
        assert_eq!(colr.base_glyph_v1(11), None);
    }

    #[test]
    fn v1_solid_paint_parsing() {
        // paint_offset = 10 means absolute offset = 34 + 10 = 44 (paints_start_v1)
        let bytes = colr_v1_solid(10, 10, 3, 0x4000);
        let colr = ColrTable::parse(&bytes).unwrap();

        let opaque = colr
            .root_paint(10, RootTransform::Exclude)
            .unwrap();
        let paint = colr
            .paint(opaque, RootTransform::Exclude)
            .unwrap();

        match paint {
            Paint::Solid(solid) => {
                assert_eq!(solid.color.palette_index, 3);
                assert_eq!(solid.color.alpha, 0x4000);
            }
            _ => panic!("expected Paint::Solid"),
        }
    }

    #[test]
    fn v1_malformed_paint_offset_is_rejected() {
        // paint_offset = 0 means absolute offset = 34, which is < paints_start_v1 (44)
        let bytes = colr_v1_solid(10, 0, 0, 0);
        let colr = ColrTable::parse(&bytes).unwrap();
        let opaque = colr
            .root_paint(10, RootTransform::Exclude)
            .unwrap();
        // The paint offset points before paints_start_v1, so it should fail
        assert!(matches!(
            colr.paint(opaque, RootTransform::Exclude),
            Err(TtError::INVALID_TABLE)
        ));
    }

    #[test]
    fn v1_format_byte_parsing_accepts_internal_formats() {
        // Internal variable formats should be accepted and mapped to public formats
        assert_eq!(PaintFormat::from_byte(17), Some(PaintFormat::Scale)); // VarScale
        assert_eq!(PaintFormat::from_byte(18), Some(PaintFormat::Scale)); // ScaleCenter
        assert_eq!(PaintFormat::from_byte(25), Some(PaintFormat::Rotate)); // VarRotate
        assert_eq!(PaintFormat::from_byte(26), Some(PaintFormat::Rotate)); // RotateCenter
        assert_eq!(PaintFormat::from_byte(29), Some(PaintFormat::Skew)); // VarSkew
        assert_eq!(PaintFormat::from_byte(30), Some(PaintFormat::Skew)); // SkewCenter
        assert_eq!(PaintFormat::from_byte(33), Some(PaintFormat::Glyph)); // Glyph2

        // Invalid formats should be rejected
        assert_eq!(PaintFormat::from_byte(0), None);
        assert_eq!(PaintFormat::from_byte(3), None);
        assert_eq!(PaintFormat::from_byte(255), None);
    }
}
