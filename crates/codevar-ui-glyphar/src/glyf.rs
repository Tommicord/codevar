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

//! Glyph outlines (`glyf`).
//!
//! [`Glyph::load`] resolves a glyph id through `loca`, decodes the
//! glyph description and hands back a [`Glyph`]: a list of
//! [`Source`] outlines, each carrying its own points, its own TrueType
//! instruction stream and the [`Affine`] transform that maps it into
//! glyph space.
//!
//! # Why sources instead of one flat point list
//!
//! Hinting a composite glyph means running each *component's* program
//! in the component's own coordinate system **before** the component
//! transform is applied: a scaled or rotated component that was
//! transformed first would round against the wrong grid. Keeping the
//! components separate lets [`crate::interpreter`] hint each source in
//! isolation and lets [`Glyph::flatten`] merge them afterwards, in
//! either order.
//!
//! # Supported glyph data
//!
//! | Kind | Detection | Notes |
//! |------|-----------|-------|
//! | Simple | `numberOfContours >= 0` | repeat-encoded flags, short/long deltas, empty glyphs |
//! | Composite | `numberOfContours < 0` | word/byte args, xy offsets, 1×/2×/2×2 scales, `MORE_COMPONENTS`, `WE_HAVE_INSTRUCTIONS` |
//! | Empty | zero-length `loca` slice or zero contours | [`Glyph::load`] returns `Ok(None)` |
//!
//! `USE_MY_METRICS` and `ROUND_XY_TO_GRID` are accepted and ignored:
//! metrics come from `hmtx`, and the component offsets this engine
//! decodes are already integers on the font grid.
//!
//! Composite *point matching* (`ARGS_ARE_XY_VALUES` clear) and nested
//! composite instruction streams are rejected with
//! [`GlypharError::Unsupported`]; both are extremely rare and would
//! require the interpreter to run programs in the middle of the
//! assembly walk.
//!
//! # Phantom points
//!
//! Every [`Source`] carries four trailing phantom points (see
//! [`Source::phantom_points`]) so the interpreter can adjust advance
//! widths with `ADJUST*` instructions. They are never part of the
//! outline: [`Glyph::flatten`] drops them and the rasterizer never
//! sees them.

use alloc::vec::Vec;

use crate::font_file::{FontFile, Reader};
use crate::{GlypharError, GlypharResult};

/// Deepest composite nesting this engine follows.
///
/// Fonts with deeper nesting are almost certainly cyclic; rejecting
/// them keeps a hostile font from exhausting the stack.
const MAX_COMPOSITE_DEPTH: u8 = 8;

/// `ARG_1_AND_2_ARE_WORDS`: the two component arguments are `int16`.
const ARG_1_AND_2_ARE_WORDS: u16 = 0x0001;
/// `ARGS_ARE_XY_VALUES`: the arguments are an x/y offset rather than a
/// point pair.
const ARGS_ARE_XY_VALUES: u16 = 0x0002;
/// `WE_HAVE_A_SCALE`: one `F2Dot14` scale for both axes.
const WE_HAVE_A_SCALE: u16 = 0x0008;
/// `MORE_COMPONENTS`: another component record follows.
const MORE_COMPONENTS: u16 = 0x0020;
/// `WE_HAVE_AN_X_AND_Y_SCALE`: separate `F2Dot14` scales.
const WE_HAVE_AN_X_AND_Y_SCALE: u16 = 0x0040;
/// `WE_HAVE_A_TWO_BY_TWO`: a full 2×2 `F2Dot14` matrix.
const WE_HAVE_A_TWO_BY_TWO: u16 = 0x0080;
/// `WE_HAVE_INSTRUCTIONS`: an instruction stream follows the last
/// component.
const WE_HAVE_INSTRUCTIONS: u16 = 0x0100;
/// `SCALED_COMPONENT_OFFSET`: scale the x/y offset with the component.
const SCALED_COMPONENT_OFFSET: u16 = 0x0800;
/// `UNSCALED_COMPONENT_OFFSET`: keep the x/y offset unscaled.
const UNSCALED_COMPONENT_OFFSET: u16 = 0x1000;

/// Simple-point flag: the point lies on the curve.
const ON_CURVE: u8 = 0x01;
/// Simple-point flag: the x delta is a signed byte.
const X_SHORT: u8 = 0x02;
/// Simple-point flag: the y delta is a signed byte.
const Y_SHORT: u8 = 0x04;
/// Simple-point flag: the following flag byte repeats.
const REPEAT: u8 = 0x08;
/// Simple-point flag: x is the same as the previous point (or the
/// short x delta is positive).
const X_SAME: u8 = 0x10;
/// Simple-point flag: y is the same as the previous point (or the
/// short y delta is positive).
const Y_SAME: u8 = 0x20;

/// Axis-aligned bounding box, in font units.
///
/// The values are copied from the glyph header (or `head`), so a font
/// that records a stale box keeps it: several hinting algorithms read
/// the box directly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BBox {
    /// Smallest x of the outline.
    pub x_min: i16,
    /// Smallest y of the outline.
    pub y_min: i16,
    /// Largest x of the outline.
    pub x_max: i16,
    /// Largest y of the outline.
    pub y_max: i16,
}

impl BBox {
    /// Smallest box that contains both inputs.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            x_min: self.x_min.min(other.x_min),
            y_min: self.y_min.min(other.y_min),
            x_max: self.x_max.max(other.x_max),
            y_max: self.y_max.max(other.y_max),
        }
    }

    /// Whether the box is degenerate (zero width or height).
    #[must_use]
    pub const fn is_degenerate(self) -> bool {
        self.x_min >= self.x_max || self.y_min >= self.y_max
    }
}

/// One contour control point in font units.
///
/// `on_curve == false` marks an off-curve (control) point; the
/// rasterizer turns the sequence into quadratic segments, inserting
/// implied on-curve midpoints between two consecutive off-curve
/// points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    /// x coordinate in font units.
    pub x: i16,
    /// y coordinate in font units.
    pub y: i16,
    /// Whether the point lies on the curve.
    pub on_curve: bool,
}

impl Point {
    /// Creates a point.
    #[inline]
    #[must_use]
    pub const fn new(x: i16, y: i16, on_curve: bool) -> Self {
        Self { x, y, on_curve }
    }
}

/// Affine transform from component space into glyph space.
///
/// `x' = a·x + b·y + e`, `y' = c·x + d·y + f`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine {
    /// x ← x scale.
    pub a: f32,
    /// x ← y scale (shear/rotation term).
    pub b: f32,
    /// y ← x scale (shear/rotation term).
    pub c: f32,
    /// y ← y scale.
    pub d: f32,
    /// x translation in font units.
    pub e: f32,
    /// y translation in font units.
    pub f: f32,
}

impl Affine {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: 0.0,
    };

    /// Pure translation.
    #[must_use]
    pub const fn translation(dx: f32, dy: f32) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: dx,
            f: dy,
        }
    }

    /// Axis-aligned scale without translation.
    #[must_use]
    pub const fn scale(sx: f32, sy: f32) -> Self {
        Self {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Composes `self` *after* `inner`, i.e. `(self ∘ inner)(p)`.
    #[must_use]
    pub fn compose(self, inner: Self) -> Self {
        Self {
            a: self.a * inner.a + self.b * inner.c,
            b: self.a * inner.b + self.b * inner.d,
            c: self.c * inner.a + self.d * inner.c,
            d: self.c * inner.b + self.d * inner.d,
            e: self.a * inner.e + self.b * inner.f + self.e,
            f: self.c * inner.e + self.d * inner.f + self.f,
        }
    }

    /// Applies the transform to `p`, rounding to the nearest font
    /// unit and saturating outside the `i16` range.
    #[inline]
    #[must_use]
    pub fn apply(self, p: Point) -> Point {
        let x = f32::from(p.x) * self.a + f32::from(p.y) * self.b + self.e;
        let y = f32::from(p.x) * self.c + f32::from(p.y) * self.d + self.f;
        Point::new(round_to_i16(x), round_to_i16(y), p.on_curve)
    }

    /// Whether the transform maps every point onto itself.
    #[must_use]
    pub fn is_identity(self) -> bool {
        self == Self::IDENTITY
    }
}

/// Rounds a float to the nearest `i16`, saturating at the bounds.
///
/// `NaN` maps to 0, which is what a corrupt `F2Dot14` in a broken font
/// deserves rather than a panic.
#[inline]
fn round_to_i16(v: f32) -> i16 {
    let rounded = v.round();
    if rounded >= 32_767.0 {
        i16::MAX
    } else if rounded <= -32_768.0 {
        i16::MIN
    } else {
        rounded as i16
    }
}

/// Converts an `F2Dot14` field to `f32`.
#[inline]
fn f2dot14(v: i16) -> f32 {
    f32::from(v) / 16_384.0
}

/// One source outline of a glyph.
///
/// A simple glyph has exactly one source; a composite glyph has one
/// per component, each with the transform that maps it into glyph
/// space. Points are stored in the *component's own* coordinate
/// system so [`crate::interpreter`] can run that component's program
/// before the transform is applied.
#[derive(Debug, Clone)]
pub struct Source<'a> {
    /// The glyph this source was loaded from.
    glyph_id: u16,
    /// Contour points, no phantom points.
    points: Vec<Point>,
    /// Exclusive end index of each contour inside `points`.
    contours: Vec<u32>,
    /// Instruction stream to run over `points` plus phantoms.
    instructions: &'a [u8],
    /// Component-space to glyph-space transform.
    transform: Affine,
    /// The four phantom points (advance width and vertical metrics).
    phantoms: [Point; 4],
    /// Bounding box recorded in the component's own glyph header.
    bbox: BBox,
}

impl<'a> Source<'a> {
    /// Glyph id this source came from.
    #[must_use]
    pub const fn glyph_id(&self) -> u16 {
        self.glyph_id
    }

    /// Contour points in component space.
    #[must_use]
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// Mutable view of the contour points, used by the interpreter to
    /// apply deltas.
    pub fn points_mut(&mut self) -> &mut [Point] {
        &mut self.points
    }

    /// Exclusive end index of each contour (a `start..end` slice into
    /// [`Source::points`]).
    #[must_use]
    pub fn contours(&self) -> &[u32] {
        &self.contours
    }

    /// TrueType instruction stream for this source.
    #[must_use]
    pub const fn instructions(&self) -> &'a [u8] {
        self.instructions
    }

    /// Transform mapping this source into glyph space.
    #[must_use]
    pub const fn transform(&self) -> Affine {
        self.transform
    }

    /// The four phantom points appended after the contour points
    /// during hinting.
    ///
    /// | Index | Point |
    /// |-------|-------|
    /// | 0 | `(x_origin, 0)` — the left side bearing point |
    /// | 1 | `(x_origin + advance_width, 0)` — the advance point |
    /// | 2 | `(x_origin, y_max)` — vertical origin |
    /// | 3 | `(x_origin, y_min)` — vertical advance |
    ///
    /// `x_origin = bbox.x_min − lsb`. The engine does not parse
    /// `vhea`/`vmtx`, so the vertical pair spans the outline's own
    /// height instead of the font's vertical metrics.
    #[must_use]
    pub const fn phantom_points(&self) -> [Point; 4] {
        self.phantoms
    }

    /// Replaces the phantom points after an `ADJUST*` instruction.
    pub fn set_phantom_points(&mut self, phantoms: [Point; 4]) {
        self.phantoms = phantoms;
    }

    /// Bounding box recorded in the component's glyph header.
    #[must_use]
    pub const fn bbox(&self) -> BBox {
        self.bbox
    }
}

/// A flattened outline: every source merged into glyph space.
///
/// Phantom points are never included; the rasterizer consumes
/// [`FlatGlyph::points`] together with [`FlatGlyph::contours`].
#[derive(Debug, Clone, Default)]
pub struct FlatGlyph {
    /// Contour points in glyph space.
    pub points: Vec<Point>,
    /// Exclusive end index of each contour inside `points`.
    pub contours: Vec<u32>,
}

impl FlatGlyph {
    /// Whether the outline has no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Total number of contours.
    #[must_use]
    pub fn contour_count(&self) -> usize {
        self.contours.len()
    }
}

/// A decoded glyph.
///
/// Instances borrow the `glyf` payload of the [`FontFile`] they were
/// loaded from, so nothing is copied except the decoded points.
#[derive(Debug, Clone)]
pub struct Glyph<'a> {
    /// Box from the glyph header.
    bbox: BBox,
    /// Component outlines in load order.
    sources: Vec<Source<'a>>,
    /// Composite-level instruction stream (empty for simple glyphs).
    instructions: &'a [u8],
}

impl<'a> Glyph<'a> {
    /// Loads and decodes `glyph`.
    ///
    /// # Errors
    ///
    /// * [`GlypharError::OutOfRange`] — `glyph` is past `maxp.numGlyphs`.
    /// * [`GlypharError::Truncated`] — the glyph data ends mid-structure.
    /// * [`GlypharError::Malformed`] — the outline violates the format
    ///   (non-monotonic contour ends, too many points, …).
    /// * [`GlypharError::Unsupported`] — a composite feature this engine
    ///   does not implement.
    pub fn load(font: &FontFile<'a>, glyph: u16) -> GlypharResult<Option<Self>> {
        let Some(range) = font.glyph_range(glyph)? else {
            return Ok(None);
        };
        let glyf = font.glyf_data();
        let raw = glyf.get(range).ok_or(GlypharError::Truncated {
            context: "glyph payload",
        })?;
        Self::parse(font, glyph, raw)
    }

    /// Decodes one raw `glyf` payload.
    ///
    /// `glyph` supplies the metrics used for the phantom points and
    /// resolves composite components; it does not have to agree with
    /// `raw`. This entry point exists so a caller that stores glyph
    /// bytes elsewhere can still reuse the decoder.
    ///
    /// # Errors
    ///
    /// See [`Glyph::load`].
    pub fn parse(font: &FontFile<'a>, glyph: u16, raw: &'a [u8]) -> GlypharResult<Option<Self>> {
        Self::parse_at_depth(font, glyph, raw, 0)
    }

    /// Shared decoder; `depth` counts how many composite levels have
    /// already been followed.
    fn parse_at_depth(
        font: &FontFile<'a>,
        glyph: u16,
        raw: &'a [u8],
        depth: u8,
    ) -> GlypharResult<Option<Self>> {
        if raw.is_empty() {
            return Ok(None);
        }
        let mut r = Reader::new(raw);
        let contour_count = r.read_i16("glyph numberOfContours")?;
        let bbox = BBox {
            x_min: r.read_i16("glyph xMin")?,
            y_min: r.read_i16("glyph yMin")?,
            x_max: r.read_i16("glyph xMax")?,
            y_max: r.read_i16("glyph yMax")?,
        };
        if contour_count >= 0 {
            Self::from_simple(font, glyph, bbox, &mut r, contour_count as u16)
        } else {
            Self::from_composite(font, bbox, &mut r, depth)
        }
    }

    /// Bounding box recorded in the glyph header.
    #[must_use]
    pub const fn bbox(&self) -> BBox {
        self.bbox
    }

    /// Component outlines in load order.
    #[must_use]
    pub fn sources(&self) -> &[Source<'a>] {
        &self.sources
    }

    /// Mutable component outlines, used by the interpreter.
    pub fn sources_mut(&mut self) -> &mut [Source<'a>] {
        &mut self.sources
    }

    /// Composite-level instruction stream, run after every source has
    /// been hinted and merged. Empty for simple glyphs.
    #[must_use]
    pub const fn instructions(&self) -> &'a [u8] {
        self.instructions
    }

    /// Whether the glyph has no outline at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sources
            .iter()
            .all(|source| source.points.is_empty())
    }

    /// Total contour points across all sources (phantoms excluded).
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.sources
            .iter()
            .map(|source| source.points.len())
            .sum()
    }

    /// Merges every source into glyph space, applying its transform.
    ///
    /// This is the *unhinted* path. The hinted path hints each source
    /// first and then performs the same merge.
    #[must_use]
    pub fn flatten(&self) -> FlatGlyph {
        let mut out = FlatGlyph::default();
        self.append_to(&mut out);
        out
    }

    /// Appends this glyph's transformed points to `out`, used both by
    /// [`Glyph::flatten`] and by the interpreter after hinting.
    pub fn append_to(&self, out: &mut FlatGlyph) {
        for source in &self.sources {
            let base = out.points.len() as u32;
            out.points.extend(
                source
                    .points
                    .iter()
                    .map(|&p| source.transform.apply(p)),
            );
            out.contours.extend(
                source
                    .contours
                    .iter()
                    .map(|&end| base.saturating_add(end)),
            );
        }
    }

    /// Builds the single-source outline of a simple glyph.
    fn from_simple(
        font: &FontFile<'a>,
        glyph: u16,
        bbox: BBox,
        r: &mut Reader<'a>,
        contour_count: u16,
    ) -> GlypharResult<Option<Self>> {
        let (points, contours, instructions) = parse_simple(r, contour_count)?;
        if points.is_empty() {
            return Ok(None);
        }
        let phantoms = phantom_points(bbox, font.advance_width(glyph), font.left_side_bearing(glyph));
        Ok(Some(Self {
            bbox,
            sources: alloc::vec![Source {
                glyph_id: glyph,
                points,
                contours,
                instructions,
                transform: Affine::IDENTITY,
                phantoms,
                bbox,
            }],
            instructions: &[],
        }))
    }

    /// Decodes a composite glyph into one source per component.
    fn from_composite(
        font: &FontFile<'a>,
        bbox: BBox,
        r: &mut Reader<'a>,
        depth: u8,
    ) -> GlypharResult<Option<Self>> {
        if depth >= MAX_COMPOSITE_DEPTH {
            return Err(GlypharError::Malformed {
                context: "glyf composite",
                detail: "component nesting exceeds the supported depth",
            });
        }
        let (sources, instructions) = parse_composite(font, r, depth)?;
        if sources.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            bbox,
            sources,
            instructions,
        }))
    }
}

/// Decodes a simple glyph's point description.
///
/// Returns `(points, contour ends, instructions)`. The contour ends
/// are *exclusive* indices into `points`.
fn parse_simple<'a>(
    r: &mut Reader<'a>,
    contour_count: u16,
) -> GlypharResult<(Vec<Point>, Vec<u32>, &'a [u8])> {
    let mut ends = Vec::with_capacity(usize::from(contour_count));
    let mut previous = -1i32;
    for _ in 0..contour_count {
        let end = r.read_u16("glyph endPtsOfContours")?;
        if i32::from(end) < previous {
            return Err(GlypharError::Malformed {
                context: "glyf contours",
                detail: "endPtsOfContours is not monotonically increasing",
            });
        }
        previous = i32::from(end);
        ends.push(end);
    }
    let point_count = match ends.last() {
        Some(&last) => usize::from(last) + 1,
        None => 0,
    };

    let instruction_length = usize::from(r.read_u16("glyph instructionLength")?);
    let instructions = r.read_bytes(instruction_length, "glyph instructions")?;

    let mut flags = Vec::with_capacity(point_count);
    while flags.len() < point_count {
        let flag = r.read_u8("glyph point flags")?;
        flags.push(flag);
        if flag & REPEAT != 0 {
            let repeat = usize::from(r.read_u8("glyph flag repeat count")?);
            for _ in 0..repeat {
                if flags.len() >= point_count {
                    break;
                }
                flags.push(flag);
            }
        }
    }

    // The spec stores the two coordinate arrays separately: all x
    // deltas first, then all y deltas.
    let mut xs = Vec::with_capacity(point_count);
    let mut x = 0i16;
    for &flag in &flags {
        x = x.wrapping_add(read_delta(r, flag, X_SHORT, X_SAME, "glyph xCoordinate")?);
        xs.push(x);
    }
    let mut points = Vec::with_capacity(point_count);
    let mut y = 0i16;
    for (&flag, &px) in flags.iter().zip(&xs) {
        y = y.wrapping_add(read_delta(r, flag, Y_SHORT, Y_SAME, "glyph yCoordinate")?);
        points.push(Point::new(px, y, flag & ON_CURVE != 0));
    }

    let contours = ends
        .iter()
        .map(|&end| u32::from(end) + 1)
        .collect();
    Ok((points, contours, instructions))
}

/// Reads one coordinate delta and applies the short/same encodings.
fn read_delta(
    r: &mut Reader<'_>,
    flag: u8,
    short_bit: u8,
    same_bit: u8,
    context: &'static str,
) -> GlypharResult<i16> {
    if flag & short_bit != 0 {
        let value = i16::from(r.read_u8(context)?);
        if flag & same_bit != 0 {
            Ok(value)
        } else {
            Ok(-value)
        }
    } else if flag & same_bit != 0 {
        Ok(0)
    } else {
        r.read_i16(context)
    }
}

/// Decodes every component of a composite glyph.
///
/// Returns the sources (points still in component space, transform
/// already composed with any nested composite transform) and the
/// composite-level instruction stream.
fn parse_composite<'a>(
    font: &FontFile<'a>,
    r: &mut Reader<'a>,
    depth: u8,
) -> GlypharResult<(Vec<Source<'a>>, &'a [u8])> {
    let mut sources = Vec::new();
    let mut has_instructions = false;
    loop {
        let flags = r.read_u16("composite component flags")?;
        let child = r.read_u16("composite glyphIndex")?;

        let (mut arg1, mut arg2) = if flags & ARG_1_AND_2_ARE_WORDS != 0 {
            (
                i32::from(r.read_i16("composite arg1")?),
                i32::from(r.read_i16("composite arg2")?),
            )
        } else {
            (
                i32::from(r.read_i8("composite arg1")?),
                i32::from(r.read_i8("composite arg2")?),
            )
        };
        if flags & ARGS_ARE_XY_VALUES == 0 {
            return Err(GlypharError::Unsupported {
                feature: "composite point matching",
            });
        }

        // Component matrix: one F2Dot14, two, or a full 2×2.
        let mut matrix = Affine::IDENTITY;
        if flags & WE_HAVE_A_SCALE != 0 {
            let scale = f2dot14(r.read_f2dot14("composite scale")?);
            matrix = Affine::scale(scale, scale);
        } else if flags & WE_HAVE_AN_X_AND_Y_SCALE != 0 {
            let sx = f2dot14(r.read_f2dot14("composite xScale")?);
            let sy = f2dot14(r.read_f2dot14("composite yScale")?);
            matrix = Affine::scale(sx, sy);
        } else if flags & WE_HAVE_A_TWO_BY_TWO != 0 {
            matrix = Affine {
                a: f2dot14(r.read_f2dot14("composite xscale")?),
                b: f2dot14(r.read_f2dot14("composite scale01")?),
                c: f2dot14(r.read_f2dot14("composite scale10")?),
                d: f2dot14(r.read_f2dot14("composite yscale")?),
                e: 0.0,
                f: 0.0,
            };
        }

        // Offset handling: only `SCALED_COMPONENT_OFFSET` scales the
        // arguments, which matches the Microsoft reading of the spec
        // (neither flag ⇒ unscaled). `ROUND_XY_TO_GRID` needs no work
        // here: the arguments are integers, and the scaled case is
        // rounded above.
        if flags & SCALED_COMPONENT_OFFSET != 0 && flags & UNSCALED_COMPONENT_OFFSET == 0 {
            let (sx, sy) = (arg1 as f32, arg2 as f32);
            arg1 = round_to_i16(sx * matrix.a + sy * matrix.b) as i32;
            arg2 = round_to_i16(sx * matrix.c + sy * matrix.d) as i32;
        }
        let transform = Affine {
            e: arg1 as f32,
            f: arg2 as f32,
            ..matrix
        };

        // Resolve the component's own outline (metrics come from the
        // child glyph id, exactly as `hmtx` describes it).
        if let Some(mut component) = load_component(font, child, depth)? {
            for mut source in component.sources.drain(..) {
                source.transform = transform.compose(source.transform);
                sources.push(source);
            }
        }

        if flags & WE_HAVE_INSTRUCTIONS != 0 {
            has_instructions = true;
        }
        if flags & MORE_COMPONENTS == 0 {
            break;
        }
    }

    let instructions = if has_instructions {
        let length = usize::from(r.read_u16("composite instructionLength")?);
        r.read_bytes(length, "composite instructions")?
    } else {
        &[]
    };
    Ok((sources, instructions))
}

/// Loads one composite component, reporting `None` for empty glyphs.
///
/// The component is decoded one nesting level deeper than its parent;
/// a composite component that carries its own program would have to
/// run in the middle of this assembly walk, which this engine does
/// not implement.
fn load_component<'a>(font: &FontFile<'a>, child: u16, depth: u8) -> GlypharResult<Option<Glyph<'a>>> {
    let range = match font.glyph_range(child)? {
        Some(range) => range,
        None => return Ok(None),
    };
    let raw = font
        .glyf_data()
        .get(range)
        .ok_or(GlypharError::Truncated {
            context: "component payload",
        })?;
    let glyph = Glyph::parse_at_depth(font, child, raw, depth + 1)?;
    match glyph {
        // Only a composite carries instructions on the glyph itself;
        // simple glyphs keep theirs on the source. A component's own
        // program would have to run in the middle of this assembly
        // walk, which this engine does not implement.
        Some(glyph) if !glyph.instructions().is_empty() => Err(GlypharError::Unsupported {
            feature: "nested composite instructions",
        }),
        other => Ok(other),
    }
}

/// Computes the four phantom points of a glyph.
///
/// See [`Source::phantom_points`] for the layout.
fn phantom_points(bbox: BBox, advance_width: u16, left_side_bearing: i16) -> [Point; 4] {
    let origin_x = i32::from(bbox.x_min) - i32::from(left_side_bearing);
    let advance_x = origin_x + i32::from(advance_width);
    let clamp = |v: i32| -> i16 {
        if v > i32::from(i16::MAX) {
            i16::MAX
        } else if v < i32::from(i16::MIN) {
            i16::MIN
        } else {
            v as i16
        }
    };
    let ox = clamp(origin_x);
    [
        Point::new(ox, 0, true),
        Point::new(clamp(advance_x), 0, true),
        Point::new(ox, bbox.y_max, true),
        Point::new(ox, bbox.y_min, true),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::SyntheticFont;

    fn font_bytes() -> Vec<u8> {
        SyntheticFont::new().build()
    }

    #[test]
    fn simple_glyph_decodes_points_and_program() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        let glyph = Glyph::load(&font, 1)
            .expect("load")
            .expect("glyph 1 has an outline");
        assert_eq!(
            glyph.bbox(),
            BBox {
                x_min: 50,
                y_min: 0,
                x_max: 550,
                y_max: 700
            }
        );
        assert_eq!(glyph.sources().len(), 1);
        let source = &glyph.sources()[0];
        let expected = [
            Point::new(50, 0, true),
            Point::new(550, 0, true),
            Point::new(550, 700, true),
            Point::new(50, 700, true),
        ];
        assert_eq!(source.points(), &expected);
        assert_eq!(source.contours(), &[4]);
        assert_eq!(source.instructions(), &[0x01, 0xB0, 0x00, 0x2F]);
        assert!(source.transform().is_identity());
        assert!(glyph.instructions().is_empty());
        assert_eq!(glyph.point_count(), 4);
        assert!(!glyph.is_empty());
    }

    #[test]
    fn off_curve_points_keep_their_flag() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        let glyph = Glyph::load(&font, 0)
            .expect("load")
            .expect("glyph 0 has an outline");
        let points = glyph.sources()[0].points();
        assert_eq!(
            points,
            &[
                Point::new(100, 0, true),
                Point::new(300, 600, false),
                Point::new(500, 0, true),
                Point::new(300, -200, false),
            ]
        );
        assert!(glyph.sources()[0].instructions().is_empty());
    }

    #[test]
    fn empty_glyph_has_no_outline() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        assert!(Glyph::load(&font, 2).expect("load").is_none());
        assert!(matches!(
            Glyph::load(&font, 99),
            Err(GlypharError::OutOfRange { .. })
        ));
    }

    #[test]
    fn phantom_points_follow_horizontal_metrics() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        let glyph = Glyph::load(&font, 1)
            .expect("load")
            .expect("outline");
        // bbox.xMin = 50, lsb = 50 → origin 0; advance 600.
        let phantoms = glyph.sources()[0].phantom_points();
        assert_eq!(phantoms[0], Point::new(0, 0, true));
        assert_eq!(phantoms[1], Point::new(600, 0, true));
        assert_eq!(phantoms[2], Point::new(0, 700, true));
        assert_eq!(phantoms[3], Point::new(0, 0, true));
    }

    #[test]
    fn composite_keeps_components_in_their_own_space() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        let glyph = Glyph::load(&font, 3)
            .expect("load")
            .expect("composite outline");
        assert_eq!(glyph.sources().len(), 2);

        let square = &glyph.sources()[0];
        assert_eq!(square.glyph_id(), 1);
        assert!(square.transform().is_identity());
        assert_eq!(square.points()[0], Point::new(50, 0, true));

        let blob = &glyph.sources()[1];
        assert_eq!(blob.glyph_id(), 0);
        assert_eq!(blob.transform(), Affine::translation(700.0, 0.0));
        assert_eq!(blob.points()[0], Point::new(100, 0, true));
        // Component space is untouched: only `flatten` applies it.
        assert_eq!(blob.contours(), &[4]);
        assert_eq!(glyph.point_count(), 8);
        assert!(glyph.instructions().is_empty());
    }

    #[test]
    fn flatten_applies_component_transforms() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        let glyph = Glyph::load(&font, 3)
            .expect("load")
            .expect("outline");
        let flat = glyph.flatten();
        assert_eq!(flat.points.len(), 8);
        assert_eq!(flat.contours, vec![4, 8]);
        assert_eq!(flat.points[4], Point::new(800, 0, true));
        assert_eq!(flat.points[5], Point::new(1000, 600, false));
        assert!(!flat.is_empty());
        assert_eq!(flat.contour_count(), 2);
    }

    #[test]
    fn flatten_of_a_simple_glyph_is_the_identity() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        let glyph = Glyph::load(&font, 1)
            .expect("load")
            .expect("outline");
        let flat = glyph.flatten();
        assert_eq!(flat.points, glyph.sources()[0].points());
        assert_eq!(flat.contours, glyph.sources()[0].contours());
    }

    #[test]
    fn truncated_glyph_data_is_reported() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        // Header claims one contour but the payload ends right after
        // the bounding box.
        let raw = [
            0x00u8, 0x01, // numberOfContours = 1
            0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x10, // bbox
        ];
        assert!(matches!(
            Glyph::parse(&font, 1, &raw),
            Err(GlypharError::Truncated { .. })
        ));
    }

    #[test]
    fn non_monotonic_contours_are_malformed() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        let mut raw = vec![
            0x00, 0x02, // numberOfContours = 2
            0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x10, // bbox
            0x00, 0x05, // endPts[0] = 5
            0x00, 0x03, // endPts[1] = 3 (smaller)
            0x00, 0x00, // instructionLength
        ];
        raw.extend_from_slice(&[0x01; 6]); // flags
        raw.extend_from_slice(&[0x00; 12]); // x/y deltas
        assert!(matches!(
            Glyph::parse(&font, 1, &raw),
            Err(GlypharError::Malformed { .. })
        ));
    }

    #[test]
    fn point_matching_composites_are_rejected() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        let raw = [
            0xFF, 0xFF, // numberOfContours = −1
            0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x10, // bbox
            0x00, 0x01, // flags: words, no xy values
            0x00, 0x01, // glyphIndex 1
            0x00, 0x00, // arg1 = point 0
            0x00, 0x01, // arg2 = point 1
            0x00, 0x00, // no MORE_COMPONENTS
        ];
        assert!(matches!(
            Glyph::parse(&font, 3, &raw),
            Err(GlypharError::Unsupported { .. })
        ));
    }

    #[test]
    fn empty_payload_decodes_to_none() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        assert!(
            Glyph::parse(&font, 1, &[])
                .expect("none")
                .is_none()
        );
    }

    #[test]
    fn affine_compose_matches_sequential_application() {
        let p = Point::new(10, 20, true);
        let scale = Affine::scale(2.0, 3.0);
        let move_ = Affine::translation(5.0, -7.0);
        let composed = move_.compose(scale);
        let expected = move_.apply(scale.apply(p));
        assert_eq!(composed.apply(p), expected);
        assert!(
            Affine::IDENTITY
                .compose(Affine::IDENTITY)
                .is_identity()
        );
        assert!(!scale.compose(move_).is_identity());
    }

    #[test]
    fn round_to_i16_saturates_instead_of_panicking() {
        assert_eq!(round_to_i16(f32::NAN), 0);
        assert_eq!(round_to_i16(1.0e12), i16::MAX);
        assert_eq!(round_to_i16(-1.0e12), i16::MIN);
        assert_eq!(round_to_i16(2.6), 3);
    }

    #[test]
    fn bbox_union_grows_to_cover_both() {
        let a = BBox {
            x_min: 0,
            y_min: 0,
            x_max: 10,
            y_max: 10,
        };
        let b = BBox {
            x_min: -5,
            y_min: 3,
            x_max: 4,
            y_max: 40,
        };
        assert_eq!(
            a.union(b),
            BBox {
                x_min: -5,
                y_min: 0,
                x_max: 10,
                y_max: 40
            }
        );
        assert!(BBox::default().is_degenerate());
    }

    #[test]
    fn composite_with_scaled_component_rounds_offsets() {
        let bytes = font_bytes();
        let font = FontFile::parse(&bytes).expect("font");
        // Component 1 with a 0.5× scale and a (3, 7) offset.
        let raw = [
            0xFF,
            0xFF, // numberOfContours = −1
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x10,
            0x00,
            0x10, // bbox
            0x00,
            0x01 | 0x0002 | 0x0008, // words + xy + scale
            0x00,
            0x01, // glyphIndex 1
            0x00,
            0x03, // dx = 3
            0x00,
            0x07, // dy = 7
            0x20,
            0x00, // F2Dot14 0.5
            0x00,
            0x00, // no MORE_COMPONENTS
        ];
        let glyph = Glyph::parse(&font, 3, &raw)
            .expect("parse")
            .expect("outline");
        let transform = glyph.sources()[0].transform();
        assert_eq!(transform.a, 0.5);
        assert_eq!(transform.d, 0.5);
        assert_eq!(transform.e, 3.0);
        assert_eq!(transform.f, 7.0);
        let flat = glyph.flatten();
        // (50, 0) → (0.5·50 + 3, 0.5·0 + 7) = (28, 7)
        assert_eq!(flat.points[0], Point::new(28, 7, true));
    }
}
