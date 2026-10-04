//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License.  You may obtain a copy of the
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

//! Synthetic `glyf`/`loca` fixtures shared by this crate's unit tests
//! and by the integration tests in `tests/`.
//!
//! [`default_font`] assembles a checksummed SFNT file whose thirteen
//! glyphs cover the whole decision tree of `ttgload.c`: an empty glyph,
//! simple outlines with long/short/repeated point flags, composites
//! driven by XY offsets, point matching and a 0.5 scale, plus one
//! malformed glyph for every error the loader reports.  The module also
//! provides a [`GlyphSlot`] shaped like the one an outline driver
//! creates and the two [`SizeMetrics`] presets the tests load with.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use codevar_truetype_core::{
    Generic, GlyphFormat, GlyphLoader, GlyphMetrics, GlyphSlot, Library, Matrix, Outline, SizeMetrics,
    SlotInternal, Tag, TtResult, Vector, div_fix, make_tag,
};
use codevar_truetype_sfnt::{
    SfntDirectory, SfntFont,
    tags::{TAG_GLYF, TAG_HEAD, TAG_HHEA, TAG_HMTX, TAG_LOCA, TAG_MAXP},
};

/// `ARGS_ARE_WORDS`: the component's arguments are 16-bit values.
const ARGS_ARE_WORDS: u16 = 0x0001;
/// `ARGS_ARE_XY_VALUES`: the arguments are an x/y offset, not point
/// indices.
const ARGS_ARE_XY_VALUES: u16 = 0x0002;
/// `MORE_COMPONENTS`: another component follows.
const MORE_COMPONENTS: u16 = 0x0020;
/// `WE_HAVE_A_SCALE`: a single F2Dot14 scale follows the arguments.
const WE_HAVE_A_SCALE: u16 = 0x0008;
/// `WE_HAVE_AN_XY_SCALE`: two F2Dot14 scales follow the arguments.
const WE_HAVE_AN_XY_SCALE: u16 = 0x0040;
/// `WE_HAVE_A_2X2`: four F2Dot14 values follow the arguments.
const WE_HAVE_A_2X2: u16 = 0x0080;
/// `SCALED_COMPONENT_OFFSET`: scale the x/y offset with the component's
/// own transform.
const SCALED_COMPONENT_OFFSET: u16 = 0x0800;
/// `USE_MY_METRICS`: let this component supply the glyph's phantom
/// points.
const USE_MY_METRICS: u16 = 0x0200;

/// Appends `value` as big-endian bytes.
pub fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends `value` as big-endian bytes.
pub fn push_i16(out: &mut Vec<u8>, value: i16) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends `value` as big-endian bytes.
pub fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// One point of a simple glyph, in font units.
#[derive(Clone, Copy)]
pub struct Point {
    /// The absolute x coordinate.
    pub x: i64,
    /// The absolute y coordinate.
    pub y: i64,
    /// `true` for an on-curve point (`CURVE_TAG_ON`), `false` for an
    /// off-curve control point (`CURVE_TAG_CONIC`).
    pub on_curve: bool,
}

impl Point {
    /// An on-curve point at `(x, y)`.
    pub const fn new(x: i64, y: i64) -> Self {
        Point { x, y, on_curve: true }
    }

    /// An off-curve control point at `(x, y)`.
    pub const fn off(x: i64, y: i64) -> Self {
        Point {
            x,
            y,
            on_curve: false,
        }
    }
}

/// A component of a composite glyph, ready to be serialized.
///
/// The constructors set the argument-width bits of `flags` to match the
/// argument values they store; `more`, `scale` and friends add the
/// optional tail of the component record.
#[derive(Clone)]
pub struct Component {
    /// The glyph index this component references.
    pub index: u16,
    /// The raw `flags` word written into `glyf`.
    pub flags: u16,
    /// The first argument, interpreted according to [`Self::flags`].
    pub arg1: i32,
    /// The second argument.
    pub arg2: i32,
    /// The F2Dot14 words appended after the arguments.
    pub transform: Vec<i16>,
}

impl Component {
    /// An x/y offset given as two 16-bit words.
    pub fn xy(index: u16, x: i16, y: i16) -> Self {
        Component {
            index,
            flags: ARGS_ARE_WORDS | ARGS_ARE_XY_VALUES,
            arg1: i32::from(x),
            arg2: i32::from(y),
            transform: Vec::new(),
        }
    }

    /// An x/y offset given as two signed bytes.
    pub fn xy_bytes(index: u16, x: i8, y: i8) -> Self {
        Component {
            index,
            flags: ARGS_ARE_XY_VALUES,
            arg1: i32::from(x),
            arg2: i32::from(y),
            transform: Vec::new(),
        }
    }

    /// Point matching: attach point `l` of this component to point `k`
    /// of the components loaded so far (unsigned indices).
    pub fn points(index: u16, k: u16, l: u16) -> Self {
        Component {
            index,
            flags: ARGS_ARE_WORDS,
            arg1: i32::from(k),
            arg2: i32::from(l),
            transform: Vec::new(),
        }
    }

    /// Flags the component as non-final (`MORE_COMPONENTS`).
    pub fn more(mut self) -> Self {
        self.flags |= MORE_COMPONENTS;
        self
    }

    /// Appends a uniform F2Dot14 scale (`0x4000` is 1.0).
    pub fn scale(mut self, value: i16) -> Self {
        self.flags |= WE_HAVE_A_SCALE;
        self.transform.push(value);
        self
    }

    /// Appends a separate horizontal/vertical F2Dot14 scale.
    pub fn xy_scale(mut self, x: i16, y: i16) -> Self {
        self.flags |= WE_HAVE_AN_XY_SCALE;
        self.transform.extend([x, y]);
        self
    }

    /// Appends a full 2x2 F2Dot14 matrix in `xx, yx, xy, yy` order.
    pub fn matrix(mut self, xx: i16, yx: i16, xy: i16, yy: i16) -> Self {
        self.flags |= WE_HAVE_A_2X2;
        self.transform.extend([xx, yx, xy, yy]);
        self
    }

    /// Sets `SCALED_COMPONENT_OFFSET` on a scaled component.
    pub fn scaled_offset(mut self) -> Self {
        self.flags |= SCALED_COMPONENT_OFFSET;
        self
    }

    /// Sets `USE_MY_METRICS` so this component keeps the phantom points
    /// it produced.
    pub fn use_my_metrics(mut self) -> Self {
        self.flags |= USE_MY_METRICS;
        self
    }
}

/// The ten-byte `glyf` header of a glyph with `n_contours`.
pub fn glyph_header(n_contours: i16, x_min: i16, y_min: i16, x_max: i16, y_max: i16) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    push_i16(&mut out, n_contours);
    push_i16(&mut out, x_min);
    push_i16(&mut out, y_min);
    push_i16(&mut out, x_max);
    push_i16(&mut out, y_max);
    out
}

/// Encodes a well-formed simple glyph.
///
/// Point flags are chosen per coordinate (`xShortVector`, the
/// "same as previous" bits) and then run-length compressed with
/// `REPEAT`, so a single call exercises every branch of
/// `read_point_flags` and `read_point_coordinates`.
///
/// # Panics
///
/// Only when `contours` is empty or holds an empty contour; both are
/// programming errors of the fixture, never of the loader under test.
pub fn simple_glyph(
    x_min: i16,
    y_min: i16,
    x_max: i16,
    y_max: i16,
    contours: &[Vec<Point>],
    instructions: &[u8],
) -> Vec<u8> {
    assert!(!contours.is_empty(), "a simple glyph needs one contour");
    let mut out = glyph_header(contours.len() as i16, x_min, y_min, x_max, y_max);

    let mut total = 0usize;
    for contour in contours {
        assert!(!contour.is_empty(), "an empty contour has no end point");
        total += contour.len();
        push_i16(&mut out, (total - 1) as i16);
    }
    push_u16(&mut out, instructions.len() as u16);
    out.extend_from_slice(instructions);

    let points: Vec<Point> = contours.iter().flatten().copied().collect();
    let (flags, x_data, y_data) = encode_coordinates(&points);
    write_repeated_flags(&mut out, &flags);
    out.extend_from_slice(&x_data);
    out.extend_from_slice(&y_data);
    out
}

/// Chooses the flag bits and the payload bytes for every point.
fn encode_coordinates(points: &[Point]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut flags = Vec::with_capacity(points.len());
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    let mut previous = 0i64;
    for point in points {
        let mut flag = if point.on_curve { 1u8 } else { 0 };
        flag |= encode_delta(point.x - previous, 0x02, 0x10, &mut x_data);
        previous = point.x;
        flags.push(flag);
    }

    let mut previous = 0i64;
    for (index, point) in points.iter().enumerate() {
        let delta = point.y - previous;
        let bits = encode_delta(delta, 0x04, 0x20, &mut y_data);
        flags[index] |= bits;
        previous = point.y;
    }

    (flags, x_data, y_data)
}

/// Encodes one delta: returns the flag bits and appends the payload.
///
/// Zero uses the "same as previous" bit alone; only magnitudes that
/// genuinely fit a byte take the short form, in either direction.
fn encode_delta(delta: i64, short_bit: u8, same_bit: u8, out: &mut Vec<u8>) -> u8 {
    if delta == 0 {
        same_bit
    } else if (1..=255).contains(&delta) {
        out.push(delta as u8);
        short_bit | same_bit
    } else if (-255..=-1).contains(&delta) {
        out.push(delta.unsigned_abs() as u8);
        short_bit
    } else {
        push_i16(out, delta as i16);
        0
    }
}

/// Writes `flags` with the `REPEAT` (0x08) run-length encoding.
fn write_repeated_flags(out: &mut Vec<u8>, flags: &[u8]) {
    let mut index = 0usize;
    while index < flags.len() {
        let flag = flags[index];
        let mut run = 1usize;
        while index + run < flags.len() && flags[index + run] == flag && run < 255 {
            run += 1;
        }
        if run == 1 {
            out.push(flag);
        } else {
            out.push(flag | 0x08);
            out.push((run - 1) as u8);
        }
        index += run;
    }
}

/// Encodes a composite glyph from `components`.
pub fn composite_glyph(x_min: i16, y_min: i16, x_max: i16, y_max: i16, components: &[Component]) -> Vec<u8> {
    let mut out = glyph_header(-1, x_min, y_min, x_max, y_max);
    for component in components {
        let words = component.flags & ARGS_ARE_WORDS != 0;
        push_u16(&mut out, component.flags);
        push_u16(&mut out, component.index);
        // The width of the arguments depends only on `ARGS_ARE_WORDS`;
        // whether they mean offsets or point indices changes nothing
        // about the byte layout.
        if words {
            push_i16(&mut out, component.arg1 as i16);
            push_i16(&mut out, component.arg2 as i16);
        } else {
            out.push(component.arg1 as u8);
            out.push(component.arg2 as u8);
        }
        for value in &component.transform {
            push_i16(&mut out, *value);
        }
    }
    out
}

/// A 54-byte `'head'` table: `unitsPerEm` 1000, long `loca`.
pub fn head_table() -> Vec<u8> {
    let mut out = Vec::with_capacity(54);
    push_u32(&mut out, 0x0001_0000);
    push_u32(&mut out, 0x0001_0000);
    push_u32(&mut out, 0);
    push_u32(&mut out, 0x5F0F_3CF5);
    push_u16(&mut out, 3);
    push_u16(&mut out, 1000);
    out.extend_from_slice(&0i64.to_be_bytes());
    out.extend_from_slice(&0i64.to_be_bytes());
    push_i16(&mut out, -100);
    push_i16(&mut out, -200);
    push_i16(&mut out, 900);
    push_i16(&mut out, 800);
    push_u16(&mut out, 0);
    push_u16(&mut out, 8);
    push_i16(&mut out, 2);
    push_i16(&mut out, 1);
    push_i16(&mut out, 0);
    out
}

/// A 32-byte `'maxp'` table (version 1.0) for `num_glyphs` glyphs.
pub fn maxp_table(num_glyphs: u16, max_component_depth: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    push_u32(&mut out, 0x0001_0000);
    push_u16(&mut out, num_glyphs);
    for field in [16u16, 4, 32, 8, 2, 64, 64, 16, 0, 256, 64, 8] {
        push_u16(&mut out, field);
    }
    push_u16(&mut out, max_component_depth);
    out
}

/// A 36-byte `'hhea'` table: ascender 800, descender -200.
pub fn hhea_table(number_of_h_metrics: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(36);
    push_u32(&mut out, 0x0001_0000);
    push_i16(&mut out, 800);
    push_i16(&mut out, -200);
    push_i16(&mut out, 90);
    push_u16(&mut out, 700);
    push_i16(&mut out, -50);
    push_i16(&mut out, -40);
    push_i16(&mut out, 900);
    push_i16(&mut out, 1);
    push_i16(&mut out, 0);
    push_i16(&mut out, 0);
    for _ in 0..4 {
        push_i16(&mut out, 0);
    }
    push_i16(&mut out, 0);
    push_u16(&mut out, number_of_h_metrics);
    out
}

/// An `'hmtx'` table with one long metric per glyph.
pub fn hmtx_table(num_glyphs: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(usize::from(num_glyphs) * 4);
    for index in 0..num_glyphs {
        push_u16(&mut out, 500 + index * 10);
        push_i16(&mut out, -10 + ((index % 7) as i16) * 3);
    }
    out
}

/// The `(glyf, loca)` pair for `glyphs`, using the long `loca` format.
pub fn glyf_and_loca(glyphs: &[Vec<u8>]) -> (Vec<u8>, Vec<u8>) {
    let mut glyf = Vec::new();
    let mut loca = Vec::new();
    push_u32(&mut loca, 0);
    for glyph in glyphs {
        glyf.extend_from_slice(glyph);
        push_u32(&mut loca, glyf.len() as u32);
    }
    (glyf, loca)
}

/// Assembles `tables` into a complete SFNT file: sorted directory,
/// correct `searchRange`/`entrySelector`/`rangeShift`, 4-byte aligned
/// offsets and spec-conformant checksums (`'head'` with
/// `checkSumAdjustment` zeroed).
pub fn build_font(tables: &[(Tag, Vec<u8>)]) -> Vec<u8> {
    let mut sorted: Vec<(Tag, &[u8])> = tables
        .iter()
        .map(|(tag, body)| (*tag, body.as_slice()))
        .collect();
    sorted.sort_by_key(|(tag, _)| *tag);
    let count = sorted.len() as u16;
    let mut entry_selector = 0u16;
    let mut largest_pow2 = 1u16;
    while largest_pow2 * 2 <= count {
        largest_pow2 *= 2;
        entry_selector += 1;
    }
    let search_range = largest_pow2 * 16;
    let range_shift = count * 16 - search_range;
    let mut out = Vec::new();
    push_u32(&mut out, 0x0001_0000);
    push_u16(&mut out, count);
    push_u16(&mut out, search_range);
    push_u16(&mut out, entry_selector);
    push_u16(&mut out, range_shift);
    let dir_start = out.len();
    out.resize(dir_start + count as usize * 16, 0);
    for (index, (tag, body)) in sorted.iter().enumerate() {
        let table_offset = out.len() as u32;
        let checksum = if *tag == TAG_HEAD && body.len() >= 12 {
            let mut head = body.to_vec();
            head[8..12].copy_from_slice(&0u32.to_be_bytes());
            SfntDirectory::checksum(&head)
        } else {
            SfntDirectory::checksum(body)
        };
        let record = dir_start + index * 16;
        out[record..record + 4].copy_from_slice(&tag.to_be_bytes());
        out[record + 4..record + 8].copy_from_slice(&checksum.to_be_bytes());
        out[record + 8..record + 12].copy_from_slice(&table_offset.to_be_bytes());
        out[record + 12..record + 16].copy_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(body);
        let padding = (4 - (body.len() % 4)) % 4;
        out.resize(out.len() + padding, 0);
    }
    out
}

/// Wraps `glyphs` into a complete, checksummed SFNT font.
pub fn font_bytes(glyphs: &[Vec<u8>], max_component_depth: u16) -> Vec<u8> {
    let (glyf, loca) = glyf_and_loca(glyphs);
    let num_glyphs = glyphs.len() as u16;
    build_font(&[
        (TAG_GLYF, glyf),
        (TAG_HEAD, head_table()),
        (TAG_HHEA, hhea_table(num_glyphs)),
        (TAG_HMTX, hmtx_table(num_glyphs)),
        (TAG_LOCA, loca),
        (TAG_MAXP, maxp_table(num_glyphs, max_component_depth)),
    ])
}

/// The rectangle of glyph 1: one contour, four on-curve points.
pub fn rect_points() -> Vec<Point> {
    vec![
        Point::new(0, 0),
        Point::new(500, 0),
        Point::new(500, 700),
        Point::new(0, 700),
    ]
}

/// The quadratic arc of glyph 2: one contour, on/off/on.
pub fn quad_points() -> Vec<Point> {
    vec![Point::new(0, 0), Point::off(250, 700), Point::new(500, 0)]
}

/// The thirteen glyphs of [`default_font`], in index order.
///
/// | index | contents |
/// |-------|----------|
/// | 0 | empty (`loca[0] == loca[1]`) |
/// | 1 | simple rectangle |
/// | 2 | simple quadratic arc |
/// | 3 | composite of 1 and 2, XY offsets |
/// | 4 | `numberOfContours == -2` |
/// | 5 | header only (no contour ends) |
/// | 6 | composite that references itself |
/// | 7 | composite of the composite (depth 2) |
/// | 8 | composite using point matching |
/// | 9 | composite with a 0.5 `WE_HAVE_A_SCALE` |
/// | 10 | simple glyph with repeated flags and instructions |
/// | 11 | unordered contour ends |
/// | 12 | `n_ins` larger than the glyph |
/// | 13 | composite with `USE_MY_METRICS` |
/// | 14 | scaled composite with `SCALED_COMPONENT_OFFSET` |
pub fn glyph_blobs() -> Vec<Vec<u8>> {
    vec![
        Vec::new(),
        simple_glyph(0, 0, 500, 700, &[rect_points()], &[]),
        simple_glyph(0, 0, 500, 700, &[quad_points()], &[]),
        composite_glyph(
            0,
            0,
            600,
            900,
            &[Component::xy(1, 100, 200).more(), Component::xy(2, 50, 60)],
        ),
        glyph_header(-2, 0, 0, 10, 10),
        glyph_header(1, 0, 0, 500, 700),
        composite_glyph(0, 0, 10, 10, &[Component::xy(6, 10, 10)]),
        composite_glyph(0, 0, 600, 900, &[Component::xy(3, 10, 10)]),
        composite_glyph(
            0,
            0,
            1000,
            700,
            &[Component::xy(1, 0, 0).more(), Component::points(2, 1, 0)],
        ),
        composite_glyph(
            0,
            0,
            600,
            800,
            &[
                Component::xy(1, 0, 0).scale(0x2000).more(),
                Component::xy(2, 100, 100),
            ],
        ),
        // Adjacent points share their flag words twice over, so the
        // decoder has to expand two `REPEAT` runs before it reaches the
        // three instruction bytes that follow the contour ends.
        simple_glyph(
            0,
            0,
            600,
            300,
            &[vec![
                Point::new(0, 0),
                Point::off(300, 0),
                Point::off(600, 0),
                Point::new(600, 300),
                Point::new(300, 300),
                Point::new(0, 300),
            ]],
            &[0x00, 0x01, 0x02],
        ),
        unordered_contours_glyph(),
        too_many_hints_glyph(),
        composite_glyph(0, 0, 600, 700, &[Component::xy(1, 0, 0).use_my_metrics()]),
        composite_glyph(
            0,
            0,
            400,
            450,
            &[Component::xy(2, 100, 100)
                .scale(0x2000)
                .scaled_offset()],
        ),
    ]
}

/// Glyph 11: two contours whose ends run backwards.
fn unordered_contours_glyph() -> Vec<u8> {
    let mut out = glyph_header(2, 0, 0, 500, 700);
    push_i16(&mut out, 3);
    push_i16(&mut out, 1);
    push_u16(&mut out, 0);
    out.extend_from_slice(&[0x01, 0x01]);
    out.extend_from_slice(&0i16.to_be_bytes());
    out.extend_from_slice(&0i16.to_be_bytes());
    out
}

/// Glyph 12: `n_ins` claims 100 bytes the glyph does not hold.
fn too_many_hints_glyph() -> Vec<u8> {
    let mut out = glyph_header(1, 0, 0, 10, 10);
    push_i16(&mut out, 0);
    push_u16(&mut out, 100);
    out
}

/// The [`font_bytes`] fixture with a permissive `maxComponentDepth`.
pub fn default_font() -> Vec<u8> {
    font_bytes(&glyph_blobs(), 5)
}

/// The same font but with `maxComponentDepth` clamped to `depth`.
///
/// `load_truetype_glyph` allows `recurse_count == 1` unconditionally, so
/// `depth == 0` only rejects glyphs nested two levels deep — glyph 7.
pub fn font_with_depth(depth: u16) -> Vec<u8> {
    font_bytes(&glyph_blobs(), depth)
}

/// Opens `bytes` as sub-font 0, the way the driver's `init_face` does.
///
/// # Errors
///
/// Whatever [`SfntFont::open`] reports for the container.
pub fn open(bytes: &[u8]) -> TtResult<SfntFont<'_>> {
    SfntFont::open(bytes, 0)
}

/// A [`GlyphSlot`] shaped like the one an outline-capable driver builds:
/// `internal` present and holding a fresh [`GlyphLoader`].
pub fn new_slot() -> GlyphSlot {
    GlyphSlot {
        library: Arc::new(Library::new()),
        generic: Generic::default(),
        metrics: GlyphMetrics::default(),
        linear_hori_advance: 0,
        linear_vert_advance: 0,
        advance: Vector::new(0, 0),
        format: GlyphFormat::None,
        bitmap: Default::default(),
        bitmap_left: 0,
        bitmap_top: 0,
        outline: Outline::new(),
        subglyphs: Vec::new(),
        control_data: None,
        lsb_delta: 0,
        rsb_delta: 0,
        other: None,
        internal: Some(SlotInternal {
            loader: Some(GlyphLoader::new()),
            flags: 0,
            glyph_transformed: false,
            glyph_matrix: Matrix::default(),
        }),
        driver_data: None,
    }
}

/// `FT_LOAD_NO_SCALE` metrics: identity scales, so every coordinate
/// survives byte-for-byte from `glyf` into the outline.
pub fn no_scale_size() -> SizeMetrics {
    SizeMetrics {
        x_ppem: 0,
        y_ppem: 0,
        x_scale: 0x1_0000,
        y_scale: 0x1_0000,
        ..SizeMetrics::default()
    }
}

/// Metrics for `ppem` pixels per em, using FreeType's
/// `FT_DivFix(ppem << 6, units_per_em)` scale.
pub fn scaled_size(units_per_em: u16, ppem: u16) -> SizeMetrics {
    let scale = div_fix(i64::from(ppem) * 64, i64::from(units_per_em));
    SizeMetrics {
        x_ppem: ppem,
        y_ppem: ppem,
        x_scale: scale,
        y_scale: scale,
        ..SizeMetrics::default()
    }
}

/// The `make_tag` helper, re-exported so the test files do not have to
/// name another crate.
pub fn tag(a: u8, b: u8, c: u8, d: u8) -> Tag {
    make_tag(a, b, c, d)
}
