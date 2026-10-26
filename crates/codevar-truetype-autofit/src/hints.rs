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

//! Glyph hint analysis (`afhints.c` / `afhints.h`).
//!
//! Every writing system shares the [`GlyphHints`] machinery: points,
//! segments and edges of one outline, plus the grid-fitting passes
//! ([`GlyphHints::align_edge_points`], [`GlyphHints::align_strong_points`]
//! and [`GlyphHints::align_weak_points`]).
//!
//! The analysis pipeline of [`GlyphHints::reload`] is split into the
//! same steps FreeType performs in one large function: table
//! allocation, point loading, contour linking, direction computation
//! and the two weak-point tagging passes.

use alloc::vec::Vec;

use codevar_truetype_core::{
    CURVE_TAG_CONIC, CURVE_TAG_CUBIC, CURVE_TAG_ON, Fixed, Orientation, Outline, Pos, corner_is_flat,
    div_fix, mul_fix,
};

use crate::metrics::StyleMetrics;
use crate::{
    DIMENSION_MAX, Dimension, Direction, SCALER_FLAG_NO_ADVANCE, SCALER_FLAG_NO_HORIZONTAL,
    SCALER_FLAG_NO_VERTICAL, SCALER_FLAG_NO_WARPER, direction_compute,
};

/// Point flag: no flags (`AF_FLAG_NONE`).
pub const FLAG_NONE: u16 = 0;
/// Point flag: conic control point (`AF_FLAG_CONIC`).
pub const FLAG_CONIC: u16 = 1 << 0;
/// Point flag: cubic control point (`AF_FLAG_CUBIC`).
pub const FLAG_CUBIC: u16 = 1 << 1;
/// Point flag: control point of any kind (`AF_FLAG_CONTROL`).
pub const FLAG_CONTROL: u16 = FLAG_CONIC | FLAG_CUBIC;
/// Point flag: X coordinate has been touched (`AF_FLAG_TOUCH_X`).
pub const FLAG_TOUCH_X: u16 = 1 << 2;
/// Point flag: Y coordinate has been touched (`AF_FLAG_TOUCH_Y`).
pub const FLAG_TOUCH_Y: u16 = 1 << 3;
/// Point flag: candidate for weak interpolation
/// (`AF_FLAG_WEAK_INTERPOLATION`).
pub const FLAG_WEAK_INTERPOLATION: u16 = 1 << 4;

/// Edge flag: a freshly created edge with no further properties
/// (`AF_EDGE_NORMAL`).
pub const EDGE_NORMAL: u8 = 0;
/// Edge flag: rounded edge (`AF_EDGE_ROUND`).
pub const EDGE_ROUND: u8 = 1 << 0;
/// Edge flag: serif edge (`AF_EDGE_SERIF`).
pub const EDGE_SERIF: u8 = 1 << 1;
/// Edge flag: edge has been hinted (`AF_EDGE_DONE`).
pub const EDGE_DONE: u8 = 1 << 2;
/// Edge flag: edge aligns to a neutral blue zone (`AF_EDGE_NEUTRAL`).
pub const EDGE_NEUTRAL: u8 = 1 << 3;

/// `AF_SEGMENTS_EMBEDDED`: initial capacity of an axis segment array.
const SEGMENTS_EMBEDDED: usize = 18;
/// `AF_EDGES_EMBEDDED`: initial capacity of an axis edge array.
const EDGES_EMBEDDED: usize = 12;
/// `AF_POINTS_EMBEDDED`: initial capacity of the point array.
const POINTS_EMBEDDED: usize = 96;
/// `AF_CONTOURS_EMBEDDED`: initial capacity of the contour array.
const CONTOURS_EMBEDDED: usize = 8;

/// `AF_PointRec` (`afhints.h`): one outline point as seen by the
/// hinter.
///
/// The C record's circular `next`/`prev` pointers become indices into
/// [`GlyphHints::points`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Point {
    /// Point flags used by the hinter (`FLAG_*`).
    pub flags: u16,
    /// Direction of the inwards vector.
    pub in_dir: Direction,
    /// Direction of the outwards vector.
    pub out_dir: Direction,
    /// Original, scaled position.
    pub ox: Pos,
    /// Original, scaled position.
    pub oy: Pos,
    /// Original, unscaled position (in font units).
    pub fx: i16,
    /// Original, unscaled position (in font units).
    pub fy: i16,
    /// Current position.
    pub x: Pos,
    /// Current position.
    pub y: Pos,
    /// Current coordinate (x or y, depending on context), also used
    /// for index deltas while loading the outline.
    pub u: Pos,
    /// Original coordinate, also used for index deltas while loading
    /// the outline.
    pub v: Pos,
    /// Index of the next point in the contour.
    pub next: usize,
    /// Index of the previous point in the contour.
    pub prev: usize,
}

/// `AF_SegmentRec` (`afhints.h`): a series of approximately aligned
/// points of one dimension.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Segment {
    /// Segment flags (currently unused by the auto-hinter).
    pub flags: u8,
    /// Segment direction.
    pub dir: Direction,
    /// Position of the segment.
    pub pos: i16,
    /// Minimum coordinate of the segment.
    pub min_coord: i16,
    /// Maximum coordinate of the segment.
    pub max_coord: i16,
    /// The hinted segment height.
    pub height: i16,
    /// Index of the segment's parent edge.
    pub edge: Option<usize>,
    /// Link to the next segment in the parent edge.
    pub edge_next: Option<usize>,
    /// (Stem) link segment.
    pub link: Option<usize>,
    /// Primary segment for serifs.
    pub serif: Option<usize>,
    /// Number of linked segments.
    pub num_linked: Pos,
    /// Used during stem matching.
    pub score: Pos,
    /// Used during stem matching.
    pub len: Pos,
    /// Index of the first point of the segment.
    pub first: Option<usize>,
    /// Index of the last point of the segment.
    pub last: Option<usize>,
}

impl Segment {
    /// `AF_SEGMENT_LEN`: the length of the segment along its axis.
    #[inline]
    pub fn length(&self) -> i32 {
        i32::from(self.max_coord) - i32::from(self.min_coord)
    }

    /// `AF_SEGMENT_DIST`: the distance between two segments along
    /// their axis.
    #[inline]
    pub fn distance(&self, other: &Segment) -> i32 {
        if self.pos > other.pos {
            i32::from(self.pos) - i32::from(other.pos)
        } else {
            i32::from(other.pos) - i32::from(self.pos)
        }
    }
}

/// `AF_EdgeRec` (`afhints.h`): one or more segments collected at a
/// single position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Edge {
    /// Original, unscaled position (in font units).
    ///
    /// Edges of an axis are sorted by `fpos`; [`AxisHints::new_edge`]
    /// maintains this invariant.
    pub fpos: i16,
    /// Original, scaled position.
    pub opos: Pos,
    /// Current position.
    pub pos: Pos,
    /// Edge flags (`EDGE_*`).
    pub flags: u8,
    /// Edge direction.
    pub dir: Direction,
    /// Used to speed up interpolation between edges.
    pub scale: Fixed,
    /// Set if this is a blue edge (copied from the blue zone of the
    /// metrics when the edge is created).
    pub blue_edge: Option<crate::Width>,
    /// Link edge.
    pub link: Option<usize>,
    /// Primary edge for serifs.
    pub serif: Option<usize>,
    /// Number of linked edges.
    pub num_linked: i16,
    /// Used during stem matching.
    pub score: i32,
    /// Index of the first segment in the edge.
    pub first: Option<usize>,
    /// Index of the last segment in the edge.
    pub last: Option<usize>,
}

/// `AF_AxisHintsRec` (`afhints.h`): segments and edges of one
/// dimension.
///
/// The C record keeps embedded arrays that are promoted to the heap
/// once full; this port simply uses pre-allocated `Vec`s with the same
/// initial capacities, which preserves the allocation behavior for
/// typical glyphs.
#[derive(Debug)]
pub struct AxisHints {
    /// Segments of this dimension (in creation order).
    pub segments: Vec<Segment>,
    /// Edges of this dimension (sorted by `fpos`).
    pub edges: Vec<Edge>,
    /// Either [`Direction::Up`] or [`Direction::Down`] for
    /// [`Dimension::Hort`], or [`Direction::Left`] /
    /// [`Direction::Right`] for [`Dimension::Vert`].
    pub major_dir: Direction,
}

impl AxisHints {
    /// Creates an axis with the embedded capacities of
    /// `AF_AxisHintsRec`.
    #[inline]
    pub fn new() -> AxisHints {
        AxisHints {
            segments: Vec::with_capacity(SEGMENTS_EMBEDDED),
            edges: Vec::with_capacity(EDGES_EMBEDDED),
            major_dir: Direction::None,
        }
    }

    /// Number of used segments (`num_segments`).
    #[inline]
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Number of used edges (`num_edges`).
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// `af_axis_hints_new_segment`: appends a new, uninitialized
    /// segment and returns its index.
    ///
    /// # Porting note
    ///
    /// FreeType returns `FT_Err_Out_Of_Memory` when the segment array
    /// cannot grow; Rust's allocator aborts instead, so this function
    /// is infallible.
    #[inline]
    pub fn new_segment(&mut self) -> usize {
        self.segments.push(Segment::default());
        self.segments.len() - 1
    }

    /// `af_axis_hints_new_edge`: inserts a new, uninitialized edge so
    /// that edges stay sorted by `fpos`, with edges of equal position
    /// ordered minor-direction first, and returns its index.
    ///
    /// # Porting note
    ///
    /// FreeType returns `FT_Err_Out_Of_Memory` when the edge array
    /// cannot grow; Rust's allocator aborts instead, so this function
    /// is infallible.
    pub fn new_edge(&mut self, fpos: i16, dir: Direction) -> usize {
        let mut index = self.num_edges();
        self.edges.push(Edge::default());
        while index > 0 {
            let prev = self.edges[index - 1];
            if prev.fpos < fpos {
                break;
            }
            // We want the edge with same position and minor direction
            // to appear before those in the major one in the list
            if prev.fpos == fpos && dir == self.major_dir {
                break;
            }
            self.edges[index] = prev;
            index -= 1;
        }
        self.edges[index] = Edge::default();
        index
    }
}

impl Default for AxisHints {
    #[inline]
    fn default() -> AxisHints {
        AxisHints::new()
    }
}

/// `AF_GlyphHintsRec` (`afhints.h`): the analysis of one glyph.
#[derive(Debug)]
pub struct GlyphHints {
    /// Scale from font units to 1/64th device pixels (horizontal).
    pub x_scale: Fixed,
    /// Added delta in 1/64th device pixels (horizontal).
    pub x_delta: Pos,
    /// Scale from font units to 1/64th device pixels (vertical).
    pub y_scale: Fixed,
    /// Added delta in 1/64th device pixels (vertical).
    pub y_delta: Pos,
    /// The points of the outline.
    pub points: Vec<Point>,
    /// Index of the first point of each contour.
    pub contours: Vec<usize>,
    /// Segment and edge analysis of both dimensions.
    pub axis: [AxisHints; DIMENSION_MAX],
    /// Copy of the scaler flags (`AF_HINTS_TEST_SCALER`).
    pub scaler_flags: u32,
    /// Free for style specific implementations (`AF_HINTS_TEST_OTHER`).
    pub other_flags: u32,
    /// The style metrics of the current glyph, cloned in
    /// [`GlyphHints::rescale`] (FreeType stores a pointer into the
    /// face globals).
    pub metrics: Option<StyleMetrics>,
    /// Used for warping (light rendering).
    pub xmin_delta: Pos,
    /// Used for warping (light rendering).
    pub xmax_delta: Pos,
    /// `face->units_per_EM`, cached from the metrics at
    /// [`GlyphHints::rescale`] time.
    pub units_per_em: u16,
}

impl Default for GlyphHints {
    /// `af_glyph_hints_init` (`afhints.c`): initializes the hint
    /// record (only the non-embedded parts are cleared in C).
    fn default() -> GlyphHints {
        GlyphHints {
            x_scale: 0,
            x_delta: 0,
            y_scale: 0,
            y_delta: 0,
            points: Vec::with_capacity(POINTS_EMBEDDED),
            contours: Vec::with_capacity(CONTOURS_EMBEDDED),
            axis: [AxisHints::new(), AxisHints::new()],
            scaler_flags: 0,
            other_flags: 0,
            metrics: None,
            xmin_delta: 0,
            xmax_delta: 0,
            units_per_em: 0,
        }
    }
}

impl GlyphHints {
    /// `af_glyph_hints_done` (`afhints.c`): releases all resources.
    ///
    /// FreeType keeps the record usable afterwards; in Rust this
    /// simply clears the vectors (keeping their capacity) and drops
    /// the metrics.
    pub fn done(&mut self) {
        self.points.clear();
        self.contours.clear();
        for axis in &mut self.axis {
            axis.segments.clear();
            axis.edges.clear();
        }
        self.scaler_flags = 0;
        self.other_flags = 0;
        self.metrics = None;
        self.x_scale = 0;
        self.x_delta = 0;
        self.y_scale = 0;
        self.y_delta = 0;
        self.xmin_delta = 0;
        self.xmax_delta = 0;
        self.units_per_em = 0;
    }

    /// `af_glyph_hints_rescale` (`afhints.c`): attaches the style
    /// metrics of the current glyph and copies the scaler flags.
    ///
    /// # Porting note
    ///
    /// FreeType stores a pointer to the metrics owned by the face
    /// globals; this port clones them, which is behaviorally identical
    /// because every mutation of the metrics happens in
    /// `style_metrics_scale` (before this call) and the clone is
    /// refreshed for each glyph.
    #[inline]
    pub fn rescale(&mut self, metrics: &StyleMetrics) {
        self.scaler_flags = metrics.scaler().flags;
        self.units_per_em = metrics.units_per_em_face();
        self.metrics = Some(metrics.clone());
    }

    /// `AF_HINTS_TEST_SCALER`: true when `flag` is not set in the
    /// scaler flags.
    #[inline]
    pub fn test_scaler(&self, flag: u32) -> bool {
        self.scaler_flags & flag == 0
    }

    /// `AF_HINTS_TEST_OTHER`: true when `flag` is set in the
    /// style-specific flags.
    #[inline]
    pub fn test_other(&self, flag: u32) -> bool {
        self.other_flags & flag != 0
    }

    /// `AF_HINTS_DO_HORIZONTAL`: horizontal hinting enabled.
    #[inline]
    pub fn do_horizontal(&self) -> bool {
        self.test_scaler(SCALER_FLAG_NO_HORIZONTAL)
    }

    /// `AF_HINTS_DO_VERTICAL`: vertical hinting enabled.
    #[inline]
    pub fn do_vertical(&self) -> bool {
        self.test_scaler(SCALER_FLAG_NO_VERTICAL)
    }

    /// `AF_HINTS_DO_ADVANCE`: advance hinting enabled.
    #[inline]
    pub fn do_advance(&self) -> bool {
        self.test_scaler(SCALER_FLAG_NO_ADVANCE)
    }

    /// `AF_HINTS_DO_WARP`: the warper may be used.
    #[inline]
    pub fn do_warp(&self) -> bool {
        self.test_scaler(SCALER_FLAG_NO_WARPER)
    }

    /// Number of used points.
    #[inline]
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Number of used contours.
    #[inline]
    pub fn num_contours(&self) -> usize {
        self.contours.len()
    }
}

/// `AF_HINTS_DO_BLUES` (`afhints.h`): blue zones take part in the
/// hinting process.
///
/// FreeType only turns this off for `FT_DEBUG_AUTOFIT` builds, so the
/// hint record is not an input of the predicate and the macro argument
/// of the C call sites is dropped here.
#[inline]
pub const fn do_blues() -> bool {
    true
}

/// `FT_Outline_Get_Orientation` (`ftoutln.c`): determines the fill
/// direction of `outline` with the nonzero winding rule, operating on
/// the polygon spanned by the control points.
///
/// Returns [`Orientation::TrueType`] for empty or malformed outlines
/// (FreeType only special-cases the empty case and reads out of bounds
/// otherwise).
///
/// # Porting note
///
/// The core crate does not export this helper, so the auto-hinter
/// carries its own copy, exactly as FreeType does for its users.
#[must_use]
pub fn outline_get_orientation(outline: &Outline) -> Orientation {
    if outline.n_points <= 0 || outline.check().is_err() {
        return Orientation::TrueType;
    }

    // We use the nonzero winding rule to find the orientation.
    // Since glyph outlines behave much more `regular' than arbitrary
    // cubic or quadratic curves, this test deals with the polygon
    // only that is spanned up by the control points.
    let cbox = outline.get_cbox();
    // Handle collapsed outlines to avoid undefined FT_MSB.
    if cbox.x_min == cbox.x_max || cbox.y_min == cbox.y_max {
        return Orientation::None;
    }
    // `FT_MSB(value) - 14`, clamped at zero (`FT_MAX(xshift, 0)`).
    let xshift = msb_shift((cbox.x_max.wrapping_abs() | cbox.x_min.wrapping_abs()) as u32);
    let yshift = msb_shift(cbox.y_max.wrapping_sub(cbox.y_min) as u32);

    let mut area: i64 = 0;
    let mut first = 0usize;
    for c in 0..outline.n_contours as usize {
        let last = outline.contours[c] as usize;
        let mut v_prev = outline.points[last];
        for n in first..=last {
            let v_cur = outline.points[n];
            area = area.wrapping_add(
                (v_cur.y.wrapping_sub(v_prev.y) >> yshift)
                    .wrapping_mul(v_cur.x.wrapping_add(v_prev.x) >> xshift),
            );
            v_prev = v_cur;
        }
        first = last + 1;
    }

    if area > 0 {
        Orientation::Postscript
    } else if area < 0 {
        Orientation::TrueType
    } else {
        Orientation::None
    }
}

/// `FT_MSB(bits) - 14`, clamped at zero, used by
/// [`outline_get_orientation`].  Returns `0` for `bits == 0` (the
/// collapsed-outline case is handled by the caller).
#[inline]
fn msb_shift(bits: u32) -> i32 {
    if bits == 0 {
        return 0;
    }
    let msb = u32::BITS - 1 - bits.leading_zeros();
    i32::try_from(msb).map_or(0, |value| (value - 14).max(0))
}

/// Converts an index delta (stored in [`Point::u`] / [`Point::v`])
/// back into a point index (`point + point->u` in C).
///
/// `reload` only ever stores differences of real point indices, so the
/// result is in bounds; the clamp keeps malformed or externally
/// modified state from panicking.
#[inline]
fn delta_index(len: usize, index: usize, delta: Pos) -> usize {
    debug_assert!(index < len);
    let target = (index as i64).saturating_add(delta);
    if target < 0 {
        0
    } else if len == 0 || target >= len as i64 {
        len.saturating_sub(1)
    } else {
        target as usize
    }
}

impl GlyphHints {
    /// `af_glyph_hints_reload` (`afhints.c`): recomputes all points
    /// from the definitions of a source outline.
    ///
    /// # Porting note
    ///
    /// FreeType assumes a well-formed outline and reads out of bounds
    /// otherwise; this port validates `outline` first (see
    /// [`Outline::check`]) and clears the analysis instead of
    /// panicking.  C also saves and restores the scale fields around
    /// the table reallocation; nothing in between modifies them, so
    /// the dead round trip is omitted here.
    pub fn reload(&mut self, outline: &Outline) {
        self.clear_analysis();
        self.set_major_dirs(outline);
        self.xmin_delta = 0;
        self.xmax_delta = 0;

        if outline.check().is_err() || outline.is_empty() {
            return;
        }

        self.allocate_tables(outline);
        self.load_points(outline);
        self.build_contour_starts(outline);
        self.compute_directions();
        self.tag_quadrant_weak_points();
        self.tag_dominant_weak_points();
    }

    /// Clears the point, contour, segment and edge arrays
    /// (`hints->num_points = 0` and friends).
    fn clear_analysis(&mut self) {
        self.points.clear();
        self.contours.clear();
        for axis in &mut self.axis {
            axis.segments.clear();
            axis.edges.clear();
        }
    }

    /// Sets the major direction of both axes from the fill orientation
    /// of `outline`.  C recomputes the orientation every time because
    /// some fonts have broken `FT_Outline.flags` values.
    fn set_major_dirs(&mut self, outline: &Outline) {
        self.axis[Dimension::Hort.index()].major_dir = Direction::Up;
        self.axis[Dimension::Vert.index()].major_dir = Direction::Left;

        if outline_get_orientation(outline) == Orientation::Postscript {
            self.axis[Dimension::Hort.index()].major_dir = Direction::Down;
            self.axis[Dimension::Vert.index()].major_dir = Direction::Right;
        }
    }

    /// Grows the point and contour tables to the size of `outline`
    /// (the caller has validated it).
    fn allocate_tables(&mut self, outline: &Outline) {
        // We reserve two additional point positions in FreeType,
        // used to hint metrics appropriately; a `Vec` grows lazily
        // instead, which is behaviorally identical here.
        self.contours
            .resize(outline.n_contours as usize, 0);
        self.points
            .resize(outline.n_points as usize, Point::default());
    }

    /// Computes coordinates and Bezier flags and wires the circular
    /// `next`/`prev` links of every point (including the `FT_MulFix`
    /// scaling).
    fn load_points(&mut self, outline: &Outline) {
        let n_points = outline.n_points as usize;
        let n_contours = outline.n_contours as usize;
        let x_scale = self.x_scale;
        let y_scale = self.y_scale;
        let x_delta = self.x_delta;
        let y_delta = self.y_delta;

        let mut prev = outline.contours[0] as usize;
        let mut end = prev;
        let mut contour_index = 0usize;

        for i in 0..n_points {
            let vec = outline.points[i];
            let tag = outline.tags[i];

            let fx = vec.x as i16;
            let fy = vec.y as i16;
            let ox = mul_fix(vec.x, x_scale) + x_delta;
            let oy = mul_fix(vec.y, y_scale) + y_delta;
            let flags = match tag & 3 {
                CURVE_TAG_CONIC => FLAG_CONIC,
                CURVE_TAG_CUBIC => FLAG_CUBIC,
                _ => FLAG_NONE,
            };

            let point = &mut self.points[i];
            point.in_dir = Direction::None;
            point.out_dir = Direction::None;
            point.fx = fx;
            point.fy = fy;
            point.ox = ox;
            point.x = ox;
            point.oy = oy;
            point.y = oy;
            point.flags = flags;
            point.prev = prev;

            self.points[prev].next = i;
            prev = i;

            if i == end {
                contour_index += 1;
                if contour_index < n_contours {
                    end = outline.contours[contour_index] as usize;
                    prev = end;
                }
            }
        }
    }

    /// Sets up the contours array: the index of the first point of
    /// each contour (`hints->contours[i] = points + idx`).
    fn build_contour_starts(&mut self, outline: &Outline) {
        let mut idx = 0usize;
        for (c, contour) in self.contours.iter_mut().enumerate() {
            *contour = idx;
            idx = outline.contours[c] as usize + 1;
        }
    }

    /// Computes the `in`/`out` directions of every point.
    ///
    /// Distances between points that are very near to each other are
    /// accumulated: the auto-hinter prepends the small vectors between
    /// near points to the first non-near vector.  All intermediate
    /// points are tagged as weak; the directions are adjusted also to
    /// be equal to the accumulated one.
    fn compute_directions(&mut self) {
        // value 20 in `near_limit' is heuristic
        let near_limit = 20 * i32::from(self.units_per_em) / 2048;
        let near_limit2 = 2 * near_limit - 1;

        let num_contours = self.contours.len();
        for ci in 0..num_contours {
            let contour_start = self.contours[ci];
            let first = adjust_contour_start(&self.points, contour_start, near_limit2);
            compute_contour_vectors(&mut self.points, first, near_limit);
        }
    }

    /// The first weak-point pass (`afhints.c`): a series of
    /// non-horizontal or non-vertical vectors pointing into the same
    /// quadrant is handled as a single, long vector; the intermediate
    /// points are tagged as weak.
    fn tag_quadrant_weak_points(&mut self) {
        for i in 0..self.points.len() {
            if self.points[i].flags & FLAG_WEAK_INTERPOLATION != 0 {
                continue;
            }

            if self.points[i].in_dir == Direction::None && self.points[i].out_dir == Direction::None {
                // Check whether both vectors point into the same quadrant
                let next_u = delta_index(self.points.len(), i, self.points[i].u);
                let prev_v = delta_index(self.points.len(), i, self.points[i].v);

                let in_x = i64::from(self.points[i].fx) - i64::from(self.points[prev_v].fx);
                let in_y = i64::from(self.points[i].fy) - i64::from(self.points[prev_v].fy);

                let out_x = i64::from(self.points[next_u].fx) - i64::from(self.points[i].fx);
                let out_y = i64::from(self.points[next_u].fy) - i64::from(self.points[i].fy);

                if (in_x ^ out_x) >= 0 && (in_y ^ out_y) >= 0 {
                    // yes, so tag current point as weak
                    // and update index deltas
                    self.points[i].flags |= FLAG_WEAK_INTERPOLATION;
                    let delta = (next_u as i64) - (prev_v as i64);
                    self.points[prev_v].u = delta;
                    self.points[next_u].v = -delta;
                }
            }
        }
    }

    /// The second weak-point pass (`afhints.c`): control points,
    /// points on a flat corner and spikes are tagged as weak;
    /// everything else not collected into edges so far is implicitly
    /// classified as strong.
    fn tag_dominant_weak_points(&mut self) {
        for i in 0..self.points.len() {
            if self.points[i].flags & FLAG_WEAK_INTERPOLATION != 0 {
                continue;
            }

            if self.points[i].flags & FLAG_CONTROL != 0 {
                // Control points are always weak
                self.points[i].flags |= FLAG_WEAK_INTERPOLATION;
            } else if self.points[i].out_dir == self.points[i].in_dir {
                if self.points[i].out_dir != Direction::None {
                    // Current point lies on a horizontal or
                    // vertical segment (but doesn't start or end it)
                    self.points[i].flags |= FLAG_WEAK_INTERPOLATION;
                } else {
                    let next_u = delta_index(self.points.len(), i, self.points[i].u);
                    let prev_v = delta_index(self.points.len(), i, self.points[i].v);

                    if corner_is_flat(
                        i64::from(self.points[i].fx) - i64::from(self.points[prev_v].fx),
                        i64::from(self.points[i].fy) - i64::from(self.points[prev_v].fy),
                        i64::from(self.points[next_u].fx) - i64::from(self.points[i].fx),
                        i64::from(self.points[next_u].fy) - i64::from(self.points[i].fy),
                    ) {
                        // either the `in' or the `out' vector is much more
                        // dominant than the other one, so tag current point
                        // as weak and update index deltas
                        let delta = (next_u as i64) - (prev_v as i64);
                        self.points[prev_v].u = delta;
                        self.points[next_u].v = -delta;

                        self.points[i].flags |= FLAG_WEAK_INTERPOLATION;
                    }
                }
            } else if i32::from(self.points[i].in_dir.code()) == -i32::from(self.points[i].out_dir.code()) {
                // Current point forms a spike
                self.points[i].flags |= FLAG_WEAK_INTERPOLATION;
            }
        }
    }

    /// `af_glyph_hints_save` (`afhints.c`): stores the hinted outline
    /// back into an [`Outline`] structure.
    ///
    /// # Porting note
    ///
    /// FreeType writes `hints->num_points` entries into the outline
    /// and assumes the caller sized it accordingly; this port clamps
    /// to the outline capacity instead of writing out of bounds.
    pub fn save(&self, outline: &mut Outline) {
        let limit = self
            .points
            .len()
            .min(outline.points.len())
            .min(outline.tags.len());
        for (i, point) in self.points.iter().take(limit).enumerate() {
            outline.points[i].x = point.x;
            outline.points[i].y = point.y;

            outline.tags[i] = if point.flags & FLAG_CONIC != 0 {
                CURVE_TAG_CONIC
            } else if point.flags & FLAG_CUBIC != 0 {
                CURVE_TAG_CUBIC
            } else {
                CURVE_TAG_ON
            };
        }
    }

    /// `af_glyph_hints_align_edge_points` (`afhints.c`): aligns all
    /// points of every segment to the position of its edge, either
    /// horizontally or vertically.
    pub fn align_edge_points(&mut self, dim: Dimension) {
        let horz = dim == Dimension::Hort;
        let touch = if horz { FLAG_TOUCH_X } else { FLAG_TOUCH_Y };
        let num_segments = self.axis[dim.index()].segments.len();
        let num_points = self.points.len();

        for s in 0..num_segments {
            let seg = self.axis[dim.index()].segments[s];
            let (Some(edge_index), Some(first), Some(last)) = (seg.edge, seg.first, seg.last) else {
                continue;
            };
            if edge_index >= self.axis[dim.index()].edges.len() || first >= num_points {
                continue;
            }
            let pos = self.axis[dim.index()].edges[edge_index].pos;

            let mut point = first;
            let mut steps = 0usize;
            loop {
                if point >= num_points {
                    break;
                }
                if horz {
                    self.points[point].x = pos;
                } else {
                    self.points[point].y = pos;
                }
                self.points[point].flags |= touch;

                if point == last {
                    break;
                }
                point = self.points[point].next;
                // The ring must reach `last` in at most one lap
                steps += 1;
                if steps > num_points {
                    break;
                }
            }
        }
    }

    /// `af_glyph_hints_align_strong_points` (`afhints.c`): hints the
    /// strong points; this is equivalent to the TrueType `IP`
    /// hinting instruction.
    pub fn align_strong_points(&mut self, dim: Dimension) {
        let touch_flag = if dim == Dimension::Hort {
            FLAG_TOUCH_X
        } else {
            FLAG_TOUCH_Y
        };
        if self.axis[dim.index()].edges.is_empty() {
            return;
        }

        for i in 0..self.points.len() {
            if self.points[i].flags & (touch_flag | FLAG_WEAK_INTERPOLATION) != 0 {
                // if this point is candidate to weak interpolation, we
                // interpolate it after all strong points have been processed
                continue;
            }

            let (fu, ou) = if dim == Dimension::Vert {
                (i64::from(self.points[i].fy), self.points[i].oy)
            } else {
                (i64::from(self.points[i].fx), self.points[i].ox)
            };

            let u = match self.locate_strong_edge(dim, fu) {
                StrongLookup::BeforeFirst => {
                    let edge = self.axis[dim.index()].edges[0];
                    edge.pos - (edge.opos - ou)
                }
                StrongLookup::AfterLast => {
                    let edges = &self.axis[dim.index()].edges;
                    let edge = edges[edges.len() - 1];
                    edge.pos + (ou - edge.opos)
                }
                StrongLookup::OnEdge(pos) => pos,
                StrongLookup::Between { before, after } => self.interpolate_strong(dim, before, after, fu),
            };

            self.store_point(i, u, touch_flag, dim);
        }
    }

    /// Locates the font-unit coordinate `u` among the edges of `dim`
    /// (the search block of `af_glyph_hints_align_strong_points`).
    ///
    /// Edges must be sorted by `fpos` (maintained by
    /// [`AxisHints::new_edge`]).
    fn locate_strong_edge(&self, dim: Dimension, u: i64) -> StrongLookup {
        let edges = &self.axis[dim.index()].edges;
        let num_edges = edges.len();
        debug_assert!(num_edges > 0);

        // Is the point before the first edge?
        if i64::from(edges[0].fpos) - u >= 0 {
            return StrongLookup::BeforeFirst;
        }
        // Is the point after the last edge?
        if u - i64::from(edges[num_edges - 1].fpos) >= 0 {
            return StrongLookup::AfterLast;
        }
        // Find enclosing edges
        let min = if num_edges <= 8 {
            // For a small number of edges, a linear search is better
            let mut nn = 0usize;
            while nn < num_edges && i64::from(edges[nn].fpos) < u {
                nn += 1;
            }
            // `edges[0].fpos < u < edges[num_edges - 1].fpos` guarantees a hit
            if nn < num_edges && i64::from(edges[nn].fpos) == u {
                return StrongLookup::OnEdge(edges[nn].pos);
            }
            nn
        } else {
            let mut min = 0usize;
            let mut max = num_edges;
            while min < max {
                let mid = (max + min) >> 1;
                let fpos = i64::from(edges[mid].fpos);
                if u < fpos {
                    max = mid;
                } else if u > fpos {
                    min = mid + 1;
                } else {
                    // We are on the edge
                    return StrongLookup::OnEdge(edges[mid].pos);
                }
            }
            min
        };

        // `edges[0].fpos < u < edges[num_edges - 1].fpos` guarantees a
        // surrounding pair of edges
        debug_assert!(min > 0 && min < num_edges);
        StrongLookup::Between {
            before: min - 1,
            after: min,
        }
    }

    /// Interpolates a point position between the `before` and `after`
    /// edges, caching the scale factor on the `before` edge like
    /// FreeType does.
    fn interpolate_strong(
        &mut self,
        dim: Dimension,
        before_index: usize,
        after_index: usize,
        fu: Pos,
    ) -> Pos {
        let after = self.axis[dim.index()].edges[after_index];
        let before_slot = &mut self.axis[dim.index()].edges[before_index];
        if before_slot.scale == 0 {
            before_slot.scale = div_fix(
                after.pos - before_slot.pos,
                i64::from(after.fpos) - i64::from(before_slot.fpos),
            );
        }
        let before = self.axis[dim.index()].edges[before_index];
        before.pos + mul_fix(fu - i64::from(before.fpos), before.scale)
    }

    /// `af_glyph_hints_align_weak_points` (`afhints.c`): shifts and
    /// interpolates all remaining points; this is equivalent to the
    /// TrueType `IUP` hinting instruction.
    pub fn align_weak_points(&mut self, dim: Dimension) {
        let touch_flag = if dim == Dimension::Hort {
            FLAG_TOUCH_X
        } else {
            FLAG_TOUCH_Y
        };
        self.stash_coord(dim);
        let num_contours = self.contours.len();
        for ci in 0..num_contours {
            let start = self.contours[ci];
            iup_contour(&mut self.points, start, touch_flag);
        }
        self.apply_coord(dim);
    }

    /// Stashes the current coordinate of dimension `dim` into `u` and
    /// the original one into `v`.
    fn stash_coord(&mut self, dim: Dimension) {
        if dim == Dimension::Hort {
            for point in &mut self.points {
                point.u = point.x;
                point.v = point.ox;
            }
        } else {
            for point in &mut self.points {
                point.u = point.y;
                point.v = point.oy;
            }
        }
    }

    /// Restores the coordinate of dimension `dim` from `u`.
    fn apply_coord(&mut self, dim: Dimension) {
        if dim == Dimension::Hort {
            for point in &mut self.points {
                point.x = point.u;
            }
        } else {
            for point in &mut self.points {
                point.y = point.u;
            }
        }
    }

    /// `af_glyph_hints_scale_dim` (`afhints.c`, only used by the
    /// warper): applies a (small) warp scale and warp delta for the
    /// given dimension.
    pub fn scale_dim(&mut self, dim: Dimension, scale: Fixed, delta: Pos) {
        if dim == Dimension::Hort {
            for point in &mut self.points {
                point.x = mul_fix(point.fx.into(), scale) + delta;
            }
        } else {
            for point in &mut self.points {
                point.y = mul_fix(point.fy.into(), scale) + delta;
            }
        }
    }

    /// Stores the hinted coordinate `u` of dimension `dim` into point
    /// `index` and marks it as touched (`Store_Point`).
    #[inline]
    fn store_point(&mut self, index: usize, u: Pos, touch_flag: u16, dim: Dimension) {
        if dim == Dimension::Hort {
            self.points[index].x = u;
        } else {
            self.points[index].y = u;
        }
        self.points[index].flags |= touch_flag;
    }
}

/// The result of locating a font-unit coordinate among the axis edges
/// (the `goto Store_Point` decisions of
/// `af_glyph_hints_align_strong_points`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StrongLookup {
    /// The point lies at or before the first edge.
    BeforeFirst,
    /// The point lies at or after the last edge.
    AfterLast,
    /// The point sits exactly on an edge (hinted position given).
    OnEdge(Pos),
    /// The point lies strictly between two edges.
    Between {
        /// Index of the enclosing edge before the point.
        before: usize,
        /// Index of the enclosing edge after the point.
        after: usize,
    },
}

/// Walks backwards from the first point of a contour while the points
/// are near to each other and returns the first non-near point
/// (`af_glyph_hints_reload`).
///
/// We use Taxicab metrics to measure the vector length.  The
/// accumulated distances so far could have the opposite direction of
/// the distance measured here; for this reason `near_limit2` is used
/// for the comparison to get a non-near point even in the worst case.
fn adjust_contour_start(points: &[Point], first: usize, near_limit2: i32) -> usize {
    let mut point = first;
    let mut prev = points[first].prev;

    while prev != first {
        let out_x = i64::from(points[point].fx) - i64::from(points[prev].fx);
        let out_y = i64::from(points[point].fy) - i64::from(points[prev].fy);

        if out_x.abs() + out_y.abs() >= i64::from(near_limit2) {
            break;
        }
        point = prev;
        prev = points[prev].prev;
    }
    point
}

/// Loops over all points of one contour to get the `in` and `out`
/// vector directions, accumulating near vectors and tagging the
/// intermediate points as weak (`af_glyph_hints_reload`).
///
/// The `u` and `v` fields are abused to store index deltas to the next
/// and previous non-near point, respectively.  To avoid problems with
/// not having non-near points, we point to `first` by default as the
/// next non-near point.
fn compute_contour_vectors(points: &mut [Point], first: usize, near_limit: i32) {
    let mut curr = first;

    points[curr].u = (first as i64) - (curr as i64);
    points[first].v = -points[curr].u;

    let mut out_x: i64 = 0;
    let mut out_y: i64 = 0;

    let mut next = first;
    loop {
        let point_i = next;
        next = points[point_i].next;

        out_x += i64::from(points[next].fx) - i64::from(points[point_i].fx);
        out_y += i64::from(points[next].fy) - i64::from(points[point_i].fy);

        if out_x.abs() + out_y.abs() < i64::from(near_limit) {
            points[next].flags |= FLAG_WEAK_INTERPOLATION;
        } else {
            points[curr].u = (next as i64) - (curr as i64);
            points[next].v = -points[curr].u;

            let out_dir = direction_compute(out_x, out_y);

            // Adjust directions for all points inbetween;
            // the loop also updates the position of `curr'
            points[curr].out_dir = out_dir;
            curr = points[curr].next;
            while curr != next {
                points[curr].in_dir = out_dir;
                points[curr].out_dir = out_dir;
                curr = points[curr].next;
            }
            points[next].in_dir = out_dir;

            points[curr].u = (first as i64) - (curr as i64);
            points[first].v = -points[curr].u;

            out_x = 0;
            out_y = 0;
        }

        if next == first {
            break;
        }
    }
}

/// Applies the IUP shift/interpolation to one contour: finds the
/// touched points of the contour starting at `start` and fills the
/// untouched points between them (`af_glyph_hints_align_weak_points`).
fn iup_contour(points: &mut [Point], start: usize, touch_flag: u16) {
    if start >= points.len() {
        return;
    }
    let end_point = points[start].prev;
    if end_point >= points.len() {
        return;
    }
    let first_point = start;

    // Find first touched point
    let mut point = start;
    let first_touched = loop {
        if point > end_point {
            // No touched point in contour
            return;
        }
        if points[point].flags & touch_flag != 0 {
            break point;
        }
        point += 1;
    };

    let mut last_touched;
    'segment: loop {
        // Skip any touched neighbours
        while point < end_point && points[point + 1].flags & touch_flag != 0 {
            point += 1;
        }
        last_touched = point;
        // Find the next touched point, if any
        point += 1;
        loop {
            if point > end_point {
                break 'segment;
            }
            if points[point].flags & touch_flag != 0 {
                break;
            }
            point += 1;
        }

        // Interpolate between last_touched and point
        iup_interp(points, last_touched + 1, point - 1, last_touched, point);
    }

    // Special case: only one point was touched
    if last_touched == first_touched {
        iup_shift(points, first_point, end_point, first_touched);
    } else if last_touched < end_point {
        // Interpolate the last part
        iup_interp(points, last_touched + 1, end_point, last_touched, first_touched);
    }
    if last_touched != first_touched && first_touched > 0 {
        iup_interp(
            points,
            first_point,
            first_touched - 1,
            last_touched,
            first_touched,
        );
    }
}

/// `af_iup_shift` (`afhints.c`): shifts the original coordinates of
/// all points between `p1` and `p2` to get hinted coordinates, using
/// the same difference as given by `ref`.
fn iup_shift(points: &mut [Point], p1: usize, p2: usize, reference: usize) {
    let delta = points[reference].u - points[reference].v;
    if delta == 0 {
        return;
    }

    let mut p = p1;
    while p < reference {
        points[p].u = points[p].v + delta;
        p += 1;
    }
    p = reference + 1;
    while p <= p2 {
        points[p].u = points[p].v + delta;
        p += 1;
    }
}

/// `af_iup_interp` (`afhints.c`): interpolates the original
/// coordinates of all points between `p1` and `p2` to get hinted
/// coordinates, using `ref1` and `ref2` as the reference points.  The
/// `u` and `v` members are the current and original coordinate values,
/// respectively.
///
/// Details can be found in the TrueType bytecode specification.
fn iup_interp(points: &mut [Point], p1: usize, p2: usize, ref1: usize, ref2: usize) {
    if p1 > p2 {
        return;
    }

    let (ref1, ref2) = if points[ref1].v > points[ref2].v {
        (ref2, ref1)
    } else {
        (ref1, ref2)
    };

    let v1 = points[ref1].v;
    let v2 = points[ref2].v;
    let u1 = points[ref1].u;
    let u2 = points[ref2].u;
    let d1 = u1 - v1;
    let d2 = u2 - v2;

    if u1 == u2 || v1 == v2 {
        for point in &mut points[p1..=p2] {
            let mut u = point.v;

            if u <= v1 {
                u += d1;
            } else if u >= v2 {
                u += d2;
            } else {
                u = u1;
            }

            point.u = u;
        }
    } else {
        let scale = div_fix(u2 - u1, v2 - v1);

        for point in &mut points[p1..=p2] {
            let mut u = point.v;

            if u <= v1 {
                u += d1;
            } else if u >= v2 {
                u += d2;
            } else {
                u = u1 + mul_fix(u - v1, scale);
            }
            point.u = u;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;

    use codevar_truetype_core::Vector;

    use crate::metrics::GlobalsShared;
    use crate::ranges::{STYLE_CLASSES, Style};

    /// Hints prepared for the given `units_per_EM` with identity
    /// scales (`font units == 1/64th device pixels`).
    fn hints_upem(upem: u16) -> GlyphHints {
        let mut hints = GlyphHints::default();
        let globals = Rc::new(GlobalsShared::new(upem));
        let metrics = StyleMetrics::new(&STYLE_CLASSES[Style::LatnDflt.index()], globals);
        hints.rescale(&metrics);
        hints.x_scale = 1 << 16;
        hints.y_scale = 1 << 16;
        hints.xmin_delta = 5;
        hints.xmax_delta = 7;
        hints
    }

    /// Four-point square; `cw` selects the clockwise (TrueType) or
    /// counter-clockwise (PostScript) point order.
    fn square(cw: bool) -> Outline {
        let coords: [(Pos, Pos); 4] = if cw {
            [(0, 0), (0, 100), (100, 100), (100, 0)]
        } else {
            [(0, 0), (100, 0), (100, 100), (0, 100)]
        };
        let mut outline = Outline::new();
        for (x, y) in coords {
            outline.points.push(Vector::new(x, y));
            outline.tags.push(CURVE_TAG_ON);
        }
        outline.contours.push(3);
        outline.n_points = 4;
        outline.n_contours = 1;
        outline
    }

    /// Builds a single-contour outline from a coordinate list.
    fn contour(coords: &[(i64, i64)], tags: &[u8]) -> Outline {
        let mut outline = Outline::new();
        for &(x, y) in coords {
            outline.points.push(Vector::new(x, y));
        }
        outline.tags.extend_from_slice(tags);
        outline.contours.push((coords.len() - 1) as i16);
        outline.n_points = coords.len() as i16;
        outline.n_contours = 1;
        outline
    }

    /// Wires the circular `next`/`prev` links of `len` points.
    fn link_ring(points: &mut [Point], len: usize) {
        for (i, point) in points.iter_mut().enumerate().take(len) {
            point.next = (i + 1) % len;
            point.prev = (i + len - 1) % len;
        }
    }

    /// Appends an edge with explicit `fpos`/`opos`/`pos`.
    fn add_edge(hints: &mut GlyphHints, dim: Dimension, fpos: i16, opos: Pos, pos: Pos) {
        let axis = &mut hints.axis[dim.index()];
        axis.edges.push(Edge {
            fpos,
            opos,
            pos,
            ..Edge::default()
        });
    }

    #[test]
    fn segment_length_and_distance() {
        let seg = Segment {
            pos: 10,
            min_coord: 5,
            max_coord: 20,
            ..Segment::default()
        };
        assert_eq!(seg.length(), 15);

        let other = Segment {
            pos: 3,
            ..Segment::default()
        };
        assert_eq!(seg.distance(&other), 7);
        assert_eq!(other.distance(&seg), 7);

        let other = Segment {
            pos: 12,
            ..Segment::default()
        };
        assert_eq!(seg.distance(&other), 2);
    }

    #[test]
    fn new_edge_sorts_by_fpos_with_minor_first() {
        let mut axis = AxisHints::new();
        assert_eq!(axis.num_edges(), 0);

        axis.major_dir = Direction::Up;
        let up10 = axis.new_edge(10, Direction::Up);
        axis.edges[up10].fpos = 10;
        axis.edges[up10].dir = Direction::Up;
        assert_eq!(up10, 0);

        let up5 = axis.new_edge(5, Direction::Up);
        axis.edges[up5].fpos = 5;
        axis.edges[up5].dir = Direction::Up;
        assert_eq!(up5, 0, "smaller fpos shifts to the front");

        let left10 = axis.new_edge(10, Direction::Left);
        axis.edges[left10].fpos = 10;
        axis.edges[left10].dir = Direction::Left;
        assert_eq!(left10, 1, "minor direction at equal fpos comes first");

        let up7 = axis.new_edge(7, Direction::Up);
        axis.edges[up7].fpos = 7;
        axis.edges[up7].dir = Direction::Up;
        assert_eq!(up7, 1);

        assert_eq!(axis.num_edges(), 4);
        let fpos: alloc::vec::Vec<i16> = axis.edges.iter().map(|e| e.fpos).collect();
        assert_eq!(fpos, [5, 7, 10, 10]);
        assert_eq!(axis.edges[2].dir, Direction::Left);
        assert_eq!(axis.edges[3].dir, Direction::Up);
    }

    #[test]
    fn new_segment_appends() {
        let mut axis = AxisHints::new();
        assert_eq!(axis.new_segment(), 0);
        assert_eq!(axis.new_segment(), 1);
        assert_eq!(axis.num_segments(), 2);
        assert_eq!(axis.num_edges(), 0);
    }

    #[test]
    fn reload_square_loads_points_and_links() {
        let mut hints = hints_upem(1000);
        hints.reload(&square(true));

        assert_eq!(hints.num_points(), 4);
        assert_eq!(hints.num_contours(), 1);
        assert_eq!(hints.contours, [0]);
        assert_eq!(hints.xmin_delta, 0);
        assert_eq!(hints.xmax_delta, 0);

        for (i, &(fx, fy)) in [(0, 0), (0, 100), (100, 100), (100, 0)]
            .iter()
            .enumerate()
        {
            let point = &hints.points[i];
            assert_eq!(point.fx, fx as i16);
            assert_eq!(point.fy, fy as i16);
            assert_eq!(point.ox, fx);
            assert_eq!(point.oy, fy);
            assert_eq!(point.x, fx);
            assert_eq!(point.y, fy);
            assert_eq!(point.flags, FLAG_NONE);
            assert_eq!(point.flags & FLAG_WEAK_INTERPOLATION, 0);
        }

        assert_eq!(hints.points[0].prev, 3);
        assert_eq!(hints.points[0].next, 1);
        assert_eq!(hints.points[1].prev, 0);
        assert_eq!(hints.points[2].next, 3);
        assert_eq!(hints.points[3].prev, 2);
        assert_eq!(hints.points[3].next, 0);

        assert_eq!(hints.points[0].in_dir, Direction::Left);
        assert_eq!(hints.points[0].out_dir, Direction::Up);
        assert_eq!(hints.points[1].in_dir, Direction::Up);
        assert_eq!(hints.points[1].out_dir, Direction::Right);
        assert_eq!(hints.points[2].in_dir, Direction::Right);
        assert_eq!(hints.points[2].out_dir, Direction::Down);
        assert_eq!(hints.points[3].in_dir, Direction::Down);
        assert_eq!(hints.points[3].out_dir, Direction::Left);

        assert_eq!(hints.axis[Dimension::Hort.index()].major_dir, Direction::Up);
        assert_eq!(hints.axis[Dimension::Vert.index()].major_dir, Direction::Left);
    }

    #[test]
    fn reload_postscript_outline_sets_down_right_major_dirs() {
        let mut hints = hints_upem(1000);
        hints.reload(&square(false));
        assert_eq!(hints.num_points(), 4);
        assert_eq!(hints.axis[Dimension::Hort.index()].major_dir, Direction::Down);
        assert_eq!(hints.axis[Dimension::Vert.index()].major_dir, Direction::Right);
    }

    #[test]
    fn reload_accumulates_near_points_as_weak() {
        let tags = [CURVE_TAG_ON; 5];
        let outline = contour(&[(0, 0), (5, 0), (100, 0), (100, 100), (0, 100)], &tags);
        let mut hints = hints_upem(1000);
        hints.reload(&outline);

        assert_eq!(hints.num_points(), 5);
        assert_eq!(hints.contours, [0]);
        assert_ne!(hints.points[1].flags & FLAG_WEAK_INTERPOLATION, 0);
        for i in [0, 2, 3, 4] {
            assert_eq!(hints.points[i].flags & FLAG_WEAK_INTERPOLATION, 0);
        }
        assert_eq!(hints.points[0].in_dir, Direction::Down);
        assert_eq!(hints.points[0].out_dir, Direction::Right);
        assert_eq!(hints.points[1].in_dir, Direction::Right);
        assert_eq!(hints.points[1].out_dir, Direction::Right);
        assert_eq!(hints.points[2].out_dir, Direction::Up);
        assert_eq!(hints.points[4].in_dir, Direction::Left);
        assert_eq!(hints.points[4].out_dir, Direction::Down);
    }

    #[test]
    fn reload_quadrant_pass_tags_same_quadrant_series() {
        let tags = [CURVE_TAG_ON; 4];
        let outline = contour(&[(0, 0), (100, 90), (200, 180), (300, 90)], &tags);
        let mut hints = hints_upem(1000);
        hints.reload(&outline);

        assert_ne!(hints.points[1].flags & FLAG_WEAK_INTERPOLATION, 0);
        assert_eq!(hints.points[0].u, 2);
        assert_eq!(hints.points[2].v, -2);
        assert_eq!(hints.points[2].flags & FLAG_WEAK_INTERPOLATION, 0);
        assert_eq!(hints.points[3].flags & FLAG_WEAK_INTERPOLATION, 0);
    }

    #[test]
    fn reload_tags_conic_control_point_weak() {
        let tags = [
            CURVE_TAG_ON,
            CURVE_TAG_CONIC,
            CURVE_TAG_ON,
            CURVE_TAG_ON,
            CURVE_TAG_ON,
        ];
        let outline = contour(&[(0, 0), (100, 50), (200, 0), (200, 100), (0, 100)], &tags);
        let mut hints = hints_upem(1000);
        hints.reload(&outline);

        let control = &hints.points[1];
        assert_ne!(control.flags & FLAG_CONIC, 0);
        assert_ne!(control.flags & FLAG_WEAK_INTERPOLATION, 0);
        for i in [0, 2, 3, 4] {
            assert_eq!(hints.points[i].flags & FLAG_WEAK_INTERPOLATION, 0);
        }
    }

    #[test]
    fn reload_malformed_outline_clears_analysis() {
        let mut hints = hints_upem(1000);
        hints.reload(&square(true));
        assert_eq!(hints.num_points(), 4);

        let mut bad = square(true);
        bad.contours = alloc::vec![10];
        hints.reload(&bad);
        assert_eq!(hints.num_points(), 0);
        assert_eq!(hints.num_contours(), 0);
        assert!(
            hints.axis[Dimension::Hort.index()]
                .segments
                .is_empty()
        );
        assert!(
            hints.axis[Dimension::Hort.index()]
                .edges
                .is_empty()
        );
        assert_eq!(hints.axis[Dimension::Hort.index()].major_dir, Direction::Up);

        let mut short = square(true);
        short.n_points = 9;
        hints.reload(&short);
        assert_eq!(hints.num_points(), 0);
    }

    #[test]
    fn reload_empty_outline_is_a_noop() {
        let mut hints = hints_upem(1000);
        hints.reload(&square(true));
        hints.reload(&Outline::new());
        assert_eq!(hints.num_points(), 0);
        assert_eq!(hints.num_contours(), 0);
        assert!(hints.points.is_empty());
        assert!(hints.contours.is_empty());
    }

    #[test]
    fn orientation_matches_fill_direction() {
        assert_eq!(outline_get_orientation(&square(true)), Orientation::TrueType);
        assert_eq!(outline_get_orientation(&square(false)), Orientation::Postscript);
        assert_eq!(outline_get_orientation(&Outline::new()), Orientation::TrueType);

        let collapsed = contour(&[(5, 0), (5, 100)], &[CURVE_TAG_ON, CURVE_TAG_ON]);
        assert_eq!(outline_get_orientation(&collapsed), Orientation::None);

        let mut bad = square(true);
        bad.contours = alloc::vec![10];
        assert_eq!(outline_get_orientation(&bad), Orientation::TrueType);
    }

    #[test]
    fn save_writes_positions_and_tags() {
        let mut hints = GlyphHints::default();
        hints.points.resize(3, Point::default());
        hints.points[0].x = 10;
        hints.points[0].y = 20;
        hints.points[0].flags = FLAG_CONIC;
        hints.points[1].x = 30;
        hints.points[1].y = 40;
        hints.points[1].flags = FLAG_CUBIC;
        hints.points[2].x = 50;
        hints.points[2].y = 60;
        hints.points[2].flags = FLAG_TOUCH_X;

        let mut outline = square(true);
        hints.save(&mut outline);
        assert_eq!(outline.points[0], Vector::new(10, 20));
        assert_eq!(outline.points[1], Vector::new(30, 40));
        assert_eq!(outline.points[2], Vector::new(50, 60));
        assert_eq!(outline.tags[0], CURVE_TAG_CONIC);
        assert_eq!(outline.tags[1], CURVE_TAG_CUBIC);
        assert_eq!(outline.tags[2], CURVE_TAG_ON);

        hints.points.resize(10, Point::default());
        hints.save(&mut outline);
        assert_eq!(outline.points[0], Vector::new(10, 20));
        assert_eq!(outline.points.len(), 4);
    }

    #[test]
    fn align_edge_points_moves_segment_to_edge() {
        let mut hints = GlyphHints::default();
        hints.points.resize(4, Point::default());
        link_ring(&mut hints.points, 4);
        let axis = &mut hints.axis[Dimension::Hort.index()];
        axis.edges.push(Edge {
            pos: 64,
            ..Edge::default()
        });
        let segment = axis.new_segment();
        axis.segments[segment].edge = Some(0);
        axis.segments[segment].first = Some(0);
        axis.segments[segment].last = Some(2);

        hints.align_edge_points(Dimension::Hort);
        assert_eq!(hints.points[0].x, 64);
        assert_eq!(hints.points[1].x, 64);
        assert_eq!(hints.points[2].x, 64);
        assert_eq!(hints.points[3].x, 0);
        assert_ne!(hints.points[0].flags & FLAG_TOUCH_X, 0);
        assert_ne!(hints.points[1].flags & FLAG_TOUCH_X, 0);
        assert_ne!(hints.points[2].flags & FLAG_TOUCH_X, 0);
        assert_eq!(hints.points[3].flags & FLAG_TOUCH_X, 0);
    }

    #[test]
    fn align_edge_points_vertical_dimension() {
        let mut hints = GlyphHints::default();
        hints.points.resize(3, Point::default());
        link_ring(&mut hints.points, 3);
        let axis = &mut hints.axis[Dimension::Vert.index()];
        axis.edges.push(Edge {
            pos: -16,
            ..Edge::default()
        });
        let segment = axis.new_segment();
        axis.segments[segment].edge = Some(0);
        axis.segments[segment].first = Some(1);
        axis.segments[segment].last = Some(2);

        hints.align_edge_points(Dimension::Vert);
        assert_eq!(hints.points[0].y, 0);
        assert_eq!(hints.points[1].y, -16);
        assert_eq!(hints.points[2].y, -16);
        assert_eq!(hints.points[1].flags & FLAG_TOUCH_X, 0);
        assert_ne!(hints.points[1].flags & FLAG_TOUCH_Y, 0);
    }

    #[test]
    fn align_edge_points_skips_invalid_segments() {
        let mut hints = GlyphHints::default();
        hints.points.resize(2, Point::default());
        link_ring(&mut hints.points, 2);
        hints.points[0].x = 7;
        hints.points[1].x = 8;
        let axis = &mut hints.axis[Dimension::Hort.index()];
        axis.edges.push(Edge {
            pos: 64,
            ..Edge::default()
        });

        let dangling = axis.new_segment();
        axis.segments[dangling].edge = Some(9);
        axis.segments[dangling].first = Some(0);
        axis.segments[dangling].last = Some(1);

        let no_edge = axis.new_segment();
        axis.segments[no_edge].edge = None;
        axis.segments[no_edge].first = Some(0);
        axis.segments[no_edge].last = Some(1);

        let bad_first = axis.new_segment();
        axis.segments[bad_first].edge = Some(0);
        axis.segments[bad_first].first = Some(9);
        axis.segments[bad_first].last = Some(1);

        let corrupted = axis.new_segment();
        axis.segments[corrupted].edge = Some(0);
        axis.segments[corrupted].first = Some(0);
        axis.segments[corrupted].last = Some(1);
        hints.points[0].next = 99;

        hints.align_edge_points(Dimension::Hort);
        assert_eq!(hints.points[0].x, 64);
        assert_eq!(hints.points[1].x, 8);
    }

    #[test]
    fn align_strong_points_extrapolates_and_interpolates() {
        let mut hints = GlyphHints::default();
        add_edge(&mut hints, Dimension::Hort, 0, 0, 0);
        add_edge(&mut hints, Dimension::Hort, 100, 100, 128);
        hints.points.resize(6, Point::default());

        hints.points[0].fx = -10;
        hints.points[0].ox = -10;
        hints.points[1].fx = 150;
        hints.points[1].ox = 150;
        hints.points[2].fx = 50;
        hints.points[2].ox = 50;
        hints.points[3].fx = 100;
        hints.points[3].ox = 100;
        hints.points[4].fx = 50;
        hints.points[4].ox = 50;
        hints.points[4].flags = FLAG_TOUCH_X;
        hints.points[4].x = 777;
        hints.points[5].fx = 50;
        hints.points[5].ox = 50;
        hints.points[5].flags = FLAG_WEAK_INTERPOLATION;
        hints.points[5].x = 888;

        hints.align_strong_points(Dimension::Hort);
        let x: alloc::vec::Vec<Pos> = hints.points.iter().map(|p| p.x).collect();
        assert_eq!(x, [-10, 178, 64, 128, 777, 888]);
        for point in hints.points.iter().take(4) {
            assert_ne!(point.flags & FLAG_TOUCH_X, 0);
        }
        assert_eq!(hints.points[4].flags & FLAG_TOUCH_X, FLAG_TOUCH_X);
        assert_eq!(hints.points[5].flags & FLAG_TOUCH_X, 0);
        let scale = hints.axis[Dimension::Hort.index()].edges[0].scale;
        assert_eq!(scale, div_fix(128, 100));
        assert_ne!(scale, 0);
    }

    #[test]
    fn align_strong_points_no_edges_is_a_noop() {
        let mut hints = GlyphHints::default();
        hints.points.resize(2, Point::default());
        hints.points[0].fx = 10;
        hints.points[0].ox = 10;
        hints.points[0].x = 42;
        hints.points[1].fx = 20;
        hints.points[1].ox = 20;
        hints.points[1].x = 43;

        hints.align_strong_points(Dimension::Hort);
        assert_eq!(hints.points[0].x, 42);
        assert_eq!(hints.points[1].x, 43);
        assert_eq!(hints.points[0].flags & FLAG_TOUCH_X, 0);
    }

    #[test]
    fn align_strong_points_binary_search_on_edge_is_exact() {
        let mut hints = GlyphHints::default();
        for fpos in (0i16..=800).step_by(100) {
            add_edge(
                &mut hints,
                Dimension::Hort,
                fpos,
                Pos::from(fpos),
                Pos::from(fpos) * 2,
            );
        }
        hints.points.resize(1, Point::default());
        hints.points[0].fx = 400;
        hints.points[0].ox = 800;

        hints.align_strong_points(Dimension::Hort);
        assert_eq!(hints.points[0].x, 800);
        assert_ne!(hints.points[0].flags & FLAG_TOUCH_X, 0);
        for edge in &hints.axis[Dimension::Hort.index()].edges {
            assert_eq!(edge.scale, 0, "exact hits must not interpolate");
        }
    }

    #[test]
    fn align_strong_points_binary_search_between_edges() {
        let mut hints = GlyphHints::default();
        for fpos in (0i16..=800).step_by(100) {
            add_edge(
                &mut hints,
                Dimension::Hort,
                fpos,
                Pos::from(fpos),
                Pos::from(fpos) * 2,
            );
        }
        hints.points.resize(1, Point::default());
        hints.points[0].fx = 350;
        hints.points[0].ox = 700;

        hints.align_strong_points(Dimension::Hort);
        assert_eq!(hints.points[0].x, 700);
        let edges = &hints.axis[Dimension::Hort.index()].edges;
        assert_eq!(edges[3].scale, div_fix(200, 100));
    }

    #[test]
    fn align_strong_points_linear_search_on_edge_is_exact() {
        let mut hints = GlyphHints::default();
        for fpos in (0i16..=700).step_by(100) {
            add_edge(
                &mut hints,
                Dimension::Hort,
                fpos,
                Pos::from(fpos),
                Pos::from(fpos),
            );
        }
        hints.points.resize(1, Point::default());
        hints.points[0].fx = 400;
        hints.points[0].ox = 400;

        hints.align_strong_points(Dimension::Hort);
        assert_eq!(hints.points[0].x, 400);
        for edge in &hints.axis[Dimension::Hort.index()].edges {
            assert_eq!(edge.scale, 0, "exact hits must not interpolate");
        }
    }

    #[test]
    fn align_weak_points_shifts_single_touched() {
        let mut hints = GlyphHints::default();
        hints.points.resize(4, Point::default());
        link_ring(&mut hints.points, 4);
        hints.contours.push(0);
        for (i, ox) in [0, 10, 20, 30].iter().enumerate() {
            hints.points[i].ox = Pos::from(*ox);
            hints.points[i].x = Pos::from(*ox);
        }
        hints.points[1].x = 15;
        hints.points[1].flags = FLAG_TOUCH_X;

        hints.align_weak_points(Dimension::Hort);
        let x: alloc::vec::Vec<Pos> = hints.points.iter().map(|p| p.x).collect();
        assert_eq!(x, [5, 15, 25, 35]);
    }

    #[test]
    fn align_weak_points_interpolates_between_two_touched() {
        let mut hints = GlyphHints::default();
        hints.points.resize(4, Point::default());
        link_ring(&mut hints.points, 4);
        hints.contours.push(0);
        for (i, ox) in [0, 10, 20, 30].iter().enumerate() {
            hints.points[i].ox = Pos::from(*ox);
            hints.points[i].x = Pos::from(*ox);
        }
        hints.points[0].x = 2;
        hints.points[0].flags = FLAG_TOUCH_X;
        hints.points[2].x = 26;
        hints.points[2].flags = FLAG_TOUCH_X;

        hints.align_weak_points(Dimension::Hort);
        let x: alloc::vec::Vec<Pos> = hints.points.iter().map(|p| p.x).collect();
        assert_eq!(x, [2, 14, 26, 36]);
    }

    #[test]
    fn align_weak_points_without_touch_keeps_coordinates() {
        let mut hints = GlyphHints::default();
        hints.points.resize(3, Point::default());
        link_ring(&mut hints.points, 3);
        hints.contours.push(0);
        for (i, ox) in [4, 8, 12].iter().enumerate() {
            hints.points[i].ox = Pos::from(*ox);
            hints.points[i].x = Pos::from(*ox) + 1;
        }

        hints.align_weak_points(Dimension::Hort);
        let x: alloc::vec::Vec<Pos> = hints.points.iter().map(|p| p.x).collect();
        assert_eq!(x, [5, 9, 13]);
    }

    #[test]
    fn iup_shift_translates_untouched_points() {
        let mut points = [Point::default(); 4];
        for (i, (v, u)) in [(0i64, 0i64), (10, 99), (20, 20), (30, 30)]
            .iter()
            .enumerate()
        {
            points[i].v = *v;
            points[i].u = *u;
        }
        iup_shift(&mut points, 0, 3, 1);
        assert_eq!(
            points
                .iter()
                .map(|p| p.u)
                .collect::<alloc::vec::Vec<_>>(),
            [89, 99, 109, 119]
        );

        let mut points = [Point::default(); 3];
        points[0].v = 5;
        points[0].u = 5;
        points[1].v = 7;
        points[1].u = 9;
        points[2].v = 6;
        points[2].u = 4;
        iup_shift(&mut points, 0, 2, 0);
        assert_eq!(
            points
                .iter()
                .map(|p| p.u)
                .collect::<alloc::vec::Vec<_>>(),
            [5, 9, 4],
            "zero delta must leave the contour untouched"
        );
    }

    #[test]
    fn iup_interp_scales_between_references() {
        let mut points = [Point::default(); 4];
        points[0].v = 0;
        points[0].u = 2;
        points[1].v = 10;
        points[1].u = 0;
        points[2].v = 20;
        points[2].u = 26;
        points[3].v = 30;
        points[3].u = 0;

        iup_interp(&mut points, 1, 1, 0, 2);
        assert_eq!(points[1].u, 14);
        iup_interp(&mut points, 3, 3, 0, 2);
        assert_eq!(points[3].u, 36);
        iup_interp(&mut points, 1, 1, 2, 0);
        assert_eq!(points[1].u, 14, "reference order must not matter");
        iup_interp(&mut points, 3, 1, 0, 2);
        assert_eq!(points[3].u, 36, "empty range must be a no-op");
    }

    #[test]
    fn iup_interp_equal_reference_positions() {
        let mut points = [Point::default(); 4];
        points[0].v = 10;
        points[0].u = 2;
        points[1].v = 5;
        points[2].v = 15;
        points[3].v = 10;
        points[3].u = 12;

        iup_interp(&mut points, 1, 2, 0, 3);
        assert_eq!(points[1].u, -3);
        assert_eq!(points[2].u, 17);
    }

    #[test]
    fn iup_interp_equal_hints_pulls_to_common_value() {
        let mut points = [Point::default(); 3];
        points[0].v = 0;
        points[0].u = 7;
        points[1].v = 10;
        points[2].v = 20;
        points[2].u = 7;

        iup_interp(&mut points, 1, 1, 0, 2);
        assert_eq!(points[1].u, 7);
    }

    #[test]
    fn scale_dim_applies_warp_scale_and_delta() {
        let mut hints = GlyphHints::default();
        hints.points.resize(3, Point::default());
        hints.points[0].fx = 1;
        hints.points[0].fy = 4;
        hints.points[1].fx = 2;
        hints.points[1].fy = 5;
        hints.points[2].fx = 3;
        hints.points[2].fy = 6;

        hints.scale_dim(Dimension::Hort, 2 << 16, 5);
        assert_eq!(hints.points[0].x, 7);
        assert_eq!(hints.points[1].x, 9);
        assert_eq!(hints.points[2].x, 11);

        hints.scale_dim(Dimension::Vert, 1 << 16, -2);
        assert_eq!(hints.points[0].y, 2);
        assert_eq!(hints.points[1].y, 3);
        assert_eq!(hints.points[2].y, 4);
    }

    #[test]
    fn rescale_done_and_scaler_flags() {
        let mut hints = GlyphHints::default();
        assert!(hints.do_horizontal());
        assert!(hints.do_vertical());
        assert!(hints.do_advance());
        assert!(hints.do_warp());

        hints.scaler_flags = SCALER_FLAG_NO_VERTICAL | SCALER_FLAG_NO_ADVANCE;
        assert!(hints.do_horizontal());
        assert!(!hints.do_vertical());
        assert!(!hints.do_advance());
        assert!(hints.do_warp());

        hints.other_flags = 4;
        assert!(hints.test_other(4));
        assert!(!hints.test_other(1));

        let globals = Rc::new(GlobalsShared::new(2048));
        let mut metrics = StyleMetrics::new(&STYLE_CLASSES[Style::LatnDflt.index()], globals);
        metrics.scaler_mut().flags = SCALER_FLAG_NO_HORIZONTAL;
        let mut hints = GlyphHints::default();
        hints.rescale(&metrics);
        assert_eq!(hints.scaler_flags, SCALER_FLAG_NO_HORIZONTAL);
        assert_eq!(hints.units_per_em, 2048);
        assert!(hints.metrics.is_some());
        assert!(!hints.do_horizontal());
        assert!(hints.do_vertical());

        hints.points.resize(1, Point::default());
        hints.contours.push(0);
        hints.done();
        assert!(hints.points.is_empty());
        assert!(hints.contours.is_empty());
        assert!(hints.metrics.is_none());
        assert_eq!(hints.scaler_flags, 0);
        assert_eq!(hints.other_flags, 0);
        assert_eq!(hints.units_per_em, 0);
        assert_eq!(hints.x_scale, 0);
    }

    #[test]
    fn msb_shift_matches_ft_msb_formula() {
        assert_eq!(msb_shift(0), 0);
        assert_eq!(msb_shift(1), 0);
        assert_eq!(msb_shift(1 << 14), 0);
        assert_eq!(msb_shift(1 << 20), 6);
        assert_eq!(msb_shift(u32::MAX), 17);
    }

    #[test]
    fn delta_index_clamps_out_of_range() {
        assert_eq!(delta_index(4, 1, 2), 3);
        assert_eq!(delta_index(4, 1, -1), 0);
        assert_eq!(delta_index(4, 1, -5), 0);
        assert_eq!(delta_index(4, 1, 10), 3);
        assert_eq!(delta_index(4, 3, 1), 3);
    }

    #[test]
    fn quadrant_pass_tags_same_quadrant_vectors() {
        let mut hints = GlyphHints::default();
        hints.points.resize(3, Point::default());
        hints.points[0].fx = 0;
        hints.points[0].fy = 0;
        hints.points[1].fx = 10;
        hints.points[1].fy = 10;
        hints.points[1].u = 1;
        hints.points[1].v = -1;
        hints.points[2].fx = 30;
        hints.points[2].fy = 40;

        hints.tag_quadrant_weak_points();
        assert_ne!(hints.points[1].flags & FLAG_WEAK_INTERPOLATION, 0);
        assert_eq!(hints.points[0].u, 2);
        assert_eq!(hints.points[2].v, -2);

        let mut hints = GlyphHints::default();
        hints.points.resize(3, Point::default());
        hints.points[1].fx = 10;
        hints.points[1].fy = 10;
        hints.points[1].u = 1;
        hints.points[1].v = -1;
        hints.points[2].fx = -10;
        hints.points[2].fy = 40;

        hints.tag_quadrant_weak_points();
        assert_eq!(hints.points[1].flags & FLAG_WEAK_INTERPOLATION, 0);
    }

    #[test]
    fn dominant_pass_classifies_control_segment_and_spike() {
        let mut hints = GlyphHints::default();
        hints.points.resize(3, Point::default());
        hints.points[0].flags = FLAG_CONIC;
        hints.points[0].in_dir = Direction::Right;
        hints.points[0].out_dir = Direction::Up;
        hints.points[1].in_dir = Direction::Up;
        hints.points[1].out_dir = Direction::Up;
        hints.points[2].in_dir = Direction::Right;
        hints.points[2].out_dir = Direction::Left;

        hints.tag_dominant_weak_points();
        for point in &hints.points {
            assert_ne!(point.flags & FLAG_WEAK_INTERPOLATION, 0);
        }
    }

    #[test]
    fn dominant_pass_tags_flat_corners_only() {
        let mut hints = GlyphHints::default();
        hints.points.resize(3, Point::default());
        hints.points[0].fx = 0;
        hints.points[0].fy = 0;
        hints.points[1].fx = 100;
        hints.points[1].fy = 100;
        hints.points[1].u = 1;
        hints.points[1].v = -1;
        hints.points[2].fx = 200;
        hints.points[2].fy = 200;

        hints.tag_dominant_weak_points();
        assert_ne!(hints.points[1].flags & FLAG_WEAK_INTERPOLATION, 0);
        assert_eq!(hints.points[0].u, 2);
        assert_eq!(hints.points[2].v, -2);

        let mut hints = GlyphHints::default();
        hints.points.resize(3, Point::default());
        hints.points[0].fx = 0;
        hints.points[0].fy = 0;
        hints.points[1].fx = 100;
        hints.points[1].fy = 100;
        hints.points[1].u = 1;
        hints.points[1].v = -1;
        hints.points[2].fx = 10;
        hints.points[2].fy = 0;

        hints.tag_dominant_weak_points();
        assert_eq!(
            hints.points[1].flags & FLAG_WEAK_INTERPOLATION,
            0,
            "a sharp corner stays strong"
        );
    }

    #[test]
    fn adjust_contour_start_walks_back_over_near_points() {
        let mut points = [Point::default(); 4];
        for (i, fx) in [0i16, 2, 4, 6].iter().enumerate() {
            points[i].fx = *fx;
        }
        link_ring(&mut points, 4);

        assert_eq!(adjust_contour_start(&points, 0, 17), 1);
        assert_eq!(adjust_contour_start(&points, 1, 17), 2);

        points[3].fx = 100;
        assert_eq!(adjust_contour_start(&points, 0, 17), 0);
    }
}
