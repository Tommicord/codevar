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
    corner_is_flat, div_fix, mul_fix, Orientation, Outline, Pos, Fixed, CURVE_TAG_CONIC,
    CURVE_TAG_CUBIC, CURVE_TAG_ON,
};

use crate::metrics::StyleMetrics;
use crate::{
    direction_compute, Dimension, Direction, DIMENSION_MAX, SCALER_FLAG_NO_ADVANCE,
    SCALER_FLAG_NO_HORIZONTAL, SCALER_FLAG_NO_VERTICAL, SCALER_FLAG_NO_WARPER,
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
        self.contours.resize(outline.n_contours as usize, 0);
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

            if self.points[i].in_dir == Direction::None
                && self.points[i].out_dir == Direction::None
            {
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
            } else if i32::from(self.points[i].in_dir.code())
                == -i32::from(self.points[i].out_dir.code())
            {
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
        let limit = self.points.len().min(outline.points.len());
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
            let (Some(edge_index), Some(first), Some(last)) = (seg.edge, seg.first, seg.last)
            else {
                continue;
            };
            if edge_index >= self.axis[dim.index()].edges.len() || first >= num_points {
                continue;
            }
            let pos = self.axis[dim.index()].edges[edge_index].pos;

            let mut point = first;
            let mut steps = 0usize;
            loop {
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
                StrongLookup::Between { before, after } => {
                    self.interpolate_strong(dim, before, after, fu)
                }
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
    let end_point = points[start].prev;
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
        iup_interp(points, first_point, first_touched - 1, last_touched, first_touched);
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
