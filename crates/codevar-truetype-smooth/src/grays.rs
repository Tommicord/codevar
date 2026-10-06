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

//! Antialiasing scan converter
//!
//! the "perfect" antialiasing renderer.  It computes the exact coverage of
//! the outline on each pixel cell: [`outline_decompose`] walks every
//! contour once, accumulating `(area, cover)` in a current [`Cell`],
//! [`GrayWorker`] records cells in a per-band index (`ycells` + linked
//! `cells`), and [`GrayWorker::sweep`] converts the accumulated areas into
//! [`Span`]s which are either written into a target [`Bitmap`] (indirect
//! mode) or handed to a callback ([`RasterFlags::DIRECT`]).
//!
//! Notes:
//!
//! * Cell/table exhaustion surfaces as
//!   [`TtError::OUT_OF_MEMORY`] so the band split ("ReduceBands") retry is
//!   preserved; stack/table index failures surface as
//!   [`TtError::RASTER_OVERFLOW`] or [`TtError::RASTER_CORRUPTED`].
//! * A rotten band (`middle == bottom`) reports [`TtError::RASTER_OVERFLOW`]
//!   and other decompose failures propagate their specific error instead of
//!   C's generic `return 1`.
//! * Direct mode invokes the callback only when both the function and the
//!   user pointer are present.
//! * Bitmap mode skips spans that fall outside the buffer instead of writing
//!   out of bounds; with the bitmap clip box this branch is unreachable.
//! * `FT_Outline_Decompose`'s `setjmp`/`longjmp` cell-exhaustion escape is
//!   expressed as a [`TtError::OUT_OF_MEMORY`] return value.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use codevar_truetype_core::{
    BBox, Bitmap, GlyphFormat, Memory, OUTLINE_EVEN_ODD_FILL, Outline, Raster, RasterFlags, RasterFuncs,
    RasterParams, Span, SpanFunc, TtError, TtResult, Vector, abs_pos,
};

/// `PIXEL_BITS`: subpixel resolution of the scan converter (must be at
/// least 6).
const PIXEL_BITS: u32 = 8;
/// `ONE_PIXEL`: one pixel in subpixel units (`1 << PIXEL_BITS`).
const ONE_PIXEL: i64 = 1 << PIXEL_BITS;
/// `UPSCALE`: `PIXEL_BITS - 6`, the shift from 26.6 to subpixel units.
const UPSCALE_SHIFT: u32 = PIXEL_BITS - 6;
/// `FT_MAX_GRAY_SPANS`: maximum spans per callback invocation.
const FT_MAX_GRAY_SPANS: usize = 32;
/// `FT_RENDER_POOL_SIZE`: size of the render pool in bytes.
const FT_RENDER_POOL_SIZE: usize = 16384;
/// `sizeof(TCell)`: cell record size in bytes (8 + 8 + 4 + 4 pad + 8).
const TCELL_SIZE: usize = 32;
/// `sizeof(PCell)`: cell pointer size in bytes (64-bit).
const PCELL_SIZE: usize = 8;
/// Number of preallocated cells (`FT_RENDER_POOL_SIZE / sizeof(TCell)`).
const NUM_CELLS: usize = FT_RENDER_POOL_SIZE / TCELL_SIZE;
/// Initial band height in scanlines
/// (`FT_RENDER_POOL_SIZE / (sizeof(TCell) * 8)`).
const BAND_SIZE: i32 = (FT_RENDER_POOL_SIZE / (TCELL_SIZE * 8)) as i32;

/// `FT_DIV_MOD(TCoord, ...)` (`ftgrays.c`): computes `dividend / divisor`
/// with the remainder forced positive, returning `(quotient, remainder)`.
///
/// # Errors
///
/// [`TtError::RASTER_OVERFLOW`] when the division is undefined (zero or
/// overflowing divisor), replacing C's undefined behavior.
#[inline]
fn div_mod_tcoord(dividend: i64, divisor: i64) -> TtResult<(i64, i64)> {
    let mut quotient = dividend
        .checked_div(divisor)
        .ok_or(TtError::RASTER_OVERFLOW)?;
    let mut remainder = dividend
        .checked_rem(divisor)
        .ok_or(TtError::RASTER_OVERFLOW)?;
    if remainder < 0 {
        quotient = quotient.wrapping_sub(1);
        remainder = remainder.wrapping_add(divisor);
    }
    Ok((quotient, remainder))
}

/// `FT_DIV_MOD(int, ...)` where both outputs are `int`
/// (`gray_render_line`'s `lift`/`rem` pair): the quotient and remainder
/// are truncated to 32 bits and the correction is applied in `int`
/// arithmetic.
///
/// # Errors
///
/// [`TtError::RASTER_OVERFLOW`] when the division is undefined.
#[inline]
fn div_mod_int_pair(dividend: i64, divisor: i64) -> TtResult<(i32, i32)> {
    let mut quotient = dividend
        .checked_div(divisor)
        .ok_or(TtError::RASTER_OVERFLOW)? as i32;
    let mut remainder = dividend
        .checked_rem(divisor)
        .ok_or(TtError::RASTER_OVERFLOW)? as i32;
    if remainder < 0 {
        quotient = quotient.wrapping_sub(1);
        remainder = remainder.wrapping_add(divisor as i32);
    }
    Ok((quotient, remainder))
}

/// `FT_DIV_MOD(int, ...)` where the quotient is an `int` but the
/// remainder lives in a `TCoord` (`long`) variable (`gray_render_line`'s
/// `delta`/`mod` pair): both intermediate values are truncated to 32 bits,
/// and the correction adds the divisor in `long` arithmetic.
///
/// # Errors
///
/// [`TtError::RASTER_OVERFLOW`] when the division is undefined.
#[inline]
fn div_mod_int_long(dividend: i64, divisor: i64) -> TtResult<(i32, i64)> {
    let truncated_div = dividend
        .checked_div(divisor)
        .ok_or(TtError::RASTER_OVERFLOW)?;
    let truncated_rem = dividend
        .checked_rem(divisor)
        .ok_or(TtError::RASTER_OVERFLOW)?;
    let mut quotient = truncated_div as i32;
    let mut remainder = truncated_rem as i32 as i64;
    if remainder < 0 {
        quotient = quotient.wrapping_sub(1);
        remainder = remainder.wrapping_add(divisor as i32 as i64);
    }
    Ok((quotient, remainder))
}

/// `FT_HYPOT(x, y)` from `ftgrays.c`: approximate `sqrt(x*x + y*y)` with
/// the alpha-max-plus-beta-min algorithm (alpha = 1, beta = 3/8, error
/// < 7%), used to decide cubic subdivision.
#[inline]
fn ft_hypot(x: i64, y: i64) -> i64 {
    let x = abs_pos(x);
    let y = abs_pos(y);
    if x > y {
        x.wrapping_add((y.wrapping_mul(3)) >> 3)
    } else {
        y.wrapping_add((x.wrapping_mul(3)) >> 3)
    }
}

/// One accumulated pixel cell (`TCell` of `ftgrays.c`): horizontal
/// position `x` (relative to `min_ex`), signed `cover` (line crossings),
/// and the doubled `area` accumulated between cell entries.
#[derive(Clone, Copy, Debug)]
struct Cell {
    /// Band-relative horizontal cell position (`gray_TWorker.ex`).
    x: i64,
    /// Accumulated line coverage (`gray_TWorker.cover`).
    cover: i64,
    /// Accumulated (doubled) cell area (`gray_TWorker.area`).
    area: i32,
    /// Next cell in the per-row linked list, or `-1` for none.
    next: i32,
}

impl Default for Cell {
    fn default() -> Self {
        Cell {
            x: 0,
            cover: 0,
            area: 0,
            next: -1,
        }
    }
}

/// A vertical band of scanlines processed by one conversion pass
/// (`gray_TBand`).
#[derive(Clone, Copy, Debug)]
struct Band {
    /// First scanline of the band (inclusive).
    min: i64,
    /// Scanline after the band (exclusive).
    max: i64,
}

/// Destination of the generated spans: either a target bitmap (indirect
/// mode) or a caller callback (direct mode).
enum Sink<'a> {
    /// Indirect mode: spans are written into the bitmap.
    Bitmap {
        /// Target bitmap, written bottom-up according to its pitch.
        bitmap: &'a mut Bitmap,
    },
    /// Direct mode: spans are delivered to the callback.
    Direct {
        /// `FT_Raster_Params.gray_spans`.
        func: Option<SpanFunc>,
        /// `FT_Raster_Params.user`.
        user: Option<&'a mut dyn core::any::Any>,
    },
}

impl Sink<'_> {
    /// Delivers one batch of `count` spans on scanline `y` to this sink.
    #[inline]
    fn emit(&mut self, y: i32, count: usize, spans: &[Span]) {
        match self {
            Sink::Bitmap { bitmap } => emit_bitmap(bitmap, y, spans),
            Sink::Direct { func, user } => {
                if let (Some(f), Some(u)) = (*func, user.as_deref_mut()) {
                    f(y, count, spans, u);
                }
            }
        }
    }
}

/// Writes `spans` of the logical bottom-up scanline `y` into `bitmap`,
/// following `gray_render_span`'s address arithmetic: for a positive
/// pitch row `y` lives at memory row `rows - 1 - y`, for a negative pitch
/// at memory row `y`, which is exactly what `Bitmap::row_mut` performs
/// for the index `rows - 1 - y`.
///
/// Spans reaching outside the buffer are skipped; with the bitmap clip
/// box this defensive branch is unreachable.
fn emit_bitmap(bitmap: &mut Bitmap, y: i32, spans: &[Span]) {
    let index = i64::from(bitmap.rows)
        .wrapping_sub(1)
        .wrapping_sub(i64::from(y));
    let Ok(index) = u32::try_from(index) else {
        return;
    };
    let Some(row) = bitmap.row_mut(index) else {
        return;
    };
    for span in spans {
        if span.coverage == 0 {
            continue;
        }
        let start = i32::from(span.x);
        let end = start.wrapping_add(i32::from(span.len));
        if start < 0 || end < start {
            continue;
        }
        let (Ok(s), Ok(e)) = (usize::try_from(start), usize::try_from(end)) else {
            continue;
        };
        if e > row.len() {
            continue;
        }
        row[s..e].fill(span.coverage);
    }
}

/// Splits of a quadratic Bézier arc of
/// five `Vector`s into two halves, x block followed by y block
///
/// # Panics
///
/// `base` must hold at least 5 elements; the only caller passes the exact
/// range `arc..arc + 5`.
fn split_conic(base: &mut [Vector]) {
    let old2x = base[2].x;
    let b = base[1].x;
    let a = old2x.wrapping_add(b) / 2;
    base[3].x = a;
    let b = base[0].x.wrapping_add(b) / 2;
    base[1].x = b;
    base[2].x = a.wrapping_add(b) / 2;
    base[4].x = old2x;

    let old2y = base[2].y;
    let b = base[1].y;
    let a = old2y.wrapping_add(b) / 2;
    base[3].y = a;
    let b = base[0].y.wrapping_add(b) / 2;
    base[1].y = b;
    base[2].y = a.wrapping_add(b) / 2;
    base[4].y = old2y;
}

/// Splits of a cubic Bézier arc of seven `Vector`s into two halves,
/// x block followed by y block
///
/// # Panics
///
/// `base` must hold at least 7 elements; the only caller passes the exact
/// range `arc..arc + 7`.
fn split_cubic(base: &mut [Vector]) {
    base[6].x = base[3].x;
    let c = base[1].x;
    let d = base[2].x;
    let a = base[0].x.wrapping_add(c) / 2;
    base[1].x = a;
    let b = base[3].x.wrapping_add(d) / 2;
    base[5].x = b;
    let c = c.wrapping_add(d) / 2;
    let a = a.wrapping_add(c) / 2;
    base[2].x = a;
    let b = b.wrapping_add(c) / 2;
    base[4].x = b;
    base[3].x = a.wrapping_add(b) / 2;

    base[6].y = base[3].y;
    let c = base[1].y;
    let d = base[2].y;
    let a = base[0].y.wrapping_add(c) / 2;
    base[1].y = a;
    let b = base[3].y.wrapping_add(d) / 2;
    base[5].y = b;
    let c = c.wrapping_add(d) / 2;
    let a = a.wrapping_add(c) / 2;
    base[2].y = a;
    let b = b.wrapping_add(c) / 2;
    base[4].y = b;
    base[3].y = a.wrapping_add(b) / 2;
}

/// Halves the band on top of `stack` after a render pool
/// overflow, pushing the lower half for processing first and leaving the
/// upper half beneath it.
///
/// Returns the `band_shoot` contribution (`bottom - top >= band_size`,
/// which for a well-formed band is never true — kept verbatim for
/// byte-exactness).
///
/// # Errors
///
/// * [`TtError::RASTER_CORRUPTED`] when the stack is unexpectedly empty.
/// * [`TtError::RASTER_OVERFLOW`] for a "rotten" band that cannot be
///   split further (`middle == bottom`).
fn reduce_band(stack: &mut Vec<Band>, band_size: i32) -> TtResult<bool> {
    let band = *stack.last().ok_or(TtError::RASTER_CORRUPTED)?;
    let bottom = band.min;
    let top = band.max;
    let middle = bottom.wrapping_add((top.wrapping_sub(bottom)) >> 1);
    if middle == bottom {
        return Err(TtError::RASTER_OVERFLOW);
    }
    let shoot = bottom.wrapping_sub(top) >= i64::from(band_size);
    let upper = Band {
        min: middle,
        max: top,
    };
    let lower = Band {
        min: bottom,
        max: middle,
    };
    *stack
        .last_mut()
        .ok_or(TtError::RASTER_CORRUPTED)? = upper;
    stack.push(lower);
    Ok(shoot)
}

/// The `gray_TWorker` of `ftgrays.c`: all state of one glyph conversion
/// (cell accumulation, Bézier flattening, band bookkeeping and span
/// emission).
struct GrayWorker<'a, 'b> {
    /// Source outline being decomposed.
    outline: &'a Outline,
    /// Destination of the generated spans.
    sink: Sink<'b>,
    /// Preallocated cell pool (`NUM_CELLS` entries).
    cells: &'b mut Vec<Cell>,
    /// Per-scanline head indices into `cells` (`-1` = empty).
    ycells: &'b mut Vec<i32>,
    /// Clip box for this render.
    clip_box: BBox,
    /// Current cell x (relative to `min_ex`).
    ex: i64,
    /// Current cell y (relative to `min_ey`).
    ey: i64,
    /// Bounding box minimum x, clipped (26.6 truncated to pixels).
    min_ex: i64,
    /// Bounding box maximum x, clipped.
    max_ex: i64,
    /// Bounding box minimum y, clipped.
    min_ey: i64,
    /// Bounding box maximum y, clipped.
    max_ey: i64,
    /// Number of cell columns in the band.
    count_ex: i64,
    /// Number of cell rows in the band.
    count_ey: i64,
    /// Accumulated area of the current cell (doubled).
    area: i32,
    /// Accumulated coverage of the current cell.
    cover: i64,
    /// `true` when the current cell lies outside the band/clip region.
    invalid: bool,
    /// Current subpixel position x.
    x: i64,
    /// Current subpixel position y.
    y: i64,
    /// Scanline of the previous endpoint (`SUBPIXELS(ey)`).
    last_ey: i64,
    /// Bézier flattening stack (`32 * 3 + 1` entries).
    bez_stack: [Vector; 97],
    /// Subdivision level stack for conic flattening.
    lev_stack: [i32; 32],
    /// Pending span batch.
    gray_spans: [Span; 32],
    /// Number of pending spans.
    num_gray_spans: i32,
    /// Scanline the pending spans belong to.
    span_y: i32,
    /// Current band height in scanlines (starts at [`BAND_SIZE`]).
    band_size: i32,
    /// Count of oversized bands, used to shrink `band_size`.
    band_shoot: i32,
    /// Number of cells used so far in this band.
    num_cells: usize,
    /// Cell capacity for this band (render pool layout).
    max_cells: usize,
    /// Number of scanlines in the current band.
    ycount: i64,
}

impl<'a, 'b> GrayWorker<'a, 'b> {
    /// Creates a worker over the reusable cell buffers of [`GrayRaster`].
    fn new(
        outline: &'a Outline,
        sink: Sink<'b>,
        cells: &'b mut Vec<Cell>,
        ycells: &'b mut Vec<i32>,
        clip_box: BBox,
    ) -> Self {
        cells.clear();
        cells.resize(NUM_CELLS, Cell::default());
        ycells.clear();
        GrayWorker {
            outline,
            sink,
            cells,
            ycells,
            clip_box,
            ex: 0,
            ey: 0,
            min_ex: 0,
            max_ex: 0,
            min_ey: 0,
            max_ey: 0,
            count_ex: 0,
            count_ey: 0,
            area: 0,
            cover: 0,
            invalid: true,
            x: 0,
            y: 0,
            last_ey: 0,
            bez_stack: [Vector::ZERO; 97],
            lev_stack: [0; 32],
            gray_spans: [Span::default(); 32],
            num_gray_spans: 0,
            span_y: 0,
            band_size: BAND_SIZE,
            band_shoot: 0,
            num_cells: 0,
            max_cells: 0,
            ycount: 0,
        }
    }

    /// Bounding box of the outline's points,
    /// truncated to integer pixels (`>> 6` and `(max + 63) >> 6`).
    ///
    /// Outlines with `n_points <= 0` produce a zero box, as in C.
    fn compute_cbox(&mut self) {
        if self.outline.n_points <= 0 {
            self.min_ex = 0;
            self.max_ex = 0;
            self.min_ey = 0;
            self.max_ey = 0;
            return;
        }
        let n = self.outline.n_points as usize;
        let points = &self.outline.points;
        let Some(first) = points.first() else {
            self.min_ex = 0;
            self.max_ex = 0;
            self.min_ey = 0;
            self.max_ey = 0;
            return;
        };
        let mut min_x = first.x;
        let mut max_x = first.x;
        let mut min_y = first.y;
        let mut max_y = first.y;
        for p in points.iter().take(n).skip(1) {
            if p.x < min_x {
                min_x = p.x;
            }
            if p.x > max_x {
                max_x = p.x;
            }
            if p.y < min_y {
                min_y = p.y;
            }
            if p.y > max_y {
                max_y = p.y;
            }
        }
        self.min_ex = min_x >> 6;
        self.min_ey = min_y >> 6;
        self.max_ex = max_x.wrapping_add(63) >> 6;
        self.max_ey = max_y.wrapping_add(63) >> 6;
    }

    /// Locates (or appends) the cell for
    /// `(self.ex, self.ey)` in the row list, clamping `x` to `count_ex`
    /// and walking the `ycells` chain sorted by `x`.
    ///
    /// # Errors
    ///
    /// * [`TtError::OUT_OF_MEMORY`] when the cell pool for this band is
    ///   exhausted (C's `ft_longjmp`), triggering the band split retry.
    /// * [`TtError::RASTER_CORRUPTED`] on an out-of-range list index.
    fn find_cell(&mut self) -> TtResult<usize> {
        let x = if self.ex > self.count_ex {
            self.count_ex
        } else {
            self.ex
        };
        let row = self.ey as usize;
        let mut prev: Option<usize> = None;
        let mut curr = *self
            .ycells
            .get(row)
            .ok_or(TtError::RASTER_CORRUPTED)?;
        loop {
            if curr < 0 {
                break;
            }
            let cell = *self
                .cells
                .get(curr as usize)
                .ok_or(TtError::RASTER_CORRUPTED)?;
            if cell.x > x {
                break;
            }
            if cell.x == x {
                return Ok(curr as usize);
            }
            prev = Some(curr as usize);
            curr = cell.next;
        }
        if self.num_cells >= self.max_cells {
            return Err(TtError::OUT_OF_MEMORY);
        }
        let idx = self.num_cells;
        self.num_cells += 1;
        {
            let cell = self
                .cells
                .get_mut(idx)
                .ok_or(TtError::RASTER_CORRUPTED)?;
            cell.x = x;
            cell.area = 0;
            cell.cover = 0;
            cell.next = curr;
        }
        match prev {
            Some(p) => {
                let link = self
                    .cells
                    .get_mut(p)
                    .ok_or(TtError::RASTER_CORRUPTED)?;
                link.next = idx as i32;
            }
            None => {
                let head = self
                    .ycells
                    .get_mut(row)
                    .ok_or(TtError::RASTER_CORRUPTED)?;
                *head = idx as i32;
            }
        }
        Ok(idx)
    }

    /// Flushes the accumulated `area`/`cover` into
    /// the current cell, if anything was accumulated.
    ///
    /// # Errors
    ///
    /// Propagates [`gray_find_cell`]'s errors.
    fn record_cell(&mut self) -> TtResult<()> {
        if (self.area as i64 | self.cover) != 0 {
            let idx = self.find_cell()?;
            let cell = self
                .cells
                .get_mut(idx)
                .ok_or(TtError::RASTER_CORRUPTED)?;
            cell.area = cell.area.wrapping_add(self.area);
            cell.cover = cell.cover.wrapping_add(self.cover);
        }
        Ok(())
    }

    /// Moves the accumulation to cell `(ex, ey)`,
    /// recording the previous cell first when it changed and was valid,
    /// then recomputes the `invalid` flag (vertical position outside the
    /// band, or horizontal position at/after `count_ex`; cells left of the
    /// clip live at `x = -1` and stay valid).
    ///
    /// # Errors
    ///
    /// Propagates [`gray_record_cell`]'s errors.
    #[inline]
    fn set_cell(&mut self, ex: i64, ey: i64) -> TtResult<()> {
        let mut ex = ex;
        let mut ey = ey;
        ey = ey.wrapping_sub(self.min_ey);
        if ex > self.max_ex {
            ex = self.max_ex;
        }
        ex = ex.wrapping_sub(self.min_ex);
        if ex < 0 {
            ex = -1;
        }
        if ex != self.ex || ey != self.ey {
            if !self.invalid {
                self.record_cell()?;
            }
            self.area = 0;
            self.cover = 0;
            self.ex = ex;
            self.ey = ey;
        }
        self.invalid = (ey as u32) >= (self.count_ey as u32) || ex >= self.count_ex;
        Ok(())
    }

    /// Begins a new contour at cell `(ex, ey)`,
    /// clamping into the clip region (left overflow parks at
    /// `min_ex - 1`) and priming `ex`/`ey` so the following
    /// [`GrayWorker::set_cell`] sees no transition.
    ///
    /// # Errors
    ///
    /// Propagates [`gray_set_cell`]'s errors.
    #[inline]
    fn start_cell(&mut self, mut ex: i64, ey: i64) -> TtResult<()> {
        if ex > self.max_ex {
            ex = self.max_ex;
        }
        if ex < self.min_ex {
            ex = self.min_ex.wrapping_sub(1);
        }
        self.area = 0;
        self.cover = 0;
        self.ex = ex.wrapping_sub(self.min_ex);
        self.ey = ey.wrapping_sub(self.min_ey);
        self.last_ey = ey.wrapping_shl(PIXEL_BITS);
        self.invalid = false;
        self.set_cell(ex, ey)
    }

    /// Renders the segment `(x1, y1) -> (x2, y2)`
    /// of scanline `ey` as one or more cells, splitting it at pixel
    /// boundaries with the `FT_DIV_MOD` Bresenham step.
    ///
    /// # Errors
    ///
    /// * [`TtError::RASTER_OVERFLOW`] when a division is undefined.
    /// * Propagates [`gray_set_cell`]'s errors.
    fn render_scanline(&mut self, ey: i64, x1: i64, y1: i64, x2: i64, y2: i64) -> TtResult<()> {
        let mut dx = x2.wrapping_sub(x1);
        let ex1 = x1 >> PIXEL_BITS;
        let ex2 = x2 >> PIXEL_BITS;
        let fx1 = x1.wrapping_sub(ex1.wrapping_shl(PIXEL_BITS));
        let fx2 = x2.wrapping_sub(ex2.wrapping_shl(PIXEL_BITS));
        if y1 == y2 {
            return self.set_cell(ex2, ey);
        }
        if ex1 == ex2 {
            let delta = y2.wrapping_sub(y1);
            self.area = self
                .area
                .wrapping_add(fx1.wrapping_add(fx2).wrapping_mul(delta) as i32);
            self.cover = self.cover.wrapping_add(delta);
            return Ok(());
        }
        let mut p = ONE_PIXEL
            .wrapping_sub(fx1)
            .wrapping_mul(y2.wrapping_sub(y1));
        let mut first = ONE_PIXEL;
        let mut incr: i32 = 1;
        if dx < 0 {
            p = fx1.wrapping_mul(y2.wrapping_sub(y1));
            first = 0;
            incr = -1;
            dx = dx.wrapping_neg();
        }
        let (mut delta, mut modulo) = div_mod_tcoord(p, dx)?;
        self.area = self
            .area
            .wrapping_add(fx1.wrapping_add(first).wrapping_mul(delta) as i32);
        self.cover = self.cover.wrapping_add(delta);
        let mut ex1 = ex1.wrapping_add(i64::from(incr));
        self.set_cell(ex1, ey)?;
        let mut y1 = y1.wrapping_add(delta);
        if ex1 != ex2 {
            p = ONE_PIXEL.wrapping_mul(y2.wrapping_sub(y1).wrapping_add(delta));
            let (lift, rem) = div_mod_tcoord(p, dx)?;
            modulo = modulo.wrapping_sub(dx as i32 as i64);
            while ex1 != ex2 {
                delta = lift;
                modulo = modulo.wrapping_add(rem);
                if modulo >= 0 {
                    modulo = modulo.wrapping_sub(dx);
                    delta = delta.wrapping_add(1);
                }
                self.area = self
                    .area
                    .wrapping_add(ONE_PIXEL.wrapping_mul(delta) as i32);
                self.cover = self.cover.wrapping_add(delta);
                y1 = y1.wrapping_add(delta);
                ex1 = ex1.wrapping_add(i64::from(incr));
                self.set_cell(ex1, ey)?;
            }
        }
        let delta = y2.wrapping_sub(y1);
        self.area = self.area.wrapping_add(
            fx2.wrapping_add(ONE_PIXEL)
                .wrapping_sub(first)
                .wrapping_mul(delta) as i32,
        );
        self.cover = self.cover.wrapping_add(delta);
        Ok(())
    }

    /// Renders a straight segment from the current
    /// position to `(to_x, to_y)` (subpixel units) as a run of scanlines,
    /// with the vertical clip, the `dx == 0` vertical fast path, and the
    /// Bresenham stepping of the general case.  Always advances the
    /// current position on exit, even when clipped.
    ///
    /// # Errors
    ///
    /// * [`TtError::RASTER_OVERFLOW`] when a division is undefined.
    fn render_line(&mut self, to_x: i64, to_y: i64) -> TtResult<()> {
        let mut ey1 = self.last_ey >> PIXEL_BITS;
        let ey2 = to_y >> PIXEL_BITS;
        let fy1 = self.y.wrapping_sub(self.last_ey);
        let fy2 = to_y.wrapping_sub(ey2.wrapping_shl(PIXEL_BITS));
        let dx = to_x.wrapping_sub(self.x);
        let mut dy = to_y.wrapping_sub(self.y);
        let (min, max) = if ey1 > ey2 { (ey2, ey1) } else { (ey1, ey2) };
        if min >= self.max_ey || max < self.min_ey {
            self.finish_line(to_x, to_y, ey2);
            return Ok(());
        }
        if ey1 == ey2 {
            self.render_scanline(ey1, self.x, fy1, to_x, fy2)?;
            self.finish_line(to_x, to_y, ey2);
            return Ok(());
        }
        let mut incr: i32 = 1;
        if dx == 0 {
            let ex = self.x >> PIXEL_BITS;
            let two_fx = self
                .x
                .wrapping_sub(ex.wrapping_shl(PIXEL_BITS))
                .wrapping_shl(1);
            let mut first = ONE_PIXEL;
            if dy < 0 {
                first = 0;
                incr = -1;
            }
            let delta = first.wrapping_sub(fy1) as i32;
            self.area = self
                .area
                .wrapping_add((two_fx as i32).wrapping_mul(delta));
            self.cover = self.cover.wrapping_add(i64::from(delta));
            ey1 = ey1.wrapping_add(i64::from(incr));
            self.set_cell(ex, ey1)?;
            let delta = first.wrapping_add(first).wrapping_sub(ONE_PIXEL) as i32;
            let area_val = (two_fx as i32).wrapping_mul(delta);
            while ey1 != ey2 {
                self.area = self.area.wrapping_add(area_val);
                self.cover = self.cover.wrapping_add(i64::from(delta));
                ey1 = ey1.wrapping_add(i64::from(incr));
                self.set_cell(ex, ey1)?;
            }
            let delta = fy2.wrapping_sub(ONE_PIXEL).wrapping_add(first) as i32;
            self.area = self
                .area
                .wrapping_add((two_fx as i32).wrapping_mul(delta));
            self.cover = self.cover.wrapping_add(i64::from(delta));
            self.finish_line(to_x, to_y, ey2);
            return Ok(());
        }
        let mut p = ONE_PIXEL.wrapping_sub(fy1).wrapping_mul(dx);
        let mut first = ONE_PIXEL;
        if dy < 0 {
            p = fy1.wrapping_mul(dx);
            first = 0;
            incr = -1;
            dy = dy.wrapping_neg();
        }
        let (mut delta, mut modulo) = div_mod_int_long(p, dy)?;
        let mut x = self.x.wrapping_add(i64::from(delta));
        self.render_scanline(ey1, self.x, fy1, x, first)?;
        ey1 = ey1.wrapping_add(i64::from(incr));
        self.set_cell(x >> PIXEL_BITS, ey1)?;
        if ey1 != ey2 {
            p = ONE_PIXEL.wrapping_mul(dx);
            let (lift, rem) = div_mod_int_pair(p, dy)?;
            modulo = modulo.wrapping_sub(dy as i32 as i64);
            while ey1 != ey2 {
                delta = lift;
                modulo = modulo.wrapping_add(i64::from(rem));
                if modulo >= 0 {
                    modulo = modulo.wrapping_sub(dy as i32 as i64);
                    delta = delta.wrapping_add(1);
                }
                let x2 = x.wrapping_add(i64::from(delta));
                self.render_scanline(ey1, x, ONE_PIXEL.wrapping_sub(first), x2, first)?;
                x = x2;
                ey1 = ey1.wrapping_add(i64::from(incr));
                self.set_cell(x >> PIXEL_BITS, ey1)?;
            }
        }
        self.render_scanline(ey1, x, ONE_PIXEL.wrapping_sub(first), to_x, fy2)?;
        self.finish_line(to_x, to_y, ey2);
        Ok(())
    }

    /// The `End` block of `gray_render_line`: advances the current
    /// position and records the final scanline.
    #[inline]
    fn finish_line(&mut self, to_x: i64, to_y: i64, ey2: i64) {
        self.x = to_x;
        self.y = to_y;
        self.last_ey = ey2.wrapping_shl(PIXEL_BITS);
    }

    /// Flattens a quadratic Bézier arc into line
    /// segments with an adaptive subdivision depth (`dx >>= 2` while
    /// `dx > ONE_PIXEL / 4`), skipping arcs that never cross the current
    /// band, and renders each resulting line.
    ///
    /// # Errors
    ///
    /// * [`TtError::RASTER_OVERFLOW`] when the level or position stacks
    /// * Propagates [`gray_render_line`]'s errors.
    fn render_conic(&mut self, control: &Vector, to: &Vector) -> TtResult<()> {
        self.bez_stack[0] = Vector {
            x: to.x.wrapping_shl(UPSCALE_SHIFT),
            y: to.y.wrapping_shl(UPSCALE_SHIFT),
        };
        self.bez_stack[1] = Vector {
            x: control.x.wrapping_shl(UPSCALE_SHIFT),
            y: control.y.wrapping_shl(UPSCALE_SHIFT),
        };
        self.bez_stack[2] = Vector { x: self.x, y: self.y };
        let mut dx = abs_pos(
            self.bez_stack[2]
                .x
                .wrapping_add(self.bez_stack[0].x)
                .wrapping_sub(self.bez_stack[1].x.wrapping_mul(2)),
        );
        let dy = abs_pos(
            self.bez_stack[2]
                .y
                .wrapping_add(self.bez_stack[0].y)
                .wrapping_sub(self.bez_stack[1].y.wrapping_mul(2)),
        );
        if dx < dy {
            dx = dy;
        }
        if dx < ONE_PIXEL / 4 {
            let (ax, ay) = (self.bez_stack[0].x, self.bez_stack[0].y);
            return self.render_line(ax, ay);
        }
        let mut min = self.bez_stack[0].y;
        let mut max = min;
        for v in [self.bez_stack[1].y, self.bez_stack[2].y] {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        if (min >> PIXEL_BITS) >= self.max_ey || (max >> PIXEL_BITS) < self.min_ey {
            let (ax, ay) = (self.bez_stack[0].x, self.bez_stack[0].y);
            return self.render_line(ax, ay);
        }
        let mut level = 0i32;
        loop {
            dx >>= 2;
            level += 1;
            if dx <= ONE_PIXEL / 4 {
                break;
            }
        }
        *self
            .lev_stack
            .get_mut(0)
            .ok_or(TtError::RASTER_OVERFLOW)? = level;
        let mut top = 0i32;
        let mut arc = 0usize;
        loop {
            let lvl = *self
                .lev_stack
                .get(top as usize)
                .ok_or(TtError::RASTER_OVERFLOW)?;
            if lvl > 0 {
                let seg = self
                    .bez_stack
                    .get_mut(arc..arc + 5)
                    .ok_or(TtError::RASTER_OVERFLOW)?;
                split_conic(seg);
                arc += 2;
                top += 1;
                *self
                    .lev_stack
                    .get_mut(top as usize)
                    .ok_or(TtError::RASTER_OVERFLOW)? = lvl - 1;
                *self
                    .lev_stack
                    .get_mut(top as usize - 1)
                    .ok_or(TtError::RASTER_OVERFLOW)? = lvl - 1;
                continue;
            }
            let (ax, ay) = {
                let p = self
                    .bez_stack
                    .get(arc)
                    .ok_or(TtError::RASTER_OVERFLOW)?;
                (p.x, p.y)
            };
            self.render_line(ax, ay)?;
            if top <= 0 {
                break;
            }
            top -= 1;
            arc -= 2;
        }
        Ok(())
    }

    /// Flattens a cubic Bézier arc using Hain's
    /// rapid termination test (perpendicular distances and dot products
    /// against the chord), skipping arcs that never cross the current
    /// band, and renders each resulting line.
    ///
    /// # Errors
    ///
    /// * [`TtError::RASTER_OVERFLOW`] when the position stack overflows
    /// * Propagates [`gray_render_line`]'s errors.
    fn render_cubic(&mut self, control1: &Vector, control2: &Vector, to: &Vector) -> TtResult<()> {
        self.bez_stack[0] = Vector {
            x: to.x.wrapping_shl(UPSCALE_SHIFT),
            y: to.y.wrapping_shl(UPSCALE_SHIFT),
        };
        self.bez_stack[1] = Vector {
            x: control2.x.wrapping_shl(UPSCALE_SHIFT),
            y: control2.y.wrapping_shl(UPSCALE_SHIFT),
        };
        self.bez_stack[2] = Vector {
            x: control1.x.wrapping_shl(UPSCALE_SHIFT),
            y: control1.y.wrapping_shl(UPSCALE_SHIFT),
        };
        self.bez_stack[3] = Vector { x: self.x, y: self.y };
        let mut min = self.bez_stack[0].y;
        let mut max = min;
        for v in [self.bez_stack[1].y, self.bez_stack[2].y, self.bez_stack[3].y] {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        if (min >> PIXEL_BITS) >= self.max_ey || (max >> PIXEL_BITS) < self.min_ey {
            let (ax, ay) = (self.bez_stack[0].x, self.bez_stack[0].y);
            return self.render_line(ax, ay);
        }
        let mut arc = 0usize;
        loop {
            let seg = {
                let s = self
                    .bez_stack
                    .get(arc..arc + 4)
                    .ok_or(TtError::RASTER_OVERFLOW)?;
                [s[0], s[1], s[2], s[3]]
            };
            let (a0, a1, a2, a3) = (seg[0], seg[1], seg[2], seg[3]);
            let dx = a3.x.wrapping_sub(a0.x);
            let dy = a3.y.wrapping_sub(a0.y);
            let l = ft_hypot(dx, dy);
            let mut split = l > 32767;
            if !split {
                let s_limit = l.wrapping_mul(ONE_PIXEL / 6);
                let dx1 = a1.x.wrapping_sub(a0.x);
                let dy1 = a1.y.wrapping_sub(a0.y);
                let s = abs_pos(
                    dy.wrapping_mul(dx1)
                        .wrapping_sub(dx.wrapping_mul(dy1)),
                );
                if s > s_limit {
                    split = true;
                } else {
                    let dx2 = a2.x.wrapping_sub(a0.x);
                    let dy2 = a2.y.wrapping_sub(a0.y);
                    let s = abs_pos(
                        dy.wrapping_mul(dx2)
                            .wrapping_sub(dx.wrapping_mul(dy2)),
                    );
                    split = s > s_limit
                        || dx1
                            .wrapping_mul(dx1.wrapping_sub(dx))
                            .wrapping_add(dy1.wrapping_mul(dy1.wrapping_sub(dy)))
                            > 0
                        || dx2
                            .wrapping_mul(dx2.wrapping_sub(dx))
                            .wrapping_add(dy2.wrapping_mul(dy2.wrapping_sub(dy)))
                            > 0;
                }
            }
            if split {
                let seg = self
                    .bez_stack
                    .get_mut(arc..arc + 7)
                    .ok_or(TtError::RASTER_OVERFLOW)?;
                split_cubic(seg);
                arc += 3;
                continue;
            }
            let (ax, ay) = {
                let p = self
                    .bez_stack
                    .get(arc)
                    .ok_or(TtError::RASTER_OVERFLOW)?;
                (p.x, p.y)
            };
            self.render_line(ax, ay)?;
            if arc == 0 {
                return Ok(());
            }
            arc -= 3;
        }
    }

    /// Emits a horizontal run of `acount` pixels at
    /// `(x, y)` with the coverage derived from `area >> 9`, applying the
    /// non-zero or even-odd fill rule, then appends it to the current
    /// span batch (merging with the previous span when adjacent and
    /// identically covered, flushing the batch first when the scanline or
    /// the 32-span limit requires it).
    ///
    /// # Errors
    ///
    /// [`TtError::RASTER_OVERFLOW`] or [`TtError::RASTER_CORRUPTED`] on
    /// an out-of-range span batch index.
    fn hline(&mut self, x: i64, y: i64, area: i64, acount: i64) -> TtResult<()> {
        let mut coverage = (area >> (PIXEL_BITS * 2 + 1 - 8)) as i32;
        if coverage < 0 {
            coverage = coverage.wrapping_neg();
        }
        if self.outline.flags & OUTLINE_EVEN_ODD_FILL != 0 {
            coverage &= 511;
            if coverage > 256 {
                coverage = 512 - coverage;
            } else if coverage == 256 {
                coverage = 255;
            }
        } else if coverage >= 256 {
            coverage = 255;
        }
        let mut y = y.wrapping_add(self.min_ey);
        let mut x = x.wrapping_add(self.min_ex);
        if x >= 32767 {
            x = 32767;
        }
        if y >= i32::MAX as i64 {
            y = i32::MAX as i64;
        }
        if coverage != 0 {
            let count = self.num_gray_spans;
            if count > 0 && i64::from(self.span_y) == y {
                let idx = (count - 1) as usize;
                let prev = *self
                    .gray_spans
                    .get(idx)
                    .ok_or(TtError::RASTER_CORRUPTED)?;
                if i32::from(prev.x).wrapping_add(i32::from(prev.len)) == x as i32
                    && i32::from(prev.coverage) == coverage
                {
                    let span = self
                        .gray_spans
                        .get_mut(idx)
                        .ok_or(TtError::RASTER_CORRUPTED)?;
                    span.len = i64::from(span.len).wrapping_add(acount) as u16;
                    return Ok(());
                }
            }
            let slot = if i64::from(self.span_y) != y || count >= FT_MAX_GRAY_SPANS as i32 {
                self.flush_spans();
                self.num_gray_spans = 0;
                self.span_y = y as i32;
                0
            } else {
                count
            };
            let span = self
                .gray_spans
                .get_mut(slot as usize)
                .ok_or(TtError::RASTER_OVERFLOW)?;
            span.x = x as i16;
            span.len = acount as u16;
            span.coverage = coverage as u8;
            self.num_gray_spans = slot + 1;
        }
        Ok(())
    }

    /// Delivers the pending span batch to the sink when non-empty (the
    /// `render_span` calls of `gray_hline`'s flush path and of
    /// `gray_sweep`'s tail).
    fn flush_spans(&mut self) {
        if self.num_gray_spans > 0 {
            let n = (self.num_gray_spans as usize).min(FT_MAX_GRAY_SPANS);
            let y = self.span_y;
            self.sink.emit(y, n, &self.gray_spans[..n]);
        }
    }

    /// Walks every cell row of the band in x order,
    /// emitting the horizontal gaps (non-zero cover between cells) and
    /// each cell's own coverage through [`GrayWorker::hline`], then
    /// flushes the final span batch.
    ///
    /// # Errors
    ///
    /// Propagates [`gray_hline`]'s and [`gray_flush`]'s errors.
    fn sweep(&mut self) -> TtResult<()> {
        if self.num_cells == 0 {
            return Ok(());
        }
        self.num_gray_spans = 0;
        for yindex in 0..self.ycount {
            let mut cover = 0i64;
            let mut x = 0i64;
            let mut idx = *self
                .ycells
                .get(yindex as usize)
                .ok_or(TtError::RASTER_CORRUPTED)?;
            while idx >= 0 {
                let cell = *self
                    .cells
                    .get(idx as usize)
                    .ok_or(TtError::RASTER_CORRUPTED)?;
                if cell.x > x && cover != 0 {
                    self.hline(
                        x,
                        yindex,
                        cover.wrapping_mul(ONE_PIXEL * 2),
                        cell.x.wrapping_sub(x),
                    )?;
                }
                cover = cover.wrapping_add(cell.cover);
                let area = cover
                    .wrapping_mul(ONE_PIXEL * 2)
                    .wrapping_sub(i64::from(cell.area));
                if area != 0 && cell.x >= 0 {
                    self.hline(cell.x, yindex, area, 1)?;
                }
                x = cell.x.wrapping_add(1);
                idx = cell.next;
            }
            if cover != 0 {
                self.hline(
                    x,
                    yindex,
                    cover.wrapping_mul(ONE_PIXEL * 2),
                    self.count_ex.wrapping_sub(x),
                )?;
            }
        }
        self.flush_spans();
        Ok(())
    }

    /// Computes the cell pool layout for `band` (`ycells` head array followed by the
    /// aligned cell array inside the 16 KiB pool) and (re)initializes the
    /// band-scoped state.
    ///
    /// Returns `false` when the pool cannot hold at least two cells for
    /// this band height, which makes the caller split the band.
    ///
    /// # Errors
    ///
    /// * [`TtError::RASTER_CORRUPTED`] when the pool layout overflows the
    ///   64-bit range or the band is malformed.
    fn setup_band(&mut self, band: Band) -> TtResult<bool> {
        let ycount = band.max.wrapping_sub(band.min);
        if ycount < 0 {
            return Err(TtError::RASTER_CORRUPTED);
        }
        let cell_start = (PCELL_SIZE as i64)
            .checked_mul(ycount)
            .ok_or(TtError::RASTER_CORRUPTED)?;
        let cell_mod = cell_start.wrapping_rem(TCELL_SIZE as i64);
        let cell_start = if cell_mod > 0 {
            cell_start
                .checked_add(TCELL_SIZE as i64 - cell_mod)
                .ok_or(TtError::RASTER_CORRUPTED)?
        } else {
            cell_start
        };
        if cell_start >= FT_RENDER_POOL_SIZE as i64 {
            return Ok(false);
        }
        let max_cells = (FT_RENDER_POOL_SIZE as i64).wrapping_sub(cell_start) / TCELL_SIZE as i64;
        if max_cells < 2 {
            return Ok(false);
        }
        self.ycells.clear();
        self.ycells.resize(ycount as usize, -1);
        self.num_cells = 0;
        self.invalid = true;
        self.min_ey = band.min;
        self.max_ey = band.max;
        self.count_ey = ycount;
        self.ycount = ycount;
        self.max_cells = max_cells as usize;
        Ok(true)
    }

    /// Decomposes the outline through [`outline_decompose`] and records
    /// the final cell, as in C (where a cell pool exhaustion `longjmp`s
    /// straight to the band splitter with `Memory_Overflow`).
    ///
    /// # Errors
    ///
    /// Propagates decompose errors ([`TtError::INVALID_OUTLINE`]) and
    /// cell recording errors ([`TtError::OUT_OF_MEMORY`]).
    fn convert_glyph_inner(&mut self) -> TtResult<()> {
        let outline = self.outline;
        outline_decompose(outline, self)?;
        if !self.invalid {
            self.record_cell()?;
        }
        Ok(())
    }

    /// Computes and clips the bounding box, then
    /// renders the outline band by band.  Each band is split on cell pool
    /// exhaustion ([`reduce_band`]) up to 39 levels; after the pass an
    /// oversized band count shrinks `band_size` for the next glyph.
    ///
    /// # Errors
    ///
    /// * [`TtError::RASTER_OVERFLOW`] for a rotten band.
    /// * [`TtError::RASTER_CORRUPTED`] on a malformed band stack.
    /// * Propagates inner conversion and sweep errors.
    fn convert_glyph(&mut self) -> TtResult<()> {
        self.compute_cbox();
        let clip = self.clip_box;
        if self.max_ex <= clip.x_min
            || self.min_ex >= clip.x_max
            || self.max_ey <= clip.y_min
            || self.min_ey >= clip.y_max
        {
            return Ok(());
        }
        if self.min_ex < clip.x_min {
            self.min_ex = clip.x_min;
        }
        if self.min_ey < clip.y_min {
            self.min_ey = clip.y_min;
        }
        if self.max_ex > clip.x_max {
            self.max_ex = clip.x_max;
        }
        if self.max_ey > clip.y_max {
            self.max_ey = clip.y_max;
        }
        self.count_ex = self.max_ex.wrapping_sub(self.min_ex);
        self.count_ey = self.max_ey.wrapping_sub(self.min_ey);
        let mut num_bands = (self.max_ey.wrapping_sub(self.min_ey) / i64::from(self.band_size)) as i32;
        if num_bands == 0 {
            num_bands = 1;
        }
        if num_bands >= 39 {
            num_bands = 39;
        }
        self.band_shoot = 0;
        let max_y = self.max_ey;
        let mut min = self.min_ey;
        for n in 0..num_bands {
            let mut max = min.wrapping_add(i64::from(self.band_size));
            if n == num_bands - 1 || max > max_y {
                max = max_y;
            }
            let mut stack: Vec<Band> = vec![Band { min, max }];
            while let Some(&band) = stack.last() {
                let mut reduce = !self.setup_band(band)?;
                if !reduce {
                    match self.convert_glyph_inner() {
                        Ok(()) => {
                            self.sweep()?;
                            stack.pop();
                        }
                        Err(e) if e == TtError::OUT_OF_MEMORY => reduce = true,
                        Err(e) => return Err(e),
                    }
                }
                if reduce && reduce_band(&mut stack, self.band_size)? {
                    self.band_shoot += 1;
                }
            }
            min = max;
        }
        if self.band_shoot > 8 && self.band_size > 16 {
            self.band_size /= 2;
        }
        Ok(())
    }
}

/// The `move_to`/`line_to`/`conic_to`/`cubic_to` emitters of
/// `FT_Outline_Funcs` (`ftgrays.c`'s `gray_move_to` and friends), driven
/// by [`outline_decompose`].
impl GrayWorker<'_, '_> {
    /// Records the previous cell, starts a new contour at
    /// the upscaled target position and adopts it as the current point.
    fn move_to(&mut self, to: &Vector) -> TtResult<()> {
        if !self.invalid {
            self.record_cell()?;
        }
        let x = to.x.wrapping_shl(UPSCALE_SHIFT);
        let y = to.y.wrapping_shl(UPSCALE_SHIFT);
        self.start_cell(x >> PIXEL_BITS, y >> PIXEL_BITS)?;
        self.x = x;
        self.y = y;
        Ok(())
    }

    /// Renders a straight segment to the upscaled target.
    fn line_to(&mut self, to: &Vector) -> TtResult<()> {
        self.render_line(to.x.wrapping_shl(UPSCALE_SHIFT), to.y.wrapping_shl(UPSCALE_SHIFT))
    }

    /// Flattens a quadratic arc to the upscaled target.
    fn conic_to(&mut self, control: &Vector, to: &Vector) -> TtResult<()> {
        self.render_conic(control, to)
    }

    /// Flattens a cubic arc to the upscaled target.
    fn cubic_to(&mut self, control1: &Vector, control2: &Vector, to: &Vector) -> TtResult<()> {
        self.render_cubic(control1, control2, to)
    }
}

/// The curve tag mask of `FT_CURVE_TAG` (`flag & 3`).
const CURVE_TAG_MASK: u8 = 0x03;
/// `FT_CURVE_TAG_ON` — the point is on the curve.
const CURVE_TAG_ON: u8 = 0x01;
/// `FT_CURVE_TAG_CONIC` — the point is a quadratic control point.
const CURVE_TAG_CONIC: u8 = 0x00;
/// `FT_CURVE_TAG_CUBIC` — the point is a cubic control point.
const CURVE_TAG_CUBIC: u8 = 0x02;

/// Faithful port of `FT_Outline_Decompose` (FreeType 2.6, `ftoutln.c`),
/// the standalone decomposer `ftgrays.c` carries for `_STANDALONE_`
/// builds.  The scaling transform (`shift` and `delta` of
/// `FT_Outline_Funcs`) is the identity, matching the smooth rasterizer
/// which always calls the decomposer with `shift = 0, delta = 0`.
///
/// Every contour is reported to `worker` as a `move_to`, a sequence of
/// `line_to`/`conic_to`/`cubic_to` calls, and a closing `line_to`.
///
/// # Errors
///
/// Returns [`TtError::INVALID_OUTLINE`] when a contour end index is
/// negative or out of range, when a contour starts with a cubic control
/// point, or when a conic arc is followed by a non-conic/non-on point —
/// the exact `Invalid_Outline` paths of the C routine.  Additional bounds
/// checks defend against contours whose indices run past the backing
/// slices (undefined behavior in C).  Errors returned by `worker` are
/// propagated unchanged.
///
/// # Complexity
///
/// O(n) in the number of points; no allocation is performed.
fn outline_decompose(outline: &Outline, worker: &mut GrayWorker<'_, '_>) -> TtResult<()> {
    let mut first: isize = 0;
    for n in 0..outline.n_contours {
        let last = *outline
            .contours
            .get(n as usize)
            .ok_or(TtError::INVALID_OUTLINE)? as isize;
        if last < 0 {
            return Err(TtError::INVALID_OUTLINE);
        }
        let mut limit = last;
        let mut v_start = decompose_point(outline, first)?;
        let v_last = decompose_point(outline, last)?;
        let mut cur = first;
        let tag_val = decompose_tag(outline, cur)?;
        if tag_val == CURVE_TAG_CUBIC {
            return Err(TtError::INVALID_OUTLINE);
        }
        if tag_val == CURVE_TAG_CONIC {
            if decompose_tag(outline, last)? == CURVE_TAG_ON {
                v_start = v_last;
                limit -= 1;
            } else {
                v_start.x = v_start.x.wrapping_add(v_last.x) / 2;
                v_start.y = v_start.y.wrapping_add(v_last.y) / 2;
            }
            cur -= 1;
        }
        worker.move_to(&v_start)?;
        let mut skip_line = false;
        'contour: while cur < limit {
            cur += 1;
            match decompose_tag(outline, cur)? {
                CURVE_TAG_ON => {
                    let vec = decompose_point(outline, cur)?;
                    worker.line_to(&vec)?;
                }
                CURVE_TAG_CONIC => {
                    let mut v_control = decompose_point(outline, cur)?;
                    'do_conic: loop {
                        if cur < limit {
                            cur += 1;
                            let next_tag = decompose_tag(outline, cur)?;
                            let vec = decompose_point(outline, cur)?;
                            if next_tag == CURVE_TAG_ON {
                                worker.conic_to(&v_control, &vec)?;
                                continue 'contour;
                            }
                            if next_tag != CURVE_TAG_CONIC {
                                return Err(TtError::INVALID_OUTLINE);
                            }
                            let mut v_middle = v_control;
                            v_middle.x = v_middle.x.wrapping_add(vec.x) / 2;
                            v_middle.y = v_middle.y.wrapping_add(vec.y) / 2;
                            worker.conic_to(&v_control, &v_middle)?;
                            v_control = vec;
                            continue 'do_conic;
                        }
                        worker.conic_to(&v_control, &v_start)?;
                        skip_line = true;
                        break 'contour;
                    }
                }
                _ => {
                    if cur + 1 > limit || decompose_tag(outline, cur + 1)? != CURVE_TAG_CUBIC {
                        return Err(TtError::INVALID_OUTLINE);
                    }
                    cur += 2;
                    let vec1 = decompose_point(outline, cur - 2)?;
                    let vec2 = decompose_point(outline, cur - 1)?;
                    if cur <= limit {
                        let vec = decompose_point(outline, cur)?;
                        worker.cubic_to(&vec1, &vec2, &vec)?;
                    } else {
                        worker.cubic_to(&vec1, &vec2, &v_start)?;
                        skip_line = true;
                        break 'contour;
                    }
                }
            }
        }
        if !skip_line {
            worker.line_to(&v_start)?;
        }
        first = last + 1;
    }
    Ok(())
}

/// Reads the point at index `i`, mapping out-of-range indices to
/// [`TtError::INVALID_OUTLINE`] instead of the C routine's undefined
/// behavior.
#[inline]
fn decompose_point(outline: &Outline, i: isize) -> TtResult<Vector> {
    if i < 0 {
        return Err(TtError::INVALID_OUTLINE);
    }
    outline
        .points
        .get(i as usize)
        .copied()
        .ok_or(TtError::INVALID_OUTLINE)
}

/// Reads the curve tag (`FT_CURVE_TAG`) of the point at index `i`.
#[inline]
fn decompose_tag(outline: &Outline, i: isize) -> TtResult<u8> {
    if i < 0 {
        return Err(TtError::INVALID_OUTLINE);
    }
    let t = *outline
        .tags
        .get(i as usize)
        .ok_or(TtError::INVALID_OUTLINE)?;
    Ok(t & CURVE_TAG_MASK)
}

/// The `ft_grays_raster` object of FreeType 2.6: an antialiasing
/// rasterizer implementing [`Raster`] over the reusable cell buffers of
/// the render pool.
///
/// `gray_raster_reset` and `gray_raster_set_mode` are no-ops in C, and
/// the raster allocates its pool internally, so no external state is
/// kept besides the cell vectors.
pub struct GrayRaster {
    /// Preallocated cell pool (`NUM_CELLS` entries), reused per band.
    cells: Vec<Cell>,
    /// Per-scanline cell list heads, reused per band.
    ycells: Vec<i32>,
}

impl GrayRaster {
    /// Creates a rasterizer with an empty cell pool (filled per render).
    pub fn new() -> Self {
        let mut cells = Vec::with_capacity(NUM_CELLS);
        cells.resize(NUM_CELLS, Cell::default());
        GrayRaster {
            cells,
            ycells: Vec::new(),
        }
    }
}

impl Default for GrayRaster {
    fn default() -> Self {
        Self::new()
    }
}

impl Raster for GrayRaster {
    /// No-op; the function ignores the render pool
    /// because this raster allocates its own workspace.
    fn reset(&mut self, _pool: Option<&mut [u8]>, _pool_size: usize) {}

    /// Accepts every mode without action.
    ///
    /// # Errors
    ///
    /// Never fails; the signature matches the [`Raster`] contract.
    fn set_mode(&mut self, _mode: u64, _value: &mut dyn core::any::Any) -> TtResult<()> {
        Ok(())
    }

    /// `gray_raster_render`: validates the outline, builds the clip box from the target (indirect mode) or
    /// the parameters (direct mode), then runs [`GrayWorker`]'s band
    /// conversion.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_OUTLINE`] for inconsistent outline geometry.
    /// * [`TtError::INVALID_ARGUMENT`] for a degenerate or short target
    ///   buffer (C reads out of bounds instead — documented deviation).
    /// * [`TtError::CANNOT_RENDER_GLYPH`] when the `AA` flag is missing.
    /// * Conversion errors of [`GrayWorker::convert_glyph`].
    fn render(&mut self, params: &mut RasterParams<'_>) -> TtResult<()> {
        let outline = params.source;
        let flags = params.flags;
        if outline.n_points == 0 || outline.n_contours <= 0 {
            return Ok(());
        }
        if outline.contours.is_empty() || outline.points.is_empty() {
            return Err(TtError::INVALID_OUTLINE);
        }
        let last = *outline
            .contours
            .get(outline.n_contours as usize - 1)
            .ok_or(TtError::INVALID_OUTLINE)?;
        if i32::from(outline.n_points) != i32::from(last) + 1 {
            return Err(TtError::INVALID_OUTLINE);
        }
        if outline.n_points > 0 && outline.points.len() < outline.n_points as usize {
            return Err(TtError::INVALID_OUTLINE);
        }
        let direct = flags.contains(RasterFlags::DIRECT);
        if !direct {
            let target = &*params.target;
            if target.width == 0 || target.rows == 0 {
                return Ok(());
            }
            if target.buffer.is_empty() {
                return Err(TtError::INVALID_ARGUMENT);
            }
            if target.stride() == 0 {
                return Err(TtError::INVALID_ARGUMENT);
            }
            let size = (target.rows as usize)
                .checked_mul(target.stride())
                .ok_or(TtError::INVALID_ARGUMENT)?;
            if target.buffer.len() < size {
                return Err(TtError::INVALID_ARGUMENT);
            }
        }
        if !flags.contains(RasterFlags::AA) {
            return Err(TtError::CANNOT_RENDER_GLYPH);
        }
        let clip_box = if !direct {
            let target = &*params.target;
            BBox::from_edges(0, 0, target.width as i64, target.rows as i64)
        } else if flags.contains(RasterFlags::CLIP) {
            params.clip_box
        } else {
            BBox::from_edges(-32768, -32768, 32767, 32767)
        };
        let sink = if direct {
            Sink::Direct {
                func: params.gray_spans,
                user: params.user.as_deref_mut(),
            }
        } else {
            Sink::Bitmap {
                bitmap: &mut *params.target,
            }
        };
        let mut worker = GrayWorker::new(outline, sink, &mut self.cells, &mut self.ycells, clip_box);
        worker.convert_glyph()
    }
}

/// `gray_raster_new`: the `FT_Raster_NewFunc` of [`FT_GRAYS_RASTER`].
///
/// The C function stores the library allocator inside the raster object;
/// this port's raster keeps its workspace in ordinary `Vec`s, so
/// `memory` is unused.
///
/// # Errors
///
/// Never fails; the signature matches [`RasterFuncs::new`].
pub fn gray_raster_new(_memory: &Memory) -> TtResult<Box<dyn Raster>> {
    Ok(Box::new(GrayRaster::new()))
}

/// `ft_grays_raster`: the `FT_Raster_Funcs` descriptor of the smooth
/// module's antialiasing rasterizer (`ftgrays.c`).
pub static FT_GRAYS_RASTER: RasterFuncs = RasterFuncs {
    glyph_format: GlyphFormat::Outline,
    new: gray_raster_new,
};

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_truetype_core::PixelMode;

    fn pt(x: i64, y: i64) -> Vector {
        Vector { x, y }
    }

    fn square(x0: i64, y0: i64, x1: i64, y1: i64) -> Outline {
        Outline {
            n_contours: 1,
            n_points: 4,
            points: vec![pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)],
            tags: vec![1, 1, 1, 1],
            contours: vec![3],
            flags: 0,
        }
    }

    fn gray_target(rows: u32, width: u32) -> Bitmap {
        Bitmap::new_sized(rows, width, PixelMode::Gray, 256).unwrap()
    }

    fn collect_spans(y: i32, count: usize, spans: &[Span], user: &mut dyn core::any::Any) {
        if let Some(log) = user.downcast_mut::<Vec<(i32, Vec<Span>)>>() {
            let mut batch = Vec::new();
            for span in spans.iter().take(count) {
                batch.push(*span);
            }
            log.push((y, batch));
        }
    }

    fn render_gray(outline: &Outline, target: &mut Bitmap) -> TtResult<()> {
        let mut raster = GrayRaster::new();
        let mut params = RasterParams {
            target,
            source: outline,
            flags: RasterFlags::AA,
            gray_spans: None,
            user: None,
            clip_box: BBox::new(),
        };
        raster.render(&mut params)
    }

    #[test]
    fn full_pixel_square_has_full_coverage() {
        let mut target = gray_target(1, 1);
        let outline = square(0, 0, 64, 64);
        render_gray(&outline, &mut target).unwrap();
        assert_eq!(target.buffer[0], 255);
    }

    #[test]
    fn half_width_square_has_half_coverage() {
        let mut target = gray_target(1, 1);
        let outline = square(0, 0, 32, 64);
        render_gray(&outline, &mut target).unwrap();
        assert_eq!(target.buffer[0], 128);
    }

    #[test]
    fn span_rows_follow_signed_pitch() {
        let outline = square(0, 64, 64, 128);
        let mut positive = gray_target(2, 1);
        render_gray(&outline, &mut positive).unwrap();
        assert_eq!(positive.buffer, vec![255, 0]);

        let mut negative = Bitmap {
            rows: 2,
            width: 1,
            pitch: -1,
            buffer: vec![0; 2],
            num_grays: 256,
            pixel_mode: PixelMode::Gray,
        };
        render_gray(&outline, &mut negative).unwrap();
        assert_eq!(negative.buffer, vec![0, 255]);
    }

    #[test]
    fn empty_outline_returns_ok() {
        let outline = Outline {
            n_contours: 0,
            n_points: 0,
            points: vec![],
            tags: vec![],
            contours: vec![],
            flags: 0,
        };
        let mut target = Bitmap::new();
        render_gray(&outline, &mut target).unwrap();
    }

    #[test]
    fn contour_point_mismatch_is_invalid_outline() {
        let outline = Outline {
            n_contours: 1,
            n_points: 4,
            points: square(0, 0, 64, 64).points,
            tags: vec![1, 1, 1, 1],
            contours: vec![2],
            flags: 0,
        };
        let mut target = gray_target(1, 1);
        let err = render_gray(&outline, &mut target).unwrap_err();
        assert_eq!(err, TtError::INVALID_OUTLINE);
    }

    #[test]
    fn missing_aa_flag_is_cannot_render_glyph() {
        let mut target = gray_target(1, 1);
        let outline = square(0, 0, 64, 64);
        let mut raster = GrayRaster::new();
        let mut params = RasterParams {
            target: &mut target,
            source: &outline,
            flags: RasterFlags::DEFAULT,
            gray_spans: None,
            user: None,
            clip_box: BBox::new(),
        };
        let err = raster.render(&mut params).unwrap_err();
        assert_eq!(err, TtError::CANNOT_RENDER_GLYPH);
    }

    #[test]
    fn direct_mode_delivers_spans_to_callback() {
        fn collect(y: i32, count: usize, spans: &[Span], user: &mut dyn core::any::Any) {
            if let Some(log) = user.downcast_mut::<Vec<(i32, Vec<Span>)>>() {
                let mut batch = Vec::new();
                for span in spans.iter().take(count) {
                    batch.push(*span);
                }
                log.push((y, batch));
            }
        }

        let outline = square(0, 0, 64, 64);
        let mut target = gray_target(1, 1);
        let mut log: Vec<(i32, Vec<Span>)> = Vec::new();
        let mut raster = GrayRaster::new();
        let mut params = RasterParams {
            target: &mut target,
            source: &outline,
            flags: RasterFlags::AA | RasterFlags::DIRECT,
            gray_spans: Some(collect),
            user: Some(&mut log),
            clip_box: BBox::new(),
        };
        raster.render(&mut params).unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].0, 0);
        assert_eq!(log[0].1.len(), 1);
        assert_eq!(
            log[0].1[0],
            Span {
                x: 0,
                len: 1,
                coverage: 255
            }
        );
        assert_eq!(target.buffer, vec![0]);
    }

    fn double_square(flags: i32) -> Outline {
        Outline {
            n_contours: 2,
            n_points: 8,
            points: vec![
                pt(0, 0),
                pt(64, 0),
                pt(64, 64),
                pt(0, 64),
                pt(0, 0),
                pt(64, 0),
                pt(64, 64),
                pt(0, 64),
            ],
            tags: vec![1, 1, 1, 1, 1, 1, 1, 1],
            contours: vec![3, 7],
            flags,
        }
    }

    #[test]
    fn even_odd_fill_rule_differs_from_nonzero() {
        let mut nonzero_target = gray_target(1, 1);
        render_gray(&double_square(0), &mut nonzero_target).unwrap();
        assert_eq!(nonzero_target.buffer[0], 255);

        let mut even_odd_target = gray_target(1, 1);
        render_gray(&double_square(OUTLINE_EVEN_ODD_FILL), &mut even_odd_target).unwrap();
        assert_eq!(even_odd_target.buffer[0], 0);
    }

    #[test]
    fn collinear_conic_edge_renders_as_full_pixel() {
        let outline = Outline {
            n_contours: 1,
            n_points: 5,
            points: vec![pt(0, 0), pt(16, 0), pt(64, 0), pt(64, 64), pt(0, 64)],
            tags: vec![1, 0, 1, 1, 1],
            contours: vec![4],
            flags: 0,
        };
        let mut target = gray_target(1, 1);
        render_gray(&outline, &mut target).unwrap();
        assert_eq!(target.buffer[0], 255);
    }

    #[test]
    fn dipping_cubic_edge_renders_as_full_pixel() {
        let outline = Outline {
            n_contours: 1,
            n_points: 6,
            points: vec![
                pt(0, 0),
                pt(-4000, -4000),
                pt(4500, -4000),
                pt(64, 0),
                pt(64, 64),
                pt(0, 64),
            ],
            tags: vec![1, 2, 2, 1, 1, 1],
            contours: vec![5],
            flags: 0,
        };
        let mut target = gray_target(1, 1);
        render_gray(&outline, &mut target).unwrap();
        assert_eq!(target.buffer[0], 255);
    }

    #[test]
    fn quarter_width_square_has_quarter_coverage() {
        let mut target = gray_target(1, 1);
        let outline = square(0, 0, 16, 64);
        render_gray(&outline, &mut target).unwrap();
        assert_eq!(target.buffer[0], 64);
    }

    fn triangle() -> Outline {
        Outline {
            n_contours: 1,
            n_points: 3,
            points: vec![pt(0, 0), pt(192, 64), pt(0, 64)],
            tags: vec![1, 1, 1],
            contours: vec![2],
            flags: 0,
        }
    }

    #[test]
    fn triangle_coverage_row_matches_exact_area() {
        let mut target = gray_target(1, 3);
        render_gray(&triangle(), &mut target).unwrap();
        let total: u32 = target.buffer.iter().map(|&b| u32::from(b)).sum();
        assert!(total > 360 && total < 410, "area {total} is not ~1.5 px^2");
        assert_eq!(target.buffer[0], 214);
        assert_eq!(target.buffer[1], 129);
        assert_eq!(target.buffer[2], 43);
    }

    #[test]
    fn triangle_direct_mode_emits_one_span_per_column() {
        let outline = triangle();
        let mut target = gray_target(1, 3);
        let mut log: Vec<(i32, Vec<Span>)> = Vec::new();
        let mut raster = GrayRaster::new();
        let mut params = RasterParams {
            target: &mut target,
            source: &outline,
            flags: RasterFlags::AA | RasterFlags::DIRECT,
            gray_spans: Some(collect_spans),
            user: Some(&mut log),
            clip_box: BBox::new(),
        };
        raster.render(&mut params).unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].0, 0);
        let spans = &log[0].1;
        assert!(spans.len() >= 3, "spans: {spans:?}");
        let mut x = 0i32;
        for span in spans.iter().take(3) {
            assert_eq!(i32::from(span.x), x);
            x += i32::from(span.len);
        }
        assert!(x <= 3, "span run {x} exceeds the target width");
        assert_eq!(target.buffer, vec![0, 0, 0]);
    }

    #[test]
    fn full_rect_coverage_sums_to_pixel_count() {
        let mut target = gray_target(2, 3);
        let outline = square(0, 0, 192, 128);
        render_gray(&outline, &mut target).unwrap();
        let total: u32 = target.buffer.iter().map(|&b| u32::from(b)).sum();
        assert_eq!(total, 1530);
    }

    #[test]
    fn glyph_larger_than_target_is_clipped() {
        let mut target = gray_target(2, 2);
        let outline = square(0, 0, 640, 640);
        render_gray(&outline, &mut target).unwrap();
        assert_eq!(target.buffer, vec![255; 4]);
    }

    #[test]
    fn zero_size_target_returns_ok() {
        let outline = square(0, 0, 64, 64);
        let mut target = Bitmap::new();
        render_gray(&outline, &mut target).unwrap();
        assert!(target.buffer.is_empty());
    }

    #[test]
    fn short_target_buffer_is_invalid_argument() {
        let outline = square(0, 0, 64, 64);
        let mut target = Bitmap {
            rows: 2,
            width: 2,
            pitch: 2,
            buffer: vec![0; 3],
            num_grays: 256,
            pixel_mode: PixelMode::Gray,
        };
        let err = render_gray(&outline, &mut target).unwrap_err();
        assert_eq!(err, TtError::INVALID_ARGUMENT);
    }

    #[test]
    fn zero_pitch_target_is_invalid_argument() {
        let outline = square(0, 0, 64, 64);
        let mut target = Bitmap {
            rows: 2,
            width: 2,
            pitch: 0,
            buffer: vec![0; 4],
            num_grays: 256,
            pixel_mode: PixelMode::Gray,
        };
        let err = render_gray(&outline, &mut target).unwrap_err();
        assert_eq!(err, TtError::INVALID_ARGUMENT);
    }

    #[test]
    fn raster_reset_and_set_mode_are_noops() {
        let mut raster = GrayRaster::new();
        raster.reset(None, 0);
        raster.reset(Some(&mut [0u8; 64]), 64);
        let mut value = 7i32;
        assert!(raster.set_mode(0x1234, &mut value).is_ok());
        assert_eq!(value, 7);
    }

    #[test]
    fn grays_raster_factory_creates_outline_raster() {
        assert_eq!(FT_GRAYS_RASTER.glyph_format, GlyphFormat::Outline);
        let mut raster = (FT_GRAYS_RASTER.new)(&Memory).unwrap();
        let outline = square(0, 0, 64, 64);
        let mut target = gray_target(1, 1);
        let mut params = RasterParams {
            target: &mut target,
            source: &outline,
            flags: RasterFlags::AA,
            gray_spans: None,
            user: None,
            clip_box: BBox::new(),
        };
        raster.render(&mut params).unwrap();
        assert_eq!(target.buffer[0], 255);
    }

    #[test]
    fn direct_mode_clip_box_excludes_outline() {
        let outline = square(0, 0, 64, 64);
        let mut target = gray_target(1, 1);
        let mut log: Vec<(i32, Vec<Span>)> = Vec::new();
        let mut raster = GrayRaster::new();
        let mut params = RasterParams {
            target: &mut target,
            source: &outline,
            flags: RasterFlags::AA | RasterFlags::DIRECT | RasterFlags::CLIP,
            gray_spans: Some(collect_spans),
            user: Some(&mut log),
            clip_box: BBox::from_edges(1000, 1000, 1010, 1010),
        };
        raster.render(&mut params).unwrap();
        assert!(log.is_empty());
    }
}
