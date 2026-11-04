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

//! `ttgload.c`: decoding of one `glyf` entry into an outline.
//!
//! [`TtLoader`] carries every piece of state FreeType keeps in
//! `TT_LoaderRec`, and [`load_glyph`] mirrors `TT_Load_Glyph`.  See the
//! crate documentation for the deviations from the C original.

use alloc::vec::Vec;

use codevar_logger::log_warn;
use codevar_truetype_core::{
    BBox, CURVE_TAG_ON, Fixed, GlyphFormat, GlyphLoader, GlyphSlot, LoadFlags, Matrix,
    OUTLINE_HIGH_PRECISION, OUTLINE_SINGLE_PASS, SUBGLYPH_FLAG_2X2, SUBGLYPH_FLAG_ARGS_ARE_WORDS,
    SUBGLYPH_FLAG_ARGS_ARE_XY_VALUES, SUBGLYPH_FLAG_ROUND_XY_TO_GRID, SUBGLYPH_FLAG_SCALE,
    SUBGLYPH_FLAG_USE_MY_METRICS, SUBGLYPH_FLAG_XY_SCALE, SizeMetrics, SubGlyph, TtError, TtResult, Vector,
    div_fix, hypot, mul_fix, pix_round, transform_points, translate_points,
};
use codevar_truetype_sfnt::{SfntFont, tags::TAG_GLYF};

use crate::{SUBGLYPH_FLAG_MORE_COMPONENTS, SUBGLYPH_FLAG_SCALED_COMPONENT_OFFSET};

/// `FT_LOAD_NO_SCALE` expressed as a 16.16 identity scale.
const IDENTITY_SCALE: Fixed = 0x1_0000;

/// The TrueType glyph loader (`TT_LoaderRec`).
///
/// One instance covers a single top-level [`load_glyph`] call: it owns
/// the [`GlyphLoader`] taken from the [`GlyphSlot`], the open `glyf`
/// byte range, the glyph header, the horizontal/vertical metrics and
/// the four phantom points.  Composite recursion re-enters the same
/// instance, exactly like FreeType's `load_truetype_glyph`.
struct TtLoader<'a, 'b> {
    /// The open font (`TT_Loader.face`).
    font: &'a SfntFont<'b>,
    /// The `glyf` table (`loader->glyf_offset`); `None` without one.
    glyf: Option<&'b [u8]>,
    /// The glyph loader shared with the slot (`loader->gloader`).
    gloader: GlyphLoader,
    /// The caller's `FT_LOAD_XXX` bits.
    load_flags: LoadFlags,
    /// Font-unit to 26.6 horizontal scale (`x_scale`), the identity
    /// under `FT_LOAD_NO_SCALE`.
    x_scale: Fixed,
    /// Vertical counterpart of `x_scale`.
    y_scale: Fixed,
    /// The open glyph frame (`stream->cursor .. stream->limit`).
    frame: &'b [u8],
    /// Read offset into `frame` (`stream->cursor`).
    cursor: usize,
    /// Whether a frame is currently open (`opened_frame`).
    frame_open: bool,
    /// `numberOfContours` of the current glyph.
    n_contours: i16,
    /// The glyph's bounding box in font units (`loader->bbox`).
    bbox: BBox,
    /// The glyph's length in bytes (`loader->byte_len`).
    byte_len: usize,
    /// Left side bearing in font units (`loader->left_bearing`).
    left_bearing: i16,
    /// Advance width in font units (`loader->advance`).
    advance: u16,
    /// Top side bearing in font units (`loader->top_bearing`).
    top_bearing: i16,
    /// Advance height in font units (`loader->vadvance`).
    vadvance: u16,
    /// The device-independent horizontal advance (`loader->linear`).
    linear: i64,
    /// Whether `linear` has been captured from `hmtx` yet
    /// (`loader->linear_def`).
    linear_def: bool,
    /// Left side bearing phantom point (`loader->pp1`).
    pp1: Vector,
    /// Right side bearing phantom point (`loader->pp2`).
    pp2: Vector,
    /// Top side bearing phantom point (`loader->pp3`).
    pp3: Vector,
    /// Bottom side bearing phantom point (`loader->pp4`).
    pp4: Vector,
    /// The composite recursion list (`loader->composites`).
    composites: Vec<u32>,
    /// `glyph->format`: [`GlyphFormat::Composite`] only after an
    /// `FT_LOAD_NO_RECURSE` load.
    format: GlyphFormat,
}

/// `TT_Load_Glyph`: loads `glyph_index` of `font` into `slot`.
///
/// `size` supplies the 16.16 scales and the ppem values; it must
/// already be populated (FreeType's `ttmetrics.valid` guard is not
/// repeated here).  `slot` is cleared first, so a failed call leaves it
/// documented as "cleared, format [`GlyphFormat::Outline`]" rather than
/// partially written.
///
/// The slot's `internal` must own a [`GlyphLoader`]; it is taken for the
/// duration of the call and handed back afterwards, whether the load
/// succeeds or not.  The caller — the base layer — is responsible for
/// resolving dependent load flags (`FT_LOAD_NO_SCALE` implying
/// `FT_LOAD_NO_HINTING`), for filling `slot.advance` and for scaling
/// `slot.linear_hori_advance`/`slot.linear_vert_advance` to 16.16, all
/// of which `FT_Load_Glyph` does around the driver call.
///
/// # Errors
///
/// * [`TtError::INVALID_ARGUMENT`] — `load_flags` contains
///   `FT_LOAD_SBITS_ONLY`; this port has no embedded-bitmap path.
/// * [`TtError::INVALID_HANDLE`] — the slot has no `internal`/loader
///   (it was not created by an outline-capable driver).
/// * [`TtError::INVALID_GLYPH_INDEX`] — `glyph_index` is past
///   `maxp.numGlyphs`.
/// * [`TtError::INVALID_TABLE`] — `loca` describes a glyph but the
///   font has no `glyf` table.
/// * [`TtError::INVALID_OUTLINE`] — the glyph bytes are malformed
///   (truncated header, unordered contour ends, short coordinate run,
///   a `numberOfContours` that is neither positive nor `-1`, ...).
/// * [`TtError::INVALID_COMPOSITE`] — a composite component is out of
///   range, the component list is truncated, or the glyph recurses into
///   itself.
/// * [`TtError::TOO_MANY_HINTS`] — the instruction count runs past the
///   end of the glyph.
/// * [`TtError::ARRAY_TOO_LARGE`] — the glyph does not fit the
///   outline's point/contour limits.
/// * [`TtError::INVALID_STREAM_SEEK`] — `loca` points outside `glyf`.
///
/// # Panics
///
/// Never: every read is bounds-checked against the glyph frame.
pub fn load_glyph(
    font: &SfntFont<'_>,
    slot: &mut GlyphSlot,
    size: &SizeMetrics,
    glyph_index: u32,
    load_flags: LoadFlags,
) -> TtResult<()> {
    // `TT_Load_Glyph` resets the slot's image before the loader runs.
    slot.clear();
    slot.format = GlyphFormat::Outline;
    slot.outline.flags = 0;

    if load_flags.contains(LoadFlags::SBITS_ONLY) {
        return Err(TtError::INVALID_ARGUMENT);
    }

    // `tt_loader_init` takes the face's glyph loader; `ft_glyphslot_init`
    // guarantees that an outline-capable driver has one.
    let mut gloader = match slot.internal.as_mut() {
        Some(internal) => internal.loader.take(),
        None => None,
    }
    .ok_or(TtError::INVALID_HANDLE)?;
    // `FT_GlyphLoader_Rewind`.
    gloader.rewind();

    let mut loader = TtLoader::new(font, gloader, size, load_flags);
    let outcome = loader.load_truetype_glyph(glyph_index, 0);
    if outcome.is_ok() {
        loader.stage_image(slot);
        loader.compute_glyph_metrics(slot);
    }
    // Hand the loader back to the slot even when the load failed.
    let gloader = core::mem::take(&mut loader.gloader);
    if let Some(internal) = slot.internal.as_mut() {
        internal.loader = Some(gloader);
    }
    outcome?;
    // `TT_Load_Glyph` sets this bit after the load, on success or not.
    if !load_flags.contains(LoadFlags::NO_SCALE) && size.y_ppem < 24 {
        slot.outline.flags |= OUTLINE_HIGH_PRECISION;
    }
    Ok(())
}

impl<'a, 'b> TtLoader<'a, 'b> {
    /// Builds a fresh loader (`tt_loader_init` without the
    /// interpreter, the glyph table lookup and the loader rewind, all
    /// of which the caller has already performed).
    fn new(font: &'a SfntFont<'b>, gloader: GlyphLoader, size: &SizeMetrics, load_flags: LoadFlags) -> Self {
        // `FT_LOAD_NO_SCALE` forces the identity scale so that the
        // "scale only when not NO_SCALE" branches of the C code collapse
        // into a single unconditional step.
        let (x_scale, y_scale) = if load_flags.contains(LoadFlags::NO_SCALE) {
            (IDENTITY_SCALE, IDENTITY_SCALE)
        } else {
            (size.x_scale, size.y_scale)
        };
        TtLoader {
            font,
            glyf: font.table(TAG_GLYF),
            gloader,
            load_flags,
            x_scale,
            y_scale,
            frame: &[],
            cursor: 0,
            frame_open: false,
            n_contours: 0,
            bbox: BBox::new(),
            byte_len: 0,
            left_bearing: 0,
            advance: 0,
            top_bearing: 0,
            vadvance: 0,
            linear: 0,
            linear_def: false,
            pp1: Vector::new(0, 0),
            pp2: Vector::new(0, 0),
            pp3: Vector::new(0, 0),
            pp4: Vector::new(0, 0),
            composites: Vec::new(),
            format: GlyphFormat::Outline,
        }
    }

    /// `IS_HINTED(load_flags)`: bytecode would run, i.e. neither
    /// `FT_LOAD_NO_HINTING` nor `FT_LOAD_NO_SCALE` is set.
    #[inline]
    fn is_hinted(&self) -> bool {
        let flags = self.load_flags;
        !flags.contains(LoadFlags::NO_HINTING) && !flags.contains(LoadFlags::NO_SCALE)
    }

    /// Bytes left in the open glyph frame.
    #[inline]
    fn remaining(&self) -> usize {
        self.frame.len().saturating_sub(self.cursor)
    }

    /// `FT_NEXT_BYTE`.
    #[inline]
    fn read_u8(&mut self) -> Option<u8> {
        let frame = self.frame;
        let cursor = self.cursor;
        let &value = frame.get(cursor)?;
        self.cursor = cursor + 1;
        Some(value)
    }

    /// `FT_NEXT_USHORT`.
    #[inline]
    fn read_u16(&mut self) -> Option<u16> {
        let frame = self.frame;
        let cursor = self.cursor;
        let end = cursor.checked_add(2)?;
        let bytes = frame.get(cursor..end)?;
        self.cursor = end;
        Some(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    /// `FT_NEXT_SHORT`.
    #[inline]
    fn read_i16(&mut self) -> Option<i16> {
        self.read_u16().map(|value| value as i16)
    }

    /// `FT_NEXT_CHAR`.
    #[inline]
    fn read_i8(&mut self) -> Option<i8> {
        self.read_u8().map(|value| value as i8)
    }

    /// Drops `count` bytes (`p += n`), reporting whether they existed.
    #[inline]
    fn skip(&mut self, count: usize) -> bool {
        let Some(end) = self.cursor.checked_add(count) else {
            return false;
        };
        if end > self.frame.len() {
            return false;
        }
        self.cursor = end;
        true
    }

    /// `TT_Access_Glyph_Frame`: slices `glyf[offset .. offset + count]`
    /// and makes it the open frame.
    ///
    /// # Errors
    ///
    /// [`TtError::INVALID_TABLE`] when the font has no `glyf` table,
    /// [`TtError::INVALID_STREAM_SEEK`] when the range falls outside it.
    fn access_glyph_frame(&mut self, offset: usize, count: usize) -> TtResult<()> {
        let glyf = self.glyf.ok_or(TtError::INVALID_TABLE)?;
        let end = offset
            .checked_add(count)
            .ok_or(TtError::INVALID_STREAM_SEEK)?;
        let frame = glyf
            .get(offset..end)
            .ok_or(TtError::INVALID_STREAM_SEEK)?;
        self.frame = frame;
        self.cursor = 0;
        self.frame_open = true;
        Ok(())
    }

    /// `TT_Forget_Glyph_Frame`.
    #[inline]
    fn forget_glyph_frame(&mut self) {
        self.frame = &[];
        self.cursor = 0;
        self.frame_open = false;
    }

    /// `TT_Load_Glyph_Header`.
    ///
    /// # Errors
    ///
    /// [`TtError::INVALID_OUTLINE`] when fewer than ten bytes remain.
    fn load_glyph_header(&mut self) -> TtResult<()> {
        if self.remaining() < 10 {
            return Err(TtError::INVALID_OUTLINE);
        }
        self.n_contours = self.read_i16().ok_or(TtError::INVALID_OUTLINE)?;
        let x_min = self.read_i16().ok_or(TtError::INVALID_OUTLINE)?;
        let y_min = self.read_i16().ok_or(TtError::INVALID_OUTLINE)?;
        let x_max = self.read_i16().ok_or(TtError::INVALID_OUTLINE)?;
        let y_max = self.read_i16().ok_or(TtError::INVALID_OUTLINE)?;
        self.bbox = BBox::from_edges(
            i64::from(x_min),
            i64::from(y_min),
            i64::from(x_max),
            i64::from(y_max),
        );
        Ok(())
    }

    /// `tt_get_metrics_v_metrics` without a `vmtx`/`OS/2` reader: the
    /// `(top side bearing, advance height)` pair FreeType derives from
    /// `hhea`.
    fn v_metrics(&self, y_max: i64) -> (i16, u16) {
        let hhea = self.font.hhea();
        let ascender = i64::from(hhea.ascender);
        let descender = i64::from(hhea.descender);
        let top_bearing = (ascender - y_max) as i16;
        let advance_height = (ascender - descender).unsigned_abs() as u16;
        (top_bearing, advance_height)
    }

    /// `tt_get_metrics`: fills the horizontal and emulated vertical
    /// metrics of `glyph_index` and captures `loader->linear`.
    fn get_metrics(&mut self, glyph_index: u32) -> TtResult<()> {
        let (advance_width, left_bearing) = self.font.metrics_for_glyph(glyph_index);
        let (top_bearing, advance_height) = self.v_metrics(self.bbox.y_max);

        self.left_bearing = left_bearing;
        self.advance = advance_width;
        self.top_bearing = top_bearing;
        self.vadvance = advance_height;

        if !self.linear_def {
            self.linear_def = true;
            self.linear = i64::from(advance_width);
        }
        Ok(())
    }

    /// `tt_loader_set_pp`: places the four phantom points from the
    /// glyph bbox and its metrics.
    ///
    /// `use_aw_2` is hard-wired to `false`: without the subpixel
    /// hinting machinery FreeType also uses formula (3) here, so
    /// `pp3.x`/`pp4.x` stay zero.
    fn set_pp(&mut self) {
        self.pp1 = Vector::new(self.bbox.x_min - i64::from(self.left_bearing), 0);
        self.pp2 = Vector::new(self.pp1.x + i64::from(self.advance), 0);
        self.pp3 = Vector::new(0, self.bbox.y_max + i64::from(self.top_bearing));
        self.pp4 = Vector::new(0, self.pp3.y - i64::from(self.vadvance));
    }

    /// Scales the phantom points with the loader's effective scales
    /// (an identity under `FT_LOAD_NO_SCALE`), covering every branch
    /// where the C code guards the step with `!FT_LOAD_NO_SCALE`.
    fn scale_phantoms(&mut self) {
        self.pp1.x = mul_fix(self.pp1.x, self.x_scale);
        self.pp2.x = mul_fix(self.pp2.x, self.x_scale);
        self.pp3.x = mul_fix(self.pp3.x, self.x_scale);
        self.pp4.x = mul_fix(self.pp4.x, self.x_scale);
        self.pp3.y = mul_fix(self.pp3.y, self.y_scale);
        self.pp4.y = mul_fix(self.pp4.y, self.y_scale);
    }

    /// `TT_Hint_Glyph`.
    ///
    /// With the bytecode interpreter compiled out the C function keeps
    /// only the "round the phantom points" step and the write-back of
    /// `loader->pp1..pp4`; the interpreter would additionally run the
    /// glyph program over the zone and store the drop-out mode in
    /// `tags[0]`.
    fn hint_glyph(&mut self) {
        self.pp1.x = pix_round(self.pp1.x);
        self.pp2.x = pix_round(self.pp2.x);
        self.pp3.y = pix_round(self.pp3.y);
        self.pp4.y = pix_round(self.pp4.y);
    }

    /// `TT_Load_Simple_Glyph`: decodes contours, instructions, flags
    /// and coordinates into the current image of the glyph loader.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_OUTLINE`] — the contour ends are not
    ///   strictly increasing, a run of flags or coordinates is
    ///   truncated, or a reserved count does not fit the table.
    /// * [`TtError::TOO_MANY_HINTS`] — the instruction count runs past
    ///   the end of the glyph.
    /// * [`TtError::ARRAY_TOO_LARGE`] — the glyph exceeds the loader's
    ///   point/contour limits.
    fn load_simple_glyph(&mut self) -> TtResult<()> {
        let n_contours = self.n_contours as usize;

        // `FT_GLYPHLOADER_CHECK_POINTS(gloader, 0, n_contours)`.
        self.gloader.check_points(0, n_contours)?;

        // `n_contours >= 0xFFF || p + (n_contours + 1) * 2 > limit`.
        if n_contours >= 0xFFF || self.remaining() < (n_contours + 1) * 2 {
            return Err(TtError::INVALID_OUTLINE);
        }

        let mut ends = Vec::with_capacity(n_contours);
        let mut previous = self.read_i16().ok_or(TtError::INVALID_OUTLINE)?;
        if previous < 0 {
            return Err(TtError::INVALID_OUTLINE);
        }
        ends.push(previous);
        for _ in 1..n_contours {
            let end = self.read_i16().ok_or(TtError::INVALID_OUTLINE)?;
            if end <= previous {
                // unordered contours: this is invalid
                return Err(TtError::INVALID_OUTLINE);
            }
            previous = end;
            ends.push(end);
        }
        let n_points = (i32::from(previous) + 1) as usize;

        // The C call is `CHECK_POINTS(gloader, n_points + 4, 0)`: the
        // four extra slots are the phantom points.
        self.gloader.check_points(n_points + 4, 0)?;

        let n_ins = self.read_u16().ok_or(TtError::INVALID_OUTLINE)?;
        if self.remaining() < usize::from(n_ins) {
            log_warn!("TT_Load_Simple_Glyph: instruction count mismatch");
            return Err(TtError::TOO_MANY_HINTS);
        }
        // The bytecode interpreter is compiled out, so the instructions
        // are validated and skipped instead of copied into `glyphIns`.
        if !self.skip(usize::from(n_ins)) {
            return Err(TtError::INVALID_OUTLINE);
        }

        let flags = self.read_point_flags(n_points)?;
        let (xs, ys) = self.read_point_coordinates(&flags)?;

        // The contour ends are relative to the current image.  Creating
        // the contours first keeps the loader's own "previous end ==
        // last point" bookkeeping from leaking into the final values,
        // so they are overwritten here, after every contour exists.
        for _ in 0..n_contours {
            self.gloader.add_contour()?;
        }
        let base_contours = self.gloader.base_outline().n_contours as usize;
        {
            let outline = self.gloader.base_outline_mut();
            for (index, end) in ends.iter().enumerate() {
                let slot = outline
                    .contours
                    .get_mut(base_contours + index)
                    .ok_or(TtError::INVALID_OUTLINE)?;
                *slot = *end;
            }
        }

        for index in 0..n_points {
            self.gloader
                .add_point_tagged(xs[index], ys[index], flags[index] & CURVE_TAG_ON)?;
        }
        Ok(())
    }

    /// Reads the packed point-flag run of a simple glyph (bit 3 is
    /// `REPEAT`, i.e. the next byte repeats the flag).
    ///
    /// # Errors
    ///
    /// [`TtError::INVALID_OUTLINE`] on a truncated run or a repeat
    /// count that would overrun `n_points`.
    fn read_point_flags(&mut self, n_points: usize) -> TtResult<Vec<u8>> {
        let mut flags: Vec<u8> = Vec::with_capacity(n_points);
        while flags.len() < n_points {
            let flag = self.read_u8().ok_or(TtError::INVALID_OUTLINE)?;
            flags.push(flag);
            if flag & 0x08 != 0 {
                let repeat = usize::from(self.read_u8().ok_or(TtError::INVALID_OUTLINE)?);
                if flags.len() + repeat > n_points {
                    return Err(TtError::INVALID_OUTLINE);
                }
                for _ in 0..repeat {
                    flags.push(flag);
                }
            }
        }
        Ok(flags)
    }

    /// Reads the delta-encoded x and y runs of a simple glyph and turns
    /// them into absolute coordinates (`xShortVector`/`yShortVector`
    /// with their "same/positive" bits).
    ///
    /// # Errors
    ///
    /// [`TtError::INVALID_OUTLINE`] when a run is truncated.
    fn read_point_coordinates(&mut self, flags: &[u8]) -> TtResult<(Vec<i64>, Vec<i64>)> {
        let mut xs = Vec::with_capacity(flags.len());
        let mut x = 0i64;
        for &flag in flags {
            let delta = if flag & 0x02 != 0 {
                let byte = i64::from(self.read_u8().ok_or(TtError::INVALID_OUTLINE)?);
                if flag & 0x10 == 0 { -byte } else { byte }
            } else if flag & 0x10 == 0 {
                i64::from(self.read_i16().ok_or(TtError::INVALID_OUTLINE)?)
            } else {
                0
            };
            x = x.wrapping_add(delta);
            xs.push(x);
        }

        let mut ys = Vec::with_capacity(flags.len());
        let mut y = 0i64;
        for &flag in flags {
            let delta = if flag & 0x04 != 0 {
                let byte = i64::from(self.read_u8().ok_or(TtError::INVALID_OUTLINE)?);
                if flag & 0x20 == 0 { -byte } else { byte }
            } else if flag & 0x20 == 0 {
                i64::from(self.read_i16().ok_or(TtError::INVALID_OUTLINE)?)
            } else {
                0
            };
            y = y.wrapping_add(delta);
            ys.push(y);
        }
        Ok((xs, ys))
    }

    /// `TT_Load_Composite_Glyph`: appends every component of the
    /// composite to the current image of the glyph loader.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_COMPOSITE`] — the component list is
    ///   truncated.
    /// * [`TtError::ARRAY_TOO_LARGE`] — the subglyph table cannot grow.
    fn load_composite_glyph(&mut self) -> TtResult<()> {
        loop {
            let flags = self
                .read_u16()
                .ok_or(TtError::INVALID_COMPOSITE)?;
            let index = self
                .read_u16()
                .ok_or(TtError::INVALID_COMPOSITE)?;

            let mut bytes = 2usize;
            if flags & SUBGLYPH_FLAG_ARGS_ARE_WORDS != 0 {
                bytes += 2;
            }
            if flags & SUBGLYPH_FLAG_SCALE != 0 {
                bytes += 2;
            } else if flags & SUBGLYPH_FLAG_XY_SCALE != 0 {
                bytes += 4;
            } else if flags & SUBGLYPH_FLAG_2X2 != 0 {
                bytes += 8;
            }
            if self.remaining() < bytes {
                return Err(TtError::INVALID_COMPOSITE);
            }

            let words = flags & SUBGLYPH_FLAG_ARGS_ARE_WORDS != 0;
            let xy_values = flags & SUBGLYPH_FLAG_ARGS_ARE_XY_VALUES != 0;
            let (arg1, arg2) = if xy_values {
                if words {
                    (
                        i64::from(
                            self.read_i16()
                                .ok_or(TtError::INVALID_COMPOSITE)?,
                        ),
                        i64::from(
                            self.read_i16()
                                .ok_or(TtError::INVALID_COMPOSITE)?,
                        ),
                    )
                } else {
                    (
                        i64::from(self.read_i8().ok_or(TtError::INVALID_COMPOSITE)?),
                        i64::from(self.read_i8().ok_or(TtError::INVALID_COMPOSITE)?),
                    )
                }
            } else if words {
                (
                    i64::from(
                        self.read_u16()
                            .ok_or(TtError::INVALID_COMPOSITE)?,
                    ),
                    i64::from(
                        self.read_u16()
                            .ok_or(TtError::INVALID_COMPOSITE)?,
                    ),
                )
            } else {
                (
                    i64::from(self.read_u8().ok_or(TtError::INVALID_COMPOSITE)?),
                    i64::from(self.read_u8().ok_or(TtError::INVALID_COMPOSITE)?),
                )
            };

            let mut transform = Matrix::IDENTITY;
            if flags & SUBGLYPH_FLAG_SCALE != 0 {
                let scale = i64::from(
                    self.read_i16()
                        .ok_or(TtError::INVALID_COMPOSITE)?,
                ) * 4;
                transform.xx = scale;
                transform.yy = scale;
            } else if flags & SUBGLYPH_FLAG_XY_SCALE != 0 {
                transform.xx = i64::from(
                    self.read_i16()
                        .ok_or(TtError::INVALID_COMPOSITE)?,
                ) * 4;
                transform.yy = i64::from(
                    self.read_i16()
                        .ok_or(TtError::INVALID_COMPOSITE)?,
                ) * 4;
            } else if flags & SUBGLYPH_FLAG_2X2 != 0 {
                transform.xx = i64::from(
                    self.read_i16()
                        .ok_or(TtError::INVALID_COMPOSITE)?,
                ) * 4;
                transform.yx = i64::from(
                    self.read_i16()
                        .ok_or(TtError::INVALID_COMPOSITE)?,
                ) * 4;
                transform.xy = i64::from(
                    self.read_i16()
                        .ok_or(TtError::INVALID_COMPOSITE)?,
                ) * 4;
                transform.yy = i64::from(
                    self.read_i16()
                        .ok_or(TtError::INVALID_COMPOSITE)?,
                ) * 4;
            }

            self.gloader.add_subglyph(SubGlyph {
                index: i32::from(index),
                flags,
                arg1: arg1 as i32,
                arg2: arg2 as i32,
                transform,
            })?;

            if flags & SUBGLYPH_FLAG_MORE_COMPONENTS == 0 {
                break;
            }
        }
        // `loader->ins_pos` (the offset of the composite's own
        // instructions) is only read back by the bytecode interpreter,
        // so it is not recorded here.
        Ok(())
    }

    /// `TT_Process_Simple_Glyph`: scales the current image, scales the
    /// phantom points and rounds them when hinted.
    ///
    /// # Errors
    ///
    /// [`TtError::INVALID_OUTLINE`] when the current image does not fit
    /// the loader's point table.
    fn process_simple_glyph(&mut self) -> TtResult<()> {
        let start = self.gloader.base_outline().n_points as usize;
        let end = start + self.gloader.current_point_count();
        let (x_scale, y_scale) = (self.x_scale, self.y_scale);

        let outline = self.gloader.base_outline_mut();
        let points = outline
            .points
            .get_mut(start..end)
            .ok_or(TtError::INVALID_OUTLINE)?;
        for point in points {
            point.x = mul_fix(point.x, x_scale);
            point.y = mul_fix(point.y, y_scale);
        }

        self.scale_phantoms();
        if self.is_hinted() {
            self.hint_glyph();
        }
        Ok(())
    }

    /// `TT_Process_Composite_Component`: transforms the component that
    /// was just loaded, derives its offset and slides it into place.
    ///
    /// `start_point` is `base.outline.n_points` at the beginning of the
    /// composite, `num_base_points` the same counter just before this
    /// component was appended.
    ///
    /// # Errors
    ///
    /// * [`TtError::INVALID_COMPOSITE`] — point-matching indices fall
    ///   outside the points loaded so far.
    /// * [`TtError::INVALID_OUTLINE`] — the component's points are
    ///   missing from the loader.
    fn process_composite_component(
        &mut self,
        subglyph: &SubGlyph,
        start_point: u32,
        num_base_points: u32,
    ) -> TtResult<()> {
        let have_scale =
            subglyph.flags & (SUBGLYPH_FLAG_SCALE | SUBGLYPH_FLAG_XY_SCALE | SUBGLYPH_FLAG_2X2) != 0;

        // FreeType transforms the component first: the point matching
        // below must see the transformed coordinates.
        if have_scale {
            let outline = self.gloader.base_outline_mut();
            let points = outline
                .points
                .get_mut(num_base_points as usize..)
                .ok_or(TtError::INVALID_OUTLINE)?;
            let _ = transform_points(points, &subglyph.transform);
        }

        let (x, y) = if subglyph.flags & SUBGLYPH_FLAG_ARGS_ARE_XY_VALUES != 0 {
            let (mut x, mut y) = (i64::from(subglyph.arg1), i64::from(subglyph.arg2));
            if x == 0 && y == 0 {
                return Ok(());
            }

            // `TT_CONFIG_OPTION_COMPONENT_OFFSET_SCALED` is undefined,
            // so the offset is scaled only when the font asks for it.
            if have_scale && subglyph.flags & SUBGLYPH_FLAG_SCALED_COMPONENT_OFFSET != 0 {
                let mac_xscale = hypot(subglyph.transform.xx, subglyph.transform.xy);
                let mac_yscale = hypot(subglyph.transform.yy, subglyph.transform.yx);
                x = mul_fix(x, mac_xscale);
                y = mul_fix(y, mac_yscale);
            }

            if !self.load_flags.contains(LoadFlags::NO_SCALE) {
                x = mul_fix(x, self.x_scale);
                y = mul_fix(y, self.y_scale);
                if subglyph.flags & SUBGLYPH_FLAG_ROUND_XY_TO_GRID != 0 {
                    x = pix_round(x);
                    y = pix_round(y);
                }
            }
            (x, y)
        } else {
            // Match the l-th point of this component to the k-th point
            // of the components loaded so far.
            let k = u32::try_from(subglyph.arg1).map_err(|_| TtError::INVALID_COMPOSITE)?;
            let l = u32::try_from(subglyph.arg2).map_err(|_| TtError::INVALID_COMPOSITE)?;
            let k = k
                .checked_add(start_point)
                .ok_or(TtError::INVALID_COMPOSITE)?;
            let l = l
                .checked_add(num_base_points)
                .ok_or(TtError::INVALID_COMPOSITE)?;
            let num_points = self.gloader.base_outline().n_points as u32;
            if k >= num_base_points || l >= num_points {
                return Err(TtError::INVALID_COMPOSITE);
            }
            let outline = self.gloader.base_outline();
            let p1 = outline
                .points
                .get(k as usize)
                .copied()
                .ok_or(TtError::INVALID_COMPOSITE)?;
            let p2 = outline
                .points
                .get(l as usize)
                .copied()
                .ok_or(TtError::INVALID_COMPOSITE)?;
            (p1.x - p2.x, p1.y - p2.y)
        };

        if x != 0 || y != 0 {
            let outline = self.gloader.base_outline_mut();
            let points = outline
                .points
                .get_mut(num_base_points as usize..)
                .ok_or(TtError::INVALID_OUTLINE)?;
            let _ = translate_points(points, x, y);
        }
        Ok(())
    }

    /// `TT_Process_Composite_Glyph` with the interpreter removed: the
    /// only remaining job is rounding the phantom points, guarded by
    /// `IS_HINTED` and `num_points > start_point` at the call site.
    fn process_composite_glyph(&mut self) {
        self.hint_glyph();
    }

    /// `load_truetype_glyph`: the recursion driver.  Closes the glyph
    /// frame on every exit, including error paths.
    ///
    /// # Errors
    ///
    /// Whatever [`TtLoader::load_glyph_data`] or a nested component
    /// reports; see [`load_glyph`].
    fn load_truetype_glyph(&mut self, glyph_index: u32, recurse_count: u32) -> TtResult<()> {
        let outcome = self.load_glyph_data(glyph_index, recurse_count);
        if self.frame_open {
            self.forget_glyph_frame();
        }
        outcome
    }

    /// The body of [`TtLoader::load_truetype_glyph`].
    ///
    /// # Errors
    ///
    /// See [`load_glyph`].
    fn load_glyph_data(&mut self, glyph_index: u32, recurse_count: u32) -> TtResult<()> {
        // Some fonts have an incorrect `maxComponentDepth`; depth 1 is
        // always allowed so that the majority of them still load.
        if recurse_count > 1 && recurse_count > u32::from(self.font.maxp().max_component_depth) {
            return Err(TtError::INVALID_COMPOSITE);
        }
        if glyph_index >= u32::from(self.font.num_glyphs()) {
            return Err(TtError::INVALID_GLYPH_INDEX);
        }

        let (start, end) = self
            .font
            .glyph_location(glyph_index)
            .unwrap_or((0, 0));
        self.byte_len = end.saturating_sub(start);

        if self.byte_len > 0 {
            self.access_glyph_frame(start, self.byte_len)?;
            self.load_glyph_header()?;
            // The metrics must be computed after the header because the
            // emulated vertical metrics need `yMax`.
            self.get_metrics(glyph_index)?;
        }

        if self.byte_len == 0 || self.n_contours == 0 {
            self.bbox = BBox::new();
            self.get_metrics(glyph_index)?;
            self.set_pp();
            self.scale_phantoms();
            return Ok(());
        }

        self.set_pp();

        if self.n_contours > 0 {
            self.load_simple_glyph()?;
            self.forget_glyph_frame();
            self.process_simple_glyph()?;
            self.gloader.add();
            return Ok(());
        }

        if self.n_contours == -1 {
            return self.load_composite_data(glyph_index, recurse_count);
        }

        // invalid composite count (negative but not -1)
        Err(TtError::INVALID_OUTLINE)
    }

    /// The `n_contours == -1` branch of
    /// [`TtLoader::load_glyph_data`].
    ///
    /// # Errors
    ///
    /// See [`load_glyph`].
    fn load_composite_data(&mut self, glyph_index: u32, recurse_count: u32) -> TtResult<()> {
        // Clear the nodes filled by sibling chains, then look for a
        // cycle.  The C list keeps one entry per nesting level and
        // re-uses the entry at `recurse_count`, which truncating the
        // vector up to that index reproduces exactly.
        self.composites.truncate(recurse_count as usize);
        if self.composites.contains(&glyph_index) {
            log_warn!("TT_Load_Composite_Glyph: infinite recursion detected");
            return Err(TtError::INVALID_COMPOSITE);
        }
        self.composites.push(glyph_index);

        let start_point = self.gloader.base_outline().n_points as u32;

        self.load_composite_glyph()?;
        self.forget_glyph_frame();

        self.scale_phantoms();

        // `FT_LOAD_NO_RECURSE`: hand the component list back untouched.
        if self.load_flags.contains(LoadFlags::NO_RECURSE) {
            self.gloader.add();
            self.format = GlyphFormat::Composite;
            return Ok(());
        }

        let num_base_subgs = self.gloader.base_subglyphs().len();
        let num_subglyphs = self.gloader.current_subglyph_count();
        self.gloader.add();

        let mut num_points = start_point;
        for n in 0..num_subglyphs {
            let subglyph = self.subglyph_at(num_base_subgs + n)?;
            let saved = (self.pp1, self.pp2, self.pp3, self.pp4);

            let num_base_points = self.gloader.base_outline().n_points as u32;
            self.load_truetype_glyph(subglyph.index as u32, recurse_count + 1)?;
            // The subglyph table may have been reallocated by the
            // nested load, so the entry is fetched again.
            let subglyph = self.subglyph_at(num_base_subgs + n)?;

            if subglyph.flags & SUBGLYPH_FLAG_USE_MY_METRICS == 0 {
                self.pp1 = saved.0;
                self.pp2 = saved.1;
                self.pp3 = saved.2;
                self.pp4 = saved.3;
            }

            num_points = self.gloader.base_outline().n_points as u32;
            if num_points == num_base_points {
                continue;
            }
            self.process_composite_component(&subglyph, start_point, num_base_points)?;
        }

        if self.is_hinted() && num_points > start_point {
            self.process_composite_glyph();
        }
        Ok(())
    }

    /// Copies subglyph `index` out of the loader's base image.
    ///
    /// # Errors
    ///
    /// [`TtError::INVALID_COMPOSITE`] when the index is out of range.
    fn subglyph_at(&self, index: usize) -> TtResult<SubGlyph> {
        self.gloader
            .base_subglyphs()
            .get(index)
            .copied()
            .ok_or(TtError::INVALID_COMPOSITE)
    }

    /// Publishes the loaded image into `slot`
    /// (`glyph->outline = loader.gloader->base.outline` plus the
    /// `FT_Outline_Translate(-pp1.x, 0)` that moves the origin).
    ///
    /// The clone is trimmed to the outline's own counters: FreeType's
    /// `FT_Outline_Get_CBox` and `FT_Outline_Translate` honour
    /// `n_points`, while the Rust helpers walk the whole slice, so an
    /// empty glyph loaded after a full one would otherwise inherit the
    /// previous glyph's box.
    fn stage_image(&self, slot: &mut GlyphSlot) {
        slot.format = self.format;
        if self.format == GlyphFormat::Composite {
            // `FT_LOAD_NO_RECURSE`: only the component list is published.
            slot.outline.flags = 0;
            slot.subglyphs = self.gloader.base_subglyphs().to_vec();
            return;
        }

        let mut outline = self.gloader.base_outline().clone();
        let n_points = outline.n_points.max(0) as usize;
        let n_contours = outline.n_contours.max(0) as usize;
        if outline.points.len() > n_points {
            outline.points.truncate(n_points);
        }
        if outline.tags.len() > n_points {
            outline.tags.truncate(n_points);
        }
        if outline.contours.len() > n_contours {
            outline.contours.truncate(n_contours);
        }

        outline.flags &= !OUTLINE_SINGLE_PASS;
        if self.pp1.x != 0 && !outline.points.is_empty() {
            let _ = translate_points(&mut outline.points, -self.pp1.x, 0);
        }
        slot.outline = outline;
    }

    /// `compute_glyph_metrics`: fills `slot.metrics` and the two
    /// device-independent advances.
    fn compute_glyph_metrics(&self, slot: &mut GlyphSlot) {
        // Only an `FT_LOAD_NO_RECURSE` load leaves `format` set to
        // `Composite`; that path has no outline, so the header's
        // font-unit bbox is used instead, exactly as in C.
        let bbox = if self.format == GlyphFormat::Composite {
            self.bbox
        } else {
            slot.outline.get_cbox()
        };

        slot.linear_hori_advance = self.linear;
        slot.metrics.hori_bearing_x = bbox.x_min;
        slot.metrics.hori_bearing_y = bbox.y_max;
        slot.metrics.hori_advance = self.pp2.x - self.pp1.x;
        slot.metrics.width = bbox.x_max - bbox.x_min;
        slot.metrics.height = bbox.y_max - bbox.y_min;

        // Vertical metrics: no `vmtx`/`OS/2` reader yet, so FreeType's
        // `hhea` fallback is always taken.
        let height = div_fix(bbox.y_max - bbox.y_min, self.y_scale) as i16;
        let hhea = self.font.hhea();
        let advance = i64::from(hhea.ascender) - i64::from(hhea.descender);
        let top = (advance - i64::from(height)) / 2;

        slot.linear_vert_advance = advance;
        let top = mul_fix(top, self.y_scale);
        let advance = mul_fix(advance, self.y_scale);

        slot.metrics.vert_bearing_x = slot.metrics.hori_bearing_x - slot.metrics.hori_advance / 2;
        slot.metrics.vert_bearing_y = top;
        slot.metrics.vert_advance = advance;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;

    /// Runs `run` with a fresh loader over [`test_util::default_font`].
    ///
    /// The fixture bytes and the [`SfntFont`] live inside the closure's
    /// scope, so a frame literal declared *before* the call outlives
    /// the loader's `'b` parameter and can be installed in it.
    fn with_loader<R>(flags: LoadFlags, run: impl FnOnce(&mut TtLoader<'_, '_>) -> R) -> R {
        let bytes = test_util::default_font();
        let font = test_util::open(&bytes).unwrap();
        let mut loader = TtLoader::new(&font, GlyphLoader::new(), &test_util::no_scale_size(), flags);
        run(&mut loader)
    }

    #[test]
    fn load_glyph_header_reads_the_bbox() {
        let bytes = test_util::default_font();
        let font = test_util::open(&bytes).unwrap();
        let mut loader = TtLoader::new(
            &font,
            GlyphLoader::new(),
            &test_util::no_scale_size(),
            LoadFlags::NO_SCALE,
        );
        let (start, end) = font.glyph_location(1).unwrap();
        loader
            .access_glyph_frame(start, end - start)
            .unwrap();
        loader.load_glyph_header().unwrap();
        assert!(loader.frame_open);
        assert_eq!(loader.n_contours, 1);
        assert_eq!(loader.bbox, BBox::from_edges(0, 0, 500, 700));
    }

    #[test]
    fn frame_reads_stop_at_the_end_of_the_frame() {
        with_loader(LoadFlags::NO_SCALE, |loader| {
            loader.access_glyph_frame(0, 4).unwrap();
            assert_eq!(loader.remaining(), 4);
            assert_eq!(loader.read_u16(), Some(1));
            assert_eq!(loader.read_u16(), Some(0));
            assert_eq!(loader.read_u8(), None);
            assert_eq!(loader.read_u16(), None);
            assert_eq!(loader.read_i16(), None);
            assert_eq!(loader.read_i8(), None);
            assert!(!loader.skip(1));
            assert_eq!(loader.cursor, 4);
            assert!(loader.access_glyph_frame(usize::MAX, 1).is_err());
            loader.forget_glyph_frame();
            assert!(!loader.frame_open);
            assert_eq!(loader.remaining(), 0);
        });
    }

    #[test]
    fn signed_reads_reinterpret_the_big_endian_words() {
        // Declared first so that the slice outlives the loader's `'b`.
        let frame = [0xFFu8, 0x80, 0x01];
        let bytes = test_util::default_font();
        let font = test_util::open(&bytes).unwrap();
        let mut loader = TtLoader::new(
            &font,
            GlyphLoader::new(),
            &test_util::no_scale_size(),
            LoadFlags::NO_SCALE,
        );
        loader.frame = &frame;
        loader.cursor = 0;
        loader.frame_open = true;
        assert_eq!(loader.read_i8(), Some(-1));
        assert_eq!(loader.read_i16(), Some(-32_767));
        assert_eq!(loader.read_u8(), None);
    }

    #[test]
    fn short_headers_are_rejected_by_the_header_reader() {
        with_loader(LoadFlags::NO_SCALE, |loader| {
            loader.access_glyph_frame(0, 9).unwrap();
            assert_eq!(loader.load_glyph_header(), Err(TtError::INVALID_OUTLINE));
        });
    }

    #[test]
    fn point_flags_expand_the_repeat_run() {
        let frame = [0x09u8, 0x03, 0x04, 0x00];
        let bytes = test_util::default_font();
        let font = test_util::open(&bytes).unwrap();
        let mut loader = TtLoader::new(
            &font,
            GlyphLoader::new(),
            &test_util::no_scale_size(),
            LoadFlags::NO_SCALE,
        );
        loader.frame = &frame;
        loader.cursor = 0;
        loader.frame_open = true;
        let flags = loader.read_point_flags(5).unwrap();
        assert_eq!(flags, vec![0x09, 0x09, 0x09, 0x09, 0x04]);

        loader.cursor = 0;
        loader.frame = &frame[..1];
        assert!(loader.read_point_flags(2).is_err());

        loader.cursor = 0;
        loader.frame = &frame;
        assert_eq!(loader.read_point_flags(3), Err(TtError::INVALID_OUTLINE));
    }

    #[test]
    fn coordinates_cover_same_short_and_long_deltas() {
        // Only the coordinate deltas live in the frame; the flags are
        // handed to `read_point_coordinates` separately.
        let frame = [100u8, 0x03, 0xE8, 50, 0x02, 0xBC];
        let flags = [0x30u8, 0x16, 0x00];
        let bytes = test_util::default_font();
        let font = test_util::open(&bytes).unwrap();
        let mut loader = TtLoader::new(
            &font,
            GlyphLoader::new(),
            &test_util::no_scale_size(),
            LoadFlags::NO_SCALE,
        );
        loader.frame = &frame;
        loader.cursor = 0;
        loader.frame_open = true;
        let (xs, ys) = loader.read_point_coordinates(&flags).unwrap();
        assert_eq!(xs, vec![0, 100, 1100]);
        assert_eq!(ys, vec![0, -50, 650]);

        loader.frame = &frame[..1];
        loader.cursor = 0;
        assert_eq!(
            loader.read_point_coordinates(&[0x00]),
            Err(TtError::INVALID_OUTLINE)
        );
    }

    #[test]
    fn hinting_depends_on_both_scale_and_hinting_flags() {
        with_loader(LoadFlags::DEFAULT, |loader| {
            assert!(loader.is_hinted());
            loader.load_flags = LoadFlags::RENDER;
            assert!(loader.is_hinted());
            loader.load_flags = LoadFlags::NO_HINTING;
            assert!(!loader.is_hinted());
            loader.load_flags = LoadFlags::NO_SCALE;
            assert!(!loader.is_hinted());
            loader.load_flags = LoadFlags::NO_SCALE | LoadFlags::NO_HINTING;
            assert!(!loader.is_hinted());
        });
    }

    #[test]
    fn phantom_points_follow_the_hhea_fallback_metrics() {
        with_loader(LoadFlags::NO_SCALE, |loader| {
            assert_eq!(loader.v_metrics(700), (100, 1000));
            assert_eq!(loader.v_metrics(0), (800, 1000));

            loader.bbox = BBox::from_edges(0, 0, 500, 700);
            loader.left_bearing = -7;
            loader.advance = 510;
            loader.top_bearing = 100;
            loader.vadvance = 1000;
            loader.set_pp();
            assert_eq!(loader.pp1, Vector::new(7, 0));
            assert_eq!(loader.pp2, Vector::new(517, 0));
            assert_eq!(loader.pp3, Vector::new(0, 800));
            assert_eq!(loader.pp4, Vector::new(0, -200));
        });
    }

    #[test]
    fn phantom_scaling_is_identity_under_no_scale() {
        with_loader(LoadFlags::NO_SCALE, |loader| {
            loader.pp1 = Vector::new(7, 0);
            loader.pp2 = Vector::new(517, 0);
            loader.pp3 = Vector::new(0, 800);
            loader.pp4 = Vector::new(0, -200);
            loader.scale_phantoms();
            assert_eq!(loader.pp1, Vector::new(7, 0));
            assert_eq!(loader.pp2, Vector::new(517, 0));
            assert_eq!(loader.pp3, Vector::new(0, 800));
            assert_eq!(loader.pp4, Vector::new(0, -200));

            loader.x_scale = 0x8000;
            loader.y_scale = 0x8000;
            loader.scale_phantoms();
            assert_eq!(loader.pp1, Vector::new(4, 0));
            assert_eq!(loader.pp2, Vector::new(259, 0));
            assert_eq!(loader.pp3, Vector::new(0, 400));
            assert_eq!(loader.pp4, Vector::new(0, -100));
        });
    }

    #[test]
    fn stage_image_drops_the_storage_of_the_previous_glyph() {
        with_loader(LoadFlags::NO_SCALE, |loader| {
            let mut slot = test_util::new_slot();

            loader.load_truetype_glyph(1, 0).unwrap();
            assert_eq!(loader.gloader.base_outline().n_points, 4);
            loader.stage_image(&mut slot);
            assert_eq!(slot.outline.points.len(), 4);
            assert!(slot.outline.check().is_ok());

            // `load_glyph` rewinds the shared loader between two glyphs;
            // glyph 1's storage is still allocated behind
            // `n_points == 0` and must not leak into the empty glyph.
            loader.gloader.rewind();
            loader.load_truetype_glyph(0, 0).unwrap();
            loader.stage_image(&mut slot);
            assert_eq!(slot.outline.n_points, 0);
            assert!(slot.outline.points.is_empty());
            assert!(slot.outline.check().is_ok());

            loader.compute_glyph_metrics(&mut slot);
            assert_eq!(slot.metrics.hori_bearing_y, 0);
            assert_eq!(slot.metrics.height, 0);
        });
    }
}
