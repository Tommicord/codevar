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

//! `FT_Outline_Decompose` (FreeType 2.6, `src/base/ftoutln.c:51`).
//!
//! Walks every contour of an [`Outline`] and reports it as a sequence of
//! `move_to`, `line_to`, `conic_to`, and `cubic_to` calls, exactly as the
//! FreeType outline decomposer does.  The scaling transform (`shift` and
//! `delta` of `FT_Outline_Funcs`) is the identity, matching the smooth
//! rasterizer which always calls `FT_Outline_Decompose` with
//! `shift = 0, delta = 0` (`ftgrays.c` interface setup).

use codevar_truetype_core::{Outline, TtError, TtResult, Vector};

/// The curve tag mask of `FT_CURVE_TAG` (`flag & 3`).
const CURVE_TAG_MASK: u8 = 0x03;
/// `FT_CURVE_TAG_ON` — the point is on the curve.
const CURVE_TAG_ON: u8 = 0x01;
/// `FT_CURVE_TAG_CONIC` — the point is a quadratic control point.
const CURVE_TAG_CONIC: u8 = 0x00;
/// `FT_CURVE_TAG_CUBIC` — the point is a cubic control point.
const CURVE_TAG_CUBIC: u8 = 0x02;

/// Receives the flattened path events produced by [`decompose`].
///
/// Mirrors `FT_Outline_Funcs` from FreeType 2.6; every method returns a
/// [`TtResult`] so a consumer (for example the gray rasterizer's worker)
/// can abort the traversal when it runs out of memory.
pub trait Decomposer {
    /// Reports the start point of a contour (`move_to`).
    fn move_to(&mut self, to: &Vector) -> TtResult<()>;

    /// Reports a straight segment (`line_to`).
    fn line_to(&mut self, to: &Vector) -> TtResult<()>;

    /// Reports a quadratic Bézier segment (`conic_to`).
    fn conic_to(&mut self, control: &Vector, to: &Vector) -> TtResult<()>;

    /// Reports a cubic Bézier segment (`cubic_to`).
    fn cubic_to(&mut self, control1: &Vector, control2: &Vector, to: &Vector) -> TtResult<()>;
}

/// Faithful port of `FT_Outline_Decompose` (FreeType 2.6, `ftoutln.c:51`).
///
/// # Errors
///
/// Returns [`TtError::INVALID_OUTLINE`] when a contour end index is
/// negative or out of range, when a contour starts with a cubic control
/// point, or when a conic arc is followed by a non-conic/non-on point —
/// the exact `Invalid_Outline` paths of the C routine.  Additional bounds
/// checks defend against contours whose indices run past the backing
/// slices (undefined behaviour in C).  Errors returned by `dec` are
/// propagated unchanged.
///
/// # Complexity
///
/// O(n) in the number of points; no allocation is performed.
pub fn decompose<D: Decomposer>(outline: &Outline, dec: &mut D) -> TtResult<()> {
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
        let mut v_start = point(outline, first)?;
        let v_last = point(outline, last)?;
        let mut v_control = v_start;
        let mut cur = first;
        let tag = tag(outline, cur)?;
        if tag == CURVE_TAG_CUBIC {
            return Err(TtError::INVALID_OUTLINE);
        }
        if tag == CURVE_TAG_CONIC {
            if tag(outline, last)? == CURVE_TAG_ON {
                v_start = v_last;
                limit -= 1;
            } else {
                v_start.x = v_start.x.wrapping_add(v_last.x) / 2;
                v_start.y = v_start.y.wrapping_add(v_last.y) / 2;
            }
            cur -= 1;
        }
        dec.move_to(&v_start)?;
        let mut skip_line = false;
        'contour: while cur < limit {
            cur += 1;
            match tag(outline, cur)? {
                CURVE_TAG_ON => {
                    let vec = point(outline, cur)?;
                    dec.line_to(&vec)?;
                }
                CURVE_TAG_CONIC => {
                    v_control = point(outline, cur)?;
                    'do_conic: loop {
                        if cur < limit {
                            cur += 1;
                            let next_tag = tag(outline, cur)?;
                            let vec = point(outline, cur)?;
                            if next_tag == CURVE_TAG_ON {
                                dec.conic_to(&v_control, &vec)?;
                                continue 'contour;
                            }
                            if next_tag != CURVE_TAG_CONIC {
                                return Err(TtError::INVALID_OUTLINE);
                            }
                            let mut v_middle = v_control;
                            v_middle.x = v_middle.x.wrapping_add(vec.x) / 2;
                            v_middle.y = v_middle.y.wrapping_add(vec.y) / 2;
                            dec.conic_to(&v_control, &v_middle)?;
                            v_control = vec;
                            continue 'do_conic;
                        }
                        dec.conic_to(&v_control, &v_start)?;
                        skip_line = true;
                        break 'contour;
                    }
                }
                _ => {
                    if cur + 1 > limit || tag(outline, cur + 1)? != CURVE_TAG_CUBIC {
                        return Err(TtError::INVALID_OUTLINE);
                    }
                    cur += 2;
                    let vec1 = point(outline, cur - 2)?;
                    let vec2 = point(outline, cur - 1)?;
                    if cur <= limit {
                        let vec = point(outline, cur)?;
                        dec.cubic_to(&vec1, &vec2, &vec)?;
                    } else {
                        dec.cubic_to(&vec1, &vec2, &v_start)?;
                        skip_line = true;
                        break 'contour;
                    }
                }
            }
        }
        if !skip_line {
            dec.line_to(&v_start)?;
        }
        first = last + 1;
    }
    Ok(())
}

/// Reads the point at index `i`, mapping out-of-range indices to
/// [`TtError::INVALID_OUTLINE`] instead of the C routine's undefined
/// behaviour.
#[inline]
fn point(outline: &Outline, i: isize) -> TtResult<Vector> {
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
fn tag(outline: &Outline, i: isize) -> TtResult<u8> {
    if i < 0 {
        return Err(TtError::INVALID_OUTLINE);
    }
    let t = *outline.tags.get(i as usize).ok_or(TtError::INVALID_OUTLINE)?;
    Ok(t & CURVE_TAG_MASK)
}
