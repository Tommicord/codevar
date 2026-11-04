// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

//! Stage 1: split packed elements into byte planes, and put them back.
//!
//! One packed element is four bytes whose entropies differ by two orders of
//! magnitude. Measured on element streams dumped from the engine (100 KiB
//! documents, bits per character of the compressed plane):
//!
//! | plane | contents | code sample | multilingual sample |
//! |-------|----------|-------------|---------------------|
//! | 0 | low byte of the codepoint | 1.563 | 2.495 |
//! | 1 | codepoint bits `8..16` | 0.010 | 0.061 |
//! | 2 | codepoint bits `16..24` | 0.000 | 0.000 |
//! | 3 | element type | 0.019 | 0.019 |
//!
//! Modelling the interleaved words instead of the planes costs 60-119% more
//! output bytes, because the order-1..6 contexts then mix a byte of ASCII
//! text with a byte of type nibble and learn neither. That is the whole
//! argument for this stage, and it is why it runs before any modelling.
//!
//! The planes are concatenated, in order, and a plane is left out of the
//! stream entirely when every element has a zero byte there - [`presence`]
//! decides that with one OR-reduce, so an all-ASCII document ships two
//! planes instead of four and the model never spends a prediction on a lane
//! that holds nothing.
//!
//! Both directions are byte-order independent: the low plane is always
//! produced by shifting right, never by reinterpreting the word in memory,
//! so a stream encoded on a little-endian target decodes on a big-endian
//! one.

use crate::element::{ELEMENT_BYTES, plane_mask};
use crate::error::{WercError, WercResult};

/// Set in [`presence`] when at least one element has a non-zero byte 1.
pub const FLAG_PLANE1: u8 = 1 << 1;

/// Set in [`presence`] when at least one element has a non-zero byte 2.
pub const FLAG_PLANE2: u8 = 1 << 2;

/// Set in [`presence`] when at least one element has a non-zero byte 3.
///
/// The type nibble lives here, so a stream of real `Writable` elements
/// always has this flag: an ordinary committed character is `0xE0` in this
/// plane.
pub const FLAG_PLANE3: u8 = 1 << 3;

/// The flags a stage-1 stream can carry. Plane 0 is always present, so it
/// has no flag: dropping it would drop the character count.
pub const FLAGS_MASK: u8 = FLAG_PLANE1 | FLAG_PLANE2 | FLAG_PLANE3;

/// Which planes a block of elements needs, as one OR-reduce over the words.
///
/// A byte of a plane is non-zero somewhere exactly when the OR of all the
/// words has that byte non-zero, so the decision costs one pass and needs no
/// per-element branching. The returned value is already masked to
/// [`FLAGS_MASK`].
///
/// # Examples
///
/// ```
/// use codevar_wredit_werc::element::{COMMON_MASK, pack};
/// use codevar_wredit_werc::plane::{FLAG_PLANE1, FLAG_PLANE3, presence};
///
/// let ascii: Vec<u32> = "abc".bytes().map(|b| pack(u32::from(b), COMMON_MASK)).collect();
/// // No codepoint reaches 0x100, so plane 1 is empty; the type byte is not.
/// assert_eq!(presence(&ascii), FLAG_PLANE3);
///
/// let wide = vec![pack(0x4E_2D, COMMON_MASK)];
/// assert_eq!(presence(&wide), FLAG_PLANE1 | FLAG_PLANE3);
/// ```
#[inline]
#[must_use]
pub fn presence(elements: &[u32]) -> u8 {
    let mut acc = 0u32;
    for &element in elements {
        acc |= element;
    }
    let mut flags = 0u8;
    for plane in 1..ELEMENT_BYTES {
        if acc & plane_mask(plane).unwrap_or_default() != 0 {
            flags |= 1 << plane;
        }
    }
    flags & FLAGS_MASK
}

/// Number of planes a stream with these flags carries, plane 0 included.
///
/// Returns `None` if `flags` has a bit set outside [`FLAGS_MASK`], which
/// would mean the caller read a flags byte from a stream this build does not
/// understand.
#[inline]
#[must_use]
pub const fn plane_count(flags: u8) -> Option<usize> {
    if flags & !FLAGS_MASK != 0 {
        return None;
    }
    let mut count = 1usize;
    let mut bit = 1u8;
    while bit <= FLAG_PLANE3 {
        if flags & bit != 0 {
            count += 1;
        }
        bit <<= 1;
    }
    Some(count)
}

/// Bytes the stage-1 body for `count` elements occupies, or `None` if the
/// flags are malformed or the arithmetic would overflow.
#[inline]
#[must_use]
pub const fn body_len(count: usize, flags: u8) -> Option<usize> {
    match plane_count(flags) {
        Some(planes) => count.checked_mul(planes),
        None => None,
    }
}

/// Split `elements` into its planes, appended in plane order to `planes`.
///
/// `flags` must be [`presence(elements)`] - or a superset of it, which only
/// wastes space - and decides which planes are written. Returns the number of
/// bytes written.
///
/// # Errors
///
/// [`WercError::OutputTooSmall`] if `planes` is shorter than
/// [`body_len`], [`WercError::Unsupported`] if `flags` carries an unknown
/// plane bit.
///
/// # Examples
///
/// ```
/// use codevar_wredit_werc::element::{COMMON_MASK, pack};
/// use codevar_wredit_werc::plane::{join, presence, split};
///
/// let elements: Vec<u32> = "hi".chars().map(|c| pack(c as u32, COMMON_MASK)).collect();
/// let flags = presence(&elements);
/// let mut body = [0u8; 8];
/// let written = split(&elements, flags, &mut body).unwrap_or(0);
/// assert_eq!(&body[..written], [b'h', b'i', 0xE0, 0xE0]);
///
/// let mut back = [0u32; 2];
/// join(flags, &body[..written], &mut back).unwrap_or(0);
/// assert_eq!(back, [elements[0], elements[1]]);
/// ```
pub fn split(elements: &[u32], flags: u8, planes: &mut [u8]) -> WercResult<usize> {
    let needed = body_len(elements.len(), flags).ok_or(WercError::Unsupported { field: "flags" })?;
    if planes.len() < needed {
        return Err(WercError::OutputTooSmall {
            needed,
            available: planes.len(),
        });
    }
    let count = elements.len();
    let mut cursor = 0usize;
    crate::simd::write_plane(elements, &mut planes[cursor..cursor + count], 0);
    cursor += count;
    if flags & FLAG_PLANE1 != 0 {
        crate::simd::write_plane(elements, &mut planes[cursor..cursor + count], 8);
        cursor += count;
    }
    if flags & FLAG_PLANE2 != 0 {
        crate::simd::write_plane(elements, &mut planes[cursor..cursor + count], 16);
        cursor += count;
    }
    if flags & FLAG_PLANE3 != 0 {
        crate::simd::write_plane(elements, &mut planes[cursor..cursor + count], 24);
    }
    Ok(needed)
}

/// Rebuild the packed elements from their planes.
///
/// `out` sets the element count, so the caller decides how many elements to
/// recover; `planes` must hold at least [`body_len(out.len(), flags)`] bytes.
/// Returns the number of elements written.
///
/// # Errors
///
/// [`WercError::InputTooShort`] if `planes` is shorter than the body,
/// [`WercError::Unsupported`] if `flags` carries an unknown plane bit.
pub fn join(flags: u8, planes: &[u8], out: &mut [u32]) -> WercResult<usize> {
    let needed = body_len(out.len(), flags).ok_or(WercError::Unsupported { field: "flags" })?;
    if planes.len() < needed {
        return Err(WercError::InputTooShort {
            needed,
            available: planes.len(),
        });
    }
    let count = out.len();
    crate::simd::read_plane(&planes[..count], out, 0, true);
    let mut cursor = count;
    if flags & FLAG_PLANE1 != 0 {
        crate::simd::read_plane(&planes[cursor..cursor + count], out, 8, false);
        cursor += count;
    }
    if flags & FLAG_PLANE2 != 0 {
        crate::simd::read_plane(&planes[cursor..cursor + count], out, 16, false);
        cursor += count;
    }
    if flags & FLAG_PLANE3 != 0 {
        crate::simd::read_plane(&planes[cursor..cursor + count], out, 24, false);
    }
    Ok(count)
}

/// Gather byte `shift / 8` of every element into `plane`.
///
/// The scalar reference for every tier in [`crate::simd`]: the shift is
/// applied before the truncation, so the plane holds exactly that byte of
/// each word whatever the target's byte order.
#[inline]
pub(crate) fn write_shifted(elements: &[u32], plane: &mut [u8], shift: u32) {
    for (dst, &element) in plane.iter_mut().zip(elements) {
        *dst = (element >> shift) as u8;
    }
}

/// OR byte `shift / 8` of every plane byte into the elements of `out`.
///
/// Each plane owns a disjoint set of bits, so OR is exact and no plane can
/// corrupt a field another already wrote.
#[inline]
pub(crate) fn read_shifted(plane: &[u8], out: &mut [u32], shift: u32) {
    for (dst, &byte) in out.iter_mut().zip(plane) {
        *dst |= u32::from(byte) << shift;
    }
}

/// Write byte `shift / 8` of every plane byte into the elements of `out`.
///
/// This is [`read_shifted`] without the OR, used for plane 0 so the
/// reconstruction never depends on what `out` held before.
#[inline]
pub(crate) fn assign_shifted(plane: &[u8], out: &mut [u32], shift: u32) {
    for (dst, &byte) in out.iter_mut().zip(plane) {
        *dst = u32::from(byte) << shift;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FLAG_PLANE1, FLAG_PLANE2, FLAG_PLANE3, FLAGS_MASK, body_len, join, plane_count, presence, split,
    };
    use crate::element::{CHAR_MASK, COMMON_MASK, MERGE_MASK, pack};
    use crate::error::WercError;

    fn doc(text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| pack(c as u32, COMMON_MASK))
            .collect()
    }

    #[test]
    fn ascii_document_ships_two_planes() {
        let elements = doc("hello world");
        assert_eq!(presence(&elements), FLAG_PLANE3);
        assert_eq!(plane_count(FLAG_PLANE3), Some(2));
        assert_eq!(body_len(elements.len(), FLAG_PLANE3), Some(2 * elements.len()));
    }

    #[test]
    fn wide_codepoints_add_their_planes() {
        assert_eq!(presence(&doc("中")), FLAG_PLANE1 | FLAG_PLANE3);
        assert_eq!(
            presence(&doc("\u{1F980}")),
            FLAG_PLANE1 | FLAG_PLANE2 | FLAG_PLANE3
        );
    }

    #[test]
    fn round_trip_keeps_every_element() {
        let elements = doc("fn main() {\n    println!(\"café 中 🦀\");\n}\n");
        let flags = presence(&elements);
        let mut body = [0u8; 4096];
        let written = split(&elements, flags, &mut body).unwrap_or_default();
        assert_eq!(written, body_len(elements.len(), flags).unwrap_or_default());
        let mut back = vec![0u32; elements.len()];
        let n = join(flags, &body[..written], &mut back).unwrap_or_default();
        assert_eq!(n, elements.len());
        assert_eq!(&back[..], &elements[..]);
    }

    #[test]
    fn merge_elements_survive_the_split() {
        // The only element type the engine uses besides a committed
        // character, and the only one whose type byte differs.
        let mut elements = doc("ab");
        elements.push(pack(b'c' as u32, MERGE_MASK));
        let flags = presence(&elements);
        assert_eq!(flags, FLAG_PLANE3);
        let mut body = [0u8; 16];
        let written = split(&elements, flags, &mut body).unwrap_or_default();
        assert_eq!(&body[..elements.len()], [b'a', b'b', b'c']);
        assert_eq!(&body[elements.len()..written], [0xE0, 0xE0, 0xD0]);
        let mut back = [0u32; 3];
        join(flags, &body[..written], &mut back).unwrap_or_default();
        assert_eq!(&back[..], &elements[..]);
    }

    #[test]
    fn round_trip_covers_the_whole_char_field() {
        // A deterministic sweep, not random data: every bit of the codepoint
        // field must land in some plane and come back from it.
        let mut elements = [0u32; 512];
        for (i, slot) in elements.iter_mut().enumerate() {
            let ch = (i as u32 * 7919) & CHAR_MASK;
            let ty = if i % 32 == 0 { MERGE_MASK } else { COMMON_MASK };
            *slot = pack(ch, ty);
        }
        let flags = presence(&elements);
        assert_eq!(flags & FLAGS_MASK, flags);
        let mut body = [0u8; 2048];
        let written = split(&elements, flags, &mut body).unwrap_or_default();
        let mut back = [0u32; 512];
        join(flags, &body[..written], &mut back).unwrap_or_default();
        assert_eq!(back, elements);
    }

    #[test]
    fn empty_input_is_an_empty_body() {
        let mut body = [0u8; 4];
        assert_eq!(split(&[], 0, &mut body), Ok(0));
        let mut out = [0u32; 1];
        assert_eq!(join(0, &[], &mut out[..0]), Ok(0));
        assert_eq!(out[0], 0);
    }

    #[test]
    fn buffers_too_small_are_reported_with_the_size() {
        // Four ASCII elements, so plane 0 and plane 3: eight bytes.
        const NEEDED: usize = 8;
        let elements = doc("abcd");
        let flags = presence(&elements);
        assert_eq!(body_len(elements.len(), flags), Some(NEEDED));
        assert_eq!(
            split(&elements, flags, &mut [0u8; NEEDED - 1]),
            Err(WercError::OutputTooSmall {
                needed: NEEDED,
                available: NEEDED - 1
            })
        );
        assert_eq!(
            join(flags, &[0u8; NEEDED - 1], &mut [0u32; 4]),
            Err(WercError::InputTooShort {
                needed: NEEDED,
                available: NEEDED - 1
            })
        );
    }

    #[test]
    fn unknown_plane_bits_are_rejected() {
        let bad = FLAG_PLANE2 | (1 << 7);
        assert_eq!(plane_count(bad), None);
        assert!(split(&doc("a"), bad, &mut [0u8; 8]).is_err());
        assert!(join(bad, &[0u8; 8], &mut [0u32; 1]).is_err());
    }
}
