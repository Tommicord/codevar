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
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The packed element word this codec compresses.
//!
//! A `Writable` document is an array of 32-bit *raw* elements: bits
//! `0..28` hold the codepoint ([`CHAR_MASK`]) and bits `28..32` hold the
//! element type ([`TYPE_MASK`] over [`SHIFT_TYPE`]). These constants
//! mirror the engine's own definitions - this crate deliberately does not
//! depend on it, so the ABI is restated here and the two are kept in step
//! by the round-trip tests, which decode real element streams.
//!
//! One element is four bytes of RAM for one character, which is what stage 1
//! exploits: the four bytes of a word are four *independent* streams with
//! wildly different entropy, so they are separated before any modelling
//! happens. [`plane_mask`] gives the mask of each of those four byte
//! planes.

/// Shift of the element type field inside a packed raw element.
pub const SHIFT_TYPE: u32 = 28;

/// Mask of the element type field (already shifted down to bits `0..4`).
///
/// [`pack`] shifts a type value by [`SHIFT_TYPE`] and masks it with this
/// constant, so only the low 4 bits of a type survive.
pub const TYPE_MASK: u32 = 0xF;

/// Mask of the codepoint field inside a packed raw element.
pub const CHAR_MASK: u32 = 0xFF_FFFF;

/// Element type: an ordinary committed character.
///
/// In a real document this type accounts for all but a handful of the
/// elements, which is why the type plane compresses to about 0.02 bits per
/// character.
pub const COMMON_MASK: u32 = 0xE;

/// Element type: on flush, overwrites the raw element that follows the
/// insertion point instead of inserting a new element.
pub const MERGE_MASK: u32 = 0xD;

/// Number of bytes the stages of this crate treat as one element.
pub const ELEMENT_BYTES: usize = 4;

/// Mask of the bytes above the low one: the codepoint bits `8..24` and the
/// type nibble, i.e. everything stage 1 moves out of plane 0.
pub const UPPER_MASK: u32 = 0xFFFF_FF00;

/// Packs a codepoint and an element type into one raw element.
///
/// Both inputs are truncated: the codepoint keeps its low 24 bits, the type
/// its low 4.
///
/// # Examples
///
/// ```
/// use codevar_wredit_werc::element::{COMMON_MASK, pack, type_of, char_of};
///
/// let element = pack(0x41, COMMON_MASK);
/// assert_eq!(element, 0xE000_0041);
/// assert_eq!(char_of(element), 0x41);
/// assert_eq!(type_of(element), COMMON_MASK);
/// ```
#[inline]
#[must_use]
pub const fn pack(ch: u32, element_type: u32) -> u32 {
    ((element_type & TYPE_MASK) << SHIFT_TYPE) | (ch & CHAR_MASK)
}

/// The codepoint field of a packed element.
#[inline]
#[must_use]
pub const fn char_of(element: u32) -> u32 {
    element & CHAR_MASK
}

/// The element type field of a packed element, unshifted.
#[inline]
#[must_use]
pub const fn type_of(element: u32) -> u32 {
    (element >> SHIFT_TYPE) & TYPE_MASK
}

/// The `plane`-th byte of every element, as a mask over the packed word.
///
/// Plane 0 is the low byte of the codepoint - for ASCII text it *is* the
/// text - and plane 3 is the type byte, because bits `24..28` of the
/// codepoint field are always zero for a well-formed element.
///
/// Returns `None` for a plane index outside `0..4`.
///
/// # Examples
///
/// ```
/// use codevar_wredit_werc::element::{COMMON_MASK, pack, plane_mask};
///
/// let element = pack(0x4E_2D, COMMON_MASK);
/// assert_eq!(plane_mask(0), Some(0x0000_00FF));
/// assert_eq!(element & plane_mask(0).unwrap_or_default(), 0x2D);
/// assert_eq!(element & plane_mask(1).unwrap_or_default(), 0x4E_00);
/// assert_eq!(plane_mask(4), None);
/// ```
#[inline]
#[must_use]
pub const fn plane_mask(plane: usize) -> Option<u32> {
    if plane >= 4 {
        return None;
    }
    Some(0xFFu32 << (8 * plane))
}

/// The byte plane a packed element's type field lands in.
///
/// [`SHIFT_TYPE`] is 28, so the type nibble lives in the top bits of plane
/// 3; stage 1 can therefore keep the type plane exactly as the word's fourth
/// byte without shifting anything.
pub const TYPE_PLANE: usize = 3;

#[cfg(test)]
mod tests {
    use super::{
        CHAR_MASK, COMMON_MASK, MERGE_MASK, SHIFT_TYPE, TYPE_MASK, char_of, pack, plane_mask, type_of,
    };

    #[test]
    fn pack_truncates_both_fields() {
        assert_eq!(pack(u32::MAX, 0xFF), (TYPE_MASK << SHIFT_TYPE) | CHAR_MASK);
        assert_eq!(char_of(pack(u32::MAX, 0xFF)), CHAR_MASK);
        assert_eq!(type_of(pack(u32::MAX, 0xFF)), TYPE_MASK);
    }

    #[test]
    fn type_and_char_never_overlap() {
        let element = pack(CHAR_MASK, MERGE_MASK);
        assert_eq!(element, (MERGE_MASK << SHIFT_TYPE) | CHAR_MASK);
        assert_eq!(element & (TYPE_MASK << SHIFT_TYPE), MERGE_MASK << SHIFT_TYPE);
    }

    #[test]
    fn type_lands_in_the_top_byte() {
        // The type nibble occupies bits 28..32, so the byte itself is
        // `type << 4`: 0xE0 for a committed character, 0xD0 for a merge.
        assert_eq!(pack(0x41, COMMON_MASK) >> 24, 0xE0);
        assert_eq!(pack(0x41, MERGE_MASK) >> 24, 0xD0);
    }

    #[test]
    fn plane_masks_cover_the_word_once() {
        let mut union = 0u32;
        for plane in 0..4 {
            let mask = plane_mask(plane).unwrap_or_default();
            assert_eq!(union & mask, 0, "planes overlap at {plane}");
            union |= mask;
        }
        assert_eq!(union, u32::MAX);
    }
}
