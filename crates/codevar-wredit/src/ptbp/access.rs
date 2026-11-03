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

//! Little-endian access to the words of a protocol image.
//!
//! The image crosses a device boundary as bytes, so nothing here reads
//! a `u32` through a pointer cast: every word is assembled from four
//! bytes, which keeps the host view independent of the target's
//! alignment and endianness rules and matches what a kernel sees when
//! it indexes the same plane.
//!
//! Bounds are checked before any read, in `u64`, because `usize` is 32
//! bits on a wasm32 target — a plane offset plus a length must not be
//! able to wrap into a valid-looking range.

use super::fields::{
    CTRL_BATCH, CTRL_BOUNDARY_COUNT, CTRL_CURSOR_COUNT, CTRL_DST_LEN, CTRL_FLAGS, CTRL_GAP_COUNT,
    CTRL_LOG_COUNT, CTRL_MAGIC, CTRL_OP_COUNT, CTRL_PAYLOAD_CAPACITY, CTRL_PAYLOAD_USED, CTRL_RESERVED,
    CTRL_SCAN_STRIDE, CTRL_SRC_LEN, CTRL_TEXT_CAPACITY, CTRL_VERSION,
};
use super::geometry::{
    CONTROL_OFFSET, CONTROL_WORDS, CURSOR_CAPACITY, IMAGE_WORDS, MAGIC, PAYLOAD_CAPACITY, Plane,
    TEXT_CAPACITY, VERSION, WORD_BYTES,
};
use crate::error::{WreditError, WreditResult};

// ---------------------------------------------------------------------------
// Word access
// ---------------------------------------------------------------------------

/// Words `data` can address; `None` when it is not whole words, or
/// when its word count does not fit a `u32`.
#[inline]
pub fn image_words(data: &[u8]) -> Option<u32> {
    if !data.len().is_multiple_of(WORD_BYTES) {
        return None;
    }
    u32::try_from(data.len() / WORD_BYTES).ok()
}

/// Byte range of `count` words from `start`, checked against `data`.
///
/// `usize` is 32 bits on a wasm32 target, so the arithmetic runs in
/// `u64` and is narrowed only once the result is known to fit.
fn word_range(data: &[u8], start: u32, count: u32) -> WreditResult<(usize, usize)> {
    let Some(total) = image_words(data) else {
        return Err(WreditError::InvalidAddress {
            address: data.len() as u64,
        });
    };
    let first = u64::from(start);
    let last = first
        .checked_add(u64::from(count))
        .ok_or(WreditError::OutOfRange {
            start: first,
            end: u64::from(u32::MAX),
        })?;
    if last > u64::from(total) {
        return Err(WreditError::OutOfRange {
            start: first,
            end: last,
        });
    }
    let first_bytes = usize::try_from(first * WORD_BYTES as u64).map_err(|_| WreditError::OutOfRange {
        start: first,
        end: last,
    })?;
    let last_bytes = usize::try_from(last * WORD_BYTES as u64).map_err(|_| WreditError::OutOfRange {
        start: first,
        end: last,
    })?;
    Ok((first_bytes, last_bytes))
}

/// Reads the little-endian word at `word`.
///
/// # Errors
///
/// [`WreditError::InvalidAddress`] when `data` is not whole words,
/// [`WreditError::OutOfRange`] when `word` escapes it.
pub fn get_word(data: &[u8], word: u32) -> WreditResult<u32> {
    let (start, end) = word_range(data, word, 1)?;
    let mut bytes = [0u8; WORD_BYTES];
    bytes.copy_from_slice(&data[start..end]);
    Ok(u32::from_le_bytes(bytes))
}

/// Writes `value` as the little-endian word at `word`.
///
/// # Errors
///
/// As [`get_word`].
pub fn put_word(data: &mut [u8], word: u32, value: u32) -> WreditResult<()> {
    let (start, end) = word_range(data, word, 1)?;
    data[start..end].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// Reads control field `field` of `image`.
///
/// # Errors
///
/// [`WreditError::OutOfRange`] when `field` escapes the control plane
/// or the image, [`WreditError::InvalidAddress`] when `image` is not
/// whole words.
#[inline]
pub fn control_at(image: &[u8], field: u32) -> WreditResult<u32> {
    if field >= CONTROL_WORDS {
        return Err(WreditError::OutOfRange {
            start: u64::from(field),
            end: u64::from(CONTROL_WORDS),
        });
    }
    get_word(image, CONTROL_OFFSET + field)
}

/// Writes control field `field` of `image`.
///
/// # Errors
///
/// As [`control_at`].
pub fn set_control(image: &mut [u8], field: u32, value: u32) -> WreditResult<()> {
    if field >= CONTROL_WORDS {
        return Err(WreditError::OutOfRange {
            start: u64::from(field),
            end: u64::from(CONTROL_WORDS),
        });
    }
    put_word(image, CONTROL_OFFSET + field, value)
}

/// Reads `count` elements of a run plane (`Payload`, `TextA`, `TextB`)
/// from element `index` into `out`.
///
/// # Errors
///
/// [`WreditError::InvalidOperation`] for a record-structured plane,
/// [`WreditError::OutOfRange`] when the run escapes the plane or the
/// image, or when `out` is shorter than `count`.
pub fn read_run(image: &[u8], plane: Plane, index: u32, count: u32, out: &mut [u32]) -> WreditResult<()> {
    if plane.record_words().is_some() {
        return Err(WreditError::InvalidOperation {
            what: "plane is record-structured",
        });
    }
    let Some(capacity) = plane.capacity() else {
        return Err(WreditError::InvalidOperation {
            what: "plane has no capacity",
        });
    };
    let Some(end) = index.checked_add(count) else {
        return Err(WreditError::OutOfRange {
            start: u64::from(index),
            end: u64::from(u32::MAX),
        });
    };
    if end > capacity || usize::try_from(count).unwrap_or(usize::MAX) > out.len() {
        return Err(WreditError::OutOfRange {
            start: u64::from(index),
            end: u64::from(end),
        });
    }
    let (start, limit) = word_range(image, plane.offset() + index, count)?;
    for (slot, bytes) in out
        .iter_mut()
        .zip(image[start..limit].chunks_exact(WORD_BYTES))
    {
        *slot = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    Ok(())
}

/// Writes `run` into a run plane (`Payload`, `TextA`, `TextB`) at
/// element `index`.
///
/// # Errors
///
/// As [`read_run`], plus [`WreditError::LimitExceeded`] when
/// `run.len()` does not fit a word index.
pub fn write_run(image: &mut [u8], plane: Plane, index: u32, run: &[u32]) -> WreditResult<()> {
    if plane.record_words().is_some() {
        return Err(WreditError::InvalidOperation {
            what: "plane is record-structured",
        });
    }
    let Some(capacity) = plane.capacity() else {
        return Err(WreditError::InvalidOperation {
            what: "plane has no capacity",
        });
    };
    let count = u32::try_from(run.len()).map_err(|_| WreditError::LimitExceeded { what: "run length" })?;
    let Some(end) = index.checked_add(count) else {
        return Err(WreditError::OutOfRange {
            start: u64::from(index),
            end: u64::from(u32::MAX),
        });
    };
    if end > capacity {
        return Err(WreditError::OutOfRange {
            start: u64::from(index),
            end: u64::from(end),
        });
    }
    let (start, limit) = word_range(image, plane.offset() + index, count)?;
    for (bytes, word) in image[start..limit]
        .chunks_exact_mut(WORD_BYTES)
        .zip(run)
    {
        bytes.copy_from_slice(&word.to_le_bytes());
    }
    Ok(())
}

/// Stamps a freshly allocated image: identity, revision, flags, and
/// the capacities a kernel reads before it trusts anything else.
///
/// The text and payload planes are left as they are — a caller that
/// needs them zeroed zeroes them, so a boot that reuses a buffer does
/// not pay for a second pass over the document.
///
/// # Errors
///
/// [`WreditError::InvalidAddress`] when `image` is not whole words,
/// [`WreditError::CapacityExhausted`] when it is shorter than
/// [`IMAGE_WORDS`].
pub fn init(image: &mut [u8], flags: u32) -> WreditResult<()> {
    let Some(words) = image_words(image) else {
        return Err(WreditError::InvalidAddress {
            address: image.len() as u64,
        });
    };
    if words < IMAGE_WORDS {
        return Err(WreditError::CapacityExhausted {
            requested: IMAGE_WORDS as usize,
            available: words as usize,
        });
    }
    set_control(image, CTRL_MAGIC, MAGIC)?;
    set_control(image, CTRL_VERSION, VERSION)?;
    set_control(image, CTRL_FLAGS, flags)?;
    set_control(image, CTRL_BATCH, 0)?;
    set_control(image, CTRL_SRC_LEN, 0)?;
    set_control(image, CTRL_DST_LEN, 0)?;
    set_control(image, CTRL_TEXT_CAPACITY, TEXT_CAPACITY)?;
    set_control(image, CTRL_OP_COUNT, 0)?;
    set_control(image, CTRL_BOUNDARY_COUNT, 0)?;
    set_control(image, CTRL_CURSOR_COUNT, CURSOR_CAPACITY)?;
    set_control(image, CTRL_GAP_COUNT, 0)?;
    set_control(image, CTRL_PAYLOAD_CAPACITY, PAYLOAD_CAPACITY)?;
    set_control(image, CTRL_PAYLOAD_USED, 0)?;
    set_control(image, CTRL_LOG_COUNT, 0)?;
    set_control(image, CTRL_SCAN_STRIDE, 0)?;
    set_control(image, CTRL_RESERVED, 0)
}

/// True when `image` carries this protocol revision.
///
/// A kernel performs the same two reads first and returns without
/// touching the text planes when they disagree, so a stale host
/// buffer cannot corrupt a document.
#[must_use]
pub fn is_conforming(image: &[u8]) -> bool {
    matches!(
        (control_at(image, CTRL_MAGIC), control_at(image, CTRL_VERSION)),
        (Ok(MAGIC), Ok(VERSION))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptbp::fields::FLAG_REMAP;
    use crate::ptbp::geometry::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn image() -> Vec<u8> {
        vec![0u8; IMAGE_WORDS as usize * WORD_BYTES]
    }

    #[test]
    fn init_stamps_a_conforming_header() {
        let mut data = image();
        assert!(!is_conforming(&data), "a zeroed image must not look conforming");
        // The image is built from the layout's own constants, so a
        // failure here means the layout is inconsistent, not the input.
        init(&mut data, FLAG_REMAP).unwrap();
        assert!(is_conforming(&data));
        assert_eq!(control_at(&data, CTRL_FLAGS).unwrap(), FLAG_REMAP);
        assert_eq!(control_at(&data, CTRL_TEXT_CAPACITY).unwrap(), TEXT_CAPACITY);
        assert_eq!(
            control_at(&data, CTRL_PAYLOAD_CAPACITY).unwrap(),
            PAYLOAD_CAPACITY
        );
        assert_eq!(control_at(&data, CTRL_CURSOR_COUNT).unwrap(), CURSOR_CAPACITY);
        assert_eq!(control_at(&data, CTRL_DST_LEN).unwrap(), 0);
        assert_eq!(control_at(&data, CTRL_RESERVED).unwrap(), 0);
    }

    #[test]
    fn init_rejects_a_short_or_misaligned_image() {
        let mut short = vec![0u8; (IMAGE_WORDS - 1) as usize * WORD_BYTES];
        assert!(
            matches!(
                init(&mut short, 0),
                Err(WreditError::CapacityExhausted { requested, available })
                    if requested == IMAGE_WORDS as usize && available == (IMAGE_WORDS - 1) as usize
            ),
            "a one-word-short image must report exactly what it lacks"
        );
        let mut ragged = image();
        ragged.truncate(ragged.len() - 1);
        assert!(matches!(
            init(&mut ragged, 0),
            Err(WreditError::InvalidAddress { .. })
        ));
    }

    #[test]
    fn a_tampered_header_stops_looking_conforming() {
        let mut data = image();
        init(&mut data, 0).unwrap();
        set_control(&mut data, CTRL_MAGIC, MAGIC ^ 1).unwrap();
        assert!(!is_conforming(&data), "a wrong magic must be rejected");
        set_control(&mut data, CTRL_MAGIC, MAGIC).unwrap();
        set_control(&mut data, CTRL_VERSION, VERSION + 1).unwrap();
        assert!(!is_conforming(&data), "a wrong revision must be rejected");
    }

    #[test]
    fn word_access_round_trips_little_endian() {
        let mut data = image();
        put_word(&mut data, TEXT_A_OFFSET + 7, 0xDEAD_BEEF).unwrap();
        assert_eq!(get_word(&data, TEXT_A_OFFSET + 7).unwrap(), 0xDEAD_BEEF);
        let tail = (TEXT_A_OFFSET as usize + 7) * WORD_BYTES;
        assert_eq!(&data[tail..tail + WORD_BYTES], &[0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn word_access_bounds_checks_before_reading() {
        let data = image();
        assert!(
            matches!(
                get_word(&data, IMAGE_WORDS),
                Err(WreditError::OutOfRange { start, end })
                    if start == IMAGE_WORDS as u64 && end == IMAGE_WORDS as u64 + 1
            ),
            "reading one past the image must name the word it wanted"
        );
        assert!(matches!(get_word(&data, IMAGE_WORDS - 1), Ok(0)));
        let ragged = vec![0u8; 6];
        assert!(matches!(
            get_word(&ragged, 0),
            Err(WreditError::InvalidAddress { .. })
        ));
    }

    #[test]
    fn a_word_index_that_would_wrap_is_reported_not_wrapped() {
        let data = image();
        assert!(
            matches!(
                get_word(&data, u32::MAX),
                Err(WreditError::OutOfRange { start, end })
                    if start == u64::from(u32::MAX) && end == u64::from(u32::MAX) + 1
            ),
            "start + 1 must be computed without wrapping a u32"
        );
    }

    #[test]
    fn control_fields_reject_a_field_outside_the_plane() {
        let mut data = image();
        assert!(
            matches!(
                control_at(&data, CONTROL_WORDS),
                Err(WreditError::OutOfRange { start, end })
                    if start == CONTROL_WORDS as u64 && end == CONTROL_WORDS as u64
            ),
            "the control plane must reject its own size as a field index"
        );
        assert!(matches!(
            set_control(&mut data, CONTROL_WORDS + 9, 1),
            Err(WreditError::OutOfRange { .. })
        ));
    }

    #[test]
    fn runs_write_and_read_back_a_plane() {
        const RUN: [u32; 5] = [1, 2, 3, 0xE100_0041, 5];
        let mut data = image();
        write_run(&mut data, Plane::TextA, 40, &RUN).unwrap();
        let mut back = [0u32; RUN.len()];
        read_run(&data, Plane::TextA, 40, RUN.len() as u32, &mut back).unwrap();
        assert_eq!(back, RUN);
        write_run(&mut data, Plane::Payload, 0, &RUN).unwrap();
        let mut at = [0u32; RUN.len()];
        read_run(&data, Plane::Payload, 0, RUN.len() as u32, &mut at).unwrap();
        assert_eq!(at, RUN);
    }

    #[test]
    fn a_run_ends_where_the_next_plane_starts() {
        let mut data = image();
        let run = [7u32; 4];
        write_run(&mut data, Plane::TextA, TEXT_CAPACITY - 4, &run).unwrap();
        let mut back = [0u32; 4];
        read_run(&data, Plane::TextA, TEXT_CAPACITY - 4, 4, &mut back).unwrap();
        assert_eq!(back, run, "the last elements of plane A must be addressable");
        assert_eq!(
            get_word(&data, TEXT_B_OFFSET).unwrap(),
            0,
            "and must not spill into plane B"
        );
    }

    #[test]
    fn runs_are_bounded_by_the_plane_and_the_out_buffer() {
        let mut data = image();
        let run = [0u32; 4];
        assert!(matches!(
            write_run(&mut data, Plane::TextA, TEXT_CAPACITY - 2, &run),
            Err(WreditError::OutOfRange { .. })
        ));
        let mut tiny = [0u32; 2];
        assert!(matches!(
            read_run(&data, Plane::Payload, 0, 4, &mut tiny),
            Err(WreditError::OutOfRange { .. })
        ));
        assert_eq!(
            write_run(&mut data, Plane::Ops, 0, &run),
            Err(WreditError::InvalidOperation {
                what: "plane is record-structured"
            })
        );
    }
}
