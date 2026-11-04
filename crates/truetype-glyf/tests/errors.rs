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

//! Every error [`load_glyph`] documents, exercised against the
//! deliberately broken glyphs of [`test_util::default_font`].

extern crate alloc;

#[allow(dead_code)]
#[path = "../src/test_util.rs"]
mod test_util;

use codevar_truetype_core::{GlyphSlot, LoadFlags, SizeMetrics, TtError, TtResult};
use codevar_truetype_glyf::load_glyph;

/// The size a load with `flags` runs against: identity for
/// `FT_LOAD_NO_SCALE`, 16 ppem otherwise.
fn size_for(flags: LoadFlags) -> SizeMetrics {
    if flags.contains(LoadFlags::NO_SCALE) {
        test_util::no_scale_size()
    } else {
        test_util::scaled_size(1000, 16)
    }
}

/// Loads `glyph_index` into `slot`, exactly as the base layer would.
fn load_into(bytes: &[u8], flags: LoadFlags, glyph_index: u32, slot: &mut GlyphSlot) -> TtResult<()> {
    let font = test_util::open(bytes)?;
    load_glyph(&font, slot, &size_for(flags), glyph_index, flags)
}

/// Loads into a fresh slot and returns whichever error it produced.
fn expect_err(bytes: &[u8], flags: LoadFlags, glyph_index: u32) -> TtError {
    let mut slot = test_util::new_slot();
    match load_into(bytes, flags, glyph_index, &mut slot) {
        Err(error) => error,
        Ok(()) => panic!("glyph {glyph_index} unexpectedly loaded"),
    }
}

#[test]
fn a_contour_count_that_is_neither_positive_nor_minus_one_is_rejected() {
    let bytes = test_util::default_font();
    assert_eq!(
        expect_err(&bytes, LoadFlags::NO_SCALE, 4),
        TtError::INVALID_OUTLINE
    );
}

#[test]
fn a_header_without_contour_ends_is_rejected() {
    let bytes = test_util::default_font();
    assert_eq!(
        expect_err(&bytes, LoadFlags::NO_SCALE, 5),
        TtError::INVALID_OUTLINE
    );
}

#[test]
fn unordered_contour_ends_are_rejected() {
    let bytes = test_util::default_font();
    assert_eq!(
        expect_err(&bytes, LoadFlags::NO_SCALE, 11),
        TtError::INVALID_OUTLINE
    );
}

#[test]
fn an_instruction_count_past_the_glyph_end_is_rejected() {
    let bytes = test_util::default_font();
    assert_eq!(
        expect_err(&bytes, LoadFlags::NO_SCALE, 12),
        TtError::TOO_MANY_HINTS
    );
}

#[test]
fn a_self_referencing_composite_is_rejected() {
    let bytes = test_util::default_font();
    assert_eq!(
        expect_err(&bytes, LoadFlags::NO_SCALE, 6),
        TtError::INVALID_COMPOSITE
    );
}

#[test]
fn nesting_deeper_than_max_component_depth_is_rejected() {
    // `maxComponentDepth` 0 only trips once the recursion count passes
    // 1, i.e. when loading glyph 7's second level.
    let shallow = test_util::font_with_depth(0);
    assert_eq!(
        expect_err(&shallow, LoadFlags::NO_SCALE, 7),
        TtError::INVALID_COMPOSITE
    );

    // The permissive fixture accepts the very same glyph.
    let deep = test_util::default_font();
    let mut slot = test_util::new_slot();
    assert!(load_into(&deep, LoadFlags::NO_SCALE, 7, &mut slot).is_ok());
    // glyph 3 is itself a two-component composite: 4 + 3 points.
    assert_eq!(slot.outline.n_points, 7);
}

#[test]
fn a_glyph_index_past_num_glyphs_is_rejected() {
    let bytes = test_util::default_font();
    let font = test_util::open(&bytes).unwrap();
    let num_glyphs = u32::from(font.num_glyphs());
    assert_eq!(
        expect_err(&bytes, LoadFlags::NO_SCALE, num_glyphs),
        TtError::INVALID_GLYPH_INDEX
    );
}

#[test]
fn sbits_only_is_rejected_because_there_is_no_bitmap_path() {
    let bytes = test_util::default_font();
    assert_eq!(
        expect_err(&bytes, LoadFlags::SBITS_ONLY, 1),
        TtError::INVALID_ARGUMENT
    );
}

#[test]
fn a_slot_without_an_internal_record_is_rejected() {
    let bytes = test_util::default_font();
    let mut slot = test_util::new_slot();
    slot.internal = None;
    assert_eq!(
        load_into(&bytes, LoadFlags::NO_SCALE, 1, &mut slot),
        Err(TtError::INVALID_HANDLE)
    );
}

#[test]
fn a_slot_whose_glyph_loader_is_gone_is_rejected() {
    let bytes = test_util::default_font();
    let mut slot = test_util::new_slot();
    slot.internal.as_mut().unwrap().loader = None;
    assert_eq!(
        load_into(&bytes, LoadFlags::NO_SCALE, 1, &mut slot),
        Err(TtError::INVALID_HANDLE)
    );
}
