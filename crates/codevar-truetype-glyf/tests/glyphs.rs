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

//! End-to-end loads of the well-formed glyphs of
//! [`test_util::default_font`], covering simple outlines, composites
//! and the metrics both feed into.

extern crate alloc;

#[allow(dead_code)]
#[path = "../src/test_util.rs"]
mod test_util;

use codevar_truetype_core::{
    GlyphFormat, GlyphSlot, LoadFlags, OUTLINE_HIGH_PRECISION, SizeMetrics, TtError, TtResult, Vector,
};
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

/// Loads into a fresh slot, keeping the outcome for assertions.
fn attempt(bytes: &[u8], flags: LoadFlags, glyph_index: u32) -> TtResult<GlyphSlot> {
    let mut slot = test_util::new_slot();
    load_into(bytes, flags, glyph_index, &mut slot)?;
    Ok(slot)
}

/// Loads into a fresh slot; only for loads that must succeed.
fn load(bytes: &[u8], flags: LoadFlags, glyph_index: u32) -> GlyphSlot {
    attempt(bytes, flags, glyph_index).unwrap()
}

/// The four points of glyph 1, as they land in the slot: the loader
/// slides the outline left by `pp1.x` so that the origin sits on the
/// left side bearing.
fn rect_slot_points() -> Vec<Vector> {
    alloc::vec![
        Vector::new(-7, 0),
        Vector::new(493, 0),
        Vector::new(493, 700),
        Vector::new(-7, 700),
    ]
}

#[test]
fn simple_glyph_decodes_points_contours_and_metrics() {
    let bytes = test_util::default_font();
    let slot = load(&bytes, LoadFlags::NO_SCALE, 1);

    assert_eq!(slot.format, GlyphFormat::Outline);
    assert_eq!(slot.outline.n_points, 4);
    assert_eq!(slot.outline.n_contours, 1);
    assert_eq!(slot.outline.contours, alloc::vec![3i16]);
    assert_eq!(slot.outline.points, rect_slot_points());
    assert_eq!(slot.outline.tags, alloc::vec![1u8, 1, 1, 1]);
    assert!(slot.outline.check().is_ok());

    // hmtx gives advance 510 and lsb -7; the phantom points become
    // `(-lsb, ..)` and `(advance, ..)` after the header offset.
    assert_eq!(slot.metrics.hori_bearing_x, -7);
    assert_eq!(slot.metrics.hori_bearing_y, 700);
    assert_eq!(slot.metrics.hori_advance, 510);
    assert_eq!(slot.metrics.width, 500);
    assert_eq!(slot.metrics.height, 700);
    assert_eq!(slot.linear_hori_advance, 510);
    // `hhea` fallback: ascender 800, descender -200, upem 1000.
    assert_eq!(slot.metrics.vert_bearing_x, -262);
    assert_eq!(slot.metrics.vert_bearing_y, 150);
    assert_eq!(slot.metrics.vert_advance, 1000);
    assert_eq!(slot.linear_vert_advance, 1000);
}

#[test]
fn quadratic_arc_keeps_its_conic_control_point() {
    let bytes = test_util::default_font();
    let slot = load(&bytes, LoadFlags::NO_SCALE, 2);

    assert_eq!(slot.outline.n_points, 3);
    assert_eq!(slot.outline.contours, alloc::vec![2i16]);
    // lsb is -4, so the outline slides right by four units.
    assert_eq!(
        slot.outline.points,
        alloc::vec![Vector::new(-4, 0), Vector::new(246, 700), Vector::new(496, 0),]
    );
    assert_eq!(slot.outline.tags, alloc::vec![1u8, 0, 1]);
    assert_eq!(slot.metrics.hori_bearing_x, -4);
    assert_eq!(slot.metrics.hori_advance, 520);
}

#[test]
fn composite_offsets_both_components_in_font_units() {
    let bytes = test_util::default_font();
    let slot = load(&bytes, LoadFlags::NO_SCALE, 3);

    assert_eq!(slot.format, GlyphFormat::Outline);
    assert_eq!(slot.outline.n_points, 7);
    assert_eq!(slot.outline.n_contours, 2);
    assert_eq!(slot.outline.contours, alloc::vec![3i16, 6]);
    assert_eq!(
        slot.outline.points,
        alloc::vec![
            // glyph 1 translated by (100, 200), then by -pp1.x
            Vector::new(99, 200),
            Vector::new(599, 200),
            Vector::new(599, 900),
            Vector::new(99, 900),
            // glyph 2 translated by (50, 60), then by -pp1.x
            Vector::new(49, 60),
            Vector::new(299, 760),
            Vector::new(549, 60),
        ]
    );
    assert!(slot.outline.check().is_ok());

    assert_eq!(slot.metrics.hori_bearing_x, 49);
    assert_eq!(slot.metrics.hori_bearing_y, 900);
    assert_eq!(slot.metrics.hori_advance, 530);
    assert_eq!(slot.metrics.width, 550);
    assert_eq!(slot.metrics.height, 840);
    assert_eq!(slot.linear_hori_advance, 530);
    assert_eq!(slot.metrics.vert_bearing_y, 80);
}

#[test]
fn point_matching_glues_the_second_component_to_the_first() {
    let bytes = test_util::default_font();
    let slot = load(&bytes, LoadFlags::NO_SCALE, 8);

    assert_eq!(slot.outline.n_points, 7);
    assert_eq!(slot.outline.contours, alloc::vec![3i16, 6]);
    // Point 0 of the arc is matched to point 1 of the rectangle, so
    // the arc slides right by 500 units; the composite's own lsb is
    // -7, which the final translate folds in.
    assert_eq!(
        slot.outline.points,
        alloc::vec![
            Vector::new(-7, 0),
            Vector::new(493, 0),
            Vector::new(493, 700),
            Vector::new(-7, 700),
            Vector::new(493, 0),
            Vector::new(743, 700),
            Vector::new(993, 0),
        ]
    );
    assert!(slot.outline.check().is_ok());

    // The matched pair really is the same point twice.
    assert_eq!(slot.outline.points[1], slot.outline.points[4]);
    assert_eq!(slot.metrics.hori_advance, 580);
    assert_eq!(slot.metrics.width, 1000);
}

#[test]
fn a_scale_transform_shrinks_only_its_own_component() {
    let bytes = test_util::default_font();
    let slot = load(&bytes, LoadFlags::NO_SCALE, 9);

    assert_eq!(slot.outline.n_points, 7);
    assert_eq!(
        slot.outline.points,
        alloc::vec![
            // glyph 1 at 0.5, lsb fold-in of -4
            Vector::new(-4, 0),
            Vector::new(246, 0),
            Vector::new(246, 350),
            Vector::new(-4, 350),
            // glyph 2 unscaled, offset by (100, 100)
            Vector::new(96, 100),
            Vector::new(346, 800),
            Vector::new(596, 100),
        ]
    );
    assert!(slot.outline.check().is_ok());
    assert_eq!(slot.metrics.hori_advance, 590);
    assert_eq!(slot.metrics.height, 800);
}

#[test]
fn repeated_flags_and_instructions_do_not_shift_the_points() {
    let bytes = test_util::default_font();
    let slot = load(&bytes, LoadFlags::NO_SCALE, 10);

    assert_eq!(slot.outline.n_points, 6);
    assert_eq!(slot.outline.contours, alloc::vec![5i16]);
    // The lsb is -1, so only one unit of slack separates the outline
    // from the advance origin.
    assert_eq!(
        slot.outline.points,
        alloc::vec![
            Vector::new(-1, 0),
            Vector::new(299, 0),
            Vector::new(599, 0),
            Vector::new(599, 300),
            Vector::new(299, 300),
            Vector::new(-1, 300),
        ]
    );
    assert_eq!(slot.outline.tags, alloc::vec![1u8, 0, 0, 1, 1, 1]);
    assert!(slot.outline.check().is_ok());
    assert_eq!(slot.metrics.hori_advance, 600);
    assert_eq!(slot.metrics.hori_bearing_x, -1);
}

#[test]
fn empty_glyph_yields_an_empty_outline() {
    let bytes = test_util::default_font();
    let slot = load(&bytes, LoadFlags::NO_SCALE, 0);

    assert_eq!(slot.format, GlyphFormat::Outline);
    assert_eq!(slot.outline.n_points, 0);
    assert_eq!(slot.outline.n_contours, 0);
    assert!(slot.outline.points.is_empty());
    assert!(slot.outline.contours.is_empty());
    assert!(slot.outline.check().is_ok());

    assert_eq!(slot.metrics.hori_bearing_x, 0);
    assert_eq!(slot.metrics.hori_bearing_y, 0);
    assert_eq!(slot.metrics.width, 0);
    assert_eq!(slot.metrics.height, 0);
    // hmtx alone still supplies the advance: 500.
    assert_eq!(slot.metrics.hori_advance, 500);
    assert_eq!(slot.linear_hori_advance, 500);
    assert_eq!(slot.metrics.vert_bearing_y, 500);
}

#[test]
fn use_my_metrics_adopts_the_components_phantom_points() {
    let bytes = test_util::default_font();
    let slot = load(&bytes, LoadFlags::NO_SCALE, 13);

    // The component's advance (510) replaces the composite's own 630.
    assert_eq!(slot.metrics.hori_advance, 510);
    assert_eq!(slot.linear_hori_advance, 630);
    assert_eq!(slot.outline.points, rect_slot_points());
    assert_eq!(slot.metrics.hori_bearing_y, 700);
}

#[test]
fn scaled_component_offset_scales_the_offset_too() {
    let bytes = test_util::default_font();
    let slot = load(&bytes, LoadFlags::NO_SCALE, 14);

    // The arc is halved to (0,0)/(125,350)/(250,0), the offset
    // (100,100) is halved by `SCALED_COMPONENT_OFFSET`, and the
    // composite's lsb of -10 shifts everything right by ten.
    assert_eq!(slot.outline.n_points, 3);
    assert_eq!(slot.outline.contours, alloc::vec![2i16]);
    assert_eq!(
        slot.outline.points,
        alloc::vec![Vector::new(40, 50), Vector::new(165, 400), Vector::new(290, 50),]
    );
    assert!(slot.outline.check().is_ok());
    assert_eq!(slot.metrics.width, 250);
    assert_eq!(slot.metrics.height, 350);
}

#[test]
fn no_recurse_reports_the_components_without_loading_them() {
    let bytes = test_util::default_font();
    let slot = load(&bytes, LoadFlags::NO_SCALE | LoadFlags::NO_RECURSE, 3);

    assert_eq!(slot.format, GlyphFormat::Composite);
    assert_eq!(slot.outline.n_points, 0);
    assert!(slot.outline.points.is_empty());
    assert_eq!(slot.subglyphs.len(), 2);
    assert_eq!(slot.subglyphs[0].index, 1);
    assert_eq!(slot.subglyphs[0].arg1, 100);
    assert_eq!(slot.subglyphs[0].arg2, 200);
    assert_eq!(slot.subglyphs[1].index, 2);
    assert_eq!(slot.subglyphs[1].arg1, 50);
    assert_eq!(slot.subglyphs[1].arg2, 60);

    // No outline: the metrics come from the glyph header instead.
    assert_eq!(slot.metrics.hori_bearing_x, 0);
    assert_eq!(slot.metrics.hori_bearing_y, 900);
    assert_eq!(slot.metrics.width, 600);
    assert_eq!(slot.metrics.height, 900);
    assert_eq!(slot.metrics.hori_advance, 530);
    assert_eq!(slot.metrics.vert_bearing_y, 50);
}

#[test]
fn hinted_load_rounds_the_phantom_points_to_the_grid() {
    let bytes = test_util::default_font();
    let slot = load(&bytes, LoadFlags::RENDER, 1);

    assert!(slot.outline.check().is_ok());
    // ppem 16 < 24, so `TT_Load_Glyph` marks the outline high precision.
    assert!(slot.outline.flags & OUTLINE_HIGH_PRECISION != 0);

    // 16/1000 of 510 is 8.16 px; the rounded phantom points sit on the
    // 64-unit grid (core's `pix_round` differs from FreeType's, so the
    // grid is asserted rather than FreeType's exact pixel value).
    assert_eq!(slot.metrics.hori_advance % 64, 0);
    assert_eq!(slot.metrics.hori_advance, 512);
    // The points themselves are scaled but never rounded.
    assert_eq!(slot.metrics.hori_bearing_x, -64);
    assert_eq!(slot.metrics.hori_bearing_y, 717);
    assert_eq!(slot.metrics.width, 512);
}

#[test]
fn a_second_load_does_not_inherit_the_previous_glyphs_points() {
    let bytes = test_util::default_font();
    let mut slot = test_util::new_slot();

    load_into(&bytes, LoadFlags::NO_SCALE, 1, &mut slot).unwrap();
    assert_eq!(slot.outline.n_points, 4);
    assert_eq!(slot.metrics.hori_bearing_y, 700);

    load_into(&bytes, LoadFlags::NO_SCALE, 0, &mut slot).unwrap();
    assert_eq!(slot.outline.n_points, 0);
    assert!(slot.outline.points.is_empty());
    assert_eq!(slot.metrics.hori_bearing_y, 0);
    assert_eq!(slot.metrics.height, 0);
    assert_eq!(slot.metrics.hori_advance, 500);
}

#[test]
fn a_failed_load_leaves_the_slot_in_a_documented_state() {
    let bytes = test_util::default_font();
    let mut slot = test_util::new_slot();
    load_into(&bytes, LoadFlags::NO_SCALE, 1, &mut slot).unwrap();

    let outcome = load_into(&bytes, LoadFlags::NO_SCALE, 4, &mut slot);
    assert_eq!(outcome, Err(TtError::INVALID_OUTLINE));
    // `slot.clear()` ran first: the slot is empty and documented as an
    // outline, never partially written.
    assert_eq!(slot.format, GlyphFormat::Outline);
    assert_eq!(slot.outline.n_points, 0);
    assert!(slot.outline.points.is_empty());
    assert_eq!(slot.metrics.hori_advance, 0);
}
