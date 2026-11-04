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
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Integration tests against a real `COLR` v1 font.
//!
//! `resources/NotoColorEmoji-colr-v1-subset.ttf` is a subset of
//! Noto Color Emoji 2.057 containing five emoji that exercise
//! `PaintTransform`, `PaintComposite`, gradients, nested layer lists
//! and glyph references.  It is kept small (85 KiB) so it can live in
//! the repository while still being byte-for-byte output of a real
//! font toolchain.

// clippy.toml exempts `#[test]` bodies, not the fixtures they share.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fs;

use codevar_truetype_core::{Bitmap, PixelMode};
use codevar_truetype_emoji_color::render::ColorFont;
use codevar_truetype_sfnt::SfntFont;

/// The real font shipped with the crate's tests.
const FONT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/resources/NotoColorEmoji-colr-v1-subset.ttf"
);

/// The emoji the subset was built from.  Each one exercises a
/// different paint feature combination.
const EMOJI: [char; 5] = ['\u{1F600}', '\u{1F980}', '\u{1F680}', '\u{1F389}', '\u{1F4BB}'];

/// Opens the resource font.
fn open(bytes: &[u8]) -> ColorFont<'_> {
    let font = SfntFont::open(bytes, 0).expect("the resource font must parse");
    let color = ColorFont::open(font);
    assert!(color.has_color(), "the resource font must carry COLR/CPAL");
    assert!(color.num_palettes() > 0, "the resource font needs a palette");
    color
}

/// The bitmap's pixels in row-major BGRA order.
fn pixels(bitmap: &Bitmap) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity((bitmap.width * bitmap.rows) as usize);
    for row in 0..bitmap.rows {
        let bytes = bitmap.row(row).expect("a row inside the bitmap");
        for pixel in bytes.chunks_exact(4) {
            out.push([pixel[0], pixel[1], pixel[2], pixel[3]]);
        }
    }
    out
}

#[test]
fn real_colr_v1_emoji_render_in_color_at_every_size() {
    let bytes = fs::read(FONT).expect("the resource font must be readable");
    let font = open(&bytes);

    for character in EMOJI {
        for ppem in [8u16, 16, 32, 64] {
            let glyph = font
                .render_char(character, ppem, 0)
                .unwrap_or_else(|error| panic!("{character:?} at {ppem}ppem failed: {error:?}"));
            assert_eq!(glyph.bitmap.pixel_mode, PixelMode::Bgra);
            assert!(
                glyph.bitmap.width > 0 && glyph.bitmap.rows > 0,
                "{character:?} at {ppem}ppem produced an empty bitmap"
            );
            // An emoji fits inside its em: allow a generous margin for
            // bearing/pixel rounding, but catch a doubled or exploded
            // transform.
            let limit = u32::from(ppem) * 2 + 8;
            assert!(
                glyph.bitmap.width <= limit && glyph.bitmap.rows <= limit,
                "{character:?} at {ppem}ppem rendered {}x{} (limit {limit})",
                glyph.bitmap.width,
                glyph.bitmap.rows
            );
            assert!(glyph.advance > 0, "{character:?} at {ppem}ppem has no advance");

            let pixels = pixels(&glyph.bitmap);
            assert!(
                pixels.iter().any(|pixel| pixel[3] > 0),
                "{character:?} at {ppem}ppem is fully transparent"
            );
            assert!(
                pixels
                    .iter()
                    .any(|pixel| pixel[3] > 0 && (pixel[0] != pixel[1] || pixel[1] != pixel[2])),
                "{character:?} at {ppem}ppem has no colored pixel"
            );
        }
    }
}

#[test]
fn real_colr_v1_emoji_keep_their_devices_bounds() {
    let bytes = fs::read(FONT).expect("the resource font must be readable");
    let font = open(&bytes);

    for character in EMOJI {
        let small = font
            .render_char(character, 16, 0)
            .expect("16ppem");
        let large = font
            .render_char(character, 32, 0)
            .expect("32ppem");
        // Doubling the ppem roughly doubles the ink, never squares it:
        // a correct transform leaves the box within a few pixels of
        // twice the 16ppem box.
        assert!(
            large.bitmap.width >= small.bitmap.width,
            "{character:?} shrank when scaled up"
        );
        assert!(
            large.bitmap.width <= small.bitmap.width * 2 + 6,
            "{character:?} grew more than 2x: {} -> {}",
            small.bitmap.width,
            large.bitmap.width
        );
        assert!(
            large.bitmap.rows <= small.bitmap.rows * 2 + 6,
            "{character:?} grew more than 2x: {} -> {}",
            small.bitmap.rows,
            large.bitmap.rows
        );
    }
}

#[test]
fn every_glyph_in_the_real_font_renders() {
    let bytes = fs::read(FONT).expect("the resource font must be readable");
    let font = open(&bytes);
    let num_glyphs = u32::from(
        SfntFont::open(&bytes, 0)
            .expect("reopen")
            .num_glyphs(),
    );

    for gid in 0..num_glyphs {
        font.render_glyph(gid, 16, 0)
            .unwrap_or_else(|error| panic!("glyph {gid} failed: {error:?}"));
    }
}

#[test]
fn the_emoji_advance_comes_from_the_font_metrics() {
    // Noto Color Emoji is monospaced: every glyph, including U+0020,
    // advances 1275 units over a 1024-unit em (1.245 em).  Twemoji's
    // emoji advance is exactly one em and Twemoji has no space glyph.
    // The wider gaps between emoji rendered with this font therefore
    // come from `hmtx`, not from a misread table offset.
    let bytes = fs::read(FONT).expect("the resource font must be readable");
    let font = open(&bytes);

    for ppem in [16u16, 32, 64] {
        let glyph = font
            .render_char('\u{1F600}', ppem, 0)
            .expect("u1F600");
        let expected = 1275 * i64::from(ppem) * 64 / 1024;
        assert_eq!(
            glyph.advance, expected,
            "u1F600 at {ppem}ppem must advance 1275/1024 em"
        );
    }
}

#[test]
fn the_paint_transform_offsets_are_applied() {
    // u1F600 has a `PaintTransform` at the root of several layers; its
    // affine record is stored behind an Offset24, not inline.  Reading
    // it inline produced garbage transforms that failed with
    // `RASTER_OVERFLOW`, so the glyph silently fell back to the gray
    // outline.
    let bytes = fs::read(FONT).expect("the resource font must be readable");
    let font = open(&bytes);
    let glyph = font
        .render_char('\u{1F600}', 16, 0)
        .expect("u1F600");
    assert_eq!(
        glyph.bitmap.pixel_mode,
        PixelMode::Bgra,
        "u1F600 fell back to the outline instead of painting"
    );
    assert_eq!((glyph.bitmap.width, glyph.bitmap.rows), (18, 17));
    assert_eq!((glyph.bitmap_left, glyph.bitmap_top), (1, 14));
    // 1275 units at 16ppem over a 1024-unit em, in 26.6 pixels.
    assert_eq!(glyph.advance, 1275);
}
