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

//! # Codevar auto-hinter
//!
//! A faithful Rust port of the FreeType 2.6 auto-hinter
//! (`src/autofit/`), the grid-fitting engine that derives stem widths,
//! blue zones and edge positions from a scalable outline without any
//! font bytecode.
//!
//! The port follows FreeType's module layout:
//!
//! | FreeType file | Section below |
//! |---------------|---------------|
//! | `aftypes.h`   | scaler, writing system, script and style types |
//! | `afblue.c/h`  | [`BLUE_STRINGS`], [`BLUE_STRINGSETS`] |
//! | `afranges.c`  | Unicode ranges of [`SCRIPT_CLASSES`] |
//! | `afangles.c`  | [`sort_positions`], [`sort_and_quantize_widths`] |
//! | `afhints.c/h` | [`GlyphHints`] and its reload/align routines |
//! | `afwarp.c`    | [`warper_compute`] |
//! | `afdummy.c`   | dummy writing system |
//! | `aflatin.c`   | [`latin`] section |
//! | `aflatin2.c`  | [`latin2`] section (alternate, unregistered) |
//! | `afcjk.c`     | [`cjk`] section |
//! | `afindic.c`   | Indic writing system (delegates to CJK) |
//! | `afglobal.c`  | [`FaceGlobals`] |
//! | `afloader.c`  | [`FaceAutohint::hint_glyph`] |
//! | `afmodule.c`  | [`Autohinter`] configuration |
//!
//! ## Configuration
//!
//! The build mirrors FreeType's `ftoption.h` defaults for the
//! auto-hinter:
//!
//! * `AF_CONFIG_OPTION_CJK`, `AF_CONFIG_OPTION_INDIC` and
//!   `AF_CONFIG_OPTION_USE_WARPER` are **enabled**;
//! * `AF_CONFIG_OPTION_CJK_BLUE_HANI_VERT` is disabled;
//! * `FT_CONFIG_OPTION_USE_HARFBUZZ` is disabled, so OpenType feature
//!   coverages (small caps, superior figures, ...) are never computed
//!   and [`af_get_coverage`] is a no-op stub exactly as in FreeType;
//! * `FT_OPTION_AUTOFIT2` is disabled, so the `latin2` writing system
//!   is **not** part of [`WRITING_SYSTEM_CLASSES`] /
//!   [`STYLE_CLASSES`]; its routines are still ported in the
//!   [`latin2`] section and can be selected per glyph through
//!   [`LoadOptions::writing_system_override`].
//!
//! ## Public API
//!
//! The integration surface is small:
//!
//! * [`GlyphProvider`] — the face abstraction (outlines, metrics and
//!   character map access) implemented by the caller;
//! * [`Autohinter`] / [`FaceAutohint`] — module level configuration
//!   and per-face cached global metrics;
//! * [`LoadOptions`] — scaler and rendering parameters for one glyph;
//! * [`HintedMetrics`] — the resulting grid-fitted glyph metrics.
//!
//! ## SIMD
//!
//! Hot loops ship scalar reference implementations plus runtime
//! dispatched vector kernels (SSE2/AVX2 on `x86_64`, NEON on
//! `aarch64`). Every kernel is selected through the `*_simd`
//! dispatcher, which is reachable from tests via the
//! [`set_simd_backend`] override hook so that scalar and vector paths
//! can be compared for bit-identical output.
#![cfg_attr(not(test), no_std)]
extern crate alloc;

use codevar_truetype_core::{Fixed, Pos, RenderMode};

/// Number of dimensions hinted by the auto-hinter (`AF_DIMENSION_MAX`).
pub const DIMENSION_MAX: usize = 2;

/// `AF_LATIN_MAX_WIDTHS`: slots in a Latin/CJK standard-width table.
pub const LATIN_MAX_WIDTHS: usize = 16;

/// `AF_BLUE_STRINGSET_MAX_LEN`: upper bound of a blue stringset.
pub const BLUE_STRINGSET_MAX_LEN: usize = 8;

/// `AF_BLUE_STRING_MAX_LEN`: upper bound of a blue string, in chars.
///
/// FreeType stores `AF_BLUE_STRING_MAX_LEN = 51` *bytes*; since the
/// longest generated string (both CJK sets) is exactly 51 characters,
/// the same bound holds for the decoded `char` count.
pub const BLUE_STRING_MAX_LEN: usize = 51;

/// `AF_ANGLE_PI`: a full turn is 512 units (`afangles.h`).
pub const ANGLE_PI: i32 = 256;
/// `AF_ANGLE_2PI`.
pub const ANGLE_2PI: i32 = ANGLE_PI * 2;
/// `AF_ANGLE_PI2`: a quarter turn.
pub const ANGLE_PI2: i32 = ANGLE_PI / 2;
/// `AF_ANGLE_PI4`: an eighth turn.
pub const ANGLE_PI4: i32 = ANGLE_PI / 4;

/// `AF_ANGLE_DIFF(result, angle1, angle2)` (aftypes.h): computes
/// `angle2 - angle1` wrapped into `[-PI .. PI]`.
#[inline]
pub fn angle_diff(angle1: i32, angle2: i32) -> i32 {
    let mut delta = angle2.wrapping_sub(angle1);
    while delta <= -ANGLE_PI {
        delta = delta.wrapping_add(ANGLE_2PI);
    }
    while delta > ANGLE_PI {
        delta = delta.wrapping_sub(ANGLE_2PI);
    }
    delta
}

/// `AF_WidthRec` (aftypes.h): a position or width at three stages of
/// the hinting process.
///
/// * `org` — original value in font units,
/// * `cur` — scaled value in 1/64th device pixels,
/// * `fit` — grid-fitted value in 1/64th device pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Width {
    /// Original position/width in font units.
    pub org: Pos,
    /// Current/scaled position/width in device sub-pixels.
    pub cur: Pos,
    /// Current/fitted position/width in device sub-pixels.
    pub fit: Pos,
}

/// `af_sort_pos` (afangles.c): insertion-sorts `count` positions in
/// ascending order.
///
/// This is the scalar reference implementation; [`sort_positions_simd`]
/// must produce a bit-identical result (a sorted array is unique, so
/// the vector path may use any sorting strategy).
#[inline]
pub fn sort_positions(count: usize, table: &mut [Pos]) {
    for i in 1..count {
        let mut j = i;
        while j > 0 && table[j] < table[j - 1] {
            table.swap(j, j - 1);
            j -= 1;
        }
    }
}

/// `af_sort_and_quantize_widths` (afangles.c): sorts a table of widths
/// in ascending order, replaces clusters that are closer together than
/// `threshold` by their mean value, and compacts zeroed entries.
///
/// `count` is updated to the number of surviving widths.
///
/// This is the scalar reference implementation;
/// [`sort_and_quantize_widths_simd`] must produce a bit-identical
/// result.
pub fn sort_and_quantize_widths(count: &mut usize, table: &mut [Width], threshold: Pos) {
    if *count <= 1 {
        return;
    }

    // sort
    for i in 1..*count {
        let mut j = i;
        while j > 0 && table[j].org < table[j - 1].org {
            table.swap(j, j - 1);
            j -= 1;
        }
    }

    let mut cur_idx = 0usize;
    let mut cur_val = table[cur_idx].org;

    // compute and use mean values for clusters not larger than
    // `threshold`; this is very primitive and might not yield the best
    // result, but normally, using reference character `o', `*count' is
    // 2, so the code below is fully sufficient
    let mut i = 1usize;
    while i < *count {
        if table[i].org.wrapping_sub(cur_val) > threshold || i == *count - 1 {
            let mut sum: Pos = 0;

            // fix loop for end of array
            let mut end = i;
            if table[i].org.wrapping_sub(cur_val) <= threshold && i == *count - 1 {
                end = i + 1;
            }

            let mut j = cur_idx;
            while j < end {
                sum = sum.wrapping_add(table[j].org);
                table[j].org = 0;
                j += 1;
            }
            table[cur_idx].org = sum / j as Pos;

            if i < *count - 1 {
                cur_idx = i + 1;
                cur_val = table[cur_idx].org;
            }
        }
        i += 1;
    }

    // compress array to remove zero values
    let mut out = 1usize;
    let mut src = 1usize;
    while src < *count {
        if table[src].org != 0 {
            table[out] = table[src];
            out += 1;
        }
        src += 1;
    }

    *count = out;
}

/// The two coordinate dimensions processed by the auto-hinter
/// (`AF_Dimension`).
///
/// Note that [`Dimension::Hort`] corresponds to *vertical* edges, since
/// they have a constant X coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Dimension {
    /// `AF_DIMENSION_HORZ`: X coordinates, vertical segments/edges.
    Hort = 0,
    /// `AF_DIMENSION_VERT`: Y coordinates, horizontal segments/edges.
    Vert = 1,
}

impl Dimension {
    /// Both dimensions, in FreeType's iteration order (`VERT` first).
    pub const ALL: [Dimension; DIMENSION_MAX] = [Dimension::Vert, Dimension::Hort];

    /// The other dimension of the pair.
    #[inline]
    pub const fn other(self) -> Dimension {
        match self {
            Dimension::Hort => Dimension::Vert,
            Dimension::Vert => Dimension::Hort,
        }
    }

    /// Numeric index (`0` for [`Dimension::Hort`], `1` for
    /// [`Dimension::Vert`]).
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            Dimension::Hort => 0,
            Dimension::Vert => 1,
        }
    }
}

/// Hint direction of a vector (`AF_Direction`).
///
/// The values are computed so that two vectors are in opposite
/// directions iff `dir1 + dir2 == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i8)]
pub enum Direction {
    /// `AF_DIR_RIGHT`.
    Right = 1,
    /// `AF_DIR_LEFT`.
    Left = -1,
    /// `AF_DIR_UP`.
    Up = 2,
    /// `AF_DIR_DOWN`.
    Down = -2,
    /// `AF_DIR_NONE`: the vector has no dominant direction.
    None = 4,
}

impl Direction {
    /// Numeric value as stored in the hint records.
    #[inline]
    pub const fn code(self) -> i8 {
        self as i8
    }

    /// Builds a direction from the raw `FT_Char` value stored in a
    /// point record, mapping unknown values to [`Direction::None`].
    #[inline]
    pub const fn from_code(code: i8) -> Direction {
        match code {
            1 => Direction::Right,
            -1 => Direction::Left,
            2 => Direction::Up,
            -2 => Direction::Down,
            _ => Direction::None,
        }
    }
}

/// `af_direction_compute` (afhints.c): returns the dominant direction
/// of the vector `(dx, dy)`, or [`Direction::None`] when the arms do
/// not differ enough (heuristic value 14, approx. 4.1 degrees).
#[inline]
pub fn direction_compute(dx: Pos, dy: Pos) -> Direction {
    let (dir, ll, ss);
    if dy >= dx {
        if dy >= -dx {
            dir = Direction::Up;
            ll = dy;
            ss = dx;
        } else {
            dir = Direction::Left;
            ll = -dx;
            ss = dy;
        }
    } else if dy >= -dx {
        dir = Direction::Right;
        ll = dx;
        ss = dy;
    } else {
        dir = Direction::Down;
        ll = -dy;
        ss = dx;
    }

    // the long arm is never negative
    if ll <= 14 * ss.abs() {
        return Direction::None;
    }
    dir
}

/// `AF_SCALER_FLAG_NO_HORIZONTAL`: disable horizontal hinting.
pub const SCALER_FLAG_NO_HORIZONTAL: u32 = 1;
/// `AF_SCALER_FLAG_NO_VERTICAL`: disable vertical hinting.
pub const SCALER_FLAG_NO_VERTICAL: u32 = 2;
/// `AF_SCALER_FLAG_NO_ADVANCE`: disable advance hinting.
pub const SCALER_FLAG_NO_ADVANCE: u32 = 4;
/// `AF_SCALER_FLAG_NO_WARPER`: disable the warper.
pub const SCALER_FLAG_NO_WARPER: u32 = 8;

/// `AF_ScalerRec` (aftypes.h): models the target pixel device that
/// will receive the auto-hinted glyph image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Scaler {
    /// From font units to 1/64th device pixels.
    pub x_scale: Fixed,
    /// From font units to 1/64th device pixels.
    pub y_scale: Fixed,
    /// In 1/64th device pixels.
    pub x_delta: Pos,
    /// In 1/64th device pixels.
    pub y_delta: Pos,
    /// Monochrome, anti-aliased, LCD, etc.
    pub render_mode: RenderMode,
    /// Additional control flags, `SCALER_FLAG_*`.
    pub flags: u32,
}

impl Default for Scaler {
    #[inline]
    fn default() -> Self {
        Scaler {
            x_scale: 0,
            y_scale: 0,
            x_delta: 0,
            y_delta: 0,
            render_mode: RenderMode::Normal,
            flags: 0,
        }
    }
}

impl Scaler {
    /// `AF_SCALER_EQUAL_SCALES(a, b)`: true when both scalers use the
    /// same scales and deltas.
    #[inline]
    pub fn equal_scales(&self, other: &Scaler) -> bool {
        self.x_scale == other.x_scale
            && self.y_scale == other.y_scale
            && self.x_delta == other.x_delta
            && self.y_delta == other.y_delta
    }
}

/// `AF_BLUE_PROPERTY_LATIN_TOP` (afblue.h): marks a top blue zone.
pub const BLUE_PROPERTY_LATIN_TOP: u16 = 1;
/// `AF_BLUE_PROPERTY_LATIN_NEUTRAL`: marks a neutral blue zone.
pub const BLUE_PROPERTY_LATIN_NEUTRAL: u16 = 2;
/// `AF_BLUE_PROPERTY_LATIN_X_HEIGHT`: marks an x-height blue zone.
pub const BLUE_PROPERTY_LATIN_X_HEIGHT: u16 = 4;
/// `AF_BLUE_PROPERTY_LATIN_LONG`: marks a long (extra tall) blue zone.
pub const BLUE_PROPERTY_LATIN_LONG: u16 = 8;

/// `AF_BLUE_PROPERTY_CJK_TOP` (afblue.h): marks a top/right blue zone.
pub const BLUE_PROPERTY_CJK_TOP: u16 = 1;
/// `AF_BLUE_PROPERTY_CJK_HORIZ`: marks a horizontal blue zone.
pub const BLUE_PROPERTY_CJK_HORIZ: u16 = 2;
/// `AF_BLUE_PROPERTY_CJK_RIGHT`: alias of [`BLUE_PROPERTY_CJK_TOP`].
pub const BLUE_PROPERTY_CJK_RIGHT: u16 = BLUE_PROPERTY_CJK_TOP;

/// Index of a blue string inside [`BLUE_STRINGS`].
///
/// FreeType addresses the strings by their byte offset into the
/// NUL-separated `af_blue_strings` array; storing the string index is
/// equivalent and keeps the table readable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum BlueString {
    /// `AF_BLUE_STRING_ARABIC_TOP`.
    ArabicTop = 0,
    /// `AF_BLUE_STRING_ARABIC_JOIN`.
    ArabicJoin = 1,
    /// `AF_BLUE_STRING_CYRILLIC_CAPITAL_TOP`.
    CyrillicCapitalTop = 2,
    /// `AF_BLUE_STRING_CYRILLIC_CAPITAL_BOTTOM`.
    CyrillicCapitalBottom = 3,
    /// `AF_BLUE_STRING_CYRILLIC_SMALL`.
    CyrillicSmall = 4,
    /// `AF_BLUE_STRING_CYRILLIC_SMALL_DESCENDER`.
    CyrillicSmallDescender = 5,
    /// `AF_BLUE_STRING_DEVANAGARI_BASE`.
    DevanagariBase = 6,
    /// `AF_BLUE_STRING_DEVANAGARI_TOP`.
    DevanagariTop = 7,
    /// `AF_BLUE_STRING_DEVANAGARI_HEAD`.
    DevanagariHead = 8,
    /// `AF_BLUE_STRING_DEVANAGARI_BOTTOM`.
    DevanagariBottom = 9,
    /// `AF_BLUE_STRING_GREEK_CAPITAL_TOP`.
    GreekCapitalTop = 10,
    /// `AF_BLUE_STRING_GREEK_CAPITAL_BOTTOM`.
    GreekCapitalBottom = 11,
    /// `AF_BLUE_STRING_GREEK_SMALL_BETA_TOP`.
    GreekSmallBetaTop = 12,
    /// `AF_BLUE_STRING_GREEK_SMALL`.
    GreekSmall = 13,
    /// `AF_BLUE_STRING_GREEK_SMALL_DESCENDER`.
    GreekSmallDescender = 14,
    /// `AF_BLUE_STRING_HEBREW_TOP`.
    HebrewTop = 15,
    /// `AF_BLUE_STRING_HEBREW_BOTTOM`.
    HebrewBottom = 16,
    /// `AF_BLUE_STRING_HEBREW_DESCENDER`.
    HebrewDescender = 17,
    /// `AF_BLUE_STRING_LATIN_CAPITAL_TOP`.
    LatinCapitalTop = 18,
    /// `AF_BLUE_STRING_LATIN_CAPITAL_BOTTOM`.
    LatinCapitalBottom = 19,
    /// `AF_BLUE_STRING_LATIN_SMALL_F_TOP`.
    LatinSmallFTop = 20,
    /// `AF_BLUE_STRING_LATIN_SMALL`.
    LatinSmall = 21,
    /// `AF_BLUE_STRING_LATIN_SMALL_DESCENDER`.
    LatinSmallDescender = 22,
    /// `AF_BLUE_STRING_TELUGU_TOP`.
    TeluguTop = 23,
    /// `AF_BLUE_STRING_TELUGU_BOTTOM`.
    TeluguBottom = 24,
    /// `AF_BLUE_STRING_THAI_TOP`.
    ThaiTop = 25,
    /// `AF_BLUE_STRING_THAI_BOTTOM`.
    ThaiBottom = 26,
    /// `AF_BLUE_STRING_THAI_ASCENDER`.
    ThaiAscender = 27,
    /// `AF_BLUE_STRING_THAI_LARGE_ASCENDER`.
    ThaiLargeAscender = 28,
    /// `AF_BLUE_STRING_THAI_DESCENDER`.
    ThaiDescender = 29,
    /// `AF_BLUE_STRING_THAI_LARGE_DESCENDER`.
    ThaiLargeDescender = 30,
    /// `AF_BLUE_STRING_THAI_DIGIT_TOP`.
    ThaiDigitTop = 31,
    /// `AF_BLUE_STRING_CJK_TOP`.
    CjkTop = 32,
    /// `AF_BLUE_STRING_CJK_BOTTOM`.
    CjkBottom = 33,
}

/// `af_blue_strings` (afblue.c): the blue strings, NUL separated in
/// FreeType and `&str` separated here (the C strings contain no space
/// characters, so the decoded code point sequences are identical).
pub static BLUE_STRINGS: [&str; 34] = [
    "اإلكطظ",   // ARABIC_TOP
    "تثطظك",    // ARABIC_JOIN
    "БВЕПЗОСЭ", // CYRILLIC_CAPITAL_TOP
    "БВЕШЗОСЭ", // CYRILLIC_CAPITAL_BOTTOM
    "хпншезос", // CYRILLIC_SMALL
    "руф",      // CYRILLIC_SMALL_DESCENDER
    "कमअआथधभश", // DEVANAGARI_BASE
    "ईऐओऔिीोौ", // DEVANAGARI_TOP
    "कमअआथधभश", // DEVANAGARI_HEAD
    "ुृ",         // DEVANAGARI_BOTTOM
    "ΓΒΕΖΘΟΩ",  // GREEK_CAPITAL_TOP
    "ΒΔΖΞΘΟ",   // GREEK_CAPITAL_BOTTOM
    "βθδζλξ",   // GREEK_SMALL_BETA_TOP
    "αειοπστω", // GREEK_SMALL
    "βγημρφχψ", // GREEK_SMALL_DESCENDER
    "בדהחךכםס", // HEBREW_TOP
    "בטכםסצ",   // HEBREW_BOTTOM
    "קךןףץ",    // HEBREW_DESCENDER
    "THEZOCQS", // LATIN_CAPITAL_TOP
    "HEZLOCUS", // LATIN_CAPITAL_BOTTOM
    "fijkdbh",  // LATIN_SMALL_F_TOP
    "xzroesc",  // LATIN_SMALL
    "pqgjy",    // LATIN_SMALL_DESCENDER
    "ఇఌఙఞణఱ౯",  // TELUGU_TOP
    "అకచరఽ౨౬",  // TELUGU_BOTTOM
    "บเแอกา",   // THAI_TOP
    "บปษฯอยฮ",  // THAI_BOTTOM
    "ปฝฟ",      // THAI_ASCENDER
    "โใไ",      // THAI_LARGE_ASCENDER
    "ฎฏฤฦ",     // THAI_DESCENDER
    "ญฐ",       // THAI_LARGE_DESCENDER
    "๐๑๓",      // THAI_DIGIT_TOP
    "他们你來們到和地对對就席我时時會来為能舰說说这這齊|军同已愿既星是景民照现現理用置要軍那配里開雷露面顾",
    // CJK_TOP (51 chars, `AF_BLUE_STRING_MAX_LEN`)
    "个为人他以们你來個們到和大对對就我时時有来為要說说|主些因它想意理生當看着置者自著裡过还进進過道還里面",
    // CJK_BOTTOM (51 chars)
];

/// `AF_Blue_StringRec` (afblue.h): one entry of a style specific blue
/// stringset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlueStringsetEntry {
    /// The string to scan, `None` terminates the set
    /// (`AF_BLUE_STRING_MAX` in FreeType).
    pub string: Option<BlueString>,
    /// `AF_BLUE_PROPERTY_*` bit mask.
    pub properties: u16,
}

/// Entry helper: builds a set entry with the given string and
/// properties.
#[inline]
const fn blue(string: BlueString, properties: u16) -> BlueStringsetEntry {
    BlueStringsetEntry {
        string: Some(string),
        properties,
    }
}

/// Entry helper: builds the end-of-set marker (`AF_BLUE_STRING_MAX`).
#[inline]
const fn blue_end() -> BlueStringsetEntry {
    BlueStringsetEntry {
        string: None,
        properties: 0,
    }
}

/// `af_blue_stringsets` (afblue.c): the per-style blue stringsets.
///
/// Sets are delimited by an entry whose `string` is `None`.
pub static BLUE_STRINGSETS: [BlueStringsetEntry; 47] = [
    // AF_BLUE_STRINGSET_ARAB
    blue(BlueString::ArabicTop, BLUE_PROPERTY_LATIN_TOP),
    blue(BlueString::ArabicJoin, 0),
    blue_end(),
    // AF_BLUE_STRINGSET_CYRL
    blue(BlueString::CyrillicCapitalTop, BLUE_PROPERTY_LATIN_TOP),
    blue(BlueString::CyrillicCapitalBottom, 0),
    blue(
        BlueString::CyrillicSmall,
        BLUE_PROPERTY_LATIN_TOP | BLUE_PROPERTY_LATIN_X_HEIGHT,
    ),
    blue(BlueString::CyrillicSmall, 0),
    blue(BlueString::CyrillicSmallDescender, 0),
    blue_end(),
    // AF_BLUE_STRINGSET_DEVA
    blue(BlueString::DevanagariTop, BLUE_PROPERTY_LATIN_TOP),
    blue(BlueString::DevanagariHead, BLUE_PROPERTY_LATIN_TOP),
    blue(
        BlueString::DevanagariBase,
        BLUE_PROPERTY_LATIN_TOP | BLUE_PROPERTY_LATIN_NEUTRAL | BLUE_PROPERTY_LATIN_X_HEIGHT,
    ),
    blue(BlueString::DevanagariBase, 0),
    blue(BlueString::DevanagariBottom, 0),
    blue_end(),
    // AF_BLUE_STRINGSET_GREK
    blue(BlueString::GreekCapitalTop, BLUE_PROPERTY_LATIN_TOP),
    blue(BlueString::GreekCapitalBottom, 0),
    blue(BlueString::GreekSmallBetaTop, BLUE_PROPERTY_LATIN_TOP),
    blue(
        BlueString::GreekSmall,
        BLUE_PROPERTY_LATIN_TOP | BLUE_PROPERTY_LATIN_X_HEIGHT,
    ),
    blue(BlueString::GreekSmall, 0),
    blue(BlueString::GreekSmallDescender, 0),
    blue_end(),
    // AF_BLUE_STRINGSET_HEBR
    blue(
        BlueString::HebrewTop,
        BLUE_PROPERTY_LATIN_TOP | BLUE_PROPERTY_LATIN_LONG,
    ),
    blue(BlueString::HebrewBottom, 0),
    blue(BlueString::HebrewDescender, 0),
    blue_end(),
    // AF_BLUE_STRINGSET_LATN
    blue(BlueString::LatinCapitalTop, BLUE_PROPERTY_LATIN_TOP),
    blue(BlueString::LatinCapitalBottom, 0),
    blue(BlueString::LatinSmallFTop, BLUE_PROPERTY_LATIN_TOP),
    blue(
        BlueString::LatinSmall,
        BLUE_PROPERTY_LATIN_TOP | BLUE_PROPERTY_LATIN_X_HEIGHT,
    ),
    blue(BlueString::LatinSmall, 0),
    blue(BlueString::LatinSmallDescender, 0),
    blue_end(),
    // AF_BLUE_STRINGSET_TELU
    blue(BlueString::TeluguTop, BLUE_PROPERTY_LATIN_TOP),
    blue(BlueString::TeluguBottom, 0),
    blue_end(),
    // AF_BLUE_STRINGSET_THAI
    blue(
        BlueString::ThaiTop,
        BLUE_PROPERTY_LATIN_TOP | BLUE_PROPERTY_LATIN_X_HEIGHT,
    ),
    blue(BlueString::ThaiBottom, 0),
    blue(BlueString::ThaiAscender, BLUE_PROPERTY_LATIN_TOP),
    blue(BlueString::ThaiLargeAscender, BLUE_PROPERTY_LATIN_TOP),
    blue(BlueString::ThaiDescender, 0),
    blue(BlueString::ThaiLargeDescender, 0),
    blue(BlueString::ThaiDigitTop, 0),
    blue_end(),
    // AF_BLUE_STRINGSET_HANI
    blue(BlueString::CjkTop, BLUE_PROPERTY_CJK_TOP),
    blue(BlueString::CjkBottom, 0),
    blue_end(),
];

/// Offset of each stringset inside [`BLUE_STRINGSETS`]
/// (`AF_BLUE_STRINGSET_*`).
pub mod blue_stringset {
    /// `AF_BLUE_STRINGSET_ARAB`.
    pub const ARAB: usize = 0;
    /// `AF_BLUE_STRINGSET_CYRL`.
    pub const CYRL: usize = 3;
    /// `AF_BLUE_STRINGSET_DEVA`.
    pub const DEVA: usize = 9;
    /// `AF_BLUE_STRINGSET_GREK`.
    pub const GREK: usize = 15;
    /// `AF_BLUE_STRINGSET_HEBR`.
    pub const HEBR: usize = 22;
    /// `AF_BLUE_STRINGSET_LATN`.
    pub const LATN: usize = 26;
    /// `AF_BLUE_STRINGSET_TELU`.
    pub const TELU: usize = 33;
    /// `AF_BLUE_STRINGSET_THAI`.
    pub const THAI: usize = 36;
    /// `AF_BLUE_STRINGSET_HANI`.
    pub const HANI: usize = 44;
    /// `AF_BLUE_STRINGSET_MAX`: number of stringset entries including
    /// the terminators (FreeType sizes `blues[]` with this bound).
    pub const MAX: usize = 48;
}
