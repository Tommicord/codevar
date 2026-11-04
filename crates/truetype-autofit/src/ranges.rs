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

//! Scripts, styles, coverages and writing systems.
//!
//! This module is a port of FreeType's `afranges.c` (the per-script
//! Unicode ranges) together with the class tables that `afglobal.c`
//! assembles from the X-macro headers `afscript.h`, `afstyles.h`,
//! `afcover.h` and `afwrtsys.h`.

/// `AF_Script_UniRangeRec` (aftypes.h): an inclusive Unicode range
/// checked against a font's character map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct UniRange {
    /// First Unicode value of the range.
    pub first: u32,
    /// Last Unicode value of the range.
    pub last: u32,
}

impl UniRange {
    /// Builds a range from its inclusive bounds (`AF_UNIRANGE_REC`).
    #[inline]
    pub const fn new(first: u32, last: u32) -> UniRange {
        UniRange { first, last }
    }
}

/// `AF_Script` (aftypes.h): the known scripts, in `afscript.h` order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Script {
    /// `AF_SCRIPT_ARAB`: Arabic.
    Arab = 0,
    /// `AF_SCRIPT_CYRL`: Cyrillic.
    Cyrl = 1,
    /// `AF_SCRIPT_DEVA`: Devanagari.
    Deva = 2,
    /// `AF_SCRIPT_GREK`: Greek.
    Grek = 3,
    /// `AF_SCRIPT_HEBR`: Hebrew.
    Hebr = 4,
    /// `AF_SCRIPT_LATN`: Latin.
    Latn = 5,
    /// `AF_SCRIPT_NONE`: no script.
    None = 6,
    /// `AF_SCRIPT_TELU`: Telugu.
    Telu = 7,
    /// `AF_SCRIPT_THAI`: Thai.
    Thai = 8,
    /// `AF_SCRIPT_BENG`: Bengali.
    Beng = 9,
    /// `AF_SCRIPT_GUJR`: Gujarati.
    Gujr = 10,
    /// `AF_SCRIPT_GURU`: Gurmukhi.
    Guru = 11,
    /// `AF_SCRIPT_KNDA`: Kannada.
    Knda = 12,
    /// `AF_SCRIPT_LIMB`: Limbu.
    Limb = 13,
    /// `AF_SCRIPT_MLYM`: Malayalam.
    Mlym = 14,
    /// `AF_SCRIPT_ORYA`: Oriya.
    Orya = 15,
    /// `AF_SCRIPT_SINH`: Sinhala.
    Sinh = 16,
    /// `AF_SCRIPT_SUND`: Sundanese.
    Sund = 17,
    /// `AF_SCRIPT_SYLO`: Syloti Nagri.
    Sylo = 18,
    /// `AF_SCRIPT_TAML`: Tamil.
    Taml = 19,
    /// `AF_SCRIPT_TIBT`: Tibetan.
    Tibt = 20,
    /// `AF_SCRIPT_HANI`: CJKV ideographs.
    Hani = 21,
}

/// `AF_SCRIPT_MAX`: the number of known scripts.
pub const SCRIPT_MAX: usize = 22;

/// `AF_SCRIPT_DEFAULT`: the default script (OpenType default;
/// ignored without HarfBuzz).
pub const SCRIPT_DEFAULT: Script = Script::Latn;

impl Script {
    /// Numeric index into [`SCRIPT_CLASSES`].
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Builds a script from its numeric index, mapping unknown values
    /// to [`Script::None`].
    #[inline]
    pub const fn from_index(index: usize) -> Script {
        match index {
            0 => Script::Arab,
            1 => Script::Cyrl,
            2 => Script::Deva,
            3 => Script::Grek,
            4 => Script::Hebr,
            5 => Script::Latn,
            6 => Script::None,
            7 => Script::Telu,
            8 => Script::Thai,
            9 => Script::Beng,
            10 => Script::Gujr,
            11 => Script::Guru,
            12 => Script::Knda,
            13 => Script::Limb,
            14 => Script::Mlym,
            15 => Script::Orya,
            16 => Script::Sinh,
            17 => Script::Sund,
            18 => Script::Sylo,
            19 => Script::Taml,
            20 => Script::Tibt,
            21 => Script::Hani,
            _ => Script::None,
        }
    }
}

/// `AF_ScriptClassRec` (aftypes.h): Unicode ranges and standard
/// characters of one script.
#[derive(Clone, Copy, Debug)]
pub struct ScriptClass {
    /// The script itself.
    pub script: Script,
    /// Unicode ranges covering the script (not terminated in Rust;
    /// [`Script::None`] uses an empty slice instead of `{ 0, 0 }`).
    pub ranges: &'static [UniRange],
    /// First standard character, used to derive stem widths.
    pub standard_char1: u32,
    /// Second standard character, used to derive stem widths.
    pub standard_char2: u32,
    /// Third standard character, used to derive stem widths.
    pub standard_char3: u32,
}

// Unicode ranges, ported from `afranges.c`.

static ARAB_RANGES: &[UniRange] = &[
    UniRange::new(0x0600, 0x06FF),   /* Arabic                                 */
    UniRange::new(0x0750, 0x07FF),   /* Arabic Supplement                      */
    UniRange::new(0x08A0, 0x08FF),   /* Arabic Extended-A                      */
    UniRange::new(0xFB50, 0xFDFF),   /* Arabic Presentation Forms-A            */
    UniRange::new(0xFE70, 0xFEFF),   /* Arabic Presentation Forms-B            */
    UniRange::new(0x1EE00, 0x1EEFF), /* Arabic Mathematical Alphabetic Symbols */
];

static CYRL_RANGES: &[UniRange] = &[
    UniRange::new(0x0400, 0x04FF), /* Cyrillic            */
    UniRange::new(0x0500, 0x052F), /* Cyrillic Supplement */
    UniRange::new(0x2DE0, 0x2DFF), /* Cyrillic Extended-A */
    UniRange::new(0xA640, 0xA69F), /* Cyrillic Extended-B */
];

// There are some characters in the Devanagari Unicode block that are
// generic to Indic scripts; we omit them so that their presence doesn't
// trigger Devanagari

static DEVA_RANGES: &[UniRange] = &[
    UniRange::new(0x0900, 0x093B), /* Devanagari (omitting U+093C nukta)      */
    UniRange::new(0x093D, 0x0950), /* (omitting U+0951 udatta, U+0952 anudatta) */
    UniRange::new(0x0953, 0x0963), /* (omitting U+0964 danda, U+0965 double danda) */
    UniRange::new(0x0966, 0x097F),
    UniRange::new(0x20B9, 0x20B9), /* (new) Rupee sign */
];

static GREK_RANGES: &[UniRange] = &[
    UniRange::new(0x0370, 0x03FF), /* Greek and Coptic */
    UniRange::new(0x1F00, 0x1FFF), /* Greek Extended   */
];

static HEBR_RANGES: &[UniRange] = &[
    UniRange::new(0x0590, 0x05FF), /* Hebrew                          */
    UniRange::new(0xFB1D, 0xFB4F), /* Alphab. Present. Forms (Hebrew) */
];

static LATN_RANGES: &[UniRange] = &[
    UniRange::new(0x0020, 0x007F),   /* Basic Latin (no control chars)         */
    UniRange::new(0x00A0, 0x00FF),   /* Latin-1 Supplement (no control chars)  */
    UniRange::new(0x0100, 0x017F),   /* Latin Extended-A                       */
    UniRange::new(0x0180, 0x024F),   /* Latin Extended-B                       */
    UniRange::new(0x0250, 0x02AF),   /* IPA Extensions                         */
    UniRange::new(0x02B0, 0x02FF),   /* Spacing Modifier Letters               */
    UniRange::new(0x0300, 0x036F),   /* Combining Diacritical Marks            */
    UniRange::new(0x1D00, 0x1D7F),   /* Phonetic Extensions                    */
    UniRange::new(0x1D80, 0x1DBF),   /* Phonetic Extensions Supplement         */
    UniRange::new(0x1DC0, 0x1DFF),   /* Combining Diacritical Marks Supplement */
    UniRange::new(0x1E00, 0x1EFF),   /* Latin Extended Additional              */
    UniRange::new(0x2000, 0x206F),   /* General Punctuation                    */
    UniRange::new(0x2070, 0x209F),   /* Superscripts and Subscripts            */
    UniRange::new(0x20A0, 0x20B8),   /* Currency Symbols ...                   */
    UniRange::new(0x20BA, 0x20CF),   /* ... except new Rupee sign              */
    UniRange::new(0x2150, 0x218F),   /* Number Forms                           */
    UniRange::new(0x2460, 0x24FF),   /* Enclosed Alphanumerics                 */
    UniRange::new(0x2C60, 0x2C7F),   /* Latin Extended-C                       */
    UniRange::new(0x2E00, 0x2E7F),   /* Supplemental Punctuation               */
    UniRange::new(0xA720, 0xA7FF),   /* Latin Extended-D                       */
    UniRange::new(0xFB00, 0xFB06),   /* Alphab. Present. Forms (Latin Ligs)    */
    UniRange::new(0x1D400, 0x1D7FF), /* Mathematical Alphanumeric Symbols      */
    UniRange::new(0x1F100, 0x1F1FF), /* Enclosed Alphanumeric Supplement       */
];

static NONE_RANGES: &[UniRange] = &[];

static TELU_RANGES: &[UniRange] = &[UniRange::new(0x0C00, 0x0C7F)]; /* Telugu */

static THAI_RANGES: &[UniRange] = &[UniRange::new(0x0E00, 0x0E7F)]; /* Thai */

static BENG_RANGES: &[UniRange] = &[UniRange::new(0x0980, 0x09FF)]; /* Bengali */

static GUJR_RANGES: &[UniRange] = &[UniRange::new(0x0A80, 0x0AFF)]; /* Gujarati */

static GURU_RANGES: &[UniRange] = &[UniRange::new(0x0A00, 0x0A7F)]; /* Gurmukhi */

static KNDA_RANGES: &[UniRange] = &[UniRange::new(0x0C80, 0x0CFF)]; /* Kannada */

static LIMB_RANGES: &[UniRange] = &[UniRange::new(0x1900, 0x194F)]; /* Limbu */

static MLYM_RANGES: &[UniRange] = &[UniRange::new(0x0D00, 0x0D7F)]; /* Malayalam */

static ORYA_RANGES: &[UniRange] = &[UniRange::new(0x0B00, 0x0B7F)]; /* Oriya */

static SINH_RANGES: &[UniRange] = &[UniRange::new(0x0D80, 0x0DFF)]; /* Sinhala */

static SUND_RANGES: &[UniRange] = &[UniRange::new(0x1B80, 0x1BBF)]; /* Sundanese */

static SYLO_RANGES: &[UniRange] = &[UniRange::new(0xA800, 0xA82F)]; /* Syloti Nagri */

static TAML_RANGES: &[UniRange] = &[UniRange::new(0x0B80, 0x0BFF)]; /* Tamil */

static TIBT_RANGES: &[UniRange] = &[UniRange::new(0x0F00, 0x0FFF)]; /* Tibetan */

// this corresponds to Unicode 6.0

#[allow(clippy::too_many_lines)]
static HANI_RANGES: &[UniRange] = &[
    UniRange::new(0x1100, 0x11FF),   /* Hangul Jamo                             */
    UniRange::new(0x2E80, 0x2EFF),   /* CJK Radicals Supplement                 */
    UniRange::new(0x2F00, 0x2FDF),   /* Kangxi Radicals                         */
    UniRange::new(0x2FF0, 0x2FFF),   /* Ideographic Description Characters      */
    UniRange::new(0x3000, 0x303F),   /* CJK Symbols and Punctuation             */
    UniRange::new(0x3040, 0x309F),   /* Hiragana                                */
    UniRange::new(0x30A0, 0x30FF),   /* Katakana                                */
    UniRange::new(0x3100, 0x312F),   /* Bopomofo                                */
    UniRange::new(0x3130, 0x318F),   /* Hangul Compatibility Jamo               */
    UniRange::new(0x3190, 0x319F),   /* Kanbun                                  */
    UniRange::new(0x31A0, 0x31BF),   /* Bopomofo Extended                       */
    UniRange::new(0x31C0, 0x31EF),   /* CJK Strokes                             */
    UniRange::new(0x31F0, 0x31FF),   /* Katakana Phonetic Extensions            */
    UniRange::new(0x3200, 0x32FF),   /* Enclosed CJK Letters and Months         */
    UniRange::new(0x3300, 0x33FF),   /* CJK Compatibility                       */
    UniRange::new(0x3400, 0x4DBF),   /* CJK Unified Ideographs Extension A      */
    UniRange::new(0x4DC0, 0x4DFF),   /* Yijing Hexagram Symbols                 */
    UniRange::new(0x4E00, 0x9FFF),   /* CJK Unified Ideographs                  */
    UniRange::new(0xA960, 0xA97F),   /* Hangul Jamo Extended-A                  */
    UniRange::new(0xAC00, 0xD7AF),   /* Hangul Syllables                        */
    UniRange::new(0xD7B0, 0xD7FF),   /* Hangul Jamo Extended-B                  */
    UniRange::new(0xF900, 0xFAFF),   /* CJK Compatibility Ideographs            */
    UniRange::new(0xFE10, 0xFE1F),   /* Vertical forms                          */
    UniRange::new(0xFE30, 0xFE4F),   /* CJK Compatibility Forms                 */
    UniRange::new(0xFF00, 0xFFEF),   /* Halfwidth and Fullwidth Forms           */
    UniRange::new(0x1B000, 0x1B0FF), /* Kana Supplement                         */
    UniRange::new(0x1D300, 0x1D35F), /* Tai Xuan Hing Symbols                   */
    UniRange::new(0x1F200, 0x1F2FF), /* Enclosed Ideographic Supplement         */
    UniRange::new(0x20000, 0x2A6DF), /* CJK Unified Ideographs Extension B      */
    UniRange::new(0x2A700, 0x2B73F), /* CJK Unified Ideographs Extension C      */
    UniRange::new(0x2B740, 0x2B81F), /* CJK Unified Ideographs Extension D      */
    UniRange::new(0x2F800, 0x2FA1F), /* CJK Compatibility Ideographs Supplement */
];

/// `af_script_classes[]` (afglobal.c, assembled from `afscript.h`):
/// the class of every known script, indexed by [`Script`].
pub static SCRIPT_CLASSES: [ScriptClass; SCRIPT_MAX] = [
    ScriptClass {
        script: Script::Arab,
        ranges: ARAB_RANGES,
        standard_char1: 0x644, /* ل */
        standard_char2: 0x62D, /* ح */
        standard_char3: 0x640, /* ـ */
    },
    ScriptClass {
        script: Script::Cyrl,
        ranges: CYRL_RANGES,
        standard_char1: 0x43E, /* о */
        standard_char2: 0x41E, /* О */
        standard_char3: 0x0,
    },
    ScriptClass {
        script: Script::Deva,
        ranges: DEVA_RANGES,
        standard_char1: 0x920, /* ठ */
        standard_char2: 0x935, /* व */
        standard_char3: 0x91F, /* ट */
    },
    ScriptClass {
        script: Script::Grek,
        ranges: GREK_RANGES,
        standard_char1: 0x3BF, /* ο */
        standard_char2: 0x39F, /* Ο */
        standard_char3: 0x0,
    },
    ScriptClass {
        script: Script::Hebr,
        ranges: HEBR_RANGES,
        standard_char1: 0x5DD, /* ם */
        standard_char2: 0x0,
        standard_char3: 0x0,
    },
    ScriptClass {
        script: Script::Latn,
        ranges: LATN_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 'O' as u32,
        standard_char3: '0' as u32,
    },
    ScriptClass {
        script: Script::None,
        ranges: NONE_RANGES,
        standard_char1: 0x0,
        standard_char2: 0x0,
        standard_char3: 0x0,
    },
    ScriptClass {
        // There are no simple forms for letters; we thus use two digit shapes
        script: Script::Telu,
        ranges: TELU_RANGES,
        standard_char1: 0xC66, /* ౦ */
        standard_char2: 0xC67, /* ౧ */
        standard_char3: 0x0,
    },
    ScriptClass {
        script: Script::Thai,
        ranges: THAI_RANGES,
        standard_char1: 0xE32, /* า */
        standard_char2: 0xE45, /* ๅ */
        standard_char3: 0xE50, /* ๐ */
    },
    ScriptClass {
        script: Script::Beng,
        ranges: BENG_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Gujr,
        ranges: GUJR_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Guru,
        ranges: GURU_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Knda,
        ranges: KNDA_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Limb,
        ranges: LIMB_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Mlym,
        ranges: MLYM_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Orya,
        ranges: ORYA_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Sinh,
        ranges: SINH_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Sund,
        ranges: SUND_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Sylo,
        ranges: SYLO_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Taml,
        ranges: TAML_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Tibt,
        ranges: TIBT_RANGES,
        standard_char1: 'o' as u32,
        standard_char2: 0x0,
        standard_char3: 0x0, /* XXX */
    },
    ScriptClass {
        script: Script::Hani,
        ranges: HANI_RANGES,
        standard_char1: 0x7530, /* 田 */
        standard_char2: 0x56D7, /* 囗 */
        standard_char3: 0x0,
    },
];

/// `AF_Coverage` (aftypes.h, assembled from `afcover.h`): an OpenType
/// feature group whose glyphs must be hinted together.
///
/// Without HarfBuzz the auto-hinter never computes non-default
/// coverages; every style effectively behaves like
/// [`Coverage::Default`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Coverage {
    /// `AF_COVERAGE_PETITE_CAPITALS_FROM_CAPITALS` (`c2cp`).
    PetiteCapitalsFromCapitals = 0,
    /// `AF_COVERAGE_SMALL_CAPITALS_FROM_CAPITALS` (`c2sc`).
    SmallCapitalsFromCapitals = 1,
    /// `AF_COVERAGE_ORDINALS` (`ordn`).
    Ordinals = 2,
    /// `AF_COVERAGE_PETITE_CAPITALS` (`pcap`).
    PetiteCapitals = 3,
    /// `AF_COVERAGE_RUBY` (`ruby`).
    Ruby = 4,
    /// `AF_COVERAGE_SCIENTIFIC_INFERIORS` (`sinf`).
    ScientificInferiors = 5,
    /// `AF_COVERAGE_SMALL_CAPITALS` (`smcp`).
    SmallCapitals = 6,
    /// `AF_COVERAGE_SUBSCRIPT` (`subs`).
    Subscript = 7,
    /// `AF_COVERAGE_SUPERSCRIPT` (`sups`).
    Superscript = 8,
    /// `AF_COVERAGE_TITLING` (`titl`).
    Titling = 9,
    /// `AF_COVERAGE_DEFAULT`: everything not listed separately,
    /// including all glyphs addressable by the character map.
    Default = 10,
}

/// `AF_STYLE_MAX` (`aftypes.h`): the number of known styles.
pub const STYLE_MAX: usize = 49;

/// `AF_STYLE_FALLBACK` (`afglobal.h`): index of the fallback style
/// used for uncovered glyphs (`AF_STYLE_HANI_DFLT` with CJK support).
pub const STYLE_FALLBACK: usize = 48;

/// `AF_STYLE_UNASSIGNED` (`afglobal.h`): bit mask indicating an
/// uncovered glyph inside `glyph_styles[]`.
pub const STYLE_UNASSIGNED: u8 = 0x7F;

/// `AF_DIGIT` (`afglobal.h`): flag marking an ASCII digit inside
/// `glyph_styles[]`.
pub const DIGIT: u8 = 0x80;

/// `AF_Style` (aftypes.h): the known styles, in `afstyles.h` order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Style {
    /// `AF_STYLE_ARAB_DFLT`: Arabic default style.
    ArabDflt = 0,
    /// `AF_STYLE_CYRL_C2CP`: Cyrillic petite capticals from capitals style.
    CyrlC2cp = 1,
    /// `AF_STYLE_CYRL_C2SC`: Cyrillic small capticals from capitals style.
    CyrlC2sc = 2,
    /// `AF_STYLE_CYRL_ORDN`: Cyrillic ordinals style.
    CyrlOrdn = 3,
    /// `AF_STYLE_CYRL_PCAP`: Cyrillic petite capitals style.
    CyrlPcap = 4,
    /// `AF_STYLE_CYRL_SINF`: Cyrillic scientific inferiors style.
    CyrlSinf = 5,
    /// `AF_STYLE_CYRL_SMCP`: Cyrillic small capitals style.
    CyrlSmcp = 6,
    /// `AF_STYLE_CYRL_SUBS`: Cyrillic subscript style.
    CyrlSubs = 7,
    /// `AF_STYLE_CYRL_SUPS`: Cyrillic superscript style.
    CyrlSups = 8,
    /// `AF_STYLE_CYRL_TITL`: Cyrillic titling style.
    CyrlTitl = 9,
    /// `AF_STYLE_CYRL_DFLT`: Cyrillic default style.
    CyrlDflt = 10,
    /// `AF_STYLE_GREK_C2CP`: Greek petite capticals from capitals style.
    GrekC2cp = 11,
    /// `AF_STYLE_GREK_C2SC`: Greek small capticals from capitals style.
    GrekC2sc = 12,
    /// `AF_STYLE_GREK_ORDN`: Greek ordinals style.
    GrekOrdn = 13,
    /// `AF_STYLE_GREK_PCAP`: Greek petite capitals style.
    GrekPcap = 14,
    /// `AF_STYLE_GREK_SINF`: Greek scientific inferiors style.
    GrekSinf = 15,
    /// `AF_STYLE_GREK_SMCP`: Greek small capitals style.
    GrekSmcp = 16,
    /// `AF_STYLE_GREK_SUBS`: Greek subscript style.
    GrekSubs = 17,
    /// `AF_STYLE_GREK_SUPS`: Greek superscript style.
    GrekSups = 18,
    /// `AF_STYLE_GREK_TITL`: Greek titling style.
    GrekTitl = 19,
    /// `AF_STYLE_GREK_DFLT`: Greek default style.
    GrekDflt = 20,
    /// `AF_STYLE_HEBR_DFLT`: Hebrew default style.
    HebrDflt = 21,
    /// `AF_STYLE_LATN_C2CP`: Latin petite capticals from capitals style.
    LatnC2cp = 22,
    /// `AF_STYLE_LATN_C2SC`: Latin small capticals from capitals style.
    LatnC2sc = 23,
    /// `AF_STYLE_LATN_ORDN`: Latin ordinals style.
    LatnOrdn = 24,
    /// `AF_STYLE_LATN_PCAP`: Latin petite capitals style.
    LatnPcap = 25,
    /// `AF_STYLE_LATN_SINF`: Latin scientific inferiors style.
    LatnSinf = 26,
    /// `AF_STYLE_LATN_SMCP`: Latin small capitals style.
    LatnSmcp = 27,
    /// `AF_STYLE_LATN_SUBS`: Latin subscript style.
    LatnSubs = 28,
    /// `AF_STYLE_LATN_SUPS`: Latin superscript style.
    LatnSups = 29,
    /// `AF_STYLE_LATN_TITL`: Latin titling style.
    LatnTitl = 30,
    /// `AF_STYLE_LATN_DFLT`: Latin default style.
    LatnDflt = 31,
    /// `AF_STYLE_DEVA_DFLT`: Devanagari default style.
    DevaDflt = 32,
    /// `AF_STYLE_NONE_DFLT`: no style (dummy writing system).
    NoneDflt = 33,
    /// `AF_STYLE_TELU_DFLT`: Telugu default style.
    TeluDflt = 34,
    /// `AF_STYLE_THAI_DFLT`: Thai default style.
    ThaiDflt = 35,
    /// `AF_STYLE_BENG_DFLT`: Bengali default style.
    BengDflt = 36,
    /// `AF_STYLE_GUJR_DFLT`: Gujarati default style.
    GujrDflt = 37,
    /// `AF_STYLE_GURU_DFLT`: Gurmukhi default style.
    GuruDflt = 38,
    /// `AF_STYLE_KNDA_DFLT`: Kannada default style.
    KndaDflt = 39,
    /// `AF_STYLE_LIMB_DFLT`: Limbu default style.
    LimbDflt = 40,
    /// `AF_STYLE_MLYM_DFLT`: Malayalam default style.
    MlymDflt = 41,
    /// `AF_STYLE_ORYA_DFLT`: Oriya default style.
    OryaDflt = 42,
    /// `AF_STYLE_SINH_DFLT`: Sinhala default style.
    SinhDflt = 43,
    /// `AF_STYLE_SUND_DFLT`: Sundanese default style.
    SundDflt = 44,
    /// `AF_STYLE_SYLO_DFLT`: Syloti Nagri default style.
    SyloDflt = 45,
    /// `AF_STYLE_TAML_DFLT`: Tamil default style.
    TamlDflt = 46,
    /// `AF_STYLE_TIBT_DFLT`: Tibetan default style.
    TibtDflt = 47,
    /// `AF_STYLE_HANI_DFLT`: CJKV ideographs default style.
    HaniDflt = 48,
}

impl Style {
    /// Numeric index into [`STYLE_CLASSES`].
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// `AF_WritingSystem` (aftypes.h, assembled from `afwrtsys.h`): the
/// known writing systems.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum WritingSystem {
    /// `AF_WRITING_SYSTEM_DUMMY`: no hinting at all.
    Dummy = 0,
    /// `AF_WRITING_SYSTEM_LATIN`: Western and other alphabetic scripts.
    Latin = 1,
    /// `AF_WRITING_SYSTEM_CJK`: East Asian ideographs.
    Cjk = 2,
    /// `AF_WRITING_SYSTEM_INDIC`: Indic scripts (delegates to CJK).
    Indic = 3,
}

/// `AF_WRITING_SYSTEM_MAX`: the number of known writing systems.
pub const WRITING_SYSTEM_MAX: usize = 4;

impl WritingSystem {
    /// Numeric index into the writing system class table.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// `AF_StyleClassRec` (aftypes.h): topmost grouping of one style.
#[derive(Clone, Copy, Debug)]
pub struct StyleClass {
    /// The style itself.
    pub style: Style,
    /// The writing system handling this style.
    pub writing_system: WritingSystem,
    /// The script of this style.
    pub script: Script,
    /// Offset into [`crate::BLUE_STRINGSETS`] for this style's blue
    /// zones (unused styles pass `(AF_Blue_Stringset)0` in C, which is
    /// represented as `None` because they never scan blue strings).
    pub blue_stringset: Option<usize>,
    /// The OpenType coverage of this style.
    pub coverage: Coverage,
    /// Human readable style name (`af_style_names[]`, the literal
    /// macro argument from `afstyles.h`).
    pub name: &'static str,
}

/// Entry helper: a `STYLE_LATIN` instance.
const fn latin_style(
    style: Style,
    script: Script,
    blue_stringset: usize,
    coverage: Coverage,
    name: &'static str,
) -> StyleClass {
    StyleClass {
        style,
        writing_system: WritingSystem::Latin,
        script,
        blue_stringset: Some(blue_stringset),
        coverage,
        name,
    }
}

/// Entry helper: a `STYLE_DEFAULT_INDIC` instance (no blue stringset).
const fn indic_style(style: Style, script: Script, name: &'static str) -> StyleClass {
    StyleClass {
        style,
        writing_system: WritingSystem::Indic,
        script,
        blue_stringset: None,
        coverage: Coverage::Default,
        name,
    }
}

/// `af_style_classes[]` (afglobal.c, assembled from `afstyles.h`):
/// the class of every known style, indexed by [`Style`].
#[allow(clippy::too_many_lines)]
pub static STYLE_CLASSES: [StyleClass; STYLE_MAX] = [
    StyleClass {
        style: Style::ArabDflt,
        writing_system: WritingSystem::Latin,
        script: Script::Arab,
        blue_stringset: Some(crate::blue_stringset::ARAB),
        coverage: Coverage::Default,
        name: "arab_dflt",
    },
    latin_style(
        Style::CyrlC2cp,
        Script::Cyrl,
        crate::blue_stringset::CYRL,
        Coverage::PetiteCapitalsFromCapitals,
        "cyrl_c2cp",
    ),
    latin_style(
        Style::CyrlC2sc,
        Script::Cyrl,
        crate::blue_stringset::CYRL,
        Coverage::SmallCapitalsFromCapitals,
        "cyrl_c2sc",
    ),
    latin_style(
        Style::CyrlOrdn,
        Script::Cyrl,
        crate::blue_stringset::CYRL,
        Coverage::Ordinals,
        "cyrl_ordn",
    ),
    latin_style(
        Style::CyrlPcap,
        Script::Cyrl,
        crate::blue_stringset::CYRL,
        Coverage::PetiteCapitals,
        "cyrl_pcap",
    ),
    latin_style(
        Style::CyrlSinf,
        Script::Cyrl,
        crate::blue_stringset::CYRL,
        Coverage::ScientificInferiors,
        "cyrl_sinf",
    ),
    latin_style(
        Style::CyrlSmcp,
        Script::Cyrl,
        crate::blue_stringset::CYRL,
        Coverage::SmallCapitals,
        "cyrl_smcp",
    ),
    latin_style(
        Style::CyrlSubs,
        Script::Cyrl,
        crate::blue_stringset::CYRL,
        Coverage::Subscript,
        "cyrl_subs",
    ),
    latin_style(
        Style::CyrlSups,
        Script::Cyrl,
        crate::blue_stringset::CYRL,
        Coverage::Superscript,
        "cyrl_sups",
    ),
    latin_style(
        Style::CyrlTitl,
        Script::Cyrl,
        crate::blue_stringset::CYRL,
        Coverage::Titling,
        "cyrl_titl",
    ),
    latin_style(
        Style::CyrlDflt,
        Script::Cyrl,
        crate::blue_stringset::CYRL,
        Coverage::Default,
        "cyrl_dflt",
    ),
    latin_style(
        Style::GrekC2cp,
        Script::Grek,
        crate::blue_stringset::GREK,
        Coverage::PetiteCapitalsFromCapitals,
        "grek_c2cp",
    ),
    latin_style(
        Style::GrekC2sc,
        Script::Grek,
        crate::blue_stringset::GREK,
        Coverage::SmallCapitalsFromCapitals,
        "grek_c2sc",
    ),
    latin_style(
        Style::GrekOrdn,
        Script::Grek,
        crate::blue_stringset::GREK,
        Coverage::Ordinals,
        "grek_ordn",
    ),
    latin_style(
        Style::GrekPcap,
        Script::Grek,
        crate::blue_stringset::GREK,
        Coverage::PetiteCapitals,
        "grek_pcap",
    ),
    latin_style(
        Style::GrekSinf,
        Script::Grek,
        crate::blue_stringset::GREK,
        Coverage::ScientificInferiors,
        "grek_sinf",
    ),
    latin_style(
        Style::GrekSmcp,
        Script::Grek,
        crate::blue_stringset::GREK,
        Coverage::SmallCapitals,
        "grek_smcp",
    ),
    latin_style(
        Style::GrekSubs,
        Script::Grek,
        crate::blue_stringset::GREK,
        Coverage::Subscript,
        "grek_subs",
    ),
    latin_style(
        Style::GrekSups,
        Script::Grek,
        crate::blue_stringset::GREK,
        Coverage::Superscript,
        "grek_sups",
    ),
    latin_style(
        Style::GrekTitl,
        Script::Grek,
        crate::blue_stringset::GREK,
        Coverage::Titling,
        "grek_titl",
    ),
    latin_style(
        Style::GrekDflt,
        Script::Grek,
        crate::blue_stringset::GREK,
        Coverage::Default,
        "grek_dflt",
    ),
    StyleClass {
        style: Style::HebrDflt,
        writing_system: WritingSystem::Latin,
        script: Script::Hebr,
        blue_stringset: Some(crate::blue_stringset::HEBR),
        coverage: Coverage::Default,
        name: "hebr_dflt",
    },
    latin_style(
        Style::LatnC2cp,
        Script::Latn,
        crate::blue_stringset::LATN,
        Coverage::PetiteCapitalsFromCapitals,
        "latn_c2cp",
    ),
    latin_style(
        Style::LatnC2sc,
        Script::Latn,
        crate::blue_stringset::LATN,
        Coverage::SmallCapitalsFromCapitals,
        "latn_c2sc",
    ),
    latin_style(
        Style::LatnOrdn,
        Script::Latn,
        crate::blue_stringset::LATN,
        Coverage::Ordinals,
        "latn_ordn",
    ),
    latin_style(
        Style::LatnPcap,
        Script::Latn,
        crate::blue_stringset::LATN,
        Coverage::PetiteCapitals,
        "latn_pcap",
    ),
    latin_style(
        Style::LatnSinf,
        Script::Latn,
        crate::blue_stringset::LATN,
        Coverage::ScientificInferiors,
        "latn_sinf",
    ),
    latin_style(
        Style::LatnSmcp,
        Script::Latn,
        crate::blue_stringset::LATN,
        Coverage::SmallCapitals,
        "latn_smcp",
    ),
    latin_style(
        Style::LatnSubs,
        Script::Latn,
        crate::blue_stringset::LATN,
        Coverage::Subscript,
        "latn_subs",
    ),
    latin_style(
        Style::LatnSups,
        Script::Latn,
        crate::blue_stringset::LATN,
        Coverage::Superscript,
        "latn_sups",
    ),
    latin_style(
        Style::LatnTitl,
        Script::Latn,
        crate::blue_stringset::LATN,
        Coverage::Titling,
        "latn_titl",
    ),
    latin_style(
        Style::LatnDflt,
        Script::Latn,
        crate::blue_stringset::LATN,
        Coverage::Default,
        "latn_dflt",
    ),
    StyleClass {
        style: Style::DevaDflt,
        writing_system: WritingSystem::Latin,
        script: Script::Deva,
        blue_stringset: Some(crate::blue_stringset::DEVA),
        coverage: Coverage::Default,
        name: "deva_dflt",
    },
    StyleClass {
        style: Style::NoneDflt,
        writing_system: WritingSystem::Dummy,
        script: Script::None,
        blue_stringset: None,
        coverage: Coverage::Default,
        name: "none_dflt",
    },
    StyleClass {
        style: Style::TeluDflt,
        writing_system: WritingSystem::Latin,
        script: Script::Telu,
        blue_stringset: Some(crate::blue_stringset::TELU),
        coverage: Coverage::Default,
        name: "telu_dflt",
    },
    StyleClass {
        style: Style::ThaiDflt,
        writing_system: WritingSystem::Latin,
        script: Script::Thai,
        blue_stringset: Some(crate::blue_stringset::THAI),
        coverage: Coverage::Default,
        name: "thai_dflt",
    },
    indic_style(Style::BengDflt, Script::Beng, "beng_dflt"),
    indic_style(Style::GujrDflt, Script::Gujr, "gujr_dflt"),
    indic_style(Style::GuruDflt, Script::Guru, "guru_dflt"),
    indic_style(Style::KndaDflt, Script::Knda, "knda_dflt"),
    indic_style(Style::LimbDflt, Script::Limb, "limb_dflt"),
    indic_style(Style::MlymDflt, Script::Mlym, "mlym_dflt"),
    indic_style(Style::OryaDflt, Script::Orya, "orya_dflt"),
    indic_style(Style::SinhDflt, Script::Sinh, "sinh_dflt"),
    indic_style(Style::SundDflt, Script::Sund, "sund_dflt"),
    indic_style(Style::SyloDflt, Script::Sylo, "sylo_dflt"),
    indic_style(Style::TamlDflt, Script::Taml, "taml_dflt"),
    indic_style(Style::TibtDflt, Script::Tibt, "tibt_dflt"),
    StyleClass {
        style: Style::HaniDflt,
        writing_system: WritingSystem::Cjk,
        script: Script::Hani,
        blue_stringset: Some(crate::blue_stringset::HANI),
        coverage: Coverage::Default,
        name: "hani_dflt",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_classes_match_their_style_index() {
        assert_eq!(STYLE_CLASSES.len(), STYLE_MAX);
        for (index, class) in STYLE_CLASSES.iter().enumerate() {
            assert_eq!(class.style.index(), index);
            assert!(!class.name.is_empty());
        }
    }

    #[test]
    fn script_classes_match_their_script_index() {
        assert_eq!(SCRIPT_CLASSES.len(), SCRIPT_MAX);
        for (index, class) in SCRIPT_CLASSES.iter().enumerate() {
            assert_eq!(class.script.index(), index);
        }
    }

    #[test]
    fn script_ranges_are_bounded_sorted_and_disjoint() {
        for class in &SCRIPT_CLASSES {
            let ranges = class.ranges;
            for range in ranges {
                assert!(range.first <= range.last, "{:?}", class.script);
            }
            for pair in ranges.windows(2) {
                assert!(
                    pair[0].last < pair[1].first,
                    "ranges of {:?} must not overlap: {:#x}..{:#x} then {:#x}..{:#x}",
                    class.script,
                    pair[0].first,
                    pair[0].last,
                    pair[1].first,
                    pair[1].last
                );
            }
        }
    }

    #[test]
    fn script_index_roundtrip_maps_unknown_to_none() {
        for index in 0..SCRIPT_MAX {
            assert_eq!(Script::from_index(index).index(), index);
        }
        assert_eq!(Script::from_index(SCRIPT_MAX), Script::None);
        assert_eq!(Script::from_index(999), Script::None);
        assert_eq!(Script::None.index(), 6);
        assert_eq!(SCRIPT_DEFAULT, Script::Latn);
    }

    #[test]
    fn writing_system_indices_match_the_enum_order() {
        assert_eq!(WritingSystem::Dummy.index(), 0);
        assert_eq!(WritingSystem::Latin.index(), 1);
        assert_eq!(WritingSystem::Cjk.index(), 2);
        assert_eq!(WritingSystem::Indic.index(), 3);
        assert_eq!(WRITING_SYSTEM_MAX, 4);
    }

    #[test]
    fn fallback_style_is_cjk_hani_dflt() {
        assert_eq!(STYLE_MAX, 49);
        assert_eq!(STYLE_FALLBACK, 48);
        let class = &STYLE_CLASSES[STYLE_FALLBACK];
        assert_eq!(class.style, Style::HaniDflt);
        assert_eq!(class.style.index(), STYLE_FALLBACK);
        assert_eq!(class.writing_system, WritingSystem::Cjk);
        assert_eq!(class.script, Script::Hani);
        assert_eq!(class.blue_stringset, Some(crate::blue_stringset::HANI));
        assert_eq!(class.name, "hani_dflt");
    }

    #[test]
    fn style_blue_stringset_offsets_are_valid() {
        for class in &STYLE_CLASSES {
            if let Some(offset) = class.blue_stringset {
                assert!(offset < crate::BLUE_STRINGSETS.len());
                assert!(
                    crate::BLUE_STRINGSETS[offset].string.is_some(),
                    "set of {} must not start with a terminator",
                    class.name
                );
            }
        }
    }

    #[test]
    fn unassigned_and_digit_flags_do_not_collide() {
        assert_eq!(STYLE_UNASSIGNED, 0x7F);
        assert_eq!(DIGIT, 0x80);
        assert_eq!(STYLE_UNASSIGNED & DIGIT, 0);
    }

    #[test]
    fn uni_range_new_stores_bounds() {
        let range = UniRange::new(0x41, 0x5A);
        assert_eq!(range.first, 0x41);
        assert_eq!(range.last, 0x5A);
        assert_eq!(
            range,
            UniRange {
                first: 0x41,
                last: 0x5A
            }
        );
    }
}
