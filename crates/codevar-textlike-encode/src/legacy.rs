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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See the
//! License for the specific language governing permissions and
//! limitations under the License.
//!
//! The encodings of the Encoding Standard
//! (<https://encoding.spec.whatwg.org/>) that this crate implements, and the
//! lookup of the standard's "get an encoding" algorithm.
//!
//! Each `Encoding` below pairs a normative pointer table from
//! [`crate::single_byte_tables`] with the run parameters that let
//! [`crate::single_byte::SingleByteEncoder`] map the contiguous part of the
//! table arithmetically instead of searching it. The run was chosen as the
//! longest ascending stretch of defined pointers, so the search only ever sees
//! the slots the run leaves out.
//!
//! [`lookup`] resolves a label exactly as the standard does: leading and
//! trailing ASCII whitespace is removed and the remainder is matched
//! ASCII-case-insensitively against the label table.

use crate::encoding::{Encoding, UTF_8, UTF_16BE, UTF_16LE, VariantEncoding};
use crate::single_byte_tables;

/// The `IBM866` encoding of the Encoding Standard: the INDEX_IBM866 table,
/// with the encoder run of 48 code points from U+0410 at pointer 0x00.
pub static IBM866: Encoding = Encoding {
    name: "IBM866",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_IBM866, 0x0410, 0x00, 48),
};

/// The `ISO-8859-2` encoding of the Encoding Standard: the INDEX_ISO_8859_2 table,
/// with the encoder run of 33 code points from U+0080 at pointer 0x00.
pub static ISO_8859_2: Encoding = Encoding {
    name: "ISO-8859-2",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_2, 0x0080, 0x00, 33),
};

/// The `ISO-8859-3` encoding of the Encoding Standard: the INDEX_ISO_8859_3 table,
/// with the encoder run of 33 code points from U+0080 at pointer 0x00.
pub static ISO_8859_3: Encoding = Encoding {
    name: "ISO-8859-3",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_3, 0x0080, 0x00, 33),
};

/// The `ISO-8859-4` encoding of the Encoding Standard: the INDEX_ISO_8859_4 table,
/// with the encoder run of 33 code points from U+0080 at pointer 0x00.
pub static ISO_8859_4: Encoding = Encoding {
    name: "ISO-8859-4",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_4, 0x0080, 0x00, 33),
};

/// The `ISO-8859-5` encoding of the Encoding Standard: the INDEX_ISO_8859_5 table,
/// with the encoder run of 66 code points from U+040E at pointer 0x2E.
pub static ISO_8859_5: Encoding = Encoding {
    name: "ISO-8859-5",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_5, 0x040E, 0x2E, 66),
};

/// The `ISO-8859-6` encoding of the Encoding Standard: the INDEX_ISO_8859_6 table,
/// with the encoder run of 33 code points from U+0080 at pointer 0x00.
pub static ISO_8859_6: Encoding = Encoding {
    name: "ISO-8859-6",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_6, 0x0080, 0x00, 33),
};

/// The `ISO-8859-7` encoding of the Encoding Standard: the INDEX_ISO_8859_7 table,
/// with the encoder run of 44 code points from U+03A3 at pointer 0x53.
pub static ISO_8859_7: Encoding = Encoding {
    name: "ISO-8859-7",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_7, 0x03A3, 0x53, 44),
};

/// The `ISO-8859-8` encoding of the Encoding Standard: the INDEX_ISO_8859_8 table,
/// with the encoder run of 33 code points from U+0080 at pointer 0x00.
pub static ISO_8859_8: Encoding = Encoding {
    name: "ISO-8859-8",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_8, 0x0080, 0x00, 33),
};

/// The `ISO-8859-8-I` encoding. The standard gives it the very same index as
/// ISO-8859-8, so it reuses INDEX_ISO_8859_8; only the bidirectional layout
/// direction differs, and that is a rendering concern this crate does not model.
pub static ISO_8859_8_I: Encoding = Encoding {
    name: "ISO-8859-8-I",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_8, 0x0080, 0x00, 33),
};

/// The `ISO-8859-10` encoding of the Encoding Standard: the INDEX_ISO_8859_10 table,
/// with the encoder run of 33 code points from U+0080 at pointer 0x00.
pub static ISO_8859_10: Encoding = Encoding {
    name: "ISO-8859-10",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_10, 0x0080, 0x00, 33),
};

/// The `ISO-8859-13` encoding of the Encoding Standard: the INDEX_ISO_8859_13 table,
/// with the encoder run of 33 code points from U+0080 at pointer 0x00.
pub static ISO_8859_13: Encoding = Encoding {
    name: "ISO-8859-13",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_13, 0x0080, 0x00, 33),
};

/// The `ISO-8859-14` encoding of the Encoding Standard: the INDEX_ISO_8859_14 table,
/// with the encoder run of 33 code points from U+0080 at pointer 0x00.
pub static ISO_8859_14: Encoding = Encoding {
    name: "ISO-8859-14",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_14, 0x0080, 0x00, 33),
};

/// The `ISO-8859-15` encoding of the Encoding Standard: the INDEX_ISO_8859_15 table,
/// with the encoder run of 65 code points from U+00BF at pointer 0x3F.
pub static ISO_8859_15: Encoding = Encoding {
    name: "ISO-8859-15",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_15, 0x00BF, 0x3F, 65),
};

/// The `ISO-8859-16` encoding of the Encoding Standard: the INDEX_ISO_8859_16 table,
/// with the encoder run of 33 code points from U+0080 at pointer 0x00.
pub static ISO_8859_16: Encoding = Encoding {
    name: "ISO-8859-16",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_ISO_8859_16, 0x0080, 0x00, 33),
};

/// The `KOI8-R` encoding of the Encoding Standard: the INDEX_KOI8_R table,
/// with the encoder run of 15 code points from U+2553 at pointer 0x24.
pub static KOI8_R: Encoding = Encoding {
    name: "KOI8-R",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_KOI8_R, 0x2553, 0x24, 15),
};

/// The `KOI8-U` encoding of the Encoding Standard: the INDEX_KOI8_U table,
/// with the encoder run of 8 code points from U+0418 at pointer 0x69.
pub static KOI8_U: Encoding = Encoding {
    name: "KOI8-U",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_KOI8_U, 0x0418, 0x69, 8),
};

/// The `macintosh` encoding of the Encoding Standard: the INDEX_MACINTOSH table,
/// with the encoder run of 3 code points from U+02D8 at pointer 0x79.
pub static MACINTOSH: Encoding = Encoding {
    name: "macintosh",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_MACINTOSH, 0x02D8, 0x79, 3),
};

/// The `windows-874` encoding of the Encoding Standard: the INDEX_WINDOWS_874 table,
/// with the encoder run of 58 code points from U+0E01 at pointer 0x21.
pub static WINDOWS_874: Encoding = Encoding {
    name: "windows-874",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_WINDOWS_874, 0x0E01, 0x21, 58),
};

/// The `windows-1250` encoding of the Encoding Standard: the INDEX_WINDOWS_1250 table,
/// with the encoder run of 5 code points from U+00B4 at pointer 0x34.
pub static WINDOWS_1250: Encoding = Encoding {
    name: "windows-1250",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_WINDOWS_1250, 0x00B4, 0x34, 5),
};

/// The `windows-1251` encoding of the Encoding Standard: the INDEX_WINDOWS_1251 table,
/// with the encoder run of 64 code points from U+0410 at pointer 0x40.
pub static WINDOWS_1251: Encoding = Encoding {
    name: "windows-1251",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_WINDOWS_1251, 0x0410, 0x40, 64),
};

/// The `windows-1252` encoding of the Encoding Standard: the INDEX_WINDOWS_1252 table,
/// with the encoder run of 96 code points from U+00A0 at pointer 0x20.
pub static WINDOWS_1252: Encoding = Encoding {
    name: "windows-1252",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_WINDOWS_1252, 0x00A0, 0x20, 96),
};

/// The `windows-1253` encoding of the Encoding Standard: the INDEX_WINDOWS_1253 table,
/// with the encoder run of 44 code points from U+03A3 at pointer 0x53.
pub static WINDOWS_1253: Encoding = Encoding {
    name: "windows-1253",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_WINDOWS_1253, 0x03A3, 0x53, 44),
};

/// The `windows-1254` encoding of the Encoding Standard: the INDEX_WINDOWS_1254 table,
/// with the encoder run of 48 code points from U+00A0 at pointer 0x20.
pub static WINDOWS_1254: Encoding = Encoding {
    name: "windows-1254",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_WINDOWS_1254, 0x00A0, 0x20, 48),
};

/// The `windows-1255` encoding of the Encoding Standard: the INDEX_WINDOWS_1255 table,
/// with the encoder run of 27 code points from U+05D0 at pointer 0x60.
pub static WINDOWS_1255: Encoding = Encoding {
    name: "windows-1255",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_WINDOWS_1255, 0x05D0, 0x60, 27),
};

/// The `windows-1256` encoding of the Encoding Standard: the INDEX_WINDOWS_1256 table,
/// with the encoder run of 22 code points from U+0621 at pointer 0x41.
pub static WINDOWS_1256: Encoding = Encoding {
    name: "windows-1256",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_WINDOWS_1256, 0x0621, 0x41, 22),
};

/// The `windows-1257` encoding of the Encoding Standard: the INDEX_WINDOWS_1257 table,
/// with the encoder run of 8 code points from U+00B0 at pointer 0x30.
pub static WINDOWS_1257: Encoding = Encoding {
    name: "windows-1257",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_WINDOWS_1257, 0x00B0, 0x30, 8),
};

/// The `windows-1258` encoding of the Encoding Standard: the INDEX_WINDOWS_1258 table,
/// with the encoder run of 35 code points from U+00A0 at pointer 0x20.
pub static WINDOWS_1258: Encoding = Encoding {
    name: "windows-1258",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_WINDOWS_1258, 0x00A0, 0x20, 35),
};

/// The `x-mac-cyrillic` encoding of the Encoding Standard: the INDEX_X_MAC_CYRILLIC table,
/// with the encoder run of 32 code points from U+0410 at pointer 0x00.
pub static X_MAC_CYRILLIC: Encoding = Encoding {
    name: "x-mac-cyrillic",
    variant: VariantEncoding::SingleByte(&single_byte_tables::INDEX_X_MAC_CYRILLIC, 0x0410, 0x00, 32),
};

/// Every encoding implemented by this crate, Unicode ones first, then the
/// single-byte ones in the order the standard lists them.
pub static ENCODINGS: [&Encoding; 31] = [
    &UTF_8,
    &UTF_16BE,
    &UTF_16LE,
    &IBM866,
    &ISO_8859_2,
    &ISO_8859_3,
    &ISO_8859_4,
    &ISO_8859_5,
    &ISO_8859_6,
    &ISO_8859_7,
    &ISO_8859_8,
    &ISO_8859_8_I,
    &ISO_8859_10,
    &ISO_8859_13,
    &ISO_8859_14,
    &ISO_8859_15,
    &ISO_8859_16,
    &KOI8_R,
    &KOI8_U,
    &MACINTOSH,
    &WINDOWS_874,
    &WINDOWS_1250,
    &WINDOWS_1251,
    &WINDOWS_1252,
    &WINDOWS_1253,
    &WINDOWS_1254,
    &WINDOWS_1255,
    &WINDOWS_1256,
    &WINDOWS_1257,
    &WINDOWS_1258,
    &X_MAC_CYRILLIC,
];

/// Every single-byte encoding implemented by this crate, in the order the
/// standard lists them.
pub static SINGLE_BYTE_ENCODINGS: [&Encoding; 28] = [
    &IBM866,
    &ISO_8859_2,
    &ISO_8859_3,
    &ISO_8859_4,
    &ISO_8859_5,
    &ISO_8859_6,
    &ISO_8859_7,
    &ISO_8859_8,
    &ISO_8859_8_I,
    &ISO_8859_10,
    &ISO_8859_13,
    &ISO_8859_14,
    &ISO_8859_15,
    &ISO_8859_16,
    &KOI8_R,
    &KOI8_U,
    &MACINTOSH,
    &WINDOWS_874,
    &WINDOWS_1250,
    &WINDOWS_1251,
    &WINDOWS_1252,
    &WINDOWS_1253,
    &WINDOWS_1254,
    &WINDOWS_1255,
    &WINDOWS_1256,
    &WINDOWS_1257,
    &WINDOWS_1258,
    &X_MAC_CYRILLIC,
];
