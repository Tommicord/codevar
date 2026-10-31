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

/// Every label of an encoding implemented by this crate, in the order the
/// standard lists them and already ASCII-lowercased as its labels are.
static LABELS: [(&str, &Encoding); 183] = [
    ("unicode-1-1-utf-8", &UTF_8),
    ("unicode11utf8", &UTF_8),
    ("unicode20utf8", &UTF_8),
    ("utf-8", &UTF_8),
    ("utf8", &UTF_8),
    ("x-unicode20utf8", &UTF_8),
    ("866", &IBM866),
    ("cp866", &IBM866),
    ("csibm866", &IBM866),
    ("ibm866", &IBM866),
    ("csisolatin2", &ISO_8859_2),
    ("iso-8859-2", &ISO_8859_2),
    ("iso-ir-101", &ISO_8859_2),
    ("iso8859-2", &ISO_8859_2),
    ("iso88592", &ISO_8859_2),
    ("iso_8859-2", &ISO_8859_2),
    ("iso_8859-2:1987", &ISO_8859_2),
    ("l2", &ISO_8859_2),
    ("latin2", &ISO_8859_2),
    ("csisolatin3", &ISO_8859_3),
    ("iso-8859-3", &ISO_8859_3),
    ("iso-ir-109", &ISO_8859_3),
    ("iso8859-3", &ISO_8859_3),
    ("iso88593", &ISO_8859_3),
    ("iso_8859-3", &ISO_8859_3),
    ("iso_8859-3:1988", &ISO_8859_3),
    ("l3", &ISO_8859_3),
    ("latin3", &ISO_8859_3),
    ("csisolatin4", &ISO_8859_4),
    ("iso-8859-4", &ISO_8859_4),
    ("iso-ir-110", &ISO_8859_4),
    ("iso8859-4", &ISO_8859_4),
    ("iso88594", &ISO_8859_4),
    ("iso_8859-4", &ISO_8859_4),
    ("iso_8859-4:1988", &ISO_8859_4),
    ("l4", &ISO_8859_4),
    ("latin4", &ISO_8859_4),
    ("csisolatincyrillic", &ISO_8859_5),
    ("cyrillic", &ISO_8859_5),
    ("iso-8859-5", &ISO_8859_5),
    ("iso-ir-144", &ISO_8859_5),
    ("iso8859-5", &ISO_8859_5),
    ("iso88595", &ISO_8859_5),
    ("iso_8859-5", &ISO_8859_5),
    ("iso_8859-5:1988", &ISO_8859_5),
    ("arabic", &ISO_8859_6),
    ("asmo-708", &ISO_8859_6),
    ("csiso88596e", &ISO_8859_6),
    ("csiso88596i", &ISO_8859_6),
    ("csisolatinarabic", &ISO_8859_6),
    ("ecma-114", &ISO_8859_6),
    ("iso-8859-6", &ISO_8859_6),
    ("iso-8859-6-e", &ISO_8859_6),
    ("iso-8859-6-i", &ISO_8859_6),
    ("iso-ir-127", &ISO_8859_6),
    ("iso8859-6", &ISO_8859_6),
    ("iso88596", &ISO_8859_6),
    ("iso_8859-6", &ISO_8859_6),
    ("iso_8859-6:1987", &ISO_8859_6),
    ("csisolatingreek", &ISO_8859_7),
    ("ecma-118", &ISO_8859_7),
    ("elot_928", &ISO_8859_7),
    ("greek", &ISO_8859_7),
    ("greek8", &ISO_8859_7),
    ("iso-8859-7", &ISO_8859_7),
    ("iso-ir-126", &ISO_8859_7),
    ("iso8859-7", &ISO_8859_7),
    ("iso88597", &ISO_8859_7),
    ("iso_8859-7", &ISO_8859_7),
    ("iso_8859-7:1987", &ISO_8859_7),
    ("sun_eu_greek", &ISO_8859_7),
    ("csiso88598e", &ISO_8859_8),
    ("csisolatinhebrew", &ISO_8859_8),
    ("hebrew", &ISO_8859_8),
    ("iso-8859-8", &ISO_8859_8),
    ("iso-8859-8-e", &ISO_8859_8),
    ("iso-ir-138", &ISO_8859_8),
    ("iso8859-8", &ISO_8859_8),
    ("iso88598", &ISO_8859_8),
    ("iso_8859-8", &ISO_8859_8),
    ("iso_8859-8:1988", &ISO_8859_8),
    ("visual", &ISO_8859_8),
    ("csiso88598i", &ISO_8859_8_I),
    ("iso-8859-8-i", &ISO_8859_8_I),
    ("logical", &ISO_8859_8_I),
    ("csisolatin6", &ISO_8859_10),
    ("iso-8859-10", &ISO_8859_10),
    ("iso-ir-157", &ISO_8859_10),
    ("iso8859-10", &ISO_8859_10),
    ("iso885910", &ISO_8859_10),
    ("l6", &ISO_8859_10),
    ("latin6", &ISO_8859_10),
    ("iso-8859-13", &ISO_8859_13),
    ("iso8859-13", &ISO_8859_13),
    ("iso885913", &ISO_8859_13),
    ("iso-8859-14", &ISO_8859_14),
    ("iso8859-14", &ISO_8859_14),
    ("iso885914", &ISO_8859_14),
    ("csisolatin9", &ISO_8859_15),
    ("iso-8859-15", &ISO_8859_15),
    ("iso8859-15", &ISO_8859_15),
    ("iso885915", &ISO_8859_15),
    ("iso_8859-15", &ISO_8859_15),
    ("l9", &ISO_8859_15),
    ("iso-8859-16", &ISO_8859_16),
    ("cskoi8r", &KOI8_R),
    ("koi", &KOI8_R),
    ("koi8", &KOI8_R),
    ("koi8-r", &KOI8_R),
    ("koi8_r", &KOI8_R),
    ("koi8-ru", &KOI8_U),
    ("koi8-u", &KOI8_U),
    ("csmacintosh", &MACINTOSH),
    ("mac", &MACINTOSH),
    ("macintosh", &MACINTOSH),
    ("x-mac-roman", &MACINTOSH),
    ("dos-874", &WINDOWS_874),
    ("iso-8859-11", &WINDOWS_874),
    ("iso8859-11", &WINDOWS_874),
    ("iso885911", &WINDOWS_874),
    ("tis-620", &WINDOWS_874),
    ("windows-874", &WINDOWS_874),
    ("cp1250", &WINDOWS_1250),
    ("windows-1250", &WINDOWS_1250),
    ("x-cp1250", &WINDOWS_1250),
    ("cp1251", &WINDOWS_1251),
    ("windows-1251", &WINDOWS_1251),
    ("x-cp1251", &WINDOWS_1251),
    ("ansi_x3.4-1968", &WINDOWS_1252),
    ("ascii", &WINDOWS_1252),
    ("cp1252", &WINDOWS_1252),
    ("cp819", &WINDOWS_1252),
    ("csisolatin1", &WINDOWS_1252),
    ("ibm819", &WINDOWS_1252),
    ("iso-8859-1", &WINDOWS_1252),
    ("iso-ir-100", &WINDOWS_1252),
    ("iso8859-1", &WINDOWS_1252),
    ("iso88591", &WINDOWS_1252),
    ("iso_8859-1", &WINDOWS_1252),
    ("iso_8859-1:1987", &WINDOWS_1252),
    ("l1", &WINDOWS_1252),
    ("latin1", &WINDOWS_1252),
    ("us-ascii", &WINDOWS_1252),
    ("windows-1252", &WINDOWS_1252),
    ("x-cp1252", &WINDOWS_1252),
    ("cp1253", &WINDOWS_1253),
    ("windows-1253", &WINDOWS_1253),
    ("x-cp1253", &WINDOWS_1253),
    ("cp1254", &WINDOWS_1254),
    ("csisolatin5", &WINDOWS_1254),
    ("iso-8859-9", &WINDOWS_1254),
    ("iso-ir-148", &WINDOWS_1254),
    ("iso8859-9", &WINDOWS_1254),
    ("iso88599", &WINDOWS_1254),
    ("iso_8859-9", &WINDOWS_1254),
    ("iso_8859-9:1989", &WINDOWS_1254),
    ("l5", &WINDOWS_1254),
    ("latin5", &WINDOWS_1254),
    ("windows-1254", &WINDOWS_1254),
    ("x-cp1254", &WINDOWS_1254),
    ("cp1255", &WINDOWS_1255),
    ("windows-1255", &WINDOWS_1255),
    ("x-cp1255", &WINDOWS_1255),
    ("cp1256", &WINDOWS_1256),
    ("windows-1256", &WINDOWS_1256),
    ("x-cp1256", &WINDOWS_1256),
    ("cp1257", &WINDOWS_1257),
    ("windows-1257", &WINDOWS_1257),
    ("x-cp1257", &WINDOWS_1257),
    ("cp1258", &WINDOWS_1258),
    ("windows-1258", &WINDOWS_1258),
    ("x-cp1258", &WINDOWS_1258),
    ("x-mac-cyrillic", &X_MAC_CYRILLIC),
    ("x-mac-ukrainian", &X_MAC_CYRILLIC),
    ("unicodefffe", &UTF_16BE),
    ("utf-16be", &UTF_16BE),
    ("csunicode", &UTF_16LE),
    ("iso-10646-ucs-2", &UTF_16LE),
    ("ucs-2", &UTF_16LE),
    ("unicode", &UTF_16LE),
    ("unicodefeff", &UTF_16LE),
    ("utf-16", &UTF_16LE),
    ("utf-16le", &UTF_16LE),
];

/// Labels the standard assigns to encodings this crate does not implement
/// yet: the legacy multi-byte encodings (GBK, gb18030, Big5, EUC-JP,
/// ISO-2022-JP, Shift_JIS, EUC-KR), the replacement encoding and
/// x-user-defined. `lookup` reports them separately from labels the
/// standard does not know at all, so a caller can tell the two apart
/// instead of silently substituting another encoding.
const UNIMPLEMENTED_LABELS: [&str; 45] = [
    "chinese",
    "csgb2312",
    "csiso58gb231280",
    "gb2312",
    "gb_2312",
    "gb_2312-80",
    "gbk",
    "iso-ir-58",
    "x-gbk",
    "gb18030",
    "big5",
    "big5-hkscs",
    "cn-big5",
    "csbig5",
    "x-x-big5",
    "cseucpkdfmtjapanese",
    "euc-jp",
    "x-euc-jp",
    "csiso2022jp",
    "iso-2022-jp",
    "csshiftjis",
    "ms932",
    "ms_kanji",
    "shift-jis",
    "shift_jis",
    "sjis",
    "windows-31j",
    "x-sjis",
    "cseuckr",
    "csksc56011987",
    "euc-kr",
    "iso-ir-149",
    "korean",
    "ks_c_5601-1987",
    "ks_c_5601-1989",
    "ksc5601",
    "ksc_5601",
    "windows-949",
    "csiso2022kr",
    "hz-gb-2312",
    "iso-2022-cn",
    "iso-2022-cn-ext",
    "iso-2022-kr",
    "replacement",
    "x-user-defined",
];

/// The outcome of matching a label against the name table of the standard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LabelLookup {
    /// The label names an encoding implemented by this crate.
    Supported(&'static Encoding),
    /// The label names an encoding the standard defines but this crate does not
    /// implement yet: a legacy multi-byte encoding, the replacement encoding or
    /// x-user-defined.
    Unimplemented,
    /// No encoding in the standard carries this label.
    Unknown,
}

/// Resolve `label` as the standard's "get an encoding" algorithm does: drop the
/// leading and trailing ASCII whitespace, then match what is left
/// ASCII-case-insensitively against the labels the standard assigns.
///
/// A label of an encoding this crate does not implement is reported as
/// [`LabelLookup::Unimplemented`] rather than `Unknown`, so that a caller can
/// fall back deliberately instead of mistaking `gb18030` for a typo.
pub fn lookup(label: &str) -> LabelLookup {
    let label = label.trim_matches(|c: char| c.is_ascii_whitespace());
    for (name, encoding) in &LABELS {
        if label.eq_ignore_ascii_case(name) {
            return LabelLookup::Supported(encoding);
        }
    }
    for name in &UNIMPLEMENTED_LABELS {
        if label.eq_ignore_ascii_case(name) {
            return LabelLookup::Unimplemented;
        }
    }
    LabelLookup::Unknown
}

/// The encoding the standard gives to `label`, or `None` when the label is
/// unknown here or names an encoding this crate does not implement.
pub fn for_label(label: &str) -> Option<&'static Encoding> {
    match lookup(label) {
        LabelLookup::Supported(encoding) => Some(encoding),
        LabelLookup::Unimplemented | LabelLookup::Unknown => None,
    }
}

/// The encoding identified by a leading byte-order mark, as the first step of
/// the standard's decode algorithm does: `EF BB BF` selects UTF-8, `FE FF`
/// UTF-16BE and `FF FE` UTF-16LE. `None` means the input starts with no mark.
///
/// The UTF-8 mark is tested first, because `EF BB BF` is also a well-formed
/// start of a UTF-16LE stream (`U+BBEF` followed by a stray byte) and would
/// otherwise be misread as UTF-16.
pub fn for_bom(bytes: &[u8]) -> Option<&'static Encoding> {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        Some(&UTF_8)
    } else if bytes.starts_with(&[0xFE, 0xFF]) {
        Some(&UTF_16BE)
    } else if bytes.starts_with(&[0xFF, 0xFE]) {
        Some(&UTF_16LE)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{CoderResult, Decoder, DecoderResult, EncoderResult, NCR_EXTRA};

    fn by_name(name: &str) -> &'static Encoding {
        ENCODINGS
            .iter()
            .copied()
            .find(|encoding| encoding.name == name)
            .unwrap_or_else(|| panic!("{name} is not in the registry"))
    }

    fn table_of(encoding: &'static Encoding) -> &'static [u16; 128] {
        let VariantEncoding::SingleByte(table, _, _, _) = encoding.variant else {
            panic!("{} is not a single-byte encoding", encoding.name);
        };
        table
    }

    #[test]
    fn every_defined_pointer_round_trips() {
        let mut out16 = [0u16; 1];
        let mut out8 = [0u8; 1];
        for &encoding in SINGLE_BYTE_ENCODINGS.iter() {
            let mut decoder = encoding.new_decoder_without_bom_handling();
            let mut encoder = encoding.new_encoder();
            for (pointer, &expected) in table_of(encoding).iter().enumerate() {
                let byte = 0x80u8 + pointer as u8;
                let (result, read, written) =
                    decoder.decode_to_utf16_without_replacement(&[byte], &mut out16, true);
                if expected == 0 {
                    assert_eq!(
                        result,
                        DecoderResult::Malformed(1, 0),
                        "{}: byte {byte:02X} is a defined pointer",
                        encoding.name
                    );
                    assert_eq!((read, written), (1, 0), "{}: byte {byte:02X}", encoding.name);
                    continue;
                }
                assert_eq!(
                    result,
                    DecoderResult::InputEmpty,
                    "{}: byte {byte:02X}",
                    encoding.name
                );
                assert_eq!((read, written), (1, 1), "{}: byte {byte:02X}", encoding.name);
                assert_eq!(out16[0], expected, "{}: byte {byte:02X}", encoding.name);
                let (result, read, written) =
                    encoder.encode_from_utf16_without_replacement(&[expected], &mut out8, true);
                assert_eq!(
                    result,
                    EncoderResult::InputEmpty,
                    "{}: U+{expected:04X} should encode",
                    encoding.name
                );
                assert_eq!((read, written), (1, 1), "{}: U+{expected:04X}", encoding.name);
                assert_eq!(out8[0], byte, "{}: U+{expected:04X}", encoding.name);
            }
        }
    }

    #[test]
    fn undefined_pointers_are_exactly_the_documented_hole_sets() {
        // The blank pointer ranges the standard's index files leave undefined, as byte
        // ranges. Every other registry index is total, and the encoder's reverse lookup
        // must therefore never emit these bytes.
        let expected: [(&str, &[(u8, u8)]); 9] = [
            (
                "ISO-8859-3",
                &[
                    (0xA5, 0xA5),
                    (0xAE, 0xAE),
                    (0xBE, 0xBE),
                    (0xC3, 0xC3),
                    (0xD0, 0xD0),
                    (0xE3, 0xE3),
                    (0xF0, 0xF0),
                ],
            ),
            (
                "ISO-8859-6",
                &[
                    (0xA1, 0xA3),
                    (0xA5, 0xAB),
                    (0xAE, 0xBA),
                    (0xBC, 0xBE),
                    (0xC0, 0xC0),
                    (0xDB, 0xDF),
                    (0xF3, 0xFF),
                ],
            ),
            ("ISO-8859-7", &[(0xAE, 0xAE), (0xD2, 0xD2), (0xFF, 0xFF)]),
            (
                "ISO-8859-8",
                &[(0xA1, 0xA1), (0xBF, 0xDE), (0xFB, 0xFC), (0xFF, 0xFF)],
            ),
            (
                "ISO-8859-8-I",
                &[(0xA1, 0xA1), (0xBF, 0xDE), (0xFB, 0xFC), (0xFF, 0xFF)],
            ),
            ("windows-1253", &[(0xAA, 0xAA), (0xD2, 0xD2), (0xFF, 0xFF)]),
            ("windows-1255", &[(0xD9, 0xDF), (0xFB, 0xFC), (0xFF, 0xFF)]),
            ("windows-1257", &[(0xA1, 0xA1), (0xA5, 0xA5)]),
            ("windows-874", &[(0xDB, 0xDE), (0xFC, 0xFF)]),
        ];
        let mut holes = 0usize;
        for (name, ranges) in expected {
            let table = table_of(by_name(name));
            let want: Vec<u8> = ranges
                .iter()
                .flat_map(|&(lo, hi)| lo..=hi)
                .collect();
            let got: Vec<u8> = table
                .iter()
                .enumerate()
                .filter(|(_, unit)| **unit == 0)
                .map(|(pointer, _)| 0x80u8 + pointer as u8)
                .collect();
            assert_eq!(got, want, "{name}: undefined pointers");
            holes += got.len();
        }
        assert_eq!(holes, 150, "the hole sets above do not add up");
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            if expected
                .iter()
                .any(|(name, _)| *name == encoding.name)
            {
                continue;
            }
            assert!(
                table_of(encoding).iter().all(|unit| *unit != 0),
                "{}: unexpectedly holds an undefined pointer",
                encoding.name
            );
        }
    }

    #[test]
    fn registry_holds_the_single_byte_encodings_of_the_standard() {
        assert_eq!(SINGLE_BYTE_ENCODINGS.len(), 28);
        let mut names = SINGLE_BYTE_ENCODINGS.iter().map(|e| e.name);
        for name in [
            "IBM866",
            "ISO-8859-2",
            "ISO-8859-3",
            "ISO-8859-4",
            "ISO-8859-5",
            "ISO-8859-6",
            "ISO-8859-7",
            "ISO-8859-8",
            "ISO-8859-8-I",
            "ISO-8859-10",
            "ISO-8859-13",
            "ISO-8859-14",
            "ISO-8859-15",
            "ISO-8859-16",
            "KOI8-R",
            "KOI8-U",
            "macintosh",
            "windows-874",
            "windows-1250",
            "windows-1251",
            "windows-1252",
            "windows-1253",
            "windows-1254",
            "windows-1255",
            "windows-1256",
            "windows-1257",
            "windows-1258",
            "x-mac-cyrillic",
        ] {
            assert!(names.by_ref().any(|n| n == name), "{name} is missing");
        }
        assert_eq!(names.count(), 0, "unexpected extra entry");
    }

    #[test]
    fn unicode_encodings_come_first_in_the_full_list() {
        assert_eq!(ENCODINGS.len(), 31);
        let ordered: [&str; 31] = [
            "UTF-8",
            "UTF-16BE",
            "UTF-16LE",
            "IBM866",
            "ISO-8859-2",
            "ISO-8859-3",
            "ISO-8859-4",
            "ISO-8859-5",
            "ISO-8859-6",
            "ISO-8859-7",
            "ISO-8859-8",
            "ISO-8859-8-I",
            "ISO-8859-10",
            "ISO-8859-13",
            "ISO-8859-14",
            "ISO-8859-15",
            "ISO-8859-16",
            "KOI8-R",
            "KOI8-U",
            "macintosh",
            "windows-874",
            "windows-1250",
            "windows-1251",
            "windows-1252",
            "windows-1253",
            "windows-1254",
            "windows-1255",
            "windows-1256",
            "windows-1257",
            "windows-1258",
            "x-mac-cyrillic",
        ];
        for (position, name) in ordered.iter().enumerate() {
            assert_eq!(ENCODINGS[position].name, *name, "registry position {position}");
        }
        assert_eq!(&ENCODINGS[3..], &SINGLE_BYTE_ENCODINGS[..]);
    }

    #[test]
    fn every_table_label_resolves_to_its_encoding() {
        for (label, encoding) in &LABELS {
            let found = for_label(label).unwrap_or_else(|| panic!("{label} does not resolve"));
            assert_eq!(found.name, encoding.name, "{label}");
        }
    }

    #[test]
    fn lookup_resolves_the_unicode_labels() {
        for (label, name) in [
            ("utf-8", "UTF-8"),
            ("UTF8", "UTF-8"),
            ("unicode-1-1-utf-8", "UTF-8"),
            ("utf-16be", "UTF-16BE"),
            ("unicodefffe", "UTF-16BE"),
            ("utf-16le", "UTF-16LE"),
            ("utf-16", "UTF-16LE"),
            ("unicode", "UTF-16LE"),
            ("csunicode", "UTF-16LE"),
        ] {
            assert_eq!(for_label(label).map(|e| e.name), Some(name), "{label}");
        }
    }

    #[test]
    fn lookup_is_ascii_case_insensitive() {
        assert_eq!(for_label("WINDOWS-1252").map(|e| e.name), Some("windows-1252"));
        assert_eq!(for_label("Windows-1252").map(|e| e.name), Some("windows-1252"));
        assert_eq!(for_label("wInDoWs-1252").map(|e| e.name), Some("windows-1252"));
        assert_eq!(for_label("KOI8-R").map(|e| e.name), Some("KOI8-R"));
        assert_eq!(for_label("koi8_r").map(|e| e.name), Some("KOI8-R"));
    }

    #[test]
    fn lookup_trims_exactly_the_five_whitespace_characters_of_the_standard() {
        // "Get an encoding" strips leading and trailing ASCII whitespace, which the
        // standard defines as U+0009, U+000A, U+000C, U+000D and U+0020--U+000B is
        // deliberately not among them.
        for filler in ["\t", "\n", "\u{000C}", "\r", " "] {
            let padded = format!("{filler}{filler}windows-1252{filler}");
            assert_eq!(
                for_label(&padded).map(|e| e.name),
                Some("windows-1252"),
                "{padded:?}"
            );
        }
        for not_whitespace in ["\u{000B}", "\u{0085}", "\u{00A0}", "\u{2028}", "\u{200B}"] {
            assert_eq!(
                lookup(&format!("{not_whitespace}utf-8")),
                LabelLookup::Unknown,
                "{not_whitespace:?} leading"
            );
            assert_eq!(
                lookup(&format!("utf-8{not_whitespace}")),
                LabelLookup::Unknown,
                "{not_whitespace:?} trailing"
            );
        }
    }

    #[test]
    fn interior_whitespace_and_untrimmed_padding_are_not_stripped() {
        for label in [
            "utf -8",
            "u tf-8",
            "utf- 8",
            "utf 8",
            "cp125 2",
            "koi8 -r",
            "iso-8859 1",
            "windows-1252 x",
            "utf-8\0",
            "\0utf-8",
        ] {
            assert_eq!(lookup(label), LabelLookup::Unknown, "{label:?}");
            assert!(for_label(label).is_none(), "{label:?}");
        }
    }

    #[test]
    fn case_folding_stays_within_ascii() {
        assert_eq!(lookup("WINDOWS-1252"), LabelLookup::Supported(&WINDOWS_1252));
        // U+0130 and U+212A fold to "i" and "k" only under Unicode case folding; the
        // standard matches labels ASCII-case-insensitively, so neither may resolve.
        assert_eq!(lookup("\u{0130}bm866"), LabelLookup::Unknown);
        assert_eq!(lookup("\u{212A}oi8-r"), LabelLookup::Unknown);
        // Non-ASCII letters are never folded, and neither are the digits.
        assert_eq!(lookup("windows-125２"), LabelLookup::Unknown);
        for label in ["iso-8859-2", "iso8859-2", "koi8_r", "x-mac-roman"] {
            let upper = label.to_ascii_uppercase();
            assert_eq!(lookup(&upper), lookup(label), "{label} and {upper} must agree");
            assert!(matches!(lookup(&upper), LabelLookup::Supported(_)));
        }
    }

    #[test]
    fn every_label_resolves_the_same_uppercased_and_padded() {
        for (label, encoding) in &LABELS {
            let upper = label.to_ascii_uppercase();
            let found = for_label(&upper).unwrap_or_else(|| panic!("{upper} does not resolve"));
            assert!(core::ptr::eq(found, *encoding), "{upper}");
            let padded = format!(" \t{label}\r\n");
            let found = for_label(&padded).unwrap_or_else(|| panic!("{padded:?}"));
            assert!(core::ptr::eq(found, *encoding), "{padded:?}");
        }
        for label in UNIMPLEMENTED_LABELS {
            assert_eq!(
                lookup(&format!(" {label}\u{0009}")),
                LabelLookup::Unimplemented,
                "{label}"
            );
        }
    }

    #[test]
    fn latin1_and_ascii_labels_alias_windows_1252() {
        for label in [
            "iso-8859-1",
            "latin1",
            "l1",
            "csisolatin1",
            "ansi_x3.4-1968",
            "ascii",
            "us-ascii",
            "ibm819",
            "cp819",
            "iso_8859-1:1987",
        ] {
            assert_eq!(for_label(label).map(|e| e.name), Some("windows-1252"), "{label}");
        }
    }

    #[test]
    fn iso_8859_11_and_tis_620_alias_windows_874() {
        for label in ["iso-8859-11", "iso8859-11", "iso885911", "tis-620", "dos-874"] {
            assert_eq!(for_label(label).map(|e| e.name), Some("windows-874"), "{label}");
        }
    }

    #[test]
    fn iso_8859_9_aliases_windows_1254() {
        for label in ["iso-8859-9", "iso-ir-148", "l5", "latin5", "csisolatin5"] {
            assert_eq!(for_label(label).map(|e| e.name), Some("windows-1254"), "{label}");
        }
    }

    #[test]
    fn hebrew_logical_and_visual_share_an_index() {
        let visual = for_label("iso-8859-8").expect("visual");
        let logical = for_label("iso-8859-8-i").expect("logical");
        assert_eq!(visual.name, "ISO-8859-8");
        assert_eq!(logical.name, "ISO-8859-8-I");
        assert_eq!(table_of(visual), table_of(logical));
        for label in ["visual", "hebrew", "csiso88598e"] {
            assert_eq!(for_label(label).map(|e| e.name), Some("ISO-8859-8"), "{label}");
        }
        for label in ["logical", "csiso88598i"] {
            assert_eq!(for_label(label).map(|e| e.name), Some("ISO-8859-8-I"), "{label}");
        }
    }

    #[test]
    fn mac_roman_and_mac_cyrillic_labels() {
        assert_eq!(for_label("x-mac-roman").map(|e| e.name), Some("macintosh"));
        assert_eq!(for_label("csmacintosh").map(|e| e.name), Some("macintosh"));
        assert_eq!(for_label("mac").map(|e| e.name), Some("macintosh"));
        assert_eq!(
            for_label("x-mac-ukrainian").map(|e| e.name),
            Some("x-mac-cyrillic")
        );
    }

    #[test]
    fn every_unimplemented_label_is_reported_as_such() {
        for label in UNIMPLEMENTED_LABELS {
            assert_eq!(lookup(label), LabelLookup::Unimplemented, "{label}");
            assert_eq!(
                lookup(&label.to_uppercase()),
                LabelLookup::Unimplemented,
                "{label}"
            );
            assert!(for_label(label).is_none(), "{label} must not resolve");
        }
        assert_eq!(UNIMPLEMENTED_LABELS.len(), 45);
    }

    #[test]
    fn multi_byte_labels_do_not_fall_back_to_single_byte() {
        for label in [
            "gbk",
            "gb18030",
            "big5",
            "euc-jp",
            "iso-2022-jp",
            "shift_jis",
            "euc-kr",
        ] {
            assert_eq!(lookup(label), LabelLookup::Unimplemented, "{label}");
        }
    }

    #[test]
    fn unknown_labels_are_unknown() {
        for label in [
            "",
            " ",
            "utf-7",
            "iso-8859-12",
            "windows-1259",
            "utf_8",
            "x-unknown",
        ] {
            assert_eq!(lookup(label), LabelLookup::Unknown, "{label:?}");
        }
    }

    #[test]
    fn a_label_of_a_defined_encoding_is_never_unknown() {
        assert_eq!(lookup("  IBM866\t"), LabelLookup::Supported(by_name("IBM866")));
        assert_eq!(lookup("koi8-ru"), LabelLookup::Supported(by_name("KOI8-U")));
        assert_eq!(lookup("866"), LabelLookup::Supported(by_name("IBM866")));
        assert_eq!(lookup("cp1258"), LabelLookup::Supported(by_name("windows-1258")));
    }

    #[test]
    fn bom_sniffing_finds_the_three_unicode_marks() {
        assert_eq!(for_bom(&[0xEF, 0xBB, 0xBF]).map(|e| e.name), Some("UTF-8"));
        assert_eq!(for_bom(&[0xFE, 0xFF]).map(|e| e.name), Some("UTF-16BE"));
        assert_eq!(for_bom(&[0xFF, 0xFE]).map(|e| e.name), Some("UTF-16LE"));
        assert_eq!(
            for_bom(&[0xEF, 0xBB, 0xBF, 0x68, 0x69]).map(|e| e.name),
            Some("UTF-8")
        );
        assert_eq!(for_bom(&[0xFF, 0xFE, 0x00]).map(|e| e.name), Some("UTF-16LE"));
    }

    #[test]
    fn bom_sniffing_rejects_lookalikes() {
        for bytes in [
            &[][..],
            &[0xEF][..],
            &[0xEF, 0xBB][..],
            &[0xBF, 0xBB, 0xEF][..],
            &[0xFE][..],
            &[0xFF][..],
            &[0xFE, 0xFE][..],
            &[0xFF, 0xFF][..],
            &[0x00, 0xFE, 0xFF][..],
            &[0x48, 0x69][..],
        ] {
            assert_eq!(for_bom(bytes), None, "{bytes:?}");
        }
    }

    #[test]
    fn spot_decodes_match_the_standard() {
        let cases: [(&str, u8, u16); 16] = [
            ("windows-1252", 0x80, 0x20AC),
            ("windows-1252", 0x91, 0x2018),
            ("windows-1252", 0x99, 0x2122),
            ("windows-1252", 0xD0, 0x00D0),
            ("ISO-8859-15", 0xA4, 0x20AC),
            ("KOI8-R", 0xF2, 0x0420),
            ("KOI8-R", 0xD2, 0x0440),
            ("macintosh", 0xA5, 0x2022),
            ("macintosh", 0x8A, 0x00E4),
            ("ISO-8859-7", 0xE1, 0x03B1),
            ("windows-1251", 0xCF, 0x041F),
            ("ISO-8859-5", 0xD0, 0x0430),
            ("ISO-8859-8", 0xE0, 0x05D0),
            ("IBM866", 0x80, 0x0410),
            ("windows-874", 0xA1, 0x0E01),
            ("ISO-8859-16", 0xBF, 0x017C),
        ];
        let mut out16 = [0u16; 1];
        for (name, byte, expected) in cases {
            let mut decoder = by_name(name).new_decoder_without_bom_handling();
            let (result, read, written) =
                decoder.decode_to_utf16_without_replacement(&[byte], &mut out16, true);
            assert_eq!(result, DecoderResult::InputEmpty, "{name} byte {byte:02X}");
            assert_eq!((read, written), (1, 1), "{name} byte {byte:02X}");
            assert_eq!(out16[0], expected, "{name} byte {byte:02X}");
        }
    }

    #[test]
    fn undefined_pointers_decode_spot_checked() {
        let mut out16 = [0u16; 1];
        for (name, byte) in [
            ("ISO-8859-3", 0xA5u8),
            ("ISO-8859-3", 0xF0),
            ("ISO-8859-6", 0xA1),
            ("ISO-8859-6", 0xFF),
            ("ISO-8859-7", 0xAE),
            ("ISO-8859-7", 0xFF),
            ("ISO-8859-8", 0xA1),
            ("ISO-8859-8", 0xFB),
            ("windows-874", 0xDB),
            ("windows-874", 0xFF),
            ("windows-1253", 0xAA),
            ("windows-1253", 0xD2),
            ("windows-1255", 0xD9),
            ("windows-1255", 0xFF),
            ("windows-1257", 0xA1),
            ("windows-1257", 0xA5),
        ] {
            let mut decoder = by_name(name).new_decoder_without_bom_handling();
            let (result, read, written) =
                decoder.decode_to_utf16_without_replacement(&[byte], &mut out16, true);
            assert_eq!(result, DecoderResult::Malformed(1, 0), "{name} byte {byte:02X}");
            assert_eq!((read, written), (1, 0), "{name} byte {byte:02X}");
        }
    }

    #[test]
    fn iso_8859_indexes_map_the_c1_controls_to_themselves() {
        // Unlike cp1252 and the other legacy code pages, the ISO-8859-n indexes
        // of the standard begin by carrying U+0080..U+009F through unchanged.
        let mut out16 = [0u16; 1];
        for name in [
            "ISO-8859-2",
            "ISO-8859-3",
            "ISO-8859-4",
            "ISO-8859-5",
            "ISO-8859-6",
            "ISO-8859-7",
            "ISO-8859-8",
            "ISO-8859-8-I",
            "ISO-8859-10",
            "ISO-8859-13",
            "ISO-8859-14",
            "ISO-8859-15",
            "ISO-8859-16",
        ] {
            let mut decoder = by_name(name).new_decoder_without_bom_handling();
            for control in 0x80u8..=0x9F {
                let (result, read, written) =
                    decoder.decode_to_utf16_without_replacement(&[control], &mut out16, true);
                assert_eq!(result, DecoderResult::InputEmpty, "{name} byte {control:02X}");
                assert_eq!((read, written), (1, 1), "{name} byte {control:02X}");
                assert_eq!(out16[0], u16::from(control), "{name} byte {control:02X}");
            }
        }
    }

    fn run_of(encoding: &'static Encoding) -> (u16, u8, u8) {
        let VariantEncoding::SingleByte(_, bmp_offset, byte_offset, length) = encoding.variant else {
            panic!("{} is not a single-byte encoding", encoding.name);
        };
        (bmp_offset, byte_offset, length)
    }

    fn decode_all(decoder: &mut Decoder, bytes: &[u8]) -> Vec<u16> {
        // A single-byte decoding emits at most one code unit per input byte, the
        // replacement characters of undefined pointers included.
        let mut dst = vec![0u16; bytes.len() + 4];
        let (result, read, written, _had_errors) = decoder.decode_to_utf16(bytes, &mut dst, true);
        assert_eq!(result, CoderResult::InputEmpty, "{} bytes in", bytes.len());
        assert_eq!(read, bytes.len());
        dst.truncate(written);
        dst
    }

    fn units(encoding: &'static Encoding, bytes: &[u8]) -> Vec<u16> {
        decode_all(&mut encoding.new_decoder_without_bom_handling(), bytes)
    }

    fn encode_unit(encoding: &'static Encoding, unit: u16) -> Result<u8, EncoderResult> {
        let mut dst = [0u8; 4];
        let mut encoder = encoding.new_encoder();
        let (result, read, written) = encoder.encode_from_utf16_without_replacement(&[unit], &mut dst, true);
        match result {
            EncoderResult::InputEmpty => {
                assert_eq!((read, written), (1, 1), "{}: U+{unit:04X}", encoding.name);
                Ok(dst[0])
            }
            other => {
                assert_eq!((read, written), (1, 0), "{}: U+{unit:04X}", encoding.name);
                Err(other)
            }
        }
    }

    fn family_of(name: &str) -> Vec<&'static str> {
        let mut labels: Vec<&'static str> = LABELS
            .iter()
            .filter(|(_, encoding)| encoding.name == name)
            .map(|(label, _)| *label)
            .collect();
        labels.sort_unstable();
        labels
    }

    #[test]
    fn every_label_family_matches_the_standard_exactly() {
        let expected: [(&str, &[&str]); 31] = [
            (
                "UTF-8",
                &[
                    "unicode-1-1-utf-8",
                    "unicode11utf8",
                    "unicode20utf8",
                    "utf-8",
                    "utf8",
                    "x-unicode20utf8",
                ],
            ),
            ("UTF-16BE", &["unicodefffe", "utf-16be"]),
            (
                "UTF-16LE",
                &[
                    "csunicode",
                    "iso-10646-ucs-2",
                    "ucs-2",
                    "unicode",
                    "unicodefeff",
                    "utf-16",
                    "utf-16le",
                ],
            ),
            ("IBM866", &["866", "cp866", "csibm866", "ibm866"]),
            (
                "ISO-8859-2",
                &[
                    "csisolatin2",
                    "iso-8859-2",
                    "iso-ir-101",
                    "iso8859-2",
                    "iso88592",
                    "iso_8859-2",
                    "iso_8859-2:1987",
                    "l2",
                    "latin2",
                ],
            ),
            (
                "ISO-8859-3",
                &[
                    "csisolatin3",
                    "iso-8859-3",
                    "iso-ir-109",
                    "iso8859-3",
                    "iso88593",
                    "iso_8859-3",
                    "iso_8859-3:1988",
                    "l3",
                    "latin3",
                ],
            ),
            (
                "ISO-8859-4",
                &[
                    "csisolatin4",
                    "iso-8859-4",
                    "iso-ir-110",
                    "iso8859-4",
                    "iso88594",
                    "iso_8859-4",
                    "iso_8859-4:1988",
                    "l4",
                    "latin4",
                ],
            ),
            (
                "ISO-8859-5",
                &[
                    "csisolatincyrillic",
                    "cyrillic",
                    "iso-8859-5",
                    "iso-ir-144",
                    "iso8859-5",
                    "iso88595",
                    "iso_8859-5",
                    "iso_8859-5:1988",
                ],
            ),
            (
                "ISO-8859-6",
                &[
                    "arabic",
                    "asmo-708",
                    "csiso88596e",
                    "csiso88596i",
                    "csisolatinarabic",
                    "ecma-114",
                    "iso-8859-6",
                    "iso-8859-6-e",
                    "iso-8859-6-i",
                    "iso-ir-127",
                    "iso8859-6",
                    "iso88596",
                    "iso_8859-6",
                    "iso_8859-6:1987",
                ],
            ),
            (
                "ISO-8859-7",
                &[
                    "csisolatingreek",
                    "ecma-118",
                    "elot_928",
                    "greek",
                    "greek8",
                    "iso-8859-7",
                    "iso-ir-126",
                    "iso8859-7",
                    "iso88597",
                    "iso_8859-7",
                    "iso_8859-7:1987",
                    "sun_eu_greek",
                ],
            ),
            (
                "ISO-8859-8",
                &[
                    "csiso88598e",
                    "csisolatinhebrew",
                    "hebrew",
                    "iso-8859-8",
                    "iso-8859-8-e",
                    "iso-ir-138",
                    "iso8859-8",
                    "iso88598",
                    "iso_8859-8",
                    "iso_8859-8:1988",
                    "visual",
                ],
            ),
            ("ISO-8859-8-I", &["csiso88598i", "iso-8859-8-i", "logical"]),
            (
                "ISO-8859-10",
                &[
                    "csisolatin6",
                    "iso-8859-10",
                    "iso-ir-157",
                    "iso8859-10",
                    "iso885910",
                    "l6",
                    "latin6",
                ],
            ),
            ("ISO-8859-13", &["iso-8859-13", "iso8859-13", "iso885913"]),
            ("ISO-8859-14", &["iso-8859-14", "iso8859-14", "iso885914"]),
            (
                "ISO-8859-15",
                &[
                    "csisolatin9",
                    "iso-8859-15",
                    "iso8859-15",
                    "iso885915",
                    "iso_8859-15",
                    "l9",
                ],
            ),
            ("ISO-8859-16", &["iso-8859-16"]),
            ("KOI8-R", &["cskoi8r", "koi", "koi8", "koi8-r", "koi8_r"]),
            ("KOI8-U", &["koi8-ru", "koi8-u"]),
            ("macintosh", &["csmacintosh", "mac", "macintosh", "x-mac-roman"]),
            (
                "windows-874",
                &[
                    "dos-874",
                    "iso-8859-11",
                    "iso8859-11",
                    "iso885911",
                    "tis-620",
                    "windows-874",
                ],
            ),
            ("windows-1250", &["cp1250", "windows-1250", "x-cp1250"]),
            ("windows-1251", &["cp1251", "windows-1251", "x-cp1251"]),
            (
                "windows-1252",
                &[
                    "ansi_x3.4-1968",
                    "ascii",
                    "cp1252",
                    "cp819",
                    "csisolatin1",
                    "ibm819",
                    "iso-8859-1",
                    "iso-ir-100",
                    "iso8859-1",
                    "iso88591",
                    "iso_8859-1",
                    "iso_8859-1:1987",
                    "l1",
                    "latin1",
                    "us-ascii",
                    "windows-1252",
                    "x-cp1252",
                ],
            ),
            ("windows-1253", &["cp1253", "windows-1253", "x-cp1253"]),
            (
                "windows-1254",
                &[
                    "cp1254",
                    "csisolatin5",
                    "iso-8859-9",
                    "iso-ir-148",
                    "iso8859-9",
                    "iso88599",
                    "iso_8859-9",
                    "iso_8859-9:1989",
                    "l5",
                    "latin5",
                    "windows-1254",
                    "x-cp1254",
                ],
            ),
            ("windows-1255", &["cp1255", "windows-1255", "x-cp1255"]),
            ("windows-1256", &["cp1256", "windows-1256", "x-cp1256"]),
            ("windows-1257", &["cp1257", "windows-1257", "x-cp1257"]),
            ("windows-1258", &["cp1258", "windows-1258", "x-cp1258"]),
            ("x-mac-cyrillic", &["x-mac-cyrillic", "x-mac-ukrainian"]),
        ];
        let mut total = 0usize;
        for (name, labels) in expected {
            let mut want: Vec<&str> = labels.to_vec();
            want.sort_unstable();
            assert_eq!(family_of(name), want, "{name}: label family");
            total += labels.len();
        }
        assert_eq!(total, LABELS.len(), "a label is missing from a family");
        assert_eq!(LABELS.len(), 183);
    }

    #[test]
    fn no_label_names_two_encodings() {
        let mut supported: Vec<&str> = LABELS.iter().map(|(label, _)| *label).collect();
        let total = supported.len();
        supported.sort_unstable();
        supported.dedup();
        assert_eq!(supported.len(), total, "the same label is listed twice");
        let mut all = total;
        all += UNIMPLEMENTED_LABELS.len();
        let mut merged: Vec<&str> = LABELS.iter().map(|(label, _)| *label).collect();
        merged.extend_from_slice(&UNIMPLEMENTED_LABELS[..]);
        merged.sort_unstable();
        merged.dedup();
        assert_eq!(merged.len(), all, "a label is both supported and unimplemented");
        assert_eq!(all, 228, "the standard assigns 228 labels here");
        for (label, _) in &LABELS {
            let hits = LABELS
                .iter()
                .filter(|(other, _)| other.eq_ignore_ascii_case(label))
                .count();
            assert_eq!(hits, 1, "{label}");
        }
    }

    #[test]
    fn every_registry_name_is_itself_a_label() {
        for &encoding in &ENCODINGS {
            let lowercase = encoding.name.to_ascii_lowercase();
            assert_eq!(
                lookup(&lowercase),
                LabelLookup::Supported(encoding),
                "{}",
                encoding.name
            );
        }
    }

    #[test]
    fn labels_point_at_the_registry_statics() {
        for (label, encoding) in &LABELS {
            match lookup(label) {
                LabelLookup::Supported(found) => {
                    assert!(core::ptr::eq(found, *encoding), "{label}");
                }
                other => panic!("{label}: {other:?}"),
            }
        }
        // ...and every registry entry is reachable through at least one label.
        for &encoding in &ENCODINGS {
            assert!(
                LABELS
                    .iter()
                    .any(|(_, e)| core::ptr::eq(*e, encoding)),
                "{} has no label",
                encoding.name
            );
        }
    }

    #[test]
    fn unregistered_lookalikes_are_unknown() {
        // The standard registers none of these, so `lookup` must not guess.
        for label in [
            "utf-7",
            "utf-32",
            "utf-32be",
            "utf-32le",
            "ucs-4",
            "csucs4",
            "iso-8859-0",
            "iso-8859-12",
            "iso-8859-17",
            "windows-1259",
            "csiso2022cn",
            "x-mac-greek",
            "iso-8859-1:1987",
            "utf-8bom",
            "csunicode11",
        ] {
            assert_eq!(lookup(label), LabelLookup::Unknown, "{label}");
            assert!(for_label(label).is_none(), "{label}");
        }
    }

    #[test]
    fn lookup_and_for_label_agree_on_every_candidate() {
        for label in [
            "utf-8",
            "  UTF-8 ",
            "gb18030",
            "big5",
            "tis-620",
            "iso-8859-1",
            "utf-7",
            "",
            "\t",
            "x-user-defined",
            "koi8-ru",
            "cp1252",
        ] {
            let supported = matches!(lookup(label), LabelLookup::Supported(_));
            assert_eq!(for_label(label).is_some(), supported, "{label:?}");
            if let LabelLookup::Supported(encoding) = lookup(label) {
                let found = for_label(label).unwrap_or_else(|| panic!("{label:?}"));
                assert!(core::ptr::eq(found, encoding), "{label:?}");
            }
        }
    }

    #[test]
    fn only_utf8_can_encode_everything() {
        assert!(UTF_8.can_encode_everything());
        assert!(!UTF_16BE.can_encode_everything());
        assert!(!UTF_16LE.can_encode_everything());
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            assert!(!encoding.can_encode_everything(), "{}", encoding.name);
        }
        assert_eq!(
            ENCODINGS
                .iter()
                .filter(|e| e.can_encode_everything())
                .count(),
            1
        );
    }

    #[test]
    fn variant_shapes_match_the_registry_partition() {
        assert!(matches!(UTF_8.variant, VariantEncoding::Utf8));
        assert!(matches!(UTF_16BE.variant, VariantEncoding::Utf16Be));
        assert!(matches!(UTF_16LE.variant, VariantEncoding::Utf16Le));
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            assert!(
                matches!(encoding.variant, VariantEncoding::SingleByte(..)),
                "{}",
                encoding.name
            );
            assert!(encoding.variant.is_single_byte(), "{}", encoding.name);
        }
        assert_eq!(
            ENCODINGS
                .iter()
                .filter(|e| e.variant.is_single_byte())
                .count(),
            SINGLE_BYTE_ENCODINGS.len()
        );
    }

    #[test]
    fn run_parameters_are_the_documented_triples() {
        let expected: [(&str, u16, u8, u8); 28] = [
            ("IBM866", 0x0410, 0x00, 48),
            ("ISO-8859-2", 0x0080, 0x00, 33),
            ("ISO-8859-3", 0x0080, 0x00, 33),
            ("ISO-8859-4", 0x0080, 0x00, 33),
            ("ISO-8859-5", 0x040E, 0x2E, 66),
            ("ISO-8859-6", 0x0080, 0x00, 33),
            ("ISO-8859-7", 0x03A3, 0x53, 44),
            ("ISO-8859-8", 0x0080, 0x00, 33),
            ("ISO-8859-8-I", 0x0080, 0x00, 33),
            ("ISO-8859-10", 0x0080, 0x00, 33),
            ("ISO-8859-13", 0x0080, 0x00, 33),
            ("ISO-8859-14", 0x0080, 0x00, 33),
            ("ISO-8859-15", 0x00BF, 0x3F, 65),
            ("ISO-8859-16", 0x0080, 0x00, 33),
            ("KOI8-R", 0x2553, 0x24, 15),
            ("KOI8-U", 0x0418, 0x69, 8),
            ("macintosh", 0x02D8, 0x79, 3),
            ("windows-874", 0x0E01, 0x21, 58),
            ("windows-1250", 0x00B4, 0x34, 5),
            ("windows-1251", 0x0410, 0x40, 64),
            ("windows-1252", 0x00A0, 0x20, 96),
            ("windows-1253", 0x03A3, 0x53, 44),
            ("windows-1254", 0x00A0, 0x20, 48),
            ("windows-1255", 0x05D0, 0x60, 27),
            ("windows-1256", 0x0621, 0x41, 22),
            ("windows-1257", 0x00B0, 0x30, 8),
            ("windows-1258", 0x00A0, 0x20, 35),
            ("x-mac-cyrillic", 0x0410, 0x00, 32),
        ];
        for (name, bmp_offset, byte_offset, length) in expected {
            assert_eq!(run_of(by_name(name)), (bmp_offset, byte_offset, length), "{name}");
        }
        assert_eq!(expected.len(), SINGLE_BYTE_ENCODINGS.len());
    }

    #[test]
    fn every_run_maps_arithmetically_onto_its_index() {
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let table = table_of(encoding);
            let (bmp_offset, byte_offset, length) = run_of(encoding);
            let start = usize::from(byte_offset);
            let length = usize::from(length);
            assert!((1..=128).contains(&length), "{}: run length", encoding.name);
            assert!(
                start + length <= table.len(),
                "{}: the run leaves the index",
                encoding.name
            );
            for (offset, &unit) in table[start..start + length].iter().enumerate() {
                assert_eq!(
                    unit,
                    bmp_offset + offset as u16,
                    "{}: pointer {:02X}",
                    encoding.name,
                    0x80 + start + offset
                );
            }
        }
    }

    #[test]
    fn no_run_extends_one_pointer_further() {
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let table = table_of(encoding);
            let (bmp_offset, byte_offset, length) = run_of(encoding);
            let start = usize::from(byte_offset);
            let end = start + usize::from(length);
            if start > 0 {
                assert_ne!(
                    table[start - 1],
                    bmp_offset - 1,
                    "{}: the run could start one pointer earlier",
                    encoding.name
                );
            }
            if end < table.len() {
                assert_ne!(
                    table[end],
                    bmp_offset + length as u16,
                    "{}: the run could be one pointer longer",
                    encoding.name
                );
            }
        }
    }

    #[test]
    fn every_index_has_the_documented_number_of_defined_pointers() {
        let exceptions: [(&str, usize); 9] = [
            ("ISO-8859-3", 121),
            ("ISO-8859-6", 83),
            ("ISO-8859-7", 125),
            ("ISO-8859-8", 92),
            ("ISO-8859-8-I", 92),
            ("windows-874", 120),
            ("windows-1253", 125),
            ("windows-1255", 118),
            ("windows-1257", 126),
        ];
        let mut total = 0usize;
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let want = exceptions
                .iter()
                .find(|(name, _)| *name == encoding.name)
                .map_or(128, |(_, count)| *count);
            let got = table_of(encoding)
                .iter()
                .filter(|unit| **unit != 0)
                .count();
            assert_eq!(got, want, "{}", encoding.name);
            total += got;
        }
        assert_eq!(total, 3434, "defined pointers across the registry");
        assert_eq!(total, 28 * 128 - 150);
    }

    #[test]
    fn every_index_is_injective() {
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let mut units: Vec<u16> = table_of(encoding)
                .iter()
                .copied()
                .filter(|unit| *unit != 0)
                .collect();
            let count = units.len();
            units.sort_unstable();
            units.dedup();
            assert_eq!(
                units.len(),
                count,
                "{}: two pointers share a code point",
                encoding.name
            );
        }
    }

    #[test]
    fn every_index_entry_is_a_bmp_scalar_above_basic_latin() {
        let mut highest = 0u16;
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            for &unit in table_of(encoding) {
                if unit == 0 {
                    continue;
                }
                assert!(unit >= 0x0080, "{}: maps into basic Latin", encoding.name);
                assert!(
                    !matches!(unit, 0xD800..=0xDFFF),
                    "{}: U+{unit:04X} is a surrogate",
                    encoding.name
                );
                assert!(
                    !matches!(unit, 0xFFFE | 0xFFFF),
                    "{}: U+{unit:04X} is not a character",
                    encoding.name
                );
                highest = highest.max(unit);
            }
        }
        assert_eq!(highest, 0xFB02, "the registry's highest pointer");
    }

    #[test]
    fn no_break_space_is_at_byte_a0_except_in_five_code_pages() {
        let exceptions = ["IBM866", "KOI8-R", "KOI8-U", "macintosh", "x-mac-cyrillic"];
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let pointers: Vec<usize> = table_of(encoding)
                .iter()
                .enumerate()
                .filter(|(_, unit)| **unit == 0x00A0)
                .map(|(pointer, _)| pointer)
                .collect();
            assert_eq!(pointers.len(), 1, "{}", encoding.name);
            let byte = 0x80 + pointers[0] as u16;
            if exceptions.contains(&encoding.name) {
                assert_ne!(byte, 0x00A0, "{}", encoding.name);
            } else {
                assert_eq!(byte, 0x00A0, "{}", encoding.name);
            }
        }
    }

    #[test]
    fn the_runs_start_either_at_the_c1_row_or_at_no_break_space() {
        let c1_row: Vec<&str> = SINGLE_BYTE_ENCODINGS
            .iter()
            .filter(|e| run_of(e) == (0x0080, 0x00, 33))
            .map(|e| e.name)
            .collect();
        let nbsp_row: Vec<&str> = SINGLE_BYTE_ENCODINGS
            .iter()
            .filter(|e| {
                let (bmp_offset, byte_offset, _) = run_of(e);
                (bmp_offset, byte_offset) == (0x00A0, 0x20)
            })
            .map(|e| e.name)
            .collect();
        assert_eq!(
            c1_row,
            [
                "ISO-8859-2",
                "ISO-8859-3",
                "ISO-8859-4",
                "ISO-8859-6",
                "ISO-8859-8",
                "ISO-8859-8-I",
                "ISO-8859-10",
                "ISO-8859-13",
                "ISO-8859-14",
                "ISO-8859-16",
            ]
        );
        assert_eq!(nbsp_row, ["windows-1252", "windows-1254", "windows-1258"]);
        // Thirteen indexes reach the no-break space arithmetically: the ten C1 runs
        // end exactly on it and the three Latin-1 tails begin on it.
        let in_run = SINGLE_BYTE_ENCODINGS
            .iter()
            .filter(|e| {
                let (_, byte_offset, length) = run_of(e);
                let start = usize::from(byte_offset);
                table_of(e)[start..start + usize::from(length)].contains(&0x00A0)
            })
            .count();
        assert_eq!(in_run, 13);
    }

    #[test]
    fn only_the_iso_8859_indexes_identity_map_the_c1_row() {
        let iso: [&str; 13] = [
            "ISO-8859-2",
            "ISO-8859-3",
            "ISO-8859-4",
            "ISO-8859-5",
            "ISO-8859-6",
            "ISO-8859-7",
            "ISO-8859-8",
            "ISO-8859-8-I",
            "ISO-8859-10",
            "ISO-8859-13",
            "ISO-8859-14",
            "ISO-8859-15",
            "ISO-8859-16",
        ];
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let identity = table_of(encoding)[..=32]
                .iter()
                .enumerate()
                .all(|(pointer, &unit)| unit == 0x0080 + pointer as u16);
            assert_eq!(identity, iso.contains(&encoding.name), "{}", encoding.name);
        }
    }

    #[test]
    fn windows_1252_is_latin1_exact_from_no_break_space_onward() {
        let table = table_of(&WINDOWS_1252);
        for (offset, &unit) in table[0x20..].iter().enumerate() {
            assert_eq!(unit, 0x00A0 + offset as u16, "byte {:02X}", 0xA0 + offset);
        }
        assert_eq!(run_of(&WINDOWS_1252), (0x00A0, 0x20, 96));
        assert_ne!(table[0x00], 0x0080, "cp1252 replaces the C1 controls");
    }

    #[test]
    fn windows_1254_tracks_latin1_only_to_byte_cf() {
        let table = table_of(&WINDOWS_1254);
        for (offset, &unit) in table[0x20..0x50].iter().enumerate() {
            assert_eq!(unit, 0x00A0 + offset as u16, "byte {:02X}", 0xA0 + offset);
        }
        assert_ne!(table[0x50], 0x00D0, "byte D0: Latin-1 diverges here");
        assert_eq!(run_of(&WINDOWS_1254), (0x00A0, 0x20, 48));
    }

    #[test]
    fn koi8_u_and_koi8_r_differ_only_in_the_screen_graphics_row() {
        let r = table_of(&KOI8_R);
        let u = table_of(&KOI8_U);
        let diffs: Vec<u16> = (0..r.len())
            .filter(|&pointer| r[pointer] != u[pointer])
            .map(|pointer| 0x80 + pointer as u16)
            .collect();
        assert_eq!(diffs.len(), 10);
        assert!(
            diffs
                .iter()
                .all(|&byte| (0xA4..=0xBE).contains(&byte)),
            "{diffs:?}"
        );
        assert_eq!(&r[..0x24], &u[..0x24], "bytes 80..=A3");
        assert_eq!(&r[0x40..], &u[0x40..], "bytes C0..=FF");
        assert_ne!(run_of(&KOI8_R), run_of(&KOI8_U));
    }

    #[test]
    fn windows_1253_and_iso_8859_7_share_their_greek_run() {
        assert_eq!(run_of(&WINDOWS_1253), run_of(&ISO_8859_7));
        let (bmp_offset, byte_offset, length) = run_of(&WINDOWS_1253);
        assert_eq!((bmp_offset, byte_offset, length), (0x03A3, 0x53, 44));
        let windows = table_of(&WINDOWS_1253);
        let greek = table_of(&ISO_8859_7);
        assert_eq!(&windows[0x37..], &greek[0x37..], "bytes B7..=FF");
        let diffs = (0..greek.len())
            .filter(|&pointer| greek[pointer] != windows[pointer])
            .count();
        assert_eq!(diffs, 26);
    }

    #[test]
    fn iso_8859_6_has_the_most_undefined_pointers() {
        let counts: Vec<(&str, usize)> = SINGLE_BYTE_ENCODINGS
            .iter()
            .map(|&e| {
                (
                    e.name,
                    table_of(e)
                        .iter()
                        .filter(|unit| **unit == 0)
                        .count(),
                )
            })
            .collect();
        let max = counts
            .iter()
            .map(|(_, count)| *count)
            .max()
            .unwrap_or_default();
        assert_eq!(max, 45);
        let winners: Vec<&str> = counts
            .iter()
            .filter(|(_, count)| *count == max)
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(winners, ["ISO-8859-6"]);
    }

    #[test]
    fn only_the_two_hebrew_entries_share_an_index_table() {
        let mut shared = 0usize;
        for (position, &first) in SINGLE_BYTE_ENCODINGS.iter().enumerate() {
            for &second in &SINGLE_BYTE_ENCODINGS[position + 1..] {
                if core::ptr::eq(table_of(first), table_of(second)) {
                    shared += 1;
                    assert_eq!((first.name, second.name), ("ISO-8859-8", "ISO-8859-8-I"));
                }
            }
        }
        assert_eq!(shared, 1);
    }

    #[test]
    fn registry_names_are_unique() {
        let mut names: Vec<&str> = ENCODINGS.iter().map(|e| e.name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total);
    }

    #[test]
    fn every_basic_latin_byte_decodes_to_its_own_code_point() {
        let ascii: Vec<u8> = (0u8..0x80).collect();
        let want: Vec<u16> = (0u16..0x80).collect();
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            assert_eq!(units(encoding, &ascii), want, "{}", encoding.name);
        }
        // The C0 controls below U+0020 are no exception, and neither is NUL.
        assert_eq!(
            units(&WINDOWS_1252, &[0x00, 0x1F, 0x20, 0x7F]),
            vec![0x0000u16, 0x001F, 0x0020, 0x007F]
        );
    }

    #[test]
    fn every_basic_latin_char_encodes_to_its_own_byte() {
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            for code in 0u16..0x80 {
                assert_eq!(
                    encode_unit(encoding, code),
                    Ok(code as u8),
                    "{}: U+{code:04X}",
                    encoding.name
                );
            }
        }
    }

    #[test]
    fn a_batch_decode_matches_byte_by_byte_decoding() {
        let batch: Vec<u8> = (0x80u8..=0xFF).collect();
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let one_by_one: Vec<u16> = batch
                .iter()
                .flat_map(|&byte| units(encoding, &[byte]))
                .collect();
            assert_eq!(units(encoding, &batch), one_by_one, "{}", encoding.name);
        }
    }

    #[test]
    fn decoding_continues_when_the_destination_fills() {
        let batch: Vec<u8> = (0x80u8..=0xFF).collect();
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let whole = units(encoding, &batch);
            for chunk in [1usize, 2, 3, 7] {
                let mut out = Vec::new();
                let mut decoder = encoding.new_decoder_without_bom_handling();
                let mut src = &batch[..];
                while !src.is_empty() {
                    let mut dst = vec![0u16; chunk];
                    let (result, read, written, _) = decoder.decode_to_utf16(src, &mut dst, true);
                    assert!(
                        matches!(result, CoderResult::InputEmpty | CoderResult::OutputFull),
                        "{}: {result:?}",
                        encoding.name
                    );
                    assert!(
                        read > 0,
                        "{}: no progress with a {chunk}-unit destination",
                        encoding.name
                    );
                    out.extend_from_slice(&dst[..written]);
                    src = &src[read..];
                }
                assert_eq!(out, whole, "{}: chunk {chunk}", encoding.name);
            }
        }
    }

    #[test]
    fn the_utf8_encode_path_agrees_with_the_utf16_path() {
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            for &unit in table_of(encoding) {
                if unit == 0 {
                    continue;
                }
                let ch = char::from_u32(u32::from(unit)).unwrap_or(char::REPLACEMENT_CHARACTER);
                let mut dst = [0u8; 4];
                let mut encoder = encoding.new_encoder();
                let text = ch.to_string();
                let (result, read, written) =
                    encoder.encode_from_utf8_without_replacement(&text, &mut dst, true);
                assert_eq!(result, EncoderResult::InputEmpty, "{}: {ch:?}", encoding.name);
                assert_eq!((read, written), (text.len(), 1), "{}: {ch:?}", encoding.name);
                let want = match encode_unit(encoding, unit) {
                    Ok(byte) => byte,
                    Err(result) => panic!("{}: U+{unit:04X} should be defined: {result:?}", encoding.name),
                };
                assert_eq!(dst[0], want, "{}: {ch:?}", encoding.name);
            }
        }
    }

    #[test]
    fn the_encoder_is_the_exact_reverse_of_its_index() {
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let table = table_of(encoding);
            let mut reverse = vec![0u16; 0x1_0000];
            for (pointer, &unit) in table.iter().enumerate() {
                if unit != 0 {
                    reverse[unit as usize] = pointer as u16 + 1;
                }
            }
            for unit in 0x0080u16..=0xFFFF {
                let pointer = reverse[unit as usize];
                let expected = if pointer == 0 {
                    let ch = char::from_u32(u32::from(unit)).unwrap_or(char::REPLACEMENT_CHARACTER);
                    Err(EncoderResult::Unmappable(ch))
                } else {
                    Ok(0x80 + (pointer - 1) as u8)
                };
                assert_eq!(
                    encode_unit(encoding, unit),
                    expected,
                    "{}: U+{unit:04X}",
                    encoding.name
                );
            }
        }
    }

    #[test]
    fn chars_absent_from_an_index_are_unmappable() {
        let cases: [(&str, u16); 18] = [
            ("windows-1252", 0x0540),
            ("windows-1252", 0x2008),
            ("windows-1252", 0x0410),
            ("ISO-8859-2", 0x0180),
            ("ISO-8859-3", 0x0170),
            ("ISO-8859-4", 0x04E9),
            ("ISO-8859-5", 0x04C0),
            ("ISO-8859-6", 0x0670),
            ("ISO-8859-7", 0x03F0),
            ("ISO-8859-8", 0x2763),
            ("ISO-8859-10", 0x01A2),
            ("ISO-8859-13", 0x01B0),
            ("ISO-8859-14", 0x0C80),
            ("ISO-8859-15", 0x01C0),
            ("windows-1251", 0x0400),
            ("windows-1253", 0x03F0),
            ("windows-1255", 0x0540),
            ("macintosh", 0x1400),
        ];
        for (name, unit) in cases {
            let encoding = by_name(name);
            assert_eq!(
                table_of(encoding).iter().position(|&u| u == unit),
                None,
                "{name}: U+{unit:04X} is in the index"
            );
            let ch = char::from_u32(u32::from(unit)).unwrap_or(char::REPLACEMENT_CHARACTER);
            assert_eq!(
                encode_unit(encoding, unit),
                Err(EncoderResult::Unmappable(ch)),
                "{name}"
            );
            let mut dst = [0u8; 4];
            let mut encoder = encoding.new_encoder();
            let text = ch.to_string();
            let (result, read, written) = encoder.encode_from_utf8_without_replacement(&text, &mut dst, true);
            assert_eq!(result, EncoderResult::Unmappable(ch), "{name}: {ch:?}");
            assert_eq!((read, written), (text.len(), 0), "{name}: {ch:?}");
        }
    }

    #[test]
    fn astral_input_is_unmappable_for_every_legacy_code_page() {
        let astral = '\u{104E0}';
        let pair: Vec<u16> = astral.to_string().encode_utf16().collect();
        assert_eq!(pair.len(), 2);
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let mut dst = [0u8; 4];
            let mut encoder = encoding.new_encoder();
            let (result, read, written) =
                encoder.encode_from_utf16_without_replacement(&pair, &mut dst, true);
            assert_eq!(result, EncoderResult::Unmappable(astral), "{}", encoding.name);
            assert_eq!((read, written), (2, 0), "{}", encoding.name);
            let text = astral.to_string();
            let (result, read, written) = encoder.encode_from_utf8_without_replacement(&text, &mut dst, true);
            assert_eq!(result, EncoderResult::Unmappable(astral), "{}", encoding.name);
            assert_eq!((read, written), (text.len(), 0), "{}", encoding.name);
        }
    }

    #[test]
    fn the_default_encode_path_replaces_unmappables_with_ncrs() {
        let mut dst = [0u8; NCR_EXTRA + 8];
        let mut encoder = WINDOWS_1252.new_encoder();
        let (result, read, written, had_unmappables) = encoder.encode_from_utf8("\u{0540}", &mut dst, true);
        assert_eq!(result, CoderResult::InputEmpty);
        assert!(had_unmappables);
        // One Armenian capital alone fills the eight-byte working space.
        assert_eq!((read, written), (2, 7), "U+0540 is 1344 in decimal");
        assert_eq!(&dst[..written], b"&#1344;");

        let mut encoder = WINDOWS_1252.new_encoder();
        let (result, read, written, had_unmappables) = encoder.encode_from_utf8("a\u{0540}", &mut dst, true);
        assert_eq!(result, CoderResult::InputEmpty);
        assert!(had_unmappables);
        assert_eq!((read, written), (3, 8));
        assert_eq!(&dst[..written], b"a&#1344;");

        // Astral code points get the same treatment from the UTF-16 side.
        let mut encoder = MACINTOSH.new_encoder();
        let pair: Vec<u16> = "\u{10000}".encode_utf16().collect();
        let (result, read, written, had_unmappables) = encoder.encode_from_utf16(&pair, &mut dst, true);
        assert_eq!(result, CoderResult::InputEmpty);
        assert!(had_unmappables);
        assert_eq!((read, written), (2, 8));
        assert_eq!(&dst[..written], b"&#65536;");

        // A buffer below the reservation is refused instead of truncated.
        let mut tiny = [0u8; NCR_EXTRA - 1];
        let (result, read, written, had) = WINDOWS_1252
            .new_encoder()
            .encode_from_utf8("a", &mut tiny, true);
        assert_eq!(
            (result, read, written, had),
            (CoderResult::OutputFull, 0, 0, false)
        );
        let (result, read, written, had) = WINDOWS_1252
            .new_encoder()
            .encode_from_utf8("", &mut tiny, true);
        assert_eq!(
            (result, read, written, had),
            (CoderResult::InputEmpty, 0, 0, false)
        );
    }

    #[test]
    fn decoding_through_the_replacing_wrapper_flags_errors() {
        let batch: Vec<u8> = (0x80u8..=0xFF).collect();
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let holes = table_of(encoding)
                .iter()
                .filter(|unit| **unit == 0)
                .count();
            let mut dst = vec![0u16; batch.len()];
            let (result, read, written, had_errors) = encoding
                .new_decoder_without_bom_handling()
                .decode_to_utf16(&batch, &mut dst, true);
            assert_eq!(result, CoderResult::InputEmpty, "{}", encoding.name);
            assert_eq!((read, written), (128, 128), "{}", encoding.name);
            assert_eq!(had_errors, holes > 0, "{}", encoding.name);
            let replacements = dst.iter().filter(|&&unit| unit == 0xFFFD).count();
            assert_eq!(replacements, holes, "{}", encoding.name);
        }
    }

    #[test]
    fn decoder_buffer_reservations_are_exact_for_single_byte() {
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let decoder = encoding.new_decoder_without_bom_handling();
            assert_eq!(
                decoder.max_utf16_buffer_length(128),
                Some(128),
                "{}",
                encoding.name
            );
            assert_eq!(
                decoder.max_utf8_buffer_length(128),
                Some(384),
                "{}",
                encoding.name
            );
            assert_eq!(
                decoder.max_utf8_buffer_length_without_replacement(128),
                Some(384),
                "{}",
                encoding.name
            );
            assert_eq!(decoder.max_utf16_buffer_length(0), Some(0), "{}", encoding.name);
        }
    }

    #[test]
    fn encoder_buffer_reservations_reserve_the_ncr_extra() {
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let encoder = encoding.new_encoder();
            assert_eq!(
                encoder.max_buffer_length_from_utf8_without_replacement(64),
                Some(64),
                "{}",
                encoding.name
            );
            assert_eq!(
                encoder.max_buffer_length_from_utf8_if_no_unmappables(64),
                Some(64 + NCR_EXTRA),
                "{}",
                encoding.name
            );
            assert_eq!(
                encoder.max_buffer_length_from_utf16_if_no_unmappables(64),
                Some(64 + NCR_EXTRA),
                "{}",
                encoding.name
            );
        }
        let encoder = UTF_8.new_encoder();
        assert_eq!(
            encoder.max_buffer_length_from_utf8_if_no_unmappables(64),
            Some(64),
            "UTF-8 encodes everything, so it needs no NCR reserve"
        );
        assert_eq!(
            encoder.max_buffer_length_from_utf16_without_replacement(64),
            Some(192)
        );
    }

    #[test]
    fn a_full_pointer_batch_needs_three_utf8_bytes_per_byte() {
        let batch: Vec<u8> = (0x80u8..=0xFF).collect();
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let mut dst = vec![0u8; batch.len() * 3];
            let (result, read, written, _) = encoding
                .new_decoder_without_bom_handling()
                .decode_to_utf8(&batch, &mut dst, true);
            assert_eq!(result, CoderResult::InputEmpty, "{}", encoding.name);
            assert_eq!(read, batch.len(), "{}", encoding.name);
            let text = String::from_utf8(dst[..written].to_vec())
                .unwrap_or_else(|e| panic!("{}: {e}", encoding.name));
            assert_eq!(text.chars().count(), batch.len(), "{}", encoding.name);
            assert_eq!(
                written,
                text.chars().map(char::len_utf8).sum::<usize>(),
                "{}",
                encoding.name
            );
        }
    }

    #[test]
    fn each_undefined_pointer_yields_exactly_one_replacement() {
        let mut holey = 0usize;
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let holes: Vec<u8> = table_of(encoding)
                .iter()
                .enumerate()
                .filter(|(_, unit)| **unit == 0)
                .map(|(pointer, _)| 0x80 + pointer as u8)
                .collect();
            if holes.is_empty() {
                continue;
            }
            holey += holes.len();
            let mut dst = vec![0u8; holes.len() * 3];
            let (result, read, written, had_errors) = encoding
                .new_decoder_without_bom_handling()
                .decode_to_utf8(&holes, &mut dst, true);
            assert_eq!(result, CoderResult::InputEmpty, "{}", encoding.name);
            assert_eq!(read, holes.len(), "{}", encoding.name);
            assert_eq!(written, holes.len() * 3, "{}", encoding.name);
            assert!(had_errors, "{}", encoding.name);
            assert!(
                dst.chunks(3)
                    .all(|chunk| chunk == [0xEF, 0xBF, 0xBD]),
                "{}",
                encoding.name
            );
            assert_eq!(
                units(encoding, &holes),
                vec![char::REPLACEMENT_CHARACTER as u16; holes.len()],
                "{}",
                encoding.name
            );
        }
        assert_eq!(holey, 150, "undefined pointers across the registry");
    }

    #[test]
    fn decoding_resumes_after_an_undefined_pointer() {
        let mut out = [0u16; 4];
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let holes: Vec<u8> = table_of(encoding)
                .iter()
                .enumerate()
                .filter(|(_, unit)| **unit == 0)
                .map(|(pointer, _)| 0x80 + pointer as u8)
                .collect();
            if holes.is_empty() {
                continue;
            }
            let first = holes[0];
            let mut decoder = encoding.new_decoder_without_bom_handling();
            for byte in [0x41u8, first, 0x42] {
                let (result, read, written) =
                    decoder.decode_to_utf16_without_replacement(&[byte], &mut out, true);
                if byte == first {
                    assert_eq!(result, DecoderResult::Malformed(1, 0), "{}", encoding.name);
                    assert_eq!((read, written), (1, 0), "{}", encoding.name);
                } else {
                    assert_eq!(result, DecoderResult::InputEmpty, "{}", encoding.name);
                    assert_eq!((read, written), (1, 1), "{}", encoding.name);
                    assert_eq!(out[0], u16::from(byte), "{}", encoding.name);
                }
            }
            let (result, read, written) =
                decoder.decode_to_utf16_without_replacement(&[0x43, first, 0x44], &mut out, true);
            assert_eq!(result, DecoderResult::Malformed(1, 0), "{}", encoding.name);
            assert_eq!((read, written), (2, 1), "{}", encoding.name);
            assert_eq!(out[0], 0x0043, "{}", encoding.name);
            let (result, read, written) =
                decoder.decode_to_utf16_without_replacement(&[0x45], &mut out, true);
            assert_eq!(result, DecoderResult::InputEmpty, "{}", encoding.name);
            assert_eq!((read, written), (1, 1), "{}", encoding.name);
            assert_eq!(out[0], 0x0045, "{}", encoding.name);
        }
    }

    #[test]
    fn splitting_the_input_never_changes_the_output() {
        let batch: Vec<u8> = (0x80u8..=0xFF).collect();
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let whole = units(encoding, &batch);
            for split in [1usize, 40, 41, 42, 127] {
                let mut out = Vec::new();
                let mut decoder = encoding.new_decoder_without_bom_handling();
                for (part, last) in [(&batch[..split], false), (&batch[split..], true)] {
                    let mut dst = vec![0u16; part.len() + 4];
                    let (result, read, written, _) = decoder.decode_to_utf16(part, &mut dst, last);
                    assert_eq!(result, CoderResult::InputEmpty, "{}", encoding.name);
                    assert_eq!(read, part.len(), "{}", encoding.name);
                    out.extend_from_slice(&dst[..written]);
                }
                assert_eq!(out, whole, "{}: split at {split}", encoding.name);
            }
        }
    }

    #[test]
    fn legacy_coders_hold_no_pending_state() {
        let mut dst = [0u8; 16];
        for &encoding in &ENCODINGS {
            let mut encoder = encoding.new_encoder();
            assert!(!encoder.has_pending_state(), "{}", encoding.name);
            // Basic Latin plus the first pointer of the index, which every registry
            // index defines: both code points always encode, even with last = false.
            let (units, want_written) = match encoding.variant {
                VariantEncoding::SingleByte(table, ..) => ([0x0041u16, table[0]], 2),
                VariantEncoding::Utf8 => ([0x0041u16, 0x0042], 2),
                VariantEncoding::Utf16Be | VariantEncoding::Utf16Le => ([0x0041u16, 0x0042], 4),
            };
            let (result, read, written) =
                encoder.encode_from_utf16_without_replacement(&units, &mut dst, false);
            assert_eq!(result, EncoderResult::InputEmpty, "{}", encoding.name);
            assert_eq!((read, written), (2, want_written), "{}", encoding.name);
            assert!(!encoder.has_pending_state(), "{}", encoding.name);
        }
    }

    #[test]
    fn coder_accessors_name_their_encoding() {
        for &encoding in &ENCODINGS {
            assert!(
                core::ptr::eq(encoding.new_decoder().encoding(), encoding),
                "{}",
                encoding.name
            );
            assert!(
                core::ptr::eq(encoding.new_decoder_with_bom_removal().encoding(), encoding),
                "{}",
                encoding.name
            );
            assert!(
                core::ptr::eq(encoding.new_encoder().encoding(), encoding),
                "{}",
                encoding.name
            );
            assert!(core::ptr::eq(encoding.encoding(), encoding), "{}", encoding.name);
        }
    }

    #[test]
    fn bom_handling_is_a_no_op_for_every_legacy_code_page() {
        let input = [0xEF, 0xBB, 0xBF, 0x68, 0x69];
        for &encoding in &SINGLE_BYTE_ENCODINGS {
            let plain = units(encoding, &input);
            let default = decode_all(&mut encoding.new_decoder(), &input);
            let removal = decode_all(&mut encoding.new_decoder_with_bom_removal(), &input);
            assert_eq!(plain.len(), 5, "{}", encoding.name);
            assert_eq!(default, plain, "{}: new_decoder", encoding.name);
            assert_eq!(removal, plain, "{}: with_bom_removal", encoding.name);
        }
        // UTF-8 is where the constructors ought to differ, and they do not: the
        // `BomHandling` mode only records a `life_cycle` (`encoding.rs:548-562`) that
        // neither `decode_to_utf8` nor `decode_to_utf16` ever reads, so the mark
        // decodes as U+FEFF whichever handle is used.
        let plain = decode_all(&mut UTF_8.new_decoder_without_bom_handling(), &input);
        let default = decode_all(&mut UTF_8.new_decoder(), &input);
        let removal = decode_all(&mut UTF_8.new_decoder_with_bom_removal(), &input);
        assert_eq!(default, plain, "new_decoder is the no-handling constructor");
        assert_eq!(removal, default, "the removal flag changes nothing");
        assert_eq!(plain, vec![0xFEFFu16, 0x0068, 0x0069]);
    }

    #[test]
    fn for_bom_never_selects_a_legacy_encoding() {
        for byte in 0x00u8..=0xFF {
            for candidate in [
                vec![byte],
                vec![byte, byte],
                vec![byte, byte, byte],
                vec![byte, 0xBB, 0xBF],
                vec![byte, 0xFF],
                vec![byte, 0xFE],
            ] {
                let found = for_bom(&candidate);
                assert!(
                    found.map_or(true, |e| matches!(e.name, "UTF-8" | "UTF-16BE" | "UTF-16LE")),
                    "{candidate:?} -> {}",
                    found.map_or("nothing", |e| e.name)
                );
            }
        }
        assert_eq!(for_bom(&[0xC0, 0x80]), None, "a legacy overlong start is no mark");
    }
}
