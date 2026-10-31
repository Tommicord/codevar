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
