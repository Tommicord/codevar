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

//! # Codevar FreeType core
//!
//! A faithful Rust port of the FreeType 2.6 core kernel: scalar type system,
//! error codes, object model (library / face / size / glyph slot / modules),
//! the base calculation, stream, memory, glyph loader, hash, debug/trace,
//! outline, bitmap, LCD filter and system interfaces, plus the MD5 helper and
//! a runtime-dispatched SIMD acceleration layer.
//!
//! The port keeps FreeType's semantics (fixed-point arithmetic, stream frame
//! discipline, bitmap flow/pitch rules) while mapping ownership onto safe
//! Rust: shared objects are reference counted (`Arc`) with interior mutability
//! ([`LibCell`]) for the fields FreeType mutates through shared handles, and
//! no panicking operation is used in production paths.

#![cfg_attr(not(test), no_std)]
extern crate alloc;

use core::fmt;

/// Result alias used by every fallible entry point of this crate.
///
/// FreeType reports failures with `Error` codes; here they are mapped to
/// `Result` so the `?` operator can be used throughout.
pub type TtResult<T> = Result<T, TtError>;

/// A 2.14 fixed-point value.
pub type F2Dot14 = i16;
/// A 26.6 fixed-point value (26 integer bits, 6 fractional bits).
pub type F26Dot6 = i64;
/// A 16.16 fixed-point value.
pub type Fixed = i64;
/// A design-space position, in 26.6 units (or font units).
pub type Pos = i64;
/// A 32-bit tag, as used by the SFNT format.
pub type Tag = u32;
/// Angle type used for FreeType trigonometry (16.16 degrees).
///
/// The type and the `ANGLE_*` constants live in `fttrigon.h`, which is not
/// part of this port's module set, but they are required by [`Hypot`]'s
/// documentation and by outline emboldening helpers, so they are defined here.
pub type Angle = Fixed;

/// `ANGLE_PI`: pi expressed in [`Angle`] units (16.16 degrees).
pub const ANGLE_PI: Angle = 180 << 16;
/// `ANGLE_2PI`: 2*pi expressed in [`Angle`] units.
pub const ANGLE_2PI: Angle = ANGLE_PI * 2;
/// `ANGLE_PI2`: pi/2 expressed in [`Angle`] units.
pub const ANGLE_PI2: Angle = ANGLE_PI / 2;
/// `ANGLE_PI4`: pi/4 expressed in [`Angle`] units.
pub const ANGLE_PI4: Angle = ANGLE_PI / 4;

/// A FreeType error code.
///
/// The numeric values are byte-identical to FreeType's `Err_XXX`
/// constants; the low byte carries the base error and bits 8..15 the module
/// that produced it (see [`TtError::module`] and [`TtError::base`]).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct TtError(pub i32);

impl TtError {
    /// `Ok` / `Err_Ok`: no error.
    pub const OK: TtError = TtError(0x00);
    /// `Cannot_Open_Resource`
    pub const CANNOT_OPEN_RESOURCE: TtError = TtError(0x01);
    /// `Unknown_File_Format`
    pub const UNKNOWN_FILE_FORMAT: TtError = TtError(0x02);
    /// `Invalid_File_Format`
    pub const INVALID_FILE_FORMAT: TtError = TtError(0x03);
    /// `Invalid_Version`
    pub const INVALID_VERSION: TtError = TtError(0x04);
    /// `Lower_Module_Version`
    pub const LOWER_MODULE_VERSION: TtError = TtError(0x05);
    /// `Invalid_Argument`
    pub const INVALID_ARGUMENT: TtError = TtError(0x06);
    /// `Unimplemented_Feature`
    pub const UNIMPLEMENTED_FEATURE: TtError = TtError(0x07);
    /// `Invalid_Table`
    pub const INVALID_TABLE: TtError = TtError(0x08);
    /// `Invalid_Offset`
    pub const INVALID_OFFSET: TtError = TtError(0x09);
    /// `Array_Too_Large`
    pub const ARRAY_TOO_LARGE: TtError = TtError(0x0A);
    /// `Missing_Module`
    pub const MISSING_MODULE: TtError = TtError(0x0B);
    /// `Missing_Property`
    pub const MISSING_PROPERTY: TtError = TtError(0x0C);
    /// `Invalid_Glyph_Index`
    pub const INVALID_GLYPH_INDEX: TtError = TtError(0x10);
    /// `Invalid_Character_Code`
    pub const INVALID_CHARACTER_CODE: TtError = TtError(0x11);
    /// `Invalid_Glyph_Format`
    pub const INVALID_GLYPH_FORMAT: TtError = TtError(0x12);
    /// `Cannot_Render_Glyph`
    pub const CANNOT_RENDER_GLYPH: TtError = TtError(0x13);
    /// `Invalid_Outline`
    pub const INVALID_OUTLINE: TtError = TtError(0x14);
    /// `Invalid_Composite`
    pub const INVALID_COMPOSITE: TtError = TtError(0x15);
    /// `Too_Many_Hints`
    pub const TOO_MANY_HINTS: TtError = TtError(0x16);
    /// `Invalid_Pixel_Size`
    pub const INVALID_PIXEL_SIZE: TtError = TtError(0x17);
    /// `Invalid_Handle`
    pub const INVALID_HANDLE: TtError = TtError(0x20);
    /// `Invalid_Library_Handle`
    pub const INVALID_LIBRARY_HANDLE: TtError = TtError(0x21);
    /// `Invalid_Driver_Handle`
    pub const INVALID_DRIVER_HANDLE: TtError = TtError(0x22);
    /// `Invalid_Face_Handle`
    pub const INVALID_FACE_HANDLE: TtError = TtError(0x23);
    /// `Invalid_Size_Handle`
    pub const INVALID_SIZE_HANDLE: TtError = TtError(0x24);
    /// `Invalid_Slot_Handle`
    pub const INVALID_SLOT_HANDLE: TtError = TtError(0x25);
    /// `Invalid_CharMap_Handle`
    pub const INVALID_CHARMAP_HANDLE: TtError = TtError(0x26);
    /// `Invalid_Cache_Handle`
    pub const INVALID_CACHE_HANDLE: TtError = TtError(0x27);
    /// `Invalid_Stream_Handle`
    pub const INVALID_STREAM_HANDLE: TtError = TtError(0x28);
    /// `Too_Many_Drivers`
    pub const TOO_MANY_DRIVERS: TtError = TtError(0x30);
    /// `Too_Many_Extensions`
    pub const TOO_MANY_EXTENSIONS: TtError = TtError(0x31);
    /// `Out_Of_Memory`
    pub const OUT_OF_MEMORY: TtError = TtError(0x40);
    /// `Unlisted_Object`
    pub const UNLISTED_OBJECT: TtError = TtError(0x41);
    /// `Cannot_Open_Stream`
    pub const CANNOT_OPEN_STREAM: TtError = TtError(0x51);
    /// `Invalid_Stream_Seek`
    pub const INVALID_STREAM_SEEK: TtError = TtError(0x52);
    /// `Invalid_Stream_Skip`
    pub const INVALID_STREAM_SKIP: TtError = TtError(0x53);
    /// `Invalid_Stream_Read`
    pub const INVALID_STREAM_READ: TtError = TtError(0x54);
    /// `Invalid_Stream_Operation`
    pub const INVALID_STREAM_OPERATION: TtError = TtError(0x55);
    /// `Invalid_Frame_Operation`
    pub const INVALID_FRAME_OPERATION: TtError = TtError(0x56);
    /// `Nested_Frame_Access`
    pub const NESTED_FRAME_ACCESS: TtError = TtError(0x57);
    /// `Invalid_Frame_Read`
    pub const INVALID_FRAME_READ: TtError = TtError(0x58);
    /// `Raster_Uninitialized`
    pub const RASTER_UNINITIALIZED: TtError = TtError(0x60);
    /// `Raster_Corrupted`
    pub const RASTER_CORRUPTED: TtError = TtError(0x61);
    /// `Raster_Overflow`
    pub const RASTER_OVERFLOW: TtError = TtError(0x62);
    /// `Raster_Negative_Height`
    pub const RASTER_NEGATIVE_HEIGHT: TtError = TtError(0x63);
    /// `Too_Many_Caches`
    pub const TOO_MANY_CACHES: TtError = TtError(0x70);
    /// `Invalid_Opcode`
    pub const INVALID_OPCODE: TtError = TtError(0x80);
    /// `Too_Few_Arguments`
    pub const TOO_FEW_ARGUMENTS: TtError = TtError(0x81);
    /// `Stack_Overflow`
    pub const STACK_OVERFLOW: TtError = TtError(0x82);
    /// `Code_Overflow`
    pub const CODE_OVERFLOW: TtError = TtError(0x83);
    /// `Bad_Argument`
    pub const BAD_ARGUMENT: TtError = TtError(0x84);
    /// `Divide_By_Zero`
    pub const DIVIDE_BY_ZERO: TtError = TtError(0x85);
    /// `Invalid_Reference`
    pub const INVALID_REFERENCE: TtError = TtError(0x86);
    /// `Debug_OpCode`
    pub const DEBUG_OPCODE: TtError = TtError(0x87);
    /// `ENDF_In_Exec_Stream`
    pub const ENDF_IN_EXEC_STREAM: TtError = TtError(0x88);
    /// `Nested_DEFS`
    pub const NESTED_DEFS: TtError = TtError(0x89);
    /// `Invalid_CodeRange`
    pub const INVALID_CODE_RANGE: TtError = TtError(0x8A);
    /// `Execution_Too_Long`
    pub const EXECUTION_TOO_LONG: TtError = TtError(0x8B);
    /// `Too_Many_Function_Defs`
    pub const TOO_MANY_FUNCTION_DEFS: TtError = TtError(0x8C);
    /// `Too_Many_Instruction_Defs`
    pub const TOO_MANY_INSTRUCTION_DEFS: TtError = TtError(0x8D);
    /// `Table_Missing`
    pub const TABLE_MISSING: TtError = TtError(0x8E);
    /// `Horiz_Header_Missing`
    pub const HORIZ_HEADER_MISSING: TtError = TtError(0x8F);
    /// `Locations_Missing`
    pub const LOCATIONS_MISSING: TtError = TtError(0x90);
    /// `Name_Table_Missing`
    pub const NAME_TABLE_MISSING: TtError = TtError(0x91);
    /// `CMap_Table_Missing`
    pub const CMAP_TABLE_MISSING: TtError = TtError(0x92);
    /// `Hmtx_Table_Missing`
    pub const HMTX_TABLE_MISSING: TtError = TtError(0x93);
    /// `Post_Table_Missing`
    pub const POST_TABLE_MISSING: TtError = TtError(0x94);
    /// `Invalid_Horiz_Metrics`
    pub const INVALID_HORIZ_METRICS: TtError = TtError(0x95);
    /// `Invalid_CharMap_Format`
    pub const INVALID_CHARMAP_FORMAT: TtError = TtError(0x96);
    /// `Invalid_PPem`
    pub const INVALID_PPEM: TtError = TtError(0x97);
    /// `Invalid_Vert_Metrics`
    pub const INVALID_VERT_METRICS: TtError = TtError(0x98);
    /// `Could_Not_Find_Context`
    pub const COULD_NOT_FIND_CONTEXT: TtError = TtError(0x99);
    /// `Invalid_Post_Table_Format`
    pub const INVALID_POST_TABLE_FORMAT: TtError = TtError(0x9A);
    /// `Invalid_Post_Table`
    pub const INVALID_POST_TABLE: TtError = TtError(0x9B);
    /// `Syntax_Error`
    pub const SYNTAX_ERROR: TtError = TtError(0xA0);
    /// `Stack_Underflow`
    pub const STACK_UNDERFLOW: TtError = TtError(0xA1);
    /// `Ignore`
    pub const IGNORE: TtError = TtError(0xA2);
    /// `No_Unicode_Glyph_Name`
    pub const NO_UNICODE_GLYPH_NAME: TtError = TtError(0xA3);
    /// `Glyph_Too_Big`
    pub const GLYPH_TOO_BIG: TtError = TtError(0xA4);
    /// `Missing_Startfont_Field`
    pub const MISSING_STARTFONT_FIELD: TtError = TtError(0xB0);
    /// `Missing_Font_Field`
    pub const MISSING_FONT_FIELD: TtError = TtError(0xB1);
    /// `Missing_Size_Field`
    pub const MISSING_SIZE_FIELD: TtError = TtError(0xB2);
    /// `Missing_Fontboundingbox_Field`
    pub const MISSING_FONTBOUNDINGBOX_FIELD: TtError = TtError(0xB3);
    /// `Missing_Chars_Field`
    pub const MISSING_CHARS_FIELD: TtError = TtError(0xB4);
    /// `Missing_Startchar_Field`
    pub const MISSING_STARTCHAR_FIELD: TtError = TtError(0xB5);
    /// `Missing_Encoding_Field`
    pub const MISSING_ENCODING_FIELD: TtError = TtError(0xB6);
    /// `Missing_Bbx_Field`
    pub const MISSING_BBX_FIELD: TtError = TtError(0xB7);
    /// `Bbx_Too_Big`
    pub const BBX_TOO_BIG: TtError = TtError(0xB8);
    /// `Corrupted_Font_Header`
    pub const CORRUPTED_FONT_HEADER: TtError = TtError(0xB9);
    /// `Corrupted_Font_Glyphs`
    pub const CORRUPTED_FONT_GLYPHS: TtError = TtError(0xBA);

    /// The raw error code as stored by FreeType.
    #[inline]
    pub const fn code(self) -> i32 {
        self.0
    }

    /// `ERROR_BASE(x)`: the low byte of the error code.
    #[inline]
    pub const fn base(self) -> i32 {
        self.0 & 0xFF
    }

    /// `ERROR_MODULE(x)`: the module bits (bits 8..15) of the error code.
    #[inline]
    pub const fn module(self) -> i32 {
        self.0 & 0xFF00
    }

    /// `true` when this is the success code (`Err_Ok`).
    #[inline]
    pub const fn is_ok(self) -> bool {
        self.0 == 0
    }

    /// The FreeType error description string (`fterrdef.h`).
    pub fn message(self) -> &'static str {
        match self.base() {
            0x00 => "no error",
            0x01 => "cannot open resource",
            0x02 => "unknown file format",
            0x03 => "broken file",
            0x04 => "invalid FreeType version",
            0x05 => "module version is too low",
            0x06 => "invalid argument",
            0x07 => "unimplemented feature",
            0x08 => "broken table",
            0x09 => "broken offset within table",
            0x0A => "array allocation size too large",
            0x0B => "missing module",
            0x0C => "missing property",
            0x10 => "invalid glyph index",
            0x11 => "invalid character code",
            0x12 => "unsupported glyph image format",
            0x13 => "cannot render this glyph format",
            0x14 => "invalid outline",
            0x15 => "invalid composite glyph",
            0x16 => "too many hints",
            0x17 => "invalid pixel size",
            0x20 => "invalid object handle",
            0x21 => "invalid library handle",
            0x22 => "invalid module handle",
            0x23 => "invalid face handle",
            0x24 => "invalid size handle",
            0x25 => "invalid glyph slot handle",
            0x26 => "invalid charmap handle",
            0x27 => "invalid cache handle",
            0x28 => "invalid stream handle",
            0x30 => "too many modules",
            0x31 => "too many extensions",
            0x40 => "out of memory",
            0x41 => "unlisted object",
            0x51 => "cannot open stream",
            0x52 => "invalid stream seek",
            0x53 => "invalid stream skip",
            0x54 => "invalid stream read",
            0x55 => "invalid stream operation",
            0x56 => "invalid frame operation",
            0x57 => "nested frame access",
            0x58 => "invalid frame read",
            0x60 => "raster uninitialized",
            0x61 => "raster corrupted",
            0x62 => "raster overflow",
            0x63 => "raster negative height",
            0x70 => "too many caches",
            0x80 => "invalid opcode",
            0x81 => "too few arguments",
            0x82 => "stack overflow",
            0x83 => "code overflow",
            0x84 => "bad argument",
            0x85 => "divide by zero",
            0x86 => "invalid reference",
            0x87 => "found debug opcode",
            0x88 => "found ENDF opcode in execution stream",
            0x89 => "nested DEFS",
            0x8A => "invalid code range",
            0x8B => "execution context too long",
            0x8C => "too many function definitions",
            0x8D => "too many instruction definitions",
            0x8E => "SFNT font table missing",
            0x8F => "horizontal header (hhea) table missing",
            0x90 => "locations (loca) table missing",
            0x91 => "name table missing",
            0x92 => "character map (cmap) table missing",
            0x93 => "horizontal metrics (hmtx) table missing",
            0x94 => "PostScript (post) table missing",
            0x95 => "invalid horizontal metrics",
            0x96 => "invalid character map (cmap) format",
            0x97 => "invalid ppem values",
            0x98 => "invalid vertical metrics",
            0x99 => "could not find context",
            0x9A => "invalid PostScript (post) table format",
            0x9B => "invalid PostScript (post) table",
            0xA0 => "syntax error",
            0xA1 => "stack underflow",
            0xA2 => "ignore",
            0xA3 => "missing Unicode glyph name",
            0xA4 => "glyph too big",
            0xB0 => "missing Startfont field",
            0xB1 => "missing Font field",
            0xB2 => "missing Size field",
            0xB3 => "missing FontBBox field",
            0xB4 => "missing Chars field",
            0xB5 => "missing Startchar field",
            0xB6 => "missing Encoding field",
            0xB7 => "missing BBox field",
            0xB8 => "BBX too big",
            0xB9 => "font header corrupted",
            0xBA => "font glyphs corrupted",
            _ => "unknown error",
        }
    }
}

impl fmt::Debug for TtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error({:#04X}: {})", self.0, self.message())
    }
}

impl fmt::Display for TtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (error {:#04X})", self.message(), self.0)
    }
}
impl core::error::Error for TtError {}

impl From<TtError> for codevar_io::Error {
    #[inline]
    fn from(err: TtError) -> Self {
        let kind = match err {
            TtError::OUT_OF_MEMORY => codevar_io::ErrorKind::OutOfMemory,
            TtError::INVALID_ARGUMENT => codevar_io::ErrorKind::InvalidInput,
            TtError::CANNOT_OPEN_RESOURCE | TtError::CANNOT_OPEN_STREAM => {
                codevar_io::ErrorKind::NotFound
            }
            TtError::INVALID_STREAM_SEEK | TtError::INVALID_STREAM_SKIP => {
                codevar_io::ErrorKind::InvalidInput
            }
            TtError::INVALID_STREAM_READ => codevar_io::ErrorKind::UnexpectedEof,
            _ => codevar_io::ErrorKind::Other,
        };
        codevar_io::Error::new(kind, err)
    }
}

/// `MAKE_TAG`: pack four bytes into a 32-bit tag (big-endian order).
#[inline]
pub const fn make_tag(x1: u8, x2: u8, x3: u8, x4: u8) -> Tag {
    ((x1 as u64) << 24 | (x2 as u64) << 16 | (x3 as u64) << 8 | x4 as u64)
        as Tag
}

/// A character map encoding identifier (usually a four-byte tag).
///
/// Modelled as a newtype instead of a Rust enum because drivers may report
/// platform encodings that are not part of FreeType's built-in list.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Encoding(pub u64);

impl Encoding {
    /// `ENCODING_NONE`
    pub const NONE: Encoding = Encoding(0);
    /// `ENCODING_MS_SYMBOL`
    pub const MS_SYMBOL: Encoding = Encoding(make_tag(b's', b'y', b'm', b'b') as u64);
    /// `ENCODING_UNICODE`
    pub const UNICODE: Encoding = Encoding(make_tag(b'u', b'n', b'i', b'c') as u64);
    /// `ENCODING_SJIS`
    pub const SJIS: Encoding = Encoding(make_tag(b's', b'j', b'i', b's') as u64);
    /// `ENCODING_GB2312`
    pub const GB2312: Encoding = Encoding(make_tag(b'g', b'b', b' ', b' ') as u64);
    /// `ENCODING_BIG5`
    pub const BIG5: Encoding = Encoding(make_tag(b'b', b'i', b'g', b'5') as u64);
    /// `ENCODING_WANSUNG`
    pub const WANSUNG: Encoding = Encoding(make_tag(b'w', b'a', b'n', b's') as u64);
    /// `ENCODING_JOHAB`
    pub const JOHAB: Encoding = Encoding(make_tag(b'j', b'o', b'h', b'a') as u64);
    /// `ENCODING_ADOBE_STANDARD`
    pub const ADOBE_STANDARD: Encoding = Encoding(make_tag(b'A', b'D', b'O', b'B') as u64);
    /// `ENCODING_ADOBE_EXPERT`
    pub const ADOBE_EXPERT: Encoding = Encoding(make_tag(b'A', b'D', b'B', b'E') as u64);
    /// `ENCODING_ADOBE_CUSTOM`
    pub const ADOBE_CUSTOM: Encoding = Encoding(make_tag(b'A', b'D', b'B', b'C') as u64);
    /// `ENCODING_ADOBE_LATIN_1`
    pub const ADOBE_LATIN_1: Encoding = Encoding(make_tag(b'l', b'a', b't', b'1') as u64);
    /// `ENCODING_OLD_LATIN_2`
    pub const OLD_LATIN_2: Encoding = Encoding(make_tag(b'l', b'a', b't', b'2') as u64);
    /// `ENCODING_APPLE_ROMAN`
    pub const APPLE_ROMAN: Encoding = Encoding(make_tag(b'a', b'r', b'm', b'n') as u64);

    /// The raw tag value.
    #[inline]
    pub const fn to_tag(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.0.to_be_bytes();
        if b[0].is_ascii_graphic() && b[1].is_ascii_graphic() && b[2].is_ascii_graphic()
            && b[3].is_ascii_graphic()
        {
            write!(
                f,
                "Encoding({}{}{}{})",
                b[0] as char, b[1] as char, b[2] as char, b[3] as char
            )
        } else {
            write!(f, "Encoding({:#010X})", self.0)
        }
    }
}

/// The format of a glyph image (a four-byte tag).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u32)]
pub enum GlyphFormat {
    /// `GLYPH_FORMAT_NONE`: the glyph image is invalid or not yet set.
    None = 0,
    /// `GLYPH_FORMAT_COMPOSITE`: a composite glyph image.
    Composite = 0x636F_6D70, // 'comp'
    /// `GLYPH_FORMAT_BITMAP`: a bitmap image.
    Bitmap = 0x6269_7473, // 'bits'
    /// `GLYPH_FORMAT_OUTLINE`: a vectorial outline.
    Outline = 0x6F75_746C, // 'outl'
    /// `GLYPH_FORMAT_PLOTTER`: a vectorial path without fill semantics.
    Plotter = 0x706C_6F74, // 'plot'
}

impl GlyphFormat {
    /// The raw four-byte tag value (`GLYPH_TAG` order).
    #[inline]
    pub const fn tag(self) -> u64 {
        self as u64
    }

    /// Reinterpret a raw tag; unknown tags map to [`GlyphFormat::None`].
    #[inline]
    pub const fn from_tag(tag: u64) -> Self {
        match tag {
            0x636F_6D70 => GlyphFormat::Composite,
            0x6269_7473 => GlyphFormat::Bitmap,
            0x6F75_746C => GlyphFormat::Outline,
            0x706C_6F74 => GlyphFormat::Plotter,
            _ => GlyphFormat::None,
        }
    }

    /// `GLYPH_FORMAT_IS_DEFAULT`: `true` for [`GlyphFormat::None`].
    #[inline]
    pub const fn is_default(self) -> bool {
        self as u32 == 0
    }
}

/// The pixel format of a bitmap (ftimage.h `PixelMode`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(i32)]
pub enum PixelMode {
    /// `PIXEL_MODE_NONE`: no pixel mode.
    #[default]
    None = 0,
    /// `PIXEL_MODE_MONO`: 1 bit per pixel.
    Mono = 1,
    /// `PIXEL_MODE_GRAY`: 8 bits per pixel.
    Gray = 2,
    /// `PIXEL_MODE_GRAY2`: 2 bits per pixel.
    Gray2 = 3,
    /// `PIXEL_MODE_GRAY4`: 4 bits per pixel.
    Gray4 = 4,
    /// `PIXEL_MODE_LCD`: 8-bit horizontal RGB subpixel.
    Lcd = 5,
    /// `PIXEL_MODE_LCD_V`: 8-bit vertical RGB subpixel.
    LcdV = 6,
    /// `PIXEL_MODE_BGRA`: 32-bit premultiplied BGRA.
    Bgra = 7,
}

impl PixelMode {
    /// `PIXEL_MODE_MAX`: one past the last valid mode.
    pub const MAX: i32 = 8;

    /// The number of bits per pixel occupied by this mode (0 for `NONE`).
    #[inline]
    pub const fn bits_per_pixel(self) -> u32 {
        match self {
            PixelMode::None => 0,
            PixelMode::Mono => 1,
            PixelMode::Gray2 => 2,
            PixelMode::Gray4 => 4,
            PixelMode::Gray | PixelMode::Lcd | PixelMode::LcdV => 8,
            PixelMode::Bgra => 32,
        }
    }
}

/// How a glyph is rendered (freetype.h `RenderMode`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(i32)]
pub enum RenderMode {
    /// `RENDER_MODE_NORMAL`: 8-bit anti-aliased rendering.
    #[default]
    Normal = 0,
    /// `RENDER_MODE_LIGHT`: same as `NORMAL`, selects light hinting.
    Light = 1,
    /// `RENDER_MODE_MONO`: 1-bit monochrome rendering.
    Mono = 2,
    /// `RENDER_MODE_LCD`: horizontal subpixel rendering (3x width).
    Lcd = 3,
    /// `RENDER_MODE_LCD_V`: vertical subpixel rendering (3x height).
    LcdV = 4,
}

impl RenderMode {
    /// `RENDER_MODE_MAX`: one past the last valid mode.
    pub const MAX: i32 = 5;
}

/// The kerning mode (freetype.h `KerningMode`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(i32)]
pub enum KerningMode {
    /// `KERNING_DEFAULT`: grid-fitted, scaled kerning distances.
    #[default]
    Default = 0,
    /// `KERNING_UNFITTED`: scaled but not grid-fitted distances.
    Unfitted = 1,
    /// `KERNING_UNSCALED`: distances in original font units.
    Unscaled = 2,
}

/// The type of size request (freetype.h `SizeRequestType`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(i32)]
pub enum SizeRequestType {
    /// `SIZE_REQUEST_TYPE_NOMINAL`: the nominal size.
    #[default]
    Nominal = 0,
    /// `SIZE_REQUEST_TYPE_REAL_DIM`: the real dimension.
    RealDim = 1,
    /// `SIZE_REQUEST_TYPE_BBOX`: the scaling to fit the font bounding box.
    BBox = 2,
    /// `SIZE_REQUEST_TYPE_CELL`: the scaling to fit a single cell.
    Cell = 3,
    /// `SIZE_REQUEST_TYPE_SCALES`: scaling values are directly 16.16.
    Scales = 4,
}

/// The LCD filter applied to subpixel-rendered glyphs (ftlcdfil.h).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(i32)]
pub enum LcdFilter {
    /// `LCD_FILTER_NONE`: no filtering.
    #[default]
    None = 0,
    /// `LCD_FILTER_DEFAULT`: the default 5-tap FIR filter.
    Default = 1,
    /// `LCD_FILTER_LIGHT`: a lighter 5-tap FIR filter.
    Light = 2,
    /// `LCD_FILTER_LEGACY1`: legacy intra-pixel filter (since 2.6.2).
    Legacy1 = 3,
    /// `LCD_FILTER_LEGACY`: the legacy intra-pixel filter.
    Legacy = 16,
}

/// The fill orientation of an outline (ftoutln.h `Orientation`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(i32)]
pub enum Orientation {
    /// `ORIENTATION_TRUETYPE`: clockwise contours are filled.
    #[default]
    TrueType = 0,
    /// `ORIENTATION_POSTSCRIPT`: counter-clockwise contours are filled.
    Postscript = 1,
    /// `ORIENTATION_FILL_RIGHT`: alias of [`Orientation::TrueType`].
    FillRight = 2,
    /// `ORIENTATION_FILL_LEFT`: alias of [`Orientation::Postscript`].
    FillLeft = 3,
    /// `ORIENTATION_NONE`: orientation cannot be determined.
    None = 4,
}

/// Flags passed to [`Load_Glyph`](LoadFlags) and friends
/// (`LOAD_XXX`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct LoadFlags(pub i32);

impl LoadFlags {
    /// `LOAD_DEFAULT`: default loading behaviour.
    pub const DEFAULT: Self = Self(0);
    /// `LOAD_NO_SCALE`: return unhinted metrics in font units.
    pub const NO_SCALE: Self = Self(1 << 0);
    /// `LOAD_NO_HINTING`: ignore hinting, load the unhinted outline.
    pub const NO_HINTING: Self = Self(1 << 1);
    /// `LOAD_RENDER`: render the glyph after loading (if scalable).
    pub const RENDER: Self = Self(1 << 2);
    /// `LOAD_NO_BITMAP`: ignore embedded bitmaps, load outline only.
    pub const NO_BITMAP: Self = Self(1 << 3);
    /// `LOAD_VERTICAL_LAYOUT`: set up vertical metrics/layout.
    pub const VERTICAL_LAYOUT: Self = Self(1 << 4);
    /// `LOAD_FORCE_AUTOHINT`: force use of the auto-hinter.
    pub const FORCE_AUTOHINT: Self = Self(1 << 5);
    /// `LOAD_CROP_BITMAP`: crop embedded bitmaps to the gray shape.
    pub const CROP_BITMAP: Self = Self(1 << 6);
    /// `LOAD_PEDANTIC`: perform pedantic glyph loading.
    pub const PEDANTIC: Self = Self(1 << 7);
    /// `LOAD_ADVANCE_ONLY`: only extract horizontal advance widths.
    pub const ADVANCE_ONLY: Self = Self(1 << 8);
    /// `LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH`: ignore `hmtx` advance widths.
    pub const IGNORE_GLOBAL_ADVANCE_WIDTH: Self = Self(1 << 9);
    /// `LOAD_NO_RECURSE`: do not load composite glyphs recursively.
    pub const NO_RECURSE: Self = Self(1 << 10);
    /// `LOAD_IGNORE_TRANSFORM`: ignore the matrix set by
    /// `Set_Transform`.
    pub const IGNORE_TRANSFORM: Self = Self(1 << 11);
    /// `LOAD_MONOCHROME`: render a monochrome bitmap (1 bit per pixel).
    pub const MONOCHROME: Self = Self(1 << 12);
    /// `LOAD_LINEAR_DESIGN`: get linear advance widths in design units.
    pub const LINEAR_DESIGN: Self = Self(1 << 13);
    /// `LOAD_SBITS_ONLY`: only extract embedded bitmaps.
    pub const SBITS_ONLY: Self = Self(1 << 14);
    /// `LOAD_NO_AUTOHINT`: never use the auto-hinter.
    pub const NO_AUTOHINT: Self = Self(1 << 15);
    /// `LOAD_COLOR`: load color glyph (BGRA) images.
    pub const COLOR: Self = Self(1 << 20);
    /// `LOAD_COMPUTE_METRICS`: compute metrics from the outline.
    pub const COMPUTE_METRICS: Self = Self(1 << 21);
    /// `LOAD_ADVANCE_FAST_ONLY`: only get fast advances (since 2.6.1).
    pub const ADVANCE_FAST_ONLY: Self = Self(0x2000_0000);

    /// `LOAD_TARGET_(x)`: pack a render mode into the target bits.
    pub const fn target(mode: RenderMode) -> Self {
        Self(((mode as i32) & 15) << 16)
    }

    /// `LOAD_TARGET_NORMAL`
    pub const TARGET_NORMAL: Self = Self::target(RenderMode::Normal);
    /// `LOAD_TARGET_LIGHT`
    pub const TARGET_LIGHT: Self = Self::target(RenderMode::Light);
    /// `LOAD_TARGET_MONO`
    pub const TARGET_MONO: Self = Self::target(RenderMode::Mono);
    /// `LOAD_TARGET_LCD`
    pub const TARGET_LCD: Self = Self::target(RenderMode::Lcd);
    /// `LOAD_TARGET_LCD_V`
    pub const TARGET_LCD_V: Self = Self::target(RenderMode::LcdV);

    /// `LOAD_TARGET_MODE(x)`: extract the render mode from the flags.
    pub const fn target_mode(self) -> RenderMode {
        match (self.0 >> 16) & 15 {
            1 => RenderMode::Light,
            2 => RenderMode::Mono,
            3 => RenderMode::Lcd,
            4 => RenderMode::LcdV,
            _ => RenderMode::Normal,
        }
    }

    /// The raw flag bits.
    #[inline]
    pub const fn bits(self) -> i32 {
        self.0
    }

    /// Build a flag set from raw bits (unknown bits are preserved).
    #[inline]
    pub const fn from_bits(bits: i32) -> Self {
        Self(bits)
    }

    /// `true` when every bit of `other` is set in `self`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// `true` when no bit is set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Set the bits of `other` on `self`.
    #[inline]
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Clear the bits of `other` from `self`.
    #[inline]
    pub fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }
}

impl core::ops::BitOr for LoadFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for LoadFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl core::ops::BitAnd for LoadFlags {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl core::ops::BitAndAssign for LoadFlags {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl core::ops::Not for LoadFlags {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

impl fmt::Debug for LoadFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LoadFlags({:#010X})", self.0)
    }
}

/// Flags for [`Open_Args::flags`] (`OPEN_XXX`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct OpenFlags(pub u32);

impl OpenFlags {
    /// `OPEN_MEMORY`: a memory-based stream.
    pub const MEMORY: Self = Self(0x1);
    /// `OPEN_STREAM`: use the user-provided stream.
    pub const STREAM: Self = Self(0x2);
    /// `OPEN_PATHNAME`: create a stream from a file path.
    pub const PATHNAME: Self = Self(0x4);
    /// `OPEN_DRIVER`: use the given driver only.
    pub const DRIVER: Self = Self(0x8);
    /// `OPEN_PARAMS`: extra open parameters are present.
    pub const PARAMS: Self = Self(0x10);

    /// The raw flag bits.
    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// `true` when every bit of `other` is set in `self`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl core::ops::BitOr for OpenFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for OpenFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl fmt::Debug for OpenFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenFlags({:#04X})", self.0)
    }
}

/// `FACE_FLAG_XXX`: properties of an [`FaceRec`].
pub const FACE_FLAG_SCALABLE: i64 = 1 << 0;
/// `FACE_FLAG_FIXED_SIZES`: the face provides bitmap strikes.
pub const FACE_FLAG_FIXED_SIZES: i64 = 1 << 1;
/// `FACE_FLAG_FIXED_WIDTH`: all glyphs have the same width.
pub const FACE_FLAG_FIXED_WIDTH: i64 = 1 << 2;
/// `FACE_FLAG_SFNT`: the face uses the SFNT storage scheme.
pub const FACE_FLAG_SFNT: i64 = 1 << 3;
/// `FACE_FLAG_HORIZONTAL`: the face has horizontal metrics.
pub const FACE_FLAG_HORIZONTAL: i64 = 1 << 4;
/// `FACE_FLAG_VERTICAL`: the face has vertical metrics.
pub const FACE_FLAG_VERTICAL: i64 = 1 << 5;
/// `FACE_FLAG_KERNING`: the face provides kerning data.
pub const FACE_FLAG_KERNING: i64 = 1 << 6;
/// `FACE_FLAG_FAST_GLYPHS`: the face contains only fast and scalable
/// glyphs.
pub const FACE_FLAG_FAST_GLYPHS: i64 = 1 << 7;
/// `FACE_FLAG_MULTIPLE_MASTERS`: the face contains multiple masters.
pub const FACE_FLAG_MULTIPLE_MASTERS: i64 = 1 << 8;
/// `FACE_FLAG_GLYPH_NAMES`: the face provides glyph name strings.
pub const FACE_FLAG_GLYPH_NAMES: i64 = 1 << 9;
/// `FACE_FLAG_EXTERNAL_STREAM`: the face's stream is externally owned.
pub const FACE_FLAG_EXTERNAL_STREAM: i64 = 1 << 10;
/// `FACE_FLAG_HINTER`: the face's driver has its own hinter.
pub const FACE_FLAG_HINTER: i64 = 1 << 11;
/// `FACE_FLAG_CID_KEYED`: the face is CID-keyed.
pub const FACE_FLAG_CID_KEYED: i64 = 1 << 12;
/// `FACE_FLAG_TRICKY`: the face requires the `tricky` hinting engine.
pub const FACE_FLAG_TRICKY: i64 = 1 << 13;
/// `FACE_FLAG_COLOR`: the face contains color glyph layers.
pub const FACE_FLAG_COLOR: i64 = 1 << 14;

/// `STYLE_FLAG_ITALIC`: the face's style is italic.
pub const STYLE_FLAG_ITALIC: i64 = 1 << 0;
/// `STYLE_FLAG_BOLD`: the face's style is bold.
pub const STYLE_FLAG_BOLD: i64 = 1 << 1;

/// `OUTLINE_NONE`: no outline flags set.
pub const OUTLINE_NONE: i32 = 0x0;
/// `OUTLINE_OWNER`: the outline owner allocated the point arrays.
pub const OUTLINE_OWNER: i32 = 0x1;
/// `OUTLINE_EVEN_ODD_FILL`: use the even-odd filling rule.
pub const OUTLINE_EVEN_ODD_FILL: i32 = 0x2;
/// `OUTLINE_REVERSE_FILL`: contours must be reversed before filling.
pub const OUTLINE_REVERSE_FILL: i32 = 0x4;
/// `OUTLINE_IGNORE_DROPOUTS`: the rasterizer ignores drop-outs.
pub const OUTLINE_IGNORE_DROPOUTS: i32 = 0x8;
/// `OUTLINE_SMART_DROPOUTS`: enable smart drop-out control.
pub const OUTLINE_SMART_DROPOUTS: i32 = 0x10;
/// `OUTLINE_INCLUDE_STUBS`: include stubs in smart drop-out control.
pub const OUTLINE_INCLUDE_STUBS: i32 = 0x20;
/// `OUTLINE_HIGH_PRECISION`: force high precision during rendering.
pub const OUTLINE_HIGH_PRECISION: i32 = 0x100;
/// `OUTLINE_SINGLE_PASS`: force a single rendering pass.
pub const OUTLINE_SINGLE_PASS: i32 = 0x200;

/// `OUTLINE_CONTOURS_MAX`: the maximum number of contours
/// (`SHRT_MAX`).
pub const OUTLINE_CONTOURS_MAX: i16 = i16::MAX;
/// `OUTLINE_POINTS_MAX`: the maximum number of points (`SHRT_MAX`).
pub const OUTLINE_POINTS_MAX: i16 = i16::MAX;

/// `curve_tag(flag)`: extract the curve type bits of a point tag.
#[inline]
pub const fn curve_tag(flag: u8) -> u8 {
    flag & 3
}

/// `CURVE_TAG_ON`: the point is on the curve.
pub const CURVE_TAG_ON: u8 = 1;
/// `CURVE_TAG_CONIC`: the point is an off-curve conic control point.
pub const CURVE_TAG_CONIC: u8 = 0;
/// `CURVE_TAG_CUBIC`: the point is an off-curve cubic control point.
pub const CURVE_TAG_CUBIC: u8 = 2;
/// `CURVE_TAG_HAS_SCANMODE`: drop-out mode stored in the tag.
pub const CURVE_TAG_HAS_SCANMODE: u8 = 4;
/// `CURVE_TAG_TOUCH_X`: reserved touch flag for the TrueType hinter.
pub const CURVE_TAG_TOUCH_X: u8 = 8;
/// `CURVE_TAG_TOUCH_Y`: reserved touch flag for the TrueType hinter.
pub const CURVE_TAG_TOUCH_Y: u8 = 16;
/// `CURVE_TAG_TOUCH_BOTH`: both touch flags set.
pub const CURVE_TAG_TOUCH_BOTH: u8 = CURVE_TAG_TOUCH_X | CURVE_TAG_TOUCH_Y;

/// `MODULE_FONT_DRIVER`: the module is a font driver.
pub const MODULE_FONT_DRIVER: u32 = 1;
/// `MODULE_RENDERER`: the module is a renderer.
pub const MODULE_RENDERER: u32 = 2;
/// `MODULE_HINTER`: the module is a glyph hinter.
pub const MODULE_HINTER: u32 = 4;
/// `MODULE_STYLER`: the module is a styler.
pub const MODULE_STYLER: u32 = 8;
/// `MODULE_DRIVER_SCALABLE`: the driver supports scalable outlines.
pub const MODULE_DRIVER_SCALABLE: u32 = 0x100;
/// `MODULE_DRIVER_NO_OUTLINES`: the driver does not produce outlines.
pub const MODULE_DRIVER_NO_OUTLINES: u32 = 0x200;
/// `MODULE_DRIVER_HAS_HINTER`: the driver provides its own hinter.
pub const MODULE_DRIVER_HAS_HINTER: u32 = 0x400;
/// `MODULE_DRIVER_HINTS_LIGHTLY`: the driver's hinter is "light".
pub const MODULE_DRIVER_HINTS_LIGHTLY: u32 = 0x800;

/// `MAX_MODULES`: the maximum number of modules per library.
pub const MAX_MODULES: usize = 32;

/// `DEBUG_HOOK_TRUETYPE`: index of the TrueType debugger hook.
pub const DEBUG_HOOK_TRUETYPE: usize = 0;

/// `GLYPH_OWN_BITMAP`: set when a slot owns its bitmap buffer.
pub const GLYPH_OWN_BITMAP: u32 = 0x1;

/// `SUBGLYPH_FLAG_ARGS_ARE_WORDS`: subglyph arguments are words.
pub const SUBGLYPH_FLAG_ARGS_ARE_WORDS: u16 = 1;
/// `SUBGLYPH_FLAG_ARGS_ARE_XY_VALUES`: arguments are x/y offsets.
pub const SUBGLYPH_FLAG_ARGS_ARE_XY_VALUES: u16 = 2;
/// `SUBGLYPH_FLAG_ROUND_XY_TO_GRID`: round offsets to the grid.
pub const SUBGLYPH_FLAG_ROUND_XY_TO_GRID: u16 = 4;
/// `SUBGLYPH_FLAG_SCALE`: subglyph has a 2x2 scale.
pub const SUBGLYPH_FLAG_SCALE: u16 = 8;
/// `SUBGLYPH_FLAG_XY_SCALE`: subglyph has separate x/y scales.
pub const SUBGLYPH_FLAG_XY_SCALE: u16 = 0x40;
/// `SUBGLYPH_FLAG_2X2`: subglyph has a full 2x2 transform.
pub const SUBGLYPH_FLAG_2X2: u16 = 0x80;
/// `SUBGLYPH_FLAG_USE_MY_METRICS`: use the subglyph's metrics.
pub const SUBGLYPH_FLAG_USE_MY_METRICS: u16 = 0x200;

/// `PARAM_TAG_INCREMENTAL`: tag for incremental loading parameters.
pub const PARAM_TAG_INCREMENTAL: u32 =
    make_tag(b'i', b'n', b'c', b'r') as u32;
/// `PARAM_TAG_IGNORE_PREFERRED_FAMILY`: ignore preferred family name.
pub const PARAM_TAG_IGNORE_PREFERRED_FAMILY: u32 =
    make_tag(b'i', b'g', b'p', b'f') as u32;
/// `PARAM_TAG_IGNORE_PREFERRED_SUBFAMILY`: ignore preferred subfamily.
pub const PARAM_TAG_IGNORE_PREFERRED_SUBFAMILY: u32 =
    make_tag(b'i', b'g', b'p', b's') as u32;
/// `PARAM_TAG_UNPATENTED_HINTING`: disable patented hinting (2.4.x).
pub const PARAM_TAG_UNPATENTED_HINTING: u32 =
    make_tag(b'u', b'n', b'p', b'a') as u32;
