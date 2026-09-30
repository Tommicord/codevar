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
#![deny(clippy::unwrap_used, clippy::expect_used)]
extern crate alloc;

use alloc::boxed::Box;
use alloc::format;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
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
            TtError::CANNOT_OPEN_RESOURCE | TtError::CANNOT_OPEN_STREAM => codevar_io::ErrorKind::NotFound,
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
    ((x1 as u64) << 24 | (x2 as u64) << 16 | (x3 as u64) << 8 | x4 as u64) as Tag
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
        if b[0].is_ascii_graphic()
            && b[1].is_ascii_graphic()
            && b[2].is_ascii_graphic()
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
pub const PARAM_TAG_INCREMENTAL: u32 = make_tag(b'i', b'n', b'c', b'r') as u32;
/// `PARAM_TAG_IGNORE_PREFERRED_FAMILY`: ignore preferred family name.
pub const PARAM_TAG_IGNORE_PREFERRED_FAMILY: u32 = make_tag(b'i', b'g', b'p', b'f') as u32;
/// `PARAM_TAG_IGNORE_PREFERRED_SUBFAMILY`: ignore preferred subfamily.
pub const PARAM_TAG_IGNORE_PREFERRED_SUBFAMILY: u32 = make_tag(b'i', b'g', b'p', b's') as u32;
/// `PARAM_TAG_UNPATENTED_HINTING`: disable patented hinting (2.4.x).
pub const PARAM_TAG_UNPATENTED_HINTING: u32 = make_tag(b'u', b'n', b'p', b'a') as u32;

/// `FT_UnitVector`: a 2D unit vector in 2.14 fixed-point format.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct UnitVector {
    /// Horizontal component (`F2Dot14`).
    pub x: F2Dot14,
    /// Vertical component (`F2Dot14`).
    pub y: F2Dot14,
}

/// `FT_Vector`: a 2D position or vector, in `Pos` units.
///
/// Depending on context the coordinates are expressed in font units, 26.6
/// fractional pixels, or 16.16 fixed-point values.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Vector {
    /// The horizontal coordinate.
    pub x: Pos,
    /// The vertical coordinate.
    pub y: Pos,
}

impl Vector {
    /// The zero vector.
    pub const ZERO: Vector = Vector { x: 0, y: 0 };

    /// Creates a new vector from its coordinates.
    #[inline]
    pub const fn new(x: Pos, y: Pos) -> Self {
        Vector { x, y }
    }
}

impl core::ops::Add for Vector {
    type Output = Vector;
    #[inline]
    fn add(self, rhs: Vector) -> Vector {
        Vector {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl core::ops::Sub for Vector {
    type Output = Vector;
    #[inline]
    fn sub(self, rhs: Vector) -> Vector {
        Vector {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

/// `FT_Matrix`: a 2x2 matrix of 16.16 fixed-point coefficients.
///
/// The transform it performs is `{ x' = x*xx + y*xy, y' = x*yx + y*yy }`.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Matrix {
    /// Matrix coefficient.
    pub xx: Fixed,
    /// Matrix coefficient.
    pub xy: Fixed,
    /// Matrix coefficient.
    pub yx: Fixed,
    /// Matrix coefficient.
    pub yy: Fixed,
}

impl Matrix {
    /// The identity matrix.
    pub const IDENTITY: Matrix = Matrix {
        xx: 0x10000,
        xy: 0,
        yx: 0,
        yy: 0x10000,
    };

    /// The all-zero matrix.
    pub const ZERO: Matrix = Matrix {
        xx: 0,
        xy: 0,
        yx: 0,
        yy: 0,
    };

    /// `true` when the matrix is the identity transform.
    #[inline]
    pub const fn is_identity(self) -> bool {
        self.xx == 0x10000 && self.xy == 0 && self.yx == 0 && self.yy == 0x10000
    }

    /// `true` when the matrix is all zeros.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.xx == 0 && self.xy == 0 && self.yx == 0 && self.yy == 0
    }
}

/// `FT_BBox`: an axis-aligned bounding box.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct BBox {
    /// The horizontal minimum (left-most).
    pub x_min: Pos,
    /// The vertical minimum (bottom-most).
    pub y_min: Pos,
    /// The horizontal maximum (right-most).
    pub x_max: Pos,
    /// The vertical maximum (top-most).
    pub y_max: Pos,
}

impl BBox {
    /// An empty box (`0, 0, 0, 0`), matching `FT_BBox_Init`.
    #[inline]
    pub const fn new() -> Self {
        BBox {
            x_min: 0,
            y_min: 0,
            x_max: 0,
            y_max: 0,
        }
    }

    /// Creates a box from its four edges.
    #[inline]
    pub const fn from_edges(x_min: Pos, y_min: Pos, x_max: Pos, y_max: Pos) -> Self {
        BBox {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    /// Expands the box to include `point`.
    #[inline]
    pub fn include(&mut self, point: Vector) {
        if point.x < self.x_min {
            self.x_min = point.x;
        }
        if point.y < self.y_min {
            self.y_min = point.y;
        }
        if point.x > self.x_max {
            self.x_max = point.x;
        }
        if point.y > self.y_max {
            self.y_max = point.y;
        }
    }

    /// The width of the box (`x_max - x_min`).
    #[inline]
    pub const fn width(self) -> Pos {
        self.x_max - self.x_min
    }

    /// The height of the box (`y_max - y_min`).
    #[inline]
    pub const fn height(self) -> Pos {
        self.y_max - self.y_min
    }
}

/// `FT_Generic`: client data plus an optional finalizer.
///
/// The Rust port stores the client data as a boxed `Any` value; the
/// finalizer runs when the data is dropped (see `Drop` semantics of
/// `Option<Box<dyn Any>>`, the finalizer is invoked by
/// [`Generic::finish`]).
#[derive(Default)]
pub struct Generic {
    /// Client-specific data.
    pub data: Option<Box<dyn core::any::Any + Send + Sync>>,
    /// Optional finalizer invoked with the client data.
    pub finalizer: Option<fn(Box<dyn core::any::Any + Send + Sync>)>,
}

impl Generic {
    /// Invokes the finalizer (if any) with the stored data and clears the
    /// generic slot.
    pub fn finish(&mut self) {
        let data = self.data.take();
        if let (Some(finalizer), Some(data)) = (self.finalizer.take(), data) {
            finalizer(data);
        }
    }
}

/// `FT_Glyph_Metrics`: the metrics of a single glyph.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct GlyphMetrics {
    /// The glyph's width in font units (or 26.6 pixels).
    pub width: Pos,
    /// The glyph's height in font units (or 26.6 pixels).
    pub height: Pos,
    /// Left side bearing for horizontal layout.
    pub hori_bearing_x: Pos,
    /// Top side bearing for horizontal layout.
    pub hori_bearing_y: Pos,
    /// Advance width for horizontal layout.
    pub hori_advance: Pos,
    /// Left side bearing for vertical layout.
    pub vert_bearing_x: Pos,
    /// Top side bearing for vertical layout.
    pub vert_bearing_y: Pos,
    /// Advance height for vertical layout.
    pub vert_advance: Pos,
}

/// `FT_Bitmap_Size`: the metrics of a bitmap strike.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct BitmapSize {
    /// The logical height of the strike in pixels.
    pub height: i16,
    /// The logical width of the strike in pixels.
    pub width: i16,
    /// The strike's nominal size in 26.6 points.
    pub size: Pos,
    /// The horizontal ppem of the strike, in 26.6 fractional pixels.
    pub x_ppem: Pos,
    /// The vertical ppem of the strike, in 26.6 fractional pixels.
    pub y_ppem: Pos,
}

/// `FT_Bitmap`: a descriptor for a run of packed pixels.
///
/// # Memory layout
///
/// `buffer` always owns exactly `rows * |pitch|` bytes stored in *top-down
/// memory order* (row 0 first). The sign of `pitch` only describes the
/// logical flow of the bitmap: a positive pitch means "down flow" (row `y`
/// at offset `y * pitch`), a negative pitch means "up flow" where row `y`
/// is located at `(rows - 1 - y) * |pitch|`. Always use [`Bitmap::row`]
/// and [`Bitmap::row_mut`] to access scanlines instead of computing raw
/// offsets by hand; this keeps negative-pitch bitmaps safe and correct.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bitmap {
    /// The number of bitmap rows.
    pub rows: u32,
    /// The number of pixels in each row.
    pub width: u32,
    /// The pitch (number of bytes per row, signed flow; absolute value is
    /// the stride in bytes).
    pub pitch: i32,
    /// The pixel buffer; see the type documentation for layout rules.
    pub buffer: Vec<u8>,
    /// The number of gray levels (2 for mono, 256 for gray, etc.).
    pub num_grays: u16,
    /// The pixel format of the buffer.
    pub pixel_mode: PixelMode,
    /// Palette format (unused, kept for API fidelity).
    pub palette_mode: u8,
    /// Optional palette (unused by the renderers, kept for API fidelity).
    pub palette: Option<Vec<u8>>,
}

impl Bitmap {
    /// Creates an empty bitmap.
    #[inline]
    pub const fn new() -> Self {
        Bitmap {
            rows: 0,
            width: 0,
            pitch: 0,
            buffer: Vec::new(),
            num_grays: 0,
            pixel_mode: PixelMode::None,
            palette_mode: 0,
            palette: None,
        }
    }

    /// The absolute row stride in bytes.
    #[inline]
    pub const fn stride(&self) -> usize {
        if self.pitch < 0 {
            (self.pitch.unsigned_abs()) as usize
        } else {
            self.pitch as usize
        }
    }

    /// The number of bytes per row implied by `pixel_mode` and `width`.
    #[inline]
    pub const fn min_pitch(width: u32, pixel_mode: PixelMode) -> i32 {
        match pixel_mode {
            PixelMode::Mono => (width as i32 + 7) / 8,
            PixelMode::Gray2 => (width as i32 + 3) / 4,
            PixelMode::Gray4 => (width as i32 + 1) / 2,
            PixelMode::Gray | PixelMode::Lcd => width as i32,
            PixelMode::LcdV => (width as i32 + 2) / 3,
            PixelMode::Bgra => width as i32 * 4,
            PixelMode::None => 0,
        }
    }

    /// Allocates a zeroed bitmap of `rows x width` for `pixel_mode`.
    ///
    /// Returns [`TtError::ARRAY_TOO_LARGE`] if the implied buffer size
    /// overflows.
    pub fn new_sized(rows: u32, width: u32, pixel_mode: PixelMode, num_grays: u16) -> TtResult<Self> {
        let pitch = Self::min_pitch(width, pixel_mode);
        let size = (rows as usize)
            .checked_mul(pitch.unsigned_abs() as usize)
            .ok_or(TtError::ARRAY_TOO_LARGE)?;
        Ok(Bitmap {
            rows,
            width,
            pitch,
            buffer: vec![0u8; size],
            num_grays,
            pixel_mode,
            palette_mode: 0,
            palette: None,
        })
    }

    /// Returns row `y`, or `None` when out of bounds or for an empty
    /// bitmap.
    #[inline]
    pub fn row(&self, y: u32) -> Option<&[u8]> {
        if y >= self.rows {
            return None;
        }
        let stride = self.stride();
        let start = if self.pitch < 0 {
            (self.rows - 1 - y) as usize * stride
        } else {
            y as usize * stride
        };
        self.buffer.get(start..start + stride)
    }

    /// Returns row `y` mutably, or `None` when out of bounds or for an
    /// empty bitmap.
    #[inline]
    pub fn row_mut(&mut self, y: u32) -> Option<&mut [u8]> {
        if y >= self.rows {
            return None;
        }
        let stride = self.stride();
        let start = if self.pitch < 0 {
            (self.rows - 1 - y) as usize * stride
        } else {
            y as usize * stride
        };
        self.buffer.get_mut(start..start + stride)
    }

    /// Resizes the bitmap to `rows x width`, zeroing the buffer.
    pub fn resize(&mut self, rows: u32, width: u32, pixel_mode: PixelMode, num_grays: u16) -> TtResult<()> {
        *self = Self::new_sized(rows, width, pixel_mode, num_grays)?;
        Ok(())
    }

    /// `true` when the buffer matches the geometry described by the
    /// bitmap's own fields.
    pub fn buffer_is_consistent(&self) -> bool {
        let expected = (self.rows as usize).saturating_mul(self.stride());
        self.buffer.len() == expected
    }
}

/// `FT_Outline`: a vectorial glyph description made of contours.
///
/// # Invariants
///
/// * `points.len() == tags.len() == n_points as usize`
/// * `contours.len() == n_contours as usize`
/// * contour end indices are strictly increasing and `< n_points`
///
/// All constructors and helpers of this port maintain these invariants;
/// `n_points`/`n_contours` always equal the slice lengths.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outline {
    /// The number of contours in the glyph.
    pub n_contours: i16,
    /// The number of points in the glyph.
    pub n_points: i16,
    /// The outline's points.
    pub points: Vec<Vector>,
    /// Per-point flags (`CURVE_TAG_XXX`).
    pub tags: Vec<u8>,
    /// The contour end point indices.
    pub contours: Vec<i16>,
    /// Outline masks (`OUTLINE_XXX`).
    pub flags: i32,
}

impl Outline {
    /// Creates an empty outline.
    #[inline]
    pub const fn new() -> Self {
        Outline {
            n_contours: 0,
            n_points: 0,
            points: Vec::new(),
            tags: Vec::new(),
            contours: Vec::new(),
            flags: 0,
        }
    }

    /// Creates an outline with reserved capacity.
    #[inline]
    pub fn with_capacity(points: usize, contours: usize) -> Self {
        Outline {
            n_contours: 0,
            n_points: 0,
            points: Vec::with_capacity(points),
            tags: Vec::with_capacity(points),
            contours: Vec::with_capacity(contours),
            flags: 0,
        }
    }

    /// Removes all points and contours, keeping the allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.points.clear();
        self.tags.clear();
        self.contours.clear();
        self.n_contours = 0;
        self.n_points = 0;
    }

    /// `FT_Outline_Check`: validates the outline invariants.
    ///
    /// Returns [`TtError::INVALID_OUTLINE`] when a contour end index is
    /// out of range, non-increasing, or when the point/contour counters
    /// disagree with the backing slices.
    pub fn check(&self) -> TtResult<()> {
        if self.points.len() != self.tags.len()
            || self.points.len() != self.n_points as usize
            || self.contours.len() != self.n_contours as usize
            || self.n_points < 0
            || self.n_contours < 0
        {
            return Err(TtError::INVALID_OUTLINE);
        }
        let mut last = -1i16;
        for &end in &self.contours {
            if end <= last || end >= self.n_points {
                return Err(TtError::INVALID_OUTLINE);
            }
            last = end;
        }
        if self.n_contours > 0 && last != self.n_points - 1 {
            return Err(TtError::INVALID_OUTLINE);
        }
        Ok(())
    }

    /// `FT_Outline_Get_CBox`: computes the control box of the outline.
    ///
    /// The control box is the bounding box of all outline points (control
    /// points included); it is returned unchanged and is `O(n)` in the
    /// number of points.
    pub fn get_cbox(&self) -> BBox {
        let mut box_ = BBox::new();
        let mut first = true;
        for &p in &self.points {
            if first {
                box_.x_min = p.x;
                box_.y_min = p.y;
                box_.x_max = p.x;
                box_.y_max = p.y;
                first = false;
            } else {
                box_.include(p);
            }
        }
        box_
    }

    /// Returns the point/flag slices of contour `i` as a range into
    /// `points`.
    ///
    /// Returns `None` when `i` is out of range.
    #[inline]
    pub fn contour_range(&self, i: usize) -> Option<(usize, usize)> {
        let end = *self.contours.get(i)? as usize;
        let start = if i == 0 {
            0
        } else {
            (*self.contours.get(i - 1)? as usize) + 1
        };
        Some((start, end + 1))
    }

    /// `true` when the outline has neither points nor contours.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_points == 0 || self.n_contours == 0
    }
}

/// `FT_Span`: a run of pixels on a single scanline, used by the direct
/// (span callback) rendering mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Span {
    /// The first pixel of the span (horizontal position).
    pub x: i16,
    /// The number of pixels in the span.
    pub len: u16,
    /// The coverage of the span (0..255).
    pub coverage: u8,
}

/// `FT_RASTER_FLAG_XXX`: flags passed to the rasterizer through
/// [`RasterParams`]-style configurations.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct RasterFlags(pub u32);

impl RasterFlags {
    /// `FT_RASTER_FLAG_DEFAULT`: monochrome, no direct rendering.
    pub const DEFAULT: Self = Self(0x0);
    /// `FT_RASTER_FLAG_AA`: anti-aliased rendering.
    pub const AA: Self = Self(0x1);
    /// `FT_RASTER_FLAG_DIRECT`: call the span callback instead of writing
    /// to the target bitmap.
    pub const DIRECT: Self = Self(0x2);
    /// `FT_RASTER_FLAG_CLIP`: clip spans to `clip_box`.
    pub const CLIP: Self = Self(0x4);

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

impl core::ops::BitOr for RasterFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for RasterFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl core::fmt::Debug for RasterFlags {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "RasterFlags({:#04X})", self.0)
    }
}

/// `FT_Memory`: the library memory manager.
///
/// FreeType delegates raw allocation to pluggable function pointers; the
/// Rust port instead routes all allocations through the global Rust
/// allocator and exposes this handle as a zero-sized context object so
/// call sites keep the shape of the original C code.
#[derive(Clone, Copy, Debug, Default)]
pub struct Memory;

impl Memory {
    /// Allocates `count` zeroed bytes.
    #[inline]
    pub fn alloc_zeroed(&self, count: usize) -> TtResult<Vec<u8>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if count > isize::MAX as usize {
            return Err(TtError::OUT_OF_MEMORY);
        }
        Ok(vec![0u8; count])
    }

    /// Allocates a `Vec` with `count` zero-initialized elements.
    #[inline]
    pub fn alloc_vec<T: Default + Clone>(&self, count: usize) -> TtResult<Vec<T>> {
        if count > isize::MAX as usize {
            return Err(TtError::OUT_OF_MEMORY);
        }
        let mut v = Vec::new();
        v.resize_with(count, T::default);
        Ok(v)
    }
}

/// A random-access byte source behind a [`Stream`].
///
/// File descriptors, user callbacks and compression layers implement this
/// trait; in-memory sources are handled directly by [`Stream`].
pub trait StreamSource: Send + Sync {
    /// The total size of the source, when known.
    fn size(&self) -> Option<u64>;

    /// Reads up to `buf.len()` bytes at absolute position `pos`.
    ///
    /// Returns the number of bytes read; `0` indicates end of input.
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> TtResult<usize>;
}

/// The backing data of an in-memory [`Stream`].
#[derive(Clone)]
pub enum StreamData {
    /// Buffers owned by the stream.
    Owned(Vec<u8>),
    /// Buffers shared between several consumers (zero-copy font faces).
    Shared(alloc::sync::Arc<[u8]>),
    /// Statically allocated buffers (embedded fonts, test fixtures).
    Static(&'static [u8]),
}

impl StreamData {
    /// The raw bytes of the buffer.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            StreamData::Owned(v) => v.as_slice(),
            StreamData::Shared(a) => a.as_ref(),
            StreamData::Static(s) => s,
        }
    }
}

impl From<Vec<u8>> for StreamData {
    #[inline]
    fn from(v: Vec<u8>) -> Self {
        StreamData::Owned(v)
    }
}

impl From<alloc::sync::Arc<[u8]>> for StreamData {
    #[inline]
    fn from(a: alloc::sync::Arc<[u8]>) -> Self {
        StreamData::Shared(a)
    }
}

/// The origin of a [`Stream`]'s bytes.
enum StreamOrigin {
    /// In-memory data; `base` never changes after construction, which is
    /// what makes zero-copy frames ([`Stream::extract_frame`]) sound.
    Memory(StreamData),
    /// An external random-access source (file, callback, compression).
    External(Arc<dyn StreamSource>),
}

/// Per-stream mutable state (position + active frame bookkeeping).
struct StreamState {
    size: u64,
    pos: u64,
    frame_active: bool,
    frame_buf: Vec<u8>,
}

/// A seekable byte stream, the Rust port of `FT_StreamRec`.
///
/// # Frame discipline
///
/// C code brackets random access with `FT_Stream_EnterFrame` /
/// `FT_Stream_ExitFrame`. Here, entering a frame returns a [`Frame`]
/// guard; dropping the guard exits the frame (RAII). Entering a frame
/// while another is active returns [`TtError::NESTED_FRAME_ACCESS`],
/// matching FreeType's rule. For memory streams the frame borrows the
/// stream data zero-copy; for external sources the frame bytes are
/// buffered inside the stream (like FreeType's `stream->read` path).
///
/// # Thread safety
///
/// A `Stream` can be shared between threads via `Arc` only if all access
/// is serialized externally (the interior state uses `RefCell`, not a
/// lock); in practice a face owns its stream exclusively.
pub struct Stream {
    origin: StreamOrigin,
    state: core::cell::RefCell<StreamState>,
}

impl Stream {
    /// Creates an in-memory stream over `data`.
    pub fn from_data(data: StreamData) -> Self {
        let size = data.as_bytes().len() as u64;
        Stream {
            origin: StreamOrigin::Memory(data),
            state: core::cell::RefCell::new(StreamState {
                size,
                pos: 0,
                frame_active: false,
                frame_buf: Vec::new(),
            }),
        }
    }

    /// Creates an in-memory stream over an owned buffer.
    #[inline]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self::from_data(StreamData::Owned(bytes))
    }

    /// Creates a stream over a `static` buffer without copying.
    #[inline]
    pub fn from_static(bytes: &'static [u8]) -> Self {
        Self::from_data(StreamData::Static(bytes))
    }

    /// Creates a stream over a shared (`Arc`) buffer without copying.
    #[inline]
    pub fn from_shared(bytes: alloc::sync::Arc<[u8]>) -> Self {
        Self::from_data(StreamData::Shared(bytes))
    }

    /// Creates a stream over an external [`StreamSource`] (file, user
    /// callback, compressed layer, ...). Size is queried lazily.
    pub fn from_source(source: Arc<dyn StreamSource>) -> Self {
        let size = source.size().unwrap_or(0);
        Stream {
            origin: StreamOrigin::External(source),
            state: core::cell::RefCell::new(StreamState {
                size,
                pos: 0,
                frame_active: false,
                frame_buf: Vec::new(),
            }),
        }
    }

    /// `FT_Stream_Size`: the total size of the stream in bytes.
    #[inline]
    pub fn size(&self) -> u64 {
        match &self.origin {
            StreamOrigin::Memory(d) => d.as_bytes().len() as u64,
            StreamOrigin::External(_) => self.state.borrow().size,
        }
    }

    /// `FT_Stream_Pos`: the current position.
    #[inline]
    pub fn pos(&self) -> u64 {
        self.state.borrow().pos
    }

    /// `FT_Stream_Seek`: moves the current position.
    ///
    /// Seeking to `size` itself is valid (it positions the cursor at
    /// end-of-stream), matching FreeType.
    pub fn seek(&self, pos: u64) -> TtResult<()> {
        if pos > self.size() {
            return Err(TtError::INVALID_STREAM_SEEK);
        }
        self.state.borrow_mut().pos = pos;
        Ok(())
    }

    /// `FT_Stream_Skip`: advances the current position by `distance`.
    ///
    /// Negative distances are rejected with
    /// [`TtError::INVALID_STREAM_OPERATION`] (as in FreeType 2.6.5).
    pub fn skip(&self, distance: i64) -> TtResult<()> {
        if distance < 0 {
            return Err(TtError::INVALID_STREAM_OPERATION);
        }
        let pos = self.pos();
        let target = pos
            .checked_add(distance as u64)
            .ok_or(TtError::INVALID_STREAM_SEEK)?;
        self.seek(target)
    }

    /// Reads `count` bytes at absolute `pos` into `buf`, advancing the
    /// position to `pos + read`.
    ///
    /// Returns [`TtError::INVALID_STREAM_OPERATION`] when the read runs
    /// past the end of the stream (FreeType's `FT_Stream_ReadAt`).
    pub fn read_at(&self, pos: u64, buf: &mut [u8]) -> TtResult<()> {
        let size = self.size();
        if pos >= size && !buf.is_empty() {
            return Err(TtError::INVALID_STREAM_OPERATION);
        }
        let available = size.saturating_sub(pos);
        let want = buf.len() as u64;
        let count = available.min(want) as usize;
        let read = match &self.origin {
            StreamOrigin::Memory(d) => {
                let bytes = d.as_bytes();
                let start = pos as usize;
                let end = start + count;
                let src = bytes
                    .get(start..end)
                    .ok_or(TtError::INVALID_STREAM_READ)?;
                buf[..count].copy_from_slice(src);
                count
            }
            StreamOrigin::External(src) => src.read_at(pos, &mut buf[..count])?,
        };
        self.state.borrow_mut().pos = pos + read as u64;
        if read < buf.len() {
            return Err(TtError::INVALID_STREAM_OPERATION);
        }
        Ok(())
    }

    /// Reads `buf.len()` bytes at the current position, advancing it.
    #[inline]
    pub fn read(&self, buf: &mut [u8]) -> TtResult<()> {
        let pos = self.pos();
        self.read_at(pos, buf)
    }

    /// `FT_Stream_TryRead`: reads up to `buf.len()` bytes and returns how
    /// many were actually read (never fails on short input).
    pub fn try_read(&self, buf: &mut [u8]) -> usize {
        let pos = self.pos();
        let size = self.size();
        if pos >= size {
            return 0;
        }
        let count = ((size - pos) as usize).min(buf.len());
        if self.read_at(pos, &mut buf[..count]).is_err() {
            return 0;
        }
        count
    }

    /// `FT_Stream_EnterFrame`: enters a frame of `count` bytes.
    ///
    /// The returned [`Frame`] reads big/little-endian values through its
    /// `get_*` methods (the `FT_GET_*` macros) and exits the frame when
    /// dropped.
    pub fn enter_frame(&self, count: usize) -> TtResult<Frame<'_>> {
        let size = self.size();
        let pos = self.pos();
        if (count as u64) > size || size - pos < count as u64 {
            return Err(TtError::INVALID_FRAME_OPERATION);
        }
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| TtError::NESTED_FRAME_ACCESS)?;
        if state.frame_active {
            return Err(TtError::NESTED_FRAME_ACCESS);
        }
        match &self.origin {
            StreamOrigin::Memory(_) => {
                state.pos += count as u64;
                state.frame_active = true;
                drop(state);
                // `base` is immutable for memory streams, so the slice
                // remains valid for as long as `&self` is borrowed.
                let data: &[u8] = match &self.origin {
                    StreamOrigin::Memory(d) => {
                        let bytes = d.as_bytes();
                        bytes
                            .get(pos as usize..pos as usize + count)
                            .ok_or(TtError::INVALID_FRAME_READ)?
                    }
                    StreamOrigin::External(_) => return Err(TtError::INVALID_FRAME_OPERATION),
                };
                Ok(Frame {
                    stream: self,
                    cursor: 0,
                    len: count,
                    data: FrameData::Borrowed(data),
                })
            }
            StreamOrigin::External(src) => {
                if count > isize::MAX as usize {
                    return Err(TtError::INVALID_FRAME_OPERATION);
                }
                state.frame_buf.clear();
                state.frame_buf.resize(count, 0);
                let read = src.read_at(pos, &mut state.frame_buf)?;
                if read < count {
                    state.frame_buf.clear();
                    return Err(TtError::INVALID_FRAME_OPERATION);
                }
                state.pos = pos + count as u64;
                state.frame_active = true;
                let data = core::cell::RefMut::map(state, |s| s.frame_buf.as_mut_slice());
                Ok(Frame {
                    stream: self,
                    cursor: 0,
                    len: count,
                    data: FrameData::Buffered(data),
                })
            }
        }
    }

    /// `FT_Stream_ExtractFrame`: extracts `count` bytes as a borrowed
    /// slice (memory streams) or an owned copy (external streams).
    ///
    /// Borrowed slices are sound because a memory stream's base never
    /// changes after construction; the borrow checker ties the slice to
    /// `&self`. External streams copy, exactly like FreeType's
    /// heap-allocated extract path.
    pub fn extract_frame(&self, count: usize) -> TtResult<CowFrame<'_>> {
        let size = self.size();
        let pos = self.pos();
        if (count as u64) > size || size - pos < count as u64 {
            return Err(TtError::INVALID_FRAME_OPERATION);
        }
        match &self.origin {
            StreamOrigin::Memory(d) => {
                let bytes = d.as_bytes();
                let slice = bytes
                    .get(pos as usize..pos as usize + count)
                    .ok_or(TtError::INVALID_FRAME_READ)?;
                self.state.borrow_mut().pos = pos + count as u64;
                Ok(CowFrame::Borrowed(slice))
            }
            StreamOrigin::External(src) => {
                let mut buf = vec![0u8; count];
                let read = src.read_at(pos, &mut buf)?;
                if read < count {
                    return Err(TtError::INVALID_FRAME_OPERATION);
                }
                self.state.borrow_mut().pos = pos + count as u64;
                Ok(CowFrame::Owned(buf))
            }
        }
    }

    /// Reads a big-endian `u8` at the current position.
    pub fn read_u8(&self) -> TtResult<u8> {
        let mut b = [0u8; 1];
        self.read(&mut b)?;
        Ok(b[0])
    }

    /// Reads a big-endian `u16` at the current position.
    pub fn read_u16(&self) -> TtResult<u16> {
        let mut b = [0u8; 2];
        self.read(&mut b)?;
        Ok(u16::from_be_bytes(b))
    }

    /// Reads a big-endian `u24` at the current position.
    pub fn read_u24(&self) -> TtResult<u32> {
        let mut b = [0u8; 3];
        self.read(&mut b)?;
        Ok(u32::from_be_bytes([0, b[0], b[1], b[2]]))
    }

    /// Reads a big-endian `u32` at the current position.
    pub fn read_u32(&self) -> TtResult<u32> {
        let mut b = [0u8; 4];
        self.read(&mut b)?;
        Ok(u32::from_be_bytes(b))
    }

    /// Reads a little-endian `u16` at the current position.
    pub fn read_u16_le(&self) -> TtResult<u16> {
        let mut b = [0u8; 2];
        self.read(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    /// Reads a little-endian `u32` at the current position.
    pub fn read_u32_le(&self) -> TtResult<u32> {
        let mut b = [0u8; 4];
        self.read(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
}

impl core::fmt::Debug for Stream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Stream")
            .field("size", &self.size())
            .field("pos", &self.pos())
            .finish()
    }
}

/// The bytes backing an active [`Frame`].
enum FrameData<'a> {
    /// Zero-copy view of the immutable memory base.
    Borrowed(&'a [u8]),
    /// Borrow of `Stream::state.frame_buf` (external sources).
    Buffered(core::cell::RefMut<'a, [u8]>),
}

/// An active stream frame (the `cursor`/`limit` window of FreeType's
/// `FT_Stream_EnterFrame`).
///
/// Dropping the frame exits it. `get_*` methods are the `FT_GET_*`
/// macros; they return [`TtError::INVALID_FRAME_READ`] if the frame is
/// exhausted, replacing C's unchecked (and unsafe) cursor advances.
pub struct Frame<'a> {
    stream: &'a Stream,
    cursor: usize,
    len: usize,
    data: FrameData<'a>,
}

impl<'a> Frame<'a> {
    /// The frame window as bytes (`limit - cursor`, shrinking as the
    /// cursor advances).
    #[inline]
    pub fn rest(&self) -> &[u8] {
        let all: &[u8] = match &self.data {
            FrameData::Borrowed(d) => d,
            FrameData::Buffered(r) => r,
        };
        all.get(self.cursor..).unwrap_or(&[])
    }

    /// The absolute cursor position within the frame.
    #[inline]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// The total length of the frame in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when the cursor reached the end of the frame.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn take(&mut self, n: usize) -> TtResult<&[u8]> {
        let all: &[u8] = match &self.data {
            FrameData::Borrowed(d) => d,
            FrameData::Buffered(r) => r,
        };
        let end = self
            .cursor
            .checked_add(n)
            .ok_or(TtError::INVALID_FRAME_READ)?;
        if end > all.len() {
            return Err(TtError::INVALID_FRAME_READ);
        }
        let slice = &all[self.cursor..end];
        self.cursor = end;
        Ok(slice)
    }

    /// `FT_Stream_GetChar`: reads one signed byte from the frame.
    pub fn get_char(&mut self) -> TtResult<i8> {
        let s = self.take(1)?;
        Ok(s[0] as i8)
    }

    /// `FT_Stream_GetUShort`: reads a big-endian `u16` from the frame.
    pub fn get_u16(&mut self) -> TtResult<u16> {
        let s = self.take(2)?;
        Ok(u16::from_be_bytes([s[0], s[1]]))
    }

    /// `FT_Stream_GetUOffset`: reads a big-endian 24-bit value.
    pub fn get_u24(&mut self) -> TtResult<u32> {
        let s = self.take(3)?;
        Ok(u32::from_be_bytes([0, s[0], s[1], s[2]]))
    }

    /// `FT_Stream_GetULong`: reads a big-endian `u32` from the frame.
    pub fn get_u32(&mut self) -> TtResult<u32> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }

    /// `FT_Stream_GetUShortLE`: reads a little-endian `u16`.
    pub fn get_u16_le(&mut self) -> TtResult<u16> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    /// `FT_Stream_GetULongLE`: reads a little-endian `u32`.
    pub fn get_u32_le(&mut self) -> TtResult<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    /// Reads `n` raw bytes from the frame.
    pub fn get_bytes(&mut self, n: usize) -> TtResult<&[u8]> {
        self.take(n)
    }

    /// Skips `n` bytes within the frame.
    pub fn skip(&mut self, n: usize) -> TtResult<()> {
        self.take(n).map(|_| ())
    }
}

impl core::fmt::Debug for Frame<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Frame")
            .field("cursor", &self.cursor)
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        // Replacing the frame data first drops the `RefMut` (releasing
        // the RefCell borrow) before the active flag is cleared.
        self.data = FrameData::Borrowed(&[]);
        if let Ok(mut state) = self.stream.state.try_borrow_mut() {
            state.frame_active = false;
        }
    }
}

/// A frame extracted with [`Stream::extract_frame`].
///
/// Mirrors FreeType's `FT_Byte*` extract: memory streams borrow without
/// copying, external streams own their bytes. Releasing is a plain drop
/// (the RAII replacement for `FT_Stream_ReleaseFrame`).
pub enum CowFrame<'a> {
    /// Zero-copy view of a memory stream's base.
    Borrowed(&'a [u8]),
    /// Owned copy of an external stream's bytes.
    Owned(Vec<u8>),
}

impl<'a> CowFrame<'a> {
    /// The frame bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            CowFrame::Borrowed(b) => b,
            CowFrame::Owned(v) => v.as_slice(),
        }
    }

    /// The number of bytes in the frame.
    #[inline]
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    /// `true` when the frame is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Consumes the frame, yielding owned bytes (copies when borrowed).
    pub fn into_owned(self) -> Vec<u8> {
        match self {
            CowFrame::Borrowed(b) => b.to_vec(),
            CowFrame::Owned(v) => v,
        }
    }
}

impl core::ops::Deref for CowFrame<'_> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_bytes()
    }
}

/// `LibCell<T>`: the interior-mutability cell used by every object that
/// FreeType mutates through a shared handle (`FT_Face`, `FT_GlyphSlot`,
/// `FT_Size`, `FT_Library`, module objects, ...).
///
/// It is a thin wrapper over [`core::cell::RefCell`] that reports both
/// runtime borrow failures as [`TtError`] instead of panicking, so no
/// production path can abort the process.
///
/// # Borrow failures
///
/// A failed borrow means the caller re-entered an object that is already
/// mutably borrowed, which is a logic error. It is reported as
/// [`TtError::INVALID_HANDLE`].
pub struct LibCell<T> {
    inner: core::cell::RefCell<T>,
}

impl<T> LibCell<T> {
    /// Creates a cell holding `value`.
    #[inline]
    pub const fn new(value: T) -> Self {
        LibCell {
            inner: core::cell::RefCell::new(value),
        }
    }

    /// Shared borrow (`FT_...` accessors that only read).
    #[inline]
    pub fn borrow(&self) -> TtResult<core::cell::Ref<'_, T>> {
        self.inner
            .try_borrow()
            .map_err(|_| TtError::INVALID_HANDLE)
    }

    /// Exclusive borrow (the mutation path used by FreeType's drivers).
    #[inline]
    pub fn borrow_mut(&self) -> TtResult<core::cell::RefMut<'_, T>> {
        self.inner
            .try_borrow_mut()
            .map_err(|_| TtError::INVALID_HANDLE)
    }

    /// Replaces the contents, returning the previous value.
    pub fn replace(&self, value: T) -> TtResult<T> {
        let mut guard = self.borrow_mut()?;
        Ok(core::mem::replace(&mut *guard, value))
    }

    /// Rebuilds the contents from a reference to the old value, returning
    /// the previous value.
    pub fn replace_with(&self, f: impl FnOnce(&mut T) -> T) -> TtResult<T> {
        let mut guard = self.borrow_mut()?;
        let replacement = f(&mut guard);
        Ok(core::mem::replace(&mut *guard, replacement))
    }

    /// Mutable access through `&mut self` (no runtime check possible).
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.inner.get_mut()
    }

    /// Unwraps the cell, yielding the inner value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.inner.into_inner()
    }
}

impl<T: Default> Default for LibCell<T> {
    #[inline]
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> From<T> for LibCell<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T: fmt::Debug> fmt::Debug for LibCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.inner.try_borrow() {
            Ok(v) => f.debug_tuple("LibCell").field(&v).finish(),
            Err(_) => f.write_str("LibCell(<borrowed>)"),
        }
    }
}

/// `FT_MOVE_SIGN(x, s)`: moves the sign of `x` into `s`, leaving `x`
/// positive (ftcalc.c).
///
/// The negation wraps exactly like FreeType's two's-complement arithmetic;
/// FreeType dropped one's-complement support in 2.x.
#[inline]
fn move_sign(x: &mut i64, s: &mut i32) {
    if *x < 0 {
        *x = x.wrapping_neg();
        *s = -*s;
    }
}

/// `FT_RoundFix`: rounds a 16.16 value to the nearest integer.
#[inline]
pub const fn round_fix(a: Fixed) -> Fixed {
    // (a + 0x8000 - (a < 0)) & ~0xFFFF
    let t = a + 0x8000;
    if a < 0 { (t - 1) & !0xFFFF } else { t & !0xFFFF }
}

/// `FT_CeilFix`: rounds a 16.16 value up to the next integer.
#[inline]
pub const fn ceil_fix(a: Fixed) -> Fixed {
    (a + 0xFFFF) & !0xFFFF
}

/// `FT_FloorFix`: rounds a 16.16 value down to the previous integer.
#[inline]
pub const fn floor_fix(a: Fixed) -> Fixed {
    a & !0xFFFF
}

/// `FT_PIX_ROUND(x)`: rounds a 26.6 value to the nearest whole pixel.
#[inline]
pub const fn pix_round(x: Pos) -> Pos {
    (x + 32 + 63) & !63
}

/// `FT_PIX_CEIL(x)`: rounds a 26.6 value up to the next whole pixel.
#[inline]
pub const fn pix_ceil(x: Pos) -> Pos {
    (x + 63) & !63
}

/// `FT_PIX_FLOOR(x)`: rounds a 26.6 value down to the previous whole pixel.
#[inline]
pub const fn pix_floor(x: Pos) -> Pos {
    x & !63
}

/// `FT_PAD_FLOOR(x, n)`: rounds `x` down to the next multiple of `n`
/// (assumes `n` is a power of two).
#[inline]
pub const fn pad_floor(x: i64, n: i64) -> i64 {
    x & !(n - 1)
}

/// `FT_PAD_ROUND(x, n)`: rounds `x` to the next multiple of `n`
/// (assumes `n` is a power of two).
#[inline]
pub const fn pad_round(x: i64, n: i64) -> i64 {
    pad_floor(x + n / 2, n)
}

/// `FT_PAD_CEIL(x, n)`: rounds `x` up to the next multiple of `n`
/// (assumes `n` is a power of two).
#[inline]
pub const fn pad_ceil(x: i64, n: i64) -> i64 {
    pad_floor(x + (n - 1), n)
}

/// `FT_MSB(z)`: index of the most significant bit of `z` (0..31).
///
/// For `z == 0` the function returns 0, exactly like FreeType.
pub fn msb(z: u32) -> i32 {
    let mut z = z;
    let mut shift = 0;
    if z & 0xFFFF_0000 != 0 {
        z >>= 16;
        shift += 16;
    }
    if z & 0x0000_FF00 != 0 {
        z >>= 8;
        shift += 8;
    }
    if z & 0x0000_00F0 != 0 {
        z >>= 4;
        shift += 4;
    }
    if z & 0x0000_000C != 0 {
        z >>= 2;
        shift += 2;
    }
    if z & 0x0000_0002 != 0 {
        shift += 1;
    }
    shift
}

/// `FT_ABS(a)`: absolute value with FreeType's wrapping semantics (the
/// negation of `i64::MIN` wraps back to `i64::MIN`, as in C).
#[inline]
pub const fn abs_pos(a: i64) -> i64 {
    if a < 0 { a.wrapping_neg() } else { a }
}

/// `FT_MulDiv(a, b, c)`: computes `a * b / c` with rounding and a 64-bit
/// intermediate (`FT_LONG64` path of ftcalc.c).
///
/// `c == 0` saturates to `0x7FFFFFFF`, matching FreeType.
pub fn mul_div(a_: i64, b_: i64, c_: i64) -> i64 {
    let mut s: i32 = 1;
    let (mut a_, mut b_, mut c_) = (a_, b_, c_);
    move_sign(&mut a_, &mut s);
    move_sign(&mut b_, &mut s);
    move_sign(&mut c_, &mut s);

    let a = a_ as u64;
    let b = b_ as u64;
    let c = c_ as u64;

    let d = if c > 0 {
        a.wrapping_mul(b).wrapping_add(c >> 1) / c
    } else {
        0x7FFF_FFFF
    };

    let d_ = d as i64;
    if s < 0 { d_.wrapping_neg() } else { d_ }
}

/// `FT_MulDiv_No_Round(a, b, c)`: like [`mul_div`] but without rounding.
pub fn mul_div_no_round(a_: i64, b_: i64, c_: i64) -> i64 {
    let mut s: i32 = 1;
    let (mut a_, mut b_, mut c_) = (a_, b_, c_);
    move_sign(&mut a_, &mut s);
    move_sign(&mut b_, &mut s);
    move_sign(&mut c_, &mut s);

    let a = a_ as u64;
    let b = b_ as u64;
    let c = c_ as u64;

    let d = if c > 0 { a.wrapping_mul(b) / c } else { 0x7FFF_FFFF };

    let d_ = d as i64;
    if s < 0 { d_.wrapping_neg() } else { d_ }
}

/// `FT_MulFix(a, b)`: multiplies two 16.16 fixed-point values.
///
/// # Porting note
///
/// The reference configuration (`FT_CONFIG_OPTION_INLINE_MULFIX` plus
/// `FT_MULFIX_ASSEMBLER == FT_MulFix_x86_64`) redirects every call through
/// the inline `FT_MulFix(a, b) -> FT_MULFIX_ASSEMBLER((FT_Int32)a,
/// (FT_Int32)b)` macro, which is what this function reproduces bit for bit:
/// the inputs are truncated to 32 bits, the product is formed in 64 bits,
/// `0x8000` plus the sign bit is added for rounding and the result is
/// truncated back to 32 bits. Results therefore always fit in an `i32`,
/// exactly like the assembly helper. The rounding direction (add
/// `0x8000 - (product < 0)`) matches the portable `FT_LONG64` path for all
/// inputs inside the 32-bit domain.
#[inline]
pub fn mul_fix(a: Fixed, b: Fixed) -> Fixed {
    let a32 = a as i32;
    let b32 = b as i32;
    let ret = i64::from(a32) * i64::from(b32);
    let tmp = ret >> 63;
    let ret = ret.wrapping_add(0x8000).wrapping_add(tmp);
    (ret >> 16) as i32 as i64
}

/// `FT_DivFix(a, b)`: divides two 16.16 fixed-point values.
///
/// `b == 0` saturates to `0x7FFFFFFF`, matching FreeType.
pub fn div_fix(a_: i64, b_: i64) -> i64 {
    let mut s: i32 = 1;
    let (mut a_, mut b_) = (a_, b_);
    move_sign(&mut a_, &mut s);
    move_sign(&mut b_, &mut s);

    let a = a_ as u64;
    let b = b_ as u64;

    let q = if b > 0 {
        a.wrapping_mul(1 << 16).wrapping_add(b >> 1) / b
    } else {
        0x7FFF_FFFF
    };

    let q_ = q as i64;
    if s < 0 { q_.wrapping_neg() } else { q_ }
}

/// `FT_Matrix_Multiply(a, b)`: `b = a * b` (16.16 coefficients).
#[inline]
pub fn matrix_multiply(a: &Matrix, b: &mut Matrix) {
    let xx = mul_fix(a.xx, b.xx) + mul_fix(a.xy, b.yx);
    let xy = mul_fix(a.xx, b.xy) + mul_fix(a.xy, b.yy);
    let yx = mul_fix(a.yx, b.xx) + mul_fix(a.yy, b.yx);
    let yy = mul_fix(a.yx, b.xy) + mul_fix(a.yy, b.yy);
    b.xx = xx;
    b.xy = xy;
    b.yx = yx;
    b.yy = yy;
}

/// `FT_Matrix_Multiply_Scaled(a, b, scaling)`: like [`matrix_multiply`]
/// but with `scaling` extra integer bits of head-room in the intermediate
/// products (used while loading composite glyphs).
pub fn matrix_multiply_scaled(a: &Matrix, b: &mut Matrix, scaling: i64) {
    let val = 0x1_0000i64.wrapping_mul(scaling);
    let xx = mul_div(a.xx, b.xx, val) + mul_div(a.xy, b.yx, val);
    let xy = mul_div(a.xx, b.xy, val) + mul_div(a.xy, b.yy, val);
    let yx = mul_div(a.yx, b.xx, val) + mul_div(a.yy, b.yx, val);
    let yy = mul_div(a.yx, b.xy, val) + mul_div(a.yy, b.yy, val);
    b.xx = xx;
    b.xy = xy;
    b.yx = yx;
    b.yy = yy;
}

/// `FT_Matrix_Invert(matrix)`: inverts a 16.16 matrix in place.
///
/// Returns [`TtError::INVALID_ARGUMENT`] for a singular matrix.
pub fn matrix_invert(matrix: &mut Matrix) -> TtResult<()> {
    let delta = mul_fix(matrix.xx, matrix.yy) - mul_fix(matrix.xy, matrix.yx);
    if delta == 0 {
        return Err(TtError::INVALID_ARGUMENT);
    }
    matrix.xy = div_fix(matrix.xy, delta).wrapping_neg();
    matrix.yx = div_fix(matrix.yx, delta).wrapping_neg();
    let xx = matrix.xx;
    let yy = matrix.yy;
    matrix.xx = div_fix(yy, delta);
    matrix.yy = div_fix(xx, delta);
    Ok(())
}

/// `FT_Vector_Transform(vector, matrix)`: transforms a vector in place.
#[inline]
pub fn vector_transform(vector: &mut Vector, matrix: &Matrix) {
    let xz = mul_fix(vector.x, matrix.xx) + mul_fix(vector.y, matrix.xy);
    let yz = mul_fix(vector.x, matrix.yx) + mul_fix(vector.y, matrix.yy);
    vector.x = xz;
    vector.y = yz;
}

/// `FT_Vector_Transform_Scaled(vector, matrix, scaling)`: transforms a
/// vector in place with extra integer head-room in the intermediates.
pub fn vector_transform_scaled(vector: &mut Vector, matrix: &Matrix, scaling: i64) {
    let val = 0x1_0000i64.wrapping_mul(scaling);
    let xz = mul_div(vector.x, matrix.xx, val) + mul_div(vector.y, matrix.xy, val);
    let yz = mul_div(vector.x, matrix.yx, val) + mul_div(vector.y, matrix.yy, val);
    vector.x = xz;
    vector.y = yz;
}

/// `FT_Vector_NormLen(vector)`: normalizes `vector` to a 16.16 unit vector
/// and returns its original length.
///
/// The computation is FreeType's overflow-free Newton iteration: the inputs
/// are truncated to 32 bits, pre-normalized so that the estimate lands in
/// `[2/3, 4/3]` of 16.16, and the reciprocal length is refined iteratively.
pub fn vector_norm_len(vector: &mut Vector) -> u32 {
    let mut x_ = vector.x as i32;
    let mut y_ = vector.y as i32;
    let mut sx: i32 = 1;
    let mut sy: i32 = 1;

    if x_ < 0 {
        x_ = x_.wrapping_neg();
        sx = -sx;
    }
    if y_ < 0 {
        y_ = y_.wrapping_neg();
        sy = -sy;
    }
    let mut x: u32 = x_ as u32;
    let mut y: u32 = y_ as u32;
    if x == 0 {
        if y > 0 {
            vector.y = i64::from(sy) * 0x1_0000;
        }
        return y;
    } else if y == 0 {
        if x > 0 {
            vector.x = i64::from(sx) * 0x1_0000;
        }
        return x;
    }

    let mut l: u32 = if x > y {
        x.wrapping_add(y >> 1)
    } else {
        y.wrapping_add(x >> 1)
    };

    let mut shift = 31 - msb(l);
    shift -= 15 + i32::from(l >= (0xAAAA_AAAAu32 >> shift));

    if shift > 0 {
        x = x.wrapping_shl(shift as u32);
        y = y.wrapping_shl(shift as u32);
        l = if x > y {
            x.wrapping_add(y >> 1)
        } else {
            y.wrapping_add(x >> 1)
        };
    } else {
        x = x.wrapping_shr((-shift) as u32);
        y = y.wrapping_shr((-shift) as u32);
        l = l.wrapping_shr((-shift) as u32);
    }

    // lower linear approximation for reciprocal length minus one
    let mut b: i32 = 0x1_0000i32.wrapping_sub(l as i32);
    let x_i = x as i32;
    let y_i = y as i32;

    // Newton's iterations
    let (mut u, mut v);
    loop {
        let un = x_i.wrapping_add(x_i.wrapping_mul(b) >> 16);
        let vn = y_i.wrapping_add(y_i.wrapping_mul(b) >> 16);
        u = un as u32;
        v = vn as u32;

        // Normalized squared length in the parentheses approaches 2^32. On
        // two's complement systems, converting to signed gives the
        // difference with 2^32 even if the expression wraps around.
        let squared = u.wrapping_mul(u).wrapping_add(v.wrapping_mul(v));
        let mut z = (squared as i32)
            .wrapping_neg()
            .wrapping_div(0x200);
        z = z
            .wrapping_mul((0x1_0000i32.wrapping_add(b)) >> 8)
            .wrapping_div(0x1_0000);

        b = b.wrapping_add(z);
        if z <= 0 {
            break;
        }
    }

    vector.x = if sx < 0 { -(u as i64) } else { u as i64 };
    vector.y = if sy < 0 { -(v as i64) } else { v as i64 };

    // Conversion to signed helps to recover from likely wrap around in
    // calculating the prenormalized length, because it gives the correct
    // difference with 2^32 on two's complement systems.
    let product = (u.wrapping_mul(x).wrapping_add(v.wrapping_mul(y))) as i32;
    let mut len = (0x1_0000i32.wrapping_add(product / 0x1_0000)) as u32;
    if shift > 0 {
        len = len
            .wrapping_add(1u32 << (shift - 1))
            .wrapping_shr(shift as u32);
    } else {
        len = len.wrapping_shl((-shift) as u32);
    }
    len
}

/// `FT_Hypot(x, y)`: the length of `(x, y)`.
///
/// This is `FT_Vector_Length` (fttrigon.c); see [`vector_length`].
#[inline]
pub fn hypot(x: Fixed, y: Fixed) -> Fixed {
    vector_length(&Vector { x, y })
}

/// `ft_corner_orientation(in, out)`: returns `+1` for a left turn, `-1` for
/// a right turn and `0` for collinear vectors.
pub fn corner_orientation(in_x: i64, in_y: i64, out_x: i64, out_y: i64) -> i32 {
    let delta = in_x.wrapping_mul(out_y) - in_y.wrapping_mul(out_x);
    i32::from(delta > 0) - i32::from(delta < 0)
}

/// `ft_corner_is_flat(in, out)`: `true` when the corner spanned by the two
/// vectors is flat enough to be rendered as a single segment.
///
/// Uses FreeType's `FT_HYPOT` approximation (alpha-max-plus-beta-min,
/// error < 7%): `d_in + d_out < 17/16 * d_hypot`.
pub fn corner_is_flat(in_x: i64, in_y: i64, out_x: i64, out_y: i64) -> bool {
    let ax = in_x.wrapping_add(out_x);
    let ay = in_y.wrapping_add(out_y);

    let d_in = ft_hypot_approx(in_x, in_y);
    let d_out = ft_hypot_approx(out_x, out_y);
    let d_hypot = ft_hypot_approx(ax, ay);

    d_in.wrapping_add(d_out).wrapping_sub(d_hypot) < (d_hypot >> 4)
}

/// `FT_HYPOT(x, y)` from `ftobjs.h`: the approximate `sqrt(x*x + y*y)`
/// using `alpha = 1`, `beta = 3/8` (mutates its arguments in C, which is
/// modelled here by taking them by value).
#[inline]
fn ft_hypot_approx(x: i64, y: i64) -> i64 {
    let x = abs_pos(x);
    let y = abs_pos(y);
    if x > y {
        x.wrapping_add(y.wrapping_mul(3) >> 3)
    } else {
        y.wrapping_add(x.wrapping_mul(3) >> 3)
    }
}

/// The point loop of `FT_Outline_Translate` (ftoutln.c): shifts every
/// point by `(dx, dy)`.
///
/// This is the scalar reference implementation; the SIMD dispatcher
/// (`translate_points_simd`) must produce bit-identical results.
#[inline]
pub fn translate_points(points: &mut [Vector], dx: Pos, dy: Pos) {
    for p in points.iter_mut() {
        p.x = p.x.wrapping_add(dx);
        p.y = p.y.wrapping_add(dy);
    }
}

/// The point loop of `FT_Outline_Transform` (ftoutln.c): applies `matrix`
/// to every point with [`vector_transform`].
///
/// This is the scalar reference implementation; the SIMD dispatcher
/// (`transform_points_simd`) must produce bit-identical results.
#[inline]
pub fn transform_points(points: &mut [Vector], matrix: &Matrix) {
    for p in points.iter_mut() {
        vector_transform(p, matrix);
    }
}

/// `FT_TRIG_SCALE`: the CORDIC shrink factor `0.858785336480436 * 2^32`.
const TRIG_SCALE: u64 = 0xDBD9_5B16;

/// `FT_TRIG_SAFE_MSB`: highest bit usable in overflow-safe vector
/// components.
const TRIG_SAFE_MSB: i32 = 29;

/// `FT_TRIG_MAX_ITERS`: number of CORDIC iterations (table length is
/// `FT_TRIG_MAX_ITERS - 1`).
const TRIG_MAX_ITERS: i32 = 23;

/// `ft_trig_arctan_table`: arctangents for `FT_ANGLE_PI = 180 << 16`.
const TRIG_ARCTAN_TABLE: [Angle; 22] = [
    1_740_967, 919_879, 466_945, 234_379, 117_304, 58_666, 29_335, 14_668, 7_334, 3_667, 1_833, 917, 458,
    229, 115, 57, 29, 14, 7, 4, 2, 1,
];

/// `ft_trig_downscale(val)`: multiplies `val` by the CORDIC shrink factor.
///
/// The `0x40000000` bias comes from a regression analysis between the true
/// and the CORDIC hypotenuse, so it minimizes the error.
fn trig_downscale(val: Fixed) -> Fixed {
    let mut val = val;
    let mut s: i32 = 1;
    if val < 0 {
        val = val.wrapping_neg();
        s = -1;
    }
    let scaled = (val as u64)
        .wrapping_mul(TRIG_SCALE)
        .wrapping_add(0x4000_0000)
        >> 32;
    let out = scaled as i64;
    if s < 0 { out.wrapping_neg() } else { out }
}

/// `ft_trig_prenorm(vec)`: scales the vector so that its largest component
/// lies just under `2^FT_TRIG_SAFE_MSB`, returning the applied shift.
fn trig_prenorm(vec: &mut Vector) -> i32 {
    let x = vec.x;
    let y = vec.y;

    let mut shift = msb((abs_pos(x) | abs_pos(y)) as u32);

    if shift <= TRIG_SAFE_MSB {
        shift = TRIG_SAFE_MSB - shift;
        vec.x = ((x as u64) << shift) as i64;
        vec.y = ((y as u64) << shift) as i64;
    } else {
        shift -= TRIG_SAFE_MSB;
        vec.x = x >> shift;
        vec.y = y >> shift;
        shift = -shift;
    }
    shift
}

/// `ft_trig_pseudo_rotate(vec, theta)`: rotates `vec` by `theta` using
/// CORDIC pseudo-rotations, keeping the vector length approximately intact.
fn trig_pseudo_rotate(vec: &mut Vector, mut theta: Angle) {
    let mut x = vec.x;
    let mut y = vec.y;

    // rotate inside [-PI/4, PI/4]
    while theta < -ANGLE_PI4 {
        let xtemp = y;
        y = x.wrapping_neg();
        x = xtemp;
        theta = theta.wrapping_add(ANGLE_PI2);
    }
    while theta > ANGLE_PI4 {
        let xtemp = y.wrapping_neg();
        y = x;
        x = xtemp;
        theta = theta.wrapping_sub(ANGLE_PI2);
    }

    for i in 1..TRIG_MAX_ITERS {
        let b = 1i64 << (i - 1);
        let entry = TRIG_ARCTAN_TABLE[(i - 1) as usize];
        if theta < 0 {
            let xtemp = x.wrapping_add(y.wrapping_add(b) >> i);
            y = y.wrapping_sub(x.wrapping_add(b) >> i);
            x = xtemp;
            theta = theta.wrapping_add(entry);
        } else {
            let xtemp = x.wrapping_sub(y.wrapping_add(b) >> i);
            y = y.wrapping_add(x.wrapping_add(b) >> i);
            x = xtemp;
            theta = theta.wrapping_sub(entry);
        }
    }

    vec.x = x;
    vec.y = y;
}

/// `ft_trig_pseudo_polarize(vec)`: converts `vec` from Cartesian to polar
/// form in place (`x` = length, `y` = angle in `Angle` units).
fn trig_pseudo_polarize(vec: &mut Vector) {
    let mut x = vec.x;
    let mut y = vec.y;
    let mut theta: Angle = 0;

    // get the vector into the [-PI/4, PI/4] sector
    if y > x {
        if y > -x {
            theta = ANGLE_PI2;
            let xtemp = y;
            y = x.wrapping_neg();
            x = xtemp;
        } else {
            theta = if y > 0 { ANGLE_PI } else { -ANGLE_PI };
            x = x.wrapping_neg();
            y = y.wrapping_neg();
        }
    } else if y < -x {
        theta = -ANGLE_PI2;
        let xtemp = y.wrapping_neg();
        y = x;
        x = xtemp;
    }

    for i in 1..TRIG_MAX_ITERS {
        let b = 1i64 << (i - 1);
        let entry = TRIG_ARCTAN_TABLE[(i - 1) as usize];
        if y > 0 {
            let xtemp = x.wrapping_add(y.wrapping_add(b) >> i);
            y = y.wrapping_sub(x.wrapping_add(b) >> i);
            x = xtemp;
            theta = theta.wrapping_add(entry);
        } else {
            let xtemp = x.wrapping_sub(y.wrapping_add(b) >> i);
            y = y.wrapping_add(x.wrapping_add(b) >> i);
            x = xtemp;
            theta = theta.wrapping_sub(entry);
        }
    }

    // round theta to acknowledge its error that mostly comes from
    // accumulated rounding errors in the arctan table
    theta = if theta >= 0 {
        pad_round(theta, 16)
    } else {
        pad_round(-theta, 16).wrapping_neg()
    };

    vec.x = x;
    vec.y = theta;
}

/// `FT_Cos(angle)`: the cosine of `angle` (16.16 result).
#[inline]
pub fn cos(angle: Angle) -> Fixed {
    let mut v = Vector::ZERO;
    vector_unit(&mut v, angle);
    v.x
}

/// `FT_Sin(angle)`: the sine of `angle` (16.16 result).
#[inline]
pub fn sin(angle: Angle) -> Fixed {
    let mut v = Vector::ZERO;
    vector_unit(&mut v, angle);
    v.y
}

/// `FT_Tan(angle)`: the tangent of `angle` (16.16 result).
#[inline]
pub fn tan(angle: Angle) -> Fixed {
    let mut v = Vector::ZERO;
    vector_unit(&mut v, angle);
    div_fix(v.y, v.x)
}

/// `FT_Atan2(dx, dy)`: returns the angle of the vector `(dx, dy)` in
/// 16.16 degrees. The zero vector maps to 0.
pub fn atan2(dx: Fixed, dy: Fixed) -> Angle {
    if dx == 0 && dy == 0 {
        return 0;
    }
    let mut v = Vector { x: dx, y: dy };
    trig_prenorm(&mut v);
    trig_pseudo_polarize(&mut v);
    v.y
}

/// `FT_Vector_Unit(vec, angle)`: sets `vec` to the unit vector at `angle`.
pub fn vector_unit(vec: &mut Vector, angle: Angle) {
    vec.x = (TRIG_SCALE >> 8) as i64;
    vec.y = 0;
    trig_pseudo_rotate(vec, angle);
    vec.x = (vec.x.wrapping_add(0x80)) >> 8;
    vec.y = (vec.y.wrapping_add(0x80)) >> 8;
}

/// `FT_Vector_Rotate(vec, angle)`: rotates a vector in place.
pub fn vector_rotate(vec: &mut Vector, angle: Angle) {
    if angle == 0 {
        return;
    }
    let mut v = *vec;
    if v.x == 0 && v.y == 0 {
        return;
    }

    let shift = trig_prenorm(&mut v);
    trig_pseudo_rotate(&mut v, angle);
    v.x = trig_downscale(v.x);
    v.y = trig_downscale(v.y);

    if shift > 0 {
        let half = 1i64 << (shift - 1);
        // FT_SIGN_LONG(x) is `x >> 63` (arithmetic shift)
        vec.x = (v.x.wrapping_add(half).wrapping_add(v.x >> 63)) >> shift;
        vec.y = (v.y.wrapping_add(half).wrapping_add(v.y >> 63)) >> shift;
    } else {
        let s = (-shift) as u32;
        vec.x = ((v.x as u64) << s) as i64;
        vec.y = ((v.y as u64) << s) as i64;
    }
}

/// `FT_Vector_Length(vec)`: the CORDIC length of a vector.
pub fn vector_length(vec: &Vector) -> Fixed {
    let mut v = *vec;

    // trivial cases
    if v.x == 0 {
        return abs_pos(v.y);
    } else if v.y == 0 {
        return abs_pos(v.x);
    }

    let shift = trig_prenorm(&mut v);
    trig_pseudo_polarize(&mut v);
    v.x = trig_downscale(v.x);

    if shift > 0 {
        return (v.x.wrapping_add(1i64 << (shift - 1))) >> shift;
    }
    ((v.x as u32) << (-shift) as u32) as i64
}

/// `FT_Vector_Polarize(vec, length, angle)`: decomposes a vector into its
/// length and angle.
///
/// # Zero vector
///
/// As in FreeType, a zero vector is rejected without touching the outputs;
/// initialize `length` and `angle` before calling if the vector may be
/// zero.
pub fn vector_polarize(vec: &Vector, length: &mut Fixed, angle: &mut Angle) {
    let mut v = *vec;
    if v.x == 0 && v.y == 0 {
        return;
    }

    let shift = trig_prenorm(&mut v);
    trig_pseudo_polarize(&mut v);
    v.x = trig_downscale(v.x);

    *length = if shift >= 0 {
        v.x >> shift
    } else {
        ((v.x as u32) << (-shift) as u32) as i64
    };
    *angle = v.y;
}

/// `FT_Vector_From_Polar(vec, length, angle)`: builds a vector from its
/// polar coordinates.
pub fn vector_from_polar(vec: &mut Vector, length: Fixed, angle: Angle) {
    vec.x = length;
    vec.y = 0;
    vector_rotate(vec, angle);
}

/// `FT_Angle_Diff(angle1, angle2)`: the signed difference of two angles,
/// normalized to `(-180, 180]` degrees.
pub fn angle_diff(angle1: Angle, angle2: Angle) -> Angle {
    let mut delta = angle2.wrapping_sub(angle1);
    while delta <= -ANGLE_PI {
        delta = delta.wrapping_add(ANGLE_2PI);
    }
    while delta > ANGLE_PI {
        delta = delta.wrapping_sub(ANGLE_2PI);
    }
    delta
}

//
// FreeType builds three debug modes (trace / error / release).  The
// reference configuration of this port (`ftoption.h`) leaves both
// `FT_DEBUG_LEVEL_ERROR` and `FT_DEBUG_LEVEL_TRACE` undefined, so the C
// build is the *release* mode: `FT_TRACE*` and `FT_ERROR` expand to
// nothing and only `ft_debug_init` survives as a no-op.
//
// The port keeps the whole tracing sub-system available instead of
// compiling it out: every toggle defaults to level 0 (quiet, matching the
// release build), all output goes through `codevar_logger` as required by
// AGENTS.md, and `ft_trace!` costs a single relaxed atomic load when the
// component is quiet.

/// Trace components, generated from `include/freetype/internal/fttrace.h`.
///
/// The discriminants are the `trace_*` values of FreeType's `FT_Trace`
/// enum, so `TraceComponent::Any` is always index 0.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(i32)]
pub enum TraceComponent {
    /// `trace_any`: matches every component.
    Any = 0,
    /// `trace_calc`: calculations (`ftcalc.c`).
    Calc,
    /// `trace_memory`: memory manager (`ftobjs.c`).
    Memory,
    /// `trace_stream`: stream manager (`ftstream.c`).
    Stream,
    /// `trace_io`: i/o interface (`ftsystem.c`).
    Io,
    /// `trace_list`: list management (`ftlist.c`).
    List,
    /// `trace_init`: initialization (`ftinit.c`).
    Init,
    /// `trace_objs`: base objects (`ftobjs.c`).
    Objs,
    /// `trace_outline`: outline management (`ftoutln.c`).
    Outline,
    /// `trace_glyph`: glyph management (`ftglyph.c`).
    Glyph,
    /// `trace_gloader`: glyph loader (`ftgloadr.c`).
    Gloader,
    /// `trace_raster`: monochrome rasterizer (`ftraster.c`).
    Raster,
    /// `trace_smooth`: anti-aliasing rasterizer (`ftgrays.c`).
    Smooth,
    /// `trace_mm`: multiple-master interface (`ftmm.c`).
    Mm,
    /// `trace_raccess`: resource fork accessor (`ftrfork.c`).
    Raccess,
    /// `trace_synth`: bold/slant synthesizer (`ftsynth.c`).
    Synth,
    /// `trace_bitmap`: bitmap checksum (`ftobjs.c`).
    Bitmap,
    /// `trace_cache`: cache sub-system (`ftcache.c`).
    Cache,
    /// `trace_sfdriver`: SFNT font driver.
    Sfdriver,
    /// `trace_sfobjs`: SFNT object handler.
    Sfobjs,
    /// `trace_ttcmap`: charmap handler.
    Ttcmap,
    /// `trace_ttkern`: kerning handler.
    Ttkern,
    /// `trace_ttload`: basic TrueType tables.
    Ttload,
    /// `trace_ttmtx`: metrics-related tables.
    Ttmtx,
    /// `trace_ttpost`: PostScript table processing.
    Ttpost,
    /// `trace_ttsbit`: embedded bitmap handling.
    Ttsbit,
    /// `trace_ttbdf`: TrueType embedded BDF.
    Ttbdf,
    /// `trace_ttdriver`: TrueType font driver.
    Ttdriver,
    /// `trace_ttgload`: TrueType glyph loader.
    Ttgload,
    /// `trace_ttinterp`: TrueType bytecode interpreter.
    Ttinterp,
    /// `trace_ttobjs`: TrueType objects manager.
    Ttobjs,
    /// `trace_ttpload`: TrueType data/program loader.
    Ttpload,
    /// `trace_ttgxvar`: TrueType GX variation handler.
    Ttgxvar,
    /// `trace_t1afm`: Type 1 AFM.
    T1afm,
    /// `trace_t1driver`: Type 1 driver.
    T1driver,
    /// `trace_t1gload`: Type 1 glyph loader.
    T1gload,
    /// `trace_t1hint`: Type 1 hinter.
    T1hint,
    /// `trace_t1load`: Type 1 loader.
    T1load,
    /// `trace_t1objs`: Type 1 objects.
    T1objs,
    /// `trace_t1parse`: Type 1 parser.
    T1parse,
    /// `trace_t1decode`: Type 1 decoder (`psaux`).
    T1decode,
    /// `trace_psobjs`: PostScript objects (`psaux`).
    Psobjs,
    /// `trace_psconv`: PostScript conversion (`psaux`).
    Psconv,
    /// `trace_pshrec`: PostScript hinter record (`pshinter`).
    Pshrec,
    /// `trace_pshalgo1`: hinter algorithm 1 (`pshinter`).
    Pshalgo1,
    /// `trace_pshalgo2`: hinter algorithm 2 (`pshinter`).
    Pshalgo2,
    /// `trace_cffdriver`: CFF driver.
    Cffdriver,
    /// `trace_cffgload`: CFF glyph loader.
    Cffgload,
    /// `trace_cffload`: CFF loader.
    Cffload,
    /// `trace_cffobjs`: CFF objects.
    Cffobjs,
    /// `trace_cffparse`: CFF parser.
    Cffparse,
    /// `trace_cf2blues`: Adobe CFF engine blues.
    Cf2blues,
    /// `trace_cf2hints`: Adobe CFF engine hints.
    Cf2hints,
    /// `trace_cf2interp`: Adobe CFF engine interpreter.
    Cf2interp,
    /// `trace_t42`: Type 42 driver.
    T42,
    /// `trace_cidafm`: CID AFM.
    Cidafm,
    /// `trace_ciddriver`: CID driver.
    Ciddriver,
    /// `trace_cidgload`: CID glyph loader.
    Cidgload,
    /// `trace_cidload`: CID loader.
    Cidload,
    /// `trace_cidobjs`: CID objects.
    Cidobjs,
    /// `trace_cidparse`: CID parser.
    Cidparse,
    /// `trace_winfnt`: Windows font driver.
    Winfnt,
    /// `trace_pcfdriver`: PCF driver.
    Pcfdriver,
    /// `trace_pcfread`: PCF reader.
    Pcfread,
    /// `trace_bdfdriver`: BDF driver.
    Bdfdriver,
    /// `trace_bdflib`: BDF library.
    Bdflib,
    /// `trace_pfr`: PFR font driver.
    Pfr,
    /// `trace_otvmodule`: OpenType validation module.
    Otvmodule,
    /// `trace_otvcommon`: OpenType validation common.
    Otvcommon,
    /// `trace_otvbase`: OpenType validation base.
    Otvbase,
    /// `trace_otvgdef`: validation of `GDEF`.
    Otvgdef,
    /// `trace_otvgpos`: validation of `GPOS`.
    Otvgpos,
    /// `trace_otvgsub`: validation of `GSUB`.
    Otvgsub,
    /// `trace_otvjstf`: validation of `JSTF`.
    Otvjstf,
    /// `trace_otvmath`: validation of `MATH`.
    Otvmath,
    /// `trace_gxvmodule`: TrueTypeGX/AAT validation module.
    Gxvmodule,
    /// `trace_gxvcommon`: TrueTypeGX/AAT validation common.
    Gxvcommon,
    /// `trace_gxvfeat`: validation of `feat`.
    Gxvfeat,
    /// `trace_gxvmort`: validation of `mort`.
    Gxvmort,
    /// `trace_gxvmorx`: validation of `morx`.
    Gxvmorx,
    /// `trace_gxvbsln`: validation of `bsln`.
    Gxvbsln,
    /// `trace_gxvjust`: validation of `just`.
    Gxvjust,
    /// `trace_gxvkern`: validation of `kern`.
    Gxvkern,
    /// `trace_gxvopbd`: validation of `opbd`.
    Gxvopbd,
    /// `trace_gxvtrak`: validation of `trak`.
    Gxvtrak,
    /// `trace_gxvprop`: validation of `prop`.
    Gxvprop,
    /// `trace_gxvlcar`: validation of `lcar`.
    Gxvlcar,
    /// `trace_afmodule`: autofit module.
    Afmodule,
    /// `trace_afhints`: autofit hints.
    Afhints,
    /// `trace_afcjk`: autofit CJK.
    Afcjk,
    /// `trace_aflatin`: autofit Latin.
    Aflatin,
    /// `trace_aflatin2`: autofit Latin (part 2).
    Aflatin2,
    /// `trace_afwarp`: autofit warper.
    Afwarp,
    /// `trace_afshaper`: autofit shaper.
    Afshaper,
    /// `trace_afglobal`: autofit global.
    Afglobal,
}

/// Number of trace components (`trace_count`).
pub const TRACE_COUNT: usize = 95;

/// Toggle names of every trace component, in `FT_Trace` order
/// (`ft_trace_toggles` in ftdebug.c).
pub const TRACE_NAMES: [&str; TRACE_COUNT] = [
    "any",
    "calc",
    "memory",
    "stream",
    "io",
    "list",
    "init",
    "objs",
    "outline",
    "glyph",
    "gloader",
    "raster",
    "smooth",
    "mm",
    "raccess",
    "synth",
    "bitmap",
    "cache",
    "sfdriver",
    "sfobjs",
    "ttcmap",
    "ttkern",
    "ttload",
    "ttmtx",
    "ttpost",
    "ttsbit",
    "ttbdf",
    "ttdriver",
    "ttgload",
    "ttinterp",
    "ttobjs",
    "ttpload",
    "ttgxvar",
    "t1afm",
    "t1driver",
    "t1gload",
    "t1hint",
    "t1load",
    "t1objs",
    "t1parse",
    "t1decode",
    "psobjs",
    "psconv",
    "pshrec",
    "pshalgo1",
    "pshalgo2",
    "cffdriver",
    "cffgload",
    "cffload",
    "cffobjs",
    "cffparse",
    "cf2blues",
    "cf2hints",
    "cf2interp",
    "t42",
    "cidafm",
    "ciddriver",
    "cidgload",
    "cidload",
    "cidobjs",
    "cidparse",
    "winfnt",
    "pcfdriver",
    "pcfread",
    "bdfdriver",
    "bdflib",
    "pfr",
    "otvmodule",
    "otvcommon",
    "otvbase",
    "otvgdef",
    "otvgpos",
    "otvgsub",
    "otvjstf",
    "otvmath",
    "gxvmodule",
    "gxvcommon",
    "gxvfeat",
    "gxvmort",
    "gxvmorx",
    "gxvbsln",
    "gxvjust",
    "gxvkern",
    "gxvopbd",
    "gxvtrak",
    "gxvprop",
    "gxvlcar",
    "afmodule",
    "afhints",
    "afcjk",
    "aflatin",
    "aflatin2",
    "afwarp",
    "afshaper",
    "afglobal",
];

impl TraceComponent {
    /// Index of the component inside [`TRACE_NAMES`] / `FT_Trace`.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Toggle name of the component (`FT_Trace_Get_Name`).
    #[inline]
    pub const fn name(self) -> &'static str {
        TRACE_NAMES[self.index()]
    }
}

/// `ft_trace_levels[trace_count]`: per-component trace levels, all 0
/// (quiet) until [`debug_init`] or [`set_trace_level`] changes them.
static TRACE_LEVELS: [core::sync::atomic::AtomicI32; TRACE_COUNT] =
    [const { core::sync::atomic::AtomicI32::new(0) }; TRACE_COUNT];

/// `FT_Trace_Get_Count()`: the number of trace components.
#[inline]
pub const fn trace_get_count() -> usize {
    TRACE_COUNT
}

/// `FT_Trace_Get_Name(idx)`: the toggle name of component `idx`, or `None`
/// when `idx` is out of range.
#[inline]
pub fn trace_get_name(idx: usize) -> Option<&'static str> {
    TRACE_NAMES.get(idx).copied()
}

/// Returns the current trace level (0..=7) of `component`.
#[inline]
pub fn trace_level(component: TraceComponent) -> i32 {
    TRACE_LEVELS[component.index()].load(core::sync::atomic::Ordering::Relaxed)
}

/// Sets the trace level (0..=7) of `component`; values outside the range
/// are clamped.
pub fn set_trace_level(component: TraceComponent, level: i32) {
    let level = level.clamp(0, 7);
    TRACE_LEVELS[component.index()].store(level, core::sync::atomic::Ordering::Relaxed);
}

/// The condition of FreeType's `FT_TRACE(level, ...)` macro: `true` when
/// the component's level is at least `level`.
#[inline]
pub fn trace_enabled(component: TraceComponent, level: i32) -> bool {
    trace_level(component) >= level
}

/// `FT_Message` for trace output: emits `message` at `codevar_logger`
/// debug level when `component` is enabled at `level`.
#[inline]
pub fn trace_message(component: TraceComponent, level: i32, message: &str) {
    if trace_enabled(component, level) {
        let _ = codevar_logger::log_with_timestamp(
            codevar_logger::LogLevel::Debug,
            &format!("[{component:?}] {message}"),
        );
    }
}

/// `FT_Message`: writes `message` to the log (stderr, via `codevar_logger`).
#[inline]
pub fn message(message: &str) {
    let _ = codevar_logger::log_with_timestamp(codevar_logger::LogLevel::Error, message);
}

/// `FT_Throw(error, line, file)` of the reference build: reports where an
/// error was raised and returns 0 (the `FT_THROW` decoration is disabled
/// when `FT_DEBUG_LEVEL_ERROR` is undefined).
pub fn throw(error: TtError, line: u32, file: &str) -> i32 {
    if trace_enabled(TraceComponent::Objs, 1) {
        message(&format!(
            "error 0x{:02X} raised at line {line} of {file}",
            error.code()
        ));
    }
    0
}

/// Replacement for `FT_Panic`: logs an assertion failure at the
/// irrecoverable level and returns instead of aborting, because panics and
/// `abort()` are forbidden in production paths.
#[inline]
pub fn assert_failed(line: u32, file: &str) {
    let _ = codevar_logger::log_with_timestamp(
        codevar_logger::LogLevel::Irr,
        &format!("assertion failed on line {line} of file {file}"),
    );
}

/// `ft_debug_init`: parses an `FT2_DEBUG`-style toggle list, for example
/// `"any:3 memory:7 stream:5"`.
///
/// # Porting note
///
/// The C implementation reads the `FT2_DEBUG` environment variable, which
/// is not available in `no_std` builds; the caller passes the string to
/// parse.  Separators are spaces, tabs, `,`, `;` and `=`, the level must be
/// a single digit `0`..`=7`, and the `any` toggle sets every component.
pub fn debug_init(spec: &str) {
    let bytes = spec.as_bytes();
    let mut p = 0usize;
    while p < bytes.len() {
        // skip leading whitespace and separators
        match bytes[p] {
            b' ' | b'\t' | b',' | b';' | b'=' => {
                p += 1;
                continue;
            }
            _ => {}
        }

        // read toggle name, followed by ':'
        let q = p;
        while p < bytes.len() && bytes[p] != b':' {
            p += 1;
        }
        if p >= bytes.len() {
            break;
        }

        if p > q {
            let name = &bytes[q..p];
            let mut found = None;
            for (n, toggle) in TRACE_NAMES.iter().enumerate() {
                if toggle.len() == name.len() && toggle.as_bytes() == name {
                    found = Some(n);
                    break;
                }
            }

            // read level
            p += 1;
            let mut level = -1i32;
            if p < bytes.len() {
                level = i32::from(bytes[p]) - i32::from(b'0');
                if !(0..=7).contains(&level) {
                    level = -1;
                }
            }

            if let Some(found) = found
                && level >= 0
            {
                if found == TraceComponent::Any.index() {
                    // special case for `any`
                    for slot in TRACE_LEVELS.iter() {
                        slot.store(level, core::sync::atomic::Ordering::Relaxed);
                    }
                } else {
                    TRACE_LEVELS[found].store(level, core::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        p += 1;
    }
}

/// Resets every trace toggle to level 0 (quiet).
pub fn debug_reset() {
    for slot in TRACE_LEVELS.iter() {
        slot.store(0, core::sync::atomic::Ordering::Relaxed);
    }
}

/// Regression tests for the stable public API that predates this module's
/// FreeType port (`FT_Stream`, `FT_Outline`, ...).
///
/// `.unwrap()` is permitted here because AGENTS.md allows it inside unit
/// tests; production paths are guarded by `#![deny(clippy::unwrap_used)]`.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn stream_memory_frame_reads_big_endian() {
        let stream = Stream::from_bytes(vec![0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03]);
        let mut frame = stream.enter_frame(8).unwrap();
        assert_eq!(frame.get_u16().unwrap(), 1);
        assert_eq!(frame.get_u16().unwrap(), 2);
        assert_eq!(frame.get_u32().unwrap(), 3);
        assert!(frame.get_char().is_err());
        drop(frame);
        assert_eq!(stream.pos(), 8);
    }

    #[test]
    fn stream_nested_frame_is_rejected() {
        let stream = Stream::from_bytes(vec![1, 2, 3, 4]);
        let _frame = stream.enter_frame(2).unwrap();
        assert_eq!(stream.enter_frame(2).unwrap_err(), TtError::NESTED_FRAME_ACCESS);
    }

    #[test]
    fn stream_extract_and_seek_semantics() {
        let stream = Stream::from_bytes((0u8..16).collect());
        stream.seek(4).unwrap();
        let frame = stream.extract_frame(4).unwrap();
        assert_eq!(frame.as_bytes(), &[4, 5, 6, 7]);
        assert_eq!(stream.pos(), 8);
        assert!(stream.seek(17).is_err());
        stream.seek(16).unwrap();
        assert!(stream.skip(-1).is_err());
        assert!(stream.skip(1).is_err());
    }

    #[test]
    fn stream_read_at_bounds() {
        let stream = Stream::from_bytes(vec![9, 8, 7]);
        let mut buf = [0u8; 5];
        assert!(stream.read_at(1, &mut buf).is_err());
        assert_eq!(&buf[..2], &[8, 7]);
        assert_eq!(stream.pos(), 3);
        let mut buf2 = [0u8; 2];
        assert!(stream.read_at(0, &mut buf2).is_ok());
        assert_eq!(buf2, [9, 8]);
    }

    #[test]
    fn outline_check_rejects_bad_contours() {
        let mut outline = Outline::new();
        outline.points = vec![Vector::new(0, 0), Vector::new(10, 0)];
        outline.tags = vec![CURVE_TAG_ON, CURVE_TAG_ON];
        outline.n_points = 2;
        outline.contours = vec![1];
        outline.n_contours = 1;
        assert!(outline.check().is_ok());

        outline.contours = vec![0];
        assert_eq!(outline.check().unwrap_err(), TtError::INVALID_OUTLINE);
    }

    #[test]
    fn outline_get_cbox_spans_all_points() {
        let mut outline = Outline::new();
        outline.points = vec![Vector::new(-5, 20), Vector::new(15, -3), Vector::new(7, 40)];
        outline.tags = vec![CURVE_TAG_ON; 3];
        outline.n_points = 3;
        let bbox = outline.get_cbox();
        assert_eq!(bbox, BBox::from_edges(-5, -3, 15, 40));
    }

    #[test]
    fn bitmap_negative_pitch_rows_are_top_down() {
        let mut bitmap = Bitmap::new_sized(3, 4, PixelMode::Gray, 256).unwrap();
        // Force an up-flow bitmap: buffer stays top-down in memory.
        bitmap.pitch = -(bitmap.stride() as i32);
        assert_eq!(bitmap.buffer.len(), 12);
        bitmap.row_mut(0).unwrap()[0] = 0xAA;
        bitmap.row_mut(2).unwrap()[0] = 0xBB;
        // Negative pitch: logical row 0 lives at the end of the buffer,
        // matching FreeType's `buffer + |pitch| * (rows - 1 - y)` rule.
        assert_eq!(bitmap.buffer[8], 0xAA);
        assert_eq!(bitmap.buffer[0], 0xBB);
        assert!(bitmap.row(3).is_none());
        assert!(bitmap.buffer_is_consistent());
    }
}
