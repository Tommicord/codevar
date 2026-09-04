//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! # Codevar File Parser
//!
//! Per-format and per-encoding utility parsers for file type detection,
//! header validation, and segmentation. Each format lives in its own module
//! (`parse_jpeg`, `parse_png`, …); text encodings live in
//! `parse_encoding_utf8` / `parse_encoding_windows12xx`.
//!
//! Implementations stay small and dependency-free (WASM-friendly), taking
//! structural cues from picojpeg / picoPNG rather than full decoders.

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(missing_docs)]
// Copyright headers use bare Apache License URLs across the workspace.
#![allow(clippy::doc_markdown)]

pub mod parse_apk;
pub mod parse_avif;
pub mod parse_common;
pub mod parse_dll;
pub mod parse_elf;
pub mod parse_encoding_iso_8859_1;
pub mod parse_encoding_us_ascii;
pub mod parse_encoding_utf16;
pub mod parse_encoding_utf32;
pub mod parse_encoding_utf8;
pub mod parse_encoding_windows12xx;
pub mod parse_ft;
pub mod parse_jpeg;
pub mod parse_lib;
pub mod parse_png;

pub use parse_apk::ApkParser;
pub use parse_avif::AvifParser;
pub use parse_common::{
    FormatParser, ParserError, ParserResult, Segment, SegmentKind, UtfParser,
    ValidationInfo, fourcc, read_u16_be, read_u16_le, read_u32_be, read_u32_le,
};
pub use parse_dll::DllParser;
pub use parse_elf::ElfParser;
pub use parse_encoding_iso_8859_1::Iso88591Parser;
pub use parse_encoding_us_ascii::UsAsciiParser;
pub use parse_encoding_utf8::Utf8Parser;
pub use parse_encoding_utf16::Utf16Parser;
pub use parse_encoding_utf32::Utf32Parser;
pub use parse_encoding_windows12xx::{Windows12xxCodePage, Windows12xxParser};
pub use parse_ft::{FileType, FileTypeDetector};
pub use parse_jpeg::JpegParser;
pub use parse_lib::LibParser;
pub use parse_png::{PNG_SIG, PngParser};
