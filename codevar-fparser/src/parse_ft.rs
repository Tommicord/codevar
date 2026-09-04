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

//! File type detection by delegating to per-format parsers.

use crate::UtfParser;
use crate::parse_apk::ApkParser;
use crate::parse_avif::AvifParser;
use crate::parse_common::FormatParser;
use crate::parse_dll::DllParser;
use crate::parse_elf::ElfParser;
use crate::parse_encoding_utf8::Utf8Parser;
use crate::parse_encoding_windows12xx::Windows12xxParser;
use crate::parse_jpeg::JpegParser;
use crate::parse_lib::LibParser;
use crate::parse_png::PngParser;

/// Detected file / content type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileType {
    /// Not recognized.
    Unknown,
    /// UTF-8 text.
    Utf8,
    /// Windows-1200 UTF-16LE.
    Utf16Le,
    /// Windows-1201 UTF-16BE.
    Utf16Be,
    /// JPEG image.
    Jpeg,
    /// PNG image.
    Png,
    /// AVIF image.
    Avif,
    /// ELF binary.
    Elf,
    /// PE / DLL / EXE.
    Dll,
    /// Static library (ar / MSVC `.lib`).
    Lib,
    /// Android APK (ZIP).
    Apk,
}

impl FileType {
    /// Human-readable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Utf8 => "utf-8",
            Self::Utf16Le => "utf-16le",
            Self::Utf16Be => "utf-16be",
            Self::Jpeg => "jpeg",
            Self::Png => "png",
            Self::Avif => "avif",
            Self::Elf => "elf",
            Self::Dll => "dll",
            Self::Lib => "lib",
            Self::Apk => "apk",
        }
    }
}

impl core::fmt::Display for FileType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Magic-byte detector that consults individual format parsers.
#[derive(Debug, Default, Clone, Copy)]
pub struct FileTypeDetector;

impl FileTypeDetector {
    /// Creates a new detector.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detects the type from a byte buffer prefix.
    #[must_use]
    pub fn detect(self, data: &[u8]) -> FileType {
        // Binary containers first (strict magic).
        if JpegParser::detect(data) {
            return FileType::Jpeg;
        }
        if PngParser::detect(data) {
            return FileType::Png;
        }
        if AvifParser::detect(data) {
            return FileType::Avif;
        }
        if ElfParser::detect(data) {
            return FileType::Elf;
        }
        if DllParser::detect(data) {
            return FileType::Dll;
        }
        if LibParser::detect(data) {
            return FileType::Lib;
        }
        if ApkParser::detect(data) {
            return FileType::Apk;
        }
        if let Some(cp) = Windows12xxParser::detect_from_bom(data) {
            return match cp.id() {
                1200 => FileType::Utf16Le,
                1201 => FileType::Utf16Be,
                _ => FileType::Unknown,
            };
        }
        if Utf8Parser::has_bom(data) {
            return FileType::Utf8;
        }
        if Utf8Parser::detect(data) {
            return FileType::Utf8;
        }
        FileType::Unknown
    }

    /// Detects from bytes, falling back to a file-name extension hint.
    #[must_use]
    pub fn detect_with_name(self, data: &[u8], name: Option<&str>) -> FileType {
        let from_bytes = self.detect(data);
        if from_bytes != FileType::Unknown {
            return from_bytes;
        }
        name.and_then(type_from_extension)
            .unwrap_or(FileType::Unknown)
    }
}

fn type_from_extension(name: &str) -> Option<FileType> {
    let ext = name.rsplit('.').next()?.to_ascii_lowercase();
    match ext.as_str() {
        "jpg" | "jpeg" | "jpe" | "jfif" => Some(FileType::Jpeg),
        "png" => Some(FileType::Png),
        "avif" => Some(FileType::Avif),
        "elf" | "o" | "so" => Some(FileType::Elf),
        "dll" | "exe" | "sys" | "ocx" => Some(FileType::Dll),
        "lib" | "a" => Some(FileType::Lib),
        "apk" => Some(FileType::Apk),
        _ => Some(FileType::Utf8),
    }
}
