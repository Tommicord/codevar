//! Copyright 2026 Codevar Project
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

//! Magic-byte validation for icons and sounds
//!
//! Validates that a byte buffer starts with the expected magic
//! bytes for supported icon and sound formats, mirroring the
//! validation performed by the C reference implementation before
//! passing blobs to the portal.

use alloc::string::ToString;
use core::fmt;

use crate::xdp_error::PortalError;

/// Supported icon formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IconFormat {
    /// Portable Network Graphics (PNG).
    Png,
    /// Microsoft Windows Icon (ICO).
    Ico,
    /// Scalable Vector Graphics (SVG) — detected by XML header.
    Svg,
}

/// Supported sound formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundFormat {
    /// Waveform Audio File Format (WAV/RIFF).
    Wav,
    /// Ogg Vorbis (OGG).
    Ogg,
}

/// Error returned when magic-byte validation fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidateError {
    /// Buffer is too short to contain the required magic bytes.
    TooShort {
        /// Minimum number of bytes required.
        required: usize,
        /// Actual buffer length.
        actual: usize,
    },
    /// Magic bytes did not match any supported format.
    UnknownFormat,
    /// The buffer claims to be SVG but does not start with an XML
    /// declaration or `<svg` element.
    InvalidSvg,
}

impl fmt::Display for ValidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { required, actual } => write!(
                f,
                "buffer too short: need at least {required} bytes, got {actual}"
            ),
            Self::UnknownFormat => write!(f, "unknown format: magic bytes do not match any supported format"),
            Self::InvalidSvg => {
                write!(f, "invalid SVG: missing XML declaration or <svg element")
            }
        }
    }
}

impl core::error::Error for ValidateError {}

impl From<ValidateError> for PortalError {
    fn from(error: ValidateError) -> Self {
        PortalError::InvalidArgument(error.to_string())
    }
}

/// Magic bytes for supported formats.
const PNG_MAGIC: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
const ICO_MAGIC: &[u8; 4] = b"\x00\x00\x01\x00";
const SVG_PREFIXES: &[&[u8]] = &[
    b"<?xml",
    b"<svg",
    b"\xef\xbb\xbf<?xml", // UTF-8 BOM + <?xml
    b"\xef\xbb\xbf<svg",  // UTF-8 BOM + <svg
];
const RIFF_MAGIC: &[u8; 4] = b"RIFF";
const WAVE_MAGIC: &[u8; 4] = b"WAVE";
const OGG_MAGIC: &[u8; 4] = b"OggS";

/// Validates that `data` starts with the magic bytes of a supported
/// icon format and returns the detected format.
///
/// # Errors
///
/// Returns [`ValidateError::TooShort`] when `data` is shorter than
/// the shortest magic sequence (4 bytes), and
/// [`ValidateError::UnknownFormat`] when the prefix does not match
/// PNG, ICO or SVG.
pub fn validate_icon(data: &[u8]) -> Result<IconFormat, ValidateError> {
    if data.len() < 4 {
        return Err(ValidateError::TooShort {
            required: 4,
            actual: data.len(),
        });
    }

    if data.starts_with(PNG_MAGIC) {
        return Ok(IconFormat::Png);
    }
    if data.starts_with(ICO_MAGIC) {
        return Ok(IconFormat::Ico);
    }
    for prefix in SVG_PREFIXES {
        if data.starts_with(prefix) {
            return Ok(IconFormat::Svg);
        }
    }

    Err(ValidateError::UnknownFormat)
}

/// Validates that `data` starts with the magic bytes of a supported
/// sound format and returns the detected format.
///
/// # Errors
///
/// Returns [`ValidateError::TooShort`] when `data` is shorter than
/// 12 bytes (required to distinguish RIFF/WAVE from other RIFF
/// containers), and [`ValidateError::UnknownFormat`] when the prefix
/// does not match WAV or OGG.
pub fn validate_sound(data: &[u8]) -> Result<SoundFormat, ValidateError> {
    if data.len() < 12 {
        return Err(ValidateError::TooShort {
            required: 12,
            actual: data.len(),
        });
    }
    if data.starts_with(RIFF_MAGIC) && &data[8..12] == WAVE_MAGIC {
        return Ok(SoundFormat::Wav);
    }
    if data.starts_with(OGG_MAGIC) {
        return Ok(SoundFormat::Ogg);
    }

    Err(ValidateError::UnknownFormat)
}

/// Validates that `data` is a supported icon or sound, returning the
/// detected format kind.
///
/// # Errors
///
/// Returns an error when `data` matches neither an icon nor a sound
/// format.
pub fn validate_icon_or_sound(data: &[u8]) -> Result<IconOrSound, ValidateError> {
    if let Ok(format) = validate_icon(data) {
        return Ok(IconOrSound::Icon(format));
    }
    if let Ok(format) = validate_sound(data) {
        return Ok(IconOrSound::Sound(format));
    }
    Err(ValidateError::UnknownFormat)
}

/// A successfully validated icon or sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IconOrSound {
    /// A validated icon.
    Icon(IconFormat),
    /// A validated sound.
    Sound(SoundFormat),
}

impl IconOrSound {
    /// Returns the MIME type associated with the format.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Icon(IconFormat::Png) => "image/png",
            Self::Icon(IconFormat::Ico) => "image/vnd.microsoft.icon",
            Self::Icon(IconFormat::Svg) => "image/svg+xml",
            Self::Sound(SoundFormat::Wav) => "audio/wav",
            Self::Sound(SoundFormat::Ogg) => "audio/ogg",
        }
    }

    /// Returns a human-readable description of the format.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Icon(IconFormat::Png) => "PNG",
            Self::Icon(IconFormat::Ico) => "ICO",
            Self::Icon(IconFormat::Svg) => "SVG",
            Self::Sound(SoundFormat::Wav) => "WAV",
            Self::Sound(SoundFormat::Ogg) => "OGG",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every unwrap() below operates on test-constructed data.

    #[test]
    fn validates_png_magic_bytes() {
        let png = b"\x89PNG\r\n\x1a\nrest of file";
        assert_eq!(validate_icon(png).unwrap(), IconFormat::Png);
    }

    #[test]
    fn validates_ico_magic_bytes() {
        let ico = b"\x00\x00\x01\x00rest";
        assert_eq!(validate_icon(ico).unwrap(), IconFormat::Ico);
    }

    #[test]
    fn validates_svg_xml_declaration() {
        let svg = b"<?xml version=\"1.0\"?><svg></svg>";
        assert_eq!(validate_icon(svg).unwrap(), IconFormat::Svg);
    }

    #[test]
    fn validates_svg_direct_element() {
        let svg = b"<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>";
        assert_eq!(validate_icon(svg).unwrap(), IconFormat::Svg);
    }

    #[test]
    fn validates_svg_with_bom() {
        let svg = b"\xef\xbb\xbf<?xml version=\"1.0\"?><svg></svg>";
        assert_eq!(validate_icon(svg).unwrap(), IconFormat::Svg);
    }

    #[test]
    fn rejects_invalid_svg() {
        let not_svg = b"<html></html>";
        assert!(matches!(
            validate_icon(not_svg),
            Err(ValidateError::UnknownFormat)
        ));
    }

    #[test]
    fn validates_wav_riff_wave() {
        let wav = b"RIFF\x24\x00\x00\x00WAVEfmt ";
        assert_eq!(validate_sound(wav).unwrap(), SoundFormat::Wav);
    }

    #[test]
    fn validates_ogg_magic_bytes() {
        let ogg = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(validate_sound(ogg).unwrap(), SoundFormat::Ogg);
    }

    #[test]
    fn rejects_riff_without_wave() {
        let riff = b"RIFF\x24\x00\x00\x00AVI ";
        assert!(matches!(validate_sound(riff), Err(ValidateError::UnknownFormat)));
    }

    #[test]
    fn rejects_too_short_buffers() {
        assert!(matches!(
            validate_icon(b""),
            Err(ValidateError::TooShort {
                required: 4,
                actual: 0
            })
        ));
        assert!(matches!(
            validate_icon(b"PNG"),
            Err(ValidateError::TooShort {
                required: 4,
                actual: 3
            })
        ));
        assert!(matches!(
            validate_sound(b""),
            Err(ValidateError::TooShort {
                required: 12,
                actual: 0
            })
        ));
        assert!(matches!(
            validate_sound(b"RIFF"),
            Err(ValidateError::TooShort {
                required: 12,
                actual: 4
            })
        ));
    }

    #[test]
    fn validates_icon_or_sound_union() {
        let png = b"\x89PNG\r\n\x1a\n";
        assert_eq!(
            validate_icon_or_sound(png).unwrap(),
            IconOrSound::Icon(IconFormat::Png)
        );

        let wav = b"RIFF\x24\x00\x00\x00WAVEfmt ";
        assert_eq!(
            validate_icon_or_sound(wav).unwrap(),
            IconOrSound::Sound(SoundFormat::Wav)
        );
    }

    #[test]
    fn converts_to_portal_error() {
        let error: PortalError = ValidateError::UnknownFormat.into();
        assert!(matches!(error, PortalError::InvalidArgument(_)));
    }

    #[test]
    fn mime_types_are_correct() {
        assert_eq!(IconOrSound::Icon(IconFormat::Png).mime_type(), "image/png");
        assert_eq!(
            IconOrSound::Icon(IconFormat::Ico).mime_type(),
            "image/vnd.microsoft.icon"
        );
        assert_eq!(IconOrSound::Icon(IconFormat::Svg).mime_type(), "image/svg+xml");
        assert_eq!(IconOrSound::Sound(SoundFormat::Wav).mime_type(), "audio/wav");
        assert_eq!(IconOrSound::Sound(SoundFormat::Ogg).mime_type(), "audio/ogg");
    }

    #[test]
    fn descriptions_are_correct() {
        assert_eq!(IconOrSound::Icon(IconFormat::Png).description(), "PNG");
        assert_eq!(IconOrSound::Icon(IconFormat::Ico).description(), "ICO");
        assert_eq!(IconOrSound::Icon(IconFormat::Svg).description(), "SVG");
        assert_eq!(IconOrSound::Sound(SoundFormat::Wav).description(), "WAV");
        assert_eq!(IconOrSound::Sound(SoundFormat::Ogg).description(), "OGG");
    }
}
