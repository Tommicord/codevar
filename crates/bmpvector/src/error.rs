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
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Error values produced by the tracing pipeline.
//!
//! The crate deliberately keeps its error surface allocation-free: every
//! variant carries either a scalar value or a `'static` field name. A caller
//! can therefore match on the exact failure without formatting a string, and
//! the crate remains usable on targets without a global allocator for the
//! error path itself.

use core::fmt;

/// Alias for results returned by this crate.
pub type Result<T> = core::result::Result<T, Error>;

/// A failure in the bitmap-to-vector pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The input image has a zero width or a zero height.
    EmptyImage,
    /// `width * height` does not fit in a `usize` on this target.
    ImageTooLarge {
        /// Declared image width.
        width: u32,
        /// Declared image height.
        height: u32,
    },
    /// The pixel buffer length does not match the declared dimensions.
    PixelCountMismatch {
        /// Number of pixels implied by the dimensions.
        expected: usize,
        /// Number of pixels actually supplied.
        actual: usize,
    },
    /// A byte buffer cannot be interpreted as the requested pixel format.
    BufferLengthMismatch {
        /// Number of bytes required by the declared dimensions and format.
        expected: usize,
        /// Number of bytes actually supplied.
        actual: usize,
    },
    /// A configuration value is outside its permitted range.
    InvalidConfig {
        /// Name of the offending field.
        field: &'static str,
    },
    /// The palette strategy selected no colors at all.
    EmptyPalette,
    /// A palette contains more entries than the index representation supports.
    PaletteTooLarge {
        /// Number of entries supplied or discovered.
        entries: usize,
    },
    /// A byte channel used as a palette entry was rejected by validation.
    InvalidPaletteEntry {
        /// Position of the rejected entry in the palette.
        index: usize,
    },
    /// A component mask was empty when contour extraction started.
    EmptyComponent {
        /// Compact component identifier.
        component: usize,
    },
    /// A component describes a region that does not exist.
    ///
    /// The pixel list and the declared area disagree, or a pixel lies outside
    /// the declared bounds. Tracing such a component would silently produce
    /// geometry that does not match the source image.
    InconsistentComponent {
        /// Compact component identifier.
        component: usize,
    },
    /// Contour following could not find a successor edge at a lattice vertex.
    BrokenContour {
        /// Compact component identifier.
        component: usize,
        /// Lattice vertex where the walk stopped.
        vertex: (u32, u32),
    },
    /// The configuration requested a palette strategy that needs an image
    /// whose alpha channel is already resolved, but transparent pixels remain.
    UnresolvedTransparency,
    /// An intermediate size computation overflowed or a numerical lookup failed.
    NumericalFailure {
        /// Name of the operation that could not be completed.
        operation: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage => f.write_str("bitmap dimensions must both be non-zero"),
            Self::ImageTooLarge { width, height } => {
                write!(f, "image {width}x{height} does not fit in memory on this target")
            }
            Self::PixelCountMismatch { expected, actual } => {
                write!(f, "expected {expected} pixels, received {actual}")
            }
            Self::BufferLengthMismatch { expected, actual } => {
                write!(f, "expected {expected} bytes, received {actual}")
            }
            Self::InvalidConfig { field } => write!(f, "invalid configuration value for `{field}`"),
            Self::EmptyPalette => f.write_str("the palette strategy produced no colors"),
            Self::PaletteTooLarge { entries } => {
                write!(f, "palette of {entries} entries exceeds the supported maximum")
            }
            Self::InvalidPaletteEntry { index } => {
                write!(f, "palette entry {index} could not be represented as a color")
            }
            Self::EmptyComponent { component } => {
                write!(f, "component {component} has no pixels to trace")
            }
            Self::InconsistentComponent { component } => {
                write!(f, "component {component} describes an inconsistent region")
            }
            Self::BrokenContour { component, vertex } => write!(
                f,
                "component {component} has an open contour at lattice vertex ({}, {})",
                vertex.0, vertex.1
            ),
            Self::UnresolvedTransparency => {
                f.write_str("transparent pixels remain after background compositing")
            }
            Self::NumericalFailure { operation } => {
                write!(f, "numerical operation `{operation}` could not be completed")
            }
        }
    }
}

impl core::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_specific_for_each_variant() {
        let messages = [
            Error::EmptyImage.to_string(),
            Error::InvalidConfig { field: "max_colors" }.to_string(),
            Error::BrokenContour {
                component: 3,
                vertex: (7, 9),
            }
            .to_string(),
        ];
        assert!(messages[0].contains("non-zero"));
        assert!(messages[1].contains("max_colors"));
        assert!(messages[2].contains("component 3"));
        assert!(messages[2].contains("(7, 9)"));
    }
}
