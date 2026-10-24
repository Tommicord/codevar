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

//! The renderer-side contract of the text pass: glyph vertices, their
//! coverage atlas and the shared frame the UI hands to the compositor.
//!
//! The backend is deliberately blind to fonts, shaping and widgets. The
//! UI side turns a run of text into two neutral products — a triangle
//! list of [`TextVertex`]es and a [`TextAtlas`] of coverage — and pushes
//! the pair, with the resolved clip, into a [`TextFrame`]. This module
//! only defines that hand-over; uploading the atlases and recording the
//! `text.vert`/`text.frag` draw calls is the text pipe's job.
//!
//! The vertex layout matches the `text` shader pair compiled by
//! `codevar-kuitui`: location 0 is the position in logical pixels,
//! location 1 the normalized atlas coordinate and location 2 the color.
//! The atlas is single-channel: its red channel is the glyph coverage the
//! fragment shader multiplies into the run color's alpha.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

use codevar_kuitui::geometry::Rect;

/// A position, atlas coordinate and color of one glyph vertex.
///
/// The layout is the input of `text.vert`: a `vec2` at location 0, a
/// `vec2` at location 1 and a `vec4` at location 2, packed without
/// padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TextVertex {
    /// Position in logical pixels, top-left origin.
    pub position: [f32; 2],
    /// Normalized atlas coordinates of the sample.
    pub uv: [f32; 2],
    /// Straight (non-premultiplied) sRGB color of the run.
    pub color: [f32; 4],
}

/// A single-channel coverage bitmap sampled by a [`TextRun`].
///
/// The atlas is validated on construction: `pixels` holds exactly
/// `width * height` bytes, one byte of coverage per pixel. The bytes are
/// row-major, with the first pixel at the top-left corner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextAtlas {
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// One byte of coverage per pixel, row-major.
    pixels: Vec<u8>,
}

impl TextAtlas {
    /// Builds an atlas from its dimensions and coverage bytes.
    ///
    /// # Errors
    ///
    /// [`TextError::EmptyAtlas`] when either dimension is zero, and
    /// [`TextError::AtlasSizeMismatch`] when `pixels` does not hold
    /// `width * height` bytes.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, TextError> {
        if width == 0 || height == 0 {
            return Err(TextError::EmptyAtlas { width, height });
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .ok_or(TextError::AtlasSizeMismatch {
                width,
                height,
                actual: pixels.len(),
            })?;
        if pixels.len() != expected {
            return Err(TextError::AtlasSizeMismatch {
                width,
                height,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// The width of the atlas, in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The height of the atlas, in pixels.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The coverage bytes, one per pixel, row-major.
    #[inline]
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

/// One shaped run of text: its glyph vertices, the coverage atlas they
/// sample and the clip that applied when the run was recorded.
///
/// Every run owns its atlas because a font packer hands back a fresh one
/// per layout. A consumer therefore uploads one texture per run; the
/// alternative — a shared atlas — is a shaping-side decision that does not
/// change this contract.
#[derive(Debug, Clone, PartialEq)]
pub struct TextRun {
    /// Six vertices per glyph quad, in paint order.
    pub vertices: Vec<TextVertex>,
    /// The coverage atlas `vertices` sample.
    pub atlas: TextAtlas,
    /// Clip resolved by the UI when the run was recorded; `None` means
    /// the whole surface.
    pub clip: Option<Rect>,
}

impl TextRun {
    /// Creates a run from its vertices, atlas and clip.
    #[must_use]
    pub const fn new(vertices: Vec<TextVertex>, atlas: TextAtlas, clip: Option<Rect>) -> Self {
        Self {
            vertices,
            atlas,
            clip,
        }
    }

    /// Whether the run holds no vertex and would draw nothing.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

/// A shared, single-threaded hand-over of shaped runs.
///
/// The UI records into the handle and the text pipe drains it, exactly
/// like [`UiFrame`](crate::ui::UiFrame) does for the polygon display list.
/// The handle is `Rc`-based because a run is produced on the UI thread and
/// is not `Send`.
#[derive(Debug, Clone, Default)]
pub struct TextFrame {
    /// The shared runs, in paint order.
    runs: Rc<RefCell<Vec<TextRun>>>,
}

impl TextFrame {
    /// Creates an empty frame.
    #[must_use]
    pub fn new() -> Self {
        Self {
            runs: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Runs `record` with exclusive access to the run list.
    ///
    /// The UI thread records one frame per call; the renderer borrows the
    /// same list from its pipe, and the two borrows never overlap because
    /// the frame is fully recorded before it is composited.
    pub fn with<R>(&self, record: impl FnOnce(&mut Vec<TextRun>) -> R) -> R {
        record(&mut self.runs.borrow_mut())
    }

    /// Whether the frame holds no run.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.runs.borrow().is_empty()
    }

    /// Empties the frame, keeping its allocation.
    pub fn clear(&self) {
        self.runs.borrow_mut().clear();
    }
}

/// Why a [`TextAtlas`] could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextError {
    /// A dimension is zero, so there is no pixel to sample.
    EmptyAtlas {
        /// The requested width.
        width: u32,
        /// The requested height.
        height: u32,
    },
    /// The coverage buffer does not match `width * height`.
    AtlasSizeMismatch {
        /// The requested width.
        width: u32,
        /// The requested height.
        height: u32,
        /// The number of bytes handed in.
        actual: usize,
    },
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAtlas { width, height } => {
                write!(f, "text atlas is empty ({width}x{height})")
            }
            Self::AtlasSizeMismatch {
                width,
                height,
                actual,
            } => write!(
                f,
                "text atlas needs {} bytes for {width}x{height} but got {actual}",
                *width as usize * *height as usize
            ),
        }
    }
}

impl core::error::Error for TextError {}

#[cfg(test)]
mod tests {
    //! The hand-over contract: atlas validation, frame ownership and clip
    //! carry-over.

    use super::*;
    use alloc::vec;
    use codevar_kuitui::geometry::Offset;

    fn atlas(width: u32, height: u32) -> TextAtlas {
        TextAtlas::new(width, height, vec![0; (width * height) as usize]).unwrap()
    }

    fn run(clip: Option<Rect>) -> TextRun {
        TextRun::new(
            vec![TextVertex {
                position: [1.0, 2.0],
                uv: [0.5, 0.25],
                color: [1.0, 1.0, 1.0, 1.0],
            }],
            atlas(1, 1),
            clip,
        )
    }

    #[test]
    fn an_atlas_accepts_one_byte_per_pixel() {
        let atlas = TextAtlas::new(4, 2, vec![7; 8]).unwrap();
        assert_eq!(atlas.width(), 4);
        assert_eq!(atlas.height(), 2);
        assert_eq!(atlas.pixels(), &[7; 8]);
    }

    #[test]
    fn an_atlas_rejects_a_short_cover_buffer() {
        assert_eq!(
            TextAtlas::new(4, 2, vec![0; 7]),
            Err(TextError::AtlasSizeMismatch {
                width: 4,
                height: 2,
                actual: 7,
            })
        );
    }

    #[test]
    fn an_atlas_rejects_a_long_cover_buffer() {
        assert_eq!(
            TextAtlas::new(2, 2, vec![0; 5]),
            Err(TextError::AtlasSizeMismatch {
                width: 2,
                height: 2,
                actual: 5,
            })
        );
    }

    #[test]
    fn an_atlas_rejects_a_zero_dimension() {
        assert_eq!(
            TextAtlas::new(0, 4, Vec::new()),
            Err(TextError::EmptyAtlas { width: 0, height: 4 })
        );
        assert_eq!(
            TextAtlas::new(4, 0, Vec::new()),
            Err(TextError::EmptyAtlas { width: 4, height: 0 })
        );
    }

    #[test]
    fn a_frame_hands_runs_over_and_empties_when_cleared() {
        let frame = TextFrame::new();
        assert!(frame.is_empty());
        frame.with(|runs| runs.push(run(None)));
        assert!(!frame.is_empty());
        frame.clear();
        assert!(frame.is_empty());
    }

    #[test]
    fn a_run_keeps_the_clip_it_was_recorded_with() {
        let clip = Rect::from_min_size(Offset::new(3.0, 4.0), 10.0, 20.0);
        let run = run(Some(clip));
        assert_eq!(run.clip, Some(clip));
        assert!(!run.is_empty());
    }

    #[test]
    fn the_vertex_layout_matches_the_text_shader() {
        assert_eq!(core::mem::size_of::<TextVertex>(), 32);
        assert_eq!(core::mem::offset_of!(TextVertex, position), 0);
        assert_eq!(core::mem::offset_of!(TextVertex, uv), 8);
        assert_eq!(core::mem::offset_of!(TextVertex, color), 16);
    }
}
