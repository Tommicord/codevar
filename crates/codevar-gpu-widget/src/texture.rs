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

//! CPU-side texture registry shared by widgets and the render pipe.
//!
//! Widgets describe *what* to draw with a [`TextureId`]; the registry
//! keeps the pixels behind that id and tracks which ids the GPU still
//! has to upload or destroy. The render pipe drains the pending lists
//! while it records a frame, so neither side needs to know about the
//! other's timing.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use core::fmt;

use crate::error::WidgetError;

/// Largest number of live textures the registry accepts.
const MAX_TEXTURES: usize = 4096;

/// Handle to a texture registered with a [`TextureRegistry`].
///
/// Ids are stable for the lifetime of the registry and are never
/// reused, so a stale id fails lookup instead of aliasing a new image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TextureId(u32);

impl TextureId {
    /// Wraps a raw id (used by tests and backends).
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// The raw numeric id.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for TextureId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "texture {}", self.0)
    }
}

/// Pixel format of a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// 8 bits per channel RGBA, straight alpha.
    Rgba8,
    /// Single alpha channel (glyph masks, soft masks).
    A8,
}

impl TextureFormat {
    /// Bytes one pixel occupies.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba8 => 4,
            Self::A8 => 1,
        }
    }
}

/// Immutable pixel payload of one texture.
///
/// The pixels are reference counted: several widgets can share one
/// image without copying it, and [`TextureRegistry::register_shared`]
/// collapses them into a single upload.
#[derive(Debug, Clone)]
pub struct TextureData {
    pixels: Arc<[u8]>,
    width: u32,
    height: u32,
    format: TextureFormat,
}

impl TextureData {
    /// Wraps `pixels` as a `width` × `height` texture.
    ///
    /// # Errors
    ///
    /// * [`WidgetError::InvalidTextureData`] — the buffer length does
    ///   not match the extent and format.
    pub fn new(
        pixels: Arc<[u8]>,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<Self, WidgetError> {
        let expected = width as usize * height as usize * format.bytes_per_pixel();
        if pixels.len() != expected {
            return Err(WidgetError::InvalidTextureData {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            pixels,
            width,
            height,
            format,
        })
    }

    /// A solid texture of one pixel in `color`.
    #[must_use]
    pub fn solid(color: crate::geometry::Color) -> Self {
        let bytes = [
            (color.r * 255.0) as u8,
            (color.g * 255.0) as u8,
            (color.b * 255.0) as u8,
            (color.a * 255.0) as u8,
        ];
        Self {
            pixels: Arc::from(bytes.as_slice()),
            width: 1,
            height: 1,
            format: TextureFormat::Rgba8,
        }
    }

    /// The pixel bytes.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Pixel format.
    #[must_use]
    pub const fn format(&self) -> TextureFormat {
        self.format
    }

    /// Whether two handles point at the same pixel allocation.
    #[must_use]
    pub fn same_allocation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.pixels, &other.pixels)
    }
}

/// Bookkeeping of one registered texture.
#[derive(Debug, Clone)]
struct TextureEntry {
    data: TextureData,
    uploaded: bool,
    destroyed: bool,
}

/// Registry mapping [`TextureId`]s to pixels plus upload/destroy
/// bookkeeping for the render pipe.
#[derive(Debug)]
pub struct TextureRegistry {
    entries: BTreeMap<TextureId, TextureEntry>,
    next_id: u32,
    pending_uploads: Vec<TextureId>,
    pending_destroys: Vec<TextureId>,
}

impl TextureRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            next_id: 1,
            pending_uploads: Vec::new(),
            pending_destroys: Vec::new(),
        }
    }

    /// Registers `data` and returns its id.
    ///
    /// The id is queued for upload on the next recorded frame.
    ///
    /// # Errors
    ///
    /// * [`WidgetError::LimitExceeded`] — the registry already holds
    ///   [`MAX_TEXTURES`] textures.
    pub fn register(&mut self, data: TextureData) -> Result<TextureId, WidgetError> {
        if self.entries.len() >= MAX_TEXTURES {
            return Err(WidgetError::LimitExceeded {
                what: "textures",
                limit: MAX_TEXTURES,
            });
        }
        let id = TextureId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.entries.insert(
            id,
            TextureEntry {
                data,
                uploaded: false,
                destroyed: false,
            },
        );
        self.pending_uploads.push(id);
        Ok(id)
    }

    /// Registers `data`, reusing the id of an already registered
    /// texture that shares its pixel allocation.
    ///
    /// This is the deduplication path for widgets built from one
    /// shared [`Arc`] payload.
    ///
    /// # Errors
    ///
    /// * [`WidgetError::LimitExceeded`] — the registry is full and the
    ///   payload was not shared yet.
    pub fn register_shared(&mut self, data: TextureData) -> Result<TextureId, WidgetError> {
        for (id, entry) in &self.entries {
            if !entry.destroyed && entry.data.same_allocation(&data) {
                return Ok(*id);
            }
        }
        self.register(data)
    }

    /// The pixels behind `id`.
    #[must_use]
    pub fn get(&self, id: TextureId) -> Option<&TextureData> {
        self.entries.get(&id).filter(|entry| !entry.destroyed).map(|entry| &entry.data)
    }

    /// Whether `id` is registered.
    #[must_use]
    pub fn contains(&self, id: TextureId) -> bool {
        self.get(id).is_some()
    }

    /// Number of live textures.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| !entry.destroyed)
            .count()
    }

    /// Whether no texture is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops `id`, queueing its GPU resources for destruction.
    ///
    /// Returns the pixels that were registered, or `None` when the id
    /// was unknown or already removed.
    pub fn remove(&mut self, id: TextureId) -> Option<TextureData> {
        let entry = self.entries.get_mut(&id)?;
        if entry.destroyed {
            return None;
        }
        entry.destroyed = true;
        if !entry.uploaded && let Some(index) = self.pending_uploads.iter().position(|it| *it == id) {
            self.pending_uploads.swap_remove(index);
        } else {
            self.pending_destroys.push(id);
        }
        let data = entry.data.clone();
        self.entries.remove(&id);
        Some(data)
    }

    /// Ids whose pixels still have to reach the GPU, in registration
    /// order; the render pipe drains this while recording.
    pub fn take_pending_uploads(&mut self) -> Vec<TextureId> {
        core::mem::take(&mut self.pending_uploads)
    }

    /// Marks `id` as uploaded so it is not scheduled again.
    pub fn mark_uploaded(&mut self, id: TextureId) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.uploaded = true;
        }
    }

    /// Ids whose GPU resources are no longer referenced by the tree
    /// and may be destroyed once the in-flight frame retires.
    pub fn take_pending_destroys(&mut self) -> Vec<TextureId> {
        core::mem::take(&mut self.pending_destroys)
    }

    /// Whether the upload list is empty.
    #[must_use]
    pub fn has_pending_uploads(&self) -> bool {
        !self.pending_uploads.is_empty()
    }
}

impl Default for TextureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn pixels(count: usize) -> Arc<[u8]> {
        Arc::from(vec![0u8; count].into_boxed_slice())
    }

    /// The length must match extent × format for both formats.
    #[test]
    fn register_validates_pixel_length() {
        let error = TextureData::new(pixels(3), 2, 2, TextureFormat::Rgba8);
        assert!(matches!(
            error,
            Err(WidgetError::InvalidTextureData {
                expected: 16,
                actual: 3
            })
        ));
        assert!(TextureData::new(pixels(4), 2, 2, TextureFormat::A8).is_ok());
    }

    /// Registered ids are unique, retrievable and removable exactly
    /// once; removal queues a destroy only for uploaded textures.
    #[test]
    fn register_lookup_and_remove() {
        let mut registry = TextureRegistry::new();
        let first = registry
            .register(TextureData::solid(crate::geometry::Color::WHITE))
            .expect("the first texture registers");
        let second = registry
            .register(TextureData::solid(crate::geometry::Color::BLACK))
            .expect("the second texture registers");
        assert_ne!(first, second);
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.take_pending_uploads(), [first, second]);

        registry.mark_uploaded(first);
        assert!(registry.remove(first).is_some());
        assert!(registry.remove(first).is_none());
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.take_pending_destroys(), [first]);
        assert!(registry.get(first).is_none());
        assert!(registry.contains(second));
    }

    /// Payloads that share their pixel allocation collapse into one
    /// id and one upload.
    #[test]
    fn shared_payloads_deduplicate() {
        let mut registry = TextureRegistry::new();
        let shared = TextureData::solid(crate::geometry::Color::WHITE);
        let first = registry.register_shared(shared.clone()).expect("first registration");
        let second = registry.register_shared(shared).expect("second registration shares the id");
        assert_eq!(first, second);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.take_pending_uploads().len(), 1);
    }

    /// Removing a texture that never reached the GPU cancels its
    /// pending upload instead of scheduling a destroy for it.
    #[test]
    fn removing_before_upload_cancels_the_upload() {
        let mut registry = TextureRegistry::new();
        let id = registry
            .register(TextureData::solid(crate::geometry::Color::WHITE))
            .expect("register");
        assert!(registry.remove(id).is_some());
        assert!(registry.take_pending_uploads().is_empty());
        assert!(registry.take_pending_destroys().is_empty());
    }
}
