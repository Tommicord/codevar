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

//! The renderer-side text contract: glyph vertices, an on-demand glyph
//! atlas and the shared frame the UI hands to the compositor.
//!
//! The backend is deliberately blind to fonts, shaping and widgets. The
//! UI side turns text into two neutral products — indexed glyph quads and
//! coverage bitmaps — and asks the [`TextAtlas`] for a place to put each
//! bitmap. This module only defines that hand-over; uploading dirty regions
//! and recording the `text.vert`/`text.frag` draw calls is the text pipe's
//! job.
//!
//! # Why the atlas is demand-driven
//!
//! A Unicode font has tens of thousands of glyphs, and a run of text
//! touches a few dozen of them. Preloading a codepoint range, or one
//! texture per glyph, spends memory on glyphs that are never drawn. The
//! atlas therefore caches only the bitmaps the UI actually rasterizes,
//! keyed by an opaque [`GlyphKey`] the UI derives from its font face,
//! glyph id, pixel size and subpixel bin.
//!
//! The vertex layout matches the `text` shader pair compiled by
//! `codevar-kuitui`: location 0 is the position in logical pixels,
//! location 1 the normalized atlas coordinate and location 2 the color.
//! The fragment shader samples the atlas's red channel as coverage.

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::{Cell, Ref, RefCell};
use core::fmt;
use core::ops::Range;

use crate::dirty::DirtySignal;
use crate::mesh::{MeshBuilder, MeshRun};

/// One texel of transparent padding kept between two glyphs.
///
/// Linear sampling reads a half-texel outside the glyph quad, so without
/// the gap the edge of one glyph could blend with its neighbor.
const GLYPH_PADDING: u32 = 0x1;

/// Largest side a page may have.
///
/// The bound keeps `page_size * page_size` bytes in a range every target
/// can address and matches the texture limits of the GPUs Codevar targets.
const MAX_PAGE_SIZE: u32 = 0x4000;

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

/// An opaque key identifying a rasterized glyph in the atlas.
///
/// The renderer does not interpret the key; the UI packs whatever
/// identifies a bitmap in its font stack into the 64 bits — typically the
/// face id, the glyph id, the raster size and the subpixel bin — so two
/// sizes of the same glyph, or two faces that share a glyph id, do not
/// collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlyphKey(u64);

impl GlyphKey {
    /// Wraps a caller-defined identifier.
    #[inline]
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The identifier the key was built from.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Sizing and storage policy of a [`TextAtlas`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtlasConfig {
    /// Side of every square page, in pixels.
    pub page_size: u32,
    /// Maximum number of pages the atlas may allocate.
    pub max_pages: usize,
    /// Whether the gray pages live in device storage instead of a host mirror.
    ///
    /// A device-backed page has no `page_pixels` bytes: the pipe owns a
    /// linear buffer per page, the caller rasterizes into it through
    /// [`TextAtlas::reserve`] and the page image is refreshed from that
    /// buffer. The color atlas always keeps its host mirror, whatever this
    /// says.
    pub device_backed: bool,
}

impl AtlasConfig {
    /// A 1024 × 1024 page size, at most four pages and host-covered pages.
    ///
    /// One full page holds about a megabyte of coverage, enough for several
    /// thousand UI glyphs; four pages cap the cache at 4 MiB.
    pub const DEFAULT: Self = Self {
        page_size: 0x800,
        max_pages: 0x4,
        device_backed: false,
    };
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A rectangle the atlas wrote since the last upload.
///
/// `page` indexes [`TextAtlas::page_pixels`]; `x` and `y` are the top-left
/// texel and `width`/`height` the extent, all in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyRect {
    /// Page the rectangle belongs to.
    pub page: u32,
    /// Left texel.
    pub x: u32,
    /// Top texel.
    pub y: u32,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
}

/// Where a glyph lives inside the atlas.
///
/// The atlas coordinate of a glyph is its pixel rectangle divided by the
/// page size. A slot is valid until the page it lives on is evicted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphSlot {
    /// Index of the page holding the glyph.
    pub page: u32,
    /// Inclusive lower atlas coordinate.
    pub uv_min: [f32; 2],
    /// Exclusive upper atlas coordinate.
    pub uv_max: [f32; 2],
    /// Bitmap width in pixels.
    pub width: u32,
    /// Bitmap height in pixels.
    pub height: u32,
}

/// Where one glyph's coverage lives inside a page's device storage.
///
/// The addressing is the rasterizer's own: the work item `(px, py)` of a
/// bottom-up glyph writes `offset + py * pitch + px`, so `offset` is the
/// byte address of the glyph's bottom row and `pitch` the distance to the
/// row above it. Because page storage is stored row-major from its top
/// edge, a page slot always has a negative pitch.
///
/// The font stack's raster destination is the same contract, so the two
/// structures map field for field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceSlot {
    /// Page whose device storage holds the bytes.
    pub page: u32,
    /// Glyph width in texels.
    pub width: usize,
    /// Glyph height in texels.
    pub height: usize,
    /// Byte address of the row the rasterizer writes first.
    pub offset: usize,
    /// Bytes from one stored row to the next, negative when the storage
    /// runs top-down.
    pub pitch: isize,
}

impl DeviceSlot {
    /// The byte address of the `row`-th row counted from the top of the
    /// glyph, which is the order a host bitmap arrives in.
    ///
    /// Returns `None` for a row outside the glyph or an address that does
    /// not fit the platform's offsets.
    #[inline]
    #[must_use]
    pub fn row_address(&self, row: usize) -> Option<usize> {
        let from_bottom = self.height.checked_sub(1)?.checked_sub(row)?;
        let address = i64::try_from(self.offset).ok()?
            + i64::try_from(from_bottom)
                .ok()?
                .checked_mul(i64::try_from(self.pitch).ok()?)?;
        usize::try_from(address).ok()
    }

    /// The smallest byte range that contains the slot.
    ///
    /// `None` when the slot has no rows or its addressing overflows, which
    /// is how a destination that reaches past the page storage is refused
    /// before anything writes it.
    #[must_use]
    pub fn bytes(&self) -> Option<Range<usize>> {
        let first = self.row_address(0)?;
        let last = self.row_address(self.height.checked_sub(1)?)?;
        let end = first.max(last).checked_add(self.width)?;
        Some(first.min(last)..end)
    }
}

/// A glyph and the device bytes that will hold its coverage.
///
/// [`TextAtlas::reserve`] returns it once the atlas has placed the glyph and
/// registered the key; writing the coverage is the caller's step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reservation {
    /// Where the glyph lives in the atlas.
    pub glyph: GlyphSlot,
    /// The bytes of its coverage in the page's storage, or `None` when the
    /// key was already resident and nothing has to be written.
    pub coverage: Option<DeviceSlot>,
}

/// Coverage a host rasterizer produced for a device-backed page.
///
/// A device page keeps no host mirror, so bytes that were rasterized on the
/// CPU wait here — usually none do — until the pipe copies them into the
/// page's storage on the next upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingCoverage {
    /// Where the bytes belong.
    pub slot: DeviceSlot,
    /// Row-major, top-down coverage, `width * height` bytes.
    pub bitmap: Vec<u8>,
}

/// Per-frame counters of a [`TextAtlas`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AtlasStats {
    /// Lookups that found a cached glyph.
    pub hits: u64,
    /// Lookups that had to place a new glyph.
    pub misses: u64,
    /// Pages cleared to make room.
    pub evictions: u64,
    /// Placements skipped because every page was pinned.
    pub blocked: u64,
    /// Number of glyphs currently resident.
    pub glyphs: u32,
    /// Number of pages currently allocated.
    pub pages: u32,
    /// Bytes of coverage the resident glyphs occupy.
    pub packed_bytes: u64,
}

/// A shelf of one page: a row of glyphs sharing a height.
#[derive(Debug)]
struct Shelf {
    /// Top texel of the shelf.
    y: u32,
    /// Height reserved for the shelf, including the padding row.
    height: u32,
    /// First free texel after the last glyph on the shelf.
    cursor_x: u32,
}

/// One square page of single-channel coverage.
#[derive(Debug)]
struct AtlasPage {
    /// Row-major coverage, `page_size * page_size` bytes, empty for a page
    /// whose coverage lives in device storage.
    pixels: Vec<u8>,
    /// Shelves allocated so far, top to bottom.
    shelves: Vec<Shelf>,
    /// First free texel below the last shelf.
    cursor_y: u32,
    /// Logical clock value of the last use.
    last_used: u64,
    /// Whether a glyph of the frame being shaped lives on this page.
    pinned: bool,
    /// Keys whose slots point at this page, for O(keys) eviction.
    keys: Vec<GlyphKey>,
}

impl AtlasPage {
    /// Creates a page, zeroed unless its coverage lives on the device.
    ///
    /// The caller validates `page_size` when it builds the
    /// [`AtlasConfig`], so the product fits `usize`.
    fn new(page_size: u32, device_backed: bool) -> Self {
        let len = if device_backed {
            0
        } else {
            page_size as usize * page_size as usize
        };
        Self {
            pixels: alloc::vec![0; len],
            shelves: Vec::new(),
            cursor_y: 0,
            last_used: 0,
            pinned: false,
            keys: Vec::new(),
        }
    }

    /// Reserves a `width` × `height` glyph, returning its top-left texel.
    ///
    /// The search is first-fit over the shelves, then a new shelf below
    /// the last one. Nothing is mutated when the page is full.
    fn allocate(&mut self, width: u32, height: u32, page_size: u32) -> Option<(u32, u32)> {
        let needed_width = width + GLYPH_PADDING;
        let needed_height = height + GLYPH_PADDING;
        for shelf in &mut self.shelves {
            if shelf.height >= needed_height && shelf.cursor_x + needed_width <= page_size {
                let x = shelf.cursor_x;
                shelf.cursor_x += needed_width;
                return Some((x, shelf.y));
            }
        }
        if self.cursor_y + needed_height > page_size {
            return None;
        }
        let y = self.cursor_y;
        self.cursor_y += needed_height;
        self.shelves.push(Shelf {
            y,
            height: needed_height,
            cursor_x: needed_width,
        });
        Some((0, y))
    }
}

/// A demand-driven, bounded glyph atlas.
///
/// The atlas starts with no page: the first bitmap allocates one. It grows
/// up to [`AtlasConfig::max_pages`], then reuses the least recently used
/// page that no glyph of the current frame references.
#[derive(Debug)]
pub struct TextAtlas {
    /// Page sizing policy.
    config: AtlasConfig,
    /// Allocated pages, in creation order.
    pages: Vec<AtlasPage>,
    /// Resident glyphs and where they live.
    glyphs: BTreeMap<GlyphKey, GlyphSlot>,
    /// Regions written since the last upload.
    dirty: Vec<DirtyRect>,
    /// Host-rasterized coverage waiting for a device page's storage.
    pending: Vec<PendingCoverage>,
    /// Monotonic use counter for the LRU order.
    clock: u64,
    /// Counters of the current frame.
    hits: u64,
    /// Counters of the current frame.
    misses: u64,
    /// Counters of the current frame.
    evictions: u64,
    /// Counters of the current frame.
    blocked: u64,
}

impl TextAtlas {
    /// Creates an atlas that has not allocated a page yet.
    ///
    /// # Errors
    ///
    /// [`TextError::ZeroPageSize`], [`TextError::ZeroPageCount`] or
    /// [`TextError::PageTooLarge`] when the configuration cannot describe
    /// an addressable page.
    pub fn new(config: AtlasConfig) -> Result<Self, TextError> {
        if config.page_size == 0 {
            return Err(TextError::ZeroPageSize);
        }
        if config.max_pages == 0 {
            return Err(TextError::ZeroPageCount);
        }
        if config.page_size > MAX_PAGE_SIZE
            || (config.page_size as usize)
                .checked_mul(config.page_size as usize)
                .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(TextError::PageTooLarge {
                page_size: config.page_size,
            });
        }
        Ok(Self {
            config,
            pages: Vec::new(),
            glyphs: BTreeMap::new(),
            dirty: Vec::new(),
            pending: Vec::new(),
            clock: 1,
            hits: 0,
            misses: 0,
            evictions: 0,
            blocked: 0,
        })
    }

    /// The sizing policy of the atlas.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> AtlasConfig {
        self.config
    }

    /// The number of pages allocated so far.
    #[inline]
    #[must_use]
    pub fn pages(&self) -> usize {
        self.pages.len()
    }

    /// The number of resident glyphs.
    #[inline]
    #[must_use]
    pub fn glyph_count(&self) -> usize {
        self.glyphs.len()
    }

    /// Bytes of coverage the resident glyphs occupy.
    ///
    /// This counts the glyph rectangles, not the reserved page area, so a
    /// sparse page reports what it actually costs on the GPU.
    #[must_use]
    pub fn packed_bytes(&self) -> usize {
        self.glyphs
            .values()
            .map(|slot| slot.width as usize * slot.height as usize)
            .sum()
    }

    /// The coverage of one page, row-major.
    ///
    /// `None` for an unknown page and for every page of a device-backed
    /// atlas, whose bytes only exist in the page's device storage.
    #[inline]
    #[must_use]
    pub fn page_pixels(&self, page: u32) -> Option<&[u8]> {
        if self.config.device_backed {
            return None;
        }
        self.pages
            .get(page as usize)
            .map(|page| page.pixels.as_slice())
    }

    /// The regions written since the last [`TextAtlas::clear_dirty`].
    #[inline]
    #[must_use]
    pub fn dirty_rects(&self) -> &[DirtyRect] {
        &self.dirty
    }

    /// Drops the dirty list once the pipe has uploaded it.
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    /// The host-rasterized coverage waiting for a device page's storage.
    #[inline]
    #[must_use]
    pub fn pending_coverage(&self) -> &[PendingCoverage] {
        &self.pending
    }

    /// Drops [`TextAtlas::pending_coverage`] once the pipe has copied it.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// The current frame's counters.
    #[must_use]
    pub fn stats(&self) -> AtlasStats {
        AtlasStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            blocked: self.blocked,
            glyphs: self.glyphs.len() as u32,
            pages: self.pages.len() as u32,
            packed_bytes: self.packed_bytes() as u64,
        }
    }

    /// Starts a frame: clears every page pin and the counters.
    ///
    /// Dirty rectangles survive; the pipe owns them until it uploads and
    /// calls [`TextAtlas::clear_dirty`].
    pub fn begin_frame(&mut self) {
        for page in &mut self.pages {
            page.pinned = false;
        }
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.blocked = 0;
    }

    /// Looks up a resident glyph and promotes its page.
    ///
    /// A hit pins the page for the rest of the frame, so eviction cannot
    /// free a glyph the frame has already referenced.
    pub fn get(&mut self, key: GlyphKey) -> Option<GlyphSlot> {
        let slot = self.glyphs.get(&key).copied()?;
        self.hits += 1;
        self.touch(slot.page);
        Some(slot)
    }

    /// Places a rasterized glyph, or returns its existing slot.
    ///
    /// `bitmap` is one byte of coverage per pixel, row-major, exactly
    /// `width * height` bytes. A zero `width` or `height` means the glyph
    /// has no bitmap (a space, for example): the call returns `Ok(None)`
    /// and allocates nothing.
    ///
    /// Re-inserting a resident key returns the old slot and ignores the
    /// new bitmap; a glyph never moves while a frame may reference it.
    ///
    /// On a device-backed atlas there is no host mirror to fill, so the
    /// bytes join [`TextAtlas::pending_coverage`] and the pipe copies them
    /// into the page's storage — the path a CPU rasterization takes when
    /// the device rasterizer is not available.
    ///
    /// # Errors
    ///
    /// * [`TextError::GlyphSizeMismatch`] when `bitmap` is the wrong size;
    /// * [`TextError::GlyphTooLarge`] when the glyph does not fit a page;
    /// * [`TextError::AtlasFull`] when every page is pinned and full.
    pub fn insert(
        &mut self,
        key: GlyphKey,
        width: u32,
        height: u32,
        bitmap: &[u8],
    ) -> Result<Option<GlyphSlot>, TextError> {
        if width == 0 || height == 0 {
            return Ok(None);
        }
        if let Some(slot) = self.glyphs.get(&key).copied() {
            self.hits += 1;
            self.touch(slot.page);
            return Ok(Some(slot));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .ok_or(TextError::GlyphTooLarge {
                width,
                height,
                page_size: self.config.page_size,
            })?;
        if bitmap.len() != expected {
            return Err(TextError::GlyphSizeMismatch {
                width,
                height,
                actual: bitmap.len(),
            });
        }
        if width > self.config.page_size || height > self.config.page_size {
            return Err(TextError::GlyphTooLarge {
                width,
                height,
                page_size: self.config.page_size,
            });
        }
        let (page, x, y) = self.place(width, height)?;
        self.misses += 1;
        self.coverage(page, x, y, width, height, bitmap);
        Ok(Some(self.register(key, page, x, y, width, height)))
    }

    /// Reserves the space of a glyph whose coverage the caller rasterizes
    /// into a device page, or returns its existing reservation.
    ///
    /// The atlas places the glyph, registers the key and marks the page
    /// dirty; the coverage itself is written by the caller through the
    /// returned [`DeviceSlot`], which addresses the page's device storage
    /// the way the rasterizer expects. Because the key is resident from the
    /// call on, a glyph whose rasterization failed has to be dropped again
    /// with [`TextAtlas::abandon`] or the atlas keeps drawing an
    /// unwritten rectangle.
    ///
    /// A zero `width` or `height` returns `Ok(None)` and allocates nothing,
    /// and a key that is already resident returns its slot with
    /// [`Reservation::coverage`] set to `None`: its bytes are where they
    /// have always been.
    ///
    /// # Errors
    ///
    /// * [`TextError::HostPages`] when the atlas has no device storage;
    /// * [`TextError::GlyphTooLarge`] when the glyph does not fit a page;
    /// * [`TextError::AtlasFull`] when every page is pinned and full.
    pub fn reserve(
        &mut self,
        key: GlyphKey,
        width: u32,
        height: u32,
    ) -> Result<Option<Reservation>, TextError> {
        if !self.config.device_backed {
            return Err(TextError::HostPages);
        }
        if width == 0 || height == 0 {
            return Ok(None);
        }
        if let Some(glyph) = self.glyphs.get(&key).copied() {
            self.hits += 1;
            self.touch(glyph.page);
            return Ok(Some(Reservation {
                coverage: None,
                glyph,
            }));
        }
        if width > self.config.page_size || height > self.config.page_size {
            return Err(TextError::GlyphTooLarge {
                width,
                height,
                page_size: self.config.page_size,
            });
        }
        let (page, x, y) = self.place(width, height)?;
        self.misses += 1;
        self.mark_dirty(page, x, y, width, height);
        let glyph = self.register(key, page, x, y, width, height);
        Ok(Some(Reservation {
            coverage: Some(self.device_slot(page, x, y, width, height)),
            glyph,
        }))
    }

    /// Drops a reserved glyph whose coverage was never written.
    ///
    /// The space it took on its shelf stays reserved until the page is
    /// reset: no slot points at it, so it costs nothing but a few bytes of
    /// an evictable page.
    pub fn abandon(&mut self, key: GlyphKey) {
        if let Some(slot) = self.glyphs.remove(&key)
            && let Some(page) = self.pages.get_mut(slot.page as usize)
            && let Some(index) = page
                .keys
                .iter()
                .position(|candidate| *candidate == key)
        {
            page.keys.swap_remove(index);
        }
    }

    /// Makes `bitmap` the coverage of the rectangle, in either storage.
    fn coverage(&mut self, page: u32, x: u32, y: u32, width: u32, height: u32, bitmap: &[u8]) {
        self.mark_dirty(page, x, y, width, height);
        if self.config.device_backed {
            // A device page has no host mirror, so bytes rasterized on the
            // CPU wait here until the pipe copies them into the storage.
            let slot = self.device_slot(page, x, y, width, height);
            self.pending.push(PendingCoverage {
                slot,
                bitmap: bitmap.to_vec(),
            });
            return;
        }
        self.write(page, x, y, width, height, bitmap);
    }

    /// Records that the page's storage changed at the rectangle.
    fn mark_dirty(&mut self, page: u32, x: u32, y: u32, width: u32, height: u32) {
        self.dirty.push(DirtyRect {
            page,
            x,
            y,
            width,
            height,
        });
    }

    /// Registers `key` at the rectangle and returns its slot.
    fn register(&mut self, key: GlyphKey, page: u32, x: u32, y: u32, width: u32, height: u32) -> GlyphSlot {
        let scale = self.config.page_size as f32;
        let slot = GlyphSlot {
            page,
            uv_min: [x as f32 / scale, y as f32 / scale],
            uv_max: [(x + width) as f32 / scale, (y + height) as f32 / scale],
            width,
            height,
        };
        self.glyphs.insert(key, slot);
        self.pages[page as usize].keys.push(key);
        self.touch(page);
        slot
    }

    /// The destination of a rectangle inside a device page's storage.
    fn device_slot(&self, page: u32, x: u32, y: u32, width: u32, height: u32) -> DeviceSlot {
        let page_size = self.config.page_size as usize;
        DeviceSlot {
            page,
            width: width as usize,
            height: height as usize,
            // The rasterizer writes the outline's bottom row first and
            // climbs, so the slot starts at the page's last covered row and
            // its pitch runs one page row upwards: the same bytes a host
            // `write` fills top-down.
            offset: (y as usize + height as usize - 1) * page_size + x as usize,
            pitch: -(page_size as isize),
        }
    }

    /// Copies `bitmap` into the page's host mirror.
    fn write(&mut self, page: u32, x: u32, y: u32, width: u32, height: u32, bitmap: &[u8]) {
        let page_size = self.config.page_size as usize;
        let target = &mut self.pages[page as usize];
        for row in 0..height as usize {
            let source = &bitmap[row * width as usize..(row + 1) * width as usize];
            let start = (y as usize + row) * page_size + x as usize;
            target.pixels[start..start + width as usize].copy_from_slice(source);
        }
    }

    /// Finds a page with room for `width` × `height`, growing or evicting.
    fn place(&mut self, width: u32, height: u32) -> Result<(u32, u32, u32), TextError> {
        let page_size = self.config.page_size;
        for (index, page) in self.pages.iter_mut().enumerate() {
            if let Some((x, y)) = page.allocate(width, height, page_size) {
                return Ok((index as u32, x, y));
            }
        }
        if self.pages.len() < self.config.max_pages {
            let index = self.pages.len();
            let mut page = AtlasPage::new(page_size, self.config.device_backed);
            let (x, y) = page
                .allocate(width, height, page_size)
                .ok_or(TextError::AtlasFull)?;
            self.pages.push(page);
            // The image starts with undefined contents, so its first
            // upload must cover the whole page, not just the new glyph.
            self.dirty.push(DirtyRect {
                page: index as u32,
                x: 0,
                y: 0,
                width: page_size,
                height: page_size,
            });
            return Ok((index as u32, x, y));
        }
        let Some(victim) = self.eviction_candidate() else {
            self.blocked += 1;
            return Err(TextError::AtlasFull);
        };
        self.evictions += 1;
        let (x, y) = self.reset_page(victim, width, height)?;
        Ok((victim as u32, x, y))
    }

    /// Index of the least recently used page that is not pinned.
    fn eviction_candidate(&self) -> Option<usize> {
        self.pages
            .iter()
            .enumerate()
            .filter(|(_, page)| !page.pinned)
            .min_by_key(|(_, page)| page.last_used)
            .map(|(index, _)| index)
    }

    /// Clears `index` and places the new glyph on its first shelf.
    fn reset_page(&mut self, index: usize, width: u32, height: u32) -> Result<(u32, u32), TextError> {
        let page = &mut self.pages[index];
        for key in page.keys.drain(..) {
            self.glyphs.remove(&key);
        }
        // The old pixels stay in the texture; no future UV references
        // them, so clearing the whole page on the CPU would only burn
        // bandwidth. The first glyph writes its own dirty rectangle.
        self.dirty
            .retain(|rect| rect.page != index as u32);
        // Coverage queued for the keys that just died would land on bytes
        // the page's next occupant owns, so it goes with them.
        self.pending
            .retain(|pending| pending.slot.page != index as u32);
        page.shelves.clear();
        page.cursor_y = 0;
        page.pinned = false;
        page.allocate(width, height, self.config.page_size)
            .ok_or(TextError::AtlasFull)
    }

    /// Marks a page as used by the current frame.
    fn touch(&mut self, page: u32) {
        if let Some(page) = self.pages.get_mut(page as usize) {
            page.last_used = self.clock;
            page.pinned = true;
        }
        self.clock = self.clock.saturating_add(1);
    }
}

/// A shared, single-threaded hand-over of an atlas and its indexed runs.
///
/// The UI calls [`TextFrame::begin_frame`] before shaping, rasterizes the
/// glyphs it needs through [`TextFrame::with`], and the text pipe drains the
/// indexed runs and uploads the dirty regions afterward. The handle is
/// `Rc`-based because both sides live on the UI thread.
#[derive(Debug, Clone)]
pub struct TextFrame {
    /// The shared gray coverage atlas.
    atlas: Rc<RefCell<TextAtlas>>,
    /// The shared color BGRA atlas.
    color_atlas: Rc<RefCell<ColorTextAtlas>>,
    /// The shared run builder for gray glyphs, in paint order.
    mesh: Rc<RefCell<MeshBuilder>>,
    /// The shared run builder for color glyphs, in paint order.
    color_mesh: Rc<RefCell<MeshBuilder>>,
    /// The producer-to-consumer dirty epoch.
    signal: Rc<Cell<DirtySignal>>,
}

impl TextFrame {
    /// Wraps `atlas` in a frame with no run.
    ///
    /// # Errors
    ///
    /// Returns [`TextError`] if the atlas configuration cannot be used to
    /// create a color atlas (this should not occur for a valid `TextAtlas`).
    pub fn new(atlas: TextAtlas) -> Result<Self, TextError> {
        let config = atlas.config();
        let color_atlas = ColorTextAtlas::new(config)?;
        Ok(Self {
            atlas: Rc::new(RefCell::new(atlas)),
            color_atlas: Rc::new(RefCell::new(color_atlas)),
            mesh: Rc::new(RefCell::new(MeshBuilder::new())),
            color_mesh: Rc::new(RefCell::new(MeshBuilder::new())),
            signal: Rc::new(Cell::new(DirtySignal::new())),
        })
    }

    /// Borrows the gray atlas for reading.
    #[must_use]
    pub fn atlas(&self) -> Ref<'_, TextAtlas> {
        self.atlas.borrow()
    }

    /// Borrows the color atlas for reading.
    #[must_use]
    pub fn color_atlas(&self) -> Ref<'_, ColorTextAtlas> {
        self.color_atlas.borrow()
    }

    /// Runs `record` with the gray atlas and the run builder borrowed mutably.
    ///
    /// This is the shaping entry point for gray glyphs: a run inserts its
    /// glyphs into the atlas and pushes the quads that reference the returned
    /// slots.
    pub fn with<R>(&self, record: impl FnOnce(&mut TextAtlas, &mut MeshBuilder) -> R) -> R {
        let mut atlas = self.atlas.borrow_mut();
        let mut mesh = self.mesh.borrow_mut();
        record(&mut atlas, &mut mesh)
    }

    /// Runs `record` with the color atlas and the run builder borrowed mutably.
    ///
    /// This is the shaping entry point for color glyphs (emoji): a run inserts
    /// its BGRA glyphs into the color atlas and pushes the quads that reference
    /// the returned slots.
    pub fn with_color<R>(&self, record: impl FnOnce(&mut ColorTextAtlas, &mut MeshBuilder) -> R) -> R {
        let mut atlas = self.color_atlas.borrow_mut();
        let mut mesh = self.color_mesh.borrow_mut();
        record(&mut atlas, &mut mesh)
    }

    /// Starts a shaping pass on both the gray and color atlases.
    pub fn begin_frame(&self) {
        self.atlas.borrow_mut().begin_frame();
        self.color_atlas.borrow_mut().begin_frame();
    }

    /// The indexed gray runs shaped this frame, in paint order.
    #[must_use]
    pub fn runs(&self) -> Ref<'_, [MeshRun]> {
        Ref::map(self.mesh.borrow(), |mesh| mesh.runs())
    }

    /// The indexed color runs shaped this frame, in paint order.
    #[must_use]
    pub fn color_runs(&self) -> Ref<'_, [MeshRun]> {
        Ref::map(self.color_mesh.borrow(), |mesh| mesh.runs())
    }

    /// Whether the frame holds no run (neither gray nor color).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mesh.borrow().is_empty() && self.color_mesh.borrow().is_empty()
    }

    /// Records that the text output changed and must be redrawn.
    ///
    /// The UI calls this when it edits its text; re-shaping identical text
    /// leaves the signal clean, so a still frame is never re-recorded.
    pub fn mark_dirty(&self) {
        let mut signal = self.signal.get();
        signal.mark();
        self.signal.set(signal);
    }

    /// Whether a change has been recorded since the pipe last drew.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.signal.get().is_dirty()
    }

    /// Clears the dirty signal and reports whether the pipe must redraw.
    pub fn take_dirty(&self) -> bool {
        let mut signal = self.signal.get();
        let dirty = signal.take();
        self.signal.set(signal);
        dirty
    }

    /// Empties the run lists, keeping their allocations.
    ///
    /// The atlases are intentionally left intact: their glyphs stay cached for
    /// the next frame, which is the point of a persistent atlas.
    pub fn clear(&self) {
        self.mesh.borrow_mut().clear();
        self.color_mesh.borrow_mut().clear();
    }
}

/// A BGRA color atlas for color glyphs (emoji).
///
/// Stores premultiplied BGRA bitmaps (4 bytes per pixel) in a page-based
/// shelf-packed layout, mirroring the structure of [`TextAtlas`] but with
/// a different pixel format. The atlas is demand-driven: pages are allocated
/// lazily, glyphs are packed into shelves, and LRU eviction reclaims space
/// when the page limit is reached.
///
/// The pixel format matches [`PixelMode::Bgra`] from `codevar-truetype-core`
/// and the output of [`codevar_truetype_emoji_color::render::ColorFont`].
#[derive(Debug)]
pub struct ColorTextAtlas {
    /// Page sizing policy.
    config: AtlasConfig,
    /// Allocated pages, in creation order.
    pages: Vec<ColorAtlasPage>,
    /// Resident glyphs and where they live.
    glyphs: BTreeMap<GlyphKey, GlyphSlot>,
    /// Regions written since the last upload.
    dirty: Vec<DirtyRect>,
    /// Monotonic use counter for the LRU order.
    clock: u64,
    /// Counters of the current frame.
    hits: u64,
    /// Counters of the current frame.
    misses: u64,
    /// Counters of the current frame.
    evictions: u64,
    /// Counters of the current frame.
    blocked: u64,
}

/// One square page of BGRA color pixels.
#[derive(Debug)]
struct ColorAtlasPage {
    /// Row-major BGRA pixels, `page_size * page_size * 4` bytes.
    pixels: Vec<u8>,
    /// Shelves allocated so far, top to bottom.
    shelves: Vec<Shelf>,
    /// First free texel below the last shelf.
    cursor_y: u32,
    /// Logical clock value of the last use.
    last_used: u64,
    /// Whether a glyph of the frame being shaped lives on this page.
    pinned: bool,
    /// Keys whose slots point at this page, for O(keys) eviction.
    keys: Vec<GlyphKey>,
}

impl ColorAtlasPage {
    /// Creates a zeroed page.
    fn new(page_size: u32) -> Self {
        let len = (page_size as usize)
            .saturating_mul(page_size as usize)
            .saturating_mul(4);
        Self {
            pixels: alloc::vec![0; len],
            shelves: Vec::new(),
            cursor_y: 0,
            last_used: 0,
            pinned: false,
            keys: Vec::new(),
        }
    }

    /// Reserves a `width` × `height` glyph, returning its top-left texel.
    fn allocate(&mut self, width: u32, height: u32, page_size: u32) -> Option<(u32, u32)> {
        let needed_width = width + GLYPH_PADDING;
        let needed_height = height + GLYPH_PADDING;
        for shelf in &mut self.shelves {
            if shelf.height >= needed_height && shelf.cursor_x + needed_width <= page_size {
                let x = shelf.cursor_x;
                shelf.cursor_x += needed_width;
                return Some((x, shelf.y));
            }
        }
        if self.cursor_y + needed_height > page_size {
            return None;
        }
        let y = self.cursor_y;
        self.cursor_y += needed_height;
        self.shelves.push(Shelf {
            y,
            height: needed_height,
            cursor_x: needed_width,
        });
        Some((0, y))
    }
}

impl ColorTextAtlas {
    /// Creates a color atlas that has not allocated a page yet.
    ///
    /// # Errors
    ///
    /// [`TextError::ZeroPageSize`], [`TextError::ZeroPageCount`] or
    /// [`TextError::PageTooLarge`] when the configuration cannot describe
    /// an addressable page.
    pub fn new(config: AtlasConfig) -> Result<Self, TextError> {
        if config.page_size == 0 {
            return Err(TextError::ZeroPageSize);
        }
        if config.max_pages == 0 {
            return Err(TextError::ZeroPageCount);
        }
        if config.page_size > MAX_PAGE_SIZE
            || (config.page_size as usize)
                .checked_mul(config.page_size as usize)
                .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(TextError::PageTooLarge {
                page_size: config.page_size,
            });
        }
        Ok(Self {
            config,
            pages: Vec::new(),
            glyphs: BTreeMap::new(),
            dirty: Vec::new(),
            clock: 1,
            hits: 0,
            misses: 0,
            evictions: 0,
            blocked: 0,
        })
    }

    /// The sizing policy of the atlas.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> AtlasConfig {
        self.config
    }

    /// The number of pages allocated so far.
    #[inline]
    #[must_use]
    pub fn pages(&self) -> usize {
        self.pages.len()
    }

    /// The number of resident glyphs.
    #[inline]
    #[must_use]
    pub fn glyph_count(&self) -> usize {
        self.glyphs.len()
    }

    /// Bytes of color data the resident glyphs occupy (4 bytes per pixel).
    #[must_use]
    pub fn packed_bytes(&self) -> usize {
        self.glyphs
            .values()
            .map(|slot| slot.width as usize * slot.height as usize * 4)
            .sum()
    }

    /// The BGRA pixels of one page, row-major, or `None` for an unknown page.
    #[inline]
    #[must_use]
    pub fn page_pixels(&self, page: u32) -> Option<&[u8]> {
        self.pages
            .get(page as usize)
            .map(|page| page.pixels.as_slice())
    }

    /// The regions written since the last [`ColorTextAtlas::clear_dirty`].
    #[inline]
    #[must_use]
    pub fn dirty_rects(&self) -> &[DirtyRect] {
        &self.dirty
    }

    /// Drops the dirty list once the pipe has uploaded it.
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    /// The current frame's counters.
    #[must_use]
    pub fn stats(&self) -> AtlasStats {
        AtlasStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            blocked: self.blocked,
            glyphs: self.glyphs.len() as u32,
            pages: self.pages.len() as u32,
            packed_bytes: self.packed_bytes() as u64,
        }
    }

    /// Starts a frame: clears every page pin and the counters.
    pub fn begin_frame(&mut self) {
        for page in &mut self.pages {
            page.pinned = false;
        }
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.blocked = 0;
    }

    /// Looks up a resident glyph and promotes its page.
    pub fn get(&mut self, key: GlyphKey) -> Option<GlyphSlot> {
        let slot = self.glyphs.get(&key).copied()?;
        self.hits += 1;
        self.touch(slot.page);
        Some(slot)
    }

    /// Places a rasterized color glyph, or returns its existing slot.
    ///
    /// `bitmap` must be 4 bytes per pixel (BGRA), row-major, exactly
    /// `width * height * 4` bytes. A zero `width` or `height` means the glyph
    /// has no bitmap: the call returns `Ok(None)` and allocates nothing.
    ///
    /// Re-inserting a resident key returns the old slot and ignores the
    /// new bitmap; a glyph never moves while a frame may reference it.
    ///
    /// # Errors
    ///
    /// * [`TextError::GlyphSizeMismatch`] when `bitmap` is the wrong size;
    /// * [`TextError::GlyphTooLarge`] when the glyph does not fit a page;
    /// * [`TextError::AtlasFull`] when every page is pinned and full.
    pub fn insert(
        &mut self,
        key: GlyphKey,
        width: u32,
        height: u32,
        bitmap: &[u8],
    ) -> Result<Option<GlyphSlot>, TextError> {
        if width == 0 || height == 0 {
            return Ok(None);
        }
        if let Some(slot) = self.glyphs.get(&key).copied() {
            self.hits += 1;
            self.touch(slot.page);
            return Ok(Some(slot));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|p| p.checked_mul(4))
            .ok_or(TextError::GlyphTooLarge {
                width,
                height,
                page_size: self.config.page_size,
            })?;
        if bitmap.len() != expected {
            return Err(TextError::GlyphSizeMismatch {
                width,
                height,
                actual: bitmap.len(),
            });
        }
        if width > self.config.page_size || height > self.config.page_size {
            return Err(TextError::GlyphTooLarge {
                width,
                height,
                page_size: self.config.page_size,
            });
        }
        let (page, x, y) = self.place(width, height)?;
        self.misses += 1;
        self.write(page, x, y, width, height, bitmap);
        let scale = self.config.page_size as f32;
        let slot = GlyphSlot {
            page,
            uv_min: [x as f32 / scale, y as f32 / scale],
            uv_max: [(x + width) as f32 / scale, (y + height) as f32 / scale],
            width,
            height,
        };
        self.glyphs.insert(key, slot);
        self.pages[page as usize].keys.push(key);
        self.touch(page);
        Ok(Some(slot))
    }

    /// Copies `bitmap` into the page and records the dirty rectangle.
    fn write(&mut self, page: u32, x: u32, y: u32, width: u32, height: u32, bitmap: &[u8]) {
        let page_size = self.config.page_size as usize;
        let target = &mut self.pages[page as usize];
        for row in 0..height as usize {
            let source = &bitmap[row * width as usize * 4..(row + 1) * width as usize * 4];
            let start = ((y as usize + row) * page_size + x as usize) * 4;
            target.pixels[start..start + width as usize * 4].copy_from_slice(source);
        }
        self.dirty.push(DirtyRect {
            page,
            x,
            y,
            width,
            height,
        });
    }

    /// Finds a page with room for `width` × `height`, growing or evicting.
    fn place(&mut self, width: u32, height: u32) -> Result<(u32, u32, u32), TextError> {
        let page_size = self.config.page_size;
        for (index, page) in self.pages.iter_mut().enumerate() {
            if let Some((x, y)) = page.allocate(width, height, page_size) {
                return Ok((index as u32, x, y));
            }
        }
        if self.pages.len() < self.config.max_pages {
            let index = self.pages.len();
            let mut page = ColorAtlasPage::new(page_size);
            let (x, y) = page
                .allocate(width, height, page_size)
                .ok_or(TextError::AtlasFull)?;
            self.pages.push(page);
            self.dirty.push(DirtyRect {
                page: index as u32,
                x: 0,
                y: 0,
                width: page_size,
                height: page_size,
            });
            return Ok((index as u32, x, y));
        }
        let Some(victim) = self.eviction_candidate() else {
            self.blocked += 1;
            return Err(TextError::AtlasFull);
        };
        self.evictions += 1;
        let (x, y) = self.reset_page(victim, width, height)?;
        Ok((victim as u32, x, y))
    }

    /// Index of the least recently used page that is not pinned.
    fn eviction_candidate(&self) -> Option<usize> {
        self.pages
            .iter()
            .enumerate()
            .filter(|(_, page)| !page.pinned)
            .min_by_key(|(_, page)| page.last_used)
            .map(|(index, _)| index)
    }

    /// Clears `index` and places the new glyph on its first shelf.
    fn reset_page(&mut self, index: usize, width: u32, height: u32) -> Result<(u32, u32), TextError> {
        let page = &mut self.pages[index];
        for key in page.keys.drain(..) {
            self.glyphs.remove(&key);
        }
        self.dirty
            .retain(|rect| rect.page != index as u32);
        page.shelves.clear();
        page.cursor_y = 0;
        page.pinned = false;
        page.allocate(width, height, self.config.page_size)
            .ok_or(TextError::AtlasFull)
    }

    /// Marks a page as used by the current frame.
    fn touch(&mut self, page: u32) {
        if let Some(page) = self.pages.get_mut(page as usize) {
            page.last_used = self.clock;
            page.pinned = true;
        }
        self.clock = self.clock.saturating_add(1);
    }
}

/// Why a [`TextAtlas`] could not be built or filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextError {
    /// The configured page side is zero.
    ZeroPageSize,
    /// The configured page count is zero.
    ZeroPageCount,
    /// The page side is larger than the renderer can address.
    PageTooLarge {
        /// The requested side.
        page_size: u32,
    },
    /// The coverage buffer does not match `width * height`.
    GlyphSizeMismatch {
        /// The requested width.
        width: u32,
        /// The requested height.
        height: u32,
        /// The number of bytes handed in.
        actual: usize,
    },
    /// The glyph does not fit inside one page.
    GlyphTooLarge {
        /// The requested width.
        width: u32,
        /// The requested height.
        height: u32,
        /// The configured page side.
        page_size: u32,
    },
    /// Every page is pinned by the current frame and full.
    AtlasFull,
    /// The atlas keeps host mirrors, so it has no device storage to reserve.
    HostPages,
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPageSize => f.write_str("the text atlas page size is zero"),
            Self::ZeroPageCount => f.write_str("the text atlas page count is zero"),
            Self::PageTooLarge { page_size } => {
                write!(f, "the text atlas page size {page_size} is too large")
            }
            Self::GlyphSizeMismatch {
                width,
                height,
                actual,
            } => write!(
                f,
                "text glyph needs {} bytes for {width}x{height} but got {actual}",
                *width as usize * *height as usize
            ),
            Self::GlyphTooLarge {
                width,
                height,
                page_size,
            } => write!(f, "text glyph {width}x{height} does not fit a {page_size} page"),
            Self::AtlasFull => f.write_str("the text atlas is full and every page is pinned"),
            Self::HostPages => f.write_str("the text atlas is host-backed and has no device storage"),
        }
    }
}

impl core::error::Error for TextError {}

#[cfg(test)]
mod tests {
    //! The atlas policy: lazy pages, shelf packing, UV math, eviction,
    //! pinning, dirty rectangles and the frame hand-over.

    use super::*;
    use alloc::vec;

    const KEY_A: GlyphKey = GlyphKey::from_raw(1);
    const KEY_B: GlyphKey = GlyphKey::from_raw(2);

    fn small(page_size: u32, max_pages: usize) -> TextAtlas {
        let config = AtlasConfig {
            page_size,
            max_pages,
            device_backed: false,
        };
        TextAtlas::new(config).unwrap()
    }

    fn device(page_size: u32, max_pages: usize) -> TextAtlas {
        let config = AtlasConfig {
            page_size,
            max_pages,
            device_backed: true,
        };
        TextAtlas::new(config).unwrap()
    }

    fn cover(count: usize, byte: u8) -> Vec<u8> {
        vec![byte; count]
    }

    #[test]
    fn the_first_glyph_allocates_the_first_page_lazily() {
        let mut atlas = small(64, 2);
        assert_eq!(atlas.pages(), 0, "no page is reserved up front");

        let slot = atlas
            .insert(KEY_A, 2, 2, &cover(4, 5))
            .unwrap()
            .unwrap();
        assert_eq!(atlas.pages(), 1);
        assert_eq!(slot.page, 0);
        assert_eq!((slot.width, slot.height), (2, 2));
        assert_eq!(slot.uv_min, [0.0, 0.0]);
        assert_eq!(slot.uv_max, [2.0 / 64.0, 2.0 / 64.0]);
        assert_eq!(atlas.glyph_count(), 1);
        assert_eq!(atlas.packed_bytes(), 4);
    }

    #[test]
    fn a_glyph_is_written_row_major_and_records_a_dirty_rect() {
        let mut atlas = small(8, 1);
        atlas
            .insert(KEY_A, 2, 2, &[1, 2, 3, 4])
            .unwrap()
            .unwrap();

        let pixels = atlas.page_pixels(0).unwrap();
        assert_eq!(&pixels[0..2], &[1, 2]);
        assert_eq!(&pixels[8..10], &[3, 4]);

        let dirty = atlas.dirty_rects();
        assert_eq!(dirty.len(), 2, "the full-page init plus the glyph");
        assert_eq!(dirty[0].width, 8);
        assert_eq!(dirty[1].width, 2);
        assert_eq!(
            dirty[1],
            DirtyRect {
                page: 0,
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            }
        );
        atlas.clear_dirty();
        assert!(atlas.dirty_rects().is_empty());
    }

    #[test]
    fn a_device_page_keeps_no_host_mirror() {
        let mut atlas = device(8, 1);
        atlas
            .insert(KEY_A, 2, 2, &[1, 2, 3, 4])
            .unwrap()
            .unwrap();

        assert!(atlas.page_pixels(0).is_none(), "the bytes are not on the host");
        let pending = atlas.pending_coverage();
        assert_eq!(pending.len(), 1, "the host rasterization waits here");
        assert_eq!(
            pending[0].slot,
            DeviceSlot {
                page: 0,
                width: 2,
                height: 2,
                offset: 8,
                pitch: -8,
            }
        );
        assert_eq!(pending[0].bitmap, vec![1, 2, 3, 4]);
        assert_eq!(atlas.dirty_rects().len(), 2, "the page still needs a flush");
        atlas.clear_pending();
        assert!(atlas.pending_coverage().is_empty());
    }

    #[test]
    fn an_eviction_drops_the_coverage_the_dead_keys_queued() {
        let mut atlas = device(4, 1);
        atlas
            .insert(KEY_A, 2, 2, &cover(4, 1))
            .unwrap()
            .unwrap();
        atlas.begin_frame();
        atlas
            .insert(KEY_B, 2, 2, &cover(4, 2))
            .unwrap()
            .unwrap();

        assert_eq!(atlas.stats().evictions, 1);
        let pending = atlas.pending_coverage();
        assert_eq!(pending.len(), 1, "only the surviving glyph is queued");
        assert_eq!(pending[0].bitmap, vec![2, 2, 2, 2]);
    }

    #[test]
    fn a_host_atlas_queues_nothing_and_refuses_a_reservation() {
        let mut atlas = small(8, 1);
        atlas
            .insert(KEY_A, 2, 2, &cover(4, 1))
            .unwrap()
            .unwrap();
        assert!(atlas.pending_coverage().is_empty());
        assert_eq!(atlas.reserve(KEY_A, 2, 2), Err(TextError::HostPages));
    }

    #[test]
    fn a_reserved_glyph_reports_the_bytes_the_rasterizer_writes() {
        let mut atlas = device(8, 1);
        atlas
            .insert(KEY_A, 2, 2, &cover(4, 1))
            .unwrap()
            .unwrap();
        let placed = atlas.reserve(KEY_B, 2, 2).unwrap().unwrap();

        assert_eq!(placed.glyph.page, 0);
        assert_eq!(placed.glyph.uv_min, [3.0 / 8.0, 0.0]);
        assert_eq!(
            placed.coverage,
            Some(DeviceSlot {
                page: 0,
                width: 2,
                height: 2,
                offset: 11,
                pitch: -8,
            })
        );
        assert_eq!(placed.coverage.unwrap().bytes(), Some(3..13));

        let again = atlas.reserve(KEY_B, 2, 2).unwrap().unwrap();
        assert_eq!(again.glyph, placed.glyph);
        assert_eq!(again.coverage, None, "a resident glyph has nothing left to write");
        assert_eq!(atlas.stats().hits, 1);
    }

    #[test]
    fn both_paths_to_a_device_page_address_the_same_bytes_as_the_host_page() {
        let mut host = small(8, 1);
        let mut atlas = device(8, 1);
        let mut storage = vec![0u8; 64];
        let mut reservations = Vec::new();
        for index in 0..4usize {
            let key = GlyphKey::from_raw(index as u64 + 1);
            // A 2x2 glyph whose top row is `top` and bottom row `bottom`.
            let top = index as u8 + 1;
            let bitmap = vec![top, top, top + 4, top + 4];
            let slot = host.insert(key, 2, 2, &bitmap).unwrap().unwrap();
            let placed = atlas.reserve(key, 2, 2).unwrap().unwrap();
            assert_eq!(placed.glyph, slot, "both atlases place the same way");
            let coverage = placed.coverage.unwrap();
            raster_into(&mut storage, coverage, &bitmap);
            reservations.push((coverage, bitmap));
        }
        let pixels = host.page_pixels(0).unwrap();
        assert_eq!(
            storage.as_slice(),
            pixels,
            "the slot addresses are the host page's rows"
        );

        // The pipe's own copy of a host-rasterized glyph lands identically.
        let mut copied = vec![0u8; 64];
        for (coverage, bitmap) in reservations {
            copy_into(&mut copied, coverage, &bitmap);
        }
        assert_eq!(
            copied.as_slice(),
            pixels,
            "`row_address` walks the same rows the rasterizer writes"
        );
    }

    #[test]
    fn an_abandoned_reservation_leaves_nothing_behind() {
        let mut atlas = device(8, 1);
        let placed = atlas.reserve(KEY_A, 2, 2).unwrap().unwrap();
        atlas.abandon(KEY_A);

        assert_eq!(atlas.glyph_count(), 0, "no slot names the reservation");
        assert!(atlas.get(KEY_A).is_none(), "the key is no longer cached");
        let again = atlas.reserve(KEY_A, 2, 2).unwrap().unwrap();
        assert!(again.coverage.is_some(), "the key is free to reserve once more");
        assert!(again.glyph != placed.glyph, "the retry takes new space");
    }

    #[test]
    fn a_slot_is_refused_when_its_rows_leave_the_storage() {
        let last_row = DeviceSlot {
            page: 0,
            width: 2,
            height: 2,
            offset: 56,
            pitch: -8,
        };
        assert_eq!(last_row.bytes(), Some(48..58));
        assert_eq!(last_row.row_address(0), Some(48));
        assert_eq!(last_row.row_address(2), None, "a row outside the glyph");

        let underflow = DeviceSlot {
            offset: 0,
            ..last_row
        };
        assert_eq!(underflow.bytes(), None, "the first row would run past the top");
        let empty = DeviceSlot {
            height: 0,
            ..last_row
        };
        assert_eq!(empty.bytes(), None, "a glyph with no rows has no bytes");
    }

    /// Writes `bitmap` into `storage` the way the rasterizer kernel does:
    /// its first row is the glyph's bottom edge and `pitch` steps upward.
    fn raster_into(storage: &mut [u8], slot: DeviceSlot, bitmap: &[u8]) {
        for py in 0..slot.height {
            let address = (slot.offset as i64 + py as i64 * slot.pitch as i64) as usize;
            for px in 0..slot.width {
                // The bitmap is top-down, the kernel's `py` is bottom-up.
                let source = (slot.height - 1 - py) * slot.width + px;
                storage[address + px] = bitmap[source];
            }
        }
    }

    /// Writes `bitmap` into `storage` the way the pipe does for a host
    /// rasterization: top-down rows at the addresses the slot reports.
    fn copy_into(storage: &mut [u8], slot: DeviceSlot, bitmap: &[u8]) {
        for row in 0..slot.height {
            let address = slot.row_address(row).unwrap();
            let source = row * slot.width..(row + 1) * slot.width;
            storage[address..address + slot.width].copy_from_slice(&bitmap[source]);
        }
    }

    #[test]
    fn a_full_shelf_starts_a_new_shelf_below_it() {
        let mut atlas = small(8, 1);
        let first = atlas
            .insert(KEY_A, 6, 2, &cover(12, 1))
            .unwrap()
            .unwrap();
        let second = atlas
            .insert(GlyphKey::from_raw(2), 6, 2, &cover(12, 1))
            .unwrap()
            .unwrap();

        assert_eq!(first.uv_min[1], 0.0);
        assert!(second.uv_min[1] > 0.0, "the second glyph opens a shelf");
        assert_eq!(second.uv_min[1], 3.0 / 8.0);
    }

    #[test]
    fn a_glyph_bigger_than_the_page_is_rejected() {
        let mut atlas = small(4, 1);
        assert_eq!(
            atlas.insert(KEY_A, 5, 5, &cover(25, 0)),
            Err(TextError::GlyphTooLarge {
                width: 5,
                height: 5,
                page_size: 4,
            })
        );
    }

    #[test]
    fn a_bitmap_of_the_wrong_length_is_rejected() {
        let mut atlas = small(8, 1);
        assert_eq!(
            atlas.insert(KEY_A, 2, 2, &cover(3, 0)),
            Err(TextError::GlyphSizeMismatch {
                width: 2,
                height: 2,
                actual: 3,
            })
        );
    }

    #[test]
    fn an_empty_glyph_allocates_nothing() {
        let mut atlas = small(8, 1);
        assert_eq!(atlas.insert(KEY_A, 0, 4, &[]), Ok(None));
        assert_eq!(atlas.pages(), 0);
        assert_eq!(atlas.glyph_count(), 0);
    }

    #[test]
    fn reinserting_a_glyph_returns_the_cached_slot() {
        let mut atlas = small(8, 1);
        let first = atlas
            .insert(KEY_A, 2, 2, &cover(4, 1))
            .unwrap()
            .unwrap();
        let second = atlas
            .insert(KEY_A, 2, 2, &cover(4, 9))
            .unwrap()
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(atlas.glyph_count(), 1);
        assert_eq!(atlas.stats().hits, 1);
    }

    #[test]
    fn a_full_atlas_evicts_the_least_recently_used_page() {
        let mut atlas = small(4, 1);
        let old = atlas
            .insert(KEY_A, 2, 2, &cover(4, 1))
            .unwrap()
            .unwrap();
        atlas.begin_frame();
        let new = atlas
            .insert(GlyphKey::from_raw(2), 2, 2, &cover(4, 2))
            .unwrap()
            .unwrap();

        assert_eq!(old.page, new.page, "the only page is reused");
        assert_eq!(atlas.pages(), 1);
        assert_eq!(atlas.stats().evictions, 1);
        assert!(atlas.get(KEY_A).is_none(), "the old glyph was evicted");
        assert!(atlas.get(GlyphKey::from_raw(2)).is_some());
    }

    #[test]
    fn a_pinned_page_is_not_evicted() {
        let mut atlas = small(4, 1);
        atlas
            .insert(KEY_A, 2, 2, &cover(4, 1))
            .unwrap()
            .unwrap();
        // No begin_frame: the page is pinned by the insert.

        assert_eq!(
            atlas.insert(GlyphKey::from_raw(2), 2, 2, &cover(4, 2)),
            Err(TextError::AtlasFull)
        );
        assert_eq!(atlas.stats().blocked, 1);
        assert!(atlas.get(KEY_A).is_some());
    }

    #[test]
    fn lookups_promote_the_page_and_pin_it() {
        let mut atlas = small(4, 1);
        atlas
            .insert(KEY_A, 2, 2, &cover(4, 1))
            .unwrap()
            .unwrap();
        atlas.begin_frame();
        assert!(atlas.get(KEY_A).is_some());
        // The lookup pinned the page again, so a new glyph cannot evict it
        // until the next begin_frame, even though the page is full.
        assert_eq!(
            atlas.insert(GlyphKey::from_raw(2), 2, 2, &cover(4, 2)),
            Err(TextError::AtlasFull)
        );
    }

    #[test]
    fn a_second_page_is_used_before_eviction() {
        let mut atlas = small(4, 2);
        atlas
            .insert(KEY_A, 2, 2, &cover(4, 1))
            .unwrap()
            .unwrap();
        atlas.begin_frame();
        let second = atlas
            .insert(GlyphKey::from_raw(2), 2, 2, &cover(4, 2))
            .unwrap()
            .unwrap();

        assert_eq!(atlas.pages(), 2);
        assert_eq!(second.page, 1);
        assert_eq!(atlas.stats().evictions, 0);
    }

    #[test]
    fn a_frame_hands_the_atlas_and_indexed_runs_to_the_pipe() {
        let frame = TextFrame::new(small(8, 1)).unwrap();
        frame.begin_frame();
        frame.with(|atlas, mesh| {
            let slot = atlas
                .insert(KEY_A, 1, 1, &cover(1, 1))
                .unwrap()
                .unwrap();
            mesh.push(&crate::mesh::GlyphQuad::from_slot(
                slot,
                [1.0, 2.0],
                [1.0; 4],
                None,
            ));
        });

        assert!(!frame.is_empty());
        let runs = frame.runs();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].page, 0);
        assert_eq!(runs[0].mesh.vertices().len(), 4);
        assert_eq!(runs[0].mesh.indices().len(), 6);
        drop(runs);
        assert_eq!(frame.atlas().glyph_count(), 1);
        frame.clear();
        assert!(frame.is_empty());
        assert_eq!(frame.atlas().glyph_count(), 1, "the cache survives");
    }

    #[test]
    fn a_frame_starts_dirty_and_goes_clean_once_taken() {
        let frame = TextFrame::new(small(8, 1)).unwrap();
        assert!(frame.is_dirty(), "the first frame must draw");
        assert!(frame.take_dirty());
        assert!(!frame.is_dirty());

        frame.mark_dirty();
        assert!(frame.is_dirty());
        assert!(frame.take_dirty());
        assert!(!frame.take_dirty(), "an idle frame reports clean");
    }

    #[test]
    fn an_invalid_config_is_rejected() {
        assert_eq!(
            TextAtlas::new(AtlasConfig {
                page_size: 0,
                max_pages: 1,
                device_backed: false,
            })
            .unwrap_err(),
            TextError::ZeroPageSize
        );
        assert_eq!(
            TextAtlas::new(AtlasConfig {
                page_size: 8,
                max_pages: 0,
                device_backed: false,
            })
            .unwrap_err(),
            TextError::ZeroPageCount
        );
        assert_eq!(
            TextAtlas::new(AtlasConfig {
                page_size: MAX_PAGE_SIZE + 1,
                max_pages: 1,
                device_backed: false,
            })
            .unwrap_err(),
            TextError::PageTooLarge {
                page_size: MAX_PAGE_SIZE + 1,
            }
        );
    }
}
