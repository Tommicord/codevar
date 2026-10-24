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

//! Indexed glyph geometry: four vertices and six indices per glyph.
//!
//! A glyph quad is a rectangle, so the two triangles that fill it share an
//! edge and two corners. Describing the rectangle as two independent
//! triangles — six vertices — repeats those two corners; emitting four
//! vertices and six [`Index`]es instead cuts the vertex traffic of a run by
//! a third, because each vertex carries a position, an atlas coordinate and
//! a color. The indices also let the backend keep one small immutable
//! `u32` buffer per run while the vertex buffer grows.
//!
//! Quads arrive in paint order and are packed by the [`MeshBuilder`] into a
//! [`MeshRun`] per `(page, clip)` pair. A run samples exactly one atlas page,
//! so the pipe binds that page once and issues one indexed draw for the whole
//! run. Only quads that are adjacent in paint order and equal in page and clip
//! share a run; interleaving two pages would change the pixel order.

use alloc::vec::Vec;

use codevar_kuitui::geometry::Rect;

use crate::atlas::{GlyphSlot, TextVertex};

/// Vertices the builder emits for one glyph.
pub const VERTICES_PER_QUAD: usize = 4;

/// Indices the builder emits for one glyph.
pub const INDICES_PER_QUAD: usize = 6;

/// Width of a vertex index.
///
/// A single run holds far fewer than four billion vertices, and a `u32`
/// index keeps the arithmetic in the backend free of narrowing conversions.
pub type Index = u32;

/// One glyph to place in a run.
///
/// The quad is axis-aligned: [`GlyphQuad::position`] is the top-left corner in
/// logical pixels, [`GlyphQuad::size`] the advance the rasterizer reported and
/// [`GlyphQuad::uv_min`]/[`GlyphQuad::uv_max`] the atlas rectangle to sample.
/// `clip` is resolved by the UI when the glyph is recorded; two quads only
/// join a run when their clips match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphQuad {
    /// Index of the atlas page the glyph samples.
    pub page: u32,
    /// Top-left corner in logical pixels.
    pub position: [f32; 2],
    /// Width and height of the quad in logical pixels.
    pub size: [f32; 2],
    /// Inclusive upper-left atlas coordinate.
    pub uv_min: [f32; 2],
    /// Exclusive lower-right atlas coordinate.
    pub uv_max: [f32; 2],
    /// Straight (non-premultiplied) sRGB color of the glyph.
    pub color: [f32; 4],
    /// Clip resolved by the UI, or `None` for the whole surface.
    pub clip: Option<Rect>,
}

impl GlyphQuad {
    /// Builds the quad of a rasterized glyph at `position`.
    ///
    /// The size comes from the slot, so the quad always covers exactly the
    /// coverage the atlas holds and the UV edges stay inside the glyph's
    /// padding.
    #[inline]
    #[must_use]
    pub fn from_slot(slot: GlyphSlot, position: [f32; 2], color: [f32; 4], clip: Option<Rect>) -> Self {
        Self {
            page: slot.page,
            position,
            size: [slot.width as f32, slot.height as f32],
            uv_min: slot.uv_min,
            uv_max: slot.uv_max,
            color,
            clip,
        }
    }
}

/// The vertices and indices of a set of adjacent glyph quads.
///
/// The mesh is grow-only: [`IndexedMesh::clear`] drops the elements but keeps
/// the allocations, so re-shaping the same run does not touch the allocator.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IndexedMesh {
    /// Four [`TextVertex`]es per quad, in clockwise order from the top-left.
    vertices: Vec<TextVertex>,
    /// Six indices per quad: `0,1,2` and `0,2,3` of its vertex quad.
    indices: Vec<Index>,
}

impl IndexedMesh {
    /// An empty mesh with no allocation.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Drops every quad, keeping the vertex and index capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Whether the mesh holds no quad.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// The number of glyph quads in the mesh.
    #[inline]
    #[must_use]
    pub fn quad_count(&self) -> usize {
        self.indices.len() / INDICES_PER_QUAD
    }

    /// The vertex array handed to the vertex buffer.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[TextVertex] {
        &self.vertices
    }

    /// The index array handed to the index buffer.
    #[inline]
    #[must_use]
    pub fn indices(&self) -> &[Index] {
        &self.indices
    }

    /// Reserves room for `quads` more glyphs.
    pub fn reserve_quads(&mut self, quads: usize) {
        self.vertices
            .reserve(quads.saturating_mul(VERTICES_PER_QUAD));
        self.indices
            .reserve(quads.saturating_mul(INDICES_PER_QUAD));
    }

    /// Appends one quad and returns whether it fit an index.
    ///
    /// The four vertices are emitted top-left, top-right, bottom-right and
    /// bottom-left; the six indices wound clockwise. `false` means the mesh
    /// already holds `u32::MAX` vertices and the caller must start a new run.
    pub fn push_quad(&mut self, quad: &GlyphQuad) -> bool {
        let base = self.vertices.len();
        if base > (Index::MAX as usize).saturating_sub(VERTICES_PER_QUAD - 1) {
            return false;
        }

        let [x, y] = quad.position;
        let [width, height] = quad.size;
        let [u_min, v_min] = quad.uv_min;
        let [u_max, v_max] = quad.uv_max;
        let color = quad.color;
        let right = x + width;
        let bottom = y + height;

        self.vertices.push(TextVertex {
            position: [x, y],
            uv: [u_min, v_min],
            color,
        });
        self.vertices.push(TextVertex {
            position: [right, y],
            uv: [u_max, v_min],
            color,
        });
        self.vertices.push(TextVertex {
            position: [right, bottom],
            uv: [u_max, v_max],
            color,
        });
        self.vertices.push(TextVertex {
            position: [x, bottom],
            uv: [u_min, v_max],
            color,
        });

        let base = base as Index;
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        true
    }
}

/// A run of glyph quads that share an atlas page and a clip.
///
/// The run is the batch unit of the pipe: one page bind, one indexed draw.
/// Runs stay in paint order, so drawing them in sequence reproduces the
/// display list even though the glyphs are packed per page.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshRun {
    /// Index of the atlas page every vertex samples.
    pub page: u32,
    /// The indexed quad geometry.
    pub mesh: IndexedMesh,
    /// Clip resolved by the UI, or `None` for the whole surface.
    pub clip: Option<Rect>,
}

impl MeshRun {
    /// Wraps a page, a mesh and a clip.
    #[inline]
    #[must_use]
    pub const fn new(page: u32, mesh: IndexedMesh, clip: Option<Rect>) -> Self {
        Self { page, mesh, clip }
    }

    /// Whether the run holds no quad.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mesh.is_empty()
    }

    /// The number of quads in the run.
    #[inline]
    #[must_use]
    pub fn quad_count(&self) -> usize {
        self.mesh.quad_count()
    }
}

/// Packs glyph quads into page runs while preserving paint order.
///
/// The builder reuses its runs, their meshes and their allocations across
/// frames: [`MeshBuilder::clear`] only resets a length. A caller that shapes
/// the same text every frame allocates nothing after the first frame.
#[derive(Debug, Default)]
pub struct MeshBuilder {
    /// Run storage, reused across frames. `active` of the entries are live.
    runs: Vec<MeshRun>,
    /// Number of entries at the front of `runs` that are live.
    active: usize,
}

impl MeshBuilder {
    /// An empty builder with no allocation.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            runs: Vec::new(),
            active: 0,
        }
    }

    /// Ends the frame, keeping every run and allocation for reuse.
    #[inline]
    pub fn clear(&mut self) {
        self.active = 0;
    }

    /// The live runs, in paint order.
    #[inline]
    #[must_use]
    pub fn runs(&self) -> &[MeshRun] {
        &self.runs[..self.active]
    }

    /// Whether the builder holds no run.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.active == 0
    }

    /// The number of quads across every live run.
    #[must_use]
    pub fn quad_count(&self) -> usize {
        self.runs().iter().map(MeshRun::quad_count).sum()
    }

    /// Appends a quad, extending the last run or opening a new one.
    ///
    /// Returns `false` only when a single run reached `u32::MAX` vertices,
    /// which a caller can only hit after placing more than a billion glyphs.
    pub fn push(&mut self, quad: &GlyphQuad) -> bool {
        if self.active > 0 {
            let last = self.active - 1;
            let run = &mut self.runs[last];
            if run.page == quad.page && run.clip == quad.clip && run.mesh.push_quad(quad) {
                return true;
            }
        }

        let index = self.active;
        if index == self.runs.len() {
            self.runs
                .push(MeshRun::new(quad.page, IndexedMesh::new(), quad.clip));
        } else {
            let run = &mut self.runs[index];
            run.page = quad.page;
            run.clip = quad.clip;
            run.mesh.clear();
        }

        if self.runs[index].mesh.push_quad(quad) {
            self.active = index + 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    //! Quad winding, index offsets and run coalescing.

    use super::*;
    use codevar_kuitui::geometry::Offset;

    const PAGE_0: u32 = 0;
    const PAGE_1: u32 = 1;

    fn quad(page: u32, x: f32, y: f32) -> GlyphQuad {
        GlyphQuad {
            page,
            position: [x, y],
            size: [4.0, 8.0],
            uv_min: [0.25, 0.5],
            uv_max: [0.5, 0.75],
            color: [1.0, 0.0, 0.0, 1.0],
            clip: None,
        }
    }

    #[test]
    fn a_quad_emits_four_vertices_and_six_indices() {
        let mut mesh = IndexedMesh::new();
        assert!(mesh.push_quad(&quad(PAGE_0, 10.0, 20.0)));

        assert_eq!(mesh.vertices().len(), VERTICES_PER_QUAD);
        assert_eq!(mesh.indices(), &[0, 1, 2, 0, 2, 3]);
        assert_eq!(mesh.quad_count(), 1);
        assert!(!mesh.is_empty());

        assert_eq!(mesh.vertices()[0].position, [10.0, 20.0]);
        assert_eq!(mesh.vertices()[1].position, [14.0, 20.0]);
        assert_eq!(mesh.vertices()[2].position, [14.0, 28.0]);
        assert_eq!(mesh.vertices()[3].position, [10.0, 28.0]);
    }

    #[test]
    fn the_uv_corners_follow_the_quad_corners() {
        let mut mesh = IndexedMesh::new();
        mesh.push_quad(&quad(PAGE_0, 0.0, 0.0));

        assert_eq!(mesh.vertices()[0].uv, [0.25, 0.5], "top-left");
        assert_eq!(mesh.vertices()[1].uv, [0.5, 0.5], "top-right");
        assert_eq!(mesh.vertices()[2].uv, [0.5, 0.75], "bottom-right");
        assert_eq!(mesh.vertices()[3].uv, [0.25, 0.75], "bottom-left");
    }

    #[test]
    fn a_second_quad_continues_the_index_base() {
        let mut mesh = IndexedMesh::new();
        mesh.push_quad(&quad(PAGE_0, 0.0, 0.0));
        mesh.push_quad(&quad(PAGE_0, 5.0, 0.0));

        assert_eq!(mesh.vertices().len(), 2 * VERTICES_PER_QUAD);
        assert_eq!(mesh.indices(), &[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
        assert_eq!(mesh.quad_count(), 2);
    }

    #[test]
    fn adjacent_quads_of_a_page_share_a_run() {
        let mut builder = MeshBuilder::new();
        builder.push(&quad(PAGE_0, 0.0, 0.0));
        builder.push(&quad(PAGE_0, 5.0, 0.0));

        assert_eq!(builder.runs().len(), 1);
        assert_eq!(builder.runs()[0].page, PAGE_0);
        assert_eq!(builder.quad_count(), 2);
    }

    #[test]
    fn a_page_change_opens_a_new_run_and_keeps_order() {
        let mut builder = MeshBuilder::new();
        builder.push(&quad(PAGE_0, 0.0, 0.0));
        builder.push(&quad(PAGE_1, 1.0, 0.0));
        builder.push(&quad(PAGE_0, 2.0, 0.0));

        let runs = builder.runs();
        assert_eq!(runs.len(), 3, "the page-0 quads are not adjacent");
        assert_eq!(runs[0].page, PAGE_0);
        assert_eq!(runs[1].page, PAGE_1);
        assert_eq!(runs[2].page, PAGE_0);
        assert_eq!(builder.quad_count(), 3);
        assert_eq!(runs[1].mesh.indices(), &[0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn a_clip_change_opens_a_new_run() {
        let clip = Rect::from_min_size(Offset::new(1.0, 1.0), 4.0, 4.0);
        let mut builder = MeshBuilder::new();
        let mut first = quad(PAGE_0, 0.0, 0.0);
        first.clip = Some(clip);
        let mut second = quad(PAGE_0, 4.0, 0.0);
        second.clip = Some(clip);
        let third = quad(PAGE_0, 8.0, 0.0);

        builder.push(&first);
        builder.push(&second);
        builder.push(&third);

        let runs = builder.runs();
        assert_eq!(runs.len(), 2, "the clipped and unclipped quads differ");
        assert_eq!(runs[0].quad_count(), 2);
        assert_eq!(runs[1].clip, None);
    }

    #[test]
    fn clearing_keeps_the_runs_but_drops_them_from_the_frame() {
        let mut builder = MeshBuilder::new();
        builder.push(&quad(PAGE_0, 0.0, 0.0));
        builder.clear();
        assert!(builder.is_empty());
        assert_eq!(builder.runs().len(), 0);
        assert_eq!(builder.quad_count(), 0);

        builder.push(&quad(PAGE_1, 3.0, 3.0));
        assert_eq!(builder.runs().len(), 1);
        assert_eq!(builder.runs()[0].page, PAGE_1);
        assert_eq!(builder.runs()[0].quad_count(), 1);
    }

    #[test]
    fn a_quad_is_built_from_an_atlas_slot() {
        let slot = GlyphSlot {
            page: PAGE_1,
            uv_min: [0.0, 0.0],
            uv_max: [0.125, 0.25],
            width: 16,
            height: 32,
        };
        let quad = GlyphQuad::from_slot(slot, [2.0, 3.0], [1.0; 4], None);
        assert_eq!(quad.page, PAGE_1);
        assert_eq!(quad.size, [16.0, 32.0]);
        assert_eq!(quad.uv_max, [0.125, 0.25]);
    }
}
