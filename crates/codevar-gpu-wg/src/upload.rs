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

//! The bytes that reach the device.
//!
//! The vertices of a frame and the coverage of the atlas are written
//! by hand, so that the crate does not depend on a crate that casts a
//! slice of structures into a slice of bytes.

use crate::mesh::Vertex;

/// The layout of the vertices of the mesh.
pub(crate) fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: core::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: core::mem::offset_of!(Vertex, position) as u64,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: core::mem::offset_of!(Vertex, shape) as u64,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: core::mem::offset_of!(Vertex, radii) as u64,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: core::mem::offset_of!(Vertex, color) as u64,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: core::mem::offset_of!(Vertex, uv) as u64,
                shader_location: 4,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: core::mem::offset_of!(Vertex, params) as u64,
                shader_location: 5,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: core::mem::offset_of!(Vertex, clip) as u64,
                shader_location: 6,
            },
        ],
    }
}

/// The bytes of the vertices of a mesh.
///
/// The vertices are written by hand so that the crate does not need a
/// crate that casts a slice of structures into a slice of bytes.
pub(crate) fn vertex_bytes(vertices: &[Vertex]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(vertices));
    for vertex in vertices {
        for value in vertex.position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in vertex.shape {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in vertex.radii {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in vertex.color {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in vertex.uv {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in vertex.params {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in vertex.clip {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// Pads the coverage of the atlas to a whole number of rows.
///
/// A texture cannot be copied row by row unless every row starts at a
/// multiple of the block size, so the coverage of a row that is
/// shorter than the row of the texture is padded with zeroes.
pub(crate) fn pad_coverage(coverage: &[u8], width: u32, height: u32) -> Vec<u8> {
    let stride = width as usize;
    let rows = height as usize;
    let mut padded = Vec::with_capacity(stride * rows);
    for row in 0..rows {
        let start = row * stride;
        let end = (start + stride).min(coverage.len());
        if start < coverage.len() {
            padded.extend_from_slice(&coverage[start..end]);
        }
        padded.resize((row + 1) * stride, 0);
    }
    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bytes_of_a_vertex_hold_every_field() {
        let vertex = Vertex {
            position: [1.0, 2.0],
            shape: [3.0, 4.0, 5.0, 6.0],
            radii: [0.0; 4],
            color: [0.25, 0.5, 0.75, 1.0],
            uv: [0.1, 0.2],
            params: [1.0, 2.0],
            clip: [0.0, 0.0, 10.0, 10.0],
        };
        let bytes = vertex_bytes(std::slice::from_ref(&vertex));
        assert_eq!(bytes.len(), core::mem::size_of::<Vertex>());
        assert_eq!(&bytes[0..4], &1.0_f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &3.0_f32.to_le_bytes());
    }

    #[test]
    fn the_coverage_of_the_atlas_is_padded_to_a_whole_row() {
        let padded = pad_coverage(&[1, 2, 3], 4, 2);
        assert_eq!(padded.len(), 8);
        assert_eq!(&padded[0..4], &[1, 2, 3, 0]);
        assert_eq!(&padded[4..8], &[0, 0, 0, 0]);
    }

    #[test]
    fn the_layout_of_the_vertices_matches_the_shader() {
        let layout = vertex_layout();
        assert_eq!(layout.array_stride, core::mem::size_of::<Vertex>() as u64);
        assert_eq!(layout.attributes.len(), 7);
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(
            layout.attributes[6].offset,
            core::mem::offset_of!(Vertex, clip) as u64
        );
    }
}
