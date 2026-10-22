// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

// The shader of the Kuitui user interface.
//
// Every primitive of a frame is a quad: the vertex shader turns the
// corners of the quad into clip coordinates, and the fragment shader
// decides which fragments of the quad are visible. The three kinds of
// primitive share one pipeline:
//
//   * a shape is a rounded rectangle, filled or outlined, and the
//     fragment shader computes the signed distance to it;
//   * a line is a segment, and the fragment shader computes the
//     distance to it;
//   * a glyph samples the coverage of the glyph atlas.
//
// The signed distance is what makes the rounded corners and the
// strokes antialiased: a fragment is drawn when the distance is
// inside the shape, and the edge of the shape is covered by the
// fragments whose distance is close to zero.

/// The uniform of a frame.
struct Uniforms {
    /// The size of the target, in device pixels.
    viewport: vec2<f32>,
    /// The number of device pixels per logical pixel.
    scale: f32,
    /// Padding, so that the uniform is a multiple of sixteen bytes.
    _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;

/// The vertex of a quad.
struct VertexInput {
    /// The corner of the quad, in logical pixels.
    @location(0) position: vec2<f32>,
    /// The shape the corner belongs to: the origin and the size of a
    /// rectangle, or the two ends of a line.
    @location(1) shape: vec4<f32>,
    /// The radii of the four corners of a rectangle.
    @location(2) radii: vec4<f32>,
    /// The color of the primitive, not premultiplied.
    @location(3) color: vec4<f32>,
    /// The texture coordinate of the corner.
    @location(4) uv: vec2<f32>,
    /// The stroke width of the primitive and the kind of the primitive.
    @location(5) params: vec2<f32>,
    /// The clipping rectangle of the primitive, in logical pixels.
    @location(6) clip: vec4<f32>,
};

/// The values the fragment shader receives.
struct VertexOutput {
    /// The position of the fragment, in clip coordinates.
    @builtin(position) position: vec4<f32>,
    /// The position of the fragment, relative to the shape.
    @location(0) local: vec2<f32>,
    /// The shape the fragment belongs to.
    @location(1) @interpolate(flat) shape: vec4<f32>,
    /// The radii of the corners of the shape.
    @location(2) @interpolate(flat) radii: vec4<f32>,
    /// The color of the primitive.
    @location(3) @interpolate(flat) color: vec4<f32>,
    /// The texture coordinate of the fragment.
    @location(4) uv: vec2<f32>,
    /// The stroke width of the primitive and the kind of the primitive.
    @location(5) @interpolate(flat) params: vec2<f32>,
    /// The clipping rectangle of the primitive.
    @location(6) @interpolate(flat) clip: vec4<f32>,
};

/// The kind of a primitive that is a shape.
const KIND_SHAPE: f32 = 0.0;
/// The kind of a primitive that is a glyph.
const KIND_GLYPH: f32 = 1.0;
/// The kind of a primitive that is a line.
const KIND_LINE: f32 = 2.0;

/// Turns the corners of a quad into clip coordinates.
///
/// The origin of the coordinate system is the top-left corner of the
/// target and the y axis points down, so the vertical coordinate is
/// flipped on the way to the clip space.
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let viewport = max(uniforms.viewport, vec2<f32>(1.0, 1.0));
    let device = input.position * uniforms.scale;
    let ndc = vec2<f32>(
        device.x / viewport.x * 2.0 - 1.0,
        1.0 - device.y / viewport.y * 2.0,
    );
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.local = input.position - input.shape.xy;
    out.shape = input.shape;
    out.radii = input.radii;
    out.color = input.color;
    out.uv = input.uv;
    out.params = input.params;
    out.clip = input.clip;
    return out;
}

/// The signed distance to a rounded rectangle.
///
/// The point is relative to the center of the rectangle, and the
/// distance is negative inside the rectangle. The radius of the corner
/// the point falls in decides the shape of that corner.
fn rounded_box_distance(point: vec2<f32>, half_size: vec2<f32>, radii: vec4<f32>) -> f32 {
    var radius: f32;
    if (point.x > 0.0) {
        if (point.y > 0.0) {
            radius = radii.z;
        } else {
            radius = radii.y;
        }
    } else if (point.y > 0.0) {
        radius = radii.w;
    } else {
        radius = radii.x;
    }
    radius = min(radius, min(half_size.x, half_size.y));
    let corner = abs(point) - half_size + vec2<f32>(radius, radius);
    return min(max(corner.x, corner.y), 0.0) + length(max(corner, vec2<f32>(0.0))) - radius;
}

/// The distance from a point to a segment.
fn segment_distance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
    let to_point = point - start;
    let segment = end - start;
    let length_squared = dot(segment, segment);
    if (length_squared <= 0.0) {
        return length(to_point);
    }
    let along = clamp(dot(to_point, segment) / length_squared, 0.0, 1.0);
    return length(to_point - segment * along);
}

/// Draws a fragment of a quad.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let absolute = in.local + in.shape.xy;
    if (absolute.x < in.clip.x || absolute.y < in.clip.y
        || absolute.x > in.clip.x + in.clip.z || absolute.y > in.clip.y + in.clip.w) {
        discard;
    }

    let kind = in.params.y;
    var coverage = 1.0;
    if (kind > 0.5 && kind < 1.5) {
        // A glyph: the atlas holds the coverage of the glyph.
        coverage = textureSampleLevel(atlas, atlas_sampler, in.uv, 0.0).r;
    } else if (kind < 0.5) {
        // A shape: the signed distance says whether the fragment is
        // inside the rounded rectangle.
        let half_size = in.shape.zw * 0.5;
        let distance = rounded_box_distance(in.local - half_size, half_size, in.radii);
        let stroke = in.params.x;
        if (stroke > 0.0) {
            if (abs(distance) > stroke * 0.5) {
                discard;
            }
        } else if (distance > 0.0) {
            discard;
        }
    } else {
        // A line: the distance to the segment says whether the fragment
        // is on the line.
        let distance = segment_distance(in.local, vec2<f32>(0.0, 0.0), in.shape.zw - in.shape.xy);
        if (distance > in.params.x * 0.5) {
            discard;
        }
    }

    let alpha = in.color.a * coverage;
    return vec4<f32>(in.color.rgb * alpha, alpha);
}
