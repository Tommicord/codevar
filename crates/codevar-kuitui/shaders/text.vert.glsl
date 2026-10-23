#version 450

// Glyph-quad vertex shader.
//
// Each glyph is a textured quad; the atlas coordinates are produced by
// the font rasterizer and the colour is the run's text colour. Like the
// geometry shader, positions are logical pixels with a top-left origin.

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;

layout(push_constant) uniform PushConstants {
    vec2 viewport;
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;

void main() {
    vec2 ndc = in_position / pc.viewport * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
}
