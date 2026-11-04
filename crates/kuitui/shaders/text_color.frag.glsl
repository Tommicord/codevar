#version 450

// Color glyph fragment shader.
//
// Samples RGBA atlas and multiplies by vertex color (white for emoji).
// Output is premultiplied for ONE/ONE_MINUS_SRC_ALPHA blending.

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform sampler2D u_atlas;

void main() {
    vec4 texel = texture(u_atlas, v_uv);
    out_color = texel * v_color;
}