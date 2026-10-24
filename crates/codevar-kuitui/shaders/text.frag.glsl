#version 450

// Glyph fragment shader.
//
// The font atlas is a single-channel coverage map: the red component
// holds the glyph coverage and is used to modulate the run color's
// alpha, which keeps the atlas four times smaller than an RGBA one.

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform sampler2D u_atlas;

void main() {
    float coverage = texture(u_atlas, v_uv).r;
    out_color = vec4(v_color.rgb, v_color.a * coverage);
}
