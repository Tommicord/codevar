#version 450

// Layer-effect vertex shader.
//
// The geometry of an effect layer is the plain UI quad: the rotation,
// the blur and the pixelation are all evaluated per fragment from the
// parameters of the layer, so the vertices stay untouched and the same
// vertex buffer serves both programs. The position is forwarded because
// the fragment shader needs it in the coordinate space of the render
// target, where the layer rectangle is expressed.

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec4 in_color;

layout(push_constant) uniform PushConstants {
    vec4 color_from;
    vec4 color_to;
    vec2 viewport;
    vec2 origin;
    vec2 size;
    vec2 axis_start;
    vec2 axis_end;
    vec2 blur_pixel;
    vec4 radii;
    float rotation;
    uint kinds;
} pc;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_position;

void main() {
    vec2 ndc = in_position / pc.viewport * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color;
    v_position = in_position;
}
