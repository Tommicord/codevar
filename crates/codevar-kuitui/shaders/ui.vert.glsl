#version 450

// UI geometry vertex shader.
//
// Positions arrive in logical pixels with the origin at the top-left of
// the render target. The push-constant viewport maps them to Vulkan's
// clip space, whose +Y axis already points down, so the UI origin needs
// no flip. Vertices carry their color, which lets the same shader draw
// solid fills, lines and polygons, and a linear gradient when the two
// ends of an edge carry the gradient's end colors.

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec4 in_color;

layout(push_constant) uniform PushConstants {
    vec2 viewport;
} pc;

layout(location = 0) out vec4 v_color;

void main() {
    vec2 ndc = in_position / pc.viewport * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color;
}
