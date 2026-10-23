#version 450

// UI geometry fragment shader: straight per-vertex colour. Because the
// colour is interpolated across the primitive, the same program renders
// a flat fill and a linear gradient.

layout(location = 0) in vec4 v_color;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = v_color;
}
