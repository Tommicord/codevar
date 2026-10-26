#version 300 es
// A fragment shader has no default float precision in GLSL ES, so it is
// spelled out here. The single output lands at location 0, matching the
// render pipeline's only color target.
precision mediump float;

out vec4 out_color;

void main() {
    out_color = vec4(0.20, 0.55, 0.90, 1.0);
}
