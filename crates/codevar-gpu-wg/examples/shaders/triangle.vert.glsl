#version 300 es
// The placeholder triangle of the Codevar graphics stack.
//
// The three clip-space vertices are hard-coded and selected by
// `gl_VertexID`, so the pipeline needs neither a vertex buffer nor a
// descriptor set. `codevar-gpu-wg` passes GLSL through to the browser
// unchanged, which is why this is GLSL ES 3.00 rather than WGSL: the
// WebGL2 backend consumes it directly and needs no naga translation.
const vec2 VERTICES[3] = vec2[3](
    vec2(0.0, 0.5),
    vec2(-0.5, -0.5),
    vec2(0.5, -0.5)
);

void main() {
    gl_Position = vec4(VERTICES[gl_VertexID], 0.0, 1.0);
}
