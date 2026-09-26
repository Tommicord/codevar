#version 450
layout(location = 0) out vec3 vColor;
void main() {
    vec2 pos[3] = vec2[](vec2(0.0, -0.6), vec2(0.6, 0.6), vec2(-0.6, 0.6));
    vec3 col[3] = vec3[](vec3(1.0, 0.2, 0.1), vec3(0.1, 1.0, 0.2), vec3(0.2, 0.3, 1.0));
    gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
    vColor = col[gl_VertexIndex];
}
