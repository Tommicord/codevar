// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

// Fullscreen triangle for the compositor's mix pass: no vertex
// buffer, three vertices generated from `gl_VertexIndex`.
#version 450

layout(location = 0) out vec2 v_uv;

void main() {
    vec2 uv = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    // Vulkan's normalized texture coordinates have (0, 0) at the top-left
    // of the image while clip space has (-1, -1) at the bottom-left, so
    // the vertical axis is flipped here.
    v_uv = vec2(uv.x, 1.0 - uv.y);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
