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

// Compositor mix pass: folds an array of offscreen framebuffers into the
// presentation target with `mix()`. Every fragment invocation runs this
// independently, so the whole framebuffer array is blended in parallel.
#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

// All slots are always bound (unused ones alias the first framebuffer),
// which keeps every descriptor statically used and indexable at runtime.
layout(set = 0, binding = 0) uniform sampler2D u_frames[8];

// The weights travel as two `vec4`s with explicit offsets so the block
// layout is identical under the std140, std430 and scalar rules; a plain
// `float[8]` would change stride depending on the rule the compiler picks.
layout(push_constant) uniform MixPush {
    uint count;
    layout(offset = 16) vec4 weights0;
    layout(offset = 32) vec4 weights1;
} u_pc;

float mix_weight(uint index) {
    if (index < 4u) {
        return clamp(u_pc.weights0[int(index)], 0.0, 1.0);
    }
    return clamp(u_pc.weights1[int(index - 4u)], 0.0, 1.0);
}

void main() {
    if (u_pc.count == 0u) {
        o_color = vec4(0.0);
        return;
    }

    vec4 color = texture(u_frames[0], v_uv);
    // `u_pc.count` is a push constant, so the index below is dynamically
    // uniform and `u_frames` needs no `nonuniform` decoration.
    for (uint i = 1u; i < u_pc.count; ++i) {
        color = mix(color, texture(u_frames[i], v_uv), mix_weight(i));
    }
    o_color = color;
}
