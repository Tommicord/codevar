// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

// AArch64 memchr: first occurrence of the low byte of `needle` in
// haystack[..len]. NEON is baseline, so no SVE variant is needed; the
// file still honours CODEVAR_SUFFIX so it can be assembled under any
// suffix (used only when the SVE library is built).
//
// ABI: x0 = haystack, w1 = needle, x2 = len; returns x0 = match or 0.

#ifndef CODEVAR_SUFFIX
#define CODEVAR_SUFFIX
#endif

#define CODEVAR_CAT2(a, b) a##b
#define CODEVAR_CAT(a, b) CODEVAR_CAT2(a, b)
#define CODEVAR_FN(name) CODEVAR_CAT(name, CODEVAR_SUFFIX)

    .text
    .align 6
    .global CODEVAR_FN(codevar_wredit_memchr)
    .type CODEVAR_FN(codevar_wredit_memchr), %function

CODEVAR_FN(codevar_wredit_memchr):
    and     w1, w1, #0xff           // clear unspecified high bits
    cbz     x2, .Lmc_none
    dup     v0.16b, w1              // broadcast the needle byte
    cmp     x2, #16
    b.lo    .Lmc_scan

.Lmc_loop:
    prfm    pldl1keep, [x0, #128]
    ldr     q1, [x0]
    cmeq    v1.16b, v1.16b, v0.16b
    umaxv   b2, v1.16b              // 0xFF if any lane matched
    umov    w3, v2.b[0]
    cbnz    w3, .Lmc_found
    add     x0, x0, #16
    subs    x2, x2, #16
    b.hs    .Lmc_loop

.Lmc_scan:
    cbz     x2, .Lmc_none
1:  ldrb    w3, [x0]
    cmp     w3, w1
    b.eq    .Lmc_found_byte
    add     x0, x0, #1
    subs    x2, x2, #1
    b.ne    1b
    b       .Lmc_none

.Lmc_found:
    // Locate the exact byte inside the 16-byte chunk.
    ldr     q1, [x0]
    cmeq    v1.16b, v1.16b, v0.16b
    umov    x3, v1.d[0]             // low 64-bit half
    cbnz    x3, 2f
    umov    x3, v1.d[1]             // high half
    add     x4, x0, #8
    b       3f
2:  mov     x4, x0
3:  // Each matching byte is 0xFF, so the lowest set bit sits at
    // byte_index * 8; rbit+clz gives that bit index.
    rbit    x3, x3
    clz     x3, x3
    lsr     x3, x3, #3
    add     x0, x4, x3
    ret

.Lmc_found_byte:
    ret

.Lmc_none:
    mov     x0, #0
    ret

    .size CODEVAR_FN(codevar_wredit_memchr), .-CODEVAR_FN(codevar_wredit_memchr)
