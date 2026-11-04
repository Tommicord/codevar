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

// AArch64 memcpy (non-overlapping). NEON baseline plus an SVE variant
// behind `-DCODEVAR_HAVE_SVE`; see memmove.S for the suffix contract.
//
// ABI: x0 = dst, x1 = src, x2 = len; returns x0 = dst.

#ifndef CODEVAR_SUFFIX
#define CODEVAR_SUFFIX
#endif

#define CODEVAR_CAT2(a, b) a##b
#define CODEVAR_CAT(a, b) CODEVAR_CAT2(a, b)
#define CODEVAR_FN(name) CODEVAR_CAT(name, CODEVAR_SUFFIX)

    .text
    .align 6
    .global CODEVAR_FN(codevar_wredit_memcpy)
    .type CODEVAR_FN(codevar_wredit_memcpy), %function

CODEVAR_FN(codevar_wredit_memcpy):
    mov     x8, x0
    cbz     x2, .Lcp_done

#ifdef CODEVAR_HAVE_SVE
    mov     x3, #0
    whilelt p0.b, x3, x2
.Lsve_loop:
    ld1b    z0.b, p0/z, [x1, x3]
    st1b    z0.b, p0, [x0, x3]
    incb    x3
    whilelt p0.b, x3, x2
    b.first .Lsve_loop
    b       .Lcp_done
#else
    cmp     x2, #64
    b.lo    .Lcp_tail
.Lcp_loop:
    prfm    pldl1keep, [x1, #128]
    prfm    pstl1keep, [x0, #128]
    ldp     q0, q1, [x1]
    ldp     q2, q3, [x1, #32]
    stp     q0, q1, [x0]
    stp     q2, q3, [x0, #32]
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x2, x2, #64
    b.hs    .Lcp_loop
.Lcp_tail:
    cbz     x2, .Lcp_done
    tbz     x2, #5, 1f
    ldp     q0, q1, [x1]
    stp     q0, q1, [x0]
    add     x1, x1, #32
    add     x0, x0, #32
1:  tbz     x2, #4, 2f
    ldr     q0, [x1]
    str     q0, [x0]
    add     x1, x1, #16
    add     x0, x0, #16
2:  tbz     x2, #3, 3f
    ldr     x3, [x1]
    str     x3, [x0]
    add     x1, x1, #8
    add     x0, x0, #8
3:  tbz     x2, #2, 4f
    ldr     w3, [x1]
    str     w3, [x0]
    add     x1, x1, #4
    add     x0, x0, #4
4:  tbz     x2, #1, 5f
    ldr     h3, [x1]
    str     h3, [x0]
    add     x1, x1, #2
    add     x0, x0, #2
5:  tbz     x2, #0, .Lcp_done
    ldr     b3, [x1]
    str     b3, [x0]
#endif // CODEVAR_HAVE_SVE

.Lcp_done:
    mov     x0, x8
    ret

    .size CODEVAR_FN(codevar_wredit_memcpy), .-CODEVAR_FN(codevar_wredit_memcpy)
