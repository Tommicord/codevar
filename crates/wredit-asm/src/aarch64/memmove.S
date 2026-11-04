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

// AArch64 memmove. Assembled twice: the baseline build uses Advanced
// SIMD (NEON), which is mandatory in ARMv8-A; the `-DCODEVAR_HAVE_SVE`
// build emits a scalable SVE version that the Rust side selects at run
// time only when the CPU reports SVE.
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
    .global CODEVAR_FN(codevar_wredit_memmove)
    .type CODEVAR_FN(codevar_wredit_memmove), %function

CODEVAR_FN(codevar_wredit_memmove):
    mov     x8, x0                  // x8 = dst (return value)
    cbz     x2, .Lmv_done
    cmp     x0, x1
    b.hs    .Lmv_backward           // dst >= src: copy high-to-low

#ifdef CODEVAR_HAVE_SVE
// ---------------------------------------------------------------------
// SVE forward copy: predicate-driven, one full vector per iteration.
// ---------------------------------------------------------------------
    mov     x3, #0                  // x3 = byte offset processed
    whilelt p0.b, x3, x2
.Lsve_fwd_loop:
    ld1b    z0.b, p0/z, [x1, x3]
    st1b    z0.b, p0, [x0, x3]
    incb    x3
    whilelt p0.b, x3, x2
    b.first .Lsve_fwd_loop
    b       .Lmv_done

// ---------------------------------------------------------------------
// SVE backward copy: whole vectors from the tail, then the head tail.
// ---------------------------------------------------------------------
.Lmv_backward:
    add     x1, x1, x2
    add     x0, x0, x2
    mov     x3, x2                  // bytes still to copy (from the front)
    cntb    x5                      // bytes per SVE vector
.Lsve_bwd_loop:
    cmp     x3, x5
    b.lt    .Lsve_bwd_head
    sub     x1, x1, x5
    sub     x0, x0, x5
    sub     x3, x3, x5
    ptrue   p0.b
    ld1b    z0.b, p0/z, [x1]
    st1b    z0.b, p0, [x0]
    b       .Lsve_bwd_loop
.Lsve_bwd_head:
    whilelt p0.b, xzr, x3
    sub     x1, x1, x3
    sub     x0, x0, x3
    ld1b    z0.b, p0/z, [x1]
    st1b    z0.b, p0, [x0]
    b       .Lmv_done

#else
// ---------------------------------------------------------------------
// NEON forward copy: 64 bytes (four Q registers) per iteration.
// ---------------------------------------------------------------------
.Lmv_forward:
    cmp     x2, #64
    b.lo    .Lmv_fwd_tail
.Lmv_fwd_loop:
    prfm    pldl1keep, [x1, #128]
    prfm    pstl1keep, [x0, #128]
    ldp     q0, q1, [x1]
    ldp     q2, q3, [x1, #32]
    stp     q0, q1, [x0]
    stp     q2, q3, [x0, #32]
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x2, x2, #64
    b.hs    .Lmv_fwd_loop

.Lmv_fwd_tail:
    cbz     x2, .Lmv_done
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
5:  tbz     x2, #0, .Lmv_done
    ldr     b3, [x1]
    str     b3, [x0]
    b       .Lmv_done

// ---------------------------------------------------------------------
// NEON backward copy: 64 bytes per iteration, walking downwards.
// ---------------------------------------------------------------------
.Lmv_backward:
    add     x1, x1, x2
    add     x0, x0, x2
    cmp     x2, #64
    b.lo    .Lmv_bwd_tail
.Lmv_bwd_loop:
    sub     x9, x1, #128
    prfm    pldl1keep, [x9]
    sub     x10, x0, #128
    prfm    pstl1keep, [x10]
    sub     x1, x1, #64
    sub     x0, x0, #64
    ldp     q0, q1, [x1]
    ldp     q2, q3, [x1, #32]
    stp     q0, q1, [x0]
    stp     q2, q3, [x0, #32]
    subs    x2, x2, #64
    b.hs    .Lmv_bwd_loop

.Lmv_bwd_tail:
    cbz     x2, .Lmv_done
    tbz     x2, #5, 1f
    sub     x1, x1, #32
    sub     x0, x0, #32
    ldp     q0, q1, [x1]
    stp     q0, q1, [x0]
1:  tbz     x2, #4, 2f
    sub     x1, x1, #16
    sub     x0, x0, #16
    ldr     q0, [x1]
    str     q0, [x0]
2:  tbz     x2, #3, 3f
    sub     x1, x1, #8
    sub     x0, x0, #8
    ldr     x3, [x1]
    str     x3, [x0]
3:  tbz     x2, #2, 4f
    sub     x1, x1, #4
    sub     x0, x0, #4
    ldr     w3, [x1]
    str     w3, [x0]
4:  tbz     x2, #1, 5f
    sub     x1, x1, #2
    sub     x0, x0, #2
    ldr     h3, [x1]
    str     h3, [x0]
5:  tbz     x2, #0, .Lmv_done
    sub     x1, x1, #1
    sub     x0, x0, #1
    ldr     b3, [x1]
    str     b3, [x0]
#endif // CODEVAR_HAVE_SVE

.Lmv_done:
    mov     x0, x8
    ret

    .size CODEVAR_FN(codevar_wredit_memmove), .-CODEVAR_FN(codevar_wredit_memmove)
