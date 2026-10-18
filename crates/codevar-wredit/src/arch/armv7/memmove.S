/* Copyright 2026 Codevar Project
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// ARMv7-A (A32) memmove.
//
// With -DCODEVAR_HAVE_NEON the bulk uses NEON `vld1.8 {q0,q1}` /
// `vst1.8` (32 bytes per pair of quad registers); otherwise, and for
// every tail, it falls back to word copies. `pld` warms the cache.
// Overlap is handled by copying forward when r0 < r1 and backward when
// r0 > r1.
//
// ABI (AAPCS): r0 = dst, r1 = src, r2 = len; returns r0.

    .syntax unified
    .text
    .arm
    .p2align 5
    .globl codevar_wredit_memmove
    .type codevar_wredit_memmove, %function

codevar_wredit_memmove:
    mov     r3, r0                  // return value
    cmp     r2, #0
    bxeq    lr
    cmp     r0, r1
    bxeq    lr
    blo     .Lmv_forward
    b       .Lmv_backward

// ---------------------------------------------------------------------
// Backward: r0 > r1.
// ---------------------------------------------------------------------
.Lmv_backward:
    add     r1, r1, r2
    add     r0, r0, r2
#ifdef CODEVAR_HAVE_NEON
    cmp     r2, #64
    blo     .Lmv_bwd_small
.Lmv_bwd_loop:
    pld     [r1, #-128]
    sub     r1, r1, #64
    sub     r0, r0, #64
    vld1.8  {q0, q1}, [r1]!
    vld1.8  {q2, q3}, [r1]!
    vst1.8  {q0, q1}, [r0]!
    vst1.8  {q2, q3}, [r0]!
    subs    r2, r2, #64
    cmp     r2, #64
    bhs     .Lmv_bwd_loop
.Lmv_bwd_small:
#endif
    cmp     r2, #4
    blo     .Lmv_bwd_bytes
.Lmv_bwd_word:
    sub     r1, r1, #4
    sub     r0, r0, #4
    ldr     r5, [r1]
    str     r5, [r0]
    subs    r2, r2, #4
    cmp     r2, #4
    bhs     .Lmv_bwd_word
.Lmv_bwd_bytes:
    cmp     r2, #0
    beq     .Lmv_done
.Lmv_bwd_byte:
    sub     r1, r1, #1
    sub     r0, r0, #1
    ldrb    r5, [r1]
    strb    r5, [r0]
    subs    r2, r2, #1
    bne     .Lmv_bwd_byte
    b       .Lmv_done

// ---------------------------------------------------------------------
// Forward: r0 < r1.
// ---------------------------------------------------------------------
.Lmv_forward:
#ifdef CODEVAR_HAVE_NEON
    cmp     r2, #64
    blo     .Lmv_fwd_small
.Lmv_fwd_loop:
    pld     [r1, #128]
    vld1.8  {q0, q1}, [r1]!
    vld1.8  {q2, q3}, [r1]!
    vst1.8  {q0, q1}, [r0]!
    vst1.8  {q2, q3}, [r0]!
    subs    r2, r2, #64
    cmp     r2, #64
    bhs     .Lmv_fwd_loop
.Lmv_fwd_small:
#endif
    cmp     r2, #4
    blo     .Lmv_fwd_bytes
.Lmv_fwd_word:
    ldr     r5, [r1], #4
    str     r5, [r0], #4
    subs    r2, r2, #4
    cmp     r2, #4
    bhs     .Lmv_fwd_word
.Lmv_fwd_bytes:
    cmp     r2, #0
    beq     .Lmv_done
.Lmv_fwd_byte:
    ldrb    r5, [r1], #1
    strb    r5, [r0], #1
    subs    r2, r2, #1
    bne     .Lmv_fwd_byte

.Lmv_done:
    mov     r0, r3
    bx      lr

    .size codevar_wredit_memmove, .-codevar_wredit_memmove
