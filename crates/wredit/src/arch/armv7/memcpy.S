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

// ARMv7-A memcpy (non-overlapping). NEON bulk when CODEVAR_HAVE_NEON,
// word/byte tail otherwise. ABI (AAPCS): r0 = dst, r1 = src, r2 = len.

    .syntax unified
    .text
    .arm
    .p2align 5
    .globl codevar_wredit_memcpy
    .type codevar_wredit_memcpy, %function

codevar_wredit_memcpy:
    mov     r3, r0
    cmp     r2, #0
    bxeq    lr
#ifdef CODEVAR_HAVE_NEON
    cmp     r2, #64
    blo     .Lcp_small
.Lcp_loop:
    pld     [r1, #128]
    vld1.8  {q0, q1}, [r1]!
    vld1.8  {q2, q3}, [r1]!
    vst1.8  {q0, q1}, [r0]!
    vst1.8  {q2, q3}, [r0]!
    subs    r2, r2, #64
    cmp     r2, #64
    bhs     .Lcp_loop
.Lcp_small:
#endif
    cmp     r2, #4
    blo     .Lcp_bytes
.Lcp_word:
    ldr     r5, [r1], #4
    str     r5, [r0], #4
    subs    r2, r2, #4
    cmp     r2, #4
    bhs     .Lcp_word
.Lcp_bytes:
    cmp     r2, #0
    beq     .Lcp_done
.Lcp_byte:
    ldrb    r5, [r1], #1
    strb    r5, [r0], #1
    subs    r2, r2, #1
    bne     .Lcp_byte
.Lcp_done:
    mov     r0, r3
    bx      lr

    .size codevar_wredit_memcpy, .-codevar_wredit_memcpy
