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

// ARMv7-A memchr: first occurrence of the low byte of r1.
// With CODEVAR_HAVE_NEON the scan compares 16 bytes with `vceq.i8` and
// locates the first hit with `rbit`/`clz`; otherwise it scans bytes.
// ABI (AAPCS): r0 = haystack, r1 = needle, r2 = len; returns r0 or 0.

    .syntax unified
    .text
    .arm
    .p2align 5
    .globl codevar_wredit_memchr
    .type codevar_wredit_memchr, %function

codevar_wredit_memchr:
    and     r1, r1, #0xff
    cmp     r2, #0
    moveq   r0, #0
    bxeq    lr

#ifdef CODEVAR_HAVE_NEON
    vdup.8  q0, r1
    cmp     r2, #16
    blo     .Lmc_bytes
.Lmc_loop:
    pld     [r0, #128]
    vld1.8  {q1}, [r0]
    vceq.i8 q1, q1, q0
    // Reduce the per-byte mask to one value: three pairwise maxima
    // leave every lane equal to 0xFF when any byte matched.
    vpmax.u8 d2, d2, d3
    vpmax.u8 d2, d2, d2
    vpmax.u8 d2, d2, d2
    vpmax.u8 d2, d2, d2
    vmov    r5, s4
    ands    r5, r5, #0xff
    bne     .Lmc_found
    add     r0, r0, #16
    subs    r2, r2, #16
    cmp     r2, #16
    bhs     .Lmc_loop
    b       .Lmc_bytes

.Lmc_found:
    // A match is guaranteed within this 16-byte chunk; locate it with a
    // short byte scan (fewer edge cases than extracting NEON lanes).
    mov     r7, #16
1:  ldrb    r5, [r0], #1
    cmp     r5, r1
    beq     2f
    subs    r7, r7, #1
    bne     1b
2:  sub     r0, r0, #1
    bx      lr
#endif

.Lmc_bytes:
    cmp     r2, #0
    moveq   r0, #0
    bxeq    lr
.Lmc_byte_loop:
    ldrb    r5, [r0]
    cmp     r5, r1
    beq     .Lmc_byte_hit
    add     r0, r0, #1
    subs    r2, r2, #1
    bne     .Lmc_byte_loop
    mov     r0, #0
    bx      lr
.Lmc_byte_hit:
    bx      lr

    .size codevar_wredit_memchr, .-codevar_wredit_memchr
