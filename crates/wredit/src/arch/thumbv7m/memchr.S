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

// Thumb (M-profile) memchr. On Thumb-2 (__ARM_ARCH >= 7) a word-
// at-a-time zero-byte test scans four bytes per iteration; ARMv6-M
// uses a plain byte scan.
// ABI (AAPCS): r0 = haystack, r1 = needle, r2 = len; returns r0 or 0.

    .syntax unified
    .text
    .thumb
    .p2align 3
    .globl codevar_wredit_memchr
    .type codevar_wredit_memchr, %function
    .thumb_func
codevar_wredit_memchr:
    push    {r4, lr}
    uxtb    r1, r1
    cmp     r2, #0
    beq     .Lmc_none
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
    // r4 = the needle byte repeated in all four bytes of a word.
    orr     r4, r1, r1, lsl #8
    orr     r4, r4, r4, lsl #16
    cmp     r2, #4
    blo     .Lmc_bytes
.Lmc_word:
    ldr     r3, [r0]
    eor     r3, r3, r4
    movw    r12, #0x0101
    movt    r12, #0x0101
    sub     r12, r3, r12        // v - 0x01010101
    mvn     r3, r3              // ~v
    and     r12, r12, r3
    movw    r3, #0x8080
    movt    r3, #0x8080
    ands    r12, r12, r3        // high bit set in a byte => match
    bne     .Lmc_found_word
    adds    r0, r0, #4
    subs    r2, r2, #4
    cmp     r2, #4
    bhs     .Lmc_word
.Lmc_bytes:
#endif
    cmp     r2, #0
    beq     .Lmc_none
1:  ldrb    r3, [r0]
    cmp     r3, r1
    beq     .Lmc_hit
    adds    r0, r0, #1
    subs    r2, r2, #1
    bne     1b
    b       .Lmc_none
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
.Lmc_found_word:
    movs    r3, #4
2:  ldrb    r12, [r0]
    cmp     r12, r1
    beq     .Lmc_hit
    adds    r0, r0, #1
    subs    r3, r3, #1
    bne     2b
#endif
.Lmc_hit:
    pop     {r4, pc}
.Lmc_none:
    movs    r0, #0
    pop     {r4, pc}

    .size codevar_wredit_memchr, .-codevar_wredit_memchr
