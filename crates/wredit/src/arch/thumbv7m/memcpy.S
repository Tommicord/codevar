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

// Thumb (M-profile) memcpy. Non-overlapping, forward only.
// The word loop is guarded for Thumb-2 (__ARM_ARCH >= 7).
// ABI (AAPCS): r0 = dst, r1 = src, r2 = len; returns r0.

    .syntax unified
    .text
    .thumb
    .p2align 3
    .globl codevar_wredit_memcpy
    .type codevar_wredit_memcpy, %function
    .thumb_func
codevar_wredit_memcpy:
    push    {r4, lr}
    mov     r3, r0
    cmp     r2, #0
    beq     .Lcp_done
    cmp     r0, r1
    beq     .Lcp_done
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
    cmp     r2, #4
    blo     .Lcp_bytes
.Lcp_word:
    ldr     r4, [r1]
    str     r4, [r0]
    adds    r1, r1, #4
    adds    r0, r0, #4
    subs    r2, r2, #4
    cmp     r2, #4
    bhs     .Lcp_word
.Lcp_bytes:
#endif
    cmp     r2, #0
    beq     .Lcp_done
1:  ldrb    r4, [r1]
    strb    r4, [r0]
    adds    r1, r1, #1
    adds    r0, r0, #1
    subs    r2, r2, #1
    bne     1b
.Lcp_done:
    mov     r0, r3
    pop     {r4, pc}

    .size codevar_wredit_memcpy, .-codevar_wredit_memcpy
