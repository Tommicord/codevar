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

// Thumb (M-profile) memmove for Cortex-M. The wide word loop needs
// Thumb-2, so it is guarded by __ARM_ARCH >= 7; ARMv6-M (Cortex-M0/M0+)
// falls back to the byte loop. There is no NEON on M-profile.
//
// ABI (AAPCS): r0 = dst, r1 = src, r2 = len; returns r0.

    .syntax unified
    .text
    .thumb
    .p2align 3
    .globl codevar_wredit_memmove
    .type codevar_wredit_memmove, %function
    .thumb_func
codevar_wredit_memmove:
    push    {r4, lr}
    mov     r3, r0                  // return value
    cmp     r2, #0
    beq     .Lmv_done
    cmp     r0, r1
    beq     .Lmv_done
    bhi     .Lmv_backward

// ---------------------------------------------------------------------
// Forward (r0 < r1).
// ---------------------------------------------------------------------
.Lmv_forward:
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
    cmp     r2, #4
    blo     .Lmv_fwd_bytes
.Lmv_fwd_word:
    ldr     r4, [r1]
    str     r4, [r0]
    adds    r1, r1, #4
    adds    r0, r0, #4
    subs    r2, r2, #4
    cmp     r2, #4
    bhs     .Lmv_fwd_word
.Lmv_fwd_bytes:
#endif
    cmp     r2, #0
    beq     .Lmv_done
1:  ldrb    r4, [r1]
    strb    r4, [r0]
    adds    r1, r1, #1
    adds    r0, r0, #1
    subs    r2, r2, #1
    bne     1b
    b       .Lmv_done

// ---------------------------------------------------------------------
// Backward (r0 > r1): walk from the high end down.
// ---------------------------------------------------------------------
.Lmv_backward:
    adds    r1, r1, r2
    adds    r0, r0, r2
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
    cmp     r2, #4
    blo     .Lmv_bwd_bytes
.Lmv_bwd_word:
    subs    r1, r1, #4
    subs    r0, r0, #4
    ldr     r4, [r1]
    str     r4, [r0]
    subs    r2, r2, #4
    cmp     r2, #4
    bhs     .Lmv_bwd_word
.Lmv_bwd_bytes:
#endif
    cmp     r2, #0
    beq     .Lmv_done
1:  subs    r1, r1, #1
    subs    r0, r0, #1
    ldrb    r4, [r1]
    strb    r4, [r0]
    subs    r2, r2, #1
    bne     1b

.Lmv_done:
    mov     r0, r3
    pop     {r4, pc}

    .size codevar_wredit_memmove, .-codevar_wredit_memmove
