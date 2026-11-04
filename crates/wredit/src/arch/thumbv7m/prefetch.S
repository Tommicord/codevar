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

// Thumb (M-profile) prefetch primitives. PLD exists from ARMv7-M; on
// ARMv6-M these are harmless no-ops. PLD has no locality selector, so
// every hint maps to the same instruction.
// ABI (AAPCS): r0 = address (and r1 = len for the range helper).

    .syntax unified
    .text
    .thumb
    .p2align 3

    .globl codevar_wredit_prefetch_t0
    .type codevar_wredit_prefetch_t0, %function
    .thumb_func
codevar_wredit_prefetch_t0:
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
    pld     [r0]
#endif
    bx      lr
    .size codevar_wredit_prefetch_t0, .-codevar_wredit_prefetch_t0

    .globl codevar_wredit_prefetch_t1
    .type codevar_wredit_prefetch_t1, %function
    .thumb_func
codevar_wredit_prefetch_t1:
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
    pld     [r0]
#endif
    bx      lr
    .size codevar_wredit_prefetch_t1, .-codevar_wredit_prefetch_t1

    .globl codevar_wredit_prefetch_t2
    .type codevar_wredit_prefetch_t2, %function
    .thumb_func
codevar_wredit_prefetch_t2:
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
    pld     [r0]
#endif
    bx      lr
    .size codevar_wredit_prefetch_t2, .-codevar_wredit_prefetch_t2

    .globl codevar_wredit_prefetch_nta
    .type codevar_wredit_prefetch_nta, %function
    .thumb_func
codevar_wredit_prefetch_nta:
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
    pld     [r0]
#endif
    bx      lr
    .size codevar_wredit_prefetch_nta, .-codevar_wredit_prefetch_nta

// void codevar_wredit_prefetch_range(const void *addr, size_t len)
    .globl codevar_wredit_prefetch_range
    .type codevar_wredit_prefetch_range, %function
    .thumb_func
codevar_wredit_prefetch_range:
#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
    cmp     r1, #64
    blo     .Lpr_done
.Lpr_loop:
    pld     [r0]
    adds    r0, r0, #64
    subs    r1, r1, #64
    cmp     r1, #64
    bhs     .Lpr_loop
.Lpr_done:
#endif
    bx      lr
    .size codevar_wredit_prefetch_range, .-codevar_wredit_prefetch_range
