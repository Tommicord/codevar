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

// ARMv7-A cache-prefetch primitives. ARM's PLD preloads data into the
// caches; it has no L2/L3 locality selector, so every hint maps to PLD.
// ABI (AAPCS): r0 = address (and r1 = len for the range helper).

    .syntax unified
    .text
    .arm
    .p2align 4

    .globl codevar_wredit_prefetch_t0
    .type codevar_wredit_prefetch_t0, %function
codevar_wredit_prefetch_t0:
    pld     [r0]
    bx      lr
    .size codevar_wredit_prefetch_t0, .-codevar_wredit_prefetch_t0

    .globl codevar_wredit_prefetch_t1
    .type codevar_wredit_prefetch_t1, %function
codevar_wredit_prefetch_t1:
    pld     [r0]
    bx      lr
    .size codevar_wredit_prefetch_t1, .-codevar_wredit_prefetch_t1

    .globl codevar_wredit_prefetch_t2
    .type codevar_wredit_prefetch_t2, %function
codevar_wredit_prefetch_t2:
    pld     [r0]
    bx      lr
    .size codevar_wredit_prefetch_t2, .-codevar_wredit_prefetch_t2

    .globl codevar_wredit_prefetch_nta
    .type codevar_wredit_prefetch_nta, %function
codevar_wredit_prefetch_nta:
    pld     [r0]
    bx      lr
    .size codevar_wredit_prefetch_nta, .-codevar_wredit_prefetch_nta

// void codevar_wredit_prefetch_range(const void *addr, size_t len)
    .globl codevar_wredit_prefetch_range
    .type codevar_wredit_prefetch_range, %function
codevar_wredit_prefetch_range:
    cmp     r1, #64
    blt     .Lpr_done
.Lpr_loop:
    pld     [r0]
    add     r0, r0, #64
    subs    r1, r1, #64
    cmp     r1, #64
    bge     .Lpr_loop
.Lpr_done:
    bx      lr
    .size codevar_wredit_prefetch_range, .-codevar_wredit_prefetch_range
