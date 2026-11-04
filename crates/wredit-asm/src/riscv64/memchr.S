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

// RISC-V RV64 memchr: first occurrence of the low byte of `needle`.
// With -DCODEVAR_HAVE_RVV the scan is vectorised (vmseq + vfirst.m);
// otherwise it is a byte loop.
// ABI (LP64): a0 = haystack, a1 = needle, a2 = len; returns a0 or 0.

    .text
    .align 3
    .globl codevar_wredit_memchr
    .type codevar_wredit_memchr, @function

codevar_wredit_memchr:
    andi    a1, a1, 0xff
    beqz    a2, .Lmc_none
#ifdef CODEVAR_HAVE_RVV
.Lmc_rvv:
    vsetvli t0, a2, e8, m8, ta, ma
    vle8.v  v0, (a0)
    vmv.v.x v1, a1
    vmseq.vv v2, v0, v1
    vfirst.m t1, v2
    bgez    t1, .Lmc_rvv_found
    add     a0, a0, t0
    sub     a2, a2, t0
    bnez    a2, .Lmc_rvv
    j       .Lmc_none
.Lmc_rvv_found:
    add     a0, a0, t1
    ret
#else
.Lmc_byte:
    lbu     t0, 0(a0)
    beq     t0, a1, .Lmc_hit
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lmc_byte
#endif

.Lmc_none:
    li      a0, 0
    ret
.Lmc_hit:
    ret

    .size codevar_wredit_memchr, .-codevar_wredit_memchr
