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

// RISC-V RV64 memcpy (non-overlapping). RVV byte-vector loop when
// -DCODEVAR_HAVE_RVV, else an 8-byte word loop.
// ABI (LP64): a0 = dst, a1 = src, a2 = len; returns a0.

    .text
    .align 3
    .globl codevar_wredit_memcpy
    .type codevar_wredit_memcpy, @function

codevar_wredit_memcpy:
    mv      a3, a0
    beqz    a2, .Lcp_done
    beq     a0, a1, .Lcp_done
#ifdef CODEVAR_HAVE_RVV
.Lcp_rvv:
    vsetvli t0, a2, e8, m8, ta, ma
    vle8.v  v0, (a1)
    vse8.v  v0, (a0)
    add     a1, a1, t0
    add     a0, a0, t0
    sub     a2, a2, t0
    bnez    a2, .Lcp_rvv
    j       .Lcp_done
#else
.Lcp_word:
    li      t0, 8
    bltu    a2, t0, .Lcp_bytes
    ld      a4, 0(a1)
    sd      a4, 0(a0)
    addi    a1, a1, 8
    addi    a0, a0, 8
    addi    a2, a2, -8
    j       .Lcp_word
.Lcp_bytes:
    beqz    a2, .Lcp_done
.Lcp_byte:
    lbu     a4, 0(a1)
    sb      a4, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lcp_byte
#endif

.Lcp_done:
    mv      a0, a3
    ret

    .size codevar_wredit_memcpy, .-codevar_wredit_memcpy
