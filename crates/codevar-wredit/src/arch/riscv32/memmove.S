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

// RISC-V RV32 memmove. Same structure as riscv64 but with 4-byte words.
// Forward copies use the RVV byte-vector loop when -DCODEVAR_HAVE_RVV.
// ABI (ILP32): a0 = dst, a1 = src, a2 = len; returns a0.

    .text
    .align 2
    .globl codevar_wredit_memmove
    .type codevar_wredit_memmove, @function

codevar_wredit_memmove:
    mv      a3, a0
    beqz    a2, .Lmm_done
    beq     a0, a1, .Lmm_done
    bltu    a0, a1, .Lmm_forward

    add     a1, a1, a2
    add     a0, a0, a2
.Lmm_bwd_word:
    li      t0, 4
    bltu    a2, t0, .Lmm_bwd_bytes
    addi    a1, a1, -4
    addi    a0, a0, -4
    lw      a4, 0(a1)
    sw      a4, 0(a0)
    addi    a2, a2, -4
    j       .Lmm_bwd_word
.Lmm_bwd_bytes:
    beqz    a2, .Lmm_done
.Lmm_bwd_byte:
    addi    a1, a1, -1
    addi    a0, a0, -1
    lbu     a4, 0(a1)
    sb      a4, 0(a0)
    addi    a2, a2, -1
    bnez    a2, .Lmm_bwd_byte
    j       .Lmm_done

.Lmm_forward:
#ifdef CODEVAR_HAVE_RVV
.Lmm_fwd_rvv:
    vsetvli t0, a2, e8, m8, ta, ma
    vle8.v  v0, (a1)
    vse8.v  v0, (a0)
    add     a1, a1, t0
    add     a0, a0, t0
    sub     a2, a2, t0
    bnez    a2, .Lmm_fwd_rvv
    j       .Lmm_done
#else
.Lmm_fwd_word:
    li      t0, 4
    bltu    a2, t0, .Lmm_fwd_bytes
    lw      a4, 0(a1)
    sw      a4, 0(a0)
    addi    a1, a1, 4
    addi    a0, a0, 4
    addi    a2, a2, -4
    j       .Lmm_fwd_word
.Lmm_fwd_bytes:
    beqz    a2, .Lmm_done
.Lmm_fwd_byte:
    lbu     a4, 0(a1)
    sb      a4, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lmm_fwd_byte
#endif

.Lmm_done:
    mv      a0, a3
    ret

    .size codevar_wredit_memmove, .-codevar_wredit_memmove
