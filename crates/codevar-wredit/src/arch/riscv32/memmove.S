/* Copyright 2026 Codevar Project
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing
   permissions and limitations under the License. */

// RISC-V RV32 memmove, 4-byte words. See riscv64/memmove.S for the
// design notes; this is the ILP32 twin with `lw`/`sw` and 8-word
// (32 byte) blocks.
//
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
    li      t0, 32
    bltu    a2, t0, .Lmm_bwd_tail
.Lmm_bwd_loop:
    addi    a1, a1, -32
    addi    a0, a0, -32
    lw      a4, 0(a1)
    lw      a5, 4(a1)
    lw      a6, 8(a1)
    lw      a7, 12(a1)
    lw      t1, 16(a1)
    lw      t2, 20(a1)
    lw      t3, 24(a1)
    lw      t4, 28(a1)
    sw      a4, 0(a0)
    sw      a5, 4(a0)
    sw      a6, 8(a0)
    sw      a7, 12(a0)
    sw      t1, 16(a0)
    sw      t2, 20(a0)
    sw      t3, 24(a0)
    sw      t4, 28(a0)
    addi    a2, a2, -32
    li      t0, 32
    bgeu    a2, t0, .Lmm_bwd_loop
.Lmm_bwd_tail:
    li      t0, 4
    bltu    a2, t0, .Lmm_bwd_bytes
.Lmm_bwd_word:
    addi    a1, a1, -4
    addi    a0, a0, -4
    lw      t1, 0(a1)
    sw      t1, 0(a0)
    addi    a2, a2, -4
    bgeu    a2, t0, .Lmm_bwd_word
.Lmm_bwd_bytes:
    beqz    a2, .Lmm_done
.Lmm_bwd_byte:
    addi    a1, a1, -1
    addi    a0, a0, -1
    lbu     t1, 0(a1)
    sb      t1, 0(a0)
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
    xor     t0, a0, a1
    andi    t0, t0, 3
    bnez    t0, .Lmm_un

.Lmm_al_head:
    andi    t0, a0, 3
    beqz    t0, .Lmm_al_words
    li      t1, 4
    sub     t1, t1, t0
.Lmm_al_head_loop:
    beqz    t1, .Lmm_al_words
    beqz    a2, .Lmm_done
    lbu     t2, 0(a1)
    sb      t2, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    addi    t1, t1, -1
    j       .Lmm_al_head_loop

.Lmm_al_words:
    li      t0, 32
    bltu    a2, t0, .Lmm_al_tail
.Lmm_al_loop:
    lw      a4, 0(a1)
    lw      a5, 4(a1)
    lw      a6, 8(a1)
    lw      a7, 12(a1)
    lw      t1, 16(a1)
    lw      t2, 20(a1)
    lw      t3, 24(a1)
    lw      t4, 28(a1)
    sw      a4, 0(a0)
    sw      a5, 4(a0)
    sw      a6, 8(a0)
    sw      a7, 12(a0)
    sw      t1, 16(a0)
    sw      t2, 20(a0)
    sw      t3, 24(a0)
    sw      t4, 28(a0)
    addi    a1, a1, 32
    addi    a0, a0, 32
    addi    a2, a2, -32
    bgeu    a2, t0, .Lmm_al_loop

.Lmm_al_tail:
    li      t0, 4
    bltu    a2, t0, .Lmm_bytes
.Lmm_al_word:
    lw      t1, 0(a1)
    sw      t1, 0(a0)
    addi    a1, a1, 4
    addi    a0, a0, 4
    addi    a2, a2, -4
    bgeu    a2, t0, .Lmm_al_word
    j       .Lmm_bytes

.Lmm_un:
    andi    t0, a0, 3
    beqz    t0, .Lmm_un_words
    li      t1, 4
    sub     t1, t1, t0
.Lmm_un_head_loop:
    beqz    t1, .Lmm_un_words
    beqz    a2, .Lmm_done
    lbu     t2, 0(a1)
    sb      t2, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    addi    t1, t1, -1
    j       .Lmm_un_head_loop

.Lmm_un_words:
    sub     a4, a1, a0
    andi    t2, a1, 3
    slli    t2, t2, 3
    li      t3, 32
    sub     t3, t3, t2
    andi    t5, a1, -4
    li      t0, 8
    bltu    a2, t0, .Lmm_un_tail
.Lmm_un_loop:
    lw      t4, 0(t5)
    lw      t6, 4(t5)
    srl     t4, t4, t2
    sll     t6, t6, t3
    or      t4, t4, t6
    sw      t4, 0(a0)
    addi    t5, t5, 4
    addi    a0, a0, 4
    addi    a2, a2, -4
    li      t0, 8
    bgeu    a2, t0, .Lmm_un_loop
.Lmm_un_tail:
    add     a1, a0, a4
#endif

.Lmm_bytes:
    beqz    a2, .Lmm_done
.Lmm_byte:
    lbu     t1, 0(a1)
    sb      t1, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lmm_byte

.Lmm_done:
    mv      a0, a3
    ret

    .size codevar_wredit_memmove, .-codevar_wredit_memmove

.section .note.GNU-stack,"",@progbits
