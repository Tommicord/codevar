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

// RISC-V RV64 memmove.
//
// The forward direction (dst < src) shares the aligned/unaligned body of
// the RV64 memcpy (eight-word unrolled `ld`/`sd` loop, or a shifted
// combine of aligned words for the misaligned case), or the byte-vector
// loop when -DCODEVAR_HAVE_RVV. The backward direction (dst > src) walks
// from the high end with an eight-word unrolled loop, loading a full
// block into registers before storing it so the written bytes never
// overtake unread source.
//
// ABI (LP64): a0 = dst, a1 = src, a2 = len; returns a0.

    .text
    .align 3
    .globl codevar_wredit_memmove
    .type codevar_wredit_memmove, @function

codevar_wredit_memmove:
    mv      a3, a0                  // return value
    beqz    a2, .Lmm_done
    beq     a0, a1, .Lmm_done
    bltu    a0, a1, .Lmm_forward

// ---------------------------------------------------------------------
// Backward (a0 > a1): walk from the high end down.
// ---------------------------------------------------------------------
    add     a1, a1, a2
    add     a0, a0, a2
    li      t0, 64
    bltu    a2, t0, .Lmm_bwd_tail
.Lmm_bwd_loop:
    addi    a1, a1, -64
    addi    a0, a0, -64
    ld      a4, 0(a1)
    ld      a5, 8(a1)
    ld      a6, 16(a1)
    ld      a7, 24(a1)
    ld      t1, 32(a1)
    ld      t2, 40(a1)
    ld      t3, 48(a1)
    ld      t4, 56(a1)
    sd      a4, 0(a0)
    sd      a5, 8(a0)
    sd      a6, 16(a0)
    sd      a7, 24(a0)
    sd      t1, 32(a0)
    sd      t2, 40(a0)
    sd      t3, 48(a0)
    sd      t4, 56(a0)
    addi    a2, a2, -64
    li      t0, 64
    bgeu    a2, t0, .Lmm_bwd_loop
.Lmm_bwd_tail:
    li      t0, 8
    bltu    a2, t0, .Lmm_bwd_bytes
.Lmm_bwd_word:
    addi    a1, a1, -8
    addi    a0, a0, -8
    ld      t1, 0(a1)
    sd      t1, 0(a0)
    addi    a2, a2, -8
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

// ---------------------------------------------------------------------
// Forward (a0 < a1).
// ---------------------------------------------------------------------
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
    andi    t0, t0, 7
    bnez    t0, .Lmm_un

.Lmm_al_head:
    andi    t0, a0, 7
    beqz    t0, .Lmm_al_words
    li      t1, 8
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
    li      t0, 64
    bltu    a2, t0, .Lmm_al_tail
.Lmm_al_loop:
    ld      a4, 0(a1)
    ld      a5, 8(a1)
    ld      a6, 16(a1)
    ld      a7, 24(a1)
    ld      t1, 32(a1)
    ld      t2, 40(a1)
    ld      t3, 48(a1)
    ld      t4, 56(a1)
    sd      a4, 0(a0)
    sd      a5, 8(a0)
    sd      a6, 16(a0)
    sd      a7, 24(a0)
    sd      t1, 32(a0)
    sd      t2, 40(a0)
    sd      t3, 48(a0)
    sd      t4, 56(a0)
    addi    a1, a1, 64
    addi    a0, a0, 64
    addi    a2, a2, -64
    bgeu    a2, t0, .Lmm_al_loop

.Lmm_al_tail:
    li      t0, 8
    bltu    a2, t0, .Lmm_bytes
.Lmm_al_word:
    ld      t1, 0(a1)
    sd      t1, 0(a0)
    addi    a1, a1, 8
    addi    a0, a0, 8
    addi    a2, a2, -8
    bgeu    a2, t0, .Lmm_al_word
    j       .Lmm_bytes

.Lmm_un:
    andi    t0, a0, 7
    beqz    t0, .Lmm_un_words
    li      t1, 8
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
    andi    t2, a1, 7
    slli    t2, t2, 3
    li      t3, 64
    sub     t3, t3, t2
    andi    t5, a1, -8
    li      t0, 16
    bltu    a2, t0, .Lmm_un_tail
.Lmm_un_loop:
    ld      t4, 0(t5)
    ld      t6, 8(t5)
    srl     t4, t4, t2
    sll     t6, t6, t3
    or      t4, t4, t6
    sd      t4, 0(a0)
    addi    t5, t5, 8
    addi    a0, a0, 8
    addi    a2, a2, -8
    li      t0, 16
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
