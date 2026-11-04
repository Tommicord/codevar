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

// RISC-V RV32 memcpy (non-overlapping), 4-byte words.
//
// Same shape as the RV64 routine (see that file): a mutually aligned
// eight-word (32 byte) unrolled `lw`/`sw` loop, or - when the alignment
// differs - a dst-aligned body that combines two aligned source words
// with a logical shift, plus a byte tail.
//
// ABI (ILP32): a0 = dst, a1 = src, a2 = len; returns a0.
// Clobbers only argument/temporary registers (no callee-saved state).

    .text
    .align 2
    .globl codevar_wredit_memcpy
    .type codevar_wredit_memcpy, @function

codevar_wredit_memcpy:
    mv      a3, a0
    beqz    a2, .Lcp_done
    beq     a0, a1, .Lcp_done
#ifdef CODEVAR_HAVE_RVV
// ---------------------------------------------------------------------
// RVV byte-vector copy.
// ---------------------------------------------------------------------
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
    xor     t0, a0, a1
    andi    t0, t0, 3
    bnez    t0, .Lcp_unaligned

// ---------------------------------------------------------------------
// Mutually aligned path.
// ---------------------------------------------------------------------
.Lcp_al_head:
    andi    t0, a0, 3
    beqz    t0, .Lcp_al_words
    li      t1, 4
    sub     t1, t1, t0
.Lcp_al_head_loop:
    beqz    t1, .Lcp_al_words
    beqz    a2, .Lcp_done
    lbu     t2, 0(a1)
    sb      t2, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    addi    t1, t1, -1
    j       .Lcp_al_head_loop

.Lcp_al_words:
    li      t0, 32
    bltu    a2, t0, .Lcp_al_tail
.Lcp_al_loop:
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
    bgeu    a2, t0, .Lcp_al_loop

.Lcp_al_tail:
    li      t0, 4
    bltu    a2, t0, .Lcp_bytes
.Lcp_al_word:
    lw      t1, 0(a1)
    sw      t1, 0(a0)
    addi    a1, a1, 4
    addi    a0, a0, 4
    addi    a2, a2, -4
    bgeu    a2, t0, .Lcp_al_word
    j       .Lcp_bytes

// ---------------------------------------------------------------------
// Misaligned path: align dst, then combine two aligned source words.
// ---------------------------------------------------------------------
.Lcp_unaligned:
    andi    t0, a0, 3
    beqz    t0, .Lcp_un_words
    li      t1, 4
    sub     t1, t1, t0
.Lcp_un_head_loop:
    beqz    t1, .Lcp_un_words
    beqz    a2, .Lcp_done
    lbu     t2, 0(a1)
    sb      t2, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    addi    t1, t1, -1
    j       .Lcp_un_head_loop

.Lcp_un_words:
    sub     a4, a1, a0              // delta = src - dst
    andi    t2, a1, 3
    slli    t2, t2, 3               // sh = (src & 3) * 8
    li      t3, 32
    sub     t3, t3, t2              // 32 - sh
    andi    t5, a1, -4              // aligned source base
    li      t0, 8                   // keep the +4 load in bounds
    bltu    a2, t0, .Lcp_un_tail
.Lcp_un_loop:
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
    bgeu    a2, t0, .Lcp_un_loop
.Lcp_un_tail:
    add     a1, a0, a4
#endif

// ---------------------------------------------------------------------
// Shared byte tail.
// ---------------------------------------------------------------------
.Lcp_bytes:
    beqz    a2, .Lcp_done
.Lcp_byte:
    lbu     t1, 0(a1)
    sb      t1, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lcp_byte

.Lcp_done:
    mv      a0, a3
    ret

    .size codevar_wredit_memcpy, .-codevar_wredit_memcpy

.section .note.GNU-stack,"",@progbits
