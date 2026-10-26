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

// RISC-V RV64 memcpy (non-overlapping).
//
// Follows the structure of the avx/riscv_memops scalar routines: when
// source and destination share the same XLEN alignment the copy runs an
// eight-word (64 byte) unrolled `ld`/`sd` loop; when they do not, the
// destination is aligned and the body uses two aligned source words
// combined with a logical shift (no unaligned loads are issued).
// A byte tail handles the residual. With -DCODEVAR_HAVE_RVV the whole
// copy is instead driven by the byte-vector loop.
//
// ABI (LP64): a0 = dst, a1 = src, a2 = len; returns a0.
// Clobbers only argument/temporary registers (no callee-saved state).

    .text
    .align 3
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
    // Same low three bits on both sides -> simple aligned word copy.
    xor     t0, a0, a1
    andi    t0, t0, 7
    bnez    t0, .Lcp_unaligned

// ---------------------------------------------------------------------
// Mutually aligned path.
// ---------------------------------------------------------------------
.Lcp_al_head:
    andi    t0, a0, 7               // bytes until dst is word aligned
    beqz    t0, .Lcp_al_words
    li      t1, 8
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
    li      t0, 64
    bltu    a2, t0, .Lcp_al_tail
.Lcp_al_loop:
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
    bgeu    a2, t0, .Lcp_al_loop

.Lcp_al_tail:
    li      t0, 8
    bltu    a2, t0, .Lcp_bytes
.Lcp_al_word:
    ld      t1, 0(a1)
    sd      t1, 0(a0)
    addi    a1, a1, 8
    addi    a0, a0, 8
    addi    a2, a2, -8
    bgeu    a2, t0, .Lcp_al_word
    j       .Lcp_bytes

// ---------------------------------------------------------------------
// Misaligned path: align dst, then combine two aligned source words.
// `a4` holds the (constant) src - dst delta so the logical source
// pointer can be reconstructed for the byte tail.
// ---------------------------------------------------------------------
.Lcp_unaligned:
    andi    t0, a0, 7
    beqz    t0, .Lcp_un_words
    li      t1, 8
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
    andi    t2, a1, 7
    slli    t2, t2, 3               // sh = (src & 7) * 8
    li      t3, 64
    sub     t3, t3, t2              // 64 - sh
    andi    t5, a1, -8              // aligned source base
    li      t0, 16                  // keep the +8 load in bounds
    bltu    a2, t0, .Lcp_un_tail
.Lcp_un_loop:
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
