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

// RISC-V RV32 (ILP32) kernel primitives used by the wredit primitives
// layer. Every symbol is `codevar_wredit_k_<name>`, follows the ILP32
// C ABI (a0-a7 arguments, a0 return, a0/a1 for a 64-bit return) and
// touches only caller-saved registers, so no callee-saved state is
// disturbed and the stack is never used.
//
// The word loops use 4-byte `lw`/`sw`; every routine keeps a correct
// scalar path, and `len`/`count == 0` returns without touching memory.

    .text
    .align 2

// ---------------------------------------------------------------------
// fill_bytes(dst = a0, value = a1, len = a2) -> a0
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_fill_bytes
    .type codevar_wredit_k_fill_bytes, @function
codevar_wredit_k_fill_bytes:
    mv      t0, a0
    andi    a1, a1, 0xff
    beqz    a2, .Lfb_done
    // Broadcast the byte to a full word without the M extension.
    slli    t1, a1, 8
    or      a1, a1, t1
    slli    t1, a1, 16
    or      a1, a1, t1
.Lfb_head:
    andi    t1, a0, 3
    beqz    t1, .Lfb_words
    sb      a1, 0(a0)
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lfb_head
    j       .Lfb_done
.Lfb_words:
    li      t1, 4
    bltu    a2, t1, .Lfb_bytes
.Lfb_wloop:
    sw      a1, 0(a0)
    addi    a0, a0, 4
    addi    a2, a2, -4
    bgeu    a2, t1, .Lfb_wloop
.Lfb_bytes:
    beqz    a2, .Lfb_done
.Lfb_bloop:
    sb      a1, 0(a0)
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lfb_bloop
.Lfb_done:
    mv      a0, t0
    ret
    .size codevar_wredit_k_fill_bytes, .-codevar_wredit_k_fill_bytes

// ---------------------------------------------------------------------
// fill_words(dst = a0, value = a1, count = a2) -> a0
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_fill_words
    .type codevar_wredit_k_fill_words, @function
codevar_wredit_k_fill_words:
    mv      t0, a0
    beqz    a2, .Lfw_done
.Lfw_loop:
    sw      a1, 0(a0)
    addi    a0, a0, 4
    addi    a2, a2, -1
    bnez    a2, .Lfw_loop
.Lfw_done:
    mv      a0, t0
    ret
    .size codevar_wredit_k_fill_words, .-codevar_wredit_k_fill_words

// ---------------------------------------------------------------------
// copy_words(dst = a0, src = a1, count = a2) -> a0
// Non-overlapping; 8 words per iteration.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_copy_words
    .type codevar_wredit_k_copy_words, @function
codevar_wredit_k_copy_words:
    mv      t0, a0
    beqz    a2, .Lcw_done
    li      t5, 8
    bltu    a2, t5, .Lcw_words
.Lcw_loop:
    lw      a3, 0(a1)
    lw      a4, 4(a1)
    lw      a5, 8(a1)
    lw      a6, 12(a1)
    lw      t1, 16(a1)
    lw      t2, 20(a1)
    lw      t3, 24(a1)
    lw      t4, 28(a1)
    sw      a3, 0(a0)
    sw      a4, 4(a0)
    sw      a5, 8(a0)
    sw      a6, 12(a0)
    sw      t1, 16(a0)
    sw      t2, 20(a0)
    sw      t3, 24(a0)
    sw      t4, 28(a0)
    addi    a1, a1, 32
    addi    a0, a0, 32
    addi    a2, a2, -8
    bgeu    a2, t5, .Lcw_loop
.Lcw_words:
    beqz    a2, .Lcw_done
.Lcw_wloop:
    lw      t1, 0(a1)
    sw      t1, 0(a0)
    addi    a1, a1, 4
    addi    a0, a0, 4
    addi    a2, a2, -1
    bnez    a2, .Lcw_wloop
.Lcw_done:
    mv      a0, t0
    ret
    .size codevar_wredit_k_copy_words, .-codevar_wredit_k_copy_words

// ---------------------------------------------------------------------
// move_words(dst = a0, src = a1, count = a2) -> a0
// Overlap-safe: forward when dst < src, backward otherwise.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_move_words
    .type codevar_wredit_k_move_words, @function
codevar_wredit_k_move_words:
    mv      t0, a0
    beqz    a2, .Lmw_done
    beq     a0, a1, .Lmw_done
    bltu    a0, a1, .Lmw_forward

    // Backward copy.
    slli    t5, a2, 2
    add     a0, a0, t5
    add     a1, a1, t5
    li      t5, 8
    bltu    a2, t5, .Lmw_bwd_words
.Lmw_bwd_loop:
    addi    a1, a1, -32
    addi    a0, a0, -32
    lw      a3, 0(a1)
    lw      a4, 4(a1)
    lw      a5, 8(a1)
    lw      a6, 12(a1)
    lw      t1, 16(a1)
    lw      t2, 20(a1)
    lw      t3, 24(a1)
    lw      t4, 28(a1)
    sw      a3, 0(a0)
    sw      a4, 4(a0)
    sw      a5, 8(a0)
    sw      a6, 12(a0)
    sw      t1, 16(a0)
    sw      t2, 20(a0)
    sw      t3, 24(a0)
    sw      t4, 28(a0)
    addi    a2, a2, -8
    bgeu    a2, t5, .Lmw_bwd_loop
.Lmw_bwd_words:
    beqz    a2, .Lmw_done
.Lmw_bwd_wloop:
    addi    a1, a1, -4
    addi    a0, a0, -4
    lw      t1, 0(a1)
    sw      t1, 0(a0)
    addi    a2, a2, -1
    bnez    a2, .Lmw_bwd_wloop
    j       .Lmw_done

    // Forward copy.
.Lmw_forward:
    li      t5, 8
    bltu    a2, t5, .Lmw_fwd_words
.Lmw_fwd_loop:
    lw      a3, 0(a1)
    lw      a4, 4(a1)
    lw      a5, 8(a1)
    lw      a6, 12(a1)
    lw      t1, 16(a1)
    lw      t2, 20(a1)
    lw      t3, 24(a1)
    lw      t4, 28(a1)
    sw      a3, 0(a0)
    sw      a4, 4(a0)
    sw      a5, 8(a0)
    sw      a6, 12(a0)
    sw      t1, 16(a0)
    sw      t2, 20(a0)
    sw      t3, 24(a0)
    sw      t4, 28(a0)
    addi    a1, a1, 32
    addi    a0, a0, 32
    addi    a2, a2, -8
    bgeu    a2, t5, .Lmw_fwd_loop
.Lmw_fwd_words:
    beqz    a2, .Lmw_done
.Lmw_fwd_wloop:
    lw      t1, 0(a1)
    sw      t1, 0(a0)
    addi    a1, a1, 4
    addi    a0, a0, 4
    addi    a2, a2, -1
    bnez    a2, .Lmw_fwd_wloop
.Lmw_done:
    mv      a0, t0
    ret
    .size codevar_wredit_k_move_words, .-codevar_wredit_k_move_words

// ---------------------------------------------------------------------
// count_byte(hay = a0, needle = a1, len = a2) -> a0 (usize)
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_count_byte
    .type codevar_wredit_k_count_byte, @function
codevar_wredit_k_count_byte:
    andi    a1, a1, 0xff
    li      t0, 0
    beqz    a2, .Lcb_done
.Lcb_loop:
    lbu     t1, 0(a0)
    bne     t1, a1, .Lcb_skip
    addi    t0, t0, 1
.Lcb_skip:
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lcb_loop
.Lcb_done:
    mv      a0, t0
    ret
    .size codevar_wredit_k_count_byte, .-codevar_wredit_k_count_byte

// ---------------------------------------------------------------------
// find_byte_rev(hay = a0, needle = a1, len = a2) -> *const u8
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_find_byte_rev
    .type codevar_wredit_k_find_byte_rev, @function
codevar_wredit_k_find_byte_rev:
    andi    a1, a1, 0xff
    beqz    a2, .Lfr_none
    add     a0, a0, a2
.Lfr_loop:
    addi    a0, a0, -1
    lbu     t0, 0(a0)
    beq     t0, a1, .Lfr_hit
    addi    a2, a2, -1
    bnez    a2, .Lfr_loop
.Lfr_none:
    li      a0, 0
    ret
.Lfr_hit:
    ret
    .size codevar_wredit_k_find_byte_rev, .-codevar_wredit_k_find_byte_rev

// ---------------------------------------------------------------------
// find_any2(hay = a0, a = a1, b = a2, len = a3) -> *const u8
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_find_any2
    .type codevar_wredit_k_find_any2, @function
codevar_wredit_k_find_any2:
    andi    a1, a1, 0xff
    andi    a2, a2, 0xff
    beqz    a3, .Lfa_none
.Lfa_loop:
    lbu     t0, 0(a0)
    beq     t0, a1, .Lfa_hit
    beq     t0, a2, .Lfa_hit
    addi    a0, a0, 1
    addi    a3, a3, -1
    bnez    a3, .Lfa_loop
.Lfa_none:
    li      a0, 0
    ret
.Lfa_hit:
    ret
    .size codevar_wredit_k_find_any2, .-codevar_wredit_k_find_any2

// ---------------------------------------------------------------------
// find_not_byte(hay = a0, needle = a1, len = a2) -> *const u8
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_find_not_byte
    .type codevar_wredit_k_find_not_byte, @function
codevar_wredit_k_find_not_byte:
    andi    a1, a1, 0xff
    beqz    a2, .Lfn_none
.Lfn_loop:
    lbu     t0, 0(a0)
    bne     t0, a1, .Lfn_hit
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lfn_loop
.Lfn_none:
    li      a0, 0
    ret
.Lfn_hit:
    ret
    .size codevar_wredit_k_find_not_byte, .-codevar_wredit_k_find_not_byte

// ---------------------------------------------------------------------
// count_eq_u32(src = a0, value = a1, mask = a2, count = a3) -> a0
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_count_eq_u32
    .type codevar_wredit_k_count_eq_u32, @function
codevar_wredit_k_count_eq_u32:
    li      t0, 0
    beqz    a3, .Lce_done
.Lce_loop:
    lw      t1, 0(a0)
    and     t1, t1, a2
    bne     t1, a1, .Lce_skip
    addi    t0, t0, 1
.Lce_skip:
    addi    a0, a0, 4
    addi    a3, a3, -1
    bnez    a3, .Lce_loop
.Lce_done:
    mv      a0, t0
    ret
    .size codevar_wredit_k_count_eq_u32, .-codevar_wredit_k_count_eq_u32

// ---------------------------------------------------------------------
// find_eq_u32(src = a0, value = a1, mask = a2, count = a3) -> *const u32
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_find_eq_u32
    .type codevar_wredit_k_find_eq_u32, @function
codevar_wredit_k_find_eq_u32:
    beqz    a3, .Lfe_none
.Lfe_loop:
    lw      t0, 0(a0)
    and     t0, t0, a2
    beq     t0, a1, .Lfe_hit
    addi    a0, a0, 4
    addi    a3, a3, -1
    bnez    a3, .Lfe_loop
.Lfe_none:
    li      a0, 0
    ret
.Lfe_hit:
    ret
    .size codevar_wredit_k_find_eq_u32, .-codevar_wredit_k_find_eq_u32

// ---------------------------------------------------------------------
// find_eq_u32_rev(src = a0, value = a1, mask = a2, count = a3) -> *const u32
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_find_eq_u32_rev
    .type codevar_wredit_k_find_eq_u32_rev, @function
codevar_wredit_k_find_eq_u32_rev:
    beqz    a3, .Lfer_none
    slli    t0, a3, 2
    add     a0, a0, t0
.Lfer_loop:
    addi    a0, a0, -4
    lw      t0, 0(a0)
    and     t0, t0, a2
    beq     t0, a1, .Lfer_hit
    addi    a3, a3, -1
    bnez    a3, .Lfer_loop
.Lfer_none:
    li      a0, 0
    ret
.Lfer_hit:
    ret
    .size codevar_wredit_k_find_eq_u32_rev, .-codevar_wredit_k_find_eq_u32_rev

// ---------------------------------------------------------------------
// memxor(dst = a0, src = a1, len = a2) -> a0
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_memxor
    .type codevar_wredit_k_memxor, @function
codevar_wredit_k_memxor:
    mv      t0, a0
    beqz    a2, .Lmx_done
    xor     t1, a0, a1
    andi    t1, t1, 3
    bnez    t1, .Lmx_bytes

    // Both pointers share alignment: word loop with a byte head/tail.
.Lmx_head:
    andi    t1, a0, 3
    beqz    t1, .Lmx_words
    lbu     t2, 0(a1)
    lbu     t3, 0(a0)
    xor     t2, t2, t3
    sb      t2, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lmx_head
    j       .Lmx_done
.Lmx_words:
    li      t1, 4
    bltu    a2, t1, .Lmx_bytes
.Lmx_wloop:
    lw      t2, 0(a1)
    lw      t3, 0(a0)
    xor     t2, t2, t3
    sw      t2, 0(a0)
    addi    a1, a1, 4
    addi    a0, a0, 4
    addi    a2, a2, -4
    bgeu    a2, t1, .Lmx_wloop

.Lmx_bytes:
    beqz    a2, .Lmx_done
.Lmx_bloop:
    lbu     t2, 0(a1)
    lbu     t3, 0(a0)
    xor     t2, t2, t3
    sb      t2, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lmx_bloop
.Lmx_done:
    mv      a0, t0
    ret
    .size codevar_wredit_k_memxor, .-codevar_wredit_k_memxor

// ---------------------------------------------------------------------
// memcmp(a = a0, b = a1, len = a2) -> i32
// 0 equal, -1 when a < b at the first difference, 1 otherwise.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_memcmp
    .type codevar_wredit_k_memcmp, @function
codevar_wredit_k_memcmp:
    beqz    a2, .Lmcq_equal
.Lmcq_loop:
    lbu     t0, 0(a0)
    lbu     t1, 0(a1)
    bne     t0, t1, .Lmcq_diff
    addi    a0, a0, 1
    addi    a1, a1, 1
    addi    a2, a2, -1
    bnez    a2, .Lmcq_loop
.Lmcq_equal:
    li      a0, 0
    ret
.Lmcq_diff:
    bltu    t0, t1, .Lmcq_less
    li      a0, 1
    ret
.Lmcq_less:
    li      a0, -1
    ret
    .size codevar_wredit_k_memcmp, .-codevar_wredit_k_memcmp

// ---------------------------------------------------------------------
// sum_bytes(src = a0, len = a1) -> u64 (a0 low, a1 high)
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_sum_bytes
    .type codevar_wredit_k_sum_bytes, @function
codevar_wredit_k_sum_bytes:
    mv      t4, a0
    mv      t5, a1
    li      t0, 0
    li      t1, 0
    beqz    t5, .Lsb_done
.Lsb_loop:
    lbu     t2, 0(t4)
    add     t0, t0, t2
    sltu    t3, t0, t2
    add     t1, t1, t3
    addi    t4, t4, 1
    addi    t5, t5, -1
    bnez    t5, .Lsb_loop
.Lsb_done:
    mv      a0, t0
    mv      a1, t1
    ret
    .size codevar_wredit_k_sum_bytes, .-codevar_wredit_k_sum_bytes

// ---------------------------------------------------------------------
// weighted_checksum(src = a0, len = a1, seed = a2) -> u64 (a0 low, a1 high)
// weight = seed % 251 + 1; sum += byte * weight; weight = weight % 251 + 1.
// Base integer only: modulo/division use shift-subtract, so no M extension.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_weighted_checksum
    .type codevar_wredit_k_weighted_checksum, @function
codevar_wredit_k_weighted_checksum:
    mv      t4, a0
    mv      t5, a1
    // t0 = seed % 251 via 32-step restoring division.
    li      t0, 0
    li      t1, 31
.Lwc_mod:
    slli    t0, t0, 1
    srl     t2, a2, t1
    andi    t2, t2, 1
    or      t0, t0, t2
    li      t3, 251
    bltu    t0, t3, .Lwc_mod_skip
    sub     t0, t0, t3
.Lwc_mod_skip:
    addi    t1, t1, -1
    bgez    t1, .Lwc_mod
    addi    a2, t0, 1              // weight

    li      a3, 0                  // sum low
    li      a4, 0                  // sum high
    beqz    t5, .Lwc_done
.Lwc_loop:
    lbu     t2, 0(t4)
    // t3 = t2 * a2 using shift-add (byte <= 255, weight <= 251).
    li      t3, 0
    mv      t6, t2
    mv      a6, a2
.Lwc_mul:
    andi    a5, a6, 1
    beqz    a5, .Lwc_mul_skip
    add     t3, t3, t6
.Lwc_mul_skip:
    slli    t6, t6, 1
    srli    a6, a6, 1
    bnez    a6, .Lwc_mul
    // 64-bit accumulate.
    add     a3, a3, t3
    sltu    a5, a3, t3
    add     a4, a4, a5
    // weight += 1; wrap to 1 above 251.
    addi    a2, a2, 1
    li      t3, 251
    bleu    a2, t3, .Lwc_w_ok
    li      a2, 1
.Lwc_w_ok:
    addi    t4, t4, 1
    addi    t5, t5, -1
    bnez    t5, .Lwc_loop
.Lwc_done:
    mv      a0, a3
    mv      a1, a4
    ret
    .size codevar_wredit_k_weighted_checksum, .-codevar_wredit_k_weighted_checksum

// ---------------------------------------------------------------------
// utf8_count_starts(src = a0, len = a1) -> a0 (usize)
// Count bytes whose (byte & 0xC0) != 0x80.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_utf8_count_starts
    .type codevar_wredit_k_utf8_count_starts, @function
codevar_wredit_k_utf8_count_starts:
    li      t2, 0
    beqz    a1, .Luc_done
.Luc_loop:
    lbu     t0, 0(a0)
    andi    t0, t0, 0xC0
    li      t1, 0x80
    beq     t0, t1, .Luc_skip
    addi    t2, t2, 1
.Luc_skip:
    addi    a0, a0, 1
    addi    a1, a1, -1
    bnez    a1, .Luc_loop
.Luc_done:
    mv      a0, t2
    ret
    .size codevar_wredit_k_utf8_count_starts, .-codevar_wredit_k_utf8_count_starts

// ---------------------------------------------------------------------
// replace_byte(dst = a0, from = a1, to = a2, len = a3) -> a0 (usize)
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_replace_byte
    .type codevar_wredit_k_replace_byte, @function
codevar_wredit_k_replace_byte:
    andi    a1, a1, 0xff
    andi    a2, a2, 0xff
    li      t0, 0
    beqz    a3, .Lrb_done
.Lrb_loop:
    lbu     t1, 0(a0)
    bne     t1, a1, .Lrb_skip
    sb      a2, 0(a0)
    addi    t0, t0, 1
.Lrb_skip:
    addi    a0, a0, 1
    addi    a3, a3, -1
    bnez    a3, .Lrb_loop
.Lrb_done:
    mv      a0, t0
    ret
    .size codevar_wredit_k_replace_byte, .-codevar_wredit_k_replace_byte

// ---------------------------------------------------------------------
// reverse_bytes(dst = a0, src = a1, len = a2) -> a0
// Non-overlapping; copies src reversed into dst.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_reverse_bytes
    .type codevar_wredit_k_reverse_bytes, @function
codevar_wredit_k_reverse_bytes:
    mv      t0, a0
    beqz    a2, .Lrv_done
    add     t1, a1, a2
    addi    t1, t1, -1
.Lrv_loop:
    lbu     t2, 0(t1)
    sb      t2, 0(a0)
    addi    t1, t1, -1
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lrv_loop
.Lrv_done:
    mv      a0, t0
    ret
    .size codevar_wredit_k_reverse_bytes, .-codevar_wredit_k_reverse_bytes

// ---------------------------------------------------------------------
// fill_pattern8(dst = a0, pattern = a1 (low) : a2 (high), len = a3) -> a0
// Repeat the 8 pattern bytes in native memory order. The current pattern
// window is kept in the a1/a2 pair and rotated left by 8 bits per byte.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_fill_pattern8
    .type codevar_wredit_k_fill_pattern8, @function
codevar_wredit_k_fill_pattern8:
    mv      t0, a0
    beqz    a3, .Lfp_done
    andi    t1, a0, 3
    bnez    t1, .Lfp_bytes
    // 4-aligned with the pattern anchored at dst: 8-byte blocks.
    li      t1, 8
    bltu    a3, t1, .Lfp_tail4
.Lfp_blocks:
    sw      a1, 0(a0)
    sw      a2, 4(a0)
    addi    a0, a0, 8
    addi    a3, a3, -8
    bgeu    a3, t1, .Lfp_blocks
.Lfp_tail4:
    li      t1, 4
    bltu    a3, t1, .Lfp_tailb
    sw      a1, 0(a0)
    addi    a0, a0, 4
    addi    a3, a3, -4
    // Rotate left by 32 bits: swap the halves.
    mv      t1, a1
    mv      a1, a2
    mv      a2, t1
.Lfp_tailb:
    beqz    a3, .Lfp_done
.Lfp_tloop:
    sb      a1, 0(a0)
    // Rotate left by 8 bits across the 64-bit pair.
    srli    t3, a1, 8
    slli    t4, a2, 24
    or      t3, t3, t4
    srli    t5, a2, 8
    slli    t6, a1, 24
    or      t5, t5, t6
    mv      a1, t3
    mv      a2, t5
    addi    a0, a0, 1
    addi    a3, a3, -1
    bnez    a3, .Lfp_tloop
    j       .Lfp_done

.Lfp_bytes:
.Lfp_bloop:
    sb      a1, 0(a0)
    srli    t3, a1, 8
    slli    t4, a2, 24
    or      t3, t3, t4
    srli    t5, a2, 8
    slli    t6, a1, 24
    or      t5, t5, t6
    mv      a1, t3
    mv      a2, t5
    addi    a0, a0, 1
    addi    a3, a3, -1
    bnez    a3, .Lfp_bloop
.Lfp_done:
    mv      a0, t0
    ret
    .size codevar_wredit_k_fill_pattern8, .-codevar_wredit_k_fill_pattern8

// ---------------------------------------------------------------------
// byte_histogram(counts = a0, src = a1, len = a2)
// Zero 256 u32 counters, then increment counts[byte].
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_byte_histogram
    .type codevar_wredit_k_byte_histogram, @function
codevar_wredit_k_byte_histogram:
    mv      t0, a0
    li      t1, 256
.Lbh_zero:
    sw      zero, 0(a0)
    addi    a0, a0, 4
    addi    t1, t1, -1
    bnez    t1, .Lbh_zero
    beqz    a2, .Lbh_done
.Lbh_loop:
    lbu     t2, 0(a1)
    slli    t2, t2, 2
    add     t2, t0, t2
    lw      t3, 0(t2)
    addi    t3, t3, 1
    sw      t3, 0(t2)
    addi    a1, a1, 1
    addi    a2, a2, -1
    bnez    a2, .Lbh_loop
.Lbh_done:
    ret
    .size codevar_wredit_k_byte_histogram, .-codevar_wredit_k_byte_histogram

    .section .note.GNU-stack,"",@progbits
