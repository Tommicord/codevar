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

// RISC-V RV64GC kernel primitives used by the wredit primitives layer.
//
// Every symbol is `codevar_wredit_k_<name>` and follows the LP64 C ABI:
// a0-a7 arguments, a0 return, t0-t6/a0-a7 caller-saved. No callee-saved
// register is touched and the stack is never used, so there is no frame
// to unwind and no state to preserve.
//
// Word loops use 8-byte `ld`/`sd` where the access is provably aligned:
// destination stores are aligned with a short byte head, and source loads
// either share the destination's alignment (copies/xor) or are aligned by
// a byte head (scans/sums). Misaligned inputs therefore stay on the byte
// path and never issue a misaligned access.
//
// `len`/`count == 0` returns before touching memory. The optional RVV
// paths are guarded by CODEVAR_HAVE_RVV; the scalar path is always
// assembled and remains correct on its own.

    .text
    .align 3

// ---------------------------------------------------------------------
// fill_bytes(dst = a0, value = a1, len = a2) -> a0
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_fill_bytes
    .type codevar_wredit_k_fill_bytes, @function
codevar_wredit_k_fill_bytes:
    mv      a3, a0
    andi    a1, a1, 0xff
    beqz    a2, .Lfb_done
    // Broadcast the byte to a full 64-bit pattern.
    slli    t0, a1, 8
    or      a1, a1, t0
    slli    t0, a1, 16
    or      a1, a1, t0
    slli    t0, a1, 32
    or      a1, a1, t0
#ifdef CODEVAR_HAVE_RVV
.Lfb_rvv:
    vsetvli t0, a2, e8, m8, ta, ma
    vmv.v.x v0, a1
    vse8.v  v0, (a0)
    add     a0, a0, t0
    sub     a2, a2, t0
    bnez    a2, .Lfb_rvv
    j       .Lfb_done
#else
    andi    t0, a0, 7
    beqz    t0, .Lfb_words
    li      t1, 8
    sub     t1, t1, t0
.Lfb_head:
    sb      a1, 0(a0)
    addi    a0, a0, 1
    addi    a2, a2, -1
    addi    t1, t1, -1
    beqz    a2, .Lfb_done
    bnez    t1, .Lfb_head
.Lfb_words:
    li      t0, 8
    bltu    a2, t0, .Lfb_bytes
.Lfb_wloop:
    sd      a1, 0(a0)
    addi    a0, a0, 8
    addi    a2, a2, -8
    bgeu    a2, t0, .Lfb_wloop
.Lfb_bytes:
    beqz    a2, .Lfb_done
.Lfb_bloop:
    sb      a1, 0(a0)
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lfb_bloop
#endif
.Lfb_done:
    mv      a0, a3
    ret
    .size codevar_wredit_k_fill_bytes, .-codevar_wredit_k_fill_bytes

// ---------------------------------------------------------------------
// fill_words(dst = a0, value = a1, count = a2) -> a0
// The byte stream is the four bytes of `value` repeated; it is filled as
// count * 4 bytes with an 8-byte pattern.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_fill_words
    .type codevar_wredit_k_fill_words, @function
codevar_wredit_k_fill_words:
    mv      a3, a0
    beqz    a2, .Lfw_done
    slli    a2, a2, 2
    slli    a4, a1, 32
    srli    a4, a4, 32              // zero-extend the u32 value
    slli    a5, a4, 32
    or      a1, a5, a4              // value | value << 32
    // `dst` is 4-byte aligned, so the head ends on a 4-byte phase and the
    // 4-byte-periodic pattern starts over cleanly at the word loop.
    andi    t0, a0, 7
    beqz    t0, .Lfw_words
    li      t1, 8
    sub     t1, t1, t0
    mv      t2, a1
.Lfw_head:
    sb      t2, 0(a0)
    srli    t2, t2, 8
    addi    a0, a0, 1
    addi    a2, a2, -1
    addi    t1, t1, -1
    beqz    a2, .Lfw_done
    bnez    t1, .Lfw_head
.Lfw_words:
    li      t0, 8
    bltu    a2, t0, .Lfw_bytes
.Lfw_wloop:
    sd      a1, 0(a0)
    addi    a0, a0, 8
    addi    a2, a2, -8
    bgeu    a2, t0, .Lfw_wloop
.Lfw_bytes:
    beqz    a2, .Lfw_done
    mv      t2, a1
.Lfw_bloop:
    sb      t2, 0(a0)
    srli    t2, t2, 8
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lfw_bloop
.Lfw_done:
    mv      a0, a3
    ret
    .size codevar_wredit_k_fill_words, .-codevar_wredit_k_fill_words

// ---------------------------------------------------------------------
// copy_words(dst = a0, src = a1, count = a2) -> a0
// Non-overlapping. A word path is used only when both pointers share the
// same 8-byte residue; otherwise the bytes are copied one at a time.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_copy_words
    .type codevar_wredit_k_copy_words, @function
codevar_wredit_k_copy_words:
    mv      a3, a0
    beqz    a2, .Lcw_done
    slli    a2, a2, 2
    beq     a0, a1, .Lcw_done
#ifdef CODEVAR_HAVE_RVV
.Lcw_rvv:
    vsetvli t0, a2, e8, m8, ta, ma
    vle8.v  v0, (a1)
    vse8.v  v0, (a0)
    add     a1, a1, t0
    add     a0, a0, t0
    sub     a2, a2, t0
    bnez    a2, .Lcw_rvv
    j       .Lcw_done
#else
    xor     t0, a0, a1
    andi    t0, t0, 7
    bnez    t0, .Lcw_bytes
    andi    t0, a0, 7
    beqz    t0, .Lcw_words
    li      t1, 8
    sub     t1, t1, t0
.Lcw_head:
    lbu     t2, 0(a1)
    sb      t2, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    addi    t1, t1, -1
    beqz    a2, .Lcw_done
    bnez    t1, .Lcw_head
.Lcw_words:
    li      t0, 8
    bltu    a2, t0, .Lcw_bytes
.Lcw_wloop:
    ld      t1, 0(a1)
    sd      t1, 0(a0)
    addi    a1, a1, 8
    addi    a0, a0, 8
    addi    a2, a2, -8
    bgeu    a2, t0, .Lcw_wloop
.Lcw_bytes:
    beqz    a2, .Lcw_done
.Lcw_bloop:
    lbu     t1, 0(a1)
    sb      t1, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lcw_bloop
#endif
.Lcw_done:
    mv      a0, a3
    ret
    .size codevar_wredit_k_copy_words, .-codevar_wredit_k_copy_words

// ---------------------------------------------------------------------
// move_words(dst = a0, src = a1, count = a2) -> a0
// Overlap-safe: forward when dst < src, backward otherwise. Both word
// paths require the two ranges to share an 8-byte residue.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_move_words
    .type codevar_wredit_k_move_words, @function
codevar_wredit_k_move_words:
    mv      a3, a0
    beqz    a2, .Lmw_done
    slli    a2, a2, 2
    beq     a0, a1, .Lmw_done
    bltu    a0, a1, .Lmw_forward

    // Backward copy: walk down from the high end.
    add     a1, a1, a2
    add     a0, a0, a2
    xor     t0, a0, a1
    andi    t0, t0, 7
    bnez    t0, .Lmw_bd_bytes
    andi    t0, a0, 7
    beqz    t0, .Lmw_bd_words
.Lmw_bd_head:
    addi    a1, a1, -1
    addi    a0, a0, -1
    lbu     t2, 0(a1)
    sb      t2, 0(a0)
    addi    a2, a2, -1
    beqz    a2, .Lmw_done
    andi    t0, a0, 7
    bnez    t0, .Lmw_bd_head
.Lmw_bd_words:
    li      t0, 8
    bltu    a2, t0, .Lmw_bd_bytes
.Lmw_bd_wloop:
    addi    a1, a1, -8
    addi    a0, a0, -8
    ld      t1, 0(a1)
    sd      t1, 0(a0)
    addi    a2, a2, -8
    bgeu    a2, t0, .Lmw_bd_wloop
.Lmw_bd_bytes:
    beqz    a2, .Lmw_done
.Lmw_bd_bloop:
    addi    a1, a1, -1
    addi    a0, a0, -1
    lbu     t1, 0(a1)
    sb      t1, 0(a0)
    addi    a2, a2, -1
    bnez    a2, .Lmw_bd_bloop
    j       .Lmw_done

    // Forward copy: dst < src, the ordinary copy direction.
.Lmw_forward:
#ifdef CODEVAR_HAVE_RVV
.Lmw_rvv:
    vsetvli t0, a2, e8, m8, ta, ma
    vle8.v  v0, (a1)
    vse8.v  v0, (a0)
    add     a1, a1, t0
    add     a0, a0, t0
    sub     a2, a2, t0
    bnez    a2, .Lmw_rvv
    j       .Lmw_done
#else
    xor     t0, a0, a1
    andi    t0, t0, 7
    bnez    t0, .Lmw_fw_bytes
    andi    t0, a0, 7
    beqz    t0, .Lmw_fw_words
    li      t1, 8
    sub     t1, t1, t0
.Lmw_fw_head:
    lbu     t2, 0(a1)
    sb      t2, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    addi    t1, t1, -1
    beqz    a2, .Lmw_done
    bnez    t1, .Lmw_fw_head
.Lmw_fw_words:
    li      t0, 8
    bltu    a2, t0, .Lmw_fw_bytes
.Lmw_fw_wloop:
    ld      t1, 0(a1)
    sd      t1, 0(a0)
    addi    a1, a1, 8
    addi    a0, a0, 8
    addi    a2, a2, -8
    bgeu    a2, t0, .Lmw_fw_wloop
.Lmw_fw_bytes:
    beqz    a2, .Lmw_done
.Lmw_fw_bloop:
    lbu     t1, 0(a1)
    sb      t1, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lmw_fw_bloop
#endif
.Lmw_done:
    mv      a0, a3
    ret
    .size codevar_wredit_k_move_words, .-codevar_wredit_k_move_words

// ---------------------------------------------------------------------
// count_byte(hay = a0, needle = a1, len = a2) -> a0 (usize)
// The word loop builds one 0x80 bit per zero byte of `word ^ needle` with
// a borrow-free SWAR test, then sums the eight flags with a multiply.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_count_byte
    .type codevar_wredit_k_count_byte, @function
codevar_wredit_k_count_byte:
    andi    a1, a1, 0xff
    li      a3, 0
    beqz    a2, .Lcb_done
    // Broadcast the needle to all eight bytes.
    slli    a4, a1, 8
    or      a4, a1, a4
    slli    t3, a4, 16
    or      a4, a4, t3
    slli    t3, a4, 32
    or      a4, a4, t3
    // t0 = 0x7f..7f, t1 = 0x80..80, t2 = 0x01..01.
    li      t0, 0x7f
    slli    t3, t0, 8
    or      t0, t0, t3
    slli    t3, t0, 16
    or      t0, t0, t3
    slli    t3, t0, 32
    or      t0, t0, t3
    li      t1, 0x80
    slli    t3, t1, 8
    or      t1, t1, t3
    slli    t3, t1, 16
    or      t1, t1, t3
    slli    t3, t1, 32
    or      t1, t1, t3
    li      t2, 0x01
    slli    t3, t2, 8
    or      t2, t2, t3
    slli    t3, t2, 16
    or      t2, t2, t3
    slli    t3, t2, 32
    or      t2, t2, t3
    // Byte head until the source is 8-byte aligned.
    andi    t3, a0, 7
    beqz    t3, .Lcb_words
.Lcb_head:
    lbu     t4, 0(a0)
    xor     t4, t4, a1
    sltiu   t4, t4, 1
    add     a3, a3, t4
    addi    a0, a0, 1
    addi    a2, a2, -1
    beqz    a2, .Lcb_done
    andi    t3, a0, 7
    bnez    t3, .Lcb_head
.Lcb_words:
    li      t3, 8
    bltu    a2, t3, .Lcb_bytes
.Lcb_wloop:
    ld      t4, 0(a0)
    xor     t4, t4, a4
    and     t5, t4, t0
    add     t5, t5, t0
    or      t5, t5, t4
    not     t5, t5
    and     t5, t5, t1
    srli    t5, t5, 7
    mul     t5, t5, t2
    srli    t5, t5, 56
    add     a3, a3, t5
    addi    a0, a0, 8
    addi    a2, a2, -8
    bgeu    a2, t3, .Lcb_wloop
.Lcb_bytes:
    beqz    a2, .Lcb_done
.Lcb_bloop:
    lbu     t4, 0(a0)
    xor     t4, t4, a1
    sltiu   t4, t4, 1
    add     a3, a3, t4
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lcb_bloop
.Lcb_done:
    mv      a0, a3
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
    slli    a1, a1, 32
    srli    a1, a1, 32
    slli    a2, a2, 32
    srli    a2, a2, 32
.Lce_loop:
    lwu     t1, 0(a0)
    and     t1, t1, a2
    xor     t1, t1, a1
    sltiu   t1, t1, 1
    add     t0, t0, t1
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
    slli    a1, a1, 32
    srli    a1, a1, 32
    slli    a2, a2, 32
    srli    a2, a2, 32
    beqz    a3, .Lfe_none
.Lfe_loop:
    lwu     t0, 0(a0)
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
    slli    a1, a1, 32
    srli    a1, a1, 32
    slli    a2, a2, 32
    srli    a2, a2, 32
    beqz    a3, .Lfer_none
    slli    t0, a3, 2
    add     a0, a0, t0
.Lfer_loop:
    addi    a0, a0, -4
    lwu     t0, 0(a0)
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
// Ranges may overlap only exactly (dst == src).
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_memxor
    .type codevar_wredit_k_memxor, @function
codevar_wredit_k_memxor:
    mv      a3, a0
    beqz    a2, .Lmx_done
#ifdef CODEVAR_HAVE_RVV
.Lmx_rvv:
    vsetvli t0, a2, e8, m8, ta, ma
    vle8.v  v0, (a1)
    vle8.v  v1, (a0)
    vxor.vv v0, v0, v1
    vse8.v  v0, (a0)
    add     a1, a1, t0
    add     a0, a0, t0
    sub     a2, a2, t0
    bnez    a2, .Lmx_rvv
    j       .Lmx_done
#else
    xor     t0, a0, a1
    andi    t0, t0, 7
    bnez    t0, .Lmx_bytes
    andi    t0, a0, 7
    beqz    t0, .Lmx_words
    li      t1, 8
    sub     t1, t1, t0
.Lmx_head:
    lbu     t2, 0(a1)
    lbu     t3, 0(a0)
    xor     t2, t2, t3
    sb      t2, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    addi    t1, t1, -1
    beqz    a2, .Lmx_done
    bnez    t1, .Lmx_head
.Lmx_words:
    li      t0, 8
    bltu    a2, t0, .Lmx_bytes
.Lmx_wloop:
    ld      t1, 0(a1)
    ld      t2, 0(a0)
    xor     t1, t1, t2
    sd      t1, 0(a0)
    addi    a1, a1, 8
    addi    a0, a0, 8
    addi    a2, a2, -8
    bgeu    a2, t0, .Lmx_wloop
.Lmx_bytes:
    beqz    a2, .Lmx_done
.Lmx_bloop:
    lbu     t1, 0(a1)
    lbu     t2, 0(a0)
    xor     t1, t1, t2
    sb      t1, 0(a0)
    addi    a1, a1, 1
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lmx_bloop
#endif
.Lmx_done:
    mv      a0, a3
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
// sum_bytes(src = a0, len = a1) -> a0 (u64)
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_sum_bytes
    .type codevar_wredit_k_sum_bytes, @function
codevar_wredit_k_sum_bytes:
    li      a2, 0
    beqz    a1, .Lsb_done
    andi    t1, a0, 7
    beqz    t1, .Lsb_words
.Lsb_head:
    lbu     t2, 0(a0)
    add     a2, a2, t2
    addi    a0, a0, 1
    addi    a1, a1, -1
    beqz    a1, .Lsb_done
    andi    t1, a0, 7
    bnez    t1, .Lsb_head
.Lsb_words:
    li      t1, 8
    bltu    a1, t1, .Lsb_bytes
.Lsb_wloop:
    ld      t2, 0(a0)
    andi    t3, t2, 0xff
    add     a2, a2, t3
    srli    t2, t2, 8
    andi    t3, t2, 0xff
    add     a2, a2, t3
    srli    t2, t2, 8
    andi    t3, t2, 0xff
    add     a2, a2, t3
    srli    t2, t2, 8
    andi    t3, t2, 0xff
    add     a2, a2, t3
    srli    t2, t2, 8
    andi    t3, t2, 0xff
    add     a2, a2, t3
    srli    t2, t2, 8
    andi    t3, t2, 0xff
    add     a2, a2, t3
    srli    t2, t2, 8
    andi    t3, t2, 0xff
    add     a2, a2, t3
    srli    t2, t2, 8
    andi    t3, t2, 0xff
    add     a2, a2, t3
    addi    a0, a0, 8
    addi    a1, a1, -8
    bgeu    a1, t1, .Lsb_wloop
.Lsb_bytes:
    beqz    a1, .Lsb_done
.Lsb_bloop:
    lbu     t2, 0(a0)
    add     a2, a2, t2
    addi    a0, a0, 1
    addi    a1, a1, -1
    bnez    a1, .Lsb_bloop
.Lsb_done:
    mv      a0, a2
    ret
    .size codevar_wredit_k_sum_bytes, .-codevar_wredit_k_sum_bytes

// ---------------------------------------------------------------------
// weighted_checksum(src = a0, len = a1, seed = a2) -> a0 (u64)
// weight = seed % 251 + 1; sum += byte * weight; weight += 1, wrapping
// back to 1 once it exceeds 251.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_weighted_checksum
    .type codevar_wredit_k_weighted_checksum, @function
codevar_wredit_k_weighted_checksum:
    slli    a2, a2, 32
    srli    a2, a2, 32              // zero-extend the u32 seed
    li      t1, 251
    remu    a2, a2, t1
    addi    a2, a2, 1              // weight
    li      a3, 0
    beqz    a1, .Lwc_done
.Lwc_loop:
    lbu     t0, 0(a0)
    mul     t0, t0, a2
    add     a3, a3, t0
    addi    a2, a2, 1
    li      t1, 251
    bleu    a2, t1, .Lwc_next
    li      a2, 1
.Lwc_next:
    addi    a0, a0, 1
    addi    a1, a1, -1
    bnez    a1, .Lwc_loop
.Lwc_done:
    mv      a0, a3
    ret
    .size codevar_wredit_k_weighted_checksum, .-codevar_wredit_k_weighted_checksum

// ---------------------------------------------------------------------
// utf8_count_starts(src = a0, len = a1) -> a0 (usize)
// Count bytes whose (byte & 0xC0) != 0x80.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_utf8_count_starts
    .type codevar_wredit_k_utf8_count_starts, @function
codevar_wredit_k_utf8_count_starts:
    li      a2, 0
    beqz    a1, .Luc_done
    li      t1, 0xc0
.Luc_loop:
    lbu     t0, 0(a0)
    and     t0, t0, t1
    xori    t0, t0, 0x80
    sltu    t0, zero, t0
    add     a2, a2, t0
    addi    a0, a0, 1
    addi    a1, a1, -1
    bnez    a1, .Luc_loop
.Luc_done:
    mv      a0, a2
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
    xor     t2, t1, a1
    sltiu   t2, t2, 1
    add     t0, t0, t2
    neg     t3, t2
    xor     t4, t1, a2
    and     t3, t3, t4
    xor     t3, t3, t1
    sb      t3, 0(a0)
    addi    a0, a0, 1
    addi    a3, a3, -1
    bnez    a3, .Lrb_loop
.Lrb_done:
    mv      a0, t0
    ret
    .size codevar_wredit_k_replace_byte, .-codevar_wredit_k_replace_byte

// ---------------------------------------------------------------------
// reverse_bytes(dst = a0, src = a1, len = a2) -> a0
// Non-overlapping; reads src forward and writes dst from the high end.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_reverse_bytes
    .type codevar_wredit_k_reverse_bytes, @function
codevar_wredit_k_reverse_bytes:
    mv      a3, a0
    beqz    a2, .Lrv_done
    add     t1, a0, a2
.Lrv_loop:
    lbu     t2, 0(a1)
    addi    a1, a1, 1
    addi    t1, t1, -1
    sb      t2, 0(t1)
    addi    a2, a2, -1
    bnez    a2, .Lrv_loop
.Lrv_done:
    mv      a0, a3
    ret
    .size codevar_wredit_k_reverse_bytes, .-codevar_wredit_k_reverse_bytes

// ---------------------------------------------------------------------
// fill_pattern8(dst = a0, pattern = a1, len = a2) -> a0
// Repeats the eight bytes of `pattern`; the final partial block uses the
// first len % 8 pattern bytes.
// ---------------------------------------------------------------------
    .globl codevar_wredit_k_fill_pattern8
    .type codevar_wredit_k_fill_pattern8, @function
codevar_wredit_k_fill_pattern8:
    mv      a3, a0
    beqz    a2, .Lfp_done
    andi    t0, a0, 7
    beqz    t0, .Lfp_words
    li      t1, 8
    sub     t1, t1, t0
    slli    t3, t1, 3              // head length in bits
    li      t4, 64
    sub     t4, t4, t3
    mv      t2, a1
.Lfp_head:
    sb      t2, 0(a0)
    srli    t2, t2, 8
    addi    a0, a0, 1
    addi    a2, a2, -1
    addi    t1, t1, -1
    beqz    a2, .Lfp_done
    bnez    t1, .Lfp_head
    // Rotate the pattern right by the head length so the word loop starts
    // at the correct phase instead of repeating from pattern byte zero.
    sll     t2, a1, t4
    srl     a1, a1, t3
    or      a1, a1, t2
.Lfp_words:
    li      t0, 8
    bltu    a2, t0, .Lfp_bytes
.Lfp_wloop:
    sd      a1, 0(a0)
    addi    a0, a0, 8
    addi    a2, a2, -8
    bgeu    a2, t0, .Lfp_wloop
.Lfp_bytes:
    beqz    a2, .Lfp_done
    mv      t2, a1
.Lfp_bloop:
    sb      t2, 0(a0)
    srli    t2, t2, 8
    addi    a0, a0, 1
    addi    a2, a2, -1
    bnez    a2, .Lfp_bloop
.Lfp_done:
    mv      a0, a3
    ret
    .size codevar_wredit_k_fill_pattern8, .-codevar_wredit_k_fill_pattern8

// ---------------------------------------------------------------------
// byte_histogram(counts = a0, src = a1, len = a2)
// Zero 256 u32 counters, then increment counts[byte] for every byte.
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
