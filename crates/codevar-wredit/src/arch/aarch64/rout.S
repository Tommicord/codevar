// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

// AArch64 (ARMv8-A) auxiliary kernels for codevar-wredit.
//
// AAPCS64 argument registers are x0-x7 and the return value is x0.
// Every routine below is a leaf function that only clobbers caller-saved
// state (x0-x15, v0-v7), so x19-x29 and the stack are never touched.
// NEON is mandatory in ARMv8-A and is therefore used unguarded for the
// bulk loops; every routine keeps a scalar path for short lengths.
//
// A length/count of zero never dereferences a pointer: each entry point
// tests the count before the first load or store and returns the
// documented zero/null/dst value.

    .text

// ---------------------------------------------------------------------
// codevar_wredit_k_fill_bytes(dst: *mut u8, value: u8, len: usize)
//   -> *mut u8
// x0 = dst, w1 = value, x2 = len; returns x0 = dst.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_fill_bytes
    .type codevar_wredit_k_fill_bytes, %function
codevar_wredit_k_fill_bytes:
    mov     x8, x0
    cbz     x2, .Lfby_done
    and     w1, w1, #0xff
    cmp     x2, #16
    b.lo    .Lfby_tail
    dup     v0.16b, w1
.Lfby_loop:
    str     q0, [x0], #16
    sub     x2, x2, #16
    cmp     x2, #16
    b.hs    .Lfby_loop
.Lfby_tail:
    cbz     x2, .Lfby_done
.Lfby_byte:
    strb    w1, [x0], #1
    sub     x2, x2, #1
    cbnz    x2, .Lfby_byte
.Lfby_done:
    mov     x0, x8
    ret
    .size codevar_wredit_k_fill_bytes, .-codevar_wredit_k_fill_bytes

// ---------------------------------------------------------------------
// codevar_wredit_k_fill_words(dst: *mut u32, value: u32, count: usize)
//   -> *mut u32
// x0 = dst, w1 = value, x2 = count; returns x0 = dst.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_fill_words
    .type codevar_wredit_k_fill_words, %function
codevar_wredit_k_fill_words:
    mov     x8, x0
    cbz     x2, .Lfwd_done
    cmp     x2, #4
    b.lo    .Lfwd_word
    dup     v0.4s, w1
.Lfwd_loop:
    str     q0, [x0], #16
    sub     x2, x2, #4
    cmp     x2, #4
    b.hs    .Lfwd_loop
.Lfwd_word:
    cbz     x2, .Lfwd_done
.Lfwd_one:
    str     w1, [x0], #4
    sub     x2, x2, #1
    cbnz    x2, .Lfwd_one
.Lfwd_done:
    mov     x0, x8
    ret
    .size codevar_wredit_k_fill_words, .-codevar_wredit_k_fill_words

// ---------------------------------------------------------------------
// codevar_wredit_k_copy_words(dst: *mut u32, src: *const u32,
//   count: usize) -> *mut u32
// x0 = dst, x1 = src, x2 = count; returns x0 = dst. No overlap allowed.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_copy_words
    .type codevar_wredit_k_copy_words, %function
codevar_wredit_k_copy_words:
    mov     x8, x0
    cbz     x2, .Lcwd_done
    cmp     x2, #8
    b.lo    .Lcwd_word
.Lcwd_loop:
    ldp     q0, q1, [x1]
    stp     q0, q1, [x0]
    add     x1, x1, #32
    add     x0, x0, #32
    sub     x2, x2, #8
    cmp     x2, #8
    b.hs    .Lcwd_loop
.Lcwd_word:
    cbz     x2, .Lcwd_done
.Lcwd_one:
    ldr     w3, [x1], #4
    str     w3, [x0], #4
    sub     x2, x2, #1
    cbnz    x2, .Lcwd_one
.Lcwd_done:
    mov     x0, x8
    ret
    .size codevar_wredit_k_copy_words, .-codevar_wredit_k_copy_words

// ---------------------------------------------------------------------
// codevar_wredit_k_move_words(dst: *mut u32, src: *const u32,
//   count: usize) -> *mut u32
// x0 = dst, x1 = src, x2 = count; returns x0 = dst. Overlap-safe:
// forward when dst < src, backward otherwise.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_move_words
    .type codevar_wredit_k_move_words, %function
codevar_wredit_k_move_words:
    mov     x8, x0
    cbz     x2, .Lmwd_done
    cmp     x0, x1
    b.eq    .Lmwd_done
    b.hs    .Lmwd_back
.Lmwd_fwd:
    ldr     w3, [x1], #4
    str     w3, [x0], #4
    sub     x2, x2, #1
    cbnz    x2, .Lmwd_fwd
    b       .Lmwd_done
.Lmwd_back:
    add     x1, x1, x2, lsl #2
    add     x0, x0, x2, lsl #2
.Lmwd_back_loop:
    ldr     w3, [x1, #-4]!
    str     w3, [x0, #-4]!
    sub     x2, x2, #1
    cbnz    x2, .Lmwd_back_loop
.Lmwd_done:
    mov     x0, x8
    ret
    .size codevar_wredit_k_move_words, .-codevar_wredit_k_move_words

// ---------------------------------------------------------------------
// codevar_wredit_k_count_byte(hay: *const u8, needle: u8, len: usize)
//   -> usize
// x0 = hay, w1 = needle, x2 = len; returns the number of matching bytes.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_count_byte
    .type codevar_wredit_k_count_byte, %function
codevar_wredit_k_count_byte:
    and     w1, w1, #0xff
    cbz     x2, .Lcnb_zero
    mov     x3, #0
    cmp     x2, #16
    b.lo    .Lcnb_tail
    dup     v0.16b, w1
.Lcnb_loop:
    ldr     q1, [x0], #16
    cmeq    v1.16b, v1.16b, v0.16b
    ushr    v1.16b, v1.16b, #7
    uaddlv  h2, v1.16b
    umov    w4, v2.h[0]
    add     x3, x3, x4
    sub     x2, x2, #16
    cmp     x2, #16
    b.hs    .Lcnb_loop
.Lcnb_tail:
    cbz     x2, .Lcnb_done
.Lcnb_byte:
    ldrb    w4, [x0], #1
    cmp     w4, w1
    cinc    x3, x3, eq
    sub     x2, x2, #1
    cbnz    x2, .Lcnb_byte
.Lcnb_done:
    mov     x0, x3
    ret
.Lcnb_zero:
    mov     x0, #0
    ret
    .size codevar_wredit_k_count_byte, .-codevar_wredit_k_count_byte

// ---------------------------------------------------------------------
// codevar_wredit_k_find_byte_rev(hay: *const u8, needle: u8, len: usize)
//   -> *const u8
// x0 = hay, w1 = needle, x2 = len; returns the last match or null.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_find_byte_rev
    .type codevar_wredit_k_find_byte_rev, %function
codevar_wredit_k_find_byte_rev:
    and     w1, w1, #0xff
    cbz     x2, .Lfbr_none
    add     x0, x0, x2
.Lfbr_loop:
    ldrb    w3, [x0, #-1]!
    cmp     w3, w1
    b.eq    .Lfbr_found
    sub     x2, x2, #1
    cbnz    x2, .Lfbr_loop
.Lfbr_none:
    mov     x0, #0
    ret
.Lfbr_found:
    ret
    .size codevar_wredit_k_find_byte_rev, .-codevar_wredit_k_find_byte_rev

// ---------------------------------------------------------------------
// codevar_wredit_k_find_any2(hay: *const u8, a: u8, b: u8, len: usize)
//   -> *const u8
// x0 = hay, w1 = a, w2 = b, x3 = len; returns the first byte equal to
// `a` or `b`, or null.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_find_any2
    .type codevar_wredit_k_find_any2, %function
codevar_wredit_k_find_any2:
    and     w1, w1, #0xff
    and     w2, w2, #0xff
    cbz     x3, .Lfa2_none
    cmp     x3, #16
    b.lo    .Lfa2_scan
    dup     v0.16b, w1
    dup     v1.16b, w2
.Lfa2_loop:
    ldr     q2, [x0]
    cmeq    v3.16b, v2.16b, v0.16b
    cmeq    v2.16b, v2.16b, v1.16b
    orr     v3.16b, v3.16b, v2.16b
    umaxv   b4, v3.16b
    umov    w4, v4.b[0]
    cbnz    w4, .Lfa2_hit
    add     x0, x0, #16
    sub     x3, x3, #16
    cmp     x3, #16
    b.hs    .Lfa2_loop
.Lfa2_scan:
    cbz     x3, .Lfa2_none
.Lfa2_byte:
    ldrb    w4, [x0]
    cmp     w4, w1
    b.eq    .Lfa2_found
    cmp     w4, w2
    b.eq    .Lfa2_found
    add     x0, x0, #1
    sub     x3, x3, #1
    cbnz    x3, .Lfa2_byte
.Lfa2_none:
    mov     x0, #0
    ret
.Lfa2_found:
    ret
.Lfa2_hit:
    // Locate the first set byte of the 16-lane match mask.
    umov    x4, v3.d[0]
    cbnz    x4, .Lfa2_hit_low
    umov    x4, v3.d[1]
    add     x0, x0, #8
.Lfa2_hit_low:
    rbit    x4, x4
    clz     x4, x4
    lsr     x4, x4, #3
    add     x0, x0, x4
    ret
    .size codevar_wredit_k_find_any2, .-codevar_wredit_k_find_any2

// ---------------------------------------------------------------------
// codevar_wredit_k_find_not_byte(hay: *const u8, needle: u8, len: usize)
//   -> *const u8
// x0 = hay, w1 = needle, x2 = len; returns the first byte different from
// `needle`, or null.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_find_not_byte
    .type codevar_wredit_k_find_not_byte, %function
codevar_wredit_k_find_not_byte:
    and     w1, w1, #0xff
    cbz     x2, .Lfnb_none
    cmp     x2, #16
    b.lo    .Lfnb_scan
    dup     v0.16b, w1
.Lfnb_loop:
    ldr     q1, [x0]
    cmeq    v1.16b, v1.16b, v0.16b
    mvn     v1.16b, v1.16b
    umaxv   b2, v1.16b
    umov    w3, v2.b[0]
    cbnz    w3, .Lfnb_hit
    add     x0, x0, #16
    sub     x2, x2, #16
    cmp     x2, #16
    b.hs    .Lfnb_loop
.Lfnb_scan:
    cbz     x2, .Lfnb_none
.Lfnb_byte:
    ldrb    w3, [x0]
    cmp     w3, w1
    b.ne    .Lfnb_found
    add     x0, x0, #1
    sub     x2, x2, #1
    cbnz    x2, .Lfnb_byte
.Lfnb_none:
    mov     x0, #0
    ret
.Lfnb_found:
    ret
.Lfnb_hit:
    // Locate the first non-needle byte in the chunk mask.
    umov    x3, v1.d[0]
    cbnz    x3, .Lfnb_hit_low
    umov    x3, v1.d[1]
    add     x0, x0, #8
.Lfnb_hit_low:
    rbit    x3, x3
    clz     x3, x3
    lsr     x3, x3, #3
    add     x0, x0, x3
    ret
    .size codevar_wredit_k_find_not_byte, .-codevar_wredit_k_find_not_byte

// ---------------------------------------------------------------------
// codevar_wredit_k_count_eq_u32(src: *const u32, value: u32, mask: u32,
//   count: usize) -> usize
// x0 = src, w1 = value, w2 = mask, x3 = count. Counts words where
// (word & mask) == value.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_count_eq_u32
    .type codevar_wredit_k_count_eq_u32, %function
codevar_wredit_k_count_eq_u32:
    cbz     x3, .Lceq_zero
    mov     x4, #0
    cmp     x3, #4
    b.lo    .Lceq_tail
    dup     v0.4s, w1
    dup     v2.4s, w2
.Lceq_loop:
    ldr     q3, [x0], #16
    and     v3.16b, v3.16b, v2.16b
    cmeq    v3.4s, v3.4s, v0.4s
    ushr    v3.4s, v3.4s, #31
    addv    s3, v3.4s
    umov    w5, v3.s[0]
    add     x4, x4, x5
    sub     x3, x3, #4
    cmp     x3, #4
    b.hs    .Lceq_loop
.Lceq_tail:
    cbz     x3, .Lceq_done
.Lceq_word:
    ldr     w5, [x0], #4
    and     w5, w5, w2
    cmp     w5, w1
    cinc    x4, x4, eq
    sub     x3, x3, #1
    cbnz    x3, .Lceq_word
.Lceq_done:
    mov     x0, x4
    ret
.Lceq_zero:
    mov     x0, #0
    ret
    .size codevar_wredit_k_count_eq_u32, .-codevar_wredit_k_count_eq_u32

// ---------------------------------------------------------------------
// codevar_wredit_k_find_eq_u32(src: *const u32, value: u32, mask: u32,
//   count: usize) -> *const u32
// x0 = src, w1 = value, w2 = mask, x3 = count; returns the first word
// where (word & mask) == value, or null.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_find_eq_u32
    .type codevar_wredit_k_find_eq_u32, %function
codevar_wredit_k_find_eq_u32:
    cbz     x3, .Lfeq_none
    cmp     x3, #4
    b.lo    .Lfeq_scan
    dup     v0.4s, w1
    dup     v2.4s, w2
.Lfeq_loop:
    ldr     q3, [x0]
    and     v3.16b, v3.16b, v2.16b
    cmeq    v3.4s, v3.4s, v0.4s
    umaxv   b4, v3.16b
    umov    w4, v4.b[0]
    cbnz    w4, .Lfeq_hit
    add     x0, x0, #16
    sub     x3, x3, #4
    cmp     x3, #4
    b.hs    .Lfeq_loop
.Lfeq_scan:
    cbz     x3, .Lfeq_none
.Lfeq_word:
    ldr     w4, [x0]
    and     w4, w4, w2
    cmp     w4, w1
    b.eq    .Lfeq_found
    add     x0, x0, #4
    sub     x3, x3, #1
    cbnz    x3, .Lfeq_word
.Lfeq_none:
    mov     x0, #0
    ret
.Lfeq_found:
    ret
.Lfeq_hit:
    // Each matching word sets 32 mask bits; find the lowest set bit.
    umov    x4, v3.d[0]
    cbnz    x4, .Lfeq_hit_low
    umov    x4, v3.d[1]
    add     x0, x0, #8
.Lfeq_hit_low:
    rbit    x4, x4
    clz     x4, x4
    lsr     x4, x4, #5
    add     x0, x0, x4, lsl #2
    ret
    .size codevar_wredit_k_find_eq_u32, .-codevar_wredit_k_find_eq_u32

// ---------------------------------------------------------------------
// codevar_wredit_k_find_eq_u32_rev(src: *const u32, value: u32,
//   mask: u32, count: usize) -> *const u32
// x0 = src, w1 = value, w2 = mask, x3 = count; returns the last word
// where (word & mask) == value, or null.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_find_eq_u32_rev
    .type codevar_wredit_k_find_eq_u32_rev, %function
codevar_wredit_k_find_eq_u32_rev:
    cbz     x3, .Lfer_none
    add     x0, x0, x3, lsl #2
.Lfer_loop:
    ldr     w4, [x0, #-4]!
    and     w4, w4, w2
    cmp     w4, w1
    b.eq    .Lfer_found
    sub     x3, x3, #1
    cbnz    x3, .Lfer_loop
.Lfer_none:
    mov     x0, #0
    ret
.Lfer_found:
    ret
    .size codevar_wredit_k_find_eq_u32_rev, .-codevar_wredit_k_find_eq_u32_rev

// ---------------------------------------------------------------------
// codevar_wredit_k_memxor(dst: *mut u8, src: *const u8, len: usize)
//   -> *mut u8
// x0 = dst, x1 = src, x2 = len; dst[i] ^= src[i]; returns x0 = dst.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_memxor
    .type codevar_wredit_k_memxor, %function
codevar_wredit_k_memxor:
    mov     x8, x0
    cbz     x2, .Lmxr_done
    cmp     x2, #16
    b.lo    .Lmxr_tail
.Lmxr_loop:
    ldr     q0, [x0]
    ldr     q1, [x1], #16
    eor     v0.16b, v0.16b, v1.16b
    str     q0, [x0], #16
    sub     x2, x2, #16
    cmp     x2, #16
    b.hs    .Lmxr_loop
.Lmxr_tail:
    cbz     x2, .Lmxr_done
.Lmxr_byte:
    ldrb    w3, [x1], #1
    ldrb    w4, [x0]
    eor     w3, w3, w4
    strb    w3, [x0], #1
    sub     x2, x2, #1
    cbnz    x2, .Lmxr_byte
.Lmxr_done:
    mov     x0, x8
    ret
    .size codevar_wredit_k_memxor, .-codevar_wredit_k_memxor

// ---------------------------------------------------------------------
// codevar_wredit_k_memcmp(a: *const u8, b: *const u8, len: usize) -> i32
// x0 = a, x1 = b, x2 = len. Returns 0 when equal, -1 when the first
// differing byte has a < b, 1 when a > b.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_memcmp
    .type codevar_wredit_k_memcmp, %function
codevar_wredit_k_memcmp:
    cbz     x2, .Lmcp_equal
    cmp     x2, #8
    b.lo    .Lmcp_bytes
.Lmcp_word_loop:
    ldr     x3, [x0], #8
    ldr     x4, [x1], #8
    cmp     x3, x4
    b.ne    .Lmcp_word_diff
    sub     x2, x2, #8
    cmp     x2, #8
    b.hs    .Lmcp_word_loop
.Lmcp_bytes:
    cbz     x2, .Lmcp_equal
.Lmcp_byte_loop:
    ldrb    w3, [x0], #1
    ldrb    w4, [x1], #1
    cmp     w3, w4
    b.ne    .Lmcp_diff
    sub     x2, x2, #1
    cbnz    x2, .Lmcp_byte_loop
.Lmcp_equal:
    mov     w0, #0
    ret
.Lmcp_word_diff:
    // x3 and x4 are little-endian words; the first differing byte is the
    // one holding the lowest differing bit.
    eor     x5, x3, x4
    rbit    x5, x5
    clz     x5, x5
    lsr     x5, x5, #3
    sub     x6, x0, #8
    sub     x7, x1, #8
    ldrb    w3, [x6, x5]
    ldrb    w4, [x7, x5]
.Lmcp_diff:
    cmp     w3, w4
    mov     w0, #1
    cneg    w0, w0, lo
    ret
    .size codevar_wredit_k_memcmp, .-codevar_wredit_k_memcmp

// ---------------------------------------------------------------------
// codevar_wredit_k_sum_bytes(src: *const u8, len: usize) -> u64
// x0 = src, x1 = len; returns the sum of zero-extended bytes.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_sum_bytes
    .type codevar_wredit_k_sum_bytes, %function
codevar_wredit_k_sum_bytes:
    cbz     x1, .Lsmb_zero
    mov     x3, #0
    cmp     x1, #16
    b.lo    .Lsmb_tail
.Lsmb_loop:
    ldr     q1, [x0], #16
    uaddlv  h2, v1.16b
    umov    w4, v2.h[0]
    add     x3, x3, x4
    sub     x1, x1, #16
    cmp     x1, #16
    b.hs    .Lsmb_loop
.Lsmb_tail:
    cbz     x1, .Lsmb_done
.Lsmb_byte:
    ldrb    w4, [x0], #1
    add     x3, x3, x4
    sub     x1, x1, #1
    cbnz    x1, .Lsmb_byte
.Lsmb_done:
    mov     x0, x3
    ret
.Lsmb_zero:
    mov     x0, #0
    ret
    .size codevar_wredit_k_sum_bytes, .-codevar_wredit_k_sum_bytes

// ---------------------------------------------------------------------
// codevar_wredit_k_weighted_checksum(src: *const u8, len: usize,
//   seed: u32) -> u64
// x0 = src, x1 = len, w2 = seed. weight starts at seed % 251 + 1 and
// walks 1..=251, wrapping to 1; sum += byte * weight.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_weighted_checksum
    .type codevar_wredit_k_weighted_checksum, %function
codevar_wredit_k_weighted_checksum:
    cbz     x1, .Lwck_zero
    mov     w3, #251
    udiv    w4, w2, w3
    msub    w3, w4, w3, w2
    add     w3, w3, #1
    mov     x4, #0
.Lwck_loop:
    ldrb    w5, [x0], #1
    mul     w6, w5, w3
    add     x4, x4, x6
    add     w3, w3, #1
    cmp     w3, #251
    b.ls    .Lwck_next
    mov     w3, #1
.Lwck_next:
    sub     x1, x1, #1
    cbnz    x1, .Lwck_loop
    mov     x0, x4
    ret
.Lwck_zero:
    mov     x0, #0
    ret
    .size codevar_wredit_k_weighted_checksum, .-codevar_wredit_k_weighted_checksum

// ---------------------------------------------------------------------
// codevar_wredit_k_utf8_count_starts(src: *const u8, len: usize) -> usize
// x0 = src, x1 = len; counts bytes with (byte & 0xC0) != 0x80.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_utf8_count_starts
    .type codevar_wredit_k_utf8_count_starts, %function
codevar_wredit_k_utf8_count_starts:
    cbz     x1, .Lutf_zero
    mov     x3, #0
    cmp     x1, #16
    b.lo    .Lutf_tail
    mov     w4, #0x80
    mov     w5, #0xc0
    dup     v0.16b, w4
    dup     v2.16b, w5
    mov     w7, #16
.Lutf_loop:
    ldr     q1, [x0], #16
    and     v1.16b, v1.16b, v2.16b
    cmeq    v1.16b, v1.16b, v0.16b
    ushr    v1.16b, v1.16b, #7
    uaddlv  h3, v1.16b
    umov    w6, v3.h[0]
    sub     w6, w7, w6
    add     x3, x3, x6
    sub     x1, x1, #16
    cmp     x1, #16
    b.hs    .Lutf_loop
.Lutf_tail:
    cbz     x1, .Lutf_done
.Lutf_byte:
    ldrb    w6, [x0], #1
    and     w6, w6, #0xc0
    cmp     w6, #0x80
    cinc    x3, x3, ne
    sub     x1, x1, #1
    cbnz    x1, .Lutf_byte
.Lutf_done:
    mov     x0, x3
    ret
.Lutf_zero:
    mov     x0, #0
    ret
    .size codevar_wredit_k_utf8_count_starts, .-codevar_wredit_k_utf8_count_starts

// ---------------------------------------------------------------------
// codevar_wredit_k_replace_byte(dst: *mut u8, from: u8, to: u8,
//   len: usize) -> usize
// x0 = dst, w1 = from, w2 = to, x3 = len; replaces in place and returns
// the number of bytes replaced.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_replace_byte
    .type codevar_wredit_k_replace_byte, %function
codevar_wredit_k_replace_byte:
    and     w1, w1, #0xff
    and     w2, w2, #0xff
    cbz     x3, .Lrpb_zero
    mov     x4, #0
    cmp     x3, #16
    b.lo    .Lrpb_tail
    dup     v0.16b, w1
    dup     v1.16b, w2
.Lrpb_loop:
    ldr     q2, [x0]
    cmeq    v3.16b, v2.16b, v0.16b
    bic     v4.16b, v2.16b, v3.16b
    and     v5.16b, v1.16b, v3.16b
    orr     v2.16b, v4.16b, v5.16b
    str     q2, [x0], #16
    ushr    v3.16b, v3.16b, #7
    uaddlv  h5, v3.16b
    umov    w6, v5.h[0]
    add     x4, x4, x6
    sub     x3, x3, #16
    cmp     x3, #16
    b.hs    .Lrpb_loop
.Lrpb_tail:
    cbz     x3, .Lrpb_done
.Lrpb_byte:
    ldrb    w6, [x0]
    cmp     w6, w1
    b.ne    .Lrpb_next
    strb    w2, [x0]
    add     x4, x4, #1
.Lrpb_next:
    add     x0, x0, #1
    sub     x3, x3, #1
    cbnz    x3, .Lrpb_byte
.Lrpb_done:
    mov     x0, x4
    ret
.Lrpb_zero:
    mov     x0, #0
    ret
    .size codevar_wredit_k_replace_byte, .-codevar_wredit_k_replace_byte

// ---------------------------------------------------------------------
// codevar_wredit_k_reverse_bytes(dst: *mut u8, src: *const u8,
//   len: usize) -> *mut u8
// x0 = dst, x1 = src, x2 = len; copies src[..len] reversed into
// dst[..len]; no overlap allowed.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_reverse_bytes
    .type codevar_wredit_k_reverse_bytes, %function
codevar_wredit_k_reverse_bytes:
    mov     x8, x0
    cbz     x2, .Lrvb_done
    add     x1, x1, x2
    cmp     x2, #8
    b.lo    .Lrvb_tail
.Lrvb_loop:
    ldr     x3, [x1, #-8]!
    rev     x3, x3
    str     x3, [x0], #8
    sub     x2, x2, #8
    cmp     x2, #8
    b.hs    .Lrvb_loop
.Lrvb_tail:
    cbz     x2, .Lrvb_done
.Lrvb_byte:
    ldrb    w3, [x1, #-1]!
    strb    w3, [x0], #1
    sub     x2, x2, #1
    cbnz    x2, .Lrvb_byte
.Lrvb_done:
    mov     x0, x8
    ret
    .size codevar_wredit_k_reverse_bytes, .-codevar_wredit_k_reverse_bytes

// ---------------------------------------------------------------------
// codevar_wredit_k_fill_pattern8(dst: *mut u8, pattern: u64,
//   len: usize) -> *mut u8
// x0 = dst, x1 = pattern, x2 = len; repeats the 8 native-order pattern
// bytes, with the final partial block using the low len % 8 bytes.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_fill_pattern8
    .type codevar_wredit_k_fill_pattern8, %function
codevar_wredit_k_fill_pattern8:
    mov     x8, x0
    cbz     x2, .Lfp8_done
    cmp     x2, #8
    b.lo    .Lfp8_tail
.Lfp8_loop:
    str     x1, [x0], #8
    sub     x2, x2, #8
    cmp     x2, #8
    b.hs    .Lfp8_loop
.Lfp8_tail:
    cbz     x2, .Lfp8_done
.Lfp8_byte:
    strb    w1, [x0], #1
    lsr     x1, x1, #8
    sub     x2, x2, #1
    cbnz    x2, .Lfp8_byte
.Lfp8_done:
    mov     x0, x8
    ret
    .size codevar_wredit_k_fill_pattern8, .-codevar_wredit_k_fill_pattern8

// ---------------------------------------------------------------------
// codevar_wredit_k_byte_histogram(counts: *mut u32, src: *const u8,
//   len: usize) -> ()
// x0 = counts, x1 = src, x2 = len; zeroes the 256 counters, then
// increments counts[byte] for every input byte.
// ---------------------------------------------------------------------
    .align 6
    .global codevar_wredit_k_byte_histogram
    .type codevar_wredit_k_byte_histogram, %function
codevar_wredit_k_byte_histogram:
    // 256 u32 counters == 1024 bytes == 32 paired 16-byte stores.
    eor     v0.16b, v0.16b, v0.16b
    mov     x3, x0
    mov     x4, #32
.Lhis_zero:
    stp     q0, q0, [x3], #32
    sub     x4, x4, #1
    cbnz    x4, .Lhis_zero
    cbz     x2, .Lhis_done
.Lhis_loop:
    ldrb    w5, [x1], #1
    ldr     w6, [x0, x5, lsl #2]
    add     w6, w6, #1
    str     w6, [x0, x5, lsl #2]
    sub     x2, x2, #1
    cbnz    x2, .Lhis_loop
.Lhis_done:
    ret
    .size codevar_wredit_k_byte_histogram, .-codevar_wredit_k_byte_histogram
