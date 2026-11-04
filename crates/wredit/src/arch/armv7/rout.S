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

// ARMv7-A (A32) scalar kernels.
//
// Every routine uses the AAPCS argument registers (r0-r3), returns in r0
// (or r0:r1 for the u64 results) and preserves every callee-saved
// register it touches. The file is position independent: the only
// immediates are movw/movt constants, never absolute addresses.
//
// All loops are scalar; NEON is not required. A length or count of zero
// never touches an input buffer (`byte_histogram` still clears its 256
// counters, as documented).
//
// `fill_pattern8` takes a 64-bit pattern as its second argument, which
// AAPCS places in the aligned register pair r2:r3 (r1 is skipped); the
// third argument (len) therefore arrives on the stack at [sp].

    .syntax unified
    .text
    .arm

// ---------------------------------------------------------------------
// 1. u8 *codevar_wredit_k_fill_bytes(u8 *dst, u8 value, size_t len)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_fill_bytes
    .type codevar_wredit_k_fill_bytes, %function
codevar_wredit_k_fill_bytes:
    mov     r3, r0
    and     r1, r1, #0xff
    cmp     r2, #0
    bxeq    lr
    orr     r12, r1, r1, lsl #8
    orr     r12, r12, r12, lsl #16
    tst     r0, #3
    beq     .Lfb_words_check
.Lfb_align:
    strb    r1, [r0], #1
    subs    r2, r2, #1
    beq     .Lfb_done
    tst     r0, #3
    bne     .Lfb_align
.Lfb_words_check:
    cmp     r2, #4
    blo     .Lfb_bytes
.Lfb_words:
    str     r12, [r0], #4
    subs    r2, r2, #4
    cmp     r2, #4
    bhs     .Lfb_words
.Lfb_bytes:
    cmp     r2, #0
    beq     .Lfb_done
.Lfb_byte_loop:
    strb    r1, [r0], #1
    subs    r2, r2, #1
    bne     .Lfb_byte_loop
.Lfb_done:
    mov     r0, r3
    bx      lr
    .size codevar_wredit_k_fill_bytes, .-codevar_wredit_k_fill_bytes

// ---------------------------------------------------------------------
// 2. u32 *codevar_wredit_k_fill_words(u32 *dst, u32 value, size_t count)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_fill_words
    .type codevar_wredit_k_fill_words, %function
codevar_wredit_k_fill_words:
    mov     r3, r0
    cmp     r2, #0
    bxeq    lr
    cmp     r2, #4
    blo     .Lfw_single
.Lfw_four:
    str     r1, [r0], #4
    str     r1, [r0], #4
    str     r1, [r0], #4
    str     r1, [r0], #4
    subs    r2, r2, #4
    cmp     r2, #4
    bhs     .Lfw_four
.Lfw_single:
    cmp     r2, #0
    beq     .Lfw_done
.Lfw_loop:
    str     r1, [r0], #4
    subs    r2, r2, #1
    bne     .Lfw_loop
.Lfw_done:
    mov     r0, r3
    bx      lr
    .size codevar_wredit_k_fill_words, .-codevar_wredit_k_fill_words

// ---------------------------------------------------------------------
// 3. u32 *codevar_wredit_k_copy_words(u32 *dst, const u32 *src, size_t count)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_copy_words
    .type codevar_wredit_k_copy_words, %function
codevar_wredit_k_copy_words:
    mov     r3, r0
    cmp     r2, #0
    bxeq    lr
    cmp     r2, #4
    blo     .Lcw_single
.Lcw_four:
    ldr     r12, [r1], #4
    str     r12, [r0], #4
    ldr     r12, [r1], #4
    str     r12, [r0], #4
    ldr     r12, [r1], #4
    str     r12, [r0], #4
    ldr     r12, [r1], #4
    str     r12, [r0], #4
    subs    r2, r2, #4
    cmp     r2, #4
    bhs     .Lcw_four
.Lcw_single:
    cmp     r2, #0
    beq     .Lcw_done
.Lcw_loop:
    ldr     r12, [r1], #4
    str     r12, [r0], #4
    subs    r2, r2, #1
    bne     .Lcw_loop
.Lcw_done:
    mov     r0, r3
    bx      lr
    .size codevar_wredit_k_copy_words, .-codevar_wredit_k_copy_words

// ---------------------------------------------------------------------
// 4. u32 *codevar_wredit_k_move_words(u32 *dst, const u32 *src, size_t count)
//
// Forward when dst < src, backward when dst > src, no-op when equal.
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_move_words
    .type codevar_wredit_k_move_words, %function
codevar_wredit_k_move_words:
    mov     r3, r0
    cmp     r2, #0
    bxeq    lr
    cmp     r0, r1
    blo     .Lmw_forward
    bxeq    lr
    add     r1, r1, r2, lsl #2
    add     r0, r0, r2, lsl #2
.Lmw_back_loop:
    sub     r1, r1, #4
    sub     r0, r0, #4
    ldr     r12, [r1]
    str     r12, [r0]
    subs    r2, r2, #1
    bne     .Lmw_back_loop
    b       .Lmw_done
.Lmw_forward:
.Lmw_fwd_loop:
    ldr     r12, [r1], #4
    str     r12, [r0], #4
    subs    r2, r2, #1
    bne     .Lmw_fwd_loop
.Lmw_done:
    mov     r0, r3
    bx      lr
    .size codevar_wredit_k_move_words, .-codevar_wredit_k_move_words

// ---------------------------------------------------------------------
// 5. size_t codevar_wredit_k_count_byte(const u8 *hay, u8 needle, size_t len)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_count_byte
    .type codevar_wredit_k_count_byte, %function
codevar_wredit_k_count_byte:
    and     r1, r1, #0xff
    mov     r3, #0
    cmp     r2, #0
    moveq   r0, #0
    bxeq    lr
.Lcb_loop:
    ldrb    r12, [r0], #1
    cmp     r12, r1
    addeq   r3, r3, #1
    subs    r2, r2, #1
    bne     .Lcb_loop
    mov     r0, r3
    bx      lr
    .size codevar_wredit_k_count_byte, .-codevar_wredit_k_count_byte

// ---------------------------------------------------------------------
// 6. const u8 *codevar_wredit_k_find_byte_rev(const u8 *hay, u8 needle,
//                                             size_t len)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_find_byte_rev
    .type codevar_wredit_k_find_byte_rev, %function
codevar_wredit_k_find_byte_rev:
    and     r1, r1, #0xff
    cmp     r2, #0
    moveq   r0, #0
    bxeq    lr
    add     r0, r0, r2
.Lfbr_loop:
    ldrb    r12, [r0, #-1]!
    cmp     r12, r1
    beq     .Lfbr_hit
    subs    r2, r2, #1
    bne     .Lfbr_loop
    mov     r0, #0
.Lfbr_hit:
    bx      lr
    .size codevar_wredit_k_find_byte_rev, .-codevar_wredit_k_find_byte_rev

// ---------------------------------------------------------------------
// 7. const u8 *codevar_wredit_k_find_any2(const u8 *hay, u8 a, u8 b,
//                                         size_t len)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_find_any2
    .type codevar_wredit_k_find_any2, %function
codevar_wredit_k_find_any2:
    and     r1, r1, #0xff
    and     r2, r2, #0xff
    cmp     r3, #0
    moveq   r0, #0
    bxeq    lr
.Lfa_loop:
    ldrb    r12, [r0]
    cmp     r12, r1
    beq     .Lfa_hit
    cmp     r12, r2
    beq     .Lfa_hit
    add     r0, r0, #1
    subs    r3, r3, #1
    bne     .Lfa_loop
    mov     r0, #0
.Lfa_hit:
    bx      lr
    .size codevar_wredit_k_find_any2, .-codevar_wredit_k_find_any2

// ---------------------------------------------------------------------
// 8. const u8 *codevar_wredit_k_find_not_byte(const u8 *hay, u8 needle,
//                                             size_t len)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_find_not_byte
    .type codevar_wredit_k_find_not_byte, %function
codevar_wredit_k_find_not_byte:
    and     r1, r1, #0xff
    cmp     r2, #0
    moveq   r0, #0
    bxeq    lr
.Lfn_loop:
    ldrb    r12, [r0]
    cmp     r12, r1
    bne     .Lfn_hit
    add     r0, r0, #1
    subs    r2, r2, #1
    bne     .Lfn_loop
    mov     r0, #0
.Lfn_hit:
    bx      lr
    .size codevar_wredit_k_find_not_byte, .-codevar_wredit_k_find_not_byte

// ---------------------------------------------------------------------
// 9. size_t codevar_wredit_k_count_eq_u32(const u32 *src, u32 value,
//                                         u32 mask, size_t count)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_count_eq_u32
    .type codevar_wredit_k_count_eq_u32, %function
codevar_wredit_k_count_eq_u32:
    push    {r4, lr}
    and     r1, r1, r2
    mov     r4, #0
    cmp     r3, #0
    moveq   r0, #0
    beq     .Lce_done
.Lce_loop:
    ldr     r12, [r0], #4
    and     r12, r12, r2
    cmp     r12, r1
    addeq   r4, r4, #1
    subs    r3, r3, #1
    bne     .Lce_loop
    mov     r0, r4
.Lce_done:
    pop     {r4, pc}
    .size codevar_wredit_k_count_eq_u32, .-codevar_wredit_k_count_eq_u32

// ---------------------------------------------------------------------
// 10. const u32 *codevar_wredit_k_find_eq_u32(const u32 *src, u32 value,
//                                             u32 mask, size_t count)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_find_eq_u32
    .type codevar_wredit_k_find_eq_u32, %function
codevar_wredit_k_find_eq_u32:
    and     r1, r1, r2
    cmp     r3, #0
    moveq   r0, #0
    bxeq    lr
.Lfe_loop:
    ldr     r12, [r0], #4
    and     r12, r12, r2
    cmp     r12, r1
    beq     .Lfe_hit
    subs    r3, r3, #1
    bne     .Lfe_loop
    mov     r0, #0
    bx      lr
.Lfe_hit:
    sub     r0, r0, #4
    bx      lr
    .size codevar_wredit_k_find_eq_u32, .-codevar_wredit_k_find_eq_u32

// ---------------------------------------------------------------------
// 11. const u32 *codevar_wredit_k_find_eq_u32_rev(const u32 *src, u32 value,
//                                                 u32 mask, size_t count)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_find_eq_u32_rev
    .type codevar_wredit_k_find_eq_u32_rev, %function
codevar_wredit_k_find_eq_u32_rev:
    and     r1, r1, r2
    cmp     r3, #0
    moveq   r0, #0
    bxeq    lr
    add     r0, r0, r3, lsl #2
.Lfer_loop:
    ldr     r12, [r0, #-4]!
    and     r12, r12, r2
    cmp     r12, r1
    beq     .Lfer_hit
    subs    r3, r3, #1
    bne     .Lfer_loop
    mov     r0, #0
.Lfer_hit:
    bx      lr
    .size codevar_wredit_k_find_eq_u32_rev, .-codevar_wredit_k_find_eq_u32_rev

// ---------------------------------------------------------------------
// 12. u8 *codevar_wredit_k_memxor(u8 *dst, const u8 *src, size_t len)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_memxor
    .type codevar_wredit_k_memxor, %function
codevar_wredit_k_memxor:
    push    {r4, lr}
    mov     r3, r0
    cmp     r2, #0
    beq     .Lmx_done
    tst     r0, #3
    bne     .Lmx_bytes
    tst     r1, #3
    bne     .Lmx_bytes
    cmp     r2, #4
    blo     .Lmx_bytes
.Lmx_word:
    ldr     r12, [r1], #4
    ldr     r4, [r0]
    eor     r4, r4, r12
    str     r4, [r0], #4
    subs    r2, r2, #4
    cmp     r2, #4
    bhs     .Lmx_word
.Lmx_bytes:
    cmp     r2, #0
    beq     .Lmx_done
.Lmx_byte:
    ldrb    r12, [r1], #1
    ldrb    r4, [r0]
    eor     r4, r4, r12
    strb    r4, [r0], #1
    subs    r2, r2, #1
    bne     .Lmx_byte
.Lmx_done:
    mov     r0, r3
    pop     {r4, pc}
    .size codevar_wredit_k_memxor, .-codevar_wredit_k_memxor

// ---------------------------------------------------------------------
// 13. i32 codevar_wredit_k_memcmp(const u8 *a, const u8 *b, size_t len)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_memcmp
    .type codevar_wredit_k_memcmp, %function
codevar_wredit_k_memcmp:
    cmp     r2, #0
    moveq   r0, #0
    bxeq    lr
.Lcmp_loop:
    ldrb    r3, [r0], #1
    ldrb    r12, [r1], #1
    cmp     r3, r12
    bne     .Lcmp_diff
    subs    r2, r2, #1
    bne     .Lcmp_loop
    mov     r0, #0
    bx      lr
.Lcmp_diff:
    movhi   r0, #1
    movlo   r0, #-1
    bx      lr
    .size codevar_wredit_k_memcmp, .-codevar_wredit_k_memcmp

// ---------------------------------------------------------------------
// 14. u64 codevar_wredit_k_sum_bytes(const u8 *src, size_t len)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_sum_bytes
    .type codevar_wredit_k_sum_bytes, %function
codevar_wredit_k_sum_bytes:
    mov     r2, #0
    mov     r3, #0
    cmp     r1, #0
    beq     .Lsum_done
.Lsum_loop:
    ldrb    r12, [r0], #1
    adds    r2, r2, r12
    adc     r3, r3, #0
    subs    r1, r1, #1
    bne     .Lsum_loop
.Lsum_done:
    mov     r0, r2
    mov     r1, r3
    bx      lr
    .size codevar_wredit_k_sum_bytes, .-codevar_wredit_k_sum_bytes

// ---------------------------------------------------------------------
// 15. u64 codevar_wredit_k_weighted_checksum(const u8 *src, size_t len,
//                                            u32 seed)
//
// weight = seed % 251 + 1; weight cycles 1..251.
//                  2190262207
// seed / 251 == (seed * ----------) >> 39, exact for every u32 seed.
//                     2^39
// The magic constant keeps the routine free of the optional ARMv7-A
// integer divide instruction.
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_weighted_checksum
    .type codevar_wredit_k_weighted_checksum, %function
codevar_wredit_k_weighted_checksum:
    push    {r4-r6, lr}
    movw    r3, #0xbfbf
    movt    r3, #0x828c
    umull   r4, r5, r2, r3
    lsr     r4, r5, #7
    mov     r5, #251
    mul     r5, r4, r5
    sub     r2, r2, r5
    add     r2, r2, #1
    mov     r4, #0
    mov     r5, #0
    cmp     r1, #0
    beq     .Lwc_done
.Lwc_loop:
    ldrb    r3, [r0], #1
    mul     r6, r3, r2
    adds    r4, r4, r6
    adc     r5, r5, #0
    add     r2, r2, #1
    cmp     r2, #251
    movhi   r2, #1
    subs    r1, r1, #1
    bne     .Lwc_loop
.Lwc_done:
    mov     r0, r4
    mov     r1, r5
    pop     {r4-r6, pc}
    .size codevar_wredit_k_weighted_checksum, .-codevar_wredit_k_weighted_checksum

// ---------------------------------------------------------------------
// 16. size_t codevar_wredit_k_utf8_count_starts(const u8 *src, size_t len)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_utf8_count_starts
    .type codevar_wredit_k_utf8_count_starts, %function
codevar_wredit_k_utf8_count_starts:
    mov     r2, #0
    cmp     r1, #0
    moveq   r0, #0
    bxeq    lr
.Luc_loop:
    ldrb    r3, [r0], #1
    and     r3, r3, #0xc0
    cmp     r3, #0x80
    addne   r2, r2, #1
    subs    r1, r1, #1
    bne     .Luc_loop
    mov     r0, r2
    bx      lr
    .size codevar_wredit_k_utf8_count_starts, .-codevar_wredit_k_utf8_count_starts

// ---------------------------------------------------------------------
// 17. size_t codevar_wredit_k_replace_byte(u8 *dst, u8 from, u8 to,
//                                          size_t len)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_replace_byte
    .type codevar_wredit_k_replace_byte, %function
codevar_wredit_k_replace_byte:
    push    {r4, lr}
    and     r1, r1, #0xff
    and     r2, r2, #0xff
    mov     r12, #0
    cmp     r3, #0
    moveq   r0, #0
    beq     .Lrp_done
.Lrp_loop:
    ldrb    r4, [r0]
    cmp     r4, r1
    bne     .Lrp_next
    strb    r2, [r0]
    add     r12, r12, #1
.Lrp_next:
    add     r0, r0, #1
    subs    r3, r3, #1
    bne     .Lrp_loop
    mov     r0, r12
.Lrp_done:
    pop     {r4, pc}
    .size codevar_wredit_k_replace_byte, .-codevar_wredit_k_replace_byte

// ---------------------------------------------------------------------
// 18. u8 *codevar_wredit_k_reverse_bytes(u8 *dst, const u8 *src, size_t len)
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_reverse_bytes
    .type codevar_wredit_k_reverse_bytes, %function
codevar_wredit_k_reverse_bytes:
    mov     r3, r0
    cmp     r2, #0
    bxeq    lr
    add     r1, r1, r2
.Lrv_byte:
    ldrb    r12, [r1, #-1]!
    strb    r12, [r0], #1
    subs    r2, r2, #1
    bne     .Lrv_byte
.Lrv_done:
    mov     r0, r3
    bx      lr
    .size codevar_wredit_k_reverse_bytes, .-codevar_wredit_k_reverse_bytes

// ---------------------------------------------------------------------
// 19. u8 *codevar_wredit_k_fill_pattern8(u8 *dst, u64 pattern, size_t len)
//
// AAPCS: pattern (64-bit) is in the aligned pair r2:r3 and len is the
// first stacked argument at [sp].
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_fill_pattern8
    .type codevar_wredit_k_fill_pattern8, %function
codevar_wredit_k_fill_pattern8:
    ldr     r1, [sp]
    mov     r12, r0
    cmp     r1, #0
    bxeq    lr
    push    {r4, lr}
    tst     r0, #3
    beq     .Lfp_blocks
.Lfp_align:
    strb    r2, [r0], #1
    and     r4, r2, #0xff
    mov     r2, r2, lsr #8
    orr     r2, r2, r3, lsl #24
    mov     r3, r3, lsr #8
    orr     r3, r3, r4, lsl #24
    subs    r1, r1, #1
    beq     .Lfp_done
    tst     r0, #3
    bne     .Lfp_align
.Lfp_blocks:
    cmp     r1, #8
    blo     .Lfp_tail
.Lfp_block:
    str     r2, [r0], #4
    str     r3, [r0], #4
    subs    r1, r1, #8
    cmp     r1, #8
    bhs     .Lfp_block
.Lfp_tail:
    cmp     r1, #0
    beq     .Lfp_done
.Lfp_tail_loop:
    strb    r2, [r0], #1
    mov     r2, r2, lsr #8
    orr     r2, r2, r3, lsl #24
    mov     r3, r3, lsr #8
    subs    r1, r1, #1
    bne     .Lfp_tail_loop
.Lfp_done:
    mov     r0, r12
    pop     {r4, pc}
    .size codevar_wredit_k_fill_pattern8, .-codevar_wredit_k_fill_pattern8

// ---------------------------------------------------------------------
// 20. void codevar_wredit_k_byte_histogram(u32 *counts, const u8 *src,
//                                          size_t len)
//
// The 256 counters are cleared before the (possibly empty) input scan.
// ---------------------------------------------------------------------
    .p2align 2
    .globl codevar_wredit_k_byte_histogram
    .type codevar_wredit_k_byte_histogram, %function
codevar_wredit_k_byte_histogram:
    push    {r4, lr}
    mov     r4, r0
    mov     r12, #0
    mov     r3, #256
.Lbh_zero:
    str     r12, [r0], #4
    subs    r3, r3, #1
    bne     .Lbh_zero
    cmp     r2, #0
    beq     .Lbh_done
.Lbh_loop:
    ldrb    r3, [r1], #1
    ldr     r12, [r4, r3, lsl #2]
    add     r12, r12, #1
    str     r12, [r4, r3, lsl #2]
    subs    r2, r2, #1
    bne     .Lbh_loop
.Lbh_done:
    pop     {r4, pc}
    .size codevar_wredit_k_byte_histogram, .-codevar_wredit_k_byte_histogram
