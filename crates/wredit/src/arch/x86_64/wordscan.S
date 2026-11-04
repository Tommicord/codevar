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

// Spill-free packed-word newline kernels.
//
// The editor stores one code point per u32 with the character in the
// low 24 bits (`CHAR_MASK = 0x00FF_FFFF`, newline = 0x0A). The Rust
// scanners are already vectorised, but the count / nth loops keep too
// many live values for LLVM and spill around the tail and the
// count-leading-ones path. These routines keep everything in
// caller-saved registers and never touch the stack.
//
// One source is assembled per ISA level, exactly like `memchr.S`:
//   SSE2   - 4 words per vector, movmskps
//   AVX2   - 8 words per vector, vmovmskps
//   AVX512 - 16 words per vector, vpcmpd + kmovw
//
// ABI (SysV AMD64):
//   count        (raw, len)              -> rax = newline count
//   find_fwd     (raw, len, from)        -> rax = index, or -1
//   find_rev     (raw, before)           -> rax = index, or -1
//   find_nth     (raw, len, n)           -> rax = index, or -1 (n > 0)
//
// All loads are unaligned and stay inside the caller's bounds. The
// direction flag is never changed. `vzeroupper` is emitted for the
// AVX variants before every return.

#ifndef CODEVAR_SUFFIX
#define CODEVAR_SUFFIX
#endif

#define CODEVAR_CAT2(a, b) a##b
#define CODEVAR_CAT(a, b) CODEVAR_CAT2(a, b)
#define CODEVAR_FN(name) CODEVAR_CAT(name, CODEVAR_SUFFIX)

#if defined(CODEVAR_HAVE_AVX512)
#define VEC 16
#define VBYTES 64
#define VZERO vzeroupper
#define SETUP \
    movl $0x00FFFFFF, %eax ; \
    vmovd %eax, %xmm2 ; \
    vpbroadcastd %xmm2, %zmm2 ; \
    movl $10, %eax ; \
    vmovd %eax, %xmm1 ; \
    vpbroadcastd %xmm1, %zmm1
#define MASK_AT(off, out) \
    vmovdqu32 off(%r8), %zmm0 ; \
    vpandd %zmm2, %zmm0, %zmm0 ; \
    vpcmpd $0, %zmm1, %zmm0, %k1 ; \
    kmovw %k1, out
#elif defined(CODEVAR_HAVE_AVX2)
#define VEC 8
#define VBYTES 32
#define VZERO vzeroupper
#define SETUP \
    movl $0x00FFFFFF, %eax ; \
    vmovd %eax, %xmm2 ; \
    vpbroadcastd %xmm2, %ymm2 ; \
    movl $10, %eax ; \
    vmovd %eax, %xmm1 ; \
    vpbroadcastd %xmm1, %ymm1
#define MASK_AT(off, out) \
    vmovdqu off(%r8), %ymm0 ; \
    vpand %ymm2, %ymm0, %ymm0 ; \
    vpcmpeqd %ymm1, %ymm0, %ymm0 ; \
    vmovmskps %ymm0, out
#else
#define VEC 4
#define VBYTES 16
#define VZERO
#define SETUP \
    movl $0x00FFFFFF, %eax ; \
    movd %eax, %xmm2 ; \
    pshufd $0, %xmm2, %xmm2 ; \
    movl $10, %eax ; \
    movd %eax, %xmm1 ; \
    pshufd $0, %xmm1, %xmm1
#define MASK_AT(off, out) \
    movdqu off(%r8), %xmm0 ; \
    pand %xmm2, %xmm0 ; \
    pcmpeqd %xmm1, %xmm0 ; \
    movmskps %xmm0, out
#endif

// Population count of the 32-bit mask in `src`, written to `dst`.
// `src` is preserved. POPCNT is available on every AVX2-class CPU;
// the SSE2 build uses a SWAR fallback so the baseline remains valid on
// the oldest x86_64 parts.
#if defined(CODEVAR_HAVE_POPCNT)
#define POPCOUNT(src, dst) popcntl src, dst
#else
#define POPCOUNT(src, dst) \
    movl src, dst ; \
    movl dst, %r11d ; \
    shrl $1, %r11d ; \
    andl $0x55555555, dst ; \
    andl $0x55555555, %r11d ; \
    addl %r11d, dst ; \
    movl dst, %r11d ; \
    shrl $2, %r11d ; \
    andl $0x33333333, dst ; \
    andl $0x33333333, %r11d ; \
    addl %r11d, dst ; \
    movl dst, %r11d ; \
    shrl $4, %r11d ; \
    addl %r11d, dst ; \
    andl $0x0F0F0F0F, dst ; \
    imull $0x01010101, dst, dst ; \
    shrl $24, dst
#endif

    .text

// =====================================================================
// count_newlines(raw: *const u32, len: usize) -> usize
// rdi = raw, rsi = len; returns rax = number of newline words.
// =====================================================================
    .p2align 5
    .globl CODEVAR_FN(codevar_wredit_wc_count)
    .type CODEVAR_FN(codevar_wredit_wc_count), @function
CODEVAR_FN(codevar_wredit_wc_count):
    testq   %rsi, %rsi
    jz      .Lwc_count_zero
    movq    %rdi, %r8
    movq    %rsi, %r9
    SETUP
    xorl    %eax, %eax
.Lwc_count_wide:
    cmpq    $(4*VEC), %r9
    jb      .Lwc_count_vec
    MASK_AT(0, %ecx)
    POPCOUNT(%ecx, %edx)
    addq    %rdx, %rax
    MASK_AT(VBYTES, %ecx)
    POPCOUNT(%ecx, %edx)
    addq    %rdx, %rax
    MASK_AT(2*VBYTES, %ecx)
    POPCOUNT(%ecx, %edx)
    addq    %rdx, %rax
    MASK_AT(3*VBYTES, %ecx)
    POPCOUNT(%ecx, %edx)
    addq    %rdx, %rax
    addq    $(4*VBYTES), %r8
    subq    $(4*VEC), %r9
    jmp     .Lwc_count_wide
.Lwc_count_vec:
    cmpq    $VEC, %r9
    jb      .Lwc_count_tail
.Lwc_count_vec_loop:
    MASK_AT(0, %ecx)
    POPCOUNT(%ecx, %edx)
    addq    %rdx, %rax
    addq    $VBYTES, %r8
    subq    $VEC, %r9
    cmpq    $VEC, %r9
    jae     .Lwc_count_vec_loop
.Lwc_count_tail:
    testq   %r9, %r9
    jz      .Lwc_count_done
.Lwc_count_tail_loop:
    movl    (%r8), %ecx
    andl    $0x00FFFFFF, %ecx
    cmpl    $10, %ecx
    sete    %cl
    movzbl  %cl, %ecx
    addq    %rcx, %rax
    addq    $4, %r8
    decq    %r9
    jnz     .Lwc_count_tail_loop
.Lwc_count_done:
    VZERO
    ret
.Lwc_count_zero:
    xorl    %eax, %eax
    VZERO
    ret
    .size CODEVAR_FN(codevar_wredit_wc_count), .-CODEVAR_FN(codevar_wredit_wc_count)

// =====================================================================
// find_newline_forward(raw, len, from) -> index or -1
// rdi = raw, rsi = len, rdx = from.
// =====================================================================
    .p2align 5
    .globl CODEVAR_FN(codevar_wredit_wc_find_fwd)
    .type CODEVAR_FN(codevar_wredit_wc_find_fwd), @function
CODEVAR_FN(codevar_wredit_wc_find_fwd):
    cmpq    %rsi, %rdx
    jae     .Lwc_ff_none
    movq    %rsi, %r9
    subq    %rdx, %r9
    leaq    (%rdi,%rdx,4), %r8
    SETUP
.Lwc_ff_wide:
    cmpq    $(4*VEC), %r9
    jb      .Lwc_ff_vec
    MASK_AT(0, %ecx)
    movl    %ecx, %r11d
    MASK_AT(VBYTES, %ecx)
    orl     %ecx, %r11d
    MASK_AT(2*VBYTES, %ecx)
    orl     %ecx, %r11d
    MASK_AT(3*VBYTES, %ecx)
    orl     %ecx, %r11d
    testl   %r11d, %r11d
    jnz     .Lwc_ff_wide_hit
    addq    $(4*VBYTES), %r8
    subq    $(4*VEC), %r9
    jmp     .Lwc_ff_wide
.Lwc_ff_wide_hit:
    MASK_AT(0, %ecx)
    testl   %ecx, %ecx
    jnz     .Lwc_ff_found
    addq    $VBYTES, %r8
    subq    $VEC, %r9
    MASK_AT(0, %ecx)
    testl   %ecx, %ecx
    jnz     .Lwc_ff_found
    addq    $VBYTES, %r8
    subq    $VEC, %r9
    MASK_AT(0, %ecx)
    testl   %ecx, %ecx
    jnz     .Lwc_ff_found
    addq    $VBYTES, %r8
    subq    $VEC, %r9
    MASK_AT(0, %ecx)
.Lwc_ff_found:
    bsfl    %ecx, %eax
    movq    %r8, %rdx
    subq    %rdi, %rdx
    shrq    $2, %rdx
    addq    %rdx, %rax
    VZERO
    ret
.Lwc_ff_vec:
    cmpq    $VEC, %r9
    jb      .Lwc_ff_tail
.Lwc_ff_vec_loop:
    MASK_AT(0, %ecx)
    testl   %ecx, %ecx
    jnz     .Lwc_ff_found
    addq    $VBYTES, %r8
    subq    $VEC, %r9
    cmpq    $VEC, %r9
    jae     .Lwc_ff_vec_loop
.Lwc_ff_tail:
    testq   %r9, %r9
    jz      .Lwc_ff_none
.Lwc_ff_tail_loop:
    movl    (%r8), %ecx
    andl    $0x00FFFFFF, %ecx
    cmpl    $10, %ecx
    je      .Lwc_ff_tail_found
    addq    $4, %r8
    decq    %r9
    jnz     .Lwc_ff_tail_loop
    jmp     .Lwc_ff_none
.Lwc_ff_tail_found:
    movq    %r8, %rax
    subq    %rdi, %rax
    shrq    $2, %rax
    VZERO
    ret
.Lwc_ff_none:
    movq    $-1, %rax
    VZERO
    ret
    .size CODEVAR_FN(codevar_wredit_wc_find_fwd), .-CODEVAR_FN(codevar_wredit_wc_find_fwd)

// =====================================================================
// find_newline_reverse(raw, before) -> index or -1
// rdi = raw, rsi = before (exclusive bound), rdx unused.
// =====================================================================
    .p2align 5
    .globl CODEVAR_FN(codevar_wredit_wc_find_rev)
    .type CODEVAR_FN(codevar_wredit_wc_find_rev), @function
CODEVAR_FN(codevar_wredit_wc_find_rev):
    testq   %rsi, %rsi
    jz      .Lwc_fr_none
    leaq    (%rdi,%rsi,4), %r8
    movq    %rsi, %r9
    SETUP
.Lwc_fr_loop:
    cmpq    $VEC, %r9
    jb      .Lwc_fr_tail
    subq    $VBYTES, %r8
    subq    $VEC, %r9
    MASK_AT(0, %ecx)
    testl   %ecx, %ecx
    jnz     .Lwc_fr_found
    jmp     .Lwc_fr_loop
.Lwc_fr_tail:
    testq   %r9, %r9
    jz      .Lwc_fr_none
.Lwc_fr_tail_loop:
    subq    $4, %r8
    decq    %r9
    movl    (%r8), %ecx
    andl    $0x00FFFFFF, %ecx
    cmpl    $10, %ecx
    je      .Lwc_fr_tail_found
    testq   %r9, %r9
    jnz     .Lwc_fr_tail_loop
    jmp     .Lwc_fr_none
.Lwc_fr_tail_found:
    movq    %r8, %rax
    subq    %rdi, %rax
    shrq    $2, %rax
    VZERO
    ret
.Lwc_fr_found:
    bsrl    %ecx, %eax
    movq    %r8, %rdx
    subq    %rdi, %rdx
    shrq    $2, %rdx
    addq    %rdx, %rax
    VZERO
    ret
.Lwc_fr_none:
    movq    $-1, %rax
    VZERO
    ret
    .size CODEVAR_FN(codevar_wredit_wc_find_rev), .-CODEVAR_FN(codevar_wredit_wc_find_rev)

// =====================================================================
// find_nth_newline(raw, len, n) -> index or -1
// rdi = raw, rsi = len, rdx = n (one-based, n > 0).
// =====================================================================
    .p2align 5
    .globl CODEVAR_FN(codevar_wredit_wc_find_nth)
    .type CODEVAR_FN(codevar_wredit_wc_find_nth), @function
CODEVAR_FN(codevar_wredit_wc_find_nth):
    testq   %rdx, %rdx
    jz      .Lwc_nth_none
    testq   %rsi, %rsi
    jz      .Lwc_nth_none
    movq    %rdx, %r10
    movq    %rdi, %r8
    movq    %rsi, %r9
    SETUP
    // Four vectors per iteration. Counting a whole 128-byte block at
    // once turns four loop branches into one, which matters for random
    // row jumps: they scan a bounded anchor block of a few lines.
.Lwc_nth_wide:
    cmpq    $(4*VEC), %r9
    jb      .Lwc_nth_vec
    MASK_AT(0, %ecx)
    POPCOUNT(%ecx, %eax)
    MASK_AT(VBYTES, %ecx)
    POPCOUNT(%ecx, %edx)
    addl    %edx, %eax
    MASK_AT(2*VBYTES, %ecx)
    POPCOUNT(%ecx, %edx)
    addl    %edx, %eax
    MASK_AT(3*VBYTES, %ecx)
    POPCOUNT(%ecx, %edx)
    addl    %edx, %eax
    cmpq    %rax, %r10
    jbe     .Lwc_nth_vec
    subq    %rax, %r10
    addq    $(4*VBYTES), %r8
    subq    $(4*VEC), %r9
    jmp     .Lwc_nth_wide
.Lwc_nth_vec:
    cmpq    $VEC, %r9
    jb      .Lwc_nth_tail
.Lwc_nth_vec_loop:
    MASK_AT(0, %ecx)
    testl   %ecx, %ecx
    jz      .Lwc_nth_skip
    POPCOUNT(%ecx, %eax)
    cmpq    %rax, %r10
    jbe     .Lwc_nth_hit
    subq    %rax, %r10
.Lwc_nth_skip:
    addq    $VBYTES, %r8
    subq    $VEC, %r9
    cmpq    $VEC, %r9
    jae     .Lwc_nth_vec_loop
.Lwc_nth_tail:
    testq   %r9, %r9
    jz      .Lwc_nth_none
.Lwc_nth_tail_loop:
    movl    (%r8), %ecx
    andl    $0x00FFFFFF, %ecx
    cmpl    $10, %ecx
    jne     .Lwc_nth_tail_next
    decq    %r10
    jz      .Lwc_nth_tail_found
.Lwc_nth_tail_next:
    addq    $4, %r8
    decq    %r9
    jnz     .Lwc_nth_tail_loop
    jmp     .Lwc_nth_none
.Lwc_nth_tail_found:
    movq    %r8, %rax
    subq    %rdi, %rax
    shrq    $2, %rax
    VZERO
    ret
.Lwc_nth_hit:
    decq    %r10
.Lwc_nth_bit:
    bsfl    %ecx, %eax
    testq   %r10, %r10
    jz      .Lwc_nth_bit_found
    leal    -1(%rcx), %edx
    andl    %edx, %ecx
    decq    %r10
    jmp     .Lwc_nth_bit
.Lwc_nth_bit_found:
    movq    %r8, %rdx
    subq    %rdi, %rdx
    shrq    $2, %rdx
    addq    %rdx, %rax
    VZERO
    ret
.Lwc_nth_none:
    movq    $-1, %rax
    VZERO
    ret
    .size CODEVAR_FN(codevar_wredit_wc_find_nth), .-CODEVAR_FN(codevar_wredit_wc_find_nth)

.section .note.GNU-stack,"",@progbits
