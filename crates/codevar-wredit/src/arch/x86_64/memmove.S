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

// Multi-variant x86_64 memmove.
//
// The forward (dst < src) direction reuses the same size ladder as
// memcpy: overlapping integer/half-vector moves, an eight-register
// middle band, a four-vector unrolled loop and `rep movsb` for the
// large blocks. The backward (dst > src) direction walks from the high
// end, copying four vectors per iteration so the destination never
// overtakes unread source.
//
// This single source is assembled once per instruction-set level and
// the resulting objects are linked side by side; the Rust side picks
// the widest one the running CPU actually supports (runtime dispatch
// through codevar_base::cpuid). The preprocessor selects the vector
// width and the instruction mnemonics:
//
//   -DCODEVAR_SUFFIX=_sse2    baseline SSE2, always present on x86_64
//   -DCODEVAR_SUFFIX=_avx2    with -mavx2
//   -DCODEVAR_SUFFIX=_avx512  with -mavx512f -mavx512bw -mavx512vl
//
// The exported symbol is `codevar_wredit_memmove<SUFFIX>`.
//
// ABI: rdi = dst, rsi = src, rdx = len; returns rax = dst.
// Overlap: copies forward when dst < src and backward when dst > src.

#ifndef CODEVAR_SUFFIX
#define CODEVAR_SUFFIX
#endif

#define CODEVAR_CAT2(a, b) a##b
#define CODEVAR_CAT(a, b) CODEVAR_CAT2(a, b)
#define CODEVAR_FN(name) CODEVAR_CAT(name, CODEVAR_SUFFIX)

#if defined(CODEVAR_HAVE_AVX512)
#define VEC 64
#define VREG(n) %zmm##n
#define XLOAD(off, base, reg)  vmovdqu64 off base, reg
#define XSTORE(reg, off, base) vmovdqu64 reg, off base
#define VZERO vzeroupper
#elif defined(CODEVAR_HAVE_AVX2)
#define VEC 32
#define VREG(n) %ymm##n
#define XLOAD(off, base, reg)  vmovdqu off base, reg
#define XSTORE(reg, off, base) vmovdqu reg, off base
#define VZERO vzeroupper
#else
#define VEC 16
#define VREG(n) %xmm##n
#define XLOAD(off, base, reg)  movdqu off base, reg
#define XSTORE(reg, off, base) movdqu reg, off base
#define VZERO
#endif

#if defined(CODEVAR_HAVE_AVX2) || defined(CODEVAR_HAVE_AVX512)
#define LCH 32
#define LREG(n) %ymm##n
#define LLOAD(off, base, reg)  vmovdqu off base, reg
#define LSTORE(reg, off, base) vmovdqu reg, off base
#define M16LOAD(off, base, reg)  vmovdqu off base, reg
#define M16STORE(reg, off, base) vmovdqu reg, off base
#else
#define LCH 16
#define LREG(n) %xmm##n
#define LLOAD(off, base, reg)  movdqu off base, reg
#define LSTORE(reg, off, base) movdqu reg, off base
#define M16LOAD(off, base, reg)  movdqu off base, reg
#define M16STORE(reg, off, base) movdqu reg, off base
#endif

#define REP_MOVSB_THRESHOLD 1024

    .text
    .p2align 5
    .globl CODEVAR_FN(codevar_wredit_memmove)
    .type CODEVAR_FN(codevar_wredit_memmove), @function

CODEVAR_FN(codevar_wredit_memmove):
    movq    %rdi, %rax              // return value
    testq   %rdx, %rdx
    jz      .Lmm_zero
    cmpq    %rsi, %rdi
    je      .Lmm_zero               // same pointer: nothing to do
    jb      .Lmm_rest               // dst < src -> forward

// =====================================================================
// Backward copy: dst > src, walk from the high end toward the low end.
// =====================================================================
    addq    %rdx, %rsi              // src + len
    addq    %rdx, %rdi              // dst + len
    movq    %rdx, %r10              // r10 = remaining
    cmpq    $(4*VEC), %rdx
    jb      .Lmm_bwd_tail

.Lmm_bwd_loop:
    subq    $(4*VEC), %rsi
    subq    $(4*VEC), %rdi
    XLOAD(0, (%rsi), VREG(0))
    XLOAD(VEC, (%rsi), VREG(1))
    XLOAD(2*VEC, (%rsi), VREG(2))
    XLOAD(3*VEC, (%rsi), VREG(3))
    XSTORE(VREG(0), 0, (%rdi))
    XSTORE(VREG(1), VEC, (%rdi))
    XSTORE(VREG(2), 2*VEC, (%rdi))
    XSTORE(VREG(3), 3*VEC, (%rdi))
    subq    $(4*VEC), %r10
    cmpq    $(4*VEC), %r10
    jae     .Lmm_bwd_loop

.Lmm_bwd_tail:
    // Each piece loads all of its words into registers before storing,
    // so writing the high end cannot clobber unread source. The bulk
    // loop can leave up to 4*VEC-1 bytes, so drain 32 at a time first.
    cmpq    $32, %r10
    jb      .Lmm_bwd_16
.Lmm_bwd_32loop:
    subq    $32, %rsi
    subq    $32, %rdi
    movq    0(%rsi), %r11
    movq    8(%rsi), %rcx
    movq    16(%rsi), %r8
    movq    24(%rsi), %r9
    movq    %r11, 0(%rdi)
    movq    %rcx, 8(%rdi)
    movq    %r8, 16(%rdi)
    movq    %r9, 24(%rdi)
    subq    $32, %r10
    cmpq    $32, %r10
    jae     .Lmm_bwd_32loop
.Lmm_bwd_16:
    testq   $16, %r10
    jz      .Lmm_bwd_8
    subq    $16, %rsi
    subq    $16, %rdi
    movq    0(%rsi), %r11
    movq    8(%rsi), %rcx
    movq    %r11, 0(%rdi)
    movq    %rcx, 8(%rdi)
.Lmm_bwd_8:
    testq   $8, %r10
    jz      .Lmm_bwd_4
    subq    $8, %rsi
    subq    $8, %rdi
    movq    0(%rsi), %r11
    movq    %r11, 0(%rdi)
.Lmm_bwd_4:
    testq   $4, %r10
    jz      .Lmm_bwd_2
    subq    $4, %rsi
    subq    $4, %rdi
    movl    0(%rsi), %r11d
    movl    %r11d, 0(%rdi)
.Lmm_bwd_2:
    testq   $2, %r10
    jz      .Lmm_bwd_1
    subq    $2, %rsi
    subq    $2, %rdi
    movw    0(%rsi), %r11w
    movw    %r11w, 0(%rdi)
.Lmm_bwd_1:
    testq   $1, %r10
    jz      .Lmm_epi
    decq    %rsi
    decq    %rdi
    movb    0(%rsi), %r11b
    movb    %r11b, 0(%rdi)
    jmp     .Lmm_epi

// =====================================================================
// Forward copy: dst < src. Same ladder as memcpy.
// =====================================================================
.Lmm_rest:
    cmpq    $16, %rdx
    jb      .Lmm_short
    cmpq    $32, %rdx
    jbe     .Lmm_16_32
    cmpq    $(8*LCH), %rdx
    ja      .Lmm_big

.Lmm_ladder:
    LLOAD(0, (%rsi), LREG(0))
    LLOAD(-LCH, (%rsi,%rdx), LREG(7))
    cmpq    $(2*LCH), %rdx
    jbe     .Lmm_st2
    LLOAD(LCH, (%rsi), LREG(1))
    LLOAD(-2*LCH, (%rsi,%rdx), LREG(6))
    cmpq    $(4*LCH), %rdx
    jbe     .Lmm_st4
    LLOAD(2*LCH, (%rsi), LREG(2))
    LLOAD(-3*LCH, (%rsi,%rdx), LREG(5))
    cmpq    $(6*LCH), %rdx
    jbe     .Lmm_st6
    LLOAD(3*LCH, (%rsi), LREG(3))
    LLOAD(-4*LCH, (%rsi,%rdx), LREG(4))
    LSTORE(LREG(3), 3*LCH, (%rdi))
    LSTORE(LREG(4), -4*LCH, (%rdi,%rdx))
.Lmm_st6:
    LSTORE(LREG(2), 2*LCH, (%rdi))
    LSTORE(LREG(5), -3*LCH, (%rdi,%rdx))
.Lmm_st4:
    LSTORE(LREG(1), LCH, (%rdi))
    LSTORE(LREG(6), -2*LCH, (%rdi,%rdx))
.Lmm_st2:
    LSTORE(LREG(0), 0, (%rdi))
    LSTORE(LREG(7), -LCH, (%rdi,%rdx))
    jmp     .Lmm_epi

.Lmm_16_32:
    M16LOAD(0, (%rsi), %xmm0)
    M16LOAD(-16, (%rsi,%rdx), %xmm1)
    M16STORE(%xmm0, 0, (%rdi))
    M16STORE(%xmm1, -16, (%rdi,%rdx))
    jmp     .Lmm_epi

.Lmm_short:
    cmpq    $8, %rdx
    jae     .Lmm_s8
    cmpq    $4, %rdx
    jae     .Lmm_s4
    cmpq    $2, %rdx
    jae     .Lmm_s2
    movzbl  (%rsi), %r8d
    movb    %r8b, (%rdi)
    jmp     .Lmm_epi
.Lmm_s2:
    movzwl  (%rsi), %r8d
    movzwl  -2(%rsi,%rdx), %r9d
    movw    %r8w, (%rdi)
    movw    %r9w, -2(%rdi,%rdx)
    jmp     .Lmm_epi
.Lmm_s4:
    movl    (%rsi), %r8d
    movl    -4(%rsi,%rdx), %r9d
    movl    %r8d, (%rdi)
    movl    %r9d, -4(%rdi,%rdx)
    jmp     .Lmm_epi
.Lmm_s8:
    movq    (%rsi), %r8
    movq    -8(%rsi,%rdx), %r9
    movq    %r8, (%rdi)
    movq    %r9, -8(%rdi,%rdx)
    jmp     .Lmm_epi

.Lmm_big:
    cmpq    $REP_MOVSB_THRESHOLD, %rdx
    jae     .Lmm_rep

    movq    %rdx, %r10
.Lmm_loop:
    XLOAD(0, (%rsi), VREG(0))
    XLOAD(VEC, (%rsi), VREG(1))
    XLOAD(2*VEC, (%rsi), VREG(2))
    XLOAD(3*VEC, (%rsi), VREG(3))
    XSTORE(VREG(0), 0, (%rdi))
    XSTORE(VREG(1), VEC, (%rdi))
    XSTORE(VREG(2), 2*VEC, (%rdi))
    XSTORE(VREG(3), 3*VEC, (%rdi))
    addq    $(4*VEC), %rsi
    addq    $(4*VEC), %rdi
    subq    $(4*VEC), %r10
    cmpq    $(4*VEC), %r10
    jae     .Lmm_loop

    movq    %r10, %rdx
    testq   %rdx, %rdx
    jz      .Lmm_epi
    jmp     .Lmm_rest

.Lmm_rep:
    movq    %rdx, %rcx
    rep     movsb

.Lmm_epi:
    VZERO
    ret

.Lmm_zero:
    ret

    .size CODEVAR_FN(codevar_wredit_memmove), .-CODEVAR_FN(codevar_wredit_memmove)

.section .note.GNU-stack,"",@progbits
