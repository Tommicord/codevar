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
#elif defined(CODEVAR_HAVE_AVX2)
#define VEC 32
#else
#define VEC 16
#endif

    .text
    .p2align 5
    .globl CODEVAR_FN(codevar_wredit_memmove)
    .type CODEVAR_FN(codevar_wredit_memmove), @function

CODEVAR_FN(codevar_wredit_memmove):
    movq    %rdi, %rax              // return value
    testq   %rdx, %rdx
    jz      .Lmm_done
    cmpq    %rsi, %rdi
    je      .Lmm_done               // same pointer: nothing to do
    jb      .Lmm_forward

// =====================================================================
// Backward copy: dst > src, walk from the high end toward the low end.
// =====================================================================
.Lmm_backward:
    leaq    (%rsi, %rdx), %rsi      // src + len
    leaq    (%rdi, %rdx), %rdi      // dst + len
    movq    %rdx, %r10              // r10 = remaining
    cmpq    $VEC, %r10
    jb      .Lmm_bwd_tail

.Lmm_bwd_loop:
    subq    $VEC, %rsi
    subq    $VEC, %rdi
#if defined(CODEVAR_HAVE_AVX512)
    vmovdqu64 (%rsi), %zmm0
    vmovdqu64 %zmm0, (%rdi)
#elif defined(CODEVAR_HAVE_AVX2)
    vmovdqu (%rsi), %ymm0
    vmovdqu %ymm0, (%rdi)
#else
    movdqu  (%rsi), %xmm0
    movdqu  %xmm0, (%rdi)
#endif
    prefetchnta -256(%rsi)
    subq    $VEC, %r10
    cmpq    $VEC, %r10
    jae     .Lmm_bwd_loop

.Lmm_bwd_tail:
    // rsi/rdi point to the end of the remaining (< VEC) region.
    // Each piece loads all of its words into registers before storing,
    // so a forward store within the piece cannot clobber unread source.
    testq   $32, %r10
    jz      .Lmm_bwd_16
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
    jz      .Lmm_done
    decq    %rsi
    decq    %rdi
    movb    0(%rsi), %r11b
    movb    %r11b, 0(%rdi)
    jmp     .Lmm_done

// =====================================================================
// Forward copy: dst < src, walk from the low end toward the high end.
// =====================================================================
.Lmm_forward:
    movq    %rdx, %r10              // r10 = remaining
    cmpq    $VEC, %r10
    jb      .Lmm_fwd_tail

.Lmm_fwd_loop:
#if defined(CODEVAR_HAVE_AVX512)
    vmovdqu64 (%rsi), %zmm0
    vmovdqu64 %zmm0, (%rdi)
#elif defined(CODEVAR_HAVE_AVX2)
    vmovdqu (%rsi), %ymm0
    vmovdqu %ymm0, (%rdi)
#else
    movdqu  (%rsi), %xmm0
    movdqu  %xmm0, (%rdi)
#endif
    prefetchnta 256(%rsi)
    addq    $VEC, %rsi
    addq    $VEC, %rdi
    subq    $VEC, %r10
    cmpq    $VEC, %r10
    jae     .Lmm_fwd_loop

.Lmm_fwd_tail:
    testq   $32, %r10
    jz      .Lmm_fwd_16
    movq    0(%rsi), %r11
    movq    8(%rsi), %rcx
    movq    16(%rsi), %r8
    movq    24(%rsi), %r9
    movq    %r11, 0(%rdi)
    movq    %rcx, 8(%rdi)
    movq    %r8, 16(%rdi)
    movq    %r9, 24(%rdi)
    addq    $32, %rsi
    addq    $32, %rdi
.Lmm_fwd_16:
    testq   $16, %r10
    jz      .Lmm_fwd_8
    movq    0(%rsi), %r11
    movq    8(%rsi), %rcx
    movq    %r11, 0(%rdi)
    movq    %rcx, 8(%rdi)
    addq    $16, %rsi
    addq    $16, %rdi
.Lmm_fwd_8:
    testq   $8, %r10
    jz      .Lmm_fwd_4
    movq    0(%rsi), %r11
    movq    %r11, 0(%rdi)
    addq    $8, %rsi
    addq    $8, %rdi
.Lmm_fwd_4:
    testq   $4, %r10
    jz      .Lmm_fwd_2
    movl    0(%rsi), %r11d
    movl    %r11d, 0(%rdi)
    addq    $4, %rsi
    addq    $4, %rdi
.Lmm_fwd_2:
    testq   $2, %r10
    jz      .Lmm_fwd_1
    movw    0(%rsi), %r11w
    movw    %r11w, 0(%rdi)
    addq    $2, %rsi
    addq    $2, %rdi
.Lmm_fwd_1:
    testq   $1, %r10
    jz      .Lmm_done
    movb    0(%rsi), %r11b
    movb    %r11b, 0(%rdi)

.Lmm_done:
#if defined(CODEVAR_HAVE_AVX2) || defined(CODEVAR_HAVE_AVX512)
    vzeroupper
#endif
    ret

    .size CODEVAR_FN(codevar_wredit_memmove), .-CODEVAR_FN(codevar_wredit_memmove)
