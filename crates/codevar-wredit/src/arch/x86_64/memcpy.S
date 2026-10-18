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

// Multi-variant x86_64 memcpy (non-overlapping).
//
// Assembled once per instruction-set level exactly like memmove.S; see
// that file for the CODEVAR_SUFFIX / CODEVAR_HAVE_* contract and the
// exported symbol `codevar_wredit_memcpy<SUFFIX>`.
//
// ABI: rdi = dst, rsi = src, rdx = len; returns rax = dst.

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
    .globl CODEVAR_FN(codevar_wredit_memcpy)
    .type CODEVAR_FN(codevar_wredit_memcpy), @function

CODEVAR_FN(codevar_wredit_memcpy):
    movq    %rdi, %rax              // return value
    testq   %rdx, %rdx
    jz      .Lcp_done
    cmpq    %rsi, %rdi
    je      .Lcp_done

    movq    %rdx, %r10              // r10 = remaining
    cmpq    $VEC, %r10
    jb      .Lcp_tail

.Lcp_loop:
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
    jae     .Lcp_loop

.Lcp_tail:
    testq   $32, %r10
    jz      .Lcp_tail_16
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
.Lcp_tail_16:
    testq   $16, %r10
    jz      .Lcp_tail_8
    movq    0(%rsi), %r11
    movq    8(%rsi), %rcx
    movq    %r11, 0(%rdi)
    movq    %rcx, 8(%rdi)
    addq    $16, %rsi
    addq    $16, %rdi
.Lcp_tail_8:
    testq   $8, %r10
    jz      .Lcp_tail_4
    movq    0(%rsi), %r11
    movq    %r11, 0(%rdi)
    addq    $8, %rsi
    addq    $8, %rdi
.Lcp_tail_4:
    testq   $4, %r10
    jz      .Lcp_tail_2
    movl    0(%rsi), %r11d
    movl    %r11d, 0(%rdi)
    addq    $4, %rsi
    addq    $4, %rdi
.Lcp_tail_2:
    testq   $2, %r10
    jz      .Lcp_tail_1
    movw    0(%rsi), %r11w
    movw    %r11w, 0(%rdi)
    addq    $2, %rsi
    addq    $2, %rdi
.Lcp_tail_1:
    testq   $1, %r10
    jz      .Lcp_done
    movb    0(%rsi), %r11b
    movb    %r11b, 0(%rdi)

.Lcp_done:
#if defined(CODEVAR_HAVE_AVX2) || defined(CODEVAR_HAVE_AVX512)
    vzeroupper
#endif
    ret

    .size CODEVAR_FN(codevar_wredit_memcpy), .-CODEVAR_FN(codevar_wredit_memcpy)
