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

// Multi-variant x86_64 memchr.
//
// Finds the first occurrence of the low byte of `needle` in
// `haystack[..len]`. Assembled once per instruction-set level; the
// exported symbol is `codevar_wredit_memchr<SUFFIX>`.
//
// Instruction selection (preprocessor):
//   SSE2   - pcmpeqb + pmovmskb, 16 bytes per iteration
//   AVX2   - vpcmpeqb + two pmovmskb (low/high 128-bit halves)
//   AVX512 - vpcmpb into a mask register + kmovq, 64 bytes per iteration
//
// ABI: rdi = haystack, esi = needle, rdx = len;
//      returns rax = pointer to first match, or NULL.

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
    .globl CODEVAR_FN(codevar_wredit_memchr)
    .type CODEVAR_FN(codevar_wredit_memchr), @function

CODEVAR_FN(codevar_wredit_memchr):
    movq    %rdi, %r10              // r10 = scan pointer
    movq    %rdx, %r8               // r8 = remaining
    testq   %r8, %r8
    jz      .Lmc_none

// Broadcast the needle byte to every lane of one vector register.
#if defined(CODEVAR_HAVE_AVX512)
    vpbroadcastb %esi, %zmm1
#elif defined(CODEVAR_HAVE_AVX2)
    // The GPR-source `vpbroadcastb` is AVX-512-only; AVX2 needs the
    // xmm-source (VEX.256.66.0F38.W0 78) form.
    vmovd   %esi, %xmm1
    vpbroadcastb %xmm1, %ymm1
#else
    movd    %esi, %xmm1
    punpcklbw %xmm1, %xmm1
    punpcklwd %xmm1, %xmm1
    pshufd  $0, %xmm1, %xmm1
#endif

    cmpq    $VEC, %r8
    jb      .Lmc_scan

.Lmc_loop:
    prefetchnta 256(%r10)
#if defined(CODEVAR_HAVE_AVX512)
    vmovdqu64 (%r10), %zmm0
    vpcmpb  $0, %zmm0, %zmm1, %k1
    kmovq   %k1, %rax
    testq   %rax, %rax
    jnz     .Lmc_found
#elif defined(CODEVAR_HAVE_AVX2)
    vmovdqu (%r10), %ymm0
    vpcmpeqb %ymm1, %ymm0, %ymm0
    vextracti128 $1, %ymm0, %xmm2
    vpmovmskb %xmm0, %eax
    vpmovmskb %xmm2, %edi
    testl   %eax, %eax
    jnz     .Lmc_found
    testl   %edi, %edi
    jnz     .Lmc_found_hi
#else
    movdqu  (%r10), %xmm0
    pcmpeqb %xmm1, %xmm0
    pmovmskb %xmm0, %eax
    testl   %eax, %eax
    jnz     .Lmc_found
#endif
    addq    $VEC, %r10
    subq    $VEC, %r8
    cmpq    $VEC, %r8
    jae     .Lmc_loop

// Scalar tail: 0..VEC-1 bytes, always in bounds.
.Lmc_scan:
    testq   %r8, %r8
    jz      .Lmc_none
.Lmc_scan_loop:
    cmpb    %sil, (%r10)
    je      .Lmc_found_byte
    incq    %r10
    decq    %r8
    jnz     .Lmc_scan_loop
    jmp     .Lmc_none

#if defined(CODEVAR_HAVE_AVX2)
.Lmc_found_hi:
    // High 128-bit half matched: bit `i` means byte `16 + i`.
    bsfq    %rdi, %rax
    addq    $16, %rax
    addq    %r10, %rax
    vzeroupper
    ret
#endif

.Lmc_found:
    // `rax` holds the match bitmask for the current VEC-byte chunk (the
    // 64-bit form is required for the AVX-512 `vpcmpb` mask).
    bsfq    %rax, %rax
    addq    %r10, %rax
#if defined(CODEVAR_HAVE_AVX2) || defined(CODEVAR_HAVE_AVX512)
    vzeroupper
#endif
    ret

.Lmc_found_byte:
    movq    %r10, %rax
#if defined(CODEVAR_HAVE_AVX2) || defined(CODEVAR_HAVE_AVX512)
    vzeroupper
#endif
    ret

.Lmc_none:
    xorl    %eax, %eax
#if defined(CODEVAR_HAVE_AVX2) || defined(CODEVAR_HAVE_AVX512)
    vzeroupper
#endif
    ret

    .size CODEVAR_FN(codevar_wredit_memchr), .-CODEVAR_FN(codevar_wredit_memchr)
