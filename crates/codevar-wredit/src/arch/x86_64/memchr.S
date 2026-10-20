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
// `haystack[..len]`. The hot loop scans four vectors (4*VEC bytes) per
// iteration and ORs their comparison masks, so a non-match retires with
// a single branch; only the rare hit re-walks the four chunks to locate
// the exact byte.
//
// Instruction selection (preprocessor):
//   SSE2   - pcmpeqb + pmovmskb, 16 bytes per vector
//   AVX2   - vpcmpeqb + two pmovmskb (low/high 128-bit halves)
//   AVX512 - vpcmpb into a mask register + kmovq
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
#define VZERO vzeroupper
// One comparison mask (64 bits) for the chunk at `off(%r10)`, in rax.
#define MASK_AT(off) \
    vmovdqu64 off(%r10), %zmm0 ; \
    vpcmpb $0, %zmm0, %zmm1, %k1 ; \
    kmovq %k1, %rax
#elif defined(CODEVAR_HAVE_AVX2)
#define VEC 32
#define VZERO vzeroupper
// One comparison mask (32 bits, as two 16-bit halves) in rax.
#define MASK_AT(off) \
    vmovdqu off(%r10), %ymm0 ; \
    vpcmpeqb %ymm1, %ymm0, %ymm0 ; \
    vextracti128 $1, %ymm0, %xmm2 ; \
    vpmovmskb %xmm0, %eax ; \
    vpmovmskb %xmm2, %r11d ; \
    shlq $16, %r11 ; \
    orq %r11, %rax
#else
#define VEC 16
#define VZERO
// One comparison mask (16 bits) in rax.
#define MASK_AT(off) \
    movdqu off(%r10), %xmm0 ; \
    pcmpeqb %xmm1, %xmm0 ; \
    pmovmskb %xmm0, %eax
#endif

    .text
    .p2align 5
    .globl CODEVAR_FN(codevar_wredit_memchr)
    .type CODEVAR_FN(codevar_wredit_memchr), @function

CODEVAR_FN(codevar_wredit_memchr):
    movq    %rdi, %r10              // r10 = scan pointer
    testq   %rdx, %rdx
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

// =====================================================================
// Wide loop: four vectors (4*VEC bytes) per iteration.
// =====================================================================
.Lmc_wide:
    cmpq    $(4*VEC), %rdx
    jb      .Lmc_vec
    MASK_AT(0)
    movq    %rax, %r8
    MASK_AT(VEC)
    orq     %rax, %r8
    MASK_AT(2*VEC)
    orq     %rax, %r8
    MASK_AT(3*VEC)
    orq     %rax, %r8
    testq   %r8, %r8
    jnz     .Lmc_wide_hit
    addq    $(4*VEC), %r10
    subq    $(4*VEC), %rdx
    jmp     .Lmc_wide

// =====================================================================
// Single-vector loop: 0..4*VEC-1 bytes left.
// =====================================================================
.Lmc_vec:
    cmpq    $VEC, %rdx
    jb      .Lmc_tail
.Lmc_vec_loop:
    MASK_AT(0)
    testq   %rax, %rax
    jnz     .Lmc_hit
    addq    $VEC, %r10
    subq    $VEC, %rdx
    cmpq    $VEC, %rdx
    jae     .Lmc_vec_loop

// Scalar tail: 0..VEC-1 bytes, always in bounds.
.Lmc_tail:
    testq   %rdx, %rdx
    jz      .Lmc_none
.Lmc_tail_loop:
    cmpb    %sil, (%r10)
    je      .Lmc_hit_byte
    incq    %r10
    decq    %rdx
    jnz     .Lmc_tail_loop
    jmp     .Lmc_none

// =====================================================================
// Slow paths.
// =====================================================================
.Lmc_wide_hit:
    MASK_AT(0)
    testq   %rax, %rax
    jnz     .Lmc_hit
    addq    $VEC, %r10
    MASK_AT(0)
    testq   %rax, %rax
    jnz     .Lmc_hit
    addq    $VEC, %r10
    MASK_AT(0)
    testq   %rax, %rax
    jnz     .Lmc_hit
    addq    $VEC, %r10
    MASK_AT(0)
    // The OR was non-zero, so the fourth chunk holds the hit.

.Lmc_hit:
    bsfq    %rax, %rax              // index of the first set bit
    addq    %r10, %rax
    VZERO
    ret

.Lmc_hit_byte:
    movq    %r10, %rax
    VZERO
    ret

.Lmc_none:
    xorl    %eax, %eax
    VZERO
    ret

    .size CODEVAR_FN(codevar_wredit_memchr), .-CODEVAR_FN(codevar_wredit_memchr)

.section .note.GNU-stack,"",@progbits
