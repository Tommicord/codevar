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
// The size dispatch follows the ladder popularised by folly's
// `folly/memcpy.S`: tiny copies use overlapping integer/16-byte moves,
// the 32..8*LCH middle band is served by an eight-register overlapping
// ladder, and everything wider falls into either a four-vector unrolled
// loop or `rep movsb` (ERMS), which streams large blocks at memory
// bandwidth on every modern x86 core.
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

// Widest vector used by the bulk loop (and zmm for AVX-512).
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

// Ladder granularity: 32 bytes whenever the ISA has ymm, 16 bytes for
// the plain SSE2 baseline. LCH never exceeds VEC, so every ladder load
// is in bounds for its guard range.
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

// `rep movsb` pays off once the ERMS engine can amortise its startup;
// below this the unrolled vector loop wins.
#define REP_MOVSB_THRESHOLD 1024

    .text
    .p2align 5
    .globl CODEVAR_FN(codevar_wredit_memcpy)
    .type CODEVAR_FN(codevar_wredit_memcpy), @function

CODEVAR_FN(codevar_wredit_memcpy):
    movq    %rdi, %rax              // return value
    testq   %rdx, %rdx
    jz      .Lcp_zero

// Re-entry point after the bulk loop: `rdx` is always <= 8*LCH here.
.Lcp_rest:
    cmpq    $16, %rdx
    jb      .Lcp_short
    cmpq    $32, %rdx
    jbe     .Lcp_16_32
    cmpq    $(8*LCH), %rdx
    ja      .Lcp_big

// =====================================================================
// 32 < len <= 8*LCH: overlapping ladder, four registers per end.
// Every load stays inside [src, src+len) because the guard below is
// taken as soon as the copied span would cover the whole range.
// =====================================================================
.Lcp_ladder:
    LLOAD(0, (%rsi), LREG(0))
    LLOAD(-LCH, (%rsi,%rdx), LREG(7))
    cmpq    $(2*LCH), %rdx
    jbe     .Lcp_st2
    LLOAD(LCH, (%rsi), LREG(1))
    LLOAD(-2*LCH, (%rsi,%rdx), LREG(6))
    cmpq    $(4*LCH), %rdx
    jbe     .Lcp_st4
    LLOAD(2*LCH, (%rsi), LREG(2))
    LLOAD(-3*LCH, (%rsi,%rdx), LREG(5))
    cmpq    $(6*LCH), %rdx
    jbe     .Lcp_st6
    LLOAD(3*LCH, (%rsi), LREG(3))
    LLOAD(-4*LCH, (%rsi,%rdx), LREG(4))
    // Stores fall through from the highest offsets down, matching
    // folly's ordering.
    LSTORE(LREG(3), 3*LCH, (%rdi))
    LSTORE(LREG(4), -4*LCH, (%rdi,%rdx))
.Lcp_st6:
    LSTORE(LREG(2), 2*LCH, (%rdi))
    LSTORE(LREG(5), -3*LCH, (%rdi,%rdx))
.Lcp_st4:
    LSTORE(LREG(1), LCH, (%rdi))
    LSTORE(LREG(6), -2*LCH, (%rdi,%rdx))
.Lcp_st2:
    LSTORE(LREG(0), 0, (%rdi))
    LSTORE(LREG(7), -LCH, (%rdi,%rdx))
    jmp     .Lcp_epi

// =====================================================================
// 16..32 bytes: two overlapping 16-byte halves.
// =====================================================================
.Lcp_16_32:
    M16LOAD(0, (%rsi), %xmm0)
    M16LOAD(-16, (%rsi,%rdx), %xmm1)
    M16STORE(%xmm0, 0, (%rdi))
    M16STORE(%xmm1, -16, (%rdi,%rdx))
    jmp     .Lcp_epi

// =====================================================================
// 1..15 bytes: overlapping byte/word/dword/qword.
// =====================================================================
.Lcp_short:
    cmpq    $8, %rdx
    jae     .Lcp_s8
    cmpq    $4, %rdx
    jae     .Lcp_s4
    cmpq    $2, %rdx
    jae     .Lcp_s2
    movzbl  (%rsi), %r8d
    movb    %r8b, (%rdi)
    jmp     .Lcp_epi
.Lcp_s2:
    movzwl  (%rsi), %r8d
    movzwl  -2(%rsi,%rdx), %r9d
    movw    %r8w, (%rdi)
    movw    %r9w, -2(%rdi,%rdx)
    jmp     .Lcp_epi
.Lcp_s4:
    movl    (%rsi), %r8d
    movl    -4(%rsi,%rdx), %r9d
    movl    %r8d, (%rdi)
    movl    %r9d, -4(%rdi,%rdx)
    jmp     .Lcp_epi
.Lcp_s8:
    movq    (%rsi), %r8
    movq    -8(%rsi,%rdx), %r9
    movq    %r8, (%rdi)
    movq    %r9, -8(%rdi,%rdx)
    jmp     .Lcp_epi

// =====================================================================
// len > 8*LCH: unrolled four-vector loop, then `rep movsb` once the
// block is large enough for the ERMS engine.
// =====================================================================
.Lcp_big:
    cmpq    $REP_MOVSB_THRESHOLD, %rdx
    jae     .Lcp_rep

    movq    %rdx, %r10
.Lcp_loop:
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
    jae     .Lcp_loop

    movq    %r10, %rdx
    testq   %rdx, %rdx
    jz      .Lcp_epi
    jmp     .Lcp_rest

.Lcp_rep:
    movq    %rdx, %rcx
    rep     movsb
    ret                             // rax already holds the original dst

.Lcp_epi:
    VZERO
    ret

.Lcp_zero:
    ret

    .size CODEVAR_FN(codevar_wredit_memcpy), .-CODEVAR_FN(codevar_wredit_memcpy)

.section .note.GNU-stack,"",@progbits
