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

// i686 memchr: first occurrence of the low byte of `needle`.
//   -DCODEVAR_HAVE_SSE2 : 16-byte pcmpeqb + pmovmskb; else scalar bytes.
// ABI (cdecl): [esp+4] = haystack, [esp+8] = needle, [esp+12] = len;
// returns pointer in eax or NULL.

    .text
    .p2align 4
    .globl codevar_wredit_memchr
    .type codevar_wredit_memchr, @function

codevar_wredit_memchr:
    pushl   %esi
    pushl   %ebx

    movl    12(%esp), %esi          // haystack
    movl    16(%esp), %ebx          // needle
    movl    20(%esp), %ecx          // len
    testl   %ecx, %ecx
    jz      .Lmc_none

#ifdef CODEVAR_HAVE_SSE2
    // Broadcast the needle byte to all 16 lanes of xmm1.
    movd    %ebx, %xmm1
    punpcklbw %xmm1, %xmm1
    punpcklwd %xmm1, %xmm1
    pshufd  $0, %xmm1, %xmm1

.Lmc_loop:
    cmpl    $16, %ecx
    jb      .Lmc_byte
    movdqu  (%esi), %xmm0
    pcmpeqb %xmm1, %xmm0
    pmovmskb %xmm0, %eax
    testl   %eax, %eax
    jnz     .Lmc_found
    addl    $16, %esi
    subl    $16, %ecx
    jmp     .Lmc_loop

.Lmc_found:
    bsfl    %eax, %eax
    addl    %esi, %eax
    jmp     .Lmc_ret
#endif

.Lmc_byte:
    testl   %ecx, %ecx
    jz      .Lmc_none
    movb    (%esi), %al
    cmpb    %bl, %al
    je      .Lmc_byte_hit
    incl    %esi
    decl    %ecx
    jmp     .Lmc_byte
.Lmc_byte_hit:
    movl    %esi, %eax
    jmp     .Lmc_ret

.Lmc_none:
    xorl    %eax, %eax
.Lmc_ret:
    popl    %ebx
    popl    %esi
    ret

    .size codevar_wredit_memchr, .-codevar_wredit_memchr
