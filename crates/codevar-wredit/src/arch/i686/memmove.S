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

// i686 memmove.
//
// Two flavours selected by the preprocessor:
//   -DCODEVAR_HAVE_SSE2 : 16-byte movdqu + prefetchnta main loop
//   otherwise           : 4-byte dword main loop (baseline i686)
// Both share a byte tail and handle overlap: forward when dst < src,
// backward when dst >= src.
//
// ABI (cdecl): [esp+4] = dst, [esp+8] = src, [esp+12] = len;
// returns dst in eax.

    .text
    .p2align 4
    .globl codevar_wredit_memmove
    .type codevar_wredit_memmove, @function

codevar_wredit_memmove:
    pushl   %esi
    pushl   %edi
    pushl   %ebx
    pushl   %ebp

    movl    20(%esp), %edi          // edi = dst
    movl    24(%esp), %esi          // esi = src
    movl    28(%esp), %ecx          // ecx = len

    movl    %edi, %eax              // return value = dst
    testl   %ecx, %ecx
    jz      .Lmm_done
    cmpl    %esi, %edi
    je      .Lmm_done
    jae     .Lmm_backward

// =====================================================================
// Forward copy (dst < src)
// =====================================================================
.Lmm_fwd_loop:
#ifdef CODEVAR_HAVE_SSE2
    cmpl    $16, %ecx
    jb      .Lmm_fwd_tail
    movdqu  (%esi), %xmm0
    movdqu  %xmm0, (%edi)
    prefetchnta 64(%esi)
    addl    $16, %esi
    addl    $16, %edi
    subl    $16, %ecx
    jmp     .Lmm_fwd_loop
#else
    cmpl    $4, %ecx
    jb      .Lmm_fwd_tail
    movl    (%esi), %ebx
    movl    %ebx, (%edi)
    addl    $4, %esi
    addl    $4, %edi
    subl    $4, %ecx
    jmp     .Lmm_fwd_loop
#endif
.Lmm_fwd_tail:
    testl   %ecx, %ecx
    jz      .Lmm_done
.Lmm_fwd_byte:
    movb    (%esi), %bl
    movb    %bl, (%edi)
    incl    %esi
    incl    %edi
    decl    %ecx
    jnz     .Lmm_fwd_byte
    jmp     .Lmm_done

// =====================================================================
// Backward copy (dst > src): walk from the high end down.
// =====================================================================
.Lmm_backward:
    addl    %ecx, %esi              // end of src
    addl    %ecx, %edi              // end of dst
.Lmm_bwd_loop:
#ifdef CODEVAR_HAVE_SSE2
    cmpl    $16, %ecx
    jb      .Lmm_bwd_tail
    subl    $16, %esi
    subl    $16, %edi
    movdqu  (%esi), %xmm0
    movdqu  %xmm0, (%edi)
    prefetchnta -64(%esi)
    subl    $16, %ecx
    jmp     .Lmm_bwd_loop
#else
    cmpl    $4, %ecx
    jb      .Lmm_bwd_tail
    subl    $4, %esi
    subl    $4, %edi
    movl    (%esi), %ebx
    movl    %ebx, (%edi)
    subl    $4, %ecx
    jmp     .Lmm_bwd_loop
#endif
.Lmm_bwd_tail:
    testl   %ecx, %ecx
    jz      .Lmm_done
.Lmm_bwd_byte:
    decl    %esi
    decl    %edi
    movb    (%esi), %bl
    movb    %bl, (%edi)
    decl    %ecx
    jnz     .Lmm_bwd_byte

.Lmm_done:
    popl    %ebp
    popl    %ebx
    popl    %edi
    popl    %esi
    ret

    .size codevar_wredit_memmove, .-codevar_wredit_memmove
