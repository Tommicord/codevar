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

// i686 memcpy (non-overlapping).
//   -DCODEVAR_HAVE_SSE2 : 16-byte movdqu loop; else 4-byte dword loop.
// ABI (cdecl): [esp+4] = dst, [esp+8] = src, [esp+12] = len; returns eax.

    .text
    .p2align 4
    .globl codevar_wredit_memcpy
    .type codevar_wredit_memcpy, @function

codevar_wredit_memcpy:
    pushl   %esi
    pushl   %edi
    pushl   %ebx

    movl    16(%esp), %edi          // dst
    movl    20(%esp), %esi          // src
    movl    24(%esp), %ecx          // len
    movl    %edi, %eax
    testl   %ecx, %ecx
    jz      .Lcp_done
    cmpl    %esi, %edi
    je      .Lcp_done

.Lcp_loop:
#ifdef CODEVAR_HAVE_SSE2
    cmpl    $16, %ecx
    jb      .Lcp_tail
    movdqu  (%esi), %xmm0
    movdqu  %xmm0, (%edi)
    prefetchnta 64(%esi)
    addl    $16, %esi
    addl    $16, %edi
    subl    $16, %ecx
    jmp     .Lcp_loop
#else
    cmpl    $4, %ecx
    jb      .Lcp_tail
    movl    (%esi), %ebx
    movl    %ebx, (%edi)
    addl    $4, %esi
    addl    $4, %edi
    subl    $4, %ecx
    jmp     .Lcp_loop
#endif
.Lcp_tail:
    testl   %ecx, %ecx
    jz      .Lcp_done
.Lcp_byte:
    movb    (%esi), %bl
    movb    %bl, (%edi)
    incl    %esi
    incl    %edi
    decl    %ecx
    jnz     .Lcp_byte

.Lcp_done:
    popl    %ebx
    popl    %edi
    popl    %esi
    ret

    .size codevar_wredit_memcpy, .-codevar_wredit_memcpy
