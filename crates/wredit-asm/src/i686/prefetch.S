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

// i686 cache-prefetch primitives. The `prefetch` instructions require
// SSE, so without -DCODEVAR_HAVE_SSE2 these become no-ops.
// ABI (cdecl): the pointer is at [esp+4], len at [esp+8].

    .text
    .p2align 4

    .globl codevar_wredit_prefetch_t0
    .type codevar_wredit_prefetch_t0, @function
codevar_wredit_prefetch_t0:
#ifdef CODEVAR_HAVE_SSE2
    movl    4(%esp), %eax
    prefetcht0 (%eax)
#endif
    ret
    .size codevar_wredit_prefetch_t0, .-codevar_wredit_prefetch_t0

    .globl codevar_wredit_prefetch_t1
    .type codevar_wredit_prefetch_t1, @function
codevar_wredit_prefetch_t1:
#ifdef CODEVAR_HAVE_SSE2
    movl    4(%esp), %eax
    prefetcht1 (%eax)
#endif
    ret
    .size codevar_wredit_prefetch_t1, .-codevar_wredit_prefetch_t1

    .globl codevar_wredit_prefetch_t2
    .type codevar_wredit_prefetch_t2, @function
codevar_wredit_prefetch_t2:
#ifdef CODEVAR_HAVE_SSE2
    movl    4(%esp), %eax
    prefetcht2 (%eax)
#endif
    ret
    .size codevar_wredit_prefetch_t2, .-codevar_wredit_prefetch_t2

    .globl codevar_wredit_prefetch_nta
    .type codevar_wredit_prefetch_nta, @function
codevar_wredit_prefetch_nta:
#ifdef CODEVAR_HAVE_SSE2
    movl    4(%esp), %eax
    prefetchnta (%eax)
#endif
    ret
    .size codevar_wredit_prefetch_nta, .-codevar_wredit_prefetch_nta

// void codevar_wredit_prefetch_range(const void *addr, size_t len)
    .globl codevar_wredit_prefetch_range
    .type codevar_wredit_prefetch_range, @function
codevar_wredit_prefetch_range:
#ifdef CODEVAR_HAVE_SSE2
    movl    4(%esp), %eax           // addr
    movl    8(%esp), %ecx           // len
    cmpl    $64, %ecx
    jb      .Lpr_done
.Lpr_loop:
    prefetchnta (%eax)
    addl    $64, %eax
    subl    $64, %ecx
    cmpl    $64, %ecx
    jae     .Lpr_loop
.Lpr_done:
#endif
    ret
    .size codevar_wredit_prefetch_range, .-codevar_wredit_prefetch_range
