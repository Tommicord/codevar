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

// x86_64 cache-prefetch primitives.
//
// `prefetch` is available since SSE and needs no ISA level, so this
// file is assembled once (no CODEVAR_SUFFIX).
//
//   T0  -> prefetcht0  prefetch into all cache levels (temporal)
//   T1  -> prefetcht1  prefetch into L2 and beyond
//   T2  -> prefetcht2  prefetch into L3 and beyond
//   NTA -> prefetchnta non-temporal (streaming), least pollution
//
// ABI: System V AMD64. `addr` in rdi, `len` in rsi.

    .text
    .p2align 4

    .globl codevar_wredit_prefetch_t0
    .type codevar_wredit_prefetch_t0, @function
codevar_wredit_prefetch_t0:
    prefetcht0 (%rdi)
    ret
    .size codevar_wredit_prefetch_t0, .-codevar_wredit_prefetch_t0

    .globl codevar_wredit_prefetch_t1
    .type codevar_wredit_prefetch_t1, @function
codevar_wredit_prefetch_t1:
    prefetcht1 (%rdi)
    ret
    .size codevar_wredit_prefetch_t1, .-codevar_wredit_prefetch_t1

    .globl codevar_wredit_prefetch_t2
    .type codevar_wredit_prefetch_t2, @function
codevar_wredit_prefetch_t2:
    prefetcht2 (%rdi)
    ret
    .size codevar_wredit_prefetch_t2, .-codevar_wredit_prefetch_t2

    .globl codevar_wredit_prefetch_nta
    .type codevar_wredit_prefetch_nta, @function
codevar_wredit_prefetch_nta:
    prefetchnta (%rdi)
    ret
    .size codevar_wredit_prefetch_nta, .-codevar_wredit_prefetch_nta

// void codevar_wredit_prefetch_range(const void *addr, size_t len)
// Prefetches every 64-byte cache line in [addr, addr + len).
    .globl codevar_wredit_prefetch_range
    .type codevar_wredit_prefetch_range, @function
codevar_wredit_prefetch_range:
    cmpq    $64, %rsi
    jb      .Lpr_done
.Lpr_loop:
    prefetchnta (%rdi)
    addq    $64, %rdi
    subq    $64, %rsi
    cmpq    $64, %rsi
    jae     .Lpr_loop
.Lpr_done:
    ret
    .size codevar_wredit_prefetch_range, .-codevar_wredit_prefetch_range
