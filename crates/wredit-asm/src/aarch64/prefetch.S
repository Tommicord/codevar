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

// AArch64 (ARMv8-A) cache prefetch primitives.
// Uses the `prfm` (prefetch memory) instruction with different
// locality and target hints.
//
// `prfm` prefetch types:
//   pldl1keep  - prefetch for load, L1, keep in all levels (temporal)
//   pldl1strm  - prefetch for load, L1, streaming (non-temporal)
//   pldl2keep  - prefetch for load, L2, keep
//   pldl3keep  - prefetch for load, L3, keep
//   pstl1keep  - prefetch for store, L1, keep
//
// For cache-line walking, ARM uses 64-byte cache lines.

    .text
    .align 3

// void codevar_wredit_prefetch_t0(const void *addr)
// Prefetch into L1 (most aggressive, highest pollution)
    .global codevar_wredit_prefetch_t0
    .type codevar_wredit_prefetch_t0, %function
codevar_wredit_prefetch_t0:
    prfm    pldl1keep, [x0]
    ret
    .size codevar_wredit_prefetch_t0, .-codevar_wredit_prefetch_t0

// void codevar_wredit_prefetch_t1(const void *addr)
// Prefetch into L2
    .global codevar_wredit_prefetch_t1
    .type codevar_wredit_prefetch_t1, %function
codevar_wredit_prefetch_t1:
    prfm    pldl2keep, [x0]
    ret
    .size codevar_wredit_prefetch_t1, .-codevar_wredit_prefetch_t1

// void codevar_wredit_prefetch_t2(const void *addr)
// Prefetch into L3
    .global codevar_wredit_prefetch_t2
    .type codevar_wredit_prefetch_t2, %function
codevar_wredit_prefetch_t2:
    prfm    pldl3keep, [x0]
    ret
    .size codevar_wredit_prefetch_t2, .-codevar_wredit_prefetch_t2

// void codevar_wredit_prefetch_nta(const void *addr)
// Non-temporal prefetch: minimal cache pollution
    .global codevar_wredit_prefetch_nta
    .type codevar_wredit_prefetch_nta, %function
codevar_wredit_prefetch_nta:
    prfm    pldl1strm, [x0]
    ret
    .size codevar_wredit_prefetch_nta, .-codevar_wredit_prefetch_nta

// void codevar_wredit_prefetch_range(const void *addr, size_t len)
// Prefetches every 64-byte cache line across [addr, addr+len)
    .global codevar_wredit_prefetch_range
    .type codevar_wredit_prefetch_range, %function
codevar_wredit_prefetch_range:
    cbz     x1, .Lpr_done
    cmp     x1, #64
    blo     .Lpr_done

.Lpr_loop:
    prfm    pldl1keep, [x0]
    add     x0, x0, #64
    subs    x1, x1, #64
    bhs     .Lpr_loop

.Lpr_done:
    ret
    .size codevar_wredit_prefetch_range, .-codevar_wredit_prefetch_range
