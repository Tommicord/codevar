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

// RISC-V cache-prefetch primitives.
//
// The Zicbop `prefetch.{i,r,w}` hints are not part of the base ISA and
// would require an extra `-march` extension, so these are deliberately
// no-ops: they keep the arch module's API uniform across targets without
// risking an illegal instruction on cores that lack the extension.
// ABI: a0 = address (and a1 = len for the range helper).

    .text
    .align 2

    .globl codevar_wredit_prefetch_t0
    .type codevar_wredit_prefetch_t0, @function
codevar_wredit_prefetch_t0:
    ret
    .size codevar_wredit_prefetch_t0, .-codevar_wredit_prefetch_t0

    .globl codevar_wredit_prefetch_t1
    .type codevar_wredit_prefetch_t1, @function
codevar_wredit_prefetch_t1:
    ret
    .size codevar_wredit_prefetch_t1, .-codevar_wredit_prefetch_t1

    .globl codevar_wredit_prefetch_t2
    .type codevar_wredit_prefetch_t2, @function
codevar_wredit_prefetch_t2:
    ret
    .size codevar_wredit_prefetch_t2, .-codevar_wredit_prefetch_t2

    .globl codevar_wredit_prefetch_nta
    .type codevar_wredit_prefetch_nta, @function
codevar_wredit_prefetch_nta:
    ret
    .size codevar_wredit_prefetch_nta, .-codevar_wredit_prefetch_nta

    .globl codevar_wredit_prefetch_range
    .type codevar_wredit_prefetch_range, @function
codevar_wredit_prefetch_range:
    ret
    .size codevar_wredit_prefetch_range, .-codevar_wredit_prefetch_range
