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

// General-purpose i686 kernels.
//
// ABI (cdecl): arguments live at 4(%esp), 8(%esp), ... on entry; the
// 64-bit results are returned in edx:eax (high:low). The caller cleans
// the arguments, so every routine ends in a plain `ret`. Only caller-
// saved registers are used unless pushed and popped here.

    .text

// =====================================================================
// fill_bytes(dst, value, len) -> dst
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_fill_bytes
    .type codevar_wredit_k_fill_bytes, @function
codevar_wredit_k_fill_bytes:
    pushl   %edi
    movl    8(%esp), %edi           // dst
    movl    12(%esp), %eax          // value
    movl    16(%esp), %ecx          // len
    movl    %edi, %edx              // preserve dst for the return
    testl   %ecx, %ecx
    jz      .Lfb_done
    rep stosb
.Lfb_done:
    movl    %edx, %eax
    popl    %edi
    ret
    .size codevar_wredit_k_fill_bytes, .-codevar_wredit_k_fill_bytes

// =====================================================================
// fill_words(dst, value, count) -> dst
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_fill_words
    .type codevar_wredit_k_fill_words, @function
codevar_wredit_k_fill_words:
    pushl   %edi
    movl    8(%esp), %edi           // dst
    movl    12(%esp), %eax          // value
    movl    16(%esp), %ecx          // count
    movl    %edi, %edx
    testl   %ecx, %ecx
    jz      .Lfw_done
    rep stosl
.Lfw_done:
    movl    %edx, %eax
    popl    %edi
    ret
    .size codevar_wredit_k_fill_words, .-codevar_wredit_k_fill_words

// =====================================================================
// copy_words(dst, src, count) -> dst   (non-overlapping)
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_copy_words
    .type codevar_wredit_k_copy_words, @function
codevar_wredit_k_copy_words:
    pushl   %esi
    pushl   %edi
    movl    12(%esp), %edi          // dst
    movl    16(%esp), %esi          // src
    movl    20(%esp), %ecx          // count
    movl    %edi, %edx
    testl   %ecx, %ecx
    jz      .Lcw_done
    rep movsl
.Lcw_done:
    movl    %edx, %eax
    popl    %edi
    popl    %esi
    ret
    .size codevar_wredit_k_copy_words, .-codevar_wredit_k_copy_words

// =====================================================================
// move_words(dst, src, count) -> dst   (overlap-safe)
// Forward when dst < src, backward otherwise.
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_move_words
    .type codevar_wredit_k_move_words, @function
codevar_wredit_k_move_words:
    pushl   %esi
    pushl   %edi
    movl    12(%esp), %edi          // dst
    movl    16(%esp), %esi          // src
    movl    20(%esp), %ecx          // count
    movl    %edi, %edx
    testl   %ecx, %ecx
    jz      .Lmw_done
    cmpl    %esi, %edi
    jb      .Lmw_forward

.Lmw_backward:
    leal    -4(%edi,%ecx,4), %edi   // last word of dst
    leal    -4(%esi,%ecx,4), %esi   // last word of src
    std
    rep movsl
    cld
    jmp     .Lmw_done

.Lmw_forward:
    rep movsl

.Lmw_done:
    movl    %edx, %eax
    popl    %edi
    popl    %esi
    ret
    .size codevar_wredit_k_move_words, .-codevar_wredit_k_move_words

// =====================================================================
// count_byte(hay, needle, len) -> count
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_count_byte
    .type codevar_wredit_k_count_byte, @function
codevar_wredit_k_count_byte:
    pushl   %esi
    pushl   %ebx
    movl    12(%esp), %esi          // hay
    movl    16(%esp), %ebx          // needle
    movl    20(%esp), %ecx          // len
    xorl    %eax, %eax
    testl   %ecx, %ecx
    jz      .Lcb_done
.Lcb_loop:
    movb    (%esi), %dl
    cmpb    %bl, %dl
    jne     .Lcb_skip
    incl    %eax
.Lcb_skip:
    incl    %esi
    decl    %ecx
    jnz     .Lcb_loop
.Lcb_done:
    popl    %ebx
    popl    %esi
    ret
    .size codevar_wredit_k_count_byte, .-codevar_wredit_k_count_byte

// =====================================================================
// find_byte_rev(hay, needle, len) -> pointer or NULL
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_find_byte_rev
    .type codevar_wredit_k_find_byte_rev, @function
codevar_wredit_k_find_byte_rev:
    pushl   %esi
    pushl   %ebx
    movl    12(%esp), %esi          // hay
    movl    16(%esp), %ebx          // needle
    movl    20(%esp), %ecx          // len
    xorl    %eax, %eax
    testl   %ecx, %ecx
    jz      .Lfbr_done
    leal    (%esi,%ecx), %edx       // one past the end
.Lfbr_loop:
    decl    %edx
    cmpb    %bl, (%edx)
    je      .Lfbr_found
    decl    %ecx
    jnz     .Lfbr_loop
    jmp     .Lfbr_done
.Lfbr_found:
    movl    %edx, %eax
.Lfbr_done:
    popl    %ebx
    popl    %esi
    ret
    .size codevar_wredit_k_find_byte_rev, .-codevar_wredit_k_find_byte_rev

// =====================================================================
// find_any2(hay, a, b, len) -> pointer or NULL
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_find_any2
    .type codevar_wredit_k_find_any2, @function
codevar_wredit_k_find_any2:
    pushl   %esi
    movl    8(%esp), %esi           // hay
    movb    12(%esp), %dl           // a
    movb    16(%esp), %dh           // b
    movl    20(%esp), %ecx          // len
    testl   %ecx, %ecx
    jz      .Lfa2_none
.Lfa2_loop:
    movb    (%esi), %al
    cmpb    %dl, %al
    je      .Lfa2_found
    cmpb    %dh, %al
    je      .Lfa2_found
    incl    %esi
    decl    %ecx
    jnz     .Lfa2_loop
.Lfa2_none:
    xorl    %eax, %eax
    jmp     .Lfa2_done
.Lfa2_found:
    movl    %esi, %eax
.Lfa2_done:
    popl    %esi
    ret
    .size codevar_wredit_k_find_any2, .-codevar_wredit_k_find_any2

// =====================================================================
// find_not_byte(hay, needle, len) -> pointer or NULL
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_find_not_byte
    .type codevar_wredit_k_find_not_byte, @function
codevar_wredit_k_find_not_byte:
    pushl   %esi
    pushl   %ebx
    movl    12(%esp), %esi          // hay
    movl    16(%esp), %ebx          // needle
    movl    20(%esp), %ecx          // len
    testl   %ecx, %ecx
    jz      .Lfnb_none
.Lfnb_loop:
    movb    (%esi), %al
    cmpb    %bl, %al
    jne     .Lfnb_found
    incl    %esi
    decl    %ecx
    jnz     .Lfnb_loop
.Lfnb_none:
    xorl    %eax, %eax
    jmp     .Lfnb_done
.Lfnb_found:
    movl    %esi, %eax
.Lfnb_done:
    popl    %ebx
    popl    %esi
    ret
    .size codevar_wredit_k_find_not_byte, .-codevar_wredit_k_find_not_byte

// =====================================================================
// count_eq_u32(src, value, mask, count) -> count
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_count_eq_u32
    .type codevar_wredit_k_count_eq_u32, @function
codevar_wredit_k_count_eq_u32:
    pushl   %esi
    pushl   %ebx
    pushl   %edi
    movl    16(%esp), %esi          // src
    movl    20(%esp), %ebx          // value
    movl    24(%esp), %edi          // mask
    movl    28(%esp), %ecx          // count
    xorl    %eax, %eax
    testl   %ecx, %ecx
    jz      .Lceu_done
.Lceu_loop:
    movl    (%esi), %edx
    andl    %edi, %edx
    cmpl    %ebx, %edx
    jne     .Lceu_skip
    incl    %eax
.Lceu_skip:
    addl    $4, %esi
    decl    %ecx
    jnz     .Lceu_loop
.Lceu_done:
    popl    %edi
    popl    %ebx
    popl    %esi
    ret
    .size codevar_wredit_k_count_eq_u32, .-codevar_wredit_k_count_eq_u32

// =====================================================================
// find_eq_u32(src, value, mask, count) -> pointer or NULL
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_find_eq_u32
    .type codevar_wredit_k_find_eq_u32, @function
codevar_wredit_k_find_eq_u32:
    pushl   %esi
    pushl   %ebx
    pushl   %edi
    movl    16(%esp), %esi          // src
    movl    20(%esp), %ebx          // value
    movl    24(%esp), %edi          // mask
    movl    28(%esp), %ecx          // count
    xorl    %eax, %eax
    testl   %ecx, %ecx
    jz      .Lfeu_done
.Lfeu_loop:
    movl    (%esi), %edx
    andl    %edi, %edx
    cmpl    %ebx, %edx
    je      .Lfeu_found
    addl    $4, %esi
    decl    %ecx
    jnz     .Lfeu_loop
    jmp     .Lfeu_done
.Lfeu_found:
    movl    %esi, %eax
.Lfeu_done:
    popl    %edi
    popl    %ebx
    popl    %esi
    ret
    .size codevar_wredit_k_find_eq_u32, .-codevar_wredit_k_find_eq_u32

// =====================================================================
// find_eq_u32_rev(src, value, mask, count) -> pointer or NULL
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_find_eq_u32_rev
    .type codevar_wredit_k_find_eq_u32_rev, @function
codevar_wredit_k_find_eq_u32_rev:
    pushl   %esi
    pushl   %ebx
    pushl   %edi
    movl    16(%esp), %esi          // src
    movl    20(%esp), %ebx          // value
    movl    24(%esp), %edi          // mask
    movl    28(%esp), %ecx          // count
    xorl    %eax, %eax
    testl   %ecx, %ecx
    jz      .Lfeur_done
    leal    (%esi,%ecx,4), %esi     // one past the end
.Lfeur_loop:
    subl    $4, %esi
    movl    (%esi), %edx
    andl    %edi, %edx
    cmpl    %ebx, %edx
    je      .Lfeur_found
    decl    %ecx
    jnz     .Lfeur_loop
    jmp     .Lfeur_done
.Lfeur_found:
    movl    %esi, %eax
.Lfeur_done:
    popl    %edi
    popl    %ebx
    popl    %esi
    ret
    .size codevar_wredit_k_find_eq_u32_rev, .-codevar_wredit_k_find_eq_u32_rev

// =====================================================================
// memxor(dst, src, len) -> dst   (overlap only when dst == src)
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_memxor
    .type codevar_wredit_k_memxor, @function
codevar_wredit_k_memxor:
    pushl   %esi
    pushl   %edi
    movl    12(%esp), %edi          // dst
    movl    16(%esp), %esi          // src
    movl    20(%esp), %ecx          // len
    movl    %edi, %eax
    testl   %ecx, %ecx
    jz      .Lmx_done
.Lmx_loop:
    cmpl    $4, %ecx
    jb      .Lmx_tail
    movl    (%esi), %edx
    xorl    %edx, (%edi)
    addl    $4, %esi
    addl    $4, %edi
    subl    $4, %ecx
    jmp     .Lmx_loop
.Lmx_tail:
    testl   %ecx, %ecx
    jz      .Lmx_done
.Lmx_byte:
    movb    (%esi), %dl
    xorb    %dl, (%edi)
    incl    %esi
    incl    %edi
    decl    %ecx
    jnz     .Lmx_byte
.Lmx_done:
    popl    %edi
    popl    %esi
    ret
    .size codevar_wredit_k_memxor, .-codevar_wredit_k_memxor

// =====================================================================
// memcmp(a, b, len) -> -1 | 0 | 1
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_memcmp
    .type codevar_wredit_k_memcmp, @function
codevar_wredit_k_memcmp:
    pushl   %esi
    pushl   %edi
    movl    12(%esp), %esi          // a
    movl    16(%esp), %edi          // b
    movl    20(%esp), %ecx          // len
    testl   %ecx, %ecx
    jz      .Lmc_equal
.Lmc_loop:
    movzbl  (%esi), %eax
    movzbl  (%edi), %edx
    cmpl    %edx, %eax
    jne     .Lmc_diff
    incl    %esi
    incl    %edi
    decl    %ecx
    jnz     .Lmc_loop
.Lmc_equal:
    xorl    %eax, %eax
    jmp     .Lmc_done
.Lmc_diff:
    jb      .Lmc_less
    movl    $1, %eax
    jmp     .Lmc_done
.Lmc_less:
    movl    $-1, %eax
.Lmc_done:
    popl    %edi
    popl    %esi
    ret
    .size codevar_wredit_k_memcmp, .-codevar_wredit_k_memcmp

// =====================================================================
// sum_bytes(src, len) -> u64   (edx:eax)
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_sum_bytes
    .type codevar_wredit_k_sum_bytes, @function
codevar_wredit_k_sum_bytes:
    pushl   %esi
    pushl   %ebx
    movl    12(%esp), %esi          // src
    movl    16(%esp), %ecx          // len
    xorl    %eax, %eax
    xorl    %edx, %edx
    testl   %ecx, %ecx
    jz      .Lsb_done
.Lsb_loop:
    cmpl    $4, %ecx
    jb      .Lsb_tail
    movzbl  (%esi), %ebx
    addl    %ebx, %eax
    adcl    $0, %edx
    movzbl  1(%esi), %ebx
    addl    %ebx, %eax
    adcl    $0, %edx
    movzbl  2(%esi), %ebx
    addl    %ebx, %eax
    adcl    $0, %edx
    movzbl  3(%esi), %ebx
    addl    %ebx, %eax
    adcl    $0, %edx
    addl    $4, %esi
    subl    $4, %ecx
    jmp     .Lsb_loop
.Lsb_tail:
    testl   %ecx, %ecx
    jz      .Lsb_done
.Lsb_byte:
    movzbl  (%esi), %ebx
    addl    %ebx, %eax
    adcl    $0, %edx
    incl    %esi
    decl    %ecx
    jnz     .Lsb_byte
.Lsb_done:
    popl    %ebx
    popl    %esi
    ret
    .size codevar_wredit_k_sum_bytes, .-codevar_wredit_k_sum_bytes

// =====================================================================
// weighted_checksum(src, len, seed) -> u64   (edx:eax)
// weight = seed % 251 + 1; wraps back to 1 after 251.
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_weighted_checksum
    .type codevar_wredit_k_weighted_checksum, @function
codevar_wredit_k_weighted_checksum:
    pushl   %esi
    pushl   %ebx
    pushl   %ebp
    movl    16(%esp), %esi          // src
    movl    20(%esp), %ecx          // len
    movl    24(%esp), %eax          // seed
    xorl    %edx, %edx
    movl    $251, %ebx
    divl    %ebx                    // edx = seed % 251
    incl    %edx
    movl    %edx, %ebp              // weight
    xorl    %eax, %eax              // accumulator low
    xorl    %edx, %edx              // accumulator high
    testl   %ecx, %ecx
    jz      .Lwc_done
.Lwc_loop:
    movzbl  (%esi), %ebx
    imull   %ebp, %ebx
    addl    %ebx, %eax
    adcl    $0, %edx
    incl    %ebp
    cmpl    $251, %ebp
    jbe     .Lwc_nowrap
    movl    $1, %ebp
.Lwc_nowrap:
    incl    %esi
    decl    %ecx
    jnz     .Lwc_loop
.Lwc_done:
    popl    %ebp
    popl    %ebx
    popl    %esi
    ret
    .size codevar_wredit_k_weighted_checksum, .-codevar_wredit_k_weighted_checksum

// =====================================================================
// utf8_count_starts(src, len) -> count
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_utf8_count_starts
    .type codevar_wredit_k_utf8_count_starts, @function
codevar_wredit_k_utf8_count_starts:
    pushl   %esi
    movl    8(%esp), %esi           // src
    movl    12(%esp), %ecx          // len
    xorl    %eax, %eax
    testl   %ecx, %ecx
    jz      .Lutf_done
.Lutf_loop:
    movzbl  (%esi), %edx
    andl    $0xC0, %edx
    cmpl    $0x80, %edx
    je      .Lutf_skip
    incl    %eax
.Lutf_skip:
    incl    %esi
    decl    %ecx
    jnz     .Lutf_loop
.Lutf_done:
    popl    %esi
    ret
    .size codevar_wredit_k_utf8_count_starts, .-codevar_wredit_k_utf8_count_starts

// =====================================================================
// replace_byte(dst, from, to, len) -> replaced count
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_replace_byte
    .type codevar_wredit_k_replace_byte, @function
codevar_wredit_k_replace_byte:
    pushl   %esi
    pushl   %ebx
    movl    12(%esp), %esi          // dst
    movb    16(%esp), %bl           // from
    movb    20(%esp), %bh           // to
    movl    24(%esp), %ecx          // len
    xorl    %eax, %eax
    testl   %ecx, %ecx
    jz      .Lrb_done
.Lrb_loop:
    movzbl  (%esi), %edx
    cmpb    %bl, %dl
    jne     .Lrb_skip
    movb    %bh, (%esi)
    incl    %eax
.Lrb_skip:
    incl    %esi
    decl    %ecx
    jnz     .Lrb_loop
.Lrb_done:
    popl    %ebx
    popl    %esi
    ret
    .size codevar_wredit_k_replace_byte, .-codevar_wredit_k_replace_byte

// =====================================================================
// reverse_bytes(dst, src, len) -> dst   (non-overlapping)
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_reverse_bytes
    .type codevar_wredit_k_reverse_bytes, @function
codevar_wredit_k_reverse_bytes:
    pushl   %esi
    pushl   %edi
    movl    12(%esp), %edi          // dst
    movl    16(%esp), %esi          // src
    movl    20(%esp), %ecx          // len
    movl    %edi, %eax
    testl   %ecx, %ecx
    jz      .Lrv_done
    addl    %ecx, %esi              // walk src from its end
.Lrv_loop:
    decl    %esi
    movb    (%esi), %dl
    movb    %dl, (%edi)
    incl    %edi
    decl    %ecx
    jnz     .Lrv_loop
.Lrv_done:
    popl    %edi
    popl    %esi
    ret
    .size codevar_wredit_k_reverse_bytes, .-codevar_wredit_k_reverse_bytes

// =====================================================================
// fill_pattern8(dst, pattern, len) -> dst
// The 8-byte pattern is stored on the stack so the tail can be copied
// with rep movsb in native byte order.
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_fill_pattern8
    .type codevar_wredit_k_fill_pattern8, @function
codevar_wredit_k_fill_pattern8:
    pushl   %edi
    pushl   %esi
    pushl   %ebx
    subl    $8, %esp                // pattern scratch
    movl    24(%esp), %edi          // dst
    movl    28(%esp), %edx          // pattern low
    movl    32(%esp), %ebx          // pattern high
    movl    %edx, 0(%esp)
    movl    %ebx, 4(%esp)
    movl    %edi, %eax
    movl    36(%esp), %ecx          // len
    testl   %ecx, %ecx
    jz      .Lfp_done
.Lfp_loop:
    cmpl    $8, %ecx
    jb      .Lfp_tail
    movl    %edx, (%edi)
    movl    %ebx, 4(%edi)
    addl    $8, %edi
    subl    $8, %ecx
    jmp     .Lfp_loop
.Lfp_tail:
    andl    $7, %ecx
    jz      .Lfp_done
    movl    %esp, %esi
    rep movsb
.Lfp_done:
    addl    $8, %esp
    popl    %ebx
    popl    %esi
    popl    %edi
    ret
    .size codevar_wredit_k_fill_pattern8, .-codevar_wredit_k_fill_pattern8

// =====================================================================
// byte_histogram(counts, src, len)
// Zeroes all 256 counters, then counts every input byte.
// =====================================================================
    .p2align 4
    .globl codevar_wredit_k_byte_histogram
    .type codevar_wredit_k_byte_histogram, @function
codevar_wredit_k_byte_histogram:
    pushl   %esi
    pushl   %edi
    movl    12(%esp), %edi          // counts
    xorl    %eax, %eax
    movl    $256, %ecx
    rep stosl
    movl    12(%esp), %edi          // reset to the counter base
    movl    16(%esp), %esi          // src
    movl    20(%esp), %ecx          // len
    testl   %ecx, %ecx
    jz      .Lbh_done
.Lbh_loop:
    movzbl  (%esi), %eax
    incl    (%edi,%eax,4)
    incl    %esi
    decl    %ecx
    jnz     .Lbh_loop
.Lbh_done:
    popl    %edi
    popl    %esi
    ret
    .size codevar_wredit_k_byte_histogram, .-codevar_wredit_k_byte_histogram
