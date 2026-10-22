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
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

// Packlow: the parallel half of the block packer.
//
// The CPU codec is a sequential pipeline: a sliding shuffle table picks a
// candidate block, the difference between the two is classified, and the
// result is fed to a range coder that cannot run in parallel at all.
//
// The device takes the two expensive, embarrassingly parallel stages and
// leaves the sequential parts on the host:
//
//   1. `block_signatures`  - one work item per block, folding the block
//                            into the signature the shuffle table keys on.
//   2. `common_prefix`     - one work item per (target, candidate) pair,
//                            measuring how far the two agree.  This is the
//                            filter that decides which pairs are worth
//                            classifying at all.
//   3. `classify_deltas`   - one work item per surviving pair, turning the
//                            difference into the per byte exponent and
//                            number the codec stores.
//   4. `apply_deltas`      - one work item per pair, rebuilding the target
//                            from the source and the corrections.
//
// The host then reads the classifications back, picks the best candidate
// per block, and range codes the token stream.  Nothing here needs a
// barrier or local memory: every work item writes only its own slots, so
// the kernels stay a pure map over the work items.
//
// Launch contract
// ---------------
//   * `block_signatures` : global size `count`, one block per work item.
//   * `common_prefix`    : global size `pair_count`, one pair per work item.
//   * `classify_deltas`  : global size `pair_count`, one pair per work item.
//   * `apply_deltas`     : global size `pair_count`, one pair per work item.
//
// The `classify_deltas` and `apply_deltas` kernels write into a per pair
// slice of `pair_count * width` bytes, where `width` is the longest block
// the host asked about; the host owns that buffer and keeps it alive until
// the event has been waited on.

/// Folds one block into the 64-bit signature the shuffle table keys on.
///
/// The recurrence matches the host implementation exactly, so a block
/// packed on the host is recognised on the device and the other way round.
#[kernel]
fn block_signatures(
    data: *const uchar,
    offsets: *const int,
    lengths: *const int,
    count: int,
    signatures: *mut long,
) -> void {
    let gid = get_global_id(0) as int;
    if gid < count {
        let base = offsets[gid];
        let len = lengths[gid];

        // The seed and the prime of the host's signature, spelled out
        // because the dialect has no top level constants.  The seed is
        // built from its two halves so it fits the literal rules, and
        // the arithmetic is unsigned so the shift is a logical one.
        let mut hash: ulong = (3421674724 as ulong) << 32 | 2216829733 as ulong;
        // The 64 bit FNV prime, the same one the host uses.
        let prime: ulong = 1099511628211;
        let mut index: ulong = 0;
        let mut word_pos = 0;
        while word_pos + 8 <= len {
            let mut word: ulong = 0;
            let mut k = 0;
            while k < 8 {
                word = word | ((data[base + word_pos + k] as ulong) << (k * 8));
                k = k + 1;
            }
            // The rotate is written with an explicit mask because the
            // dialect's shift on a 64 bit value is arithmetic, and the
            // host rotates the bits.
            hash = ((hash << 7) | ((hash >> 57) & 127)) ^ (word * prime + index);
            index = index + 1;
            word_pos = word_pos + 8;
        }

        // The tail is the bytes that did not fill a word.  Its length is
        // computed first, because the dialect does not fold a compound
        // comparison into the loop condition the way one would expect.
        let mut tail_len = len - word_pos;
        let mut tail: ulong = 0;
        let mut k = 0;
        while k < tail_len {
            tail = tail | ((data[base + word_pos + k] as ulong) << (k * 8));
            k = k + 1;
        }
        hash = ((hash << 7) | ((hash >> 57) & 127)) ^ (tail * prime + index);
        signatures[gid] = hash ^ (hash >> 29);
    }
}

/// Measures how far a target block and a candidate block agree.
///
/// One work item per pair: the pairs are independent, so the filter the
/// host uses to skip hopeless candidates runs entirely on the device.
#[kernel]
fn common_prefix(
    targets: *const uchar,
    target_offsets: *const int,
    target_lengths: *const int,
    sources: *const uchar,
    source_offsets: *const int,
    pair_target: *const int,
    pair_source: *const int,
    pair_count: int,
    prefixes: *mut int,
) -> void {
    let gid = get_global_id(0) as int;
    if gid < pair_count {
        let t = pair_target[gid];
        let s = pair_source[gid];
        let tbase = target_offsets[t];
        let tlen = target_lengths[t];
        let sbase = source_offsets[s];

        let mut agreed = 0;
        let mut i = 0;
        while i < tlen {
            if targets[tbase + i] != sources[sbase + i] {
                break;
            }
            agreed = i + 1;
            i = i + 1;
        }
        prefixes[gid] = agreed;
    }
}

/// Turns the difference between a target block and a candidate block into
/// the per byte exponent and number the codec stores.
///
/// A byte is reachable when its difference is `number - 3` scaled by
/// `2^exponent`, with the number in `-3..=4` and the exponent in `0..=3`.
/// A negative difference may only reach -3, a positive one +4, so the
/// scale a difference needs depends on its sign.
#[kernel]
fn classify_deltas(
    targets: *const uchar,
    target_offsets: *const int,
    target_lengths: *const int,
    sources: *const uchar,
    source_offsets: *const int,
    pair_target: *const int,
    pair_source: *const int,
    pair_count: int,
    width: int,
    exponents: *mut uchar,
    numbers: *mut uchar,
    representable: *mut int,
) -> void {
    let gid = get_global_id(0) as int;
    if gid < pair_count {
        let t = pair_target[gid];
        let s = pair_source[gid];
        let tbase = target_offsets[t];
        let tlen = target_lengths[t];
        let sbase = source_offsets[s];
        let out = gid * width;

        let mut good = 0;
        let mut i = 0;
        let mut stopped = 0;
        while i < tlen && stopped == 0 {
            let delta = (targets[tbase + i] as int) - (sources[sbase + i] as int);
            let mut magnitude = delta;
            if magnitude < 0 {
                magnitude = -magnitude;
            }

            // The smallest scale whose limit the difference fits under and
            // whose scale divides it.
            let mut scale = 0;
            let mut limit = 4;
            let mut divisor = 1;
            let mut found = 0;
            while scale < 4 && found == 0 {
                let mut cap = limit;
                if delta < 0 {
                    cap = limit - 1;
                }
                if magnitude <= cap && magnitude - (magnitude / divisor) * divisor == 0 {
                    found = 1;
                } else {
                    scale = scale + 1;
                    limit = limit * 2;
                    divisor = divisor * 2;
                }
            }

            if found == 0 {
                stopped = 1;
            } else {
                let mut quotient = magnitude / (1 << scale);
                if delta < 0 {
                    quotient = -quotient;
                }
                exponents[out + i] = scale as uchar;
                numbers[out + i] = (quotient + 3) as uchar;
                good = i + 1;
            }
            i = i + 1;
        }
        representable[gid] = good;
    }
}

/// Rebuilds a target block from a candidate block and the corrections.
///
/// One work item per pair, so the whole reconstruction of a workspace
/// runs at once.
#[kernel]
fn apply_deltas(
    sources: *const uchar,
    source_offsets: *const int,
    exponents: *const uchar,
    numbers: *mut uchar,
    representable: *const int,
    pair_target: *const int,
    pair_source: *const int,
    pair_count: int,
    width: int,
    targets: *mut uchar,
    target_offsets: *const int,
) -> void {
    let gid = get_global_id(0) as int;
    if gid < pair_count {
        let t = pair_target[gid];
        let s = pair_source[gid];
        let sbase = source_offsets[s];
        let tbase = target_offsets[t];
        let good = representable[gid];
        let base = gid * width;
        let mut i = 0;
        while i < good {
            let exponent = exponents[base + i] as int;
            let number = numbers[base + i] as int;
            let scale = 1 << exponent;
            let offset = (number - 3) * scale;
            let value = (sources[sbase + i] as int) + offset;
            targets[tbase + i] = (value & 255) as uchar;
            i = i + 1;
        }
    }
}

/// Counts how often every byte value occurs in a block.
///
/// The host uses the counts to seed the literal model, so the entropy
/// coder starts from a good guess instead of a flat distribution.
#[kernel]
fn block_histograms(
    data: *const uchar,
    offsets: *const int,
    lengths: *const int,
    count: int,
    histograms: *mut int,
) -> void {
    let gid = get_global_id(0) as int;
    if gid < count {
        let base = offsets[gid];
        let len = lengths[gid];
        let out = gid * 256;
        // The histogram has to start at zero; the buffer is reused
        // between launches, so the kernel clears its own slot first.
        let mut k = 0;
        while k < 256 {
            histograms[out + k] = 0;
            k = k + 1;
        }
        let mut i = 0;
        while i < len {
            histograms[out + (data[base + i] as int)] =
                histograms[out + (data[base + i] as int)] + 1;
            i = i + 1;
        }
    }
}
