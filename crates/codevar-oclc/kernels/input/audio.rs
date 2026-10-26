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
// CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and
// limitations under the License.

// Microphone level analysis: the parallel half of block measurement.
//
// A capture block is measured independently of the sample rate: root mean
// square, peak, clipped samples and zero crossings are all reductions over
// the sample stream, so the device runs them and the host folds the partials
// into the one record the UI reads.
//
// The three kernels differ only in how a sample is decoded, matching the
// three `SampleFormat`s the control layer reports:
//
//   1. `audio_levels_u8`  - one unsigned byte per sample, bias 128.
//   2. `audio_levels_i16` - two bytes per sample, little-endian, the common
//                            microphone format.
//   3. `audio_levels_f32` - one float per sample, already normalized.
//
// Each work item owns a chunk of 128 consecutive samples and writes eight
// floats of partial results at `gid * 8`:
//
//   0. `sum_sq` - the sum of the squared normalized samples of the chunk.
//   1. `peak`   - the largest absolute sample of the chunk.
//   2. `clipped`- samples that reached the format's full scale.
//   3. `count`  - samples in the chunk; a partial tail chunk has fewer.
//   4. `zcr`    - sign changes between consecutive samples *inside* the chunk.
//   5. `first`  - the first sample of the chunk, for the boundary test.
//   6. `last`   - the last sample of the chunk, for the boundary test.
//   7. reserved - always zero, so the stride stays a multiple of four.
//
// The host performs the final reduction: it sums `sum_sq`, `clipped`, `count`
// and `zcr`, takes the maximum `peak`, and adds the crossings *between*
// neighbouring chunks from their `last` and `first` values. Summing the
// squares in a chunk of 128 rather than across the whole block also keeps the
// float accumulation close to the host's double-precision reference.
//
// Interleaved multi-channel blocks need no special case: the reduction runs
// over every sample value it is handed, exactly as the host reference does.
//
// Launch with a global size of `ceil(samples / 128)` and a buffer of
// `32 * size` bytes. Every item writes its slot, including items whose chunk
// is empty, so the host can reduce a launch without clearing anything first.

/// Measures chunks of an 8-bit unsigned capture block.
#[kernel]
fn audio_levels_u8(data: *const uchar, len: int, partials: *mut float) -> void {
    let gid = get_global_id(0) as int;
    let base = gid * 128;
    let out = gid * 8;
    let mut sum_sq = 0.0;
    let mut peak = 0.0;
    let mut clipped = 0;
    let mut count = 0;
    let mut zcr = 0;
    let mut first = 0.0;
    let mut last = 0.0;
    let mut prev = 0.0;
    let mut j = 0;
    while j < 128 {
        let i = base + j;
        if i < len {
            let byte = data[i] as int;
            let sample = ((byte - 128) as float) / 128.0;
            let mut magnitude = sample;
            if sample < 0.0 {
                magnitude = -sample;
            }
            if magnitude > peak {
                peak = magnitude;
            }
            if byte == 0 || byte == 255 {
                clipped = clipped + 1;
            }
            sum_sq = sum_sq + sample * sample;
            count = count + 1;
            if count == 1 {
                first = sample;
            } else if (prev < 0.0 && sample >= 0.0) || (prev >= 0.0 && sample < 0.0) {
                zcr = zcr + 1;
            }
            prev = sample;
            last = sample;
        }
        j = j + 1;
    }
    partials[out] = sum_sq;
    partials[out + 1] = peak;
    partials[out + 2] = clipped as float;
    partials[out + 3] = count as float;
    partials[out + 4] = zcr as float;
    partials[out + 5] = first;
    partials[out + 6] = last;
    partials[out + 7] = 0.0;
}

/// Measures chunks of a 16-bit little-endian capture block.
///
/// A trailing odd byte is not a sample and is left out of every statistic,
/// matching the host reference.
#[kernel]
fn audio_levels_i16(data: *const uchar, len: int, partials: *mut float) -> void {
    let gid = get_global_id(0) as int;
    let base = gid * 128;
    let out = gid * 8;
    let total = len / 2;
    let mut sum_sq = 0.0;
    let mut peak = 0.0;
    let mut clipped = 0;
    let mut count = 0;
    let mut zcr = 0;
    let mut first = 0.0;
    let mut last = 0.0;
    let mut prev = 0.0;
    let mut j = 0;
    while j < 128 {
        let i = base + j;
        if i < total {
            let off = i * 2;
            let raw = ((data[off + 1] as int) << 8) + (data[off] as int);
            let mut chosen = raw;
            if raw >= 32768 {
                chosen = raw - 65536;
            }
            let sample = (chosen as float) / 32768.0;
            let mut magnitude = sample;
            if sample < 0.0 {
                magnitude = -sample;
            }
            if magnitude > peak {
                peak = magnitude;
            }
            if chosen == 32767 || chosen == -32768 {
                clipped = clipped + 1;
            }
            sum_sq = sum_sq + sample * sample;
            count = count + 1;
            if count == 1 {
                first = sample;
            } else if (prev < 0.0 && sample >= 0.0) || (prev >= 0.0 && sample < 0.0) {
                zcr = zcr + 1;
            }
            prev = sample;
            last = sample;
        }
        j = j + 1;
    }
    partials[out] = sum_sq;
    partials[out + 1] = peak;
    partials[out + 2] = clipped as float;
    partials[out + 3] = count as float;
    partials[out + 4] = zcr as float;
    partials[out + 5] = first;
    partials[out + 6] = last;
    partials[out + 7] = 0.0;
}

/// Measures chunks of a 32-bit float capture block.
///
/// The buffer is read little-endian, the same byte order the control layer
/// documents for every `AudioFormat`; a sample that is not finite contributes
/// zero so one corrupt value cannot turn the block's level into `NaN`.
#[kernel]
fn audio_levels_f32(data: *const float, len: int, partials: *mut float) -> void {
    let gid = get_global_id(0) as int;
    let base = gid * 128;
    let out = gid * 8;
    let mut sum_sq = 0.0;
    let mut peak = 0.0;
    let mut clipped = 0;
    let mut count = 0;
    let mut zcr = 0;
    let mut first = 0.0;
    let mut last = 0.0;
    let mut prev = 0.0;
    let mut j = 0;
    while j < 128 {
        let i = base + j;
        if i < len {
            let raw = data[i];
            // A finite sample minus itself is zero; NaN and either infinity
            // are not, so one subtraction rejects every non-finite value
            // without a special-case for each of them.
            let finite = raw - raw;
            let sample = if finite != 0.0 { 0.0 } else { raw };
            let mut magnitude = sample;
            if sample < 0.0 {
                magnitude = -sample;
            }
            if magnitude > peak {
                peak = magnitude;
            }
            if sample >= 1.0 || sample <= -1.0 {
                clipped = clipped + 1;
            }
            sum_sq = sum_sq + sample * sample;
            count = count + 1;
            if count == 1 {
                first = sample;
            } else if (prev < 0.0 && sample >= 0.0) || (prev >= 0.0 && sample < 0.0) {
                zcr = zcr + 1;
            }
            prev = sample;
            last = sample;
        }
        j = j + 1;
    }
    partials[out] = sum_sq;
    partials[out + 1] = peak;
    partials[out + 2] = clipped as float;
    partials[out + 3] = count as float;
    partials[out + 4] = zcr as float;
    partials[out + 5] = first;
    partials[out + 6] = last;
    partials[out + 7] = 0.0;
}
