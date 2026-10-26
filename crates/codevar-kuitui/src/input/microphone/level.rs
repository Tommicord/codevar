//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The level of one block of captured samples.
//!
//! [`AudioLevels::analyze`] is the whole of the audio measurement: it reads
//! the interleaved bytes of an [`AudioFormat`] and produces the root mean
//! square, the peak, the number of clipped samples and the number of sign
//! crossings of the block. Everything else the microphone module keeps is
//! derived from those numbers, so this is where a level meter is accurate or
//! not.
//!
//! # Sample-rate independence
//!
//! Nothing here is averaged across blocks and nothing depends on the sample
//! rate: the values describe exactly the samples that were passed in, so a
//! block of 480 samples at 48 kHz and a block of 80 samples at 8 kHz each
//! report their own level. The helpers that need a rate
//! ([`AudioLevels::frames`], [`AudioLevels::duration_micros`],
//! [`AudioLevels::zero_crossings_per_second`]) take the [`AudioFormat`] as an
//! argument, and time-based smoothing lives in the microphone state, where the
//! coefficient is derived from the rate of the block being applied. That is
//! what makes one handler behave identically at 8, 16, 44.1 and 48 kHz.
//!
//! # Kernels
//!
//! The sum of squares, the peak and the clipped count reduce over independent
//! samples, so on x86-64 they run one sample per lane — SSE2 for the baseline
//! and AVX2 when [`codevar_base::cpuid`] reports it — and WebAssembly uses
//! simd128 for the `F32` format when the build has it. A kernel accumulates
//! the squares in `f32` lanes and folds them into an `f64` total every 512
//! samples, so its root mean square differs from the scalar reference by less
//! than the width of a meter, while the peak and the two counts stay exact.
//! The crossing count has a serial dependency from one sample to the next, so
//! it always comes from the same scalar pass and cannot disagree between
//! paths. Each kernel leaves the samples that do not fill a whole vector to
//! the reference, so a short block costs nothing but a branch. The kernels
//! stop at 256 bits: an audio block is a byte stream, so the loop is bound by
//! the load and the widening conversion rather than by the arithmetic width.

use libm::{log10f, sqrt};

use codevar_kuitui_control::{AudioFormat, SampleFormat};

/// Samples a vector kernel accumulates before it folds its lanes into the
/// `f64` total.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const FLUSH_SAMPLES: usize = 512;

/// The scale that turns a signed 16-bit sample into a unit level.
const I16_SCALE: f32 = 1.0 / 32_768.0;

/// The level of one block of samples.
///
/// The values describe the block that was just analyzed; nothing here is
/// averaged across blocks, so a quiet block after a loud one reports the
/// quiet level instead of a lingering decay.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AudioLevels {
    /// Root mean square of the samples, in `0.0..=1.0` for well-formed input.
    pub rms: f32,
    /// The largest absolute sample of the block.
    pub peak: f32,
    /// Number of samples that reached the format's full scale.
    pub clipped: u32,
    /// Number of samples analyzed; a trailing partial sample is ignored.
    pub samples: u32,
    /// Number of times the sign of the interleaved sample stream changed.
    ///
    /// The crossings are counted over the interleaved stream, so a stereo
    /// block reports the crossings of both channels taken in turn.
    pub crossings: u32,
}

impl AudioLevels {
    /// The level of an empty or silent block.
    pub const SILENT: AudioLevels = AudioLevels {
        rms: 0.0,
        peak: 0.0,
        clipped: 0,
        samples: 0,
        crossings: 0,
    };

    /// Measures one block of raw samples.
    ///
    /// The bytes are read little-endian, matching [`AudioFormat`]. A block
    /// whose length is not a whole number of samples loses its trailing
    /// bytes; a non-finite `F32` sample contributes zero, so one corrupt
    /// value cannot turn the whole block into `NaN`.
    #[must_use]
    pub fn analyze(format: AudioFormat, data: &[u8]) -> Self {
        let crossings = crossings_of(format.sample_format, data);
        measure(format.sample_format, data).into_levels(crossings)
    }

    /// Whether the block reached full scale at least once.
    #[must_use]
    pub const fn is_clipping(self) -> bool {
        self.clipped > 0
    }

    /// The loudness of the block in decibels relative to full scale.
    ///
    /// Silence is [`f32::NEG_INFINITY`], which a meter draws as an empty bar
    /// instead of as a very low number.
    #[must_use]
    pub fn rms_dbfs(self) -> f32 {
        dbfs(self.rms)
    }

    /// The peak of the block in decibels relative to full scale.
    #[must_use]
    pub fn peak_dbfs(self) -> f32 {
        dbfs(self.peak)
    }

    /// Number of interleaved frames in the block.
    ///
    /// `None` when the format has no channel, which no real backend reports
    /// but a caller still has to survive.
    #[must_use]
    pub const fn frames(self, format: AudioFormat) -> Option<u64> {
        if format.channels == 0 {
            return None;
        }
        Some(self.samples as u64 / format.channels as u64)
    }

    /// How long the block lasts, in microseconds.
    ///
    /// The duration comes from the format's rate, so the same sample count is
    /// six times longer at 8 kHz than at 48 kHz. `None` when the format has
    /// no rate or no channel.
    #[must_use]
    pub const fn duration_micros(self, format: AudioFormat) -> Option<u64> {
        if format.sample_rate == 0 || format.channels == 0 {
            return None;
        }
        let frames = self.samples as u64 / format.channels as u64;
        Some(frames.saturating_mul(1_000_000) / format.sample_rate as u64)
    }
    /// Sign crossings per second: the crude pitch indicator of a meter, and
    /// the value a voice-activity gate reads.
    ///
    /// `None` for a block with no frame or a format with no rate.
    #[must_use]
    pub fn zero_crossings_per_second(self, format: AudioFormat) -> Option<f32> {
        let frames = self.frames(format)?;
        if frames == 0 || format.sample_rate == 0 {
            return None;
        }
        let per_frame = f64::from(self.crossings) / frames as f64;
        Some((per_frame * f64::from(format.sample_rate)) as f32)
    }
}

/// Measures one block with the widest kernel the target offers.
fn measure(format: SampleFormat, data: &[u8]) -> Measure {
    match format {
        SampleFormat::U8 => scalar::u8(data),
        SampleFormat::I16 => {
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(measure) = x86::i16(data) {
                    return measure;
                }
            }
            scalar::i16(data)
        }
        SampleFormat::F32 => {
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(measure) = x86::f32(data) {
                    return measure;
                }
            }
            scalar::f32(data)
        }
    }
}

/// What a kernel measured, before the root mean square and the crossings.
///
/// Every path fills this in and hands it to [`Measure::into_levels`], so the
/// definition of `rms` lives in one place and an accelerated path cannot
/// drift from the reference.
struct Measure {
    /// Sum of the squared samples.
    sum_sq: f64,
    /// Largest absolute sample.
    peak: f32,
    /// Samples at full scale.
    clipped: u32,
    /// Samples measured.
    samples: u32,
}

impl Measure {
    /// An empty accumulator.
    #[inline]
    const fn empty() -> Self {
        Self {
            sum_sq: 0.0,
            peak: 0.0,
            clipped: 0,
            samples: 0,
        }
    }

    /// Completes the measurement with its crossings and its root mean square.
    #[inline]
    fn into_levels(self, crossings: u32) -> AudioLevels {
        AudioLevels {
            rms: rms_of(self.sum_sq, self.samples),
            peak: self.peak,
            clipped: self.clipped,
            samples: self.samples,
            crossings,
        }
    }

    /// Folds one sample, given as a unit level, into the accumulator.
    #[inline]
    fn add(&mut self, sample: f32, clipped: bool) {
        let magnitude = if sample < 0.0 { -sample } else { sample };
        if magnitude > self.peak {
            self.peak = magnitude;
        }
        if clipped {
            self.clipped = self.clipped.saturating_add(1);
        }
        self.sum_sq += f64::from(sample) * f64::from(sample);
        self.samples = self.samples.saturating_add(1);
    }

    /// Adds a measurement taken over another part of the same block.
    #[inline]
    fn merge(&mut self, other: Measure) {
        self.sum_sq += other.sum_sq;
        if other.peak > self.peak {
            self.peak = other.peak;
        }
        self.clipped = self.clipped.saturating_add(other.clipped);
        self.samples = self.samples.saturating_add(other.samples);
    }
}

/// Turns a sum of squares and a sample count into a root mean square.
#[inline]
fn rms_of(sum_sq: f64, samples: u32) -> f32 {
    if samples == 0 {
        return 0.0;
    }
    sqrt(sum_sq / f64::from(samples)) as f32
}

/// One linear level in decibels relative to full scale.
#[inline]
fn dbfs(level: f32) -> f32 {
    if level <= 0.0 {
        return f32::NEG_INFINITY;
    }
    20.0 * log10f(level)
}

/// Counts the sign changes of the decoded sample stream.
///
/// The first sample of the block is the reference, so a one-sample block has
/// no crossings. A negative zero counts as negative, which is the sign its
/// bits carry, so the reference and the accelerated paths agree by
/// construction.
fn crossings_of(format: SampleFormat, data: &[u8]) -> u32 {
    let mut crossings = Crossings::default();
    match format {
        SampleFormat::U8 => {
            for &byte in data {
                crossings.add(byte < 128);
            }
        }
        SampleFormat::I16 => {
            for pair in data.chunks_exact(2) {
                crossings.add(i16::from_le_bytes([pair[0], pair[1]]) < 0);
            }
        }
        SampleFormat::F32 => {
            for quad in data.chunks_exact(4) {
                let sample = f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]);
                // The measurement substitutes a zero for a non-finite sample,
                // so the crossing pass reads that zero's sign too.
                let sample = if sample.is_finite() { sample } else { 0.0 };
                crossings.add(sample < 0.0);
            }
        }
    }
    crossings.count()
}

/// The running sign-change count of one block.
#[derive(Debug, Default)]
struct Crossings {
    /// Sign of the previous sample, once there is one.
    previous: Option<bool>,
    /// Crossings counted so far.
    count: u32,
}

impl Crossings {
    /// Compares one sample's sign with the previous one and remembers it.
    #[inline]
    fn add(&mut self, negative: bool) {
        if let Some(previous) = self.previous {
            self.count = self
                .count
                .saturating_add(u32::from(negative != previous));
        }
        self.previous = Some(negative);
    }

    /// The number of crossings of the block.
    #[inline]
    const fn count(self) -> u32 {
        self.count
    }
}

/// The scalar reference: one sample at a time, with an `f64` total.
///
/// Every target compiles it, and it is the oracle the vector kernels are
/// tested against. It also measures the tail of a block that a kernel cannot
/// cover with a whole vector.
mod scalar {
    use super::{I16_SCALE, Measure};

    /// Measures a `U8` block: biased by half a scale, with full scale at both
    /// ends.
    pub(super) fn u8(data: &[u8]) -> Measure {
        let mut measure = Measure::empty();
        for &byte in data {
            let sample = (f32::from(byte) - 128.0) / 128.0;
            measure.add(sample, byte == 0 || byte == u8::MAX);
        }
        measure
    }

    /// Measures an `I16` block read little-endian.
    pub(super) fn i16(data: &[u8]) -> Measure {
        let mut measure = Measure::empty();
        for pair in data.chunks_exact(2) {
            let raw = i16::from_le_bytes([pair[0], pair[1]]);
            measure.add(f32::from(raw) * I16_SCALE, raw == i16::MIN || raw == i16::MAX);
        }
        measure
    }

    /// Measures an `F32` block, reading a non-finite sample as zero.
    pub(super) fn f32(data: &[u8]) -> Measure {
        let mut measure = Measure::empty();
        for quad in data.chunks_exact(4) {
            let sample = f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]);
            let sample = if sample.is_finite() { sample } else { 0.0 };
            measure.add(sample, sample >= 1.0 || sample <= -1.0);
        }
        measure
    }
}

/// The x86-64 kernels, one sample per lane.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{__m128, __m256};

    use codevar_base::cpuid::{self, Feature};

    use super::{FLUSH_SAMPLES, I16_SCALE, Measure, scalar};

    /// Measures a block of signed 16-bit samples.
    ///
    /// Returns `None` when not even SSE2 is available, which leaves the
    /// scalar reference in charge.
    pub(super) fn i16(data: &[u8]) -> Option<Measure> {
        if cpuid::has(Feature::Avx2) {
            // SAFETY: the probe reports AVX2 together with the OS-enabled
            // YMM state, which is exactly the precondition of the `avx2`
            // target feature. The kernel reads only `data`.
            return Some(unsafe { i16_avx2(data) });
        }
        if cpuid::has(Feature::Sse2) {
            // SAFETY: the probe reports SSE2, the precondition of the `sse2`
            // target feature. The kernel reads only `data`.
            return Some(unsafe { i16_sse2(data) });
        }
        None
    }

    /// Measures a block of 32-bit floating point samples.
    pub(super) fn f32(data: &[u8]) -> Option<Measure> {
        if cpuid::has(Feature::Avx2) {
            // SAFETY: as for [`i16`]; the kernel reads only whole vectors of
            // `data`.
            return Some(unsafe { f32_avx2(data) });
        }
        if cpuid::has(Feature::Sse2) {
            // SAFETY: as for [`i16`].
            return Some(unsafe { f32_sse2(data) });
        }
        None
    }

    /// The `I16` block with eight samples per 128-bit register.
    ///
    /// # Safety
    ///
    /// The caller must have verified SSE2 support.
    #[target_feature(enable = "sse2")]
    unsafe fn i16_sse2(data: &[u8]) -> Measure {
        use core::arch::x86_64::{
            _mm_add_ps, _mm_cmpeq_epi16, _mm_cvtepi32_ps, _mm_loadu_si128, _mm_max_ps, _mm_movemask_epi8,
            _mm_mul_ps, _mm_or_si128, _mm_set1_epi16, _mm_set1_ps, _mm_setzero_ps, _mm_slli_epi32,
            _mm_srai_epi32, _mm_sub_ps,
        };

        let body_len = data.len() - data.len() % 16;
        let (body, tail) = data.split_at(body_len);
        // SAFETY: the caller guarantees SSE2; each load reads one whole
        // 16-byte chunk of `data` and every other operation stays in
        // registers.
        unsafe {
            let mut measure = Measure::empty();
            let scale = _mm_set1_ps(I16_SCALE);
            let zero = _mm_setzero_ps();
            let low_full = _mm_set1_epi16(-32_768);
            let high_full = _mm_set1_epi16(32_767);
            let mut squares = [zero; 2];
            let mut peak = zero;
            let mut since = 0usize;
            for chunk in body.chunks(16) {
                let raw = _mm_loadu_si128(chunk.as_ptr().cast());
                // One 32-bit lane holds two samples: shifting left and back
                // sign-extends the even sample, an arithmetic shift alone
                // sign-extends the odd one.
                let even = _mm_mul_ps(
                    _mm_cvtepi32_ps(_mm_srai_epi32(_mm_slli_epi32(raw, 16), 16)),
                    scale,
                );
                let odd = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(raw, 16)), scale);
                squares = [
                    _mm_add_ps(squares[0], _mm_mul_ps(even, even)),
                    _mm_add_ps(squares[1], _mm_mul_ps(odd, odd)),
                ];
                let magnitude = _mm_max_ps(
                    _mm_sub_ps(zero, even),
                    _mm_max_ps(even, _mm_max_ps(_mm_sub_ps(zero, odd), odd)),
                );
                peak = _mm_max_ps(peak, magnitude);
                let full = _mm_or_si128(_mm_cmpeq_epi16(raw, low_full), _mm_cmpeq_epi16(raw, high_full));
                // One 16-bit result sets two bytes of the mask.
                measure.clipped += _mm_movemask_epi8(full).count_ones() / 2;
                measure.samples += 8;
                since += 8;
                if since >= FLUSH_SAMPLES {
                    measure.sum_sq += fold(squares);
                    squares = [zero; 2];
                    since = 0;
                }
            }
            measure.sum_sq += fold(squares);
            measure.peak = maximum128(peak);
            measure.merge(scalar::i16(tail));
            measure
        }
    }

    /// The `I16` block with sixteen samples per 256-bit register.
    ///
    /// # Safety
    ///
    /// The caller must have verified AVX2 support.
    #[target_feature(enable = "avx2")]
    unsafe fn i16_avx2(data: &[u8]) -> Measure {
        use core::arch::x86_64::{
            _mm_cmpeq_epi16, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi16, _mm256_add_ps,
            _mm256_cvtepi16_epi32, _mm256_cvtepi32_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_set1_ps,
            _mm256_setzero_ps, _mm256_sub_ps,
        };

        let body_len = data.len() - data.len() % 32;
        let (body, tail) = data.split_at(body_len);
        // SAFETY: the caller guarantees AVX2; each pair of loads reads one
        // whole 32-byte chunk of `data`.
        unsafe {
            let mut measure = Measure::empty();
            let scale = _mm256_set1_ps(I16_SCALE);
            let zero = _mm256_setzero_ps();
            let low_full = _mm_set1_epi16(-32_768);
            let high_full = _mm_set1_epi16(32_767);
            let mut squares = [zero; 2];
            let mut peak = zero;
            let mut since = 0usize;
            for chunk in body.chunks(32) {
                let first = _mm_loadu_si128(chunk.as_ptr().cast());
                let second = _mm_loadu_si128(chunk.as_ptr().add(16).cast());
                let a = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(first)), scale);
                let b = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(second)), scale);
                squares = [
                    _mm256_add_ps(squares[0], _mm256_mul_ps(a, a)),
                    _mm256_add_ps(squares[1], _mm256_mul_ps(b, b)),
                ];
                let magnitude = _mm256_max_ps(
                    _mm256_sub_ps(zero, a),
                    _mm256_max_ps(a, _mm256_max_ps(_mm256_sub_ps(zero, b), b)),
                );
                peak = _mm256_max_ps(peak, magnitude);
                // The full-scale test stays in the integer domain: the
                // positive extreme of the format is just under unity, so a
                // floating point comparison would never reach it.
                for raw in [first, second] {
                    let full = _mm_or_si128(_mm_cmpeq_epi16(raw, low_full), _mm_cmpeq_epi16(raw, high_full));
                    measure.clipped += _mm_movemask_epi8(full).count_ones() / 2;
                }
                measure.samples += 16;
                since += 16;
                if since >= FLUSH_SAMPLES {
                    measure.sum_sq += fold256(squares);
                    squares = [zero; 2];
                    since = 0;
                }
            }
            measure.sum_sq += fold256(squares);
            measure.peak = maximum256(peak);
            measure.merge(scalar::i16(tail));
            measure
        }
    }

    /// The `F32` block with four samples per 128-bit register.
    ///
    /// # Safety
    ///
    /// The caller must have verified SSE2 support.
    #[target_feature(enable = "sse2")]
    unsafe fn f32_sse2(data: &[u8]) -> Measure {
        use core::arch::x86_64::{
            _mm_add_ps, _mm_and_ps, _mm_cmpeq_ps, _mm_cmpge_ps, _mm_loadu_ps, _mm_max_ps, _mm_movemask_ps,
            _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps, _mm_sub_ps,
        };

        let body_len = data.len() - data.len() % 16;
        let (body, tail) = data.split_at(body_len);
        // SAFETY: the caller guarantees SSE2; each load reads one whole
        // 16-byte chunk of `data`.
        unsafe {
            let mut measure = Measure::empty();
            let zero = _mm_setzero_ps();
            let one = _mm_set1_ps(1.0);
            let mut squares = zero;
            let mut peak = zero;
            let mut since = 0usize;
            for chunk in body.chunks(16) {
                let raw = _mm_loadu_ps(chunk.as_ptr().cast());
                // A finite sample minus itself is a zero, so an equality test
                // against zero is true exactly for the finite lanes; the mask
                // replaces `NaN` and infinity with zero.
                let sample = _mm_and_ps(raw, _mm_cmpeq_ps(_mm_sub_ps(raw, raw), zero));
                squares = _mm_add_ps(squares, _mm_mul_ps(sample, sample));
                let magnitude = _mm_max_ps(_mm_sub_ps(zero, sample), sample);
                peak = _mm_max_ps(peak, magnitude);
                measure.clipped += _mm_movemask_ps(_mm_cmpge_ps(magnitude, one)).count_ones();
                measure.samples += 4;
                since += 4;
                if since >= FLUSH_SAMPLES {
                    measure.sum_sq += fold([squares, zero]);
                    squares = zero;
                    since = 0;
                }
            }
            measure.sum_sq += fold([squares, zero]);
            measure.peak = maximum128(peak);
            measure.merge(scalar::f32(tail));
            measure
        }
    }

    /// The `F32` block with eight samples per 256-bit register.
    ///
    /// # Safety
    ///
    /// The caller must have verified AVX2 support.
    #[target_feature(enable = "avx2")]
    unsafe fn f32_avx2(data: &[u8]) -> Measure {
        use core::arch::x86_64::{
            _CMP_EQ_OQ, _CMP_GE_OQ, _mm256_add_ps, _mm256_and_ps, _mm256_cmp_ps, _mm256_loadu_ps,
            _mm256_max_ps, _mm256_movemask_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps,
            _mm256_sub_ps,
        };

        let body_len = data.len() - data.len() % 32;
        let (body, tail) = data.split_at(body_len);
        // SAFETY: the caller guarantees AVX2; each load reads one whole
        // 32-byte chunk of `data`.
        unsafe {
            let mut measure = Measure::empty();
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);
            let mut squares = zero;
            let mut peak = zero;
            let mut since = 0usize;
            for chunk in body.chunks(32) {
                let raw = _mm256_loadu_ps(chunk.as_ptr().cast());
                let finite = _mm256_cmp_ps(_mm256_sub_ps(raw, raw), zero, _CMP_EQ_OQ);
                let sample = _mm256_and_ps(raw, finite);
                squares = _mm256_add_ps(squares, _mm256_mul_ps(sample, sample));
                let magnitude = _mm256_max_ps(_mm256_sub_ps(zero, sample), sample);
                peak = _mm256_max_ps(peak, magnitude);
                let full = _mm256_cmp_ps(magnitude, one, _CMP_GE_OQ);
                measure.clipped += _mm256_movemask_ps(full).count_ones();
                measure.samples += 8;
                since += 8;
                if since >= FLUSH_SAMPLES {
                    measure.sum_sq += fold256([squares, zero]);
                    squares = zero;
                    since = 0;
                }
            }
            measure.sum_sq += fold256([squares, zero]);
            measure.peak = maximum256(peak);
            measure.merge(scalar::f32(tail));
            measure
        }
    }

    /// The total of two 128-bit square accumulators, in `f64`.
    ///
    /// # Safety
    ///
    /// The caller must have verified SSE2 support.
    #[target_feature(enable = "sse2")]
    unsafe fn fold(squares: [__m128; 2]) -> f64 {
        use core::arch::x86_64::{_mm_add_ps, _mm_storeu_ps};

        let mut lanes = [0.0f32; 4];
        // SAFETY: the caller guarantees SSE2; the store writes four floats
        // into the local array.
        unsafe {
            _mm_storeu_ps(lanes.as_mut_ptr(), _mm_add_ps(squares[0], squares[1]));
        }
        total(&lanes)
    }

    /// The total of two 256-bit square accumulators, in `f64`.
    ///
    /// # Safety
    ///
    /// The caller must have verified AVX2 support.
    #[target_feature(enable = "avx2")]
    unsafe fn fold256(squares: [__m256; 2]) -> f64 {
        use core::arch::x86_64::{_mm256_add_ps, _mm256_storeu_ps};

        let mut lanes = [0.0f32; 8];
        // SAFETY: the caller guarantees AVX2; the store writes eight floats
        // into the local array.
        unsafe {
            _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(squares[0], squares[1]));
        }
        total(&lanes)
    }

    /// The largest lane of a 128-bit vector.
    ///
    /// # Safety
    ///
    /// The caller must have verified SSE2 support.
    #[target_feature(enable = "sse2")]
    unsafe fn maximum128(value: __m128) -> f32 {
        use core::arch::x86_64::{_mm_cvtss_f32, _mm_max_ps, _mm_shuffle_ps};

        // `0b01_00_11_10` pairs lane `i` with lane `i + 2`, and the second
        // shuffle pairs lane 0 with lane 1.
        let pair = _mm_max_ps(value, _mm_shuffle_ps(value, value, 0b01_00_11_10));
        _mm_cvtss_f32(_mm_max_ps(pair, _mm_shuffle_ps(pair, pair, 0b00_00_00_01)))
    }

    /// The largest lane of a 256-bit vector.
    ///
    /// # Safety
    ///
    /// The caller must have verified AVX2 support.
    #[target_feature(enable = "avx2")]
    unsafe fn maximum256(value: __m256) -> f32 {
        use core::arch::x86_64::{
            _mm_cvtss_f32, _mm_max_ps, _mm_shuffle_ps, _mm256_castps256_ps128, _mm256_extractf128_ps,
        };

        let halves = _mm_max_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps::<1>(value));
        let pair = _mm_max_ps(halves, _mm_shuffle_ps(halves, halves, 0b01_00_11_10));
        _mm_cvtss_f32(_mm_max_ps(pair, _mm_shuffle_ps(pair, pair, 0b00_00_00_01)))
    }

    /// Adds the lanes in `f64`, which is where the reference keeps its total.
    fn total(lanes: &[f32]) -> f64 {
        lanes
            .iter()
            .fold(0.0f64, |sum, &lane| sum + f64::from(lane))
    }
}
#[cfg(test)]
mod tests {
    use libm::sqrt;

    use codevar_kuitui_control::{AudioFormat, SampleFormat};

    use super::{AudioLevels, I16_SCALE};

    fn layout(sample_rate: u32, channels: u16, sample_format: SampleFormat) -> AudioFormat {
        AudioFormat {
            sample_rate,
            channels,
            sample_format,
        }
    }

    /// A deterministic sample source, so a failure is reproducible.
    struct Noise(u64);

    impl Noise {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
    }

    impl Iterator for Noise {
        type Item = i16;

        fn next(&mut self) -> Option<i16> {
            // A 64-bit xorshift; the top bits of the state are the strongest,
            // so the sample comes from there.
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            Some((self.0 >> 40) as i16)
        }
    }

    /// The reference the accelerated paths are measured against.
    fn reference(samples: &[i16]) -> AudioLevels {
        let mut sum_sq = 0.0f64;
        let mut peak = 0.0f32;
        let mut clipped = 0u32;
        let mut crossings = 0u32;
        let mut previous: Option<bool> = None;
        for &raw in samples {
            let sample = f32::from(raw) * I16_SCALE;
            sum_sq += f64::from(sample) * f64::from(sample);
            let magnitude = if sample < 0.0 { -sample } else { sample };
            if magnitude > peak {
                peak = magnitude;
            }
            clipped += u32::from(raw == i16::MIN || raw == i16::MAX);
            let negative = sample < 0.0;
            if let Some(was) = previous {
                crossings += u32::from(negative != was);
            }
            previous = Some(negative);
        }
        AudioLevels {
            rms: sqrt(sum_sq / samples.len() as f64) as f32,
            peak,
            clipped,
            samples: samples.len() as u32,
            crossings,
        }
    }

    fn levels_of(samples: &[i16], rate: u32) -> AudioLevels {
        let mut data = Vec::with_capacity(samples.len() * 2);
        for &raw in samples {
            data.extend_from_slice(&raw.to_le_bytes());
        }
        AudioLevels::analyze(layout(rate, 1, SampleFormat::I16), &data)
    }

    #[test]
    fn an_empty_block_is_silent() {
        assert_eq!(
            AudioLevels::analyze(layout(48_000, 1, SampleFormat::I16), &[]),
            AudioLevels::SILENT
        );
        assert_eq!(AudioLevels::default(), AudioLevels::SILENT);
    }

    #[test]
    fn a_block_longer_than_the_flush_window_matches_the_reference() {
        // 1000 samples is not a whole number of 8- or 16-sample vectors, so
        // the tail path is taken as well.
        let samples: Vec<i16> = Noise::new(0x5eed).take(1_000).collect();
        let measured = levels_of(&samples, 48_000);
        let expected = reference(&samples);
        assert_eq!(measured.peak, expected.peak);
        assert_eq!(measured.samples, expected.samples);
        assert_eq!(measured.clipped, expected.clipped);
        assert_eq!(measured.crossings, expected.crossings);
        let drift = (measured.rms - expected.rms).abs() / expected.rms;
        assert!(
            drift < 1e-5,
            "rms drift {drift} with {measured:?} against {expected:?}"
        );
    }

    #[test]
    fn a_short_block_measured_without_a_whole_vector_still_counts() {
        let samples: Vec<i16> = Noise::new(0x1234).take(7).collect();
        assert_eq!(levels_of(&samples, 8_000), reference(&samples));
    }

    #[test]
    fn full_scale_counts_once_for_each_extreme() {
        let samples = [i16::MIN, 1_000, i16::MAX, 0, i16::MAX];
        let measured = levels_of(&samples, 48_000);
        assert_eq!(measured.clipped, 3);
        assert!(measured.is_clipping());
        assert_eq!(measured.peak, 1.0);
        assert_eq!(measured.peak_dbfs(), 0.0);
    }

    #[test]
    fn a_non_finite_float_sample_reads_as_zero() {
        let mut data = Vec::new();
        for sample in [0.5f32, f32::NAN, f32::NEG_INFINITY, -0.25, 1.0] {
            data.extend_from_slice(&sample.to_le_bytes());
        }
        let measured = AudioLevels::analyze(layout(16_000, 1, SampleFormat::F32), &data);
        assert_eq!(measured.samples, 5);
        assert_eq!(measured.peak, 1.0);
        // Only the last sample reaches full scale: the infinite one is gone
        // before the comparison.
        assert_eq!(measured.clipped, 1);
        assert!(measured.rms.is_finite());
        assert_eq!(
            measured.zero_crossings_per_second(layout(16_000, 1, SampleFormat::F32)),
            Some(6_400.0)
        );
    }

    #[test]
    fn an_eight_bit_block_is_biased_by_half_a_scale() {
        let measured = AudioLevels::analyze(layout(8_000, 1, SampleFormat::U8), &[128, 0, 255, 128, 64]);
        assert_eq!(measured.samples, 5);
        assert_eq!(measured.peak, 1.0);
        assert_eq!(measured.clipped, 2);
        // Zero, the two full-scale ends and the half-scale dip alternate
        // three times across the block.
        assert_eq!(measured.crossings, 3);
    }

    #[test]
    fn the_duration_and_the_crossing_rate_follow_the_sample_rate() {
        let samples: Vec<i16> = Noise::new(0xabcd).take(480).collect();
        let loud = levels_of(&samples, 48_000);
        let quiet = levels_of(&samples, 8_000);
        assert_eq!(
            loud.duration_micros(layout(48_000, 1, SampleFormat::I16)),
            Some(10_000)
        );
        assert_eq!(
            quiet.duration_micros(layout(8_000, 1, SampleFormat::I16)),
            Some(60_000)
        );
        // The same samples at six times the rate cross six times as often.
        let fast = loud.zero_crossings_per_second(layout(48_000, 1, SampleFormat::I16));
        let slow = quiet.zero_crossings_per_second(layout(8_000, 1, SampleFormat::I16));
        assert!(
            fast.is_some_and(|rate| slow.is_some_and(|other| (rate - other * 6.0).abs() < 1.0)),
            "{fast:?} against {slow:?}"
        );
    }

    #[test]
    fn stereo_samples_count_frames_not_samples() {
        let samples: Vec<i16> = Noise::new(0x77).take(64).collect();
        let measured = levels_of(&samples, 48_000);
        let stereo = layout(48_000, 2, SampleFormat::I16);
        assert_eq!(measured.frames(stereo), Some(32));
        assert_eq!(measured.frames(layout(48_000, 0, SampleFormat::I16)), None);
        assert_eq!(measured.duration_micros(stereo), Some(666));
    }

    #[test]
    fn a_square_wave_reports_its_real_crossing_rate() {
        let samples = [-1_000i16, 1_000, -1_000, 1_000, -1_000, 1_000, -1_000, 1_000];
        let measured = levels_of(&samples, 8_000);
        let format = layout(8_000, 1, SampleFormat::I16);
        assert_eq!(measured.crossings, 7);
        assert_eq!(measured.duration_micros(format), Some(1_000));
        assert_eq!(measured.zero_crossings_per_second(format), Some(7_000.0));
    }

    #[test]
    fn a_format_without_a_rate_or_a_channel_has_no_time() {
        let samples = [-1_000i16, 1_000];
        let measured = levels_of(&samples, 8_000);
        assert_eq!(measured.duration_micros(layout(0, 1, SampleFormat::I16)), None);
        assert_eq!(measured.frames(layout(8_000, 0, SampleFormat::I16)), None);
        assert_eq!(
            measured.zero_crossings_per_second(layout(8_000, 0, SampleFormat::I16)),
            None
        );
    }
}
