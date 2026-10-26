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

//! The shared microphone level and the snapshot a widget reads.
//!
//! A microphone is not a discrete device like a keyboard: it produces
//! blocks of samples several thousand times per second. Keeping every block
//! would allocate on the audio path and hand the UI more data than it can
//! draw, so the state keeps what a level meter needs — root mean square,
//! peak, clipping — and the handler receives the block itself while it is
//! still in memory.
//!
//! [`MicrophoneState`] follows the same protocol as the other states: one
//! writer (the pump) bumps a sequence counter, updates plain atomics and
//! bumps it again, and a reader copies between two equal, even values. The
//! samples are analyzed *before* the writer opens that window, so decoding a
//! large block never blocks a reader.

use core::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

use codevar_kuitui_control::{AudioEvent, AudioFormat, DeviceId, SampleFormat, Timestamp};
use libm::sqrt;

/// Bit of [`MicrophoneState::flags`] that marks a received block.
const HAS_EVENT: u8 = 1 << 0;
/// Bit of [`MicrophoneState::flags`] set when the last block clipped.
const CLIPPING: u8 = 1 << 1;

/// Encoding of [`SampleFormat::U8`] in the state.
const FORMAT_U8: u8 = 0;
/// Encoding of [`SampleFormat::I16`] in the state.
const FORMAT_I16: u8 = 1;
/// Encoding of [`SampleFormat::F32`] in the state.
const FORMAT_F32: u8 = 2;

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
}

impl AudioLevels {
    /// The level of an empty or silent block.
    pub const SILENT: AudioLevels = AudioLevels {
        rms: 0.0,
        peak: 0.0,
        clipped: 0,
        samples: 0,
    };

    /// Measures one block of raw samples.
    ///
    /// The bytes are read little-endian, matching [`AudioFormat`]. A block
    /// whose length is not a whole number of samples loses its trailing
    /// bytes; a non-finite `F32` sample contributes zero, so one corrupt
    /// value cannot turn the whole block into `NaN`.
    #[must_use]
    pub fn analyze(format: AudioFormat, data: &[u8]) -> Self {
        let mut sum_sq = 0.0f64;
        let mut levels = Self::SILENT;
        match format.sample_format {
            SampleFormat::U8 => {
                for &byte in data {
                    let sample = (f32::from(byte) - 128.0) / 128.0;
                    let clipped = byte == 0 || byte == u8::MAX;
                    measure(sample, clipped, &mut sum_sq, &mut levels);
                }
            }
            SampleFormat::I16 => {
                for pair in data.chunks_exact(2) {
                    let sample = i16::from_le_bytes([pair[0], pair[1]]);
                    let clipped = sample == i16::MIN || sample == i16::MAX;
                    measure(f32::from(sample) / 32_768.0, clipped, &mut sum_sq, &mut levels);
                }
            }
            SampleFormat::F32 => {
                for quad in data.chunks_exact(4) {
                    let sample = f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]);
                    let sample = if sample.is_finite() { sample } else { 0.0 };
                    let clipped = sample >= 1.0 || sample <= -1.0;
                    measure(sample, clipped, &mut sum_sq, &mut levels);
                }
            }
        }
        if levels.samples > 0 {
            levels.rms = sqrt(sum_sq / f64::from(levels.samples)) as f32;
        }
        levels
    }

    /// Whether the block reached full scale at least once.
    #[must_use]
    pub const fn is_clipping(self) -> bool {
        self.clipped > 0
    }
}

/// Folds one sample into the running level.
#[inline]
fn measure(sample: f32, clipped: bool, sum_sq: &mut f64, levels: &mut AudioLevels) {
    let magnitude = if sample < 0.0 { -sample } else { sample };
    if magnitude > levels.peak {
        levels.peak = magnitude;
    }
    if clipped {
        levels.clipped = levels.clipped.saturating_add(1);
    }
    *sum_sq += f64::from(sample) * f64::from(sample);
    levels.samples = levels.samples.saturating_add(1);
}

/// A consistent copy of the microphone state.
///
/// A widget reads this when the level, the format and the counts must
/// describe the same block. The snapshot does not carry the samples
/// themselves; those are only available in the handler's block update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MicrophoneSnapshot {
    /// When the backend captured the block.
    pub timestamp: Timestamp,
    /// The microphone that captured it, or `None` before the first block.
    pub device: Option<DeviceId>,
    /// The layout of the block, or `None` before the first block.
    pub format: Option<AudioFormat>,
    /// The level of the last block.
    pub levels: AudioLevels,
    /// Number of blocks applied since the state was created.
    pub blocks: u64,
    /// Number of samples applied since the state was created.
    pub total_samples: u64,
    /// The sequence value the snapshot was taken at.
    pub sequence: u64,
}

impl MicrophoneSnapshot {
    /// Whether the last block reached full scale.
    #[must_use]
    pub const fn is_clipping(&self) -> bool {
        self.levels.clipped > 0
    }

    /// The sample rate of the last block, or `None` before the first one.
    #[must_use]
    pub const fn sample_rate(&self) -> Option<u32> {
        match self.format {
            Some(format) => Some(format.sample_rate),
            None => None,
        }
    }

    /// The channel count of the last block, or `None` before the first one.
    #[must_use]
    pub const fn channels(&self) -> Option<u16> {
        match self.format {
            Some(format) => Some(format.channels),
            None => None,
        }
    }
}

/// The microphone state shared by the pump and every `MicrophoneHandler`.
///
/// The write side is [`MicrophoneState::apply`], meant for the single
/// `InputPump` that owns the stream. Readers use
/// [`MicrophoneState::snapshot`] for a consistent copy or the individual
/// accessors when one field is enough; a level meter that only draws the
/// loudness reads [`MicrophoneState::rms`] and never touches the rest.
pub struct MicrophoneState {
    /// Sequence counter; odd while a writer is active.
    sequence: AtomicU64,
    /// Root mean square of the last block, as `f32` bits.
    rms: AtomicU32,
    /// Peak of the last block, as `f32` bits.
    peak: AtomicU32,
    /// Clipped samples in the last block.
    clipped: AtomicU32,
    /// Samples in the last block.
    samples: AtomicU32,
    /// Samples applied since the state was created.
    total_samples: AtomicU64,
    /// Blocks applied since the state was created.
    blocks: AtomicU64,
    /// Sample rate of the last block.
    sample_rate: AtomicU32,
    /// Channel count of the last block.
    channels: AtomicU32,
    /// [`SampleFormat`] of the last block.
    format: AtomicU8,
    /// [`HAS_EVENT`] and [`CLIPPING`].
    flags: AtomicU8,
    /// The microphone that captured the last block.
    device: AtomicU64,
    /// Time of the last block.
    timestamp: AtomicU64,
}

impl MicrophoneState {
    /// Creates an empty state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sequence: AtomicU64::new(0),
            rms: AtomicU32::new(0),
            peak: AtomicU32::new(0),
            clipped: AtomicU32::new(0),
            samples: AtomicU32::new(0),
            total_samples: AtomicU64::new(0),
            blocks: AtomicU64::new(0),
            sample_rate: AtomicU32::new(0),
            channels: AtomicU32::new(0),
            format: AtomicU8::new(FORMAT_U8),
            flags: AtomicU8::new(0),
            device: AtomicU64::new(0),
            timestamp: AtomicU64::new(0),
        }
    }

    /// Records one block of samples.
    ///
    /// This is the write side of the state and is meant for the single
    /// `InputPump` that owns the stream. The block is analyzed before the
    /// sequence counter moves, so the critical section stays a handful of
    /// atomic stores regardless of the block size.
    pub fn apply(&self, event: &AudioEvent, timestamp: Timestamp) {
        let levels = AudioLevels::analyze(event.format, &event.data);
        self.apply_levels(event.format, event.device, levels, timestamp);
    }

    /// Stores a block that has already been analyzed.
    pub(crate) fn apply_levels(
        &self,
        format: AudioFormat,
        device: DeviceId,
        levels: AudioLevels,
        timestamp: Timestamp,
    ) {
        self.sequence.fetch_add(1, Ordering::AcqRel);
        self.rms
            .store(levels.rms.to_bits(), Ordering::Relaxed);
        self.peak
            .store(levels.peak.to_bits(), Ordering::Relaxed);
        self.clipped
            .store(levels.clipped, Ordering::Relaxed);
        self.samples
            .store(levels.samples, Ordering::Relaxed);
        self.total_samples
            .fetch_add(u64::from(levels.samples), Ordering::Relaxed);
        self.blocks.fetch_add(1, Ordering::Relaxed);
        self.sample_rate
            .store(format.sample_rate, Ordering::Relaxed);
        self.channels
            .store(u32::from(format.channels), Ordering::Relaxed);
        self.format
            .store(encode_format(format.sample_format), Ordering::Relaxed);
        self.device.store(device.get(), Ordering::Relaxed);
        self.flags.fetch_or(HAS_EVENT, Ordering::Relaxed);
        if levels.is_clipping() {
            self.flags.fetch_or(CLIPPING, Ordering::Relaxed);
        } else {
            self.flags.fetch_and(!CLIPPING, Ordering::Relaxed);
        }
        self.timestamp
            .store(timestamp.micros(), Ordering::Relaxed);
        self.sequence.fetch_add(1, Ordering::Release);
    }

    /// Copies every field between two equal, even sequence values.
    ///
    /// The copy is consistent even when the pump is applying a block on
    /// another thread; the loop retries while a write is in progress.
    #[must_use]
    pub fn snapshot(&self) -> MicrophoneSnapshot {
        loop {
            let before = self.sequence.load(Ordering::Acquire);
            if before & 1 != 0 {
                core::hint::spin_loop();
                continue;
            }
            let flags = self.flags.load(Ordering::Relaxed);
            let format = format_of(
                decode_format(self.format.load(Ordering::Relaxed)),
                self.sample_rate.load(Ordering::Relaxed),
                self.channels.load(Ordering::Relaxed),
                flags & HAS_EVENT != 0,
            );
            let snapshot = MicrophoneSnapshot {
                timestamp: Timestamp::from_micros(self.timestamp.load(Ordering::Relaxed)),
                device: (flags & HAS_EVENT != 0).then(|| DeviceId::new(self.device.load(Ordering::Relaxed))),
                format,
                levels: AudioLevels {
                    rms: f32::from_bits(self.rms.load(Ordering::Relaxed)),
                    peak: f32::from_bits(self.peak.load(Ordering::Relaxed)),
                    clipped: self.clipped.load(Ordering::Relaxed),
                    samples: self.samples.load(Ordering::Relaxed),
                },
                blocks: self.blocks.load(Ordering::Relaxed),
                total_samples: self.total_samples.load(Ordering::Relaxed),
                sequence: before,
            };
            if self.sequence.load(Ordering::Acquire) == before {
                return snapshot;
            }
        }
    }

    /// The loudness of the last block.
    #[must_use]
    pub fn rms(&self) -> f32 {
        f32::from_bits(self.rms.load(Ordering::Relaxed))
    }

    /// The largest absolute sample of the last block.
    #[must_use]
    pub fn peak(&self) -> f32 {
        f32::from_bits(self.peak.load(Ordering::Relaxed))
    }

    /// Whether the last block reached full scale.
    #[must_use]
    pub fn is_clipping(&self) -> bool {
        self.flags.load(Ordering::Relaxed) & CLIPPING != 0
    }

    /// The number of clipped samples in the last block.
    #[must_use]
    pub fn clipped_samples(&self) -> u32 {
        self.clipped.load(Ordering::Relaxed)
    }

    /// The number of samples in the last block.
    #[must_use]
    pub fn block_samples(&self) -> u32 {
        self.samples.load(Ordering::Relaxed)
    }

    /// The number of samples applied since the state was created.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.total_samples.load(Ordering::Relaxed)
    }

    /// The number of blocks applied since the state was created.
    #[must_use]
    pub fn blocks(&self) -> u64 {
        self.blocks.load(Ordering::Relaxed)
    }

    /// The layout of the last block, or `None` before the first one.
    #[must_use]
    pub fn format(&self) -> Option<AudioFormat> {
        format_of(
            decode_format(self.format.load(Ordering::Relaxed)),
            self.sample_rate.load(Ordering::Relaxed),
            self.channels.load(Ordering::Relaxed),
            self.flags.load(Ordering::Relaxed) & HAS_EVENT != 0,
        )
    }

    /// The microphone that captured the last block.
    #[must_use]
    pub fn device(&self) -> Option<DeviceId> {
        (self.flags.load(Ordering::Relaxed) & HAS_EVENT != 0)
            .then(|| DeviceId::new(self.device.load(Ordering::Relaxed)))
    }

    /// The time of the last block.
    #[must_use]
    pub fn timestamp(&self) -> Timestamp {
        Timestamp::from_micros(self.timestamp.load(Ordering::Relaxed))
    }

    /// The current sequence value; even between blocks.
    #[must_use]
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::Acquire)
    }
}

impl Default for MicrophoneState {
    fn default() -> Self {
        Self::new()
    }
}

/// Encodes a sample format for the state.
#[inline]
const fn encode_format(format: SampleFormat) -> u8 {
    match format {
        SampleFormat::U8 => FORMAT_U8,
        SampleFormat::I16 => FORMAT_I16,
        SampleFormat::F32 => FORMAT_F32,
    }
}

/// Decodes a sample format written by [`encode_format`].
#[inline]
const fn decode_format(value: u8) -> Option<SampleFormat> {
    match value {
        FORMAT_U8 => Some(SampleFormat::U8),
        FORMAT_I16 => Some(SampleFormat::I16),
        FORMAT_F32 => Some(SampleFormat::F32),
        _ => None,
    }
}

/// Rebuilds the layout from its decoded parts, or `None` before a block.
#[inline]
const fn format_of(
    sample_format: Option<SampleFormat>,
    sample_rate: u32,
    channels: u32,
    received: bool,
) -> Option<AudioFormat> {
    if !received {
        return None;
    }
    match sample_format {
        Some(sample_format) => Some(AudioFormat {
            sample_rate,
            channels: channels as u16,
            sample_format,
        }),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioLevels, MicrophoneState};
    use codevar_kuitui_control::{AudioEvent, AudioFormat, DeviceId, SampleFormat, Timestamp};

    fn i16_event(samples: &[i16], channels: u16) -> AudioEvent {
        let mut data = Vec::with_capacity(samples.len() * 2);
        for sample in samples {
            data.extend_from_slice(&sample.to_le_bytes());
        }
        AudioEvent {
            device: DeviceId::new(9),
            format: AudioFormat {
                sample_rate: 48_000,
                channels,
                sample_format: SampleFormat::I16,
            },
            data,
        }
    }

    fn silence() -> AudioEvent {
        i16_event(&[0, 0, 0, 0], 1)
    }

    fn timestamp(micros: u64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    #[test]
    fn a_silent_block_has_no_level() {
        let levels = AudioLevels::analyze(silence().format, &silence().data);
        assert_eq!(levels.samples, 4);
        assert_eq!(levels.rms, 0.0);
        assert_eq!(levels.peak, 0.0);
        assert!(!levels.is_clipping());
    }

    #[test]
    fn a_square_wave_is_measured_in_root_mean_square() {
        let levels = AudioLevels::analyze(silence().format, &i16_event(&[16_384, -16_384], 1).data);
        assert_eq!(levels.samples, 2);
        assert!((levels.rms - 0.5).abs() < 1e-6);
        assert!((levels.peak - 0.5).abs() < 1e-6);
        assert!(!levels.is_clipping());
    }

    #[test]
    fn full_scale_samples_count_as_clipping() {
        let levels = AudioLevels::analyze(silence().format, &i16_event(&[i16::MAX, i16::MIN], 1).data);
        assert_eq!(levels.clipped, 2);
        assert!((levels.peak - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unsigned_and_float_blocks_are_decoded() {
        let u8_format = AudioFormat {
            sample_rate: 8_000,
            channels: 1,
            sample_format: SampleFormat::U8,
        };
        let u8_levels = AudioLevels::analyze(u8_format, &[255, 1]);
        assert_eq!(u8_levels.samples, 2);
        assert_eq!(u8_levels.clipped, 1);
        assert!(u8_levels.rms > 0.99);

        let f32_format = AudioFormat {
            sample_rate: 48_000,
            channels: 1,
            sample_format: SampleFormat::F32,
        };
        let mut data = Vec::new();
        for sample in [0.25f32, -0.25, 2.0] {
            data.extend_from_slice(&sample.to_le_bytes());
        }
        let f32_levels = AudioLevels::analyze(f32_format, &data);
        assert_eq!(f32_levels.samples, 3);
        assert_eq!(f32_levels.clipped, 1);
        assert!((f32_levels.peak - 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_trailing_partial_sample_is_ignored() {
        let event = i16_event(&[1_000, 2_000], 1);
        let levels = AudioLevels::analyze(event.format, &event.data[..3]);
        assert_eq!(levels.samples, 1);
    }

    #[test]
    fn non_finite_float_samples_contribute_zero() {
        let format = AudioFormat {
            sample_rate: 48_000,
            channels: 1,
            sample_format: SampleFormat::F32,
        };
        let mut data = Vec::new();
        for sample in [f32::NAN, f32::INFINITY, -1.0] {
            data.extend_from_slice(&sample.to_le_bytes());
        }
        let levels = AudioLevels::analyze(format, &data);
        assert_eq!(levels.samples, 3);
        assert_eq!(levels.clipped, 1);
        assert!((levels.rms - 0.577_350_3).abs() < 1e-5);
    }

    #[test]
    fn the_state_tracks_the_last_block_and_the_totals() {
        let state = MicrophoneState::new();
        assert_eq!(state.device(), None);
        assert_eq!(state.format(), None);
        assert!(!state.is_clipping());

        let first = i16_event(&[i16::MAX, i16::MIN], 2);
        state.apply(&first, timestamp(10));
        assert_eq!(state.blocks(), 1);
        assert_eq!(state.total_samples(), 2);
        assert_eq!(state.block_samples(), 2);
        assert_eq!(state.clipped_samples(), 2);
        assert!(state.is_clipping());
        assert_eq!(state.device(), Some(DeviceId::new(9)));
        let format = state.format();
        assert!(format.is_some_and(|format| format.channels == 2 && format.sample_rate == 48_000));
        assert_eq!(state.timestamp(), timestamp(10));

        let second = silence();
        state.apply(&second, timestamp(20));
        assert_eq!(state.blocks(), 2);
        assert_eq!(state.total_samples(), 6);
        assert!(!state.is_clipping());
        assert_eq!(state.rms(), 0.0);

        let snapshot = state.snapshot();
        assert_eq!(snapshot.blocks, 2);
        assert_eq!(snapshot.total_samples, 6);
        assert_eq!(snapshot.levels.samples, 4);
        assert!(!snapshot.is_clipping());
        assert_eq!(snapshot.sample_rate(), Some(48_000));
        assert_eq!(snapshot.channels(), Some(1));
        assert_eq!(snapshot.device, Some(DeviceId::new(9)));
        assert_eq!(snapshot.timestamp, timestamp(20));
        assert_eq!(snapshot.sequence, state.sequence());
        assert_eq!(snapshot.sequence & 1, 0);
    }
}
