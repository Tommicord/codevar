//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! FcWare-backed compression for history transaction units (TXUs).
//!
//! Large cold TXU payloads (`buffer` / `ppbuff`) are compressed in-memory
//! under memory pressure or when history grows large. Wire serialization
//! always decompresses first so the bytecode format stays unchanged.

use crate::base::base_memory::MemoryPressure;
use crate::edit::wredit_history::History;
use crate::edit::wredit_history_txu::HistoryTXU;
use crate::fcware::compression::{Codec, compress, decompress};
use crate::fcware::compression_error::Error as CompressionError;
use std::sync::Arc;

/// Minimum payload size (bytes) before a TXU blob is considered compressible.
pub const TXU_BLOB_COMPRESS_THRESHOLD: usize = 4_096;
/// Lower threshold used under high / critical memory pressure.
pub const TXU_BLOB_COMPRESS_THRESHOLD_AGGRESSIVE: usize = 1_024;
/// Compress cold TXUs when total resident history exceeds this size.
pub const HISTORY_TOTAL_COMPRESS_THRESHOLD: usize = 256 * 1024;
/// Number of newest TXUs kept uncompressed for fast undo/redo.
pub const KEEP_RECENT_UNCOMPRESSED: usize = 8;

/// Result of a TXU compress / decompress attempt.
pub type TxuCompressResult<T> = Result<T, CompressionError>;

impl HistoryTXU {
    /// Estimated resident bytes for this TXU (compressed or plain).
    #[must_use]
    pub fn estimated_resident_bytes(&self) -> usize {
        let mut total = self.buffer().len()
            + self
                .delta_offset()
                .len()
                .saturating_mul(size_of::<usize>())
            + self.hash().len()
            + size_of::<Self>();
        let mut delta = Some(self.delta());
        while let Some(d) = delta {
            total = total.saturating_add(d.ppbuff().len());
            total = total.saturating_add(64);
            delta = d.next().map(std::convert::AsRef::as_ref);
        }
        total
    }

    /// Whether any payload blob is currently FcWare-compressed.
    #[inline]
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.payload_compressed()
    }

    /// Uncompressed payload size estimate used for compress eligibility.
    #[must_use]
    pub fn uncompressed_payload_bytes(&self) -> usize {
        if self.is_compressed() {
            // Prefer logical buffer length plus current ppbuff storage.
            return self
                .buffer_length()
                .saturating_add(self.delta().ppbuff().len());
        }
        self.buffer()
            .len()
            .saturating_add(self.delta().ppbuff().len())
    }

    /// Whether this TXU should be compressed at the given `blob_threshold`.
    #[inline]
    #[must_use]
    pub fn should_compress(&self, blob_threshold: usize) -> bool {
        !self.is_compressed() && self.uncompressed_payload_bytes() >= blob_threshold
    }

    /// Compresses large `buffer` / `ppbuff` blobs with FcWare Dysu.
    ///
    /// Returns `true` if any blob was compressed. Failures leave the TXU
    /// unchanged and return `Ok(false)` after logging when compression is
    /// skipped for size; codec errors are returned to the caller.
    ///
    /// # Errors
    ///
    /// Returns a [`CompressionError`] when FcWare encoding fails.
    pub fn try_compress(&mut self, blob_threshold: usize) -> TxuCompressResult<bool> {
        if self.is_compressed() || !self.should_compress(blob_threshold) {
            return Ok(false);
        }

        let codec = Codec::dysu_default();
        let mut changed = false;

        if self.buffer().len() >= blob_threshold {
            match compress(self.buffer(), codec) {
                Ok(frame) if frame.len() < self.buffer().len() => {
                    *self.buffer_mut() = frame;
                    changed = true;
                }
                Ok(_) => {}
                Err(err) => {
                    return Err(err);
                }
            }
        }
        let ppbuff_len = self.delta().ppbuff().len();
        if ppbuff_len >= blob_threshold {
            let plain = self.delta().ppbuff().clone();
            match compress(&plain, codec) {
                Ok(frame) if frame.len() < plain.len() => {
                    *self.delta_mut().ppbuff_mut() = frame;
                    changed = true;
                }
                Ok(_) => {}
                Err(err) => {
                    // Roll back buffer compression if ppbuff failed mid-way.
                    if changed {
                        let _ = self.ensure_decompressed();
                    }
                    return Err(err);
                }
            }
        }
        if changed {
            self.set_payload_compressed(true);
        }
        Ok(changed)
    }

    /// Ensures `buffer` and `ppbuff` are plain (decompresses if needed).
    ///
    /// # Errors
    ///
    /// Returns a [`CompressionError`] when an FcWare frame cannot be decoded.
    pub fn ensure_decompressed(&mut self) -> TxuCompressResult<()> {
        if !self.is_compressed() {
            return Ok(());
        }
        if looks_like_fcware_frame(self.buffer()) {
            let plain = decompress(self.buffer())?;
            *self.buffer_length_mut() = plain.len();
            *self.buffer_mut() = plain;
        }
        if looks_like_fcware_frame(self.delta().ppbuff()) {
            let plain = decompress(self.delta().ppbuff())?;
            *self.delta_mut().ppbuff_mut() = plain;
        }
        // Linked next deltas may also be compressed when stored as nested TXUs
        // in memory; walk and decompress.
        let mut cursor = self.delta_mut().next.as_mut();
        while let Some(next) = cursor {
            if looks_like_fcware_frame(next.ppbuff()) {
                match decompress(next.ppbuff()) {
                    Ok(plain) => *next.ppbuff_mut() = plain,
                    Err(err) => {
                        return Err(err);
                    }
                }
            }
            cursor = next.next.as_mut();
        }

        self.set_payload_compressed(false);
        Ok(())
    }
}

impl History {
    /// Estimated resident bytes across construct buffer and all TXUs.
    #[must_use]
    pub fn estimated_resident_bytes(&self) -> usize {
        let mut total = self.construct().len().saturating_add(32);
        if let Ok(txus) = self.txus().read() {
            for txu in txus.iter() {
                total = total.saturating_add(txu.estimated_resident_bytes());
            }
        }
        for txu in self.pending_changes() {
            total = total.saturating_add(txu.estimated_resident_bytes());
        }
        total
    }

    /// Blob size threshold for the given pressure level.
    #[inline]
    #[must_use]
    pub fn blob_threshold_for(pressure: MemoryPressure) -> usize {
        match pressure {
            MemoryPressure::Normal | MemoryPressure::Elevated => {
                TXU_BLOB_COMPRESS_THRESHOLD
            }
            MemoryPressure::High | MemoryPressure::Critical => {
                TXU_BLOB_COMPRESS_THRESHOLD_AGGRESSIVE
            }
        }
    }

    /// Compresses cold TXUs when history is large or memory pressure is high.
    ///
    /// Keeps the newest [`KEEP_RECENT_UNCOMPRESSED`] TXUs plain for undo latency.
    /// Returns the number of TXUs that were newly compressed.
    pub fn maybe_compress_under_pressure(&mut self, pressure: MemoryPressure) -> usize {
        let total = self.estimated_resident_bytes();
        let large_history = total >= HISTORY_TOTAL_COMPRESS_THRESHOLD;
        if pressure == MemoryPressure::Normal && !large_history {
            return 0;
        }

        let threshold = Self::blob_threshold_for(pressure);
        let keep_recent = match pressure {
            MemoryPressure::Critical => 2,
            MemoryPressure::High => 4,
            _ => KEEP_RECENT_UNCOMPRESSED,
        };

        self.compress_cold_txus(threshold, keep_recent)
    }

    /// Compresses TXUs older than `keep_recent` that exceed `blob_threshold`.
    pub fn compress_cold_txus(
        &mut self,
        blob_threshold: usize,
        keep_recent: usize,
    ) -> usize {
        let Ok(mut guard) = self.txus().write() else {
            return 0;
        };
        let len = guard.len();
        if len <= keep_recent {
            return 0;
        }
        let cold_end = len.saturating_sub(keep_recent);
        let mut vec = guard.to_vec();
        let mut compressed_count = 0usize;
        for txu in &mut vec[..cold_end] {
            match txu.try_compress(blob_threshold) {
                Ok(true) => compressed_count = compressed_count.saturating_add(1),
                Ok(false) => {}
                Err(err) => {
                    crate::warn!("Skipping TXU compression after error: {}", err);
                }
            }
        }
        *guard = Arc::new(vec.into_boxed_slice());
        compressed_count
    }

    /// Ensures the TXU at `index` is decompressed for undo/redo.
    ///
    /// # Errors
    ///
    /// Returns a [`CompressionError`] when decompression of the TXU fails.
    pub fn ensure_txu_decompressed(&mut self, index: usize) -> TxuCompressResult<()> {
        let Ok(mut guard) = self.txus().write() else {
            return Ok(());
        };
        if index >= guard.len() {
            return Ok(());
        }
        let mut vec = guard.to_vec();
        let result = vec[index].ensure_decompressed();
        *guard = Arc::new(vec.into_boxed_slice());
        result
    }
}

#[inline]
fn looks_like_fcware_frame(data: &[u8]) -> bool {
    crate::fcware::compression::detect_frame(data).is_some()
}

#[cfg(test)]
mod tests {
    use super::{
        HISTORY_TOTAL_COMPRESS_THRESHOLD, KEEP_RECENT_UNCOMPRESSED,
        TXU_BLOB_COMPRESS_THRESHOLD,
    };
    use crate::base::base_memory::MemoryPressure;
    use crate::edit::wredit_base_writable::ENCODING_UTF8;
    use crate::edit::wredit_history::History;
    use crate::edit::wredit_history_txu::{
        HistoryTXU, HistoryTXUDelta, HistoryTXUDeltaType,
    };

    fn large_text(len: usize) -> Vec<u8> {
        // Highly repetitive source-like payload so Dysu can shrink it.
        b"fn example_function_name() { let value = 42; }\n"
            .iter()
            .copied()
            .cycle()
            .take(len)
            .collect()
    }

    fn make_txu(payload: Vec<u8>) -> HistoryTXU {
        let delta = HistoryTXUDelta::new(
            HistoryTXUDeltaType::CodeAdd,
            0,
            1,
            0,
            payload.clone(),
            ENCODING_UTF8,
            None,
        );
        let mut txu = HistoryTXU::new::<u8, u8, 4096>(delta);
        *txu.buffer_mut() = payload.clone();
        *txu.buffer_length_mut() = payload.len();
        txu
    }

    #[test]
    fn compress_and_decompress_roundtrip() {
        let payload = large_text(TXU_BLOB_COMPRESS_THRESHOLD * 2);
        let mut txu = make_txu(payload.clone());
        assert!(txu.should_compress(TXU_BLOB_COMPRESS_THRESHOLD));
        let compressed = txu
            .try_compress(TXU_BLOB_COMPRESS_THRESHOLD)
            .expect("compress");
        assert!(compressed);
        assert!(txu.is_compressed());
        assert!(txu.estimated_resident_bytes() < payload.len().saturating_mul(2));

        txu.ensure_decompressed().expect("decompress");
        assert!(!txu.is_compressed());
        assert_eq!(txu.buffer(), payload.as_slice());
        assert_eq!(txu.delta().ppbuff().as_slice(), payload.as_slice());
    }

    #[test]
    fn history_compresses_cold_txus() {
        let mut history = History::new();
        let payload = large_text(TXU_BLOB_COMPRESS_THRESHOLD * 2);
        let total = KEEP_RECENT_UNCOMPRESSED + 4;
        for _ in 0..total {
            history.add_txu(make_txu(payload.clone()));
        }
        let compressed = history.maybe_compress_under_pressure(MemoryPressure::High);
        assert!(compressed > 0);

        let guard = history.txus().read().expect("lock");
        let cold = guard.len().saturating_sub(4);
        for txu in &guard[..cold] {
            assert!(txu.is_compressed());
        }
        for txu in &guard[cold..] {
            assert!(!txu.is_compressed());
        }
    }

    #[test]
    fn small_txu_skipped() {
        let mut txu = make_txu(b"tiny".to_vec());
        let compressed = txu
            .try_compress(TXU_BLOB_COMPRESS_THRESHOLD)
            .expect("compress");
        assert!(!compressed);
        assert!(!txu.is_compressed());
    }

    #[test]
    fn history_total_threshold_constant_reasonable() {
        assert!(HISTORY_TOTAL_COMPRESS_THRESHOLD >= TXU_BLOB_COMPRESS_THRESHOLD);
    }
}
