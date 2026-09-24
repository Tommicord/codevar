//! Copyright 2026 Codevar Project
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

//! TLS presentation language codec (RFC 8446 §3 / RFC 5246 §4).

use crate::tls_error::{TlsError, TlsResult};

/// Sequential reader over a TLS-encoded byte slice.
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    buf: &'a [u8],
}

impl<'a> Reader<'a> {
    /// Creates a reader over `buf`.
    #[must_use]
    pub const fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    /// Remaining unread bytes.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.buf.len()
    }

    /// Returns true when no bytes remain.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Returns the unread slice.
    #[must_use]
    pub const fn rest(&self) -> &'a [u8] {
        self.buf
    }

    /// Fails if any bytes remain.
    pub fn expect_empty(&self, ctx: &str) -> TlsResult<()> {
        if self.buf.is_empty() {
            Ok(())
        } else {
            Err(TlsError::decode(format!(
                "{ctx}: {n} trailing bytes",
                n = self.buf.len()
            )))
        }
    }

    /// Reads `n` bytes.
    pub fn bytes(&mut self, n: usize) -> TlsResult<&'a [u8]> {
        if self.buf.len() < n {
            return Err(TlsError::decode(format!(
                "need {n} bytes, have {}",
                self.buf.len()
            )));
        }
        let (head, tail) = self.buf.split_at(n);
        self.buf = tail;
        Ok(head)
    }

    /// Reads a single byte.
    pub fn u8(&mut self) -> TlsResult<u8> {
        Ok(self.bytes(1)?[0])
    }

    /// Reads a big-endian `u16`.
    pub fn u16(&mut self) -> TlsResult<u16> {
        let b = self.bytes(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    /// Reads a big-endian 24-bit integer.
    pub fn u24(&mut self) -> TlsResult<u32> {
        let b = self.bytes(3)?;
        Ok(u32::from_be_bytes([0, b[0], b[1], b[2]]))
    }

    /// Reads a big-endian `u32`.
    pub fn u32(&mut self) -> TlsResult<u32> {
        let b = self.bytes(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Reads a length-prefixed vector with an 8-bit length.
    pub fn vec_u8(&mut self) -> TlsResult<&'a [u8]> {
        let len = usize::from(self.u8()?);
        self.bytes(len)
    }

    /// Reads a length-prefixed vector with a 16-bit length.
    pub fn vec_u16(&mut self) -> TlsResult<&'a [u8]> {
        let len = usize::from(self.u16()?);
        self.bytes(len)
    }

    /// Reads a length-prefixed vector with a 24-bit length.
    pub fn vec_u24(&mut self) -> TlsResult<&'a [u8]> {
        let len = self.u24()? as usize;
        self.bytes(len)
    }
}

/// Appends a big-endian `u16`.
#[inline]
pub fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// Appends a big-endian 24-bit integer.
#[inline]
pub fn put_u24(out: &mut Vec<u8>, v: u32) {
    out.push(((v >> 16) & 0xff) as u8);
    out.push(((v >> 8) & 0xff) as u8);
    out.push((v & 0xff) as u8);
}

/// Appends a big-endian `u32`.
#[inline]
pub fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// Appends an 8-bit length-prefixed vector.
pub fn put_vec_u8(out: &mut Vec<u8>, data: &[u8]) -> TlsResult<()> {
    if data.len() > 255 {
        return Err(TlsError::Internal("vector exceeds u8 length prefix".into()));
    }
    out.push(data.len() as u8);
    out.extend_from_slice(data);
    Ok(())
}

/// Appends a 16-bit length-prefixed vector.
pub fn put_vec_u16(out: &mut Vec<u8>, data: &[u8]) -> TlsResult<()> {
    if data.len() > 65535 {
        return Err(TlsError::Internal(
            "vector exceeds u16 length prefix".into(),
        ));
    }
    put_u16(out, data.len() as u16);
    out.extend_from_slice(data);
    Ok(())
}

/// Appends a 24-bit length-prefixed vector.
pub fn put_vec_u24(out: &mut Vec<u8>, data: &[u8]) -> TlsResult<()> {
    if data.len() > 0xff_ffff {
        return Err(TlsError::Internal(
            "vector exceeds u24 length prefix".into(),
        ));
    }
    put_u24(out, data.len() as u32);
    out.extend_from_slice(data);
    Ok(())
}

/// Reserves a 16-bit length prefix and returns its index; call [`fill_u16_len`].
#[must_use]
pub fn start_u16_vec(out: &mut Vec<u8>) -> usize {
    let idx = out.len();
    out.extend_from_slice(&[0, 0]);
    idx
}

/// Writes the length of bytes after `idx+2` into a previously reserved u16 prefix.
pub fn fill_u16_len(out: &mut [u8], idx: usize) -> TlsResult<()> {
    let len = out.len().saturating_sub(idx + 2);
    if len > 65535 {
        return Err(TlsError::Internal(
            "vector exceeds u16 length prefix".into(),
        ));
    }
    let bytes = u16::try_from(len)
        .map_err(|_| TlsError::Internal("vector exceeds u16 length prefix".into()))?
        .to_be_bytes();
    out[idx] = bytes[0];
    out[idx + 1] = bytes[1];
    Ok(())
}

/// Reserves a 24-bit length prefix.
#[must_use]
pub fn start_u24_vec(out: &mut Vec<u8>) -> usize {
    let idx = out.len();
    out.extend_from_slice(&[0, 0, 0]);
    idx
}

/// Writes the length of bytes after `idx+3` into a previously reserved u24 prefix.
pub fn fill_u24_len(out: &mut [u8], idx: usize) -> TlsResult<()> {
    let len = out.len().saturating_sub(idx + 3);
    if len > 0xff_ffff {
        return Err(TlsError::Internal(
            "vector exceeds u24 length prefix".into(),
        ));
    }
    out[idx] = ((len >> 16) & 0xff) as u8;
    out[idx + 1] = ((len >> 8) & 0xff) as u8;
    out[idx + 2] = (len & 0xff) as u8;
    Ok(())
}

/// Reserves an 8-bit length prefix.
#[must_use]
pub fn start_u8_vec(out: &mut Vec<u8>) -> usize {
    let idx = out.len();
    out.push(0);
    idx
}

/// Fills a previously reserved u8 length prefix.
pub fn fill_u8_len(out: &mut [u8], idx: usize) -> TlsResult<()> {
    let len = out.len().saturating_sub(idx + 1);
    if len > 255 {
        return Err(TlsError::Internal("vector exceeds u8 length prefix".into()));
    }
    out[idx] = len as u8;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_empty_input_behaves() {
        let mut r = Reader::new(&[]);
        assert!(r.is_empty());
        assert_eq!(r.remaining(), 0);
        assert!(r.rest().is_empty());
        assert!(r.expect_empty("empty").is_ok());
        assert!(r.u8().is_err());
        assert!(r.u16().is_err());
        assert!(r.u24().is_err());
        assert!(r.u32().is_err());
        assert!(r.bytes(1).is_err());
        assert!(r.bytes(0).unwrap().is_empty());
    }

    #[test]
    fn primitives_round_trip_big_endian() {
        let mut out = Vec::new();
        out.push(0x7f);
        put_u16(&mut out, 0x1234);
        put_u24(&mut out, 0xab_cdef);
        put_u32(&mut out, 0xdead_beef);
        let mut r = Reader::new(&out);
        assert_eq!(r.u8().unwrap(), 0x7f);
        assert_eq!(r.u16().unwrap(), 0x1234);
        assert_eq!(r.u24().unwrap(), 0xab_cdef);
        assert_eq!(r.u32().unwrap(), 0xdead_beef);
        assert_eq!(r.remaining(), 0);
        assert!(r.expect_empty("primitives").is_ok());
    }

    #[test]
    fn truncated_primitive_reads_error_at_every_offset() {
        let data = [0x11u8, 0x22, 0x33, 0x44];
        assert!(Reader::new(&data[..0]).u8().is_err());
        for n in 0..2 {
            assert!(Reader::new(&data[..n]).u16().is_err(), "u16 n={n}");
        }
        for n in 0..3 {
            assert!(Reader::new(&data[..n]).u24().is_err(), "u24 n={n}");
        }
        for n in 0..4 {
            assert!(Reader::new(&data[..n]).u32().is_err(), "u32 n={n}");
        }
        let mut exact = Reader::new(&data);
        assert_eq!(exact.u8().unwrap(), 0x11);
        assert_eq!(exact.u16().unwrap(), 0x2233);
        assert_eq!(exact.u8().unwrap(), 0x44);
        assert!(exact.is_empty());
        assert!(exact.u8().is_err());
    }

    #[test]
    fn length_prefixed_vectors_round_trip_at_boundaries() {
        for len in [0usize, 1, 255] {
            let payload = vec![0xA5u8; len];
            let mut out = Vec::new();
            put_vec_u8(&mut out, &payload).unwrap();
            let mut r = Reader::new(&out);
            assert_eq!(r.vec_u8().unwrap(), payload.as_slice());
            assert!(r.is_empty());
        }
        for len in [0usize, 1, 65535] {
            let payload = vec![0x5Au8; len];
            let mut out = Vec::new();
            put_vec_u16(&mut out, &payload).unwrap();
            let mut r = Reader::new(&out);
            assert_eq!(r.vec_u16().unwrap(), payload.as_slice());
            assert!(r.is_empty());
        }
        let payload = vec![0x3Cu8; 0x01_0000];
        let mut out = Vec::new();
        put_vec_u24(&mut out, &payload).unwrap();
        let mut r = Reader::new(&out);
        assert_eq!(r.vec_u24().unwrap(), payload.as_slice());
        assert!(r.is_empty());
    }

    #[test]
    fn oversized_vector_writes_rejected_before_mutation() {
        let mut out = vec![0xEE];
        assert!(put_vec_u8(&mut out, &[0u8; 256]).is_err());
        assert_eq!(out, vec![0xEE]);
        assert!(put_vec_u16(&mut out, &[0u8; 65536]).is_err());
        assert_eq!(out, vec![0xEE]);
        let huge = vec![0u8; 0xff_ffff + 1];
        assert!(put_vec_u24(&mut out, &huge).is_err());
        assert_eq!(out, vec![0xEE]);
    }

    #[test]
    fn u24_vector_exact_max_length_accepted() {
        let payload = vec![7u8; 0xff_ffff];
        let mut out = Vec::new();
        put_vec_u24(&mut out, &payload).unwrap();
        assert_eq!(out.len(), 3 + 0xff_ffff);
        let mut r = Reader::new(&out);
        assert_eq!(r.vec_u24().unwrap().len(), 0xff_ffff);
    }

    #[test]
    fn declared_lengths_beyond_buffer_rejected_with_tiny_input() {
        let mut r16 = Reader::new(&[0xff, 0xff, 0x01, 0x02]);
        assert!(r16.vec_u16().is_err());
        let mut r24 = Reader::new(&[0xff, 0xff, 0xff, 0x01]);
        assert!(r24.vec_u24().is_err());
        let mut r8 = Reader::new(&[0xff]);
        assert!(r8.vec_u8().is_err());
        let mut none = Reader::new(&[]);
        assert!(none.vec_u8().is_err());
        assert!(none.vec_u16().is_err());
        assert!(none.vec_u24().is_err());
    }

    #[test]
    fn zero_length_vectors_yield_empty_slices() {
        let mut r8 = Reader::new(&[0x00]);
        assert!(r8.vec_u8().unwrap().is_empty());
        let mut r16 = Reader::new(&[0x00, 0x00]);
        assert!(r16.vec_u16().unwrap().is_empty());
        let mut r24 = Reader::new(&[0x00, 0x00, 0x00]);
        assert!(r24.vec_u24().unwrap().is_empty());
        assert!(r8.is_empty());
        assert!(r16.is_empty());
        assert!(r24.is_empty());
    }

    #[test]
    fn truncated_vector_body_errors_at_every_offset() {
        let wire = [0x00, 0x03, 0xaa, 0xbb, 0xcc];
        for n in 0..5 {
            assert!(Reader::new(&wire[..n]).vec_u16().is_err(), "n={n}");
        }
        let mut exact = Reader::new(&wire);
        assert_eq!(exact.vec_u16().unwrap(), &[0xaa, 0xbb, 0xcc]);
        assert!(exact.is_empty());
    }

    #[test]
    fn expect_empty_reports_trailing_bytes() {
        let mut r = Reader::new(&[1, 2, 3]);
        let err = r.expect_empty("frame").unwrap_err();
        let text = err.to_string();
        assert!(text.contains("frame"), "text={text}");
        assert!(text.contains("3 trailing bytes"), "text={text}");
        assert!(r.u8().is_ok());
        assert!(r.expect_empty("frame").is_err());
        assert!(r.u8().is_ok());
        let _ = r.u8().unwrap();
        assert!(r.expect_empty("frame").is_ok());
    }

    #[test]
    fn bytes_read_past_end_errors_without_consuming() {
        let mut r = Reader::new(&[1, 2]);
        assert!(r.bytes(3).is_err());
        assert_eq!(r.remaining(), 2);
        assert_eq!(r.bytes(2).unwrap(), &[1, 2]);
        assert!(r.bytes(1).is_err());
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn rest_returns_unread_suffix() {
        let mut r = Reader::new(&[1, 2, 3, 4]);
        assert_eq!(r.rest(), &[1, 2, 3, 4]);
        let head = r.bytes(2).unwrap();
        assert_eq!(head, &[1, 2]);
        assert_eq!(r.rest(), &[3, 4]);
        assert_eq!(r.remaining(), 2);
    }

    #[test]
    fn put_u24_truncates_bits_above_24() {
        let mut out = Vec::new();
        put_u24(&mut out, 0xff_ffff);
        assert_eq!(out, vec![0xff, 0xff, 0xff]);
        out.clear();
        put_u24(&mut out, 0x0100_0000);
        assert_eq!(out, vec![0x00, 0x00, 0x00]);
        out.clear();
        put_u24(&mut out, 0x00ab_cdef);
        assert_eq!(out, vec![0xab, 0xcd, 0xef]);
    }

    #[test]
    fn fill_length_helpers_round_trip() {
        let mut out8 = Vec::new();
        let i8 = start_u8_vec(&mut out8);
        out8.extend_from_slice(b"abc");
        fill_u8_len(&mut out8, i8).unwrap();
        assert_eq!(out8, vec![3, b'a', b'b', b'c']);

        let mut out16 = Vec::new();
        let i16 = start_u16_vec(&mut out16);
        out16.extend_from_slice(&[9, 9]);
        fill_u16_len(&mut out16, i16).unwrap();
        assert_eq!(out16, vec![0, 2, 9, 9]);

        let mut out24 = Vec::new();
        let i24 = start_u24_vec(&mut out24);
        out24.extend_from_slice(&[1, 2, 3]);
        fill_u24_len(&mut out24, i24).unwrap();
        assert_eq!(out24, vec![0, 0, 3, 1, 2, 3]);

        let mut empty16 = Vec::new();
        let ie = start_u16_vec(&mut empty16);
        fill_u16_len(&mut empty16, ie).unwrap();
        assert_eq!(empty16, vec![0, 0]);
    }

    #[test]
    fn fill_length_overflow_rejected_prefix_untouched() {
        let mut out16 = Vec::new();
        let idx = start_u16_vec(&mut out16);
        out16.resize(out16.len() + 65536, 0);
        assert!(fill_u16_len(&mut out16, idx).is_err());
        assert_eq!(&out16[..2], &[0, 0]);

        let mut out8 = Vec::new();
        let idx8 = start_u8_vec(&mut out8);
        out8.resize(out8.len() + 256, 0);
        assert!(fill_u8_len(&mut out8, idx8).is_err());
        assert_eq!(out8[0], 0);

        let mut out24 = Vec::new();
        let idx24 = start_u24_vec(&mut out24);
        out24.resize(out24.len() + 0xff_ffff + 1, 0);
        assert!(fill_u24_len(&mut out24, idx24).is_err());
        assert_eq!(&out24[..3], &[0, 0, 0]);
    }

    #[test]
    fn encoder_reuses_large_buffer_across_iterations() {
        let mut out = Vec::with_capacity(2 * 1024 * 1024);
        let payload = vec![0x42u8; 64 * 1024 - 1];
        for i in 0..32u32 {
            out.clear();
            put_u16(&mut out, 0x1234);
            put_vec_u16(&mut out, &payload).unwrap();
            put_u24(&mut out, i);
            put_u32(&mut out, 0xfeed_face);
            let mut r = Reader::new(&out);
            assert_eq!(r.u16().unwrap(), 0x1234);
            assert_eq!(r.vec_u16().unwrap().len(), payload.len());
            assert_eq!(r.u24().unwrap(), i);
            assert_eq!(r.u32().unwrap(), 0xfeed_face);
            assert!(r.expect_empty("iter").is_ok());
        }
        assert!(out.capacity() >= 1024 * 1024);
    }
}
