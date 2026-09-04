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

//! TLS presentation language codec (RFC 8446 §3 / RFC 5246 §4).

use crate::network::tls_error::{TlsError, TlsResult};

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
pub fn fill_u16_len(out: &mut Vec<u8>, idx: usize) -> TlsResult<()> {
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
pub fn fill_u24_len(out: &mut Vec<u8>, idx: usize) -> TlsResult<()> {
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
pub fn fill_u8_len(out: &mut Vec<u8>, idx: usize) -> TlsResult<()> {
    let len = out.len().saturating_sub(idx + 1);
    if len > 255 {
        return Err(TlsError::Internal("vector exceeds u8 length prefix".into()));
    }
    out[idx] = len as u8;
    Ok(())
}
