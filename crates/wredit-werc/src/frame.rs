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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Stage 3's envelope: how a coded stream declares itself.
//!
//! A context-mixing stream is unreadable without the model that produced it,
//! and the model is sized by three numbers. The header carries those, the
//! stage-1 plane flags and two lengths, so a decoder can rebuild the exact
//! model and stop as soon as the bytes disagree with the declaration instead
//! of decoding a truncated document into garbage.
//!
//! ```text
//!  0  magic + version  "WEZC\x01"  5
//!  5  mem_bits                     1
//!  6  sse_bits                     1
//!  7  lr_shift                     1
//!  8  plane flags                  1
//!  9  element count        u32 LE  4
//! 13  payload length       u32 LE  4
//! ```
//!
//! The plane body length is not stored: [`crate::plane::body_len`] derives it
//! from the count and the flags, so a stream cannot contradict itself about
//! how many bytes the model has to produce. What the header does store is the
//! *coded* length, which is not derivable and is what lets a frame sit inside
//! a larger blob.

use crate::error::{WercError, WercResult};
use crate::model::Model;
use crate::plane;

/// Magic bytes and the version they carry. A stream whose version byte (or
/// whole magic) differs was written by a build with a different model, and
/// reading it as this one would produce a document nobody asked for.
pub const MAGIC: [u8; 5] = *b"WEZC\x01";

/// Bytes a header occupies, [`MAGIC`] included.
pub const HEADER_SIZE: usize = 17;

/// The model parameters a stream was coded with.
///
/// These are the prototype's tuning knobs, and they are the only way a
/// decoder knows how to size its tables. [`Params::DEFAULT`] is the desktop
/// budget; a caller with less RAM lowers `mem_bits` and accepts the ratio it
/// still gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    /// Log2 of the context and match tables.
    pub mem_bits: u8,
    /// Log2 of the SSE table.
    pub sse_bits: u8,
    /// Learning-rate shift for the mixer weights.
    pub lr_shift: u8,
}

impl Params {
    /// The parameters the ratio was measured at: 4 Mi entries of context and
    /// 64 Ki SSE slots on a desktop, which is the budget a document has to
    /// beat to be worth coding at all.
    pub const DEFAULT: Self = Self {
        mem_bits: 22,
        sse_bits: 16,
        lr_shift: 11,
    };

    /// The smallest parameters that still build a model, for a caller that
    /// has to live inside a microcontroller's RAM.
    pub const MINIMAL: Self = Self {
        mem_bits: 8,
        sse_bits: 4,
        lr_shift: 11,
    };

    /// Build the model these parameters describe.
    ///
    /// # Errors
    ///
    /// [`WercError::Unsupported`] if a byte is outside the range the tables
    /// were built for; see [`Model::new`].
    pub fn model(self) -> WercResult<Model> {
        Model::new(
            u32::from(self.mem_bits),
            u32::from(self.sse_bits),
            u32::from(self.lr_shift),
        )
    }
}

/// A parsed header: everything a decoder needs before it reads a payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Model the payload was coded with.
    pub params: Params,
    /// Stage-1 plane flags, from [`plane::presence`].
    pub flags: u8,
    /// Elements the document holds.
    pub count: u32,
    /// Bytes of arithmetic-coded payload that follow the header.
    pub payload_len: u32,
}

impl Header {
    /// The header for a block of elements and the payload that follows it.
    #[must_use]
    pub const fn new(params: Params, flags: u8, count: u32, payload_len: u32) -> Self {
        Self {
            params,
            flags,
            count,
            payload_len,
        }
    }

    /// Number of plane bytes this stream's model has to produce.
    ///
    /// `None` if the flags carry an unknown plane bit or the count is large
    /// enough to overflow the product - both mean the stream is not one this
    /// build can lay out.
    #[must_use]
    pub const fn body_len(&self) -> Option<usize> {
        plane::body_len(self.count as usize, self.flags)
    }

    /// Write the header to the front of `out`, returning its length.
    ///
    /// # Errors
    ///
    /// [`WercError::OutputTooSmall`] if `out` is shorter than
    /// [`HEADER_SIZE`].
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_wredit_werc::frame::{Header, Params, HEADER_SIZE};
    ///
    /// let mut bytes = [0u8; HEADER_SIZE];
    /// let header = Header::new(Params::MINIMAL, 0b1000, 7, 12);
    /// assert_eq!(header.write_to(&mut bytes).unwrap_or(0), HEADER_SIZE);
    /// assert_eq!(Header::parse(&bytes).unwrap_or(header), header);
    /// ```
    pub fn write_to(&self, out: &mut [u8]) -> WercResult<usize> {
        if out.len() < HEADER_SIZE {
            return Err(WercError::OutputTooSmall {
                needed: HEADER_SIZE,
                available: out.len(),
            });
        }
        out[..MAGIC.len()].copy_from_slice(&MAGIC);
        out[5] = self.params.mem_bits;
        out[6] = self.params.sse_bits;
        out[7] = self.params.lr_shift;
        out[8] = self.flags;
        out[9..13].copy_from_slice(&self.count.to_le_bytes());
        out[13..17].copy_from_slice(&self.payload_len.to_le_bytes());
        Ok(HEADER_SIZE)
    }

    /// Read a header, checking it describes a stream this build can decode.
    ///
    /// The flags and the count are validated as they are read, so a hostile
    /// header is refused here rather than at the first out-of-bounds plane
    /// index.
    ///
    /// # Errors
    ///
    /// [`WercError::InputTooShort`] if `bytes` is shorter than
    /// [`HEADER_SIZE`], [`WercError::NotWezc`] if the magic differs,
    /// [`WercError::Unsupported`] if a flag bit is unknown or the count's
    /// plane body does not fit the target's `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_wredit_werc::frame::{Header, Params, HEADER_SIZE};
    /// use codevar_wredit_werc::error::WercError;
    ///
    /// let mut bytes = [0u8; HEADER_SIZE];
    /// Header::new(Params::MINIMAL, 0b1000, 1, 0)
    ///     .write_to(&mut bytes)
    ///     .unwrap_or(0);
    /// bytes[0] = b'X';
    /// assert_eq!(Header::parse(&bytes), Err(WercError::NotWezc));
    /// ```
    pub fn parse(bytes: &[u8]) -> WercResult<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(WercError::InputTooShort {
                needed: HEADER_SIZE,
                available: bytes.len(),
            });
        }
        if bytes[..MAGIC.len()] != MAGIC {
            return Err(WercError::NotWezc);
        }
        let header = Self {
            params: Params {
                mem_bits: bytes[5],
                sse_bits: bytes[6],
                lr_shift: bytes[7],
            },
            flags: bytes[8],
            count: u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]),
            payload_len: u32::from_le_bytes([bytes[13], bytes[14], bytes[15], bytes[16]]),
        };
        if plane::plane_count(header.flags).is_none() {
            return Err(WercError::Unsupported { field: "flags" });
        }
        if header.body_len().is_none() {
            return Err(WercError::Unsupported { field: "count" });
        }
        Ok(header)
    }
}

#[cfg(test)]
mod tests {
    use super::{Header, MAGIC, Params};
    use crate::error::WercError;
    use crate::plane::{FLAG_PLANE1, FLAG_PLANE2, FLAG_PLANE3};

    const fn header(flags: u8, count: u32) -> Header {
        Header::new(Params::MINIMAL, flags, count, 0)
    }

    #[test]
    fn a_header_round_trips_through_its_own_bytes() {
        let mut bytes = [0u8; super::HEADER_SIZE];
        let original = Header::new(Params::DEFAULT, FLAG_PLANE3, 4096, 1234);
        assert_eq!(original.write_to(&mut bytes).ok(), Some(super::HEADER_SIZE));
        assert_eq!(Header::parse(&bytes), Ok(original));
        assert_eq!(&bytes[..MAGIC.len()], &MAGIC);
    }

    #[test]
    fn the_body_length_is_derived_not_stored() {
        // The header never says how many plane bytes follow: it is the count
        // times the planes its flags name, so a stream cannot lie about its
        // own size. Plane 0 has no flag and is always there.
        assert_eq!(
            header(FLAG_PLANE1 | FLAG_PLANE2 | FLAG_PLANE3, 3).body_len(),
            Some(12)
        );
        assert_eq!(header(FLAG_PLANE3, 3).body_len(), Some(6));
        assert_eq!(header(0, 3).body_len(), Some(3));
        assert_eq!(header(0, 0).body_len(), Some(0));
    }

    #[test]
    fn a_write_into_a_short_buffer_reports_the_size() {
        let mut bytes = [0u8; super::HEADER_SIZE - 1];
        assert_eq!(
            header(FLAG_PLANE3, 1).write_to(&mut bytes),
            Err(WercError::OutputTooSmall {
                needed: super::HEADER_SIZE,
                available: super::HEADER_SIZE - 1
            })
        );
    }

    #[test]
    fn a_parse_of_a_short_or_foreign_buffer_fails() {
        let mut bytes = [0u8; super::HEADER_SIZE];
        header(FLAG_PLANE3, 1)
            .write_to(&mut bytes)
            .unwrap_or_default();
        assert_eq!(
            Header::parse(&bytes[..8]),
            Err(WercError::InputTooShort {
                needed: super::HEADER_SIZE,
                available: 8
            })
        );
        bytes[4] ^= 0xFF;
        assert_eq!(Header::parse(&bytes), Err(WercError::NotWezc));
    }

    #[test]
    fn a_stream_with_planes_this_build_cannot_read_is_rejected() {
        let mut bytes = [0u8; super::HEADER_SIZE];
        // Plane 0 has no flag, so any other bit in the byte is unknown.
        header(1 << 4, 1)
            .write_to(&mut bytes)
            .unwrap_or_default();
        assert_eq!(
            Header::parse(&bytes),
            Err(WercError::Unsupported { field: "flags" })
        );
    }

    #[test]
    fn a_count_is_only_rejected_when_its_body_cannot_be_laid_out() {
        // The guard is the multiply, not the field width: a count whose plane
        // body overflows the target's usize is refused at parse, and on a
        // 64-bit host the largest count a header can carry still fits.
        let huge = header(FLAG_PLANE1 | FLAG_PLANE2 | FLAG_PLANE3, u32::MAX);
        if usize::BITS > 32 {
            assert_eq!(huge.body_len(), Some(u32::MAX as usize * 4));
        } else {
            assert_eq!(huge.body_len(), None);
        }
    }

    #[test]
    fn every_declared_parameter_reaches_the_model() {
        assert!(Params::MINIMAL.model().is_ok());
        let mut bytes = [0u8; super::HEADER_SIZE];
        header(FLAG_PLANE3, 1)
            .write_to(&mut bytes)
            .unwrap_or_default();
        bytes[5] = 27; // one bit past the largest table this build sizes
        // A header that fails to parse here would fall back to the minimal
        // parameters, whose model builds, and the assertion below would fail.
        let parsed = Header::parse(&bytes).unwrap_or(header(FLAG_PLANE3, 1));
        assert_eq!(
            parsed.params.model().err().map(|e| e.label()),
            Some("unsupported stream parameter")
        );
    }
}
