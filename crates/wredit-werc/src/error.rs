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

//! Error type shared by every fallible operation in the crate.
//!
//! Every public operation that can fail returns [`WercResult`]; the crate
//! never panics in production code paths. The failures are all about buffer
//! sizes and stream framing - the model itself cannot fail, because a
//! context-mixing pass over a well-formed buffer always produces a
//! well-formed stream.

/// Shorthand for the results returned by this crate.
pub type WercResult<T> = Result<T, WercError>;

/// Reasons a WEZC operation can fail.
///
/// The enum is `#[non_exhaustive]`: a later revision may add variants (for
/// example a dictionary-substitution error) without breaking a caller that
/// matches exhaustively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WercError {
    /// A caller-supplied output buffer is smaller than the stream needs.
    ///
    /// `needed` is the exact number of bytes the operation would have
    /// written, so a caller can size the buffer and retry.
    OutputTooSmall {
        /// Bytes required by the operation.
        needed: usize,
        /// Bytes available in the caller's buffer.
        available: usize,
    },
    /// A caller-supplied input buffer is shorter than the stream it claims
    /// to hold.
    ///
    /// Reported while reading a fixed-width header or a plane of known
    /// length; `needed` is the length that element of the format requires.
    InputTooShort {
        /// Bytes required to read the next part of the stream.
        needed: usize,
        /// Bytes available in the caller's buffer.
        available: usize,
    },
    /// The stream does not start with the WEZC magic, so the bytes handed
    /// to the decoder are not a WEZC container.
    NotWezc,
    /// The container parses but declares a parameter this build does not
    /// implement: an unsupported stage-1 transform, or a model parameter
    /// outside the range the tables were built for.
    Unsupported {
        /// Field that was rejected, for the log line.
        field: &'static str,
    },
    /// The arithmetic decoder reached the end of the payload while bits
    /// were still outstanding.
    ///
    /// Either the payload was truncated or the encoder and decoder disagree
    /// about the stream; both mean the output would be a silently wrong
    /// document, so the decode is abandoned instead.
    TruncatedPayload,
}

impl WercError {
    /// Bytes required by the error, when the variant carries a size.
    ///
    /// Lets a caller that passed a probe buffer re-run the operation with
    /// `size_hint() + already_written` without recomputing the layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use codevar_wredit_werc::error::WercError;
    ///
    /// let err = WercError::OutputTooSmall { needed: 4096, available: 1024 };
    /// assert_eq!(err.size_hint(), Some(4096));
    /// assert_eq!(err.label(), "output buffer too small");
    /// ```
    #[inline]
    #[must_use]
    pub const fn size_hint(&self) -> Option<usize> {
        match self {
            Self::OutputTooSmall { needed, .. } | Self::InputTooShort { needed, .. } => Some(*needed),
            Self::NotWezc | Self::TruncatedPayload | Self::Unsupported { .. } => None,
        }
    }

    /// Stable, lowercase description of the failure.
    #[inline]
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::OutputTooSmall { .. } => "output buffer too small",
            Self::InputTooShort { .. } => "input buffer too short",
            Self::NotWezc => "not a wezc container",
            Self::Unsupported { .. } => "unsupported stream parameter",
            Self::TruncatedPayload => "truncated payload",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::WercError;

    #[test]
    fn size_hint_reports_only_sizing_errors() {
        assert_eq!(
            WercError::OutputTooSmall {
                needed: 12,
                available: 4
            }
            .size_hint(),
            Some(12)
        );
        assert_eq!(
            WercError::InputTooShort {
                needed: 8,
                available: 2
            }
            .size_hint(),
            Some(8)
        );
        assert_eq!(WercError::NotWezc.size_hint(), None);
    }

    #[test]
    fn errors_compare_by_value() {
        assert_eq!(
            WercError::Unsupported { field: "mode" },
            WercError::Unsupported { field: "mode" }
        );
        assert_ne!(
            WercError::Unsupported { field: "mode" },
            WercError::Unsupported { field: "mem" }
        );
    }
}
