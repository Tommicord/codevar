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

use core::fmt;

/// Enumeration of possible methods to seek within an I/O object.
///
/// This is the `codevar-io` equivalent of [`std::io::SeekFrom`].
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum SeekFrom {
    /// Sets the offset to the provided number of bytes.
    Start(u64),
    /// Sets the offset to the size of this object plus the specified number of bytes.
    End(i64),
    /// Sets the offset to the current position plus the specified number of bytes.
    Current(i64),
}

/// Possible kinds of errors.
///
/// This list is intended to grow over time, and it is not recommended to
/// exhaustively match against it. In application code, use `match` for the `ErrorKind`
/// values you are expecting; use `_` to match "all other errors".
///
/// This is the `codevar-io` equivalent of [`std::io::ErrorKind`], except with the following changes:
///
/// - `WouldBlock` is removed, since `codevar-io` traits are always blocking. See the [crate-level documentation](crate) for details.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[non_exhaustive]
pub enum ErrorKind {
    /// Unspecified error kind.
    Other,

    /// An entity was not found, often a file.
    NotFound,
    /// The operation lacked the necessary privileges to complete.
    PermissionDenied,
    /// The connection was refused by the remote server.
    ConnectionRefused,
    /// The connection was reset by the remote server.
    ConnectionReset,
    /// The connection was aborted (terminated) by the remote server.
    ConnectionAborted,
    /// The network operation failed because it was not connected yet.
    NotConnected,
    /// A socket address could not be bound because the address is already in
    /// use elsewhere.
    AddrInUse,
    /// A nonexistent interface was requested or the requested address was not
    /// local.
    AddrNotAvailable,
    /// The operation failed because a pipe was closed.
    BrokenPipe,
    /// An entity already exists, often a file.
    AlreadyExists,
    /// A parameter was incorrect.
    InvalidInput,
    /// Data not valid for the operation were encountered.
    ///
    /// Unlike [`InvalidInput`], this typically means that the operation
    /// parameters were valid, however the error was caused by malformed
    /// input data.
    ///
    /// For example, a function that reads a file into a string will error with
    /// `InvalidData` if the file's contents are not valid UTF-8.
    ///
    /// [`InvalidInput`]: ErrorKind::InvalidInput
    InvalidData,
    /// The I/O operation's timeout expired, causing it to be canceled.
    TimedOut,
    /// This operation was interrupted.
    ///
    /// Interrupted operations can typically be retried.
    Interrupted,
    /// This operation is unsupported on this platform.
    ///
    /// This means that the operation can never succeed.
    Unsupported,
    /// An operation could not be completed, because it failed
    /// to allocate enough memory.
    OutOfMemory,
    /// An attempted write could not write any data.
    WriteZero,
}

/// Error trait.
///
/// This trait allows generic code to do limited inspecting of errors,
/// to react differently to different kinds.
pub trait Error: core::error::Error {
    /// Get the kind of this error.
    fn kind(&self) -> ErrorKind;
}

impl Error for core::convert::Infallible {
    fn kind(&self) -> ErrorKind {
        match *self {}
    }
}

impl Error for ErrorKind {
    fn kind(&self) -> ErrorKind {
        *self
    }
}

impl core::error::Error for ErrorKind {}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Base trait for all IO traits, defining the error type.
///
/// All IO operations of all traits return the error defined in this trait.
///
/// Having a shared trait instead of having every trait define its own
/// `Error` associated type enforces all impls on the same type use the same error.
/// This is very convenient when writing generic code, it means you have to
/// handle a single error type `T::Error`, instead of `<T as Read>::Error` and `<T as Write>::Error`
/// which might be different types.
pub trait ErrorType {
    /// Error type of all the IO operations on this type.
    type Error: Error;
}

impl<T: ?Sized + ErrorType> ErrorType for &mut T {
    type Error = T::Error;
}

/// Error returned by [`Read::read_exact`]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum ReadExactError<E> {
    /// An EOF error was encountered before reading the exact amount of requested bytes.
    UnexpectedEof,
    /// Error returned by the inner Read.
    Other(E),
}

impl<E> From<E> for ReadExactError<E> {
    fn from(err: E) -> Self {
        Self::Other(err)
    }
}

impl<E: fmt::Debug> fmt::Display for ReadExactError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<E: core::error::Error + 'static> core::error::Error for ReadExactError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::UnexpectedEof => None,
            Self::Other(error) => Some(error),
        }
    }
}

/// Errors that could be returned by `Write` on `&mut [u8]`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
#[non_exhaustive]
pub enum SliceWriteError {
    /// The target slice was full and so could not receive any new data.
    Full,
}

impl Error for SliceWriteError {
    fn kind(&self) -> ErrorKind {
        match self {
            SliceWriteError::Full => ErrorKind::WriteZero,
        }
    }
}

impl From<ErrorKind> for SliceWriteError {
    fn from(_kind: ErrorKind) -> Self {
        SliceWriteError::Full
    }
}

impl fmt::Display for SliceWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::error::Error for SliceWriteError {}

/// Error returned by [`Write::write_fmt`]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum WriteFmtError<E> {
    /// An error was encountered while formatting.
    FmtError,
    /// Error returned by the inner Write.
    Other(E),
}

impl<E> From<E> for WriteFmtError<E> {
    fn from(err: E) -> Self {
        Self::Other(err)
    }
}

impl<E: fmt::Debug> fmt::Display for WriteFmtError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<E: core::error::Error + 'static> core::error::Error for WriteFmtError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::FmtError => None,
            Self::Other(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, Read, io_error::Error as IoError};
    use std::error::Error as StdError;

    #[test]
    fn test_seek_from_variants() {
        let start = SeekFrom::Start(100);
        assert_eq!(start, SeekFrom::Start(100));

        let end = SeekFrom::End(-50);
        assert_eq!(end, SeekFrom::End(-50));

        let current = SeekFrom::Current(10);
        assert_eq!(current, SeekFrom::Current(10));
    }

    #[test]
    fn test_error_kind_equality() {
        assert_eq!(ErrorKind::Other, ErrorKind::Other);
        assert_eq!(ErrorKind::NotFound, ErrorKind::NotFound);
        assert_ne!(ErrorKind::Other, ErrorKind::NotFound);
    }

    #[test]
    fn test_error_kind_debug() {
        let kind = ErrorKind::NotFound;
        let debug_str = format!("{:?}", kind);
        assert_eq!(debug_str, "NotFound");
    }

    #[test]
    fn test_error_kind_display() {
        let kind = ErrorKind::PermissionDenied;
        let display_str = format!("{}", kind);
        assert_eq!(display_str, "PermissionDenied");
    }

    #[test]
    fn test_error_trait_for_infallible() {
        // Verify Infallible implements Error trait - use type ascription
        fn assert_error<E: IoError>() {}
        assert_error::<core::convert::Infallible>();
    }

    #[test]
    fn test_error_trait_for_error_kind() {
        let kind = ErrorKind::InvalidInput;
        assert_eq!(kind.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn test_error_kind_error_trait() {
        let kind = ErrorKind::OutOfMemory;
        let _error: &dyn core::error::Error = &kind;
    }

    #[test]
    fn test_read_exact_error_from() {
        let err = ReadExactError::from(ErrorKind::TimedOut);
        assert!(matches!(err, ReadExactError::Other(ErrorKind::TimedOut)));
    }

    #[test]
    fn test_read_exact_error_display() {
        let err: ReadExactError<ErrorKind> = ReadExactError::UnexpectedEof;
        let display_str = format!("{}", err);
        assert!(display_str.contains("UnexpectedEof"));
    }

    #[test]
    fn test_read_exact_error_source() {
        let err: ReadExactError<ErrorKind> = ReadExactError::UnexpectedEof;
        assert!(err.source().is_none());

        let err: ReadExactError<ErrorKind> = ReadExactError::Other(ErrorKind::Other);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_slice_write_error_display() {
        let err = SliceWriteError::Full;
        let display_str = format!("{}", err);
        assert!(display_str.contains("Full"));
    }

    #[test]
    fn test_slice_write_error_source() {
        let err = SliceWriteError::Full;
        assert!(err.source().is_none());
    }

    #[test]
    fn test_write_fmt_error_from() {
        let err = WriteFmtError::from(ErrorKind::Other);
        assert!(matches!(err, WriteFmtError::Other(ErrorKind::Other)));
    }

    #[test]
    fn test_write_fmt_error_display() {
        let err: WriteFmtError<ErrorKind> = WriteFmtError::FmtError;
        let display_str = format!("{}", err);
        assert!(display_str.contains("FmtError"));
    }

    #[test]
    fn test_write_fmt_error_source() {
        let err: WriteFmtError<ErrorKind> = WriteFmtError::FmtError;
        assert!(err.source().is_none());

        let err: WriteFmtError<ErrorKind> = WriteFmtError::Other(ErrorKind::Other);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_error_type_for_mut() {
        struct MyReader;
        impl ErrorType for MyReader {
            type Error = ErrorKind;
        }
        impl Read for MyReader {
            fn read(&mut self, _buf: &mut [u8]) -> Result<usize, Self::Error> {
                Ok(0)
            }
        }

        let mut reader = MyReader;
        let _mut_ref = &mut reader;
        // Verify ErrorType is implemented for &mut T
        let _: <&mut MyReader as ErrorType>::Error = ErrorKind::Other;
    }
}
