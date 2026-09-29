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

use crate::{ErrorKind, ErrorType, ReadExactError, SeekFrom, WriteFmtError};

/// Blocking reader.
///
/// This trait is the `codevar-io` equivalent of [`std::io::Read`].
pub trait Read: ErrorType {
    /// Read some bytes from this source into the specified buffer, returning how many bytes were read.
    ///
    /// If no bytes are currently available to read:
    /// - The method blocks until at least one byte becomes available;
    /// - Once at least one (or more) bytes become available, a non-zero amount of those is copied to the
    ///   beginning of `buf`, and the amount is returned, *without waiting or blocking any further for
    ///   more bytes to become available*.
    ///
    /// If bytes are available to read:
    /// - A non-zero amount of bytes is read to the beginning of `buf`, and the amount is returned immediately,
    ///   *without blocking and waiting for more bytes to become available*;
    ///
    /// Note that once some bytes are available to read, it is *not* guaranteed that all available bytes are returned.
    /// It is possible for the implementation to read an amount of bytes less than `buf.len()` while there are more
    /// bytes immediately available.
    ///
    /// This blocking behavior is important for the cases where `Read` represents the "read" leg of a pipe-like
    /// protocol (a socket, a pipe, a serial line etc.). The semantics is that the caller - by passing a non-empty
    /// buffer - does expect _some_ data (one or more bytes) - but _not necessarily `buf.len()` or more bytes_ -
    /// to become available, before the peer represented by `Read` would stop sending bytes due to
    /// application-specific reasons (as in the peer waiting for a response to the data it had sent so far).
    ///
    /// If the reader is at end-of-file (EOF), `Ok(0)` is returned. There is no guarantee that a reader at EOF
    /// will always be so in the future, for example a reader can stop being at EOF if another process appends
    /// more bytes to the underlying file.
    ///
    /// If `buf.len() == 0`, `read` returns without blocking, with either `Ok(0)` or an error.
    /// The `Ok(0)` doesn't indicate EOF, unlike when called with a non-empty buffer.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error>;

    /// Read the exact number of bytes required to fill `buf`.
    ///
    /// This function calls `read()` in a loop until exactly `buf.len()` bytes have
    /// been read, blocking if needed.
    ///
    /// If you are using [`ReadReady`] to avoid blocking, you should not use this function.
    /// `ReadReady::read_ready()` returning true only guarantees the first call to `read()` will
    /// not block, so this function may still block in subsequent calls.
    fn read_exact(&mut self, mut buf: &mut [u8]) -> Result<(), ReadExactError<Self::Error>> {
        while !buf.is_empty() {
            match self.read(buf) {
                Ok(0) => break,
                Ok(n) => buf = &mut buf[n..],
                Err(e) => return Err(ReadExactError::Other(e)),
            }
        }
        if buf.is_empty() {
            Ok(())
        } else {
            Err(ReadExactError::UnexpectedEof)
        }
    }
}

/// Blocking buffered reader.
///
/// This trait is the `codevar-io` equivalent of [`std::io::BufRead`].
pub trait BufRead: Read {
    /// Return the contents of the internal buffer, filling it with more data from the inner reader if it is empty.
    ///
    /// If no bytes are currently available to read, this function blocks until at least one byte is available.
    ///
    /// If the reader is at end-of-file (EOF), an empty slice is returned. There is no guarantee that a reader at EOF
    /// will always be so in the future, for example a reader can stop being at EOF if another process appends
    /// more bytes to the underlying file.
    fn fill_buf(&mut self) -> Result<&[u8], Self::Error>;

    /// Tell this buffer that `amt` bytes have been consumed from the buffer, so they should no longer be returned in calls to `fill_buf`.
    fn consume(&mut self, amt: usize);
}

/// Blocking writer.
///
/// This trait is the `codevar-io` equivalent of [`std::io::Write`].
pub trait Write: ErrorType {
    /// Write a buffer into this writer, returning how many bytes were written.
    ///
    /// If the writer is not currently ready to accept more bytes (for example, its buffer is full),
    /// this function blocks until it is ready to accept least one byte.
    ///
    /// If it's ready to accept bytes, a non-zero amount of bytes is written from the beginning of `buf`, and the amount
    /// is returned. It is not guaranteed that *all* available buffer space is filled, i.e. it is possible for the
    /// implementation to write an amount of bytes less than `buf.len()` while the writer continues to be
    /// ready to accept more bytes immediately.
    ///
    /// Implementations must not return `Ok(0)` unless `buf` is empty. Situations where the
    /// writer is not able to accept more bytes must instead be indicated with an error,
    /// where the `ErrorKind` is `WriteZero`.
    ///
    /// If `buf` is empty, `write` returns without blocking, with either `Ok(0)` or an error.
    /// `Ok(0)` doesn't indicate an error.
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error>;

    /// Flush this output stream, blocking until all intermediately buffered contents reach their destination.
    fn flush(&mut self) -> Result<(), Self::Error>;

    /// Write an entire buffer into this writer.
    ///
    /// This function calls `write()` in a loop until exactly `buf.len()` bytes have
    /// been written, blocking if needed.
    ///
    /// If you are using [`WriteReady`] to avoid blocking, you should not use this function.
    /// `WriteReady::write_ready()` returning true only guarantees the first call to `write()` will
    /// not block, so this function may still block in subsequent calls.
    ///
    /// Returns an error if `write()` returns `Ok(0)` when `buf` is not empty,
    /// indicating that the writer is not able to accept more bytes.
    fn write_all(&mut self, mut buf: &[u8]) -> Result<(), Self::Error>
    where
        Self::Error: From<ErrorKind>,
    {
        while !buf.is_empty() {
            match self.write(buf) {
                Ok(0) => return Err(ErrorKind::WriteZero.into()),
                Ok(n) => buf = &buf[n..],
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Write a formatted string into this writer, returning any error encountered.
    ///
    /// This function calls `write()` in a loop until the entire formatted string has
    /// been written, blocking if needed.
    ///
    /// If you are using [`WriteReady`] to avoid blocking, you should not use this function.
    /// `WriteReady::write_ready()` returning true only guarantees the first call to `write()` will
    /// not block, so this function may still block in subsequent calls.
    ///
    /// Unlike [`Write::write`], the number of bytes written is not returned. However, in the case of
    /// writing to an `&mut [u8]` its possible to calculate the number of bytes written by subtracting
    /// the length of the slice after the write, from the initial length of the slice.
    ///
    /// ```rust
    /// # use codevar_io::{Cursor, Write};
    /// let mut buf = [0u8; 256];
    /// let mut cursor = Cursor::new(&mut buf[..]);
    /// let start = cursor.get_ref().len();
    /// cursor.write_fmt(format_args!("{}", "Test")).expect("write_fmt should succeed");
    /// let len = start - cursor.get_ref().len();
    /// ```
    fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> Result<(), WriteFmtError<Self::Error>>
    where
        Self::Error: From<ErrorKind>,
    {
        // Create a shim which translates a Write to a fmt::Write and saves
        // off I/O errors. instead of discarding them
        struct Adapter<'a, T: Write + ?Sized + 'a>
        where
            T::Error: From<ErrorKind>,
        {
            inner: &'a mut T,
            error: Result<(), T::Error>,
        }

        impl<T: Write + ?Sized> fmt::Write for Adapter<'_, T>
        where
            T::Error: From<ErrorKind>,
        {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                match self.inner.write_all(s.as_bytes()) {
                    Ok(()) => Ok(()),
                    Err(e) => {
                        self.error = Err(e);
                        Err(fmt::Error)
                    }
                }
            }
        }

        let mut output = Adapter {
            inner: self,
            error: Ok(()),
        };
        match fmt::write(&mut output, fmt) {
            Ok(()) => Ok(()),
            Err(..) => match output.error {
                // check if the error came from the underlying `Write` or not
                Err(e) => Err(WriteFmtError::Other(e)),
                Ok(()) => Err(WriteFmtError::FmtError),
            },
        }
    }
}

/// Blocking seek within streams.
///
/// The `Seek` trait provides a cursor which can be moved within a stream of
/// bytes.
///
/// The stream typically has a fixed size, allowing seeking relative to either
/// end or the current offset.
///
/// This trait is the `codevar-io` equivalent of [`std::io::Seek`].
pub trait Seek: ErrorType {
    /// Seek to an offset, in bytes, in a stream.
    /// A seek beyond the end of a stream is allowed, but behavior is defined
    /// by the implementation.
    ///
    /// If the seek operation completed successfully,
    /// this method returns the new position from the start of the stream.
    /// That position can be used later with [`SeekFrom::Start`].
    ///
    /// # Errors
    ///
    /// Seeking can fail, for example because it might involve flushing a buffer.
    ///
    /// Seeking to a negative offset is considered an error.
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error>;

    /// Rewind to the beginning of a stream.
    ///
    /// This is a convenience method, equivalent to `seek(SeekFrom::Start(0))`.
    ///
    /// # Errors
    ///
    /// Rewinding can fail, for example because it might involve flushing a buffer.
    fn rewind(&mut self) -> Result<(), Self::Error> {
        self.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    /// Returns the current seek position from the start of the stream.
    ///
    /// This is equivalent to `self.seek(SeekFrom::Current(0))`.
    fn stream_position(&mut self) -> Result<u64, Self::Error> {
        self.seek(SeekFrom::Current(0))
    }

    /// Seeks relative to the current position.
    ///
    /// This is equivalent to `self.seek(SeekFrom::Current(offset))` but
    /// doesn't return the new position which can allow some implementations
    /// to perform more efficient seeks.
    fn seek_relative(&mut self, offset: i64) -> Result<(), Self::Error> {
        self.seek(SeekFrom::Current(offset))?;
        Ok(())
    }
}

/// Get whether a reader is ready.
///
/// This allows using a [`Read`] or [`BufRead`] in a nonblocking fashion, i.e. trying to read
/// only when it is ready.
pub trait ReadReady: ErrorType {
    /// Get whether the reader is ready for immediately reading.
    ///
    /// This usually means that there is either some bytes have been received and are buffered and ready to be read,
    /// or that the reader is at EOF.
    ///
    /// If this returns `true`, it's guaranteed that the next call to [`Read::read`] or [`BufRead::fill_buf`] will not block.
    fn read_ready(&mut self) -> Result<bool, Self::Error>;
}

/// Get whether a writer is ready.
///
/// This allows using a [`Write`] in a nonblocking fashion, i.e. trying to write
/// only when it is ready.
pub trait WriteReady: ErrorType {
    /// Get whether the writer is ready for immediately writing.
    ///
    /// This usually means that there is free space in the internal transmit buffer.
    ///
    /// If this returns `true`, it's guaranteed that the next call to [`Write::write`] will not block.
    fn write_ready(&mut self) -> Result<bool, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cursor, ErrorKind, ReadExactError, SliceWriteError, io_error::Error};

    #[test]
    fn test_read_trait_bounds() {
        fn assert_read<R: Read>(_r: R) {}
        let cursor = Cursor::new(vec![1, 2, 3]);
        assert_read(cursor);
    }

    #[test]
    fn test_write_trait_bounds() {
        fn assert_write<W: Write>(_w: W) {}
        let cursor = Cursor::new(vec![1, 2, 3]);
        assert_write(cursor);
    }

    #[test]
    fn test_seek_trait_bounds() {
        fn assert_seek<S: Seek>(_s: S) {}
        let cursor = Cursor::new(vec![1, 2, 3]);
        assert_seek(cursor);
    }

    #[test]
    fn test_buf_read_trait_bounds() {
        fn assert_buf_read<B: BufRead>(_b: B) {}
        let cursor = Cursor::new(vec![1, 2, 3]);
        assert_buf_read(cursor);
    }

    #[test]
    fn test_read_ready_trait_bounds() {
        fn assert_read_ready<R: ReadReady>(_r: R) {}
        let cursor = Cursor::new(vec![1, 2, 3]);
        assert_read_ready(cursor);
    }

    #[test]
    fn test_write_ready_trait_bounds() {
        fn assert_write_ready<W: WriteReady>(_w: W) {}
        let cursor = Cursor::new(vec![1, 2, 3]);
        assert_write_ready(cursor);
    }

    #[test]
    fn test_read_exact_error_creation() {
        let err: ReadExactError<ErrorKind> = ReadExactError::UnexpectedEof;
        assert!(matches!(err, ReadExactError::UnexpectedEof));

        let err: ReadExactError<ErrorKind> = ReadExactError::Other(ErrorKind::Other);
        assert!(matches!(err, ReadExactError::Other(ErrorKind::Other)));
    }

    #[test]
    fn test_write_fmt_error_creation() {
        use crate::WriteFmtError;
        let err: WriteFmtError<ErrorKind> = WriteFmtError::FmtError;
        assert!(matches!(err, WriteFmtError::FmtError));

        let err: WriteFmtError<ErrorKind> = WriteFmtError::Other(ErrorKind::Other);
        assert!(matches!(err, WriteFmtError::Other(ErrorKind::Other)));
    }
}
