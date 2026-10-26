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

#![allow(async_fn_in_trait)]

use crate::{ErrorKind, ErrorType, ReadExactError, SeekFrom};

/// Async reader.
///
/// This trait is the `codevar-io` async equivalent of [`std::io::Read`].
pub trait AsyncRead: ErrorType {
    /// Read some bytes from this source into the specified buffer, returning how many bytes were read.
    ///
    /// If no bytes are currently available to read:
    /// - The method waits until at least one byte becomes available;
    /// - Once at least one (or more) bytes become available, a non-zero amount of those is copied to the
    ///   beginning of `buf`, and the amount is returned, *without waiting any further for more bytes to
    ///   become available*.
    ///
    /// If bytes are available to read:
    /// - A non-zero amount of bytes is read to the beginning of `buf`, and the amount is returned immediately,
    ///   *without waiting for more bytes to become available*;
    ///
    /// Note that once some bytes are available to read, it is *not* guaranteed that all available bytes are returned.
    /// It is possible for the implementation to read an amount of bytes less than `buf.len()` while there are more
    /// bytes immediately available.
    ///
    /// This waiting behavior is important for the cases where `Read` represents the "read" leg of a pipe-like
    /// protocol (a socket, a pipe, a serial line etc.). The semantics is that the caller - by passing a non-empty
    /// buffer - does expect _some_ data (one or more bytes) - but _not necessarily `buf.len()` or more bytes_ -
    /// to become available, before the peer represented by `Read` would stop sending bytes due to
    /// application-specific reasons (as in the peer waiting for a response to the data it had sent so far).
    ///
    /// If the reader is at end-of-file (EOF), `Ok(0)` is returned. There is no guarantee that a reader at EOF
    /// will always be so in the future, for example a reader can stop being at EOF if another process appends
    /// more bytes to the underlying file.
    ///
    /// If `buf.len() == 0`, `read` returns without waiting, with either `Ok(0)` or an error.
    /// The `Ok(0)` doesn't indicate EOF, unlike when called with a non-empty buffer.
    ///
    /// Implementations are encouraged to make this function side-effect-free on cancel (AKA "cancel-safe"), i.e.
    /// guarantee that if you cancel (drop) a `read()` future that hasn't completed yet, the stream's
    /// state hasn't changed (no bytes have been read).
    ///
    /// This is not a requirement to allow implementations that read into the user's buffer straight from
    /// the hardware with e.g. DMA.
    ///
    /// Implementations should document whether they're actually side-effect-free on cancel or not.
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error>;

    /// Read the exact number of bytes required to fill `buf`.
    ///
    /// This function calls `read()` in a loop until exactly `buf.len()` bytes have
    /// been read, waiting if needed.
    ///
    /// This function is not side-effect-free on cancel (AKA "cancel-safe"), i.e. if you cancel (drop) a returned
    /// future that hasn't completed yet, some bytes might have already been read, which will get lost.
    async fn read_exact(&mut self, mut buf: &mut [u8]) -> Result<(), ReadExactError<Self::Error>> {
        while !buf.is_empty() {
            match self.read(buf).await {
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

/// Async buffered reader.
///
/// This trait is the `codevar-io` async equivalent of [`std::io::BufRead`].
pub trait AsyncBufRead: AsyncRead {
    /// Return the contents of the internal buffer, filling it with more data from the inner reader if it is empty.
    ///
    /// If no bytes are currently available to read, this function waits until at least one byte is available.
    ///
    /// If the reader is at end-of-file (EOF), an empty slice is returned. There is no guarantee that a reader at EOF
    /// will always be so in the future, for example a reader can stop being at EOF if another process appends
    /// more bytes to the underlying file.
    async fn fill_buf(&mut self) -> Result<&[u8], Self::Error>;

    /// Tell this buffer that `amt` bytes have been consumed from the buffer, so they should no longer be returned in calls to `fill_buf`.
    fn consume(&mut self, amt: usize);
}

/// Async writer.
///
/// This trait is the `codevar-io` async equivalent of [`std::io::Write`].
pub trait AsyncWrite: ErrorType {
    /// Write a buffer into this writer, returning how many bytes were written.
    ///
    /// If the writer is not currently ready to accept more bytes (for example, its buffer is full),
    /// this function waits until it is ready to accept least one byte.
    ///
    /// If it's ready to accept bytes, a non-zero amount of bytes is written from the beginning of `buf`, and the amount
    /// is returned. It is not guaranteed that *all* available buffer space is filled, i.e. it is possible for the
    /// implementation to write an amount of bytes less than `buf.len()` while the writer continues to be
    /// ready to accept more bytes immediately.
    ///
    /// Implementations should never return `Ok(0)` when `buf.len() != 0`. Situations where the writer is not
    /// able to accept more bytes and likely never will are better indicated with errors.
    ///
    /// If `buf.len() == 0`, `write` returns without waiting, with either `Ok(0)` or an error.
    /// The `Ok(0)` doesn't indicate an error.
    ///
    /// Implementations are encouraged to make this function side-effect-free on cancel (AKA "cancel-safe"), i.e.
    /// guarantee that if you cancel (drop) a `write()` future that hasn't completed yet, the stream's
    /// state hasn't changed (no bytes have been written).
    ///
    /// This is not a requirement to allow implementations that write from the user's buffer straight to
    /// the hardware with e.g. DMA.
    ///
    /// Implementations should document whether they're actually side-effect-free on cancel or not.
    async fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error>;

    /// Flush this output stream, ensuring that all intermediately buffered contents reach their destination.
    async fn flush(&mut self) -> Result<(), Self::Error>;

    /// Write an entire buffer into this writer.
    ///
    /// This function calls `write()` in a loop until exactly `buf.len()` bytes have
    /// been written, waiting if needed.
    ///
    /// This function is not side-effect-free on cancel (AKA "cancel-safe"), i.e. if you cancel (drop) a returned
    /// future that hasn't completed yet, some bytes might have already been written.
    ///
    /// Returns an error if `write()` returns `Ok(0)` when `buf` is not empty,
    /// indicating that the writer is not able to accept more bytes.
    async fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error>
    where
        Self::Error: From<ErrorKind>,
    {
        let mut buf = buf;
        while !buf.is_empty() {
            match self.write(buf).await {
                Ok(0) => return Err(ErrorKind::WriteZero.into()),
                Ok(n) => buf = &buf[n..],
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

/// Async seek within streams.
///
/// This trait is the `codevar-io` async equivalent of [`std::io::Seek`].
pub trait AsyncSeek: ErrorType {
    /// Seek to an offset, in bytes, in a stream.
    async fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error>;

    /// Rewind to the beginning of a stream.
    async fn rewind(&mut self) -> Result<(), Self::Error> {
        self.seek(SeekFrom::Start(0)).await?;
        Ok(())
    }

    /// Returns the current seek position from the start of the stream.
    async fn stream_position(&mut self) -> Result<u64, Self::Error> {
        self.seek(SeekFrom::Current(0)).await
    }
}

/// Get whether an async reader is ready.
///
/// This allows using a [`AsyncRead`] or [`AsyncBufRead`] in a nonblocking fashion, i.e. trying to read
/// only when it is ready.
pub trait AsyncReadReady: ErrorType {
    /// Get whether the reader is ready for immediately reading.
    ///
    /// This usually means that there is either some bytes have been received and are buffered and ready to be read,
    /// or that the reader is at EOF.
    ///
    /// If this returns `true`, it's guaranteed that the next call to [`AsyncRead::read`] or [`AsyncBufRead::fill_buf`] will not wait.
    async fn read_ready(&mut self) -> Result<bool, Self::Error>;
}

/// Get whether an async writer is ready.
///
/// This allows using a [`AsyncWrite`] in a nonblocking fashion, i.e. trying to write
/// only when it is ready.
pub trait AsyncWriteReady: ErrorType {
    /// Get whether the writer is ready for immediately writing.
    ///
    /// This usually means that there is free space in the internal transmit buffer.
    ///
    /// If this returns `true`, it's guaranteed that the next call to [`AsyncWrite::write`] will not wait.
    async fn write_ready(&mut self) -> Result<bool, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_async_read_trait_bounds() {
        fn assert_async_read<R: AsyncRead>(_r: R) {}
        // Can't easily test without an implementation, but verify trait compiles
    }

    #[test]
    fn test_async_write_trait_bounds() {
        fn assert_async_write<W: AsyncWrite>(_w: W) {}
    }

    #[test]
    fn test_async_seek_trait_bounds() {
        fn assert_async_seek<S: AsyncSeek>(_s: S) {}
    }

    #[test]
    fn test_async_buf_read_trait_bounds() {
        fn assert_async_buf_read<B: AsyncBufRead>(_b: B) {}
    }

    #[test]
    fn test_async_read_ready_trait_bounds() {
        fn assert_async_read_ready<R: AsyncReadReady>(_r: R) {}
    }

    #[test]
    fn test_async_write_ready_trait_bounds() {
        fn assert_async_write_ready<W: AsyncWriteReady>(_w: W) {}
    }

    #[test]
    fn test_read_exact_error_creation() {
        let err: ReadExactError<ErrorKind> = ReadExactError::UnexpectedEof;
        assert!(matches!(err, ReadExactError::UnexpectedEof));

        let err: ReadExactError<ErrorKind> = ReadExactError::Other(ErrorKind::Other);
        assert!(matches!(err, ReadExactError::Other(ErrorKind::Other)));
    }
}
