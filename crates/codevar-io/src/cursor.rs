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

use alloc::vec::Vec;
use core::cmp;
use core::fmt;

use crate::error::IoResult;
use crate::{
    BufRead, ErrorType, Read, ReadExactError, ReadReady, Seek, SeekFrom, SliceWriteError, Write, WriteReady,
};

/// A cursor that wraps an in-memory buffer and provides `Seek`, `Read`, `Write`, and `BufRead`
/// implementations.
///
/// This is the `codevar-io` equivalent of [`std::io::Cursor`].
///
/// Cursors are used with in-memory buffers, anything implementing `AsRef<[u8]>`, to allow them to
/// implement `Read` and/or `Write`, allowing these buffers to be used anywhere you might use a
/// reader or writer that does actual I/O.
///
/// The cursor maintains an internal position, initially 0, which can be moved using `Seek`.
/// Reading and writing operations advance the position accordingly.
///
/// # Examples
///
/// ```rust
/// # use codevar_io::{Cursor, Read, Write, Seek, SeekFrom};
/// let mut cursor = Cursor::new(vec![1, 2, 3, 4, 5]);
/// let mut buf = [0u8; 3];
/// cursor.read_exact(&mut buf).expect("read_exact should succeed");
/// assert_eq!(buf, [1, 2, 3]);
///
/// cursor.seek(SeekFrom::Start(0)).expect("seek should succeed");
/// cursor.write(&[6, 7]).expect("write should succeed");
/// assert_eq!(cursor.into_inner(), vec![6, 7, 3, 4, 5]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cursor<T> {
    inner: T,
    pos: u64,
}

impl<T> Cursor<T> {
    /// Creates a new cursor wrapping the provided underlying in-memory buffer.
    ///
    /// Cursor initial position is 0 even if underlying buffer (e.g., `Vec`) is not empty.
    /// So writing to cursor starts with overwriting `Vec` content, not with appending to it.
    pub const fn new(inner: T) -> Self {
        Self { inner, pos: 0 }
    }

    /// Consumes this cursor, returning the underlying value.
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Gets a reference to the underlying value in this cursor.
    pub const fn get_ref(&self) -> &T {
        &self.inner
    }

    /// Gets a mutable reference to the underlying value in this cursor.
    ///
    /// Care should be taken to avoid modifying the internal I/O state of the underlying value
    /// in a way that would break the cursor's position tracking.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Returns the current position of this cursor.
    pub const fn position(&self) -> u64 {
        self.pos
    }

    /// Sets the position of this cursor.
    ///
    /// # Panics
    ///
    /// Panics if `pos > u64::MAX` (which can never happen on 64-bit platforms).
    pub fn set_position(&mut self, pos: u64) {
        self.pos = pos;
    }
}

impl<T> Cursor<T>
where
    T: AsRef<[u8]>,
{
    /// Returns the remaining slice from the current position.
    fn remaining_slice(&self) -> &[u8] {
        let pos = self.pos as usize;
        let slice = self.inner.as_ref();
        if pos >= slice.len() { &[] } else { &slice[pos..] }
    }
}

impl<T> Cursor<T>
where
    T: AsMut<[u8]>,
{
    /// Returns the remaining mutable slice from the current position.
    fn remaining_slice_mut(&mut self) -> &mut [u8] {
        let pos = self.pos as usize;
        let slice = self.inner.as_mut();
        if pos >= slice.len() {
            &mut []
        } else {
            &mut slice[pos..]
        }
    }
}

impl ErrorType for Cursor<&[u8]> {
    type Error = core::convert::Infallible;
}

impl ErrorType for Cursor<&mut [u8]> {
    type Error = SliceWriteError;
}

impl ErrorType for Cursor<Vec<u8>> {
    type Error = core::convert::Infallible;
}

impl Read for Cursor<&[u8]> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let remaining = self.remaining_slice();
        let amt = cmp::min(buf.len(), remaining.len());
        if amt > 0 {
            buf[..amt].copy_from_slice(&remaining[..amt]);
            self.pos += amt as u64;
        }
        Ok(amt)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ReadExactError<Self::Error>> {
        let remaining = self.remaining_slice();
        if remaining.len() < buf.len() {
            return Err(ReadExactError::UnexpectedEof);
        }
        self.read(buf)?;
        Ok(())
    }
}

impl Read for Cursor<&mut [u8]> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let remaining = self.remaining_slice();
        let amt = cmp::min(buf.len(), remaining.len());
        if amt > 0 {
            buf[..amt].copy_from_slice(&remaining[..amt]);
            self.pos += amt as u64;
        }
        Ok(amt)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ReadExactError<Self::Error>> {
        let remaining = self.remaining_slice();
        if remaining.len() < buf.len() {
            return Err(ReadExactError::UnexpectedEof);
        }
        self.read(buf)?;
        Ok(())
    }
}

impl Read for Cursor<Vec<u8>> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let remaining = self.remaining_slice();
        let amt = cmp::min(buf.len(), remaining.len());
        if amt > 0 {
            buf[..amt].copy_from_slice(&remaining[..amt]);
            self.pos += amt as u64;
        }
        Ok(amt)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ReadExactError<Self::Error>> {
        let remaining = self.remaining_slice();
        if remaining.len() < buf.len() {
            return Err(ReadExactError::UnexpectedEof);
        }
        self.read(buf)?;
        Ok(())
    }
}

impl BufRead for Cursor<&[u8]> {
    #[inline]
    fn fill_buf(&mut self) -> IoResult<&[u8]> {
        Ok(self.remaining_slice())
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        self.pos = self.pos.saturating_add(amt as u64);
    }
}

// Implement BufRead for Cursor<&mut [u8]>
impl BufRead for Cursor<&mut [u8]> {
    #[inline]
    fn fill_buf(&mut self) -> IoResult<&[u8]> {
        Ok(self.remaining_slice())
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        self.pos = self.pos.saturating_add(amt as u64);
    }
}

impl BufRead for Cursor<Vec<u8>> {
    #[inline]
    fn fill_buf(&mut self) -> IoResult<&[u8]> {
        Ok(self.remaining_slice())
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        self.pos = self.pos.saturating_add(amt as u64);
    }
}

impl Write for Cursor<&mut [u8]> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        let remaining = self.remaining_slice_mut();
        let amt = cmp::min(buf.len(), remaining.len());
        if !buf.is_empty() && amt == 0 {
            return Err(SliceWriteError::Full);
        }
        if amt > 0 {
            remaining[..amt].copy_from_slice(&buf[..amt]);
            self.pos += amt as u64;
        }
        Ok(amt)
    }

    #[inline]
    fn flush(&mut self) -> Result<(), Self::Error> {
        // In-memory buffer doesn't need flushing
        Ok(())
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error> {
        let remaining = self.remaining_slice_mut();
        if remaining.len() < buf.len() {
            return Err(SliceWriteError::Full);
        }
        self.write(buf)?;
        Ok(())
    }
}

impl Write for Cursor<Vec<u8>> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        let pos = self.pos as usize;
        let slice = &mut self.inner;
        if pos + buf.len() > slice.len() {
            slice.resize(pos + buf.len(), 0);
        }
        slice[pos..pos + buf.len()].copy_from_slice(buf);
        self.pos += buf.len() as u64;
        Ok(buf.len())
    }

    #[inline]
    fn flush(&mut self) -> Result<(), Self::Error> {
        // In-memory buffer doesn't need flushing
        Ok(())
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error> {
        self.write(buf)?;
        Ok(())
    }
}

impl Seek for Cursor<&[u8]> {
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error> {
        let len = self.inner.as_ref().len() as u64;
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset,
            SeekFrom::End(offset) => len.saturating_add_signed(offset),
            SeekFrom::Current(offset) => self.pos.saturating_add_signed(offset),
        };
        self.pos = new_pos.min(len);
        Ok(self.pos)
    }
}

impl Seek for Cursor<&mut [u8]> {
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error> {
        let len = self.inner.as_ref().len() as u64;
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset,
            SeekFrom::End(offset) => len.saturating_add_signed(offset),
            SeekFrom::Current(offset) => self.pos.saturating_add_signed(offset),
        };
        self.pos = new_pos.min(len);
        Ok(self.pos)
    }
}

impl Seek for Cursor<Vec<u8>> {
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error> {
        let len = self.inner.len() as u64;
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset,
            SeekFrom::End(offset) => len.saturating_add_signed(offset),
            SeekFrom::Current(offset) => self.pos.saturating_add_signed(offset),
        };
        self.pos = new_pos;
        Ok(self.pos)
    }
}

impl ReadReady for Cursor<&[u8]> {
    #[inline]
    fn read_ready(&mut self) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

impl ReadReady for Cursor<&mut [u8]> {
    #[inline]
    fn read_ready(&mut self) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

impl ReadReady for Cursor<Vec<u8>> {
    #[inline]
    fn read_ready(&mut self) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

impl WriteReady for Cursor<&mut [u8]> {
    #[inline]
    fn write_ready(&mut self) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

impl WriteReady for Cursor<Vec<u8>> {
    #[inline]
    fn write_ready(&mut self) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

// Display implementation for Cursor
impl<T: fmt::Debug> fmt::Display for Cursor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cursor {{ pos: {}, inner: {:?} }}", self.pos, self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_new() {
        let cursor = Cursor::new(vec![1, 2, 3]);
        assert_eq!(cursor.position(), 0);
        assert_eq!(cursor.get_ref(), &vec![1, 2, 3]);
    }

    #[test]
    fn test_cursor_into_inner() {
        let cursor = Cursor::new(vec![1, 2, 3]);
        let inner = cursor.into_inner();
        assert_eq!(inner, vec![1, 2, 3]);
    }

    #[test]
    fn test_cursor_get_mut() {
        let mut cursor = Cursor::new(vec![1, 2, 3]);
        cursor.get_mut()[0] = 10;
        assert_eq!(cursor.get_ref()[0], 10);
    }

    #[test]
    fn test_cursor_set_position() {
        let mut cursor = Cursor::new(vec![1, 2, 3, 4, 5]);
        cursor.set_position(2);
        assert_eq!(cursor.position(), 2);
    }

    #[test]
    fn test_read_slice() {
        let data = b"Hello, World!";
        let mut cursor = Cursor::new(&data[..]);
        let mut buf = [0u8; 5];
        let n = cursor
            .read(&mut buf)
            .expect("read should succeed");
        assert_eq!(n, 5);
        assert_eq!(&buf, b"Hello");

        let n = cursor
            .read(&mut buf)
            .expect("read should succeed");
        assert_eq!(n, 5);
        assert_eq!(&buf, b", Wor");

        let n = cursor
            .read(&mut buf)
            .expect("read should succeed");
        assert_eq!(n, 3);
        assert_eq!(&buf[..3], b"ld!");

        let n = cursor
            .read(&mut buf)
            .expect("read should succeed");
        assert_eq!(n, 0);
    }

    #[test]
    fn test_read_exact_slice() {
        let data = b"Hello, World!";
        let mut cursor = Cursor::new(&data[..]);
        let mut buf = [0u8; 5];
        cursor
            .read_exact(&mut buf)
            .expect("read_exact should succeed");
        assert_eq!(&buf, b"Hello");

        cursor
            .read_exact(&mut buf)
            .expect("read_exact should succeed");
        assert_eq!(&buf, b", Wor");

        let mut buf2 = [0u8; 3];
        cursor
            .read_exact(&mut buf2)
            .expect("read_exact should succeed");
        assert_eq!(&buf2, b"ld!");

        // Should fail on EOF
        let mut buf3 = [0u8; 1];
        let result = cursor.read_exact(&mut buf3);
        assert!(result.is_err());
        assert!(matches!(result, Err(ReadExactError::UnexpectedEof)));
    }

    #[test]
    fn test_read_mut_slice() {
        let mut buf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut cursor = Cursor::new(&mut buf[..]);
        let mut dest = [0u8; 5];
        let n = cursor
            .read(&mut dest)
            .expect("read should succeed");
        assert_eq!(n, 5);
        assert_eq!(dest, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_write_slice() {
        let mut buf = [0u8; 20];
        let mut cursor = Cursor::new(&mut buf[..]);
        let n = cursor
            .write(b"Hello")
            .expect("write should succeed");
        assert_eq!(n, 5);

        let n = cursor
            .write(b" World")
            .expect("write should succeed");
        assert_eq!(n, 6);

        // Check contents after all writes
        assert_eq!(&buf[..11], b"Hello World");
    }

    #[test]
    fn test_write_all_slice() {
        let mut buf = [0u8; 10];
        {
            let mut cursor = Cursor::new(&mut buf[..]);
            cursor
                .write_all(b"Hello")
                .expect("write_all should succeed");
        } // cursor goes out of scope here
        assert_eq!(&buf[..5], b"Hello");

        // Should fail when buffer is full (trying to write 6 bytes but only 5 remaining)
        // Use the same cursor position by seeking to position 5
        let mut cursor = Cursor::new(&mut buf[..]);
        cursor
            .seek(SeekFrom::Start(5))
            .expect("seek should succeed");
        let result = cursor.write_all(b" World!");
        assert!(result.is_err());
        assert!(matches!(result, Err(SliceWriteError::Full)));
    }

    #[test]
    fn test_write_vec() {
        let mut cursor = Cursor::new(Vec::new());
        cursor
            .write(b"Hello")
            .expect("write should succeed");
        assert_eq!(cursor.into_inner(), b"Hello");
    }

    #[test]
    fn test_write_all_vec() {
        let mut cursor = Cursor::new(vec![1, 2, 3, 4, 5]);
        cursor
            .seek(SeekFrom::Start(0))
            .expect("seek should succeed");
        cursor
            .write(&[6, 7])
            .expect("write should succeed");
        assert_eq!(cursor.into_inner(), vec![6, 7, 3, 4, 5]);
    }

    #[test]
    fn test_seek_start() {
        let data = b"Hello, World!";
        let mut cursor = Cursor::new(&data[..]);
        let pos = cursor
            .seek(SeekFrom::Start(7))
            .expect("seek should succeed");
        assert_eq!(pos, 7);
        let mut buf = [0u8; 5];
        cursor
            .read_exact(&mut buf)
            .expect("read_exact should succeed");
        assert_eq!(&buf, b"World");
    }

    #[test]
    fn test_seek_current() {
        let data = b"Hello, World!";
        let mut cursor = Cursor::new(&data[..]);
        cursor
            .seek(SeekFrom::Start(7))
            .expect("seek should succeed");
        let pos = cursor
            .seek(SeekFrom::Current(-5))
            .expect("seek should succeed");
        // Seeking from position 7 by -5 should go to position 2
        assert_eq!(pos, 2);
        let mut buf = [0u8; 5];
        cursor
            .read_exact(&mut buf)
            .expect("read_exact should succeed");
        // Position 2 in "Hello, World!" is the second 'l', so we get "llo, "
        assert_eq!(&buf, b"llo, ");
    }

    #[test]
    fn test_seek_end() {
        let data = b"Hello, World!";
        let mut cursor = Cursor::new(&data[..]);
        let pos = cursor
            .seek(SeekFrom::End(-6))
            .expect("seek should succeed");
        assert_eq!(pos, 7);
        let mut buf = [0u8; 5];
        cursor
            .read_exact(&mut buf)
            .expect("read_exact should succeed");
        assert_eq!(&buf, b"World");
    }

    #[test]
    fn test_rewind() {
        let data = b"Hello, World!";
        let mut cursor = Cursor::new(&data[..]);
        cursor
            .seek(SeekFrom::Start(7))
            .expect("seek should succeed");
        cursor.rewind().expect("rewind should succeed");
        assert_eq!(cursor.position(), 0);
    }

    #[test]
    fn test_stream_position() {
        let data = b"Hello, World!";
        let mut cursor = Cursor::new(&data[..]);
        assert_eq!(
            cursor
                .stream_position()
                .expect("stream_position should succeed"),
            0
        );
        cursor
            .read(&mut [0u8; 5])
            .expect("read should succeed");
        assert_eq!(
            cursor
                .stream_position()
                .expect("stream_position should succeed"),
            5
        );
        cursor
            .seek(SeekFrom::Start(10))
            .expect("seek should succeed");
        assert_eq!(
            cursor
                .stream_position()
                .expect("stream_position should succeed"),
            10
        );
    }

    #[test]
    fn test_seek_relative() {
        let data = b"Hello, World!";
        let mut cursor = Cursor::new(&data[..]);
        cursor
            .seek(SeekFrom::Start(5))
            .expect("seek should succeed");
        cursor
            .seek_relative(3)
            .expect("seek_relative should succeed");
        assert_eq!(cursor.position(), 8);

        cursor
            .seek_relative(-4)
            .expect("seek_relative should succeed");
        assert_eq!(cursor.position(), 4);
    }

    #[test]
    fn test_buf_read_slice() {
        let data = b"Hello, World!";
        let mut cursor = Cursor::new(&data[..]);
        let buf = cursor
            .fill_buf()
            .expect("fill_buf should succeed");
        assert_eq!(buf, b"Hello, World!");

        cursor.consume(7);
        let buf = cursor
            .fill_buf()
            .expect("fill_buf should succeed");
        assert_eq!(buf, b"World!");

        cursor.consume(6);
        let buf = cursor
            .fill_buf()
            .expect("fill_buf should succeed");
        assert_eq!(buf, b"");
    }

    #[test]
    fn test_read_ready_write_ready_slice() {
        let data = b"Hello";
        let mut cursor = Cursor::new(&data[..]);
        assert!(
            cursor
                .read_ready()
                .expect("read_ready should succeed")
        );

        let mut buf = [0u8; 10];
        let mut cursor = Cursor::new(&mut buf[..]);
        assert!(
            cursor
                .write_ready()
                .expect("write_ready should succeed")
        );
    }

    #[test]
    fn test_cursor_display() {
        let cursor = Cursor::new(vec![1, 2, 3]);
        let display_str = format!("{}", cursor);
        assert!(display_str.contains("Cursor"));
        assert!(display_str.contains("pos: 0"));
    }

    #[test]
    fn test_cursor_seek_beyond_end_slice() {
        let data = b"Hello";
        let mut cursor = Cursor::new(&data[..]);
        let pos = cursor
            .seek(SeekFrom::Start(10))
            .expect("seek should succeed");
        // Position is clamped to length
        assert_eq!(pos, 5);
    }

    #[test]
    fn test_cursor_seek_beyond_end_vec() {
        let mut cursor = Cursor::new(vec![1, 2, 3]);
        let pos = cursor
            .seek(SeekFrom::Start(10))
            .expect("seek should succeed");
        // Position can go beyond end for Vec
        assert_eq!(pos, 10);

        // Writing beyond end should extend the buffer
        cursor
            .write(&[4, 5])
            .expect("write should succeed");
        assert_eq!(cursor.into_inner(), vec![1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 4, 5]);
    }
}
