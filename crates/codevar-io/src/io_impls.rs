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

#![deny(
    clippy::missing_trait_methods,
    reason = "Methods should be forwarded to the underlying type"
)]
use core::fmt;

use crate::{
    BufRead, ErrorKind, Read, ReadExactError, ReadReady, Seek, SeekFrom, Write, WriteFmtError, WriteReady,
};

impl<T: ?Sized + Read> Read for &mut T {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        T::read(self, buf)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ReadExactError<Self::Error>> {
        T::read_exact(self, buf)
    }
}

impl<T: ?Sized + BufRead> BufRead for &mut T {
    #[inline]
    fn fill_buf(&mut self) -> Result<&[u8], Self::Error> {
        T::fill_buf(self)
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        T::consume(self, amt);
    }
}

impl<T: ?Sized + Write> Write for &mut T {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        T::write(self, buf)
    }

    #[inline]
    fn flush(&mut self) -> Result<(), Self::Error> {
        T::flush(self)
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error>
    where
        Self::Error: From<ErrorKind>,
    {
        T::write_all(self, buf)
    }

    #[inline]
    fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> Result<(), WriteFmtError<Self::Error>>
    where
        Self::Error: From<ErrorKind>,
    {
        T::write_fmt(self, fmt)
    }
}

impl<T: ?Sized + Seek> Seek for &mut T {
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error> {
        T::seek(self, pos)
    }

    #[inline]
    fn rewind(&mut self) -> Result<(), Self::Error> {
        T::rewind(self)
    }

    #[inline]
    fn stream_position(&mut self) -> Result<u64, Self::Error> {
        T::stream_position(self)
    }

    #[inline]
    fn seek_relative(&mut self, offset: i64) -> Result<(), Self::Error> {
        T::seek_relative(self, offset)
    }
}

impl<T: ?Sized + ReadReady> ReadReady for &mut T {
    #[inline]
    fn read_ready(&mut self) -> Result<bool, Self::Error> {
        T::read_ready(self)
    }
}

impl<T: ?Sized + WriteReady> WriteReady for &mut T {
    #[inline]
    fn write_ready(&mut self) -> Result<bool, Self::Error> {
        T::write_ready(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cursor;

    #[test]
    fn test_read_for_mut() {
        let mut cursor = Cursor::new(vec![1, 2, 3, 4, 5]);
        let mut_ref: &mut dyn Read<Error = _> = &mut cursor;
        let mut buf = [0u8; 3];
        let n = mut_ref
            .read(&mut buf)
            .expect("read should succeed");
        assert_eq!(n, 3);
        assert_eq!(buf, [1, 2, 3]);
    }

    #[test]
    fn test_read_exact_for_mut() {
        let mut cursor = Cursor::new(vec![1, 2, 3, 4, 5]);
        let mut_ref: &mut dyn Read<Error = _> = &mut cursor;
        let mut buf = [0u8; 5];
        mut_ref
            .read_exact(&mut buf)
            .expect("read_exact should succeed");
        assert_eq!(buf, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_buf_read_for_mut() {
        let mut cursor = Cursor::new(vec![1, 2, 3, 4, 5]);
        let mut_ref: &mut dyn BufRead<Error = _> = &mut cursor;
        let buf = mut_ref
            .fill_buf()
            .expect("fill_buf should succeed");
        assert_eq!(buf, &[1, 2, 3, 4, 5]);
        mut_ref.consume(2);
        let buf = mut_ref
            .fill_buf()
            .expect("fill_buf should succeed");
        assert_eq!(buf, &[3, 4, 5]);
    }

    #[test]
    fn test_write_for_mut() {
        let mut buf = [0u8; 10];
        let mut cursor = Cursor::new(&mut buf[..]);
        let mut_ref: &mut dyn Write<Error = _> = &mut cursor;
        let n = mut_ref
            .write(b"Hello")
            .expect("write should succeed");
        assert_eq!(n, 5);
        assert_eq!(&buf[..5], b"Hello");
    }

    #[test]
    fn test_flush_for_mut() {
        let mut buf = [0u8; 10];
        let mut cursor = Cursor::new(&mut buf[..]);
        let mut_ref: &mut dyn Write<Error = _> = &mut cursor;
        mut_ref
            .write(b"Hello")
            .expect("write should succeed");
        mut_ref.flush().expect("flush should succeed");
    }

    #[test]
    fn test_write_all_for_mut() {
        let mut buf = [0u8; 10];
        let mut cursor = Cursor::new(&mut buf[..]);
        let mut_ref: &mut dyn Write<Error = _> = &mut cursor;
        mut_ref
            .write_all(b"Hello")
            .expect("write_all should succeed");
        assert_eq!(&buf[..5], b"Hello");
    }

    #[test]
    fn test_seek_for_mut() {
        let mut cursor = Cursor::new(vec![1, 2, 3, 4, 5]);
        let mut_ref: &mut dyn Seek<Error = _> = &mut cursor;
        let pos = mut_ref
            .seek(SeekFrom::Start(2))
            .expect("seek should succeed");
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_rewind_for_mut() {
        let mut cursor = Cursor::new(vec![1, 2, 3, 4, 5]);
        let mut_ref: &mut dyn Seek<Error = _> = &mut cursor;
        mut_ref
            .seek(SeekFrom::Start(3))
            .expect("seek should succeed");
        mut_ref.rewind().expect("rewind should succeed");
        assert_eq!(
            mut_ref
                .stream_position()
                .expect("stream_position should succeed"),
            0
        );
    }

    #[test]
    fn test_stream_position_for_mut() {
        let mut cursor = Cursor::new(vec![1, 2, 3, 4, 5]);
        let mut_ref: &mut dyn Seek<Error = _> = &mut cursor;
        assert_eq!(
            mut_ref
                .stream_position()
                .expect("stream_position should succeed"),
            0
        );
        mut_ref
            .seek(SeekFrom::Start(2))
            .expect("seek should succeed");
        assert_eq!(
            mut_ref
                .stream_position()
                .expect("stream_position should succeed"),
            2
        );
    }

    #[test]
    fn test_seek_relative_for_mut() {
        let mut cursor = Cursor::new(vec![1, 2, 3, 4, 5]);
        let mut_ref: &mut dyn Seek<Error = _> = &mut cursor;
        mut_ref
            .seek(SeekFrom::Start(2))
            .expect("seek should succeed");
        mut_ref
            .seek_relative(2)
            .expect("seek_relative should succeed");
        assert_eq!(
            mut_ref
                .stream_position()
                .expect("stream_position should succeed"),
            4
        );
    }

    #[test]
    fn test_read_ready_for_mut() {
        let mut cursor = Cursor::new(vec![1, 2, 3]);
        let mut_ref: &mut dyn ReadReady<Error = _> = &mut cursor;
        assert!(
            mut_ref
                .read_ready()
                .expect("read_ready should succeed")
        );
    }

    #[test]
    fn test_write_ready_for_mut() {
        let mut buf = [0u8; 10];
        let mut cursor = Cursor::new(&mut buf[..]);
        let mut_ref: &mut dyn WriteReady<Error = _> = &mut cursor;
        assert!(
            mut_ref
                .write_ready()
                .expect("write_ready should succeed")
        );
    }
}
