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

use crate::error::IoResult;
use crate::{
    BufRead, ErrorKind, IoError, Read, ReadExactError, ReadReady, Seek, SeekFrom, Write, WriteFmtError,
    WriteReady,
};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

#[deny(
    clippy::missing_trait_methods,
    reason = "Methods should be forwarded to the underlying type"
)]
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

#[deny(
    clippy::missing_trait_methods,
    reason = "Methods should be forwarded to the underlying type"
)]
impl<T: ?Sized + BufRead> BufRead for &mut T {
    #[inline]
    fn fill_buf(&mut self) -> Result<&[u8], IoError> {
        T::fill_buf(self)
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        T::consume(self, amt);
    }

    #[inline]
    fn has_data_left(&mut self) -> IoResult<bool> {
        T::has_data_left(self)
    }

    #[inline]
    fn read_until(&mut self, byte: u8, buf: &mut Vec<u8>) -> IoResult<usize> {
        T::read_until(self, byte, buf)
    }

    #[inline]
    fn skip_until(&mut self, byte: u8) -> IoResult<usize> {
        T::skip_until(self, byte)
    }

    #[inline]
    fn read_line(&mut self, buf: &mut String) -> IoResult<usize> {
        T::read_line(self, buf)
    }

    #[inline]
    fn split(self, byte: u8) -> Split<Self> {
        Split {
            buf: self,
            delim: byte,
        }
    }

    #[inline]
    fn lines(self) -> Lines<Self> {
        Lines { buf: self }
    }
}

#[deny(
    clippy::missing_trait_methods,
    reason = "Methods should be forwarded to the underlying type"
)]
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

#[deny(
    clippy::missing_trait_methods,
    reason = "Methods should be forwarded to the underlying type"
)]
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

struct Guard<'a> {
    buf: &'a mut Vec<u8>,
    len: usize,
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        unsafe {
            self.buf.set_len(self.len);
        }
    }
}

pub(crate) unsafe fn append_to_string<F>(buf: &mut String, f: F) -> IoResult<usize>
where
    F: FnOnce(&mut Vec<u8>) -> IoResult<usize>,
{
    let mut g = Guard {
        len: buf.len(),
        buf: unsafe { buf.as_mut_vec() },
    };
    let ret = f(g.buf);

    // SAFETY: the caller promises to only append data to `buf`
    let appended = unsafe { g.buf.get_unchecked(g.len..) };
    if str::from_utf8(appended).is_err() {
        ret.and_then(|_| Err(IoError::from(ErrorKind::InvalidData)))
    } else {
        g.len = g.buf.len();
        ret
    }
}

#[derive(Debug)]
pub struct Split<B> {
    pub(crate) buf: B,
    pub(crate) delim: u8,
}

impl<B: BufRead> Iterator for Split<B> {
    type Item = IoResult<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = Vec::new();
        match read_until(&mut self.buf, self.delim, &mut buf) {
            Ok(0) => None,
            Ok(_) => {
                if buf.last() == Some(&self.delim) {
                    let _ = buf.pop();
                }
                Some(Ok(buf))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

pub struct Lines<B> {
    pub(crate) buf: B,
}

impl<B: BufRead> Iterator for Lines<B> {
    type Item = IoResult<String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = String::new();
        match self.buf.read_line(&mut buf) {
            Ok(0) => None,
            Ok(_) => {
                if buf.ends_with('\n') {
                    let _ = buf.pop();
                    if buf.ends_with('\r') {
                        let _ = buf.pop();
                    }
                }
                Some(Ok(buf))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

pub(crate) fn read_until<R: BufRead + ?Sized>(r: &mut R, delim: u8, buf: &mut Vec<u8>) -> IoResult<usize> {
    let mut read = 0;
    loop {
        let (done, used) = {
            let available = r.fill_buf()?;
            match memchr::memchr(delim, available) {
                Some(i) => {
                    buf.extend_from_slice(&available[..=i]);
                    (true, i + 1)
                }
                None => {
                    buf.extend_from_slice(available);
                    (false, available.len())
                }
            }
        };
        r.consume(used);
        read += used;
        if done || used == 0 {
            return Ok(read);
        }
    }
}

pub(crate) fn skip_until<R: BufRead + ?Sized>(r: &mut R, delim: u8) -> IoResult<usize> {
    let mut read = 0;
    loop {
        let (done, used) = {
            let available = match r.fill_buf() {
                Ok(n) => n,
                Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            match memchr::memchr(delim, available) {
                Some(i) => (true, i + 1),
                None => (false, available.len()),
            }
        };
        r.consume(used);
        read += used;
        if done || used == 0 {
            return Ok(read);
        }
    }
}

#[deny(
    clippy::missing_trait_methods,
    reason = "Methods should be forwarded to the underlying type"
)]
impl<T: ?Sized + ReadReady> ReadReady for &mut T {
    #[inline]
    fn read_ready(&mut self) -> Result<bool, Self::Error> {
        T::read_ready(self)
    }
}

#[deny(
    clippy::missing_trait_methods,
    reason = "Methods should be forwarded to the underlying type"
)]
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
