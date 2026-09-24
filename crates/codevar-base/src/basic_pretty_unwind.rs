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

//! Pretty-printing for stack unwind frames.
//!
//! Formats [`Frame`] objects from [`crate::basic_unwind`] into human-readable
//! text using the [`core::fmt::Write`] trait. All helpers write into a
//! caller-provided sink, so this module performs **no heap allocation** and is
//! usable from `no_std` and async-signal-safe contexts (for example writing
//! into a fixed stack buffer inside a signal handler).

use crate::basic_unwind::Frame;
use core::fmt::{self, Write};

/// Formats a single stack frame into `w`.
///
/// The output has the form:
///
/// ```text
///    #0: 0x12345678 0x9abcdef0 +0x100
/// ```
///
/// The `index` parameter is the frame's position in the call stack. When the
/// frame has no module base, the `+0x...` offset portion is omitted.
///
/// # Performance
///
/// Writes only to the provided sink; allocates nothing.
#[inline]
pub fn write_frame<W: Write + ?Sized>(
    w: &mut W,
    frame: &Frame,
    index: usize,
) -> fmt::Result {
    let ip = frame.ip();
    let sp = frame.sp();
    let module_base = frame.module_base_address();

    write!(w, "   {index:>3}: 0x{ip:016x} 0x{sp:016x}")?;

    if let Some(base) = module_base {
        let offset = ip.wrapping_sub(base);
        write!(w, "+0x{offset:x}")?;
    }
    Ok(())
}

/// Formats an iterator of [`Frame`] objects into `w` as multiple lines.
///
/// Each frame is formatted by [`write_frame`], prefixed with `#`, and
/// separated by a newline:
///
/// ```text
/// #   0: 0x12345678 0x9abcdef0 +0x100
/// #   1: 0x12345700 0x9abcde00 +0x200
/// #   2: 0x12345800 0x9abcdd00 +0x300
/// ```
///
/// # Performance
///
/// Writes only to the provided sink; allocates nothing.
#[inline]
pub fn write_frames<'a, W: Write + ?Sized>(
    w: &mut W,
    frames: impl Iterator<Item = &'a Frame>,
) -> fmt::Result {
    for (i, frame) in frames.enumerate() {
        if i > 0 {
            writeln!(w)?;
        }
        w.write_char('#')?;
        write_frame(w, frame, i)?;
    }
    Ok(())
}

/// Writes the symbol address of `frame` into `w` as a 16-digit hex string.
///
/// Uses [`Frame::symbol_address`] and formats the pointer value.
///
/// # Performance
///
/// Writes only to the provided sink; allocates nothing.
#[inline]
pub fn write_symbol_address<W: Write + ?Sized>(w: &mut W, frame: &Frame) -> fmt::Result {
    let symbol_addr = frame.symbol_address() as usize;
    write!(w, "{symbol_addr:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic_unwind::Frame;

    /// Fixed-capacity `fmt::Write` sink backed by a stack buffer (no heap).
    struct StackBuf<const N: usize> {
        buf: [u8; N],
        len: usize,
    }

    impl<const N: usize> StackBuf<N> {
        const fn new() -> Self {
            Self {
                buf: [0; N],
                len: 0,
            }
        }

        fn as_str(&self) -> &str {
            core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
        }
    }

    impl<const N: usize> Write for StackBuf<N> {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            let bytes = s.as_bytes();
            let avail = N - self.len;
            let take = bytes.len().min(avail);
            self.buf[self.len..self.len + take].copy_from_slice(&bytes[..take]);
            self.len += take;
            Ok(())
        }
    }

    fn make_frame(ip: usize, sp: usize, module_base: Option<usize>) -> Frame {
        Frame::new(ip, sp, module_base)
    }

    #[test]
    fn write_frame_with_module_base() {
        let frame = make_frame(0x12345678, 0x9abcdef0, Some(0x12345000));
        let mut out = StackBuf::<256>::new();
        write_frame(&mut out, &frame, 0).unwrap_or(());
        let formatted = out.as_str();
        assert!(formatted.contains("0x0000000012345678"));
        assert!(formatted.contains("0x000000009abcdef0"));
        assert!(formatted.contains("+0x678"));
    }

    #[test]
    fn write_frame_without_module_base() {
        let frame = make_frame(0xdeadbeef, 0xcafebabe, None);
        let mut out = StackBuf::<256>::new();
        write_frame(&mut out, &frame, 1).unwrap_or(());
        let formatted = out.as_str();
        assert!(formatted.contains("0x00000000deadbeef"));
        assert!(formatted.contains("0x00000000cafebabe"));
        assert!(!formatted.contains('+'));
    }

    #[test]
    fn write_frames_multiple() {
        let frames = [
            make_frame(0x1000, 0x2000, None),
            make_frame(0x3000, 0x4000, None),
        ];
        let mut out = StackBuf::<512>::new();
        write_frames(&mut out, frames.iter()).unwrap_or(());
        let formatted = out.as_str();
        let mut lines = formatted.split('\n');
        let first = lines.next().unwrap_or("");
        let second = lines.next().unwrap_or("");
        assert!(lines.next().is_none());
        assert!(first.contains("0x0000000000001000"));
        assert!(second.contains("0x0000000000003000"));
    }

    #[test]
    fn write_symbol_address_non_null() {
        let frame = make_frame(0x1000, 0x2000, None);
        let mut out = StackBuf::<64>::new();
        write_symbol_address(&mut out, &frame).unwrap_or(());
        assert!(out.as_str().contains("1000"));
    }

    #[test]
    fn write_frame_index_padding() {
        let frame = make_frame(0x1, 0x2, None);
        let mut out = StackBuf::<128>::new();
        write_frame(&mut out, &frame, 42).unwrap_or(());
        assert!(out.as_str().contains("   42:"));
    }

    #[test]
    fn write_frame_truncates_safely_when_sink_is_full() {
        let frame = make_frame(0x1, 0x2, None);
        let mut out = StackBuf::<4>::new();
        // Must not panic or overrun; simply stops writing at capacity.
        write_frame(&mut out, &frame, 0).unwrap_or(());
        assert_eq!(out.len, 4);
    }
}
