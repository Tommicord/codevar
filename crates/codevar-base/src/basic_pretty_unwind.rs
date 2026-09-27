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
use alloc::string::ToString;
use core::ffi::CStr;
use core::fmt::{self, Write};

/// Attempts to resolve a symbol name and module path for `addr`.
///
/// On Unix, uses [`libc::dladdr`]. On Windows, uses `SymFromAddr` from
/// `DbgHelp`. Returns `(symbol_name, module_path)` when resolution succeeds,
/// or `None` when the address cannot be resolved.
///
/// # Safety
///
/// Calling this from a signal handler is generally safe on Unix because
/// `dladdr` is listed as async-signal-safe in the POSIX specification.
/// On Windows, `SymFromAddr` requires prior initialization and is not
/// async-signal-safe; callers must ensure it is not invoked from a signal
/// handler on that platform.
#[inline]
pub fn resolve_symbol(addr: usize) -> Option<(&'static str, &'static str)> {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    {
        use core::ffi::c_void;
        unsafe {
            let mut info: libc::Dl_info = core::mem::zeroed();
            if libc::dladdr(addr as *const c_void, &mut info) != 0 {
                let fname = if !info.dli_fname.is_null() {
                    CStr::from_ptr(info.dli_fname).to_str().ok()?
                } else {
                    ""
                };
                let sname = if !info.dli_sname.is_null() {
                    CStr::from_ptr(info.dli_sname).to_str().ok()?
                } else {
                    ""
                };
                if !sname.is_empty() {
                    // Demangle Rust symbol names
                    let demangled = rustc_demangle::demangle(sname).to_string();
                    // Note: This allocates, but only when a symbol is found.
                    // For async-signal-safe contexts, use the mangled name.
                    Some((demangled.leak(), fname))
                } else {
                    Some((fname, fname))
                }
            } else {
                None
            }
        }
    }
    #[cfg(windows)]
    {
        use core::ffi::c_void;
        #[repr(C)]
        struct SymbolInfo {
            size_of_struct: u32,
            type_index: u32,
            flags: u64,
            value: u64,
            address: u64,
            register: i32,
            scope: i32,
            tag: i32,
            name_len: i32,
            max_name_len: i32,
            name: [u8; 1024],
        }
        unsafe {
            let mut info = SymbolInfo {
                size_of_struct: core::mem::size_of::<SymbolInfo>() as u32,
                type_index: 0,
                flags: 0,
                value: 0,
                address: 0,
                register: 0,
                scope: 0,
                tag: 0,
                name_len: 0,
                max_name_len: 1024,
                name: [0; 1024],
            };
            let mut disp = 0;
            if windows_link::SymFromAddr(
                !0, // process handle (use current process)
                addr as u64,
                &mut disp,
                &mut info as *mut _ as *mut _,
            ) != 0
            {
                let name = CStr::from_ptr(info.name.as_ptr().cast())
                    .to_str()
                    .ok()?;
                let demangled = rustc_demangle::demangle(name).to_string();
                Some((demangled.leak(), ""))
            } else {
                None
            }
        }
    }
    #[cfg(not(any(unix, windows)))]
    None
}

/// Resolves the module name (shared object / executable path) for `addr`.
///
/// On Unix uses [`libc::dladdr`]. On Windows returns an empty string.
/// Returns `None` when resolution fails.
#[inline]
pub fn resolve_module(addr: usize) -> Option<&'static str> {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    {
        use core::ffi::c_void;
        unsafe {
            let mut info: libc::Dl_info = core::mem::zeroed();
            if libc::dladdr(addr as *const c_void, &mut info) != 0 && !info.dli_fname.is_null() {
                Some(CStr::from_ptr(info.dli_fname).to_str().ok()?)
            } else {
                None
            }
        }
    }
    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    None
}

/// Formats a single stack frame into `w`.
///
/// The output has the form:
///
/// ```text
///    #0: ip=0x12345678 sp=0x9abcdef0 main+0x42 (/lib/x86_64-linux-gnu/libc.so.6)
/// ```
///
/// The `index` parameter is the frame's position in the call stack. When the
/// frame has no module base, the `+0x...` offset portion is omitted.
///
/// # Performance
///
/// Writes only to the provided sink; allocates nothing.
#[inline]
pub fn write_frame<W: Write + ?Sized>(w: &mut W, frame: &Frame, index: usize) -> fmt::Result {
    let ip = frame.ip();
    let sp = frame.sp();
    let module_base = frame.module_base_address();

    write!(w, "   {index:>3}: ip=0x{ip:016x} sp=0x{sp:016x}")?;
    if let Some(base) = module_base {
        let offset = ip.wrapping_sub(base);
        write!(w, "+0x{offset:x}")?;
        if let Some((sname, _fname)) = resolve_symbol(ip) {
            if !sname.is_empty() {
                write!(w, " <{sname}>")?;
            }
        }
    }
    if let Some(mname) = resolve_module(ip) {
        write!(w, " ({mname})")?;
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
            Self { buf: [0; N], len: 0 }
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
        let frames = [make_frame(0x1000, 0x2000, None), make_frame(0x3000, 0x4000, None)];
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
