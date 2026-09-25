//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Path manipulation, file reading, and `file:` URI conversion.
//!
//! The primary type is [`PathBuf`], an owning, mutable path stored as
//! a [`String`] (UTF-8 paths only). Unlike [`std::path::PathBuf`] this
//! module has no `std` dependency and is fully `no_std` compatible.
//!
//! ## Fluent builder
//!
//! Use [`PathBuilder`] to construct validated paths step by step:
//!
//! ```rust
//! # use codevar_base::basic_pathbuf::PathBuilder;
//! let path = PathBuilder::new()
//!     .root()
//!     .push("usr").unwrap()
//!     .push("local").unwrap()
//!     .file("bin").unwrap()
//!     .with_extension("sh").unwrap()
//!     .build()
//!     .expect("valid path");
//! assert_eq!(path.as_str(), "/usr/local/bin.sh");
//! ```
//!
//! ## Platform-specific file reading
//!
//! Reading files is dispatched to the best available backend for the
//! target:
//!
//! | Target            | Backend                       |
//! |-------------------|------------------------------- |
//! | Unix (incl. macOS)| [`libc`] `open`/`read`/`close` |
//! | Windows           | Win32 `CreateFileW`/`ReadFile` |
//! | other             | [`PathError::Unsupported`]     |
//!
//! See [`read`] and [`read_to_string`].
//!
//! ## URI conversion
//!
//! [`to_file_uri`] percent-encodes a path into a RFC-3986 `file:` URI
//! keeping path separators unescaped, and [`from_file_uri`] reverses
//! that transformation (decoding and normalizing `.`/`..` segments).
//!
//! [`libc`]: https://docs.rs/libc

use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

cfg_if::cfg_if! {
    if #[cfg(all(unix, not(target_arch = "wasm32")))] {
        use libc;
    }
}

/// Maximum file size (16 MiB) accepted by [`read`] and [`read_to_string`].
const MAX_FILE_SIZE: usize = 16 * 1024 * 1024;

/// Initial read buffer size for file I/O.
const READ_BUF_SIZE: usize = 4096;

/// Separator used by the [`PathBuf`] builder.
#[cfg(not(windows))]
pub const MAIN_SEPARATOR: char = '/';

/// Separator used by the [`PathBuf`] builder on Windows.
#[cfg(windows)]
pub const MAIN_SEPARATOR: char = '/';

/// An owning, mutable path stored as a UTF-8 [`String`].
///
/// Provides the same shape of API as [`std::path::PathBuf`] without a
/// `std` dependency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathBuf {
    inner: String,
}

impl PathBuf {
    /// Creates an empty [`PathBuf`].
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { inner: String::new() }
    }

    /// Creates a [`PathBuf`] from an existing [`String`].
    #[inline]
    #[must_use]
    pub const fn from_string(inner: String) -> Self {
        Self { inner }
    }

    /// Creates a [`PathBuf`] from a string slice, validated.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::Empty`] when `path` is empty, or
    /// [`PathError::InvalidCharacter`] when `path` contains a NUL or
    /// control character.
    pub fn from_str(path: &str) -> Result<Self, PathError> {
        validate(path)?;
        Ok(Self {
            inner: path.to_owned(),
        })
    }

    /// Creates a [`PathBuf`] by parsing a string slice.
    ///
    /// This is equivalent to [`from_str`] and enables using `PathBuf`
    /// with the `FromStr` trait.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::Empty`] when `path` is empty, or
    /// [`PathError::InvalidCharacter`] when `path` contains a NUL or
    /// control character.
    pub fn parse_str(path: &str) -> Result<Self, PathError> {
        Self::from_str(path)
    }

    /// Returns the path as a string slice.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.inner
    }

    /// Consumes the [`PathBuf`] and returns the inner [`String`].
    #[inline]
    #[must_use]
    pub fn into_string(self) -> String {
        self.inner
    }

    /// Applies a function to the path if it is valid, returning the
    /// result. If the path is invalid (contains NUL or control
    /// characters), returns the error.
    pub fn and_then<F, U>(self, f: F) -> Result<U, PathError>
    where
        F: FnOnce(Self) -> Result<U, PathError>,
    {
        validate(self.as_str())?;
        f(self)
    }

    /// Returns `true` when the path is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the length of the path in bytes.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Pushes `component` onto the path.
    ///
    /// If `component` is absolute (starts with `/`) it replaces the
    /// current path. Trailing `/` on `component` is stripped.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::InvalidCharacter`] when `component` is not
    /// valid.
    pub fn push(&mut self, component: &str) -> Result<&mut Self, PathError> {
        validate_component(component)?;
        if component.is_empty() {
            return Ok(self);
        }
        if component.starts_with('/') {
            self.inner = component.trim_end_matches('/').to_owned();
        } else {
            if !self.inner.is_empty() && !self.inner.ends_with('/') {
                self.inner.push('/');
            }
            self.inner.push_str(component.trim_end_matches('/'));
        }
        Ok(self)
    }

    /// Appends a path component like [`push`].
    #[inline]
    pub fn join(&mut self, component: &str) -> Result<&mut Self, PathError> {
        self.push(component)
    }

    /// Removes the last component from the path, if any. Returns
    /// `true` when a component was removed.
    #[must_use]
    pub fn pop(&mut self) -> bool {
        if self.inner.is_empty() {
            return false;
        }
        if self.inner == "/" {
            return false;
        }
        let stripped = self.inner.trim_end_matches('/');
        if let Some(idx) = stripped.rfind('/') {
            if idx == 0 {
                self.inner = String::from("/");
            } else {
                self.inner.truncate(idx);
            }
            true
        } else {
            self.inner.clear();
            true
        }
    }

    /// Returns the last component of the path, or `None` when the path
    /// ends with `/` or is empty.
    #[inline]
    #[must_use]
    pub fn file_name(&self) -> Option<&str> {
        let stripped = self.inner.trim_end_matches('/');
        if stripped.is_empty() {
            return None;
        }
        stripped.rfind('/').map(|i| &stripped[i + 1..])
    }

    /// Returns the file stem (the part before the last `.` in the last
    /// component), or `None` when the path has no file name.
    #[inline]
    #[must_use]
    pub fn file_stem(&self) -> Option<&str> {
        let name = self.file_name()?;
        name.rfind('.').map(|i| &name[..i])
    }

    /// Returns the file extension (the part after the last `.` in the
    /// last component), or `None` when there is no extension.
    #[inline]
    #[must_use]
    pub fn extension(&self) -> Option<&str> {
        let name = self.file_name()?;
        name.rfind('.').map(|i| &name[i + 1..])
    }

    /// Sets the last component of the path to `name`.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::InvalidCharacter`] when `name` is empty or
    /// invalid.
    pub fn set_file_name(&mut self, name: &str) -> Result<&mut Self, PathError> {
        validate_component(name)?;
        if name.is_empty() {
            return Err(PathError::Empty);
        }
        if let Some(idx) = self.inner.rfind('/') {
            self.inner.truncate(idx + 1);
        } else {
            self.inner.clear();
        }
        self.inner.push_str(name);
        Ok(self)
    }

    /// Sets the extension of the last component. Returns `true` when
    /// the extension was changed.
    ///
    /// If `ext` is `None` the existing extension is removed.
    /// If there's no existing extension, one is added.
    pub fn set_extension(&mut self, ext: Option<&str>) -> bool {
        match ext {
            None => {
                if let Some(idx) = self.inner.rfind('.') {
                    self.inner.truncate(idx);
                    true
                } else {
                    false
                }
            }
            Some(ext) => {
                if validate_component(ext).is_err() {
                    return false;
                }
                if let Some(idx) = self.inner.rfind('.') {
                    self.inner.truncate(idx);
                    self.inner.push('.');
                    self.inner.push_str(ext);
                    true
                } else {
                    // No existing extension - add one
                    self.inner.push('.');
                    self.inner.push_str(ext);
                    true
                }
            }
        }
    }

    /// Returns a new [`PathBuf`] with the last component replaced by
    /// `name`.
    pub fn with_file_name(&self, name: &str) -> Result<PathBuf, PathError> {
        let mut p = self.clone();
        p.set_file_name(name)?;
        Ok(p)
    }

    /// Returns a new [`PathBuf`] with the extension replaced by `ext`.
    pub fn with_extension(&self, ext: &str) -> Result<PathBuf, PathError> {
        let mut p = self.clone();
        p.set_extension(Some(ext));
        Ok(p)
    }

    /// Returns the parent directory of the path, or `None` when the
    /// path has no parent.
    #[inline]
    #[must_use]
    pub fn parent(&self) -> Option<&str> {
        let stripped = self.inner.trim_end_matches('/');
        if stripped.is_empty() || stripped == "/" {
            return None;
        }
        let idx = stripped.rfind('/')?;
        if idx == 0 {
            Some("/")
        } else {
            Some(&stripped[..idx])
        }
    }

    /// Returns an iterator over the ancestors of the path.
    pub fn ancestors(&self) -> impl Iterator<Item = &str> {
        Ancestors {
            current: self.as_str(),
            done: false,
        }
    }

    /// Returns `true` when the path starts with `/` or a Windows
    /// drive letter prefix such as `C:/`.
    #[inline]
    #[must_use]
    pub fn is_absolute(&self) -> bool {
        self.inner.starts_with('/') || is_windows_drive_absolute(&self.inner)
    }

    /// Returns `true` when the path does not start with a root.
    #[inline]
    #[must_use]
    pub fn is_relative(&self) -> bool {
        !self.is_absolute()
    }

    /// Returns `true` when the path starts with a root component.
    #[inline]
    #[must_use]
    pub fn has_root(&self) -> bool {
        self.is_absolute()
    }

    /// Normalises the path by resolving `.` and `..` segments and
    /// collapsing repeated separators.
    #[must_use]
    pub fn normalize(&self) -> PathBuf {
        let mut segments: Vec<&str> = Vec::new();
        for segment in self.inner.split('/') {
            match segment {
                "" | "." => {}
                ".." => {
                    segments.pop();
                }
                other => segments.push(other),
            }
        }
        if segments.is_empty() {
            return PathBuf::from_string(String::from("/"));
        }
        let mut out = String::with_capacity(self.inner.len());
        for segment in &segments {
            out.push('/');
            out.push_str(segment);
        }
        PathBuf::from_string(out)
    }

    /// Returns an iterator over the path components between separators.
    pub fn components(&self) -> impl Iterator<Item = &str> {
        self.inner.split('/').filter(|s| !s.is_empty())
    }

    /// Returns the number of components in the path.
    #[inline]
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.components().count()
    }
}

impl Default for PathBuf {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AsRef<str> for PathBuf {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for PathBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner)
    }
}

/// Ancestor iterator for [`PathBuf::ancestors`].
struct Ancestors<'a> {
    current: &'a str,
    done: bool,
}

impl<'a> Iterator for Ancestors<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let stripped = self.current.trim_end_matches('/');
        if stripped.is_empty() || stripped == "/" {
            return None;
        }
        let idx = stripped.rfind('/')?;
        let parent = if idx == 0 { "/" } else { &stripped[..idx] };
        self.current = parent;
        if parent == "/" {
            self.done = true;
        }
        Some(parent)
    }
}

/// Fluent builder for constructing a validated [`PathBuf`].
#[derive(Debug, Clone)]
pub struct PathBuilder {
    buf: PathBuf,
    built: bool,
}

impl PathBuilder {
    /// Creates a new [`PathBuilder`].
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: PathBuf::new(),
            built: false,
        }
    }

    /// Sets the path to start from the root (`/`).
    pub fn root(mut self) -> Self {
        self.buf.inner.push('/');
        self
    }

    /// Pushes a path segment.
    pub fn push(mut self, component: &str) -> Result<Self, PathError> {
        validate_component(component)?;
        self.buf.push(component)?;
        Ok(self)
    }

    /// Appends a path segment.
    pub fn join(self, component: &str) -> Result<Self, PathError> {
        self.push(component)
    }

    /// Sets the final component of the path as a file name.
    pub fn file(mut self, name: &str) -> Result<Self, PathError> {
        validate_component(name)?;
        self.buf.push(name)?;
        Ok(self)
    }

    /// Sets the extension of the final component.
    pub fn with_extension(mut self, ext: &str) -> Result<Self, PathError> {
        validate_component(ext)?;
        self.buf.set_extension(Some(ext));
        Ok(self)
    }

    /// Builds the [`PathBuf`], validating and normalising the path.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::Empty`] when the path is empty, or
    /// [`PathError::InvalidCharacter`] when it contains invalid bytes.
    pub fn build(self) -> Result<PathBuf, PathError> {
        if self.built {
            return Err(PathError::InvalidCharacter(0));
        }
        let mut p = self.buf;
        if p.is_empty() {
            return Err(PathError::Empty);
        }
        validate(p.as_str())?;
        p.inner = p.normalize().into_string();
        Ok(p)
    }
}

impl Default for PathBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Validates a full path.
///
/// # Errors
///
/// Returns [`PathError::Empty`] when `path` is empty, or
/// [`PathError::InvalidCharacter`] when `path` contains a NUL or
/// control character.
pub fn validate(path: &str) -> Result<(), PathError> {
    if path.is_empty() {
        return Err(PathError::Empty);
    }
    for (i, byte) in path.bytes().enumerate() {
        if byte == 0 {
            return Err(PathError::InvalidCharacter(i));
        }
        if byte < 0x20 || byte == 0x7f {
            return Err(PathError::InvalidCharacter(i));
        }
    }
    #[cfg(windows)]
    validate_windows_path(path)?;
    Ok(())
}

/// Returns `true` when the path passes [`validate`].
#[must_use]
pub fn is_valid(path: &str) -> bool {
    validate(path).is_ok()
}

/// Validates a single path component.
///
/// Rejects empty strings, `"."`, `".."`, NUL bytes, and control
/// characters. On Windows targets reserved device names are rejected.
pub fn validate_component(name: &str) -> Result<(), PathError> {
    if name.is_empty() {
        return Err(PathError::Empty);
    }
    if name == "." || name == ".." {
        return Err(PathError::ReservedName);
    }
    for (i, byte) in name.bytes().enumerate() {
        if byte == 0 {
            return Err(PathError::InvalidCharacter(i));
        }
        if byte < 0x20 || byte == 0x7f {
            return Err(PathError::InvalidCharacter(i));
        }
    }
    #[cfg(windows)]
    if is_reserved_device_name(name) {
        return Err(PathError::ReservedName);
    }
    Ok(())
}

#[cfg(windows)]
fn validate_windows_path(path: &str) -> Result<(), PathError> {
    if path.ends_with('.') || path.ends_with(' ') {
        return Err(PathError::ReservedName);
    }
    for component in path.split('/') {
        if is_reserved_device_name(component) {
            return Err(PathError::ReservedName);
        }
    }
    Ok(())
}

#[cfg(windows)]
fn is_reserved_device_name(name: &str) -> bool {
    let upper = name.to_uppercase();
    match upper.as_str() {
        "CON" | "PRN" | "AUX" | "NUL" => true,
        _ => {
            if let Some(stripped) = upper.strip_prefix("COM") {
                stripped.len() <= 2 && stripped.bytes().all(|b| b.is_ascii_digit())
            } else if let Some(stripped) = upper.strip_prefix("LPT") {
                stripped.len() <= 2 && stripped.bytes().all(|b| b.is_ascii_digit())
            } else {
                false
            }
        }
    }
}

cfg_if::cfg_if! {
    if #[cfg(all(unix, not(target_arch = "wasm32")))] {
        use crate::basic_pathbuf::unix as sys;
    } else if #[cfg(windows)] {
        use crate::basic_pathbuf::windows as sys;
    } else {
        use crate::basic_pathbuf::unsupported as sys;
    }
}

/// Reads a file into bytes.
///
/// Dispatches to the platform backend. Returns
/// [`PathError::Unsupported`] on targets without a backend.
///
/// # Errors
///
/// Returns [`PathError::Io`] when the file cannot be opened or read.
pub fn read(path: &str) -> Result<Vec<u8>, PathError> {
    sys::read(path)
}

/// Reads a file into a [`String`].
///
/// The file is limited to 16 MiB. Returns
/// [`PathError::NotUtf8`] when the file contents are not valid
/// UTF-8, or [`PathError::Io`] on I/O errors.
///
/// # Errors
///
/// See [`read`].
pub fn read_to_string(path: &str) -> Result<String, PathError> {
    let bytes = sys::read(path)?;
    if bytes.len() > MAX_FILE_SIZE {
        return Err(PathError::Io { op: "read", code: 0 });
    }
    String::from_utf8(bytes).map_err(|_| PathError::NotUtf8)
}

/// Returns `true` when the file at `path` exists and is readable.
pub fn exists(path: &str) -> bool {
    sys::exists(path)
}

/// Returns the current working directory.
pub fn current_dir() -> Result<PathBuf, PathError> {
    sys::current_dir()
}

/// Unix implementation using [`libc`].
#[cfg(all(unix, not(target_arch = "wasm32")))]
mod unix {
    use alloc::vec;
    use super::*;

    pub(super) fn read(path: &str) -> Result<Vec<u8>, PathError> {
        use alloc::ffi::CString;

        let c_path = CString::new(path).map_err(|_| PathError::InvalidCharacter(0))?;
        // SAFETY: `c_path` is a valid NUL-terminated path; the
        // descriptor is checked against -1 before use.
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        if fd < 0 {
            // SAFETY: `fd` is negative, so `errno` describes the
            // failure.
            let errno = unsafe { errno() };
            return Err(PathError::Io {
                op: "open",
                code: errno,
            });
        }
        let mut bytes = Vec::with_capacity(READ_BUF_SIZE);
        let mut buffer = [0u8; READ_BUF_SIZE];
        loop {
            // SAFETY: `buffer` is writable for its full length and
            // `fd` refers to an open file.
            let count = unsafe { libc::read(fd, buffer.as_mut_ptr().cast(), buffer.len()) };
            if count < 0 {
                let errno = unsafe { errno() };
                // SAFETY: the descriptor is open and will be closed.
                unsafe { libc::close(fd) };
                if errno == libc::EINTR {
                    continue;
                }
                return Err(PathError::Io {
                    op: "read",
                    code: errno,
                });
            }
            if count == 0 {
                break;
            }
            let Ok(count) = usize::try_from(count) else {
                // SAFETY: the descriptor is open and will be closed.
                unsafe { libc::close(fd) };
                return Err(PathError::Io { op: "read", code: 0 });
            };
            if bytes.len() + count > MAX_FILE_SIZE {
                // SAFETY: the descriptor is open and will be closed.
                unsafe { libc::close(fd) };
                return Err(PathError::Io { op: "read", code: 0 });
            }
            bytes.extend_from_slice(&buffer[..count]);
        }
        // SAFETY: the descriptor is open and will be closed.
        unsafe { libc::close(fd) };
        Ok(bytes)
    }

    pub(super) fn exists(path: &str) -> bool {
        use alloc::ffi::CString;
        let Ok(c_path) = CString::new(path) else {
            return false;
        };
        // SAFETY: `c_path` is a valid NUL-terminated path; `stat`
        // returns -1 on failure.
        let rc = unsafe { libc::stat(c_path.as_ptr(), core::ptr::null_mut()) };
        rc == 0
    }

    pub(super) fn current_dir() -> Result<PathBuf, PathError> {
        // SAFETY: `getcwd` writes at most 4096 bytes into the buffer.
        let mut buf = vec![0u8; 4096];
        let ptr = unsafe { libc::getcwd(buf.as_mut_ptr().cast(), buf.len()) };
        if ptr.is_null() {
            let errno = unsafe { errno() };
            return Err(PathError::Io {
                op: "getcwd",
                code: errno,
            });
        }
        let len = unsafe { core::ffi::CStr::from_ptr(ptr) }.to_bytes().len();
        // SAFETY: `getcwd` null-terminated the string at `len`.
        unsafe { buf.set_len(len) };
        let s = String::from_utf8(buf).map_err(|_| PathError::NotUtf8)?;
        Ok(PathBuf::from_string(s))
    }

    /// Returns the thread-local errno value on this Unix target.
    ///
    /// # Safety
    ///
    /// Must be called on the same thread, immediately after the
    /// syscall that failed, before any other libc call can clobber
    /// the value.
    unsafe fn errno() -> i32 {
        cfg_if::cfg_if! {
            if #[cfg(target_os = "android")] {
                // SAFETY: `__errno` returns a valid pointer to the
                // calling thread's errno slot; dereferencing is safe
                // on the current thread immediately after a syscall.
                unsafe { *libc::__errno() }
            } else if #[cfg(any(target_os = "linux", target_os = "emscripten"))] {
                // SAFETY: `__errno_location` returns a valid pointer
                // to the calling thread's errno slot (glibc/musl
                // contract).
                unsafe { *libc::__errno_location() }
            } else if #[cfg(any(
                target_os = "macos",
                target_os = "ios",
                target_os = "tvos",
                target_os = "watchos",
                target_os = "visionos",
                target_os = "freebsd",
                target_os = "dragonfly",
            ))] {
                // SAFETY: `__error` returns a valid pointer to the
                // calling thread's errno slot (BSD/libSystem
                // contract).
                unsafe { *libc::__error() }
            } else {
                0
            }
        }
    }
}

/// Windows implementation using the Win32 API.
#[cfg(windows)]
mod windows {
    use super::*;
    use core::ffi::c_void;

    windows_link::link!(
        "kernel32.dll" "system"
        fn CreateFileW(
            lpFileName: *const u16,
            dwDesiredAccess: u32,
            dwShareMode: u32,
            lpSecurityAttributes: *const c_void,
            dwCreationDisposition: u32,
            dwFlagsAndAttributes: u32,
            hTemplateFile: *mut c_void,
        ) -> *mut c_void
    );
    windows_link::link!(
        "kernel32.dll" "system"
        fn ReadFile(
            hFile: *mut c_void,
            lpBuffer: *mut u8,
            nNumberOfBytesToRead: u32,
            lpNumberOfBytesRead: *mut u32,
            lpOverlapped: *mut c_void,
        ) -> i32
    );
    windows_link::link!(
        "kernel32.dll" "system"
        fn CloseHandle(hObject: *mut c_void) -> i32
    );
    windows_link::link!(
        "kernel32.dll" "system"
        fn GetCurrentDirectoryW(nBufferLength: u32, lpBuffer: *mut u16) -> u32
    );
    windows_link::link!(
        "kernel32.dll" "system"
        fn GetLastError() -> u32
    );

    const GENERIC_READ: u32 = 0x8000_0000;
    const OPEN_EXISTING: u32 = 3;
    const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
    const INVALID_HANDLE: isize = -1;

    pub(super) fn read(path: &str) -> Result<Vec<u8>, PathError> {
        let utf16 = to_utf16(path);
        // SAFETY: `utf16` is a valid null-terminated UTF-16LE path.
        let handle = unsafe {
            CreateFileW(
                utf16.as_ptr(),
                GENERIC_READ,
                0,
                core::ptr::null(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                core::ptr::null_mut(),
            )
        };
        if handle as isize == INVALID_HANDLE {
            let code = unsafe { GetLastError() };
            return Err(PathError::Io {
                op: "CreateFileW",
                code: code as i32,
            });
        }
        let mut bytes = Vec::with_capacity(READ_BUF_SIZE);
        let mut buffer = [0u8; READ_BUF_SIZE];
        loop {
            let mut read = 0u32;
            // SAFETY: `handle` is valid, `buffer` is writable.
            let ok = unsafe {
                ReadFile(
                    handle,
                    buffer.as_mut_ptr(),
                    buffer.len() as u32,
                    &mut read,
                    core::ptr::null_mut(),
                )
            };
            if ok == 0 || read == 0 {
                // SAFETY: the handle is valid and will be closed.
                unsafe { CloseHandle(handle) };
                if read == 0 {
                    break;
                }
                let code = unsafe { GetLastError() };
                return Err(PathError::Io {
                    op: "ReadFile",
                    code: code as i32,
                });
            }
            if bytes.len() + read as usize > MAX_FILE_SIZE {
                // SAFETY: the handle is valid and will be closed.
                unsafe { CloseHandle(handle) };
                return Err(PathError::Io {
                    op: "ReadFile",
                    code: 0,
                });
            }
            bytes.extend_from_slice(&buffer[..read as usize]);
        }
        Ok(bytes)
    }

    pub(super) fn exists(path: &str) -> bool {
        read(path).is_ok()
    }

    pub(super) fn current_dir() -> Result<PathBuf, PathError> {
        // SAFETY: `GetCurrentDirectoryW` writes at most `nBufferLength`
        // UTF-16 code units into `lpBuffer`.
        let mut buf = vec![0u16; 4096];
        let len = unsafe { GetCurrentDirectoryW(buf.len() as u32, buf.as_mut_ptr()) };
        if len == 0 || len as usize >= buf.len() {
            let code = unsafe { GetLastError() };
            return Err(PathError::Io {
                op: "GetCurrentDirectoryW",
                code: code as i32,
            });
        }
        // SAFETY: `len` code units form a valid UTF-16LE string.
        let utf16 = &buf[..len as usize];
        let s = String::from_utf16(utf16).map_err(|_| PathError::NotUtf8)?;
        Ok(PathBuf::from_string(s))
    }

    fn to_utf16(s: &str) -> Vec<u16> {
        let mut out: Vec<u16> = Vec::with_capacity(s.len());
        encode_utf16::encode_utf16(s.as_bytes(), &mut out);
        out.push(0);
        out
    }
}

/// UTF-16 encoding helper for Windows.
#[cfg(windows)]
mod encode_utf16 {
    pub(super) fn encode_utf16(input: &[u8], out: &mut Vec<u16>) {
        let mut i = 0;
        while i < input.len() {
            let byte = input[i];
            if byte < 0x80 {
                out.push(byte as u16);
                i += 1;
            } else if byte < 0xe0 {
                if i + 1 >= input.len() {
                    break;
                }
                let ch = ((byte & 0x1f) as u32) << 6 | ((input[i + 1] & 0x3f) as u32);
                out.push(ch as u16);
                i += 2;
            } else if byte < 0xf0 {
                if i + 2 >= input.len() {
                    break;
                }
                let ch = ((byte & 0x0f) as u32) << 12
                    | ((input[i + 1] & 0x3f) as u32) << 6
                    | ((input[i + 2] & 0x3f) as u32);
                out.push(ch as u16);
                i += 3;
            } else {
                if i + 3 >= input.len() {
                    break;
                }
                let ch = ((byte & 0x07) as u32) << 18
                    | ((input[i + 1] & 0x3f) as u32) << 12
                    | ((input[i + 2] & 0x3f) as u32) << 6
                    | ((input[i + 3] & 0x3f) as u32);
                let cp = ch as u32;
                let lead = 0xd800 + ((cp - 0x10000) >> 10);
                let trail = 0xdc00 + ((cp - 0x10000) & 0x3ff);
                out.push(lead as u16);
                out.push(trail as u16);
                i += 4;
            }
        }
    }
}

/// Fallback for targets without a file I/O backend (e.g. `wasm32`).
#[cfg(not(any(all(unix, not(target_arch = "wasm32")), windows,)))]
mod unsupported {
    use super::*;

    pub(super) fn read(_path: &str) -> Result<Vec<u8>, PathError> {
        Err(PathError::Unsupported)
    }

    pub(super) fn exists(_path: &str) -> bool {
        false
    }

    pub(super) fn current_dir() -> Result<PathBuf, PathError> {
        Err(PathError::Unsupported)
    }
}

/// Converts a Windows-style path to a [`PathBuf`] with `/` separators.
#[cfg(windows)]
pub fn from_windows_path(path: &str) -> Result<PathBuf, PathError> {
    validate(path)?;
    let normalized = path.replace('\\', "/");
    Ok(PathBuf::from_string(normalized))
}

#[cfg(not(windows))]
fn is_windows_drive_absolute(_path: &str) -> bool {
    false
}

#[cfg(windows)]
fn is_windows_drive_absolute(path: &str) -> bool {
    let mut chars = path.chars();
    match (chars.next(), chars.next(), chars.next()) {
        (Some(a), Some(b), Some(c)) if c == ':' && b == '/' => a.is_ascii_alphabetic(),
        _ => false,
    }
}

// ──── URI conversion ─────────────────────────────────────────

/// Characters that may appear unescaped in a `file:` URI path segment.
const fn is_uri_path_safe(byte: u8) -> bool {
    matches!(
        byte,
        b'/'
            | b'!'
            | b'$'
            | b'&'
            | b'\''
            | b'('
            | b')'
            | b'*'
            | b'+'
            | b','
            | b'-'
            | b'.'
            | b'0'..=b'9'
            | b':'
            | b'='
            | b'@'
            | b'A'..=b'Z'
            | b'_'
            | b'a'..=b'z'
            | b'~'
    )
}

/// Converts an absolute path to a `file:` URI.
///
/// Percent-encodes every byte outside the safe set, keeping `/`
/// unescaped as the path separator.
///
/// # Errors
///
/// Returns [`PathError::NotAbsolute`] when `path` is not absolute, or
/// [`PathError::InvalidCharacter`] when the path contains a NUL or
/// control character.
pub fn to_file_uri(path: &str) -> Result<String, PathError> {
    validate(path)?;
    let normalized = normalize_for_uri(path);
    if !normalized.is_absolute() {
        return Err(PathError::NotAbsolute);
    }
    let mut uri = String::with_capacity("file://".len() + normalized.len());
    uri.push_str("file://");
    for &byte in normalized.as_str().as_bytes() {
        if is_uri_path_safe(byte) {
            uri.push(char::from(byte));
        } else {
            uri.push('%');
            uri.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
            uri.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
        }
    }
    Ok(uri)
}

/// Converts a `file:` URI back to a [`PathBuf`].
///
/// Percent-decodes the path and resolves `.`/`..` segments. Returns
/// `None` for URIs without a `file` scheme or with a malformed path.
pub fn from_file_uri(uri: &str) -> Result<PathBuf, PathError> {
    let (scheme, rest) = uri.split_once(':').ok_or(PathError::InvalidUri)?;
    if !scheme.eq_ignore_ascii_case("file") {
        return Err(PathError::InvalidUri);
    }
    let raw_path = match rest.strip_prefix("//") {
        Some(after) if after.starts_with('/') => after,
        Some(after) => &after[after.find('/').ok_or(PathError::InvalidUri)?..],
        None if rest.starts_with('/') => rest,
        None => return Err(PathError::InvalidUri),
    };
    let end = raw_path.find(['?', '#']).unwrap_or(raw_path.len());
    let decoded = decode_percent_encoding(&raw_path[..end])?;
    let path = PathBuf::from_string(normalize(&decoded));
    validate(path.as_str())?;
    Ok(path)
}

/// Normalises a path for URI conversion.
#[cfg(not(windows))]
fn normalize_for_uri(path: &str) -> PathBuf {
    PathBuf::from_string(path.to_owned())
}

/// Normalises a path for URI conversion (Windows: backslashes → `/`, uppercase drive).
#[cfg(windows)]
fn normalize_for_uri(path: &str) -> PathBuf {
    let normalized: String = path
        .replace('\\', "/")
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if i == 0 && c.is_ascii_alphabetic() {
                c.to_ascii_uppercase()
            } else {
                c
            }
        })
        .collect();
    PathBuf::from_string(normalized)
}

/// Percent-decodes a URI path string.
fn decode_percent_encoding(value: &str) -> Result<String, PathError> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            let byte = bytes[index];
            if byte == 0 {
                return Err(PathError::InvalidCharacter(index));
            }
            // Allow literal '/' as path separator (only encoded '/' is rejected)
            decoded.push(byte);
            index += 1;
            continue;
        }
        let high =
            hex_digit(*bytes.get(index + 1).ok_or(PathError::InvalidUri)?).ok_or(PathError::InvalidUri)?;
        let low =
            hex_digit(*bytes.get(index + 2).ok_or(PathError::InvalidUri)?).ok_or(PathError::InvalidUri)?;
        let byte = high << 4 | low;
        if byte == 0 || byte == b'/' {
            return Err(PathError::InvalidCharacter(index));
        }
        decoded.push(byte);
        index += 3;
    }
    String::from_utf8(decoded).map_err(|_| PathError::NotUtf8)
}

/// Returns the value of a hexadecimal digit byte.
fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Normalises a path string by resolving `.`/`..` segments.
fn normalize(path: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    if segments.is_empty() {
        return String::from("/");
    }
    let mut out = String::with_capacity(path.len());
    for segment in &segments {
        out.push('/');
        out.push_str(segment);
    }
    out
}

/// Uppercase hex digits used to render one byte as `%XX`.
const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

/// Errors produced by path construction, validation, file I/O, and
/// URI conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathError {
    /// The path string is empty.
    Empty,
    /// A byte outside the allowed set; holds the byte offset.
    InvalidCharacter(usize),
    /// The path is not absolute when an absolute path was required.
    NotAbsolute,
    /// A reserved device name (Windows) or an invalid component name.
    ReservedName,
    /// An I/O operation failed. `op` names the operation; `code` is
    /// the platform error code.
    Io { op: &'static str, code: i32 },
    /// The file contents are not valid UTF-8.
    NotUtf8,
    /// The URI is malformed or does not use the `file` scheme.
    InvalidUri,
    /// The platform has no file I/O backend.
    Unsupported,
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("path is empty"),
            Self::InvalidCharacter(offset) => {
                write!(f, "invalid character at byte offset {offset}")
            }
            Self::NotAbsolute => f.write_str("path is not absolute"),
            Self::ReservedName => f.write_str("reserved name or invalid component"),
            Self::Io { op, code } => {
                write!(f, "`{op}` failed (code {code})")
            }
            Self::NotUtf8 => f.write_str("path is not valid UTF-8"),
            Self::InvalidUri => f.write_str("invalid file URI"),
            Self::Unsupported => f.write_str("file I/O not supported on this target"),
        }
    }
}

impl core::error::Error for PathError {}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_from_str_and_as_str() {
        let p = PathBuf::from_str("/usr/local/bin").unwrap();
        assert_eq!(p.as_str(), "/usr/local/bin");
    }

    #[test]
    fn test_from_str_rejects_empty() {
        assert!(matches!(PathBuf::from_str(""), Err(PathError::Empty)));
    }

    #[test]
    fn test_from_str_rejects_nul() {
        assert!(matches!(
            PathBuf::from_str("/foo\x00bar"),
            Err(PathError::InvalidCharacter(_))
        ));
    }

    #[test]
    fn test_new_is_empty() {
        assert!(PathBuf::new().is_empty());
    }
    #[test]
    fn test_push_builds_path() {
        let mut p = PathBuf::new();
        p.push("usr").unwrap();
        p.push("local").unwrap();
        p.push("bin").unwrap();
        assert_eq!(p.as_str(), "usr/local/bin");
    }

    #[test]
    fn test_push_absolute_replaces() {
        let mut p = PathBuf::new();
        p.push("usr").unwrap();
        p.push("/local").unwrap();
        assert_eq!(p.as_str(), "/local");
    }

    #[test]
    fn test_push_strips_trailing_slash() {
        let mut p = PathBuf::new();
        p.push("usr/").unwrap();
        assert_eq!(p.as_str(), "usr");
    }

    #[test]
    fn test_join_is_same_as_push() {
        let mut p = PathBuf::new();
        p.join("foo").unwrap();
        p.join("bar").unwrap();
        assert_eq!(p.as_str(), "foo/bar");
    }
    #[test]
    fn test_pop_removes_component() {
        let mut p = PathBuf::from_str("/usr/local/bin").unwrap();
        assert!(p.pop());
        assert_eq!(p.as_str(), "/usr/local");
    }

    #[test]
    fn test_pop_root_returns_false() {
        let mut p = PathBuf::from_str("/").unwrap();
        assert!(!p.pop());
    }
    #[test]
    fn test_file_name() {
        let p = PathBuf::from_str("/usr/local/bin.rs").unwrap();
        assert_eq!(p.file_name(), Some("bin.rs"));
    }

    #[test]
    fn test_file_name_empty() {
        let p = PathBuf::from_str("/").unwrap();
        assert_eq!(p.file_name(), None);
    }

    #[test]
    fn test_file_stem() {
        let p = PathBuf::from_str("/usr/local/bin.rs").unwrap();
        assert_eq!(p.file_stem(), Some("bin"));
    }

    #[test]
    fn test_extension() {
        let p = PathBuf::from_str("/usr/local/bin.rs").unwrap();
        assert_eq!(p.extension(), Some("rs"));
    }

    #[test]
    fn test_extension_none() {
        let p = PathBuf::from_str("/usr/local/bin").unwrap();
        assert_eq!(p.extension(), None);
    }

    #[test]
    fn test_set_file_name() {
        let mut p = PathBuf::from_str("/usr/local/bin").unwrap();
        p.set_file_name("app").unwrap();
        assert_eq!(p.as_str(), "/usr/local/app");
    }

    #[test]
    fn test_set_extension() {
        let mut p = PathBuf::from_str("/usr/local/bin.rs").unwrap();
        assert!(p.set_extension(Some("txt")));
        assert_eq!(p.as_str(), "/usr/local/bin.txt");
    }

    #[test]
    fn test_set_extension_remove() {
        let mut p = PathBuf::from_str("/usr/local/bin.rs").unwrap();
        assert!(p.set_extension(None));
        assert_eq!(p.as_str(), "/usr/local/bin");
    }

    #[test]
    fn test_with_file_name() {
        let p = PathBuf::from_str("/usr/local/bin").unwrap();
        let q = p.with_file_name("app").unwrap();
        assert_eq!(q.as_str(), "/usr/local/app");
    }

    #[test]
    fn test_with_extension() {
        let p = PathBuf::from_str("/usr/local/bin.rs").unwrap();
        let q = p.with_extension("txt").unwrap();
        assert_eq!(q.as_str(), "/usr/local/bin.txt");
    }

    #[test]
    fn test_parent() {
        let p = PathBuf::from_str("/usr/local/bin").unwrap();
        assert_eq!(p.parent(), Some("/usr/local"));
    }

    #[test]
    fn test_parent_root() {
        let p = PathBuf::from_str("/").unwrap();
        assert_eq!(p.parent(), None);
    }
    #[test]
    fn test_ancestors() {
        let p = PathBuf::from_str("/usr/local/bin").unwrap();
        let mut iter = p.ancestors();
        assert_eq!(iter.next(), Some("/usr/local"));
        assert_eq!(iter.next(), Some("/usr"));
        assert_eq!(iter.next(), Some("/"));
        assert_eq!(iter.next(), None);
    }
    #[test]
    fn test_is_absolute_with_slash() {
        let p = PathBuf::from_str("/usr/local").unwrap();
        assert!(p.is_absolute());
    }

    #[test]
    fn test_is_relative() {
        let p = PathBuf::from_str("usr/local").unwrap();
        assert!(!p.is_absolute());
    }
    #[test]
    fn test_normalize_dots() {
        let p = PathBuf::from_str("/usr/./local/../bin").unwrap();
        assert_eq!(p.normalize().as_str(), "/usr/bin");
    }

    #[test]
    fn test_normalize_double_dots_escape() {
        let p = PathBuf::from_str("/usr/../../bin").unwrap();
        assert_eq!(p.normalize().as_str(), "/bin");
    }

    #[test]
    fn test_normalize_empty_is_root() {
        let p = PathBuf::new();
        assert_eq!(p.normalize().as_str(), "/");
    }

    #[test]
    fn test_normalize_collapses_separators() {
        let p = PathBuf::from_str("//usr///local").unwrap();
        assert_eq!(p.normalize().as_str(), "/usr/local");
    }

    #[test]
    fn test_components() {
        let p = PathBuf::from_str("/usr/local/bin").unwrap();
        let v: Vec<&str> = p.components().collect();
        assert_eq!(v, ["usr", "local", "bin"]);
    }

    #[test]
    fn test_component_count() {
        let p = PathBuf::from_str("/usr/local/bin").unwrap();
        assert_eq!(p.component_count(), 3);
    }
    #[test]
    fn test_builder_simple() {
        let p = PathBuilder::new()
            .root()
            .push("a")
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(p.as_str(), "/a");
    }

    #[test]
    fn test_builder_two_push() {
        let p = PathBuilder::new()
            .root()
            .push("a")
            .unwrap()
            .push("b")
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(p.as_str(), "/a/b");
    }

    #[test]
    fn test_builder_file() {
        let p = PathBuilder::new()
            .root()
            .push("a")
            .unwrap()
            .file("b")
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(p.as_str(), "/a/b");
    }

    #[test]
    fn test_builder_extension() {
        let p = PathBuilder::new()
            .root()
            .push("a")
            .unwrap()
            .file("b")
            .unwrap()
            .with_extension("txt")
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(p.as_str(), "/a/b.txt");
    }

    #[test]
    fn test_builder_builds_path() {
        let p = PathBuilder::new()
            .root()
            .push("usr")
            .unwrap()
            .push("local")
            .unwrap()
            .file("bin")
            .unwrap()
            .with_extension("sh")
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(p.as_str(), "/usr/local/bin.sh");
    }

    #[test]
    fn test_builder_rejects_empty() {
        assert!(matches!(PathBuilder::new().build(), Err(PathError::Empty)));
    }

    #[test]
    fn test_builder_rejects_invalid_component() {
        assert!(matches!(
            PathBuilder::new().root().push("foo\x00bar"),
            Err(PathError::InvalidCharacter(_))
        ));
    }
    #[test]
    fn test_validate_accepts_valid() {
        assert!(is_valid("/usr/local/bin"));
    }

    #[test]
    fn test_validate_rejects_empty() {
        assert!(matches!(validate(""), Err(PathError::Empty)));
    }

    #[test]
    fn test_validate_rejects_nul() {
        assert!(matches!(
            validate("/foo\x00bar"),
            Err(PathError::InvalidCharacter(4))
        ));
    }

    #[test]
    fn test_validate_rejects_control() {
        assert!(matches!(
            validate("/foo\x01bar"),
            Err(PathError::InvalidCharacter(4))
        ));
    }

    #[test]
    fn test_validate_component_rejects_dotdot() {
        assert!(matches!(validate_component(".."), Err(PathError::ReservedName)));
    }

    #[test]
    fn test_validate_component_rejects_empty() {
        assert!(matches!(validate_component(""), Err(PathError::Empty)));
    }
    #[test]
    fn test_to_file_uri() {
        let uri = to_file_uri("/usr/local/bin").unwrap();
        assert_eq!(uri, "file:///usr/local/bin");
    }

    #[test]
    fn test_to_file_uri_encodes_special() {
        let uri = to_file_uri("/usr/local/hello world").unwrap();
        assert!(uri.contains("%20"));
    }

    #[test]
    fn test_to_file_uri_rejects_relative() {
        assert!(matches!(to_file_uri("usr/local"), Err(PathError::NotAbsolute)));
    }

    #[test]
    fn test_from_file_uri() {
        let p = from_file_uri("file:///usr/local/bin").unwrap();
        assert_eq!(p.as_str(), "/usr/local/bin");
    }

    #[test]
    fn test_from_file_uri_decodes() {
        let p = from_file_uri("file:///usr/local/hello%20world").unwrap();
        assert_eq!(p.as_str(), "/usr/local/hello world");
    }

    #[test]
    fn test_from_file_uri_rejects_non_file() {
        assert!(matches!(
            from_file_uri("http://example.com/"),
            Err(PathError::InvalidUri)
        ));
    }

    #[test]
    fn test_round_trip_uri() {
        let path = "/usr/local/bin";
        let uri = to_file_uri(path).unwrap();
        let p = from_file_uri(&uri).unwrap();
        assert_eq!(p.as_str(), "/usr/local/bin");
    }

    // ── current_dir (Unix only) ──

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_current_dir_returns_path() {
        let dir = current_dir().unwrap();
        assert!(dir.is_absolute());
    }
}
