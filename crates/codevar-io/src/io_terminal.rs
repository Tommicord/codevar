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

//! Terminal detection utilities for standard I/O streams.
//!
//! This module provides the [`IsTerminal`] trait for detecting whether a given
//! stream is connected to a terminal (TTY). This is useful for enabling
//! ANSI colors and interactive features only when appropriate.
//!
//! # Example
//!
//! ```
//! use codevar_io::{Stdout, IsTerminal};
//!
//! let stdout = Stdout::new();
//! if stdout.is_terminal() {
//!     println!("Terminal detected - enabling colors");
//! } else {
//!     println!("Not a terminal - disabling colors");
//! }
//! ```

#[cfg(all(target_family = "unix", not(target_os = "wasi"), not(target_os = "hermit")))]
mod unix_impls {
    use super::IsTerminal;
    use crate::{Stderr, Stdin, Stdout};
    use libc::{STDERR_FILENO, STDIN_FILENO, STDOUT_FILENO};

    impl IsTerminal for Stdin {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDIN_FILENO) != 0 }
        }
    }

    impl IsTerminal for Stdout {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDOUT_FILENO) != 0 }
        }
    }

    impl IsTerminal for Stderr {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDERR_FILENO) != 0 }
        }
    }

    impl<'a> IsTerminal for crate::io_unix::StdinLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDIN_FILENO) != 0 }
        }
    }

    impl<'a> IsTerminal for crate::io_unix::StdoutLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDOUT_FILENO) != 0 }
        }
    }

    impl<'a> IsTerminal for crate::io_unix::StderrLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDERR_FILENO) != 0 }
        }
    }
}

#[cfg(target_os = "wasi")]
mod wasi_impls {
    use super::IsTerminal;
    use crate::{Stderr, Stdin, Stdout};
    use libc::{STDERR_FILENO, STDIN_FILENO, STDOUT_FILENO};

    impl IsTerminal for Stdin {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDIN_FILENO) != 0 }
        }
    }

    impl IsTerminal for Stdout {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDOUT_FILENO) != 0 }
        }
    }

    impl IsTerminal for Stderr {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDERR_FILENO) != 0 }
        }
    }

    impl<'a> IsTerminal for crate::io_wasi::StdinLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDIN_FILENO) != 0 }
        }
    }

    impl<'a> IsTerminal for crate::io_wasi::StdoutLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDOUT_FILENO) != 0 }
        }
    }

    impl<'a> IsTerminal for crate::io_wasi::StderrLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            unsafe { libc::isatty(STDERR_FILENO) != 0 }
        }
    }
}

#[cfg(target_os = "hermit")]
mod hermit_impls {
    use super::IsTerminal;
    use crate::{Stderr, Stdin, Stdout};
    use hermit_abi::{STDERR_FILENO, STDIN_FILENO, STDOUT_FILENO};

    impl IsTerminal for Stdin {
        #[inline]
        fn is_terminal(&self) -> bool {
            hermit_abi::isatty(STDIN_FILENO)
        }
    }

    impl IsTerminal for Stdout {
        #[inline]
        fn is_terminal(&self) -> bool {
            hermit_abi::isatty(STDOUT_FILENO)
        }
    }

    impl IsTerminal for Stderr {
        #[inline]
        fn is_terminal(&self) -> bool {
            hermit_abi::isatty(STDERR_FILENO)
        }
    }
}

#[cfg(windows)]
mod windows_impls {
    use super::IsTerminal;
    use crate::{Stderr, Stdin, Stdout};
    use windows_sys::Win32::{
        Foundation::{HANDLE, INVALID_HANDLE_VALUE},
        Storage::FileSystem::{FILE_TYPE_PIPE, FileNameInfo, GetFileInformationByHandleEx, GetFileType},
        System::Console::GetConsoleMode,
    };

    const STD_INPUT_HANDLE: u32 = 0xFFFFFFF6;
    const STD_OUTPUT_HANDLE: u32 = 0xFFFFFFF5;
    const STD_ERROR_HANDLE: u32 = 0xFFFFFFF4;

    unsafe extern "system" {
        fn GetStdHandle(n_std_handle: u32) -> *mut core::ffi::c_void;
    }

    fn get_handle(handle_id: u32) -> Option<*mut core::ffi::c_void> {
        let handle = unsafe { GetStdHandle(handle_id) };
        if handle.is_null() || handle as isize == INVALID_HANDLE_VALUE {
            None
        } else {
            Some(handle)
        }
    }

    fn is_console(handle: *mut core::ffi::c_void) -> bool {
        let mut mode = 0;
        unsafe { GetConsoleMode(handle, &mut mode) != 0 }
    }

    fn msys_tty_on(handle: HANDLE) -> bool {
        // Early return if the handle is not a pipe.
        if unsafe { GetFileType(handle) } != FILE_TYPE_PIPE {
            return false;
        }

        /// Mirrors windows_sys::Win32::Storage::FileSystem::FILE_NAME_INFO, giving
        /// it a fixed length that we can stack allocate
        #[repr(C)]
        #[allow(non_snake_case)]
        struct FILE_NAME_INFO {
            FileNameLength: u32,
            FileName: [u16; MAX_PATH as usize],
        }
        let mut name_info = FILE_NAME_INFO {
            FileNameLength: 0,
            FileName: [0; MAX_PATH as usize],
        };
        // Safety: buffer length is fixed.
        let res = unsafe {
            GetFileInformationByHandleEx(
                handle,
                FileNameInfo,
                &mut name_info as *mut _ as *mut core::ffi::c_void,
                core::mem::size_of::<FILE_NAME_INFO>() as u32,
            )
        };
        if res == 0 {
            return false;
        }

        // Use `get` because `FileNameLength` can be out of range.
        let s = match name_info
            .FileName
            .get(..name_info.FileNameLength as usize / 2)
        {
            None => return false,
            Some(s) => s,
        };
        let name = alloc::string::String::from_utf16_lossy(s);
        // Get the file name only.
        let name = name.rsplit('\\').next().unwrap_or(&name);
        // This checks whether 'pty' exists in the file name, which indicates that
        // a pseudo-terminal is attached. To mitigate against false positives
        // (e.g., an actual file name that contains 'pty'), we also require that
        // the file name begins with either the strings 'msys-' or 'cygwin-'.)
        let is_msys = name.starts_with("msys-") || name.starts_with("cygwin-");
        let is_pty = name.contains("-pty");
        is_msys && is_pty
    }

    impl IsTerminal for Stdin {
        #[inline]
        fn is_terminal(&self) -> bool {
            get_handle(STD_INPUT_HANDLE)
                .map(is_console)
                .unwrap_or(false)
        }
    }

    impl IsTerminal for Stdout {
        #[inline]
        fn is_terminal(&self) -> bool {
            get_handle(STD_OUTPUT_HANDLE)
                .map(is_console)
                .unwrap_or(false)
        }
    }

    impl IsTerminal for Stderr {
        #[inline]
        fn is_terminal(&self) -> bool {
            get_handle(STD_ERROR_HANDLE)
                .map(|h| {
                    is_console(h) || {
                        // Check for MSYS/Cygwin pty
                        let mut mode = 0;
                        if unsafe { GetConsoleMode(h, &mut mode) } == 0 {
                            msys_tty_on(h as HANDLE)
                        } else {
                            false
                        }
                    }
                })
                .unwrap_or(false)
        }
    }

    impl<'a> IsTerminal for crate::io_windows::StdinLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            Stdin::new().is_terminal()
        }
    }

    impl<'a> IsTerminal for crate::io_windows::StdoutLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            Stdout::new().is_terminal()
        }
    }

    impl<'a> IsTerminal for crate::io_windows::StderrLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            Stderr::new().is_terminal()
        }
    }
}

#[cfg(not(any(
    target_family = "unix",
    target_os = "windows",
    target_os = "wasi",
    target_os = "hermit"
)))]
mod fallback_impls {
    use super::IsTerminal;
    use crate::{Stderr, Stdin, Stdout};

    impl IsTerminal for Stdin {
        #[inline]
        fn is_terminal(&self) -> bool {
            false
        }
    }

    impl IsTerminal for Stdout {
        #[inline]
        fn is_terminal(&self) -> bool {
            false
        }
    }

    impl IsTerminal for Stderr {
        #[inline]
        fn is_terminal(&self) -> bool {
            false
        }
    }

    impl<'a> IsTerminal for crate::io_fallback::StdinLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            false
        }
    }

    impl<'a> IsTerminal for crate::io_fallback::StdoutLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            false
        }
    }

    impl<'a> IsTerminal for crate::io_fallback::StderrLock<'a> {
        #[inline]
        fn is_terminal(&self) -> bool {
            false
        }
    }
}

/// Extension trait to check whether a stream is a terminal.
pub trait IsTerminal {
    /// Returns `true` if this stream is a terminal.
    ///
    /// # Example
    ///
    /// ```
    /// use codevar_io::{Stdout, IsTerminal};
    ///
    /// let stdout = Stdout::new();
    /// if stdout.is_terminal() {
    ///     println!("stdout is a terminal");
    /// }
    /// ```
    fn is_terminal(&self) -> bool;
}

/// Returns `true` if `stream` is a terminal.
///
/// This is a convenience function equivalent to calling `stream.is_terminal()`.
///
/// # Example
///
/// ```
/// use codevar_io::{Stdout, IsTerminal, is_terminal};
///
/// let stdout = Stdout::new();
/// if is_terminal(stdout) {
///     println!("stdout is a terminal");
/// }
/// ```
pub fn is_terminal<T: IsTerminal>(stream: T) -> bool {
    stream.is_terminal()
}

#[cfg(test)]
mod tests {
    use super::IsTerminal;

    #[test]
    #[cfg(all(target_family = "unix", not(target_os = "wasi")))]
    fn stdin() {
        use crate::io_unix::Stdin;
        assert_eq!(
            unsafe { libc::isatty(libc::STDIN_FILENO) != 0 },
            Stdin::new().is_terminal()
        )
    }

    #[test]
    #[cfg(all(target_family = "unix", not(target_os = "wasi")))]
    fn stdout() {
        use crate::io_unix::Stdout;
        assert_eq!(
            unsafe { libc::isatty(libc::STDOUT_FILENO) != 0 },
            Stdout::new().is_terminal()
        )
    }

    #[test]
    #[cfg(all(target_family = "unix", not(target_os = "wasi")))]
    fn stderr() {
        use crate::io_unix::Stderr;
        assert_eq!(
            unsafe { libc::isatty(libc::STDERR_FILENO) != 0 },
            Stderr::new().is_terminal()
        )
    }
}
