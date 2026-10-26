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

//! Raw dynamic loading of OpenCL client and vendor libraries.
//!
//! This is the `no_std` counterpart of what `libloading` does: a thin wrapper
//! over `dlopen`/`dlsym` on unix and `LoadLibraryA`/`GetProcAddress` on
//! Windows, with no link-time dependency on OpenCL. It mirrors the loader
//! code that [`codevar-gpu-core`] already uses for Vulkan, so the two GPU
//! stacks behave identically.
//!
//! # Library lifetime
//!
//! An opened library handle is intentionally never closed. Vendor drivers
//! spawn their own threads (compiler workers, queue managers) and unloading
//! the library while such a thread is running would crash the process; the
//! extracted entry-point pointers are likewise only valid while the library
//! stays mapped. Every handle therefore lives until process exit, matching
//! the documented behaviour of `codevar-gpu-core`'s Vulkan loader.
//!
//! [`codevar-gpu-core`]: https://docs.rs/codevar-gpu-core
//!
//! # Reference
//!
//! * <https://github.com/KhronosGroup/OpenCL-ICD-Loader> — how the canonical
//!   client loader opens vendor libraries.
//! * <https://github.com/OCL-dev/ocl-icd> — the Debian/Ubuntu loader whose
//!   `.icd` handling [`crate::icd`] mirrors.

use alloc::string::String;
use core::ffi::{CStr, c_void};

use crate::error::{Error, Result};

/// An open handle to a shared library that exports OpenCL entry points.
///
/// The handle keeps the library (and therefore every function pointer
/// resolved from it) mapped for the lifetime of the process; see the module
/// documentation for why it is never released. The raw handle is inert
/// between `dlopen`/`LoadLibrary` and process exit, which makes it safe to
/// share across threads.
pub struct Library {
    handle: *mut c_void,
}

// SAFETY: the field is only ever passed to the platform dynamic loader, which
// is thread-safe, and the library is never closed, so the handle cannot be
// invalidated while another thread uses it.
unsafe impl Send for Library {}
// SAFETY: see the `Send` implementation above.
unsafe impl Sync for Library {}

impl Library {
    /// Opens `name`, trying the platform dynamic loader.
    ///
    /// `name` may be a soname (`c"libOpenCL.so.1"`), an absolute path, or a
    /// DLL name (`c"OpenCL.dll"`), depending on what the platform loader
    /// accepts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LibraryOpen`] carrying the loader's diagnostic
    /// message when the library does not exist or cannot be mapped.
    pub(crate) fn open(name: &CStr) -> Result<Self> {
        let handle = open_raw(name)?;
        Ok(Self { handle })
    }

    /// Resolves the named symbol in this library.
    ///
    /// Returns `None` when the symbol is absent. A null return cannot
    /// distinguish "missing symbol" from a legitimately null symbol, but no
    /// OpenCL entry point in this crate's subset is ever null.
    pub(crate) fn symbol(&self, name: &CStr) -> Option<*mut c_void> {
        symbol_raw(self.handle, name)
    }

    /// Returns the raw loader handle (used by tests to assert non-nullness).
    #[cfg(test)]
    pub(crate) const fn as_raw(&self) -> *mut c_void {
        self.handle
    }
}

/// Candidate names of the OpenCL *client* loader, in preference order.
///
/// These are the libraries that export the full `cl*` API: the Khronos ICD
/// loader (`libOpenCL.so.1`, `OpenCL.dll`) or, on macOS, the (deprecated)
/// OpenCL framework, which exports the API directly.
pub(crate) fn client_candidates() -> &'static [&'static CStr] {
    #[cfg(target_os = "macos")]
    {
        &[
            c"/System/Library/Frameworks/OpenCL.framework/OpenCL",
            c"libOpenCL.dylib",
        ]
    }
    #[cfg(windows)]
    {
        &[c"OpenCL.dll"]
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        &[c"libOpenCL.so.1", c"libOpenCL.so"]
    }
    #[cfg(not(any(unix, windows)))]
    {
        &[]
    }
}

/// Opens the first candidate that can be mapped, reporting the last failure.
///
/// # Errors
///
/// Returns [`Error::LibraryOpen`] with the diagnostic of the final candidate
/// when no candidate could be opened.
pub(crate) fn open_first(candidates: &[&CStr]) -> Result<(Library, String)> {
    let mut last_error = Error::NoLoader;
    for name in candidates {
        match Library::open(name) {
            Ok(library) => {
                let origin = name.to_string_lossy().into_owned();
                return Ok((library, origin));
            }
            Err(error) => last_error = error,
        }
    }
    Err(last_error)
}

#[cfg(unix)]
fn open_raw(name: &CStr) -> Result<*mut c_void> {
    // SAFETY: `name` is a valid NUL-terminated constant string and the
    // RTLD_* flags are defined POSIX values. A null return only means the
    // library could not be opened; the error below is fetched instead.
    let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if !handle.is_null() {
        return Ok(handle);
    }
    // SAFETY: `dlerror` returns either null or a pointer to a static,
    // NUL-terminated string owned by the dynamic loader. It is only valid
    // until the next dlerror/dlopen call, so it is copied immediately.
    let message = unsafe { libc::dlerror() };
    let message = if message.is_null() {
        String::from("the dynamic loader reported no diagnostic")
    } else {
        // SAFETY: a non-null `dlerror` result is a NUL-terminated C string.
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(Error::LibraryOpen {
        library: name.to_string_lossy().into_owned(),
        message,
    })
}

#[cfg(unix)]
fn symbol_raw(handle: *mut c_void, name: &CStr) -> Option<*mut c_void> {
    // SAFETY: `handle` came from a successful `dlopen` in `open_raw` and the
    // library is never closed, so the handle stays valid for the process
    // lifetime. `name` is a valid NUL-terminated constant string.
    let symbol = unsafe { libc::dlsym(handle, name.as_ptr()) };
    (!symbol.is_null()).then_some(symbol)
}

#[cfg(windows)]
fn open_raw(name: &CStr) -> Result<*mut c_void> {
    use alloc::format;

    windows_link::link!("kernel32.dll" "system"
        fn LoadLibraryA(lp_lib_file_name: *const u8) -> *mut c_void
    );
    windows_link::link!("kernel32.dll" "system"
        fn GetLastError() -> u32
    );

    // SAFETY: `name` is a valid NUL-terminated constant string; the loader
    // copies the path during the call. A null return means failure and the
    // error code is queried immediately afterward.
    let handle = unsafe { LoadLibraryA(name.as_ptr().cast()) };
    if handle.is_null() {
        let code = unsafe { GetLastError() };
        return Err(Error::LibraryOpen {
            library: name.to_string_lossy().into_owned(),
            message: format!("LoadLibraryA failed with error code {code}"),
        });
    }
    Ok(handle)
}

#[cfg(windows)]
fn symbol_raw(handle: *mut c_void, name: &CStr) -> Option<*mut c_void> {
    windows_link::link!("kernel32.dll" "system"
        fn GetProcAddress(h_module: *mut c_void, lp_proc_name: *const u8) -> *mut c_void
    );

    // SAFETY: `handle` came from a successful `LoadLibraryA` and the library
    // is never freed, so the handle stays valid for the process lifetime.
    // `name` is a valid NUL-terminated constant string.
    let symbol = unsafe { GetProcAddress(handle, name.as_ptr().cast()) };
    (!symbol.is_null()).then_some(symbol)
}

#[cfg(not(any(unix, windows)))]
fn open_raw(name: &CStr) -> Result<*mut c_void> {
    Err(Error::LibraryOpen {
        library: name.to_string_lossy().into_owned(),
        message: String::from("no dynamic-loader backend exists for this target platform"),
    })
}

#[cfg(not(any(unix, windows)))]
fn symbol_raw(_handle: *mut c_void, _name: &CStr) -> Option<*mut c_void> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_candidates_are_platform_specific_and_non_empty() {
        let candidates = client_candidates();
        assert!(!candidates.is_empty());
        for name in candidates {
            assert!(!name.to_bytes().is_empty());
        }
    }

    #[test]
    fn open_reports_a_diagnostic_for_a_missing_library() {
        let error = Library::open(c"libcodevar_definitely_missing_ocl.so").err();
        assert!(
            matches!(
                error,
                Some(Error::LibraryOpen { ref library, ref message })
                    if library == "libcodevar_definitely_missing_ocl.so" && !message.is_empty()
            ),
            "expected Error::LibraryOpen with a message, got {error:?}"
        );
    }

    #[test]
    fn opening_the_client_loader_resolves_the_platform_entry_point() {
        let candidates = client_candidates();
        let mut failures = 0usize;
        for name in candidates {
            match Library::open(name) {
                Ok(library) => {
                    assert!(!library.as_raw().is_null());
                    let symbol = library.symbol(c"clGetPlatformIDs");
                    assert!(
                        symbol.is_some(),
                        "{name:?} opened but does not export clGetPlatformIDs"
                    );
                    return;
                }
                Err(error) => {
                    assert!(
                        matches!(error, Error::LibraryOpen { .. }),
                        "unexpected error kind opening {name:?}: {error:?}"
                    );
                    failures += 1;
                }
            }
        }
        // Without any OpenCL installation every candidate must fail with a
        // LibraryOpen diagnostic rather than a different error kind.
        assert_eq!(failures, candidates.len());
    }

    #[test]
    fn open_first_retries_every_candidate() {
        let candidates: [&CStr; 2] = [c"libcodevar_missing_one.so", c"libcodevar_missing_two.so"];
        let error = open_first(&candidates).err();
        assert!(
            matches!(
                error,
                Some(Error::LibraryOpen { ref library, .. })
                    if library == "libcodevar_missing_two.so"
            ),
            "open_first must report the final candidate, got {error:?}"
        );
    }
}
