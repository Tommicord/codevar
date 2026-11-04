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

//! Vendor-driver discovery and platform enumeration.
//!
//! Modern OpenCL installations split into two halves: the *client loader*
//! (`libOpenCL.so.1`, `OpenCL.dll`) which exports the full `cl*` API, and
//! the *vendor ICDs* (`libigdrcl.so`, `libnvidia-opencl.so.1`, …) which
//! export only [`sys::IcdGetPlatformIdsKhr`] plus a `cl_icd_dispatch`
//! table (see [`crate::dispatch`]). This module abstracts both:
//!
//! * [`candidates`] lists every vendor library to try, in preference
//!   order: `OCL_ICD_FILENAMES`, then `*.icd` registration files
//!   (`OCL_ICD_VENDORS` or `/etc/OpenCL/vendors`), then a list of builtin
//!   sonames, deduplicated.
//! * [`query_platforms`] enumerates platforms through a library that
//!   exports the full API (client loader or legacy vendor).
//! * [`vendor_platforms`] enumerates platforms through a modern vendor
//!   library's ICD entry point and builds one [`Api`] per platform from
//!   its dispatch table.
//!
//! Every opened library is intentionally never closed; see
//! [`crate::loader`] for the lifetime rules.
//!
//! [`sys::IcdGetPlatformIdsKhr`]: crate::sys::IcdGetPlatformIdsKhr

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use codevar_pathbuf::PathBuf;
use core::ffi::c_void;
use core::mem::{size_of, transmute};
use core::ptr::null_mut;

use crate::api::Api;
use crate::dispatch::dispatch_of;
use crate::error::{Error, Result, status};
use crate::loader::Library;
use crate::platform::Platform;
use crate::query::MAX_HANDLES;
use crate::sys::{self, GetPlatformIds, IcdGetPlatformIdsKhr, PlatformHandle};

/// A vendor-driver library discovered on this machine.
pub(crate) struct Candidate {
    /// Path or soname handed to the dynamic loader.
    pub(crate) path: PathBuf,
    /// Where discovery found it (used in diagnostics only).
    pub(crate) origin: &'static str,
}

/// Returns every candidate vendor-driver library, deduplicated and in the
/// preference order documented in the module documentation.
///
/// On non-unix targets directory scanning is unavailable, so only
/// `OCL_ICD_FILENAMES` and the builtin list contribute entries.
pub(crate) fn candidates() -> Vec<Candidate> {
    let mut out: Vec<Candidate> = Vec::new();
    if let Some(value) = codevar_env::env_var("OCL_ICD_FILENAMES") {
        let separator = if cfg!(windows) { ';' } else { ':' };
        for path in value.split(separator) {
            push_candidate(&mut out, path, "OCL_ICD_FILENAMES");
        }
    }
    for dir in vendor_directories() {
        for path in icd_registration_files(&dir) {
            push_path(&mut out, path, "vendor .icd file");
        }
    }
    for path in builtin_candidates() {
        push_candidate(&mut out, path, "builtin candidate");
    }
    out
}

/// Appends a trimmed, non-empty path unless it is already present.
///
/// Paths that fail validation (empty, control characters) are skipped:
/// they could never be opened by the dynamic loader anyway.
fn push_candidate(out: &mut Vec<Candidate>, path: &str, origin: &'static str) {
    let Ok(path) = PathBuf::from_str(path.trim()) else {
        return;
    };
    push_path(out, path, origin);
}

/// Appends an already-built [`PathBuf`] unless it is already present.
fn push_path(out: &mut Vec<Candidate>, path: PathBuf, origin: &'static str) {
    if out.iter().any(|existing| existing.path == path) {
        return;
    }
    out.push(Candidate { path, origin });
}

/// Directories scanned for `*.icd` registration files.
///
/// `OCL_ICD_VENDORS` replaces the default when set (matching the ocl-icd
/// and Khronos loaders).
fn vendor_directories() -> Vec<PathBuf> {
    if let Some(dir) = codevar_env::env_var("OCL_ICD_VENDORS")
        && let Ok(path) = PathBuf::from_str(dir.trim())
    {
        return alloc::vec![path];
    }
    let mut dirs = Vec::new();
    #[cfg(unix)]
    {
        // A compile-time constant, so it always validates.
        dirs.push(PathBuf::from_string(String::from("/etc/OpenCL/vendors")));
    }
    dirs
}

/// Extracts the library path from the contents of a `*.icd` file: the
/// first non-empty, non-comment line, trimmed.
#[cfg(unix)]
fn icd_library_path(contents: &str) -> Option<&str> {
    contents
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
}

/// Returns the library paths registered by every `*.icd` file in `dir`.
///
/// Entries that cannot be read or parsed are skipped; a broken registration
/// file must not hide the working drivers next to it.
#[cfg(unix)]
fn icd_registration_files(dir: &PathBuf) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for name in list_dir(dir.as_str()) {
        if !name.ends_with(".icd") {
            continue;
        }
        let mut path = dir.clone();
        if path.push(&name).is_err() {
            continue;
        }
        let Ok(contents) = codevar_pathbuf::read_to_string(path.as_str()) else {
            continue;
        };
        if let Some(library) = icd_library_path(&contents)
            && let Ok(path) = PathBuf::from_str(library)
        {
            out.push(path);
        }
    }
    out
}

/// Directory scanning is unavailable off-unix; only environment variables
/// and builtin candidates contribute there.
#[cfg(not(unix))]
fn icd_registration_files(_dir: &PathBuf) -> Vec<PathBuf> {
    Vec::new()
}

/// Builtin library candidates, in preference order, tried when neither
/// environment variables nor registration files yielded a usable driver.
#[cfg(all(unix, not(target_os = "macos")))]
fn builtin_candidates() -> &'static [&'static str] {
    &[
        // Intel GPUs (Debian/Ubuntu package layout, then ldconfig soname).
        "/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so",
        "/usr/lib64/intel-opencl/libigdrcl.so",
        "libigdrcl.so",
        // NVIDIA.
        "libnvidia-opencl.so.1",
        // AMD (ROCm and legacy Catalyst layouts).
        "libamdocl64.so",
        "libamdocl.so",
        // Portable Computing Language.
        "libpocl.so.2",
        "libpocl.so",
        // Last resort: the client loader itself, which exports the full API.
        "libOpenCL.so.1",
        "libOpenCL.so",
    ]
}

/// Builtin candidates on macOS: only the (deprecated) system framework.
#[cfg(target_os = "macos")]
fn builtin_candidates() -> &'static [&'static str] {
    &[
        "/System/Library/Frameworks/OpenCL.framework/OpenCL",
        "libOpenCL.dylib",
    ]
}

/// Builtin candidates on Windows: the ICD loader DLL exports the full API.
#[cfg(windows)]
fn builtin_candidates() -> &'static [&'static str] {
    &["OpenCL.dll"]
}

/// No builtin candidates exist on targets without OpenCL.
#[cfg(not(any(unix, windows)))]
fn builtin_candidates() -> &'static [&'static str] {
    &[]
}

/// Shared two-call `clGetPlatformIDs` protocol, used both by the client
/// loader path and by `clIcdGetPlatformIDsKHR` (identical signatures).
///
/// An empty result — either zero platforms or the loader's
/// [`sys::PLATFORM_NOT_FOUND_KHR`] — is `Ok(Vec::new())`, because "this
/// driver has no platforms" is a legitimate machine state for the caller
/// to skip over while scanning candidates.
///
/// # Errors
///
/// Returns [`Error::Status`] when the driver rejects a call and
/// [`Error::InvalidArgument`] when it reports an implausible count.
fn enumerate_platforms(
    mut get: impl FnMut(u32, *mut PlatformHandle, *mut u32) -> i32,
    context: &'static str,
) -> Result<Vec<PlatformHandle>> {
    let mut count: u32 = 0;
    // The null/zero probe is the documented way to query the platform
    // count; `count` is a valid out-pointer.
    let code = get(0, null_mut(), &mut count);
    if code == sys::PLATFORM_NOT_FOUND_KHR || (code == sys::SUCCESS && count == 0) {
        return Ok(Vec::new());
    }
    status(code, context)?;
    if count > MAX_HANDLES {
        return Err(Error::InvalidArgument {
            what: "clGetPlatformIDs reported an implausible platform count",
        });
    }
    let mut handles: Vec<PlatformHandle> = (0..count)
        .map(|_| PlatformHandle::from_raw(null_mut()))
        .collect();
    // The buffer has room for `count` handles, matching the count the
    // driver just confirmed.
    let code = get(count, handles.as_mut_ptr(), null_mut());
    status(code, context)?;
    handles.retain(|handle| !handle.is_null());
    Ok(handles)
}

/// Enumerates the platforms of a library that exports the full `cl*` API
/// (the Khronos client loader or a legacy vendor library).
///
/// Returns an empty vector when the library reports no platforms.
///
/// # Errors
///
/// See [`enumerate_platforms`].
pub(crate) fn query_platforms(api: &Arc<Api>) -> Result<Vec<Platform>> {
    let handles = enumerate_platforms(
        |entries, platforms, total| unsafe { (api.get_platform_ids)(entries, platforms, total) },
        "clGetPlatformIDs",
    )?;
    Ok(handles
        .into_iter()
        .map(|raw| Platform::new(api.clone(), raw))
        .collect())
}

/// Enumerates the platforms of a modern vendor library through
/// `clIcdGetPlatformIDsKHR` and builds one [`Api`] per platform from its
/// `cl_icd_dispatch` table.
///
/// # Errors
///
/// Returns [`Error::MissingSymbol`] when the library exports no ICD entry
/// point (it is not a vendor ICD), [`Error::Status`] /
/// [`Error::InvalidArgument`] on enumeration failures, and
/// [`Error::MissingSymbol`] when a required dispatch-table slot is null.
pub(crate) fn vendor_platforms(library: &Library) -> Result<Vec<Platform>> {
    let Some(address) = library.symbol(c"clIcdGetPlatformIDsKHR") else {
        return Err(Error::MissingSymbol {
            symbol: "clIcdGetPlatformIDsKHR",
        });
    };
    debug_assert_eq!(size_of::<IcdGetPlatformIdsKhr>(), size_of::<GetPlatformIds>());
    // SAFETY: `address` is the address of the exported symbol
    // `clIcdGetPlatformIDsKHR`, whose C signature is exactly
    // `GetPlatformIds` (checked above); function pointers are
    // pointer-sized, so the bit pattern is preserved unchanged.
    let get: GetPlatformIds = unsafe { transmute::<*mut c_void, GetPlatformIds>(address) };
    let handles = enumerate_platforms(|e, p, n| unsafe { get(e, p, n) }, "clIcdGetPlatformIDsKHR")?;
    let mut platforms = Vec::with_capacity(handles.len());
    for raw in handles {
        // SAFETY: `raw` is a live platform handle produced by the driver
        // behind `library`, so its first word is that driver's
        // `cl_icd_dispatch` pointer per the ICD ABI.
        let table = unsafe { dispatch_of(raw) };
        if table.is_null() {
            return Err(Error::MissingSymbol {
                symbol: "clIcdGetPlatformIDsKHR",
            });
        }
        // SAFETY: the caller guarantees `library` stays open (this crate
        // never closes libraries), so the dispatch table the platform
        // object points at remains mapped for the process lifetime; the
        // null check above guarantees a table is present.
        let api = unsafe { Api::from_dispatch(table)? };
        platforms.push(Platform::new(Arc::new(api), raw));
    }
    Ok(platforms)
}

/// Lists the entries of `dir` without `.`/`..`, or an empty vector when
/// the directory cannot be opened.
#[cfg(unix)]
fn list_dir(dir: &str) -> Vec<String> {
    use alloc::ffi::CString;
    use core::ffi::CStr;

    let Ok(c_dir) = CString::new(dir) else {
        return Vec::new();
    };
    // SAFETY: `c_dir` is a valid NUL-terminated directory path; a null
    // result only means the directory does not exist.
    let handle = unsafe { libc::opendir(c_dir.as_ptr()) };
    if handle.is_null() {
        return Vec::new();
    }
    let mut names = Vec::new();
    loop {
        // SAFETY: `handle` is an open stream; the entry (when non-null)
        // is valid until the next `readdir` call, and the borrow does
        // not outlive this loop iteration.
        let Some(entry) = (unsafe { libc::readdir(handle).as_ref() }) else {
            break;
        };
        // SAFETY: `d_name` is a NUL-terminated byte array inside the
        // live entry.
        let name = unsafe { CStr::from_ptr(entry.d_name.as_ptr()) };
        let Ok(name) = name.to_str() else {
            continue;
        };
        if name == "." || name == ".." {
            continue;
        }
        names.push(String::from(name));
    }
    // SAFETY: `handle` was opened above and is not used afterward.
    unsafe { libc::closedir(handle) };
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn icd_library_path_takes_the_first_content_line() {
        assert_eq!(
            icd_library_path("/opt/vendor/libdriver.so\n"),
            Some("/opt/vendor/libdriver.so")
        );
        assert_eq!(icd_library_path("  libdriver.so  \n"), Some("libdriver.so"));
        assert_eq!(
            icd_library_path("# comment\nlibdriver.so\n"),
            Some("libdriver.so")
        );
        assert_eq!(icd_library_path("\n\n  \n"), None);
        assert_eq!(icd_library_path(""), None);
    }

    #[test]
    fn push_candidate_trims_and_deduplicates() {
        let mut out = Vec::new();
        push_candidate(&mut out, "  liba.so ", "one");
        push_candidate(&mut out, "liba.so", "two");
        push_candidate(&mut out, "   ", "one");
        push_candidate(&mut out, "libb.so", "one");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].path.as_str(), "liba.so");
        assert_eq!(out[0].origin, "one");
        assert_eq!(out[1].path.as_str(), "libb.so");
    }

    /// The candidate list must never contain duplicates, whichever
    /// discovery sources contributed them.
    #[test]
    fn candidates_are_unique() {
        let candidates = candidates();
        for (index, candidate) in candidates.iter().enumerate() {
            let duplicates = candidates[index + 1..]
                .iter()
                .filter(|other| other.path == candidate.path)
                .count();
            assert_eq!(duplicates, 0, "duplicate candidate {}", candidate.path);
        }
    }

    #[test]
    fn enumerate_platforms_handles_empty_results() {
        let empty = |_entries: u32, _out: *mut PlatformHandle, _total: *mut u32| sys::SUCCESS;
        assert!(enumerate_platforms(empty, "clGetPlatformIDs").is_ok_and(|h| h.is_empty()));

        let missing =
            |_entries: u32, _out: *mut PlatformHandle, _total: *mut u32| sys::PLATFORM_NOT_FOUND_KHR;
        assert!(enumerate_platforms(missing, "clGetPlatformIDs").is_ok_and(|h| h.is_empty()));
    }

    #[test]
    fn enumerate_platforms_rejects_implausible_counts() {
        let huge = |_entries: u32, _out: *mut PlatformHandle, total: *mut u32| {
            if !total.is_null() {
                // SAFETY: the caller passes a valid out-pointer.
                unsafe { total.write(u32::MAX) };
            }
            sys::SUCCESS
        };
        assert!(matches!(
            enumerate_platforms(huge, "clGetPlatformIDs"),
            Err(Error::InvalidArgument { .. })
        ));
    }

    #[test]
    fn enumerate_platforms_fills_handles_with_the_two_call_protocol() {
        use core::cell::Cell;

        let raw = 0x1234usize as *mut c_void;
        let calls = Cell::new(0u32);
        let fill = |entries: u32, out: *mut PlatformHandle, total: *mut u32| {
            calls.set(calls.get() + 1);
            if out.is_null() {
                if !total.is_null() {
                    // SAFETY: the caller passes a valid out-pointer.
                    unsafe { total.write(1) };
                }
                return sys::SUCCESS;
            }
            if entries < 1 {
                return sys::INVALID_VALUE;
            }
            // SAFETY: `out` points at the caller's one-element buffer.
            unsafe { out.write(PlatformHandle::from_raw(raw)) };
            sys::SUCCESS
        };
        let handles = enumerate_platforms(fill, "clGetPlatformIDs")
            .ok()
            .unwrap_or_default();
        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0].as_raw(), raw);
        assert_eq!(calls.get(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn registration_files_of_a_missing_directory_are_empty() {
        let dir = PathBuf::from_string(String::from("/codevar/definitely/missing"));
        assert!(icd_registration_files(&dir).is_empty());
    }
}
