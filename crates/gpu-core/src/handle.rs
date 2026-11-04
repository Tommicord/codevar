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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Owned operating-system handles shared by the graphics backends.
//!
//! [`OwnedFd`] is the `no_std` counterpart of `std::os::fd::OwnedFd`: the
//! raw descriptor is closed exactly once when the value drops. Vulkan
//! uses it for dma-buf and `sync_file` descriptors; a backend that has
//! no such handles never constructs one.

/// An owned operating-system file descriptor (a dma-buf or `sync_file`).
///
/// The raw descriptor is closed exactly once when the value drops, moved
/// with [`OwnedFd::into_raw`], adopted with [`OwnedFd::from_raw`] and
/// inspected with [`OwnedFd::as_raw`].
///
/// On non-unix, non-Windows targets dropping is a no-op; the dma-buf and
/// `sync_file` extensions this type wraps are only reachable on unix (and
/// Win32 handles are not small integers, so they never enter this type).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct OwnedFd {
    raw: core::ffi::c_int,
}

impl OwnedFd {
    /// Adopts a raw descriptor that is already open.
    ///
    /// # Safety
    ///
    /// `raw` must be an open descriptor that is owned by the caller and
    /// has not been borrowed by any other owner. After this call the
    /// descriptor is closed by [`OwnedFd`]'s `Drop`.
    #[inline]
    #[must_use]
    pub const unsafe fn from_raw(raw: core::ffi::c_int) -> Self {
        Self { raw }
    }

    /// Returns the raw descriptor without giving up ownership.
    #[inline]
    #[must_use]
    pub const fn as_raw(&self) -> core::ffi::c_int {
        self.raw
    }

    /// Returns the raw descriptor and gives up ownership; the caller must
    /// close it.
    #[inline]
    #[must_use]
    pub const fn into_raw(self) -> core::ffi::c_int {
        let raw = self.raw;
        core::mem::forget(self);
        raw
    }
}

impl Drop for OwnedFd {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: `raw` was open when adopted by `from_raw` and this `Drop`
        // runs exactly once per descriptor (moves go through `into_raw`,
        // which forgets `self`); closing a descriptor twice is therefore
        // impossible.
        unsafe {
            libc::close(self.raw);
        }
        #[cfg(windows)]
        {
            windows_link::link!(
                "kernel32.dll" "system"
                fn CloseHandle(hObject: *mut core::ffi::c_void) -> i32
            );
            // SAFETY: as above for the descriptor itself; `CloseHandle` only
            // fails on an already-closed handle, which cannot happen here.
            unsafe {
                let _ = CloseHandle(self.raw as isize as *mut core::ffi::c_void);
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = self.raw;
        }
    }
}

#[cfg(test)]
mod tests {
    //! Ownership transfer of the descriptor wrapper.

    use super::*;

    #[test]
    fn into_raw_forgets_the_owner() {
        #[cfg(unix)]
        {
            // A descriptor that is never closed by the wrapper: the test
            // closes it itself after taking ownership back.
            // SAFETY: `dup` returns a fresh descriptor owned by this test.
            let raw = unsafe { libc::dup(0) };
            assert!(raw >= 0);
            // SAFETY: `raw` is a fresh descriptor.
            let owned = unsafe { OwnedFd::from_raw(raw) };
            assert_eq!(owned.as_raw(), raw);
            let recovered = owned.into_raw();
            assert_eq!(recovered, raw);
            // SAFETY: the descriptor is still open and owned by the test.
            unsafe { libc::close(recovered) };
        }
    }
}
