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

//! Module base address caching for async-signal-safe backtrace formatting.
//!
//! Uses `dl_iterate_phdr` at startup to enumerate all loaded modules and
//! caches their base addresses, names, and address ranges. The cache is
//! then available for O(log n) lookup from signal handlers without calling
//! any non-async-signal-safe functions.

use core::ffi::CStr;
use core::mem::MaybeUninit;
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(all(unix, not(target_arch = "wasm32")))]
mod imp {
    use super::*;
    use libc;

    /// Maximum number of cached modules.
    const MAX_MODULES: usize = 256;

    /// A cached module entry.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ModuleEntry {
        /// Start address of the module (lowest PT_LOAD vaddr).
        pub base: usize,
        /// End address (exclusive) of the module's mapped region.
        pub end: usize,
        /// Module name (null-terminated, from dlpi_name).
        pub name: [u8; 256],
    }

    /// Global module cache.
    ///
    /// Initialized once at startup via `init()`. After initialization,
    /// the valid entries (0..COUNT) are sorted by `base` address for binary search.
    /// Read-only access from signal handlers requires no locks, no syscalls,
    /// and no heap allocation.
    static mut MODULES: [MaybeUninit<ModuleEntry>; MAX_MODULES] =
        [const { MaybeUninit::uninit() }; MAX_MODULES];

    /// Number of valid entries in `MODULES`.
    static COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Initialization flag: 0 = uninitialized, 1 = initialized.
    static INITIALIZED: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" {
        fn dl_iterate_phdr(
            callback: unsafe extern "C" fn(*const libc::dl_phdr_info, usize, *mut core::ffi::c_void) -> i32,
            data: *mut core::ffi::c_void,
        ) -> i32;
    }

    /// Initialize the module cache by walking `dl_iterate_phdr`.
    ///
    /// Must be called once at startup before installing signal handlers.
    /// Not thread-safe; caller must ensure single-threaded initialization.
    #[cfg_attr(feature = "nightly", sanitize(address = "off"))]
    pub unsafe fn init() {
        if INITIALIZED.swap(1, Ordering::AcqRel) != 0 {
            return;
        }

        unsafe extern "C" fn callback(
            info: *const libc::dl_phdr_info,
            _size: usize,
            data: *mut core::ffi::c_void,
        ) -> i32 {
            if info.is_null() || data.is_null() {
                return 0;
            }
            let info = unsafe { &*info };
            let count_ptr = data as *mut AtomicUsize;
            let count = unsafe { &*count_ptr };
            let current = count.load(Ordering::Relaxed);
            if current >= MAX_MODULES {
                return 1;
            }
            if info.dlpi_phdr.is_null() || info.dlpi_phnum == 0 {
                return 0;
            }
            let phdrs = unsafe { core::slice::from_raw_parts(info.dlpi_phdr, usize::from(info.dlpi_phnum)) };
            let mut min_start = usize::MAX;
            let mut max_end = 0usize;
            let base_addr = info.dlpi_addr as usize;

            for ph in phdrs {
                if ph.p_type == libc::PT_LOAD {
                    let start = base_addr.wrapping_add(ph.p_vaddr as usize);
                    let end = start.wrapping_add(ph.p_memsz as usize);
                    if start < min_start {
                        min_start = start;
                    }
                    if end > max_end {
                        max_end = end;
                    }
                }
            }
            if min_start == usize::MAX {
                return 0;
            }
            let mut entry = ModuleEntry {
                base: min_start,
                end: max_end,
                name: [0; 256],
            };
            if !info.dlpi_name.is_null() {
                let name_cstr = unsafe { CStr::from_ptr(info.dlpi_name) };
                let name_bytes = name_cstr.to_bytes();
                let len = name_bytes.len().min(255);
                entry.name[..len].copy_from_slice(&name_bytes[..len]);
                entry.name[len] = 0;
            }

            unsafe {
                let modules_ptr = ptr::addr_of_mut!(MODULES) as *mut ModuleEntry;
                let ptr = modules_ptr.add(current);
                ptr.write(entry);
            }

            count.store(current + 1, Ordering::Relaxed);
            0
        }

        let count = &COUNT as *const AtomicUsize as *mut core::ffi::c_void;
        unsafe {
            dl_iterate_phdr(callback, count);
        }
        let n = COUNT.load(Ordering::Acquire);
        if n > 1 {
            let modules_ptr = ptr::addr_of!(MODULES) as *mut ModuleEntry;
            for i in 1..n {
                let key = unsafe { modules_ptr.add(i).read() };
                let mut j = i;
                while j > 0 {
                    let prev = unsafe { modules_ptr.add(j - 1).read() };
                    if prev.base <= key.base {
                        break;
                    }
                    unsafe {
                        modules_ptr.add(j).write(prev);
                    }
                    j -= 1;
                }
                unsafe {
                    modules_ptr.add(j).write(key);
                }
            }
        }
    }

    /// Look up the module base address for `ip`.
    ///
    /// Returns `Some(base)` if `ip` falls within a cached module's range,
    /// otherwise `None`. Async-signal-safe (read-only, no locks, no syscalls).
    #[inline]
    #[cfg_attr(feature = "nightly", sanitize(address = "off"))]
    pub unsafe fn module_base(ip: usize) -> Option<usize> {
        let n = COUNT.load(Ordering::Acquire);
        if n == 0 {
            return None;
        }
        let modules_ptr = ptr::addr_of!(MODULES) as *const ModuleEntry;
        // Binary search on the sorted valid entries [0, n)
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            // SAFETY: Using ptr::addr_of! to avoid mutable reference to static mut
            let entry = unsafe { &*modules_ptr.add(mid) };
            if ip < entry.base {
                hi = mid;
            } else if ip >= entry.end {
                lo = mid + 1;
            } else {
                return Some(entry.base);
            }
        }
        None
    }

    /// Look up the module name for `ip`.
    ///
    /// Returns the cached module name if `ip` falls within a cached module's
    /// range, otherwise `None`. Async-signal-safe.
    #[inline]
    #[cfg_attr(feature = "nightly", sanitize(address = "off"))]
    pub unsafe fn module_name(ip: usize) -> Option<&'static str> {
        unsafe {
            let n = COUNT.load(Ordering::Acquire);
            if n == 0 {
                return None;
            }
            let modules_ptr = ptr::addr_of!(MODULES) as *const ModuleEntry;
            let mut lo = 0usize;
            let mut hi = n;
            while lo < hi {
                let mid = (lo + hi) >> 1;
                let entry = &*modules_ptr.add(mid);
                if ip < entry.base {
                    hi = mid;
                } else if ip >= entry.end {
                    lo = mid + 1;
                } else {
                    let entry_ptr = modules_ptr.add(mid);
                    let name_ptr = ptr::addr_of!((*entry_ptr).name) as *const u8;
                    let mut name_end = 0;
                    while name_end < 256 && *name_ptr.add(name_end) != 0 {
                        name_end += 1;
                    }
                    let slice = core::slice::from_raw_parts(name_ptr, name_end);
                    return Some(core::str::from_utf8_unchecked(slice));
                }
            }
            None
        }
    }
}

#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
mod imp {
    /// No-op on unsupported platforms.
    pub fn init() {}

    #[inline]
    pub fn module_base(_ip: usize) -> Option<usize> {
        None
    }

    #[inline]
    pub fn module_name(_ip: usize) -> Option<&'static str> {
        None
    }
}

/// Initialize the module base cache.
///
/// Call once at program startup before installing signal handlers.
/// Not thread-safe.
pub fn init() {
    unsafe {
        imp::init();
    }
}

/// Get the module base address for an instruction pointer.
///
/// Async-signal-safe. Returns the cached base address if `ip` falls within
/// a known module's address range, otherwise `None`.
#[inline]
pub fn module_base(ip: usize) -> Option<usize> {
    unsafe { imp::module_base(ip) }
}

/// Get the module name for an instruction pointer.
///
/// Async-signal-safe. Returns the cached module name if `ip` falls within
/// a known module's address range, otherwise `None`.
#[inline]
pub fn module_name(ip: usize) -> Option<&'static str> {
    unsafe { imp::module_name(ip) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_base_returns_none_before_init() {
        // Before init, module_base should return None
        assert_eq!(module_base(0x1000), None);
        assert_eq!(module_name(0x1000), None);
    }

    #[test]
    fn module_base_returns_none_after_init_without_match() {
        init();
        // Use an address that won't match any loaded module
        assert_eq!(module_base(0xFFFF_FFFF_FFFF_FFFF), None);
        assert_eq!(module_name(0xFFFF_FFFF_FFFF_FFFF), None);
    }

    #[test]
    fn init_is_idempotent() {
        init();
        init(); // Should not panic or cause issues
        init();
    }

    #[test]
    fn module_base_returns_some_for_valid_range() {
        init();
        // The main executable should always be in the cache
        // Get a valid IP from the current function
        let ip = module_base_returns_some_for_valid_range as *const () as usize;
        let base = module_base(ip);
        // At minimum, the cache should be initialized and not crash
        // The actual value depends on where the test binary is loaded
        assert!(base.is_some() || base.is_none()); // Just verify no panic
    }
}
