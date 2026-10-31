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

//! Resolved OpenCL entry points shared by every object in the crate.
//!
//! An [`Api`] holds one function pointer per OpenCL entry point this crate
//! uses. There are exactly two ways to build one, and both are generated
//! from a single macro list so the two paths can never drift apart:
//!
//! * [`Api::from_symbols`] resolves the `cl*` symbols directly out of a
//!   library that exports them — the Khronos ICD loader (`libOpenCL.so.1`,
//!   `OpenCL.dll`), the macOS OpenCL framework, or a legacy vendor library
//!   that exports the full API.
//! * [`Api::from_dispatch`] copies the pointers out of a vendor ICD's
//!   `cl_icd_dispatch` table (see [`crate::dispatch`]), which is the only
//!   way to use modern vendors such as Intel NEO, AMD's ROCm loader or
//!   NVIDIA's `libnvidia-opencl.so.1` directly.
//!
//! Required entry points (the OpenCL 1.0/1.1 core this crate depends on)
//! must resolve or construction fails with [`Error::MissingSymbol`].
//! Optional entry points become `None` and surface as [`Error::Unsupported`]
//! only when a caller actually uses them.
//!
//! The struct contains nothing but function pointers, so it is
//! automatically `Send + Sync` and is shared by all objects through `Arc`.

use alloc::ffi::CString;
use core::ffi::{CStr, c_void};
use core::mem::{size_of, transmute};

use crate::dispatch::IcdDispatch;
use crate::error::{Error, Result};
use crate::loader::Library;
use crate::sys;

/// Resolves `name` in `library` to a raw symbol address.
fn lookup(library: &Library, name: &'static str) -> Result<*mut c_void> {
    // The names are compile-time constants without interior NULs, so this
    // can only fail for a bug in this crate.
    let Ok(c_name) = CString::new(name) else {
        return Err(Error::MissingSymbol { symbol: name });
    };
    library
        .symbol(&c_name)
        .ok_or(Error::MissingSymbol { symbol: name })
}

macro_rules! define_api {
    (
        required: [$($req:ident : $req_ty:ident => $req_sym:literal),* $(,)?],
        optional: [$($opt:ident : $opt_ty:ident => $opt_sym:literal),* $(,)?]
    ) => {
        /// The set of OpenCL entry points resolved from one driver library.
        ///
        /// See the [module documentation](self) for the two construction
        /// paths and the required/optional split.
        // The dispatch table is a fixed, driver-wide set of entry points:
        // some slots (program/kernel info queries, buffer fill, extension
        // addresses) belong to planned features and are not read yet.
        #[allow(dead_code)]
        pub(crate) struct Api {
            $(
                #[doc = concat!("`", $req_sym, "`, resolved or construction failed.")]
                pub(crate) $req: sys::$req_ty,
            )*
            $(
                #[doc = concat!("`", $opt_sym, "` when the driver exports it, `None` otherwise.")]
                pub(crate) $opt: Option<sys::$opt_ty>,
            )*
        }

        impl Api {
            /// Resolves every entry point as a symbol of `library`.
            ///
            /// # Errors
            ///
            /// Returns [`Error::MissingSymbol`] when a required `cl*`
            /// symbol is absent, which means the library is not an OpenCL
            /// implementation (or is a stub loader without the core API).
            pub(crate) fn from_symbols(library: &Library) -> Result<Self> {
                Ok(Self {
                    $(
                        $req: {
                            let raw = lookup(library, $req_sym)?;
                            debug_assert_eq!(size_of::<sys::$req_ty>(), size_of::<*mut c_void>());
                            // SAFETY: `raw` is the address of the exported
                            // symbol named `$req_sym`, whose C signature is
                            // exactly `sys::$req_ty`; function pointers are
                            // pointer-sized (checked above), so the bit pattern
                            // is preserved unchanged.
                            unsafe { transmute::<*mut c_void, sys::$req_ty>(raw) }
                        },
                    )*
                    $(
                        $opt: {
                            let raw = lookup(library, $opt_sym);
                            match raw {
                                Ok(raw) => {
                                    debug_assert_eq!(
                                        size_of::<sys::$opt_ty>(),
                                        size_of::<*mut c_void>()
                                    );
                                    // SAFETY: same argument as the required
                                    // case, for the optional symbol named
                                    // `$opt_sym`.
                                    Some(unsafe { transmute::<*mut c_void, sys::$opt_ty>(raw) })
                                }
                                Err(_) => None,
                            }
                        },
                    )*
                })
            }

            /// Copies every entry point out of a vendor ICD dispatch table.
            ///
            /// # Errors
            ///
            /// Returns [`Error::MissingSymbol`] when a required slot of the
            /// table is null, i.e. the vendor implements an older OpenCL
            /// revision than this crate requires.
            ///
            /// # Safety
            ///
            /// `dispatch` must point to a complete `cl_icd_dispatch` table
            /// of a library that is still loaded, as produced by
            /// [`crate::dispatch::dispatch_of`] for a platform handle of
            /// that library.
            pub(crate) unsafe fn from_dispatch(dispatch: *const IcdDispatch) -> Result<Self> {
                // SAFETY: guaranteed by the caller: the pointer addresses a
                // full dispatch table of a live driver library.
                let table = unsafe { &*dispatch };
                Ok(Self {
                    $($req: table.$req.ok_or(Error::MissingSymbol { symbol: $req_sym })?,)*
                    $($opt: table.$opt,)*
                })
            }
        }
    };
}

define_api! {
    required: [
        get_platform_ids: GetPlatformIds => "clGetPlatformIDs",
        get_platform_info: GetPlatformInfo => "clGetPlatformInfo",
        get_device_ids: GetDeviceIds => "clGetDeviceIDs",
        get_device_info: GetDeviceInfo => "clGetDeviceInfo",
        create_context: CreateContext => "clCreateContext",
        release_context: ReleaseContext => "clReleaseContext",
        create_command_queue: CreateCommandQueue => "clCreateCommandQueue",
        release_command_queue: ReleaseCommandQueue => "clReleaseCommandQueue",
        flush: Flush => "clFlush",
        finish: Finish => "clFinish",
        create_buffer: CreateBuffer => "clCreateBuffer",
        release_mem_object: ReleaseMemObject => "clReleaseMemObject",
        create_program_with_source: CreateProgramWithSource => "clCreateProgramWithSource",
        release_program: ReleaseProgram => "clReleaseProgram",
        build_program: BuildProgram => "clBuildProgram",
        get_program_info: GetProgramInfo => "clGetProgramInfo",
        get_program_build_info: GetProgramBuildInfo => "clGetProgramBuildInfo",
        create_kernel: CreateKernel => "clCreateKernel",
        release_kernel: ReleaseKernel => "clReleaseKernel",
        set_kernel_arg: SetKernelArg => "clSetKernelArg",
        get_kernel_info: GetKernelInfo => "clGetKernelInfo",
        get_kernel_work_group_info: GetKernelWorkGroupInfo => "clGetKernelWorkGroupInfo",
        enqueue_read_buffer: EnqueueReadBuffer => "clEnqueueReadBuffer",
        enqueue_write_buffer: EnqueueWriteBuffer => "clEnqueueWriteBuffer",
        enqueue_copy_buffer: EnqueueCopyBuffer => "clEnqueueCopyBuffer",
        enqueue_nd_range_kernel: EnqueueNdRangeKernel => "clEnqueueNDRangeKernel",
        wait_for_events: WaitForEvents => "clWaitForEvents",
        release_event: ReleaseEvent => "clReleaseEvent",
        get_event_info: GetEventInfo => "clGetEventInfo",
        get_event_profiling_info: GetEventProfilingInfo => "clGetEventProfilingInfo",
    ],
    optional: [
        create_program_with_il: CreateProgramWithIL => "clCreateProgramWithIL",
        enqueue_fill_buffer: EnqueueFillBuffer => "clEnqueueFillBuffer",
        create_buffer_with_properties: CreateBufferWithProperties
            => "clCreateBufferWithProperties",
        get_extension_function_address: GetExtensionFunctionAddress
            => "clGetExtensionFunctionAddress",
        get_extension_function_address_for_platform: GetExtensionFunctionAddressForPlatform
            => "clGetExtensionFunctionAddressForPlatform",
    ]
}

impl Api {
    /// Unwraps an optional entry point, converting absence into
    /// [`Error::Unsupported`] naming `symbol`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the driver does not export the
    /// entry point.
    #[inline]
    pub(crate) fn optional<T: Copy>(value: Option<T>, symbol: &'static str) -> Result<T> {
        value.ok_or(Error::Unsupported { symbol })
    }

    /// Resolves a `clXxxKHR` extension entry point of one platform.
    ///
    /// The platform-scoped query is tried first: under an ICD loader the
    /// unscoped [`GetExtensionFunctionAddress`] has no implementation to
    /// dispatch to and returns null on multi-vendor installations, while the
    /// scoped form reaches the vendor that owns `platform`. Core names return
    /// `None` from both forms, as mandated by the specification.
    ///
    /// # Safety
    ///
    /// `platform` must be a platform handle of the implementation this
    /// [`Api`] was resolved from, and the returned pointer must only be
    /// called with the signature the extension defines, and only while the
    /// driver library that provided this [`Api`] stays loaded.
    pub(crate) unsafe fn extension_function(
        &self,
        platform: sys::PlatformHandle,
        name: &CStr,
    ) -> Option<*mut c_void> {
        // SAFETY: the caller upholds the handle-identity and lifetime
        // requirements; a null result simply means "not supported".
        let function = unsafe {
            match self.get_extension_function_address_for_platform {
                Some(get) => get(platform, name.as_ptr()),
                // Pre-1.2 drivers only export the unscoped form; using it
                // keeps single-implementation installations working.
                None => match self.get_extension_function_address {
                    Some(get) => get(name.as_ptr()),
                    None => return None,
                },
            }
        };
        (!function.is_null()).then_some(function)
    }

    /// Resolves the extension entry point named by `name` as the typed
    /// function pointer `T`.
    ///
    /// Returns `None` when the implementation does not export it; callers
    /// turn absence into [`Error::Unsupported`] with the extension name.
    ///
    /// # Safety
    ///
    /// `T` must be the entry-point type declared in [`sys`] for the function
    /// named by `name`, and `platform` must satisfy the requirements of
    /// [`Self::extension_function`].
    pub(crate) unsafe fn extension_fn<T: Copy>(
        &self,
        platform: sys::PlatformHandle,
        name: &CStr,
    ) -> Option<T> {
        let raw = unsafe { self.extension_function(platform, name) }?;
        let address = raw as usize;
        debug_assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
        // SAFETY: `T` is the entry-point type of the function named by
        // `name` (caller's obligation) and every entry-point type in [`sys`]
        // is a plain, pointer-wide function pointer, so copying
        // `size_of::<T>()` bytes out of `address` reproduces the pointer
        // that `extension_function` returned.
        Some(unsafe { core::mem::transmute_copy::<usize, T>(&address) })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use alloc::boxed::Box;
    use core::ffi::c_char;

    #[test]
    fn optional_reports_unsupported_entry_points() {
        let result: Result<unsafe extern "system" fn() -> i32> = Api::optional(None, "clEnqueueFillBuffer");
        assert!(matches!(
            result,
            Err(Error::Unsupported {
                symbol: "clEnqueueFillBuffer"
            })
        ));
        assert!(Api::optional(Some(1u32), "anything").is_ok());
    }

    #[test]
    fn from_dispatch_rejects_a_table_with_null_required_slots() {
        // SAFETY: zeroing is valid for every field — `Option<fn>` maps null
        // to `None` and raw pointers accept null — so the value is complete.
        let table: Box<IcdDispatch> = unsafe { Box::new(core::mem::MaybeUninit::zeroed().assume_init()) };

        // SAFETY: `table` is a live, fully initialized dispatch table kept
        // alive by the box for the whole test.
        let error = unsafe { Api::from_dispatch(&*table) }.err();
        assert!(
            matches!(
                error,
                Some(Error::MissingSymbol {
                    symbol: "clGetPlatformIDs"
                })
            ),
            "expected the first required slot to be reported, got {error:?}"
        );
    }

    /// The address a fake driver hands back for a resolved extension.
    const SENTINEL: usize = 0xdead_beef;

    /// The address of `function`, as the raw slot value a dispatch table
    /// stores.
    const fn slot_of(function: *const ()) -> *mut c_void {
        function as *mut c_void
    }

    /// The platform handle the fake resolvers are called with; the stubs
    /// ignore it and the tests never dereference it.
    pub(crate) fn api_platform() -> sys::PlatformHandle {
        sys::PlatformHandle::from_raw(core::ptr::dangling_mut::<c_void>())
    }

    /// A platform-scoped resolver that only knows the semaphore entry
    /// points, like a driver without `cl_khr_external_memory`.
    unsafe extern "system" fn scoped_stub(
        _platform: sys::PlatformHandle,
        name: *const c_char,
    ) -> *mut c_void {
        // SAFETY: `extension_fn` always passes the pointer of a `c"…"`
        // literal, which stays valid for the duration of the call.
        let text = unsafe { CStr::from_ptr(name) };
        let known = text == c"clCreateSemaphoreWithPropertiesKHR"
            || text == c"clEnqueueWaitSemaphoresKHR"
            || text == c"clRetainSemaphoreKHR"
            || text == c"clReleaseSemaphoreKHR";
        if known {
            SENTINEL as *mut c_void
        } else {
            core::ptr::null_mut()
        }
    }

    /// The legacy unscoped resolver, used only when the platform-scoped
    /// slot is absent.
    unsafe extern "system" fn unscoped_stub(name: *const c_char) -> *mut c_void {
        // SAFETY: as for `scoped_stub`.
        let text = unsafe { CStr::from_ptr(name) };
        if text == c"clEnqueueWaitSemaphoresKHR" {
            SENTINEL as *mut c_void
        } else {
            core::ptr::null_mut()
        }
    }

    /// A live function used to fill the slots the tests never call, so that
    /// every required slot of the fake table is non-null.
    unsafe extern "system" fn filler() -> i32 {
        sys::SUCCESS
    }

    /// Builds an [`Api`] from a synthetic dispatch table whose required
    /// slots are non-null and whose resolver slots (65 and 107 of
    /// `cl_icd_dispatch`) are set on demand.
    pub(crate) fn api_with_fake_dispatch(scoped: bool, unscoped: bool) -> Api {
        let filler_slot = slot_of(filler as *const ());
        let mut slots = [filler_slot; 150];
        slots[65] = if unscoped {
            slot_of(unscoped_stub as *const ())
        } else {
            core::ptr::null_mut()
        };
        slots[107] = if scoped {
            slot_of(scoped_stub as *const ())
        } else {
            core::ptr::null_mut()
        };
        // SAFETY: the array is 150 pointer-wide slots, the exact layout of
        // `IcdDispatch` (asserted by `dispatch::tests`), and each value is
        // either null or the address of a live `extern "system"` function.
        // The typed slots are never called, only copied into the `Api`.
        unsafe { Api::from_dispatch(slots.as_ptr().cast()) }
            .expect("a table with non-null required slots must resolve")
    }

    /// A driver that exports the platform-scoped resolver must be queried
    /// through it: that is the only form an ICD loader dispatches correctly.
    #[test]
    fn extension_resolution_prefers_the_platform_scoped_resolver() {
        let api = api_with_fake_dispatch(true, false);
        // Only the scoped stub answers this name, so a hit proves which slot
        // was called.
        let resolved: Option<sys::CreateSemaphoreWithProperties> =
            unsafe { api.extension_fn(api_platform(), c"clCreateSemaphoreWithPropertiesKHR") };
        assert!(resolved.is_some());
    }

    /// Without the scoped slot the legacy resolver still has to work.
    #[test]
    fn extension_resolution_falls_back_to_the_unscoped_resolver() {
        let api = api_with_fake_dispatch(false, true);
        let resolved: Option<sys::EnqueueWaitSemaphores> =
            unsafe { api.extension_fn(api_platform(), c"clEnqueueWaitSemaphoresKHR") };
        assert!(resolved.is_some());
    }

    /// An unknown extension name must resolve to `None`, not a wild pointer.
    #[test]
    fn extension_resolution_reports_unknown_names_as_none() {
        let api = api_with_fake_dispatch(true, true);
        let resolved: Option<sys::EnqueueAcquireExternalMemObjects> =
            unsafe { api.extension_fn(api_platform(), c"clEnqueueAcquireExternalMemObjectsKHR") };
        assert!(resolved.is_none());
    }

    /// The typed wrapper must hand back exactly the pointer the driver
    /// returned — a lost or truncated address would crash on first use.
    #[test]
    fn extension_resolution_keeps_the_driver_pointer() {
        let api = api_with_fake_dispatch(true, false);
        let raw = unsafe { api.extension_function(api_platform(), c"clEnqueueWaitSemaphoresKHR") };
        // SAFETY: reading a pointer's address is exactly what this test
        // checks; the value is the sentinel the stub returned, never called.
        assert_eq!(raw.expect("the stub resolves this name") as usize, SENTINEL);
    }

    #[test]
    fn from_symbols_resolves_a_real_icd_loader_when_one_is_installed() {
        // On machines with an OpenCL installation this validates the whole
        // symbol-resolution path; without one it only validates that the
        // failure stays a LibraryOpen diagnostic.
        let opened = crate::loader::open_first(crate::loader::client_candidates());
        match opened {
            Ok((library, _origin)) => {
                let built = Api::from_symbols(&library);
                assert!(
                    built.is_ok(),
                    "an installed ICD loader must export every required entry point: {:?}",
                    built.err()
                );
            }
            Err(error) => {
                assert!(
                    matches!(error, Error::LibraryOpen { .. }),
                    "expected Error::LibraryOpen when no loader exists, got {error:?}"
                );
            }
        }
    }
}
