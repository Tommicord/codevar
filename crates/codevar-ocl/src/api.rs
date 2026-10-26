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
        get_extension_function_address: GetExtensionFunctionAddress
            => "clGetExtensionFunctionAddress",
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

    /// Resolves an extension entry point by its `clXxxKHR` name.
    ///
    /// Core names return `None`, as mandated by the ICD specification (and
    /// verified against Intel's driver): only extension names resolve.
    ///
    /// # Safety
    ///
    /// The returned pointer must only be called with the signature the
    /// extension defines, and only while the driver library that provided
    /// this [`Api`] stays loaded.
    #[allow(dead_code)] // reserved for the planned extension-call surface
    pub(crate) unsafe fn extension_function(&self, name: &CStr) -> Option<*mut c_void> {
        let get = self.get_extension_function_address?;
        // SAFETY: the caller upholds the driver-lifetime requirement; this
        // only forwards the name to the driver, which returns either null
        // or a pointer to the requested extension function.
        let function = unsafe { get(name.as_ptr()) };
        (!function.is_null()).then_some(function)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;

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
