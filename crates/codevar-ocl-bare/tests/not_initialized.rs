//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Driver behavior *before* [`codevar_ocl_bare::init`] has run.
//!
//! Every `cl*` entry point must fail the way a loader with an empty ICD
//! manifest would — `CL_PLATFORM_NOT_FOUND_KHR` wherever the platform is
//! required — instead of dereferencing unpublished state. This binary
//! never initializes the platform, so the assertions hold no matter which
//! test of this file runs first.

use codevar_ocl_bare::api::{clCreateContext, clGetPlatformIDs};
use codevar_ocl_bare::sys::{self, DeviceHandle, PlatformHandle};
use codevar_ocl_bare::{Error, platform};

#[test]
fn platform_ids_reports_zero_platforms_before_init() {
    let mut count = usize::MAX;
    let code = unsafe { clGetPlatformIDs(0, core::ptr::null_mut(), &mut count) };
    assert_eq!(code, sys::PLATFORM_NOT_FOUND_KHR);
    assert_eq!(count, 0);
}

#[test]
fn platform_ids_ignores_a_missing_count_slot() {
    // The count slot is optional in the OpenCL specification: a null
    // slot is ignored, so the missing platform is what the caller
    // learns about.
    let code = unsafe { clGetPlatformIDs(4, core::ptr::null_mut(), core::ptr::null_mut()) };
    assert_eq!(code, sys::PLATFORM_NOT_FOUND_KHR);
}

#[test]
fn platform_array_without_entries_is_checked_after_the_platform_state() {
    // The caller asks for zero entries — a local argument error once the
    // platform exists. Before `init`, the missing platform wins, so a
    // discovering loader learns about the empty image first.
    let mut slot = PlatformHandle::from_raw(core::ptr::null_mut());
    let mut count = usize::MAX;
    let code = unsafe { clGetPlatformIDs(0, &mut slot, &mut count) };
    assert_eq!(code, sys::PLATFORM_NOT_FOUND_KHR);
    assert_eq!(count, 0);
    assert!(slot.is_null());
}

#[test]
fn context_creation_reports_the_missing_platform() {
    // The device array is validated only after the platform resolves, so
    // its contents are never read on this path.
    let devices = [DeviceHandle::from_raw(core::ptr::null_mut())];
    let mut errcode = sys::SUCCESS;
    let context = unsafe {
        clCreateContext(
            core::ptr::null(),
            1,
            devices.as_ptr(),
            core::ptr::null(),
            core::ptr::null_mut(),
            &mut errcode,
        )
    };
    assert!(context.is_null());
    assert_eq!(errcode, sys::PLATFORM_NOT_FOUND_KHR);
}

#[test]
fn crate_state_reports_not_initialized() {
    let error = platform().err().expect("no platform before init");
    assert_eq!(error, Error::NotInitialized);
    assert_eq!(error.to_cl(), sys::PLATFORM_NOT_FOUND_KHR);
}
