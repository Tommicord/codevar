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

//! Platform publication and discovery: `init` under concurrency, the
//! platform queries, device enumeration with type filters, and the
//! `clGetDeviceInfo` descriptions of two registered back ends.
//!
//! The board this file registers is one SMP CPU device and one simulated
//! AMP peer, with a two-kernel registry — enough to exercise every
//! discovery path the embedded profile exposes.

use core::ffi::c_void;
use std::borrow::Cow;
use std::sync::OnceLock;

use codevar_ocl_bare::api::{clGetDeviceIDs, clGetDeviceInfo, clGetPlatformIDs, clGetPlatformInfo};
use codevar_ocl_bare::sys::{self, DeviceHandle, PlatformHandle};
use codevar_ocl_bare::{
    AmpDevice, ArgKind, Device, DeviceInfo, KernelDef, KernelModule, PlatformConfig, ServiceMode, SmpDevice,
    SpirvBinding, WorkItem, init, platform,
};

/// Registry entry with one buffer argument, reported as `probe`.
const PROBE: KernelDef = KernelDef {
    name: Cow::Borrowed("probe"),
    func: probe,
    kinds: Cow::Borrowed(&[ArgKind::Buffer]),
    binding: SpirvBinding::NONE,
};

/// Registry entry with a buffer and a scalar, reported as `scale`.
const SCALE: KernelDef = KernelDef {
    name: Cow::Borrowed("scale"),
    func: probe,
    kinds: Cow::Borrowed(&[ArgKind::Buffer, ArgKind::Scalar(4)]),
    binding: SpirvBinding::NONE,
};

static MODULE: KernelModule = KernelModule {
    name: "discovery",
    kernels: &[PROBE, SCALE],
};

static MODULES: &[&KernelModule] = &[&MODULE];

/// Never executed here: discovery only resolves the definitions.
unsafe extern "C" fn probe(_item: *mut WorkItem) {}

static SMP_INFO: DeviceInfo = DeviceInfo::new(
    "smp-cpu",
    "codevar-tests",
    sys::DEVICE_TYPE_CPU,
    3,
    64,
    1 << 20,
    4096,
    256,
);

static AMP_INFO: DeviceInfo = DeviceInfo::new(
    "amp-peer",
    "codevar-tests",
    sys::DEVICE_TYPE_ACCELERATOR,
    1,
    64,
    1 << 20,
    4096,
    256,
);

/// Builds (once) the platform description this file's tests register.
fn config() -> &'static PlatformConfig {
    static CELL: OnceLock<PlatformConfig> = OnceLock::new();
    CELL.get_or_init(|| {
        let smp = SmpDevice::new(SMP_INFO.clone(), 3).unwrap();
        let amp = AmpDevice::simulated(AMP_INFO.clone(), ServiceMode::Cooperative).unwrap();
        let devices: &'static [&'static dyn Device] =
            Box::leak(vec![smp as &dyn Device, amp as &dyn Device].into_boxed_slice());
        PlatformConfig {
            name: "bare-test-board",
            vendor: "codevar-tests",
            modules: MODULES,
            devices,
        }
    })
}

/// Registers the board once; safe to call from every test and thread
/// because every caller passes the identical configuration.
fn ensure_init() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| init(*config()).expect("board registers"));
}

/// The platform handle as a discovering application sees it.
fn platform_handle() -> PlatformHandle {
    ensure_init();
    let mut handle = PlatformHandle::from_raw(core::ptr::null_mut());
    let mut count = 0usize;
    let code = unsafe { clGetPlatformIDs(1, &mut handle, &mut count) };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(count, 1);
    assert!(!handle.is_null());
    handle
}

/// Runs an `clGet*Info` string query through the size-probe protocol and
/// returns the payload without its terminator.
fn query_string(query: impl Fn(usize, *mut c_void, *mut usize) -> i32) -> String {
    let mut size = 0usize;
    assert_eq!(query(0, core::ptr::null_mut(), &mut size), sys::SUCCESS);
    let mut buf = vec![0u8; size];
    assert_eq!(
        query(size, buf.as_mut_ptr().cast(), core::ptr::null_mut()),
        sys::SUCCESS
    );
    assert_eq!(buf.last().copied(), Some(0));
    let text = String::from_utf8(buf).expect("info strings are UTF-8");
    text.trim_end_matches('\0').to_string()
}

/// Runs an `clGet*Info` scalar query through the size-probe protocol.
fn query_scalar(query: impl Fn(usize, *mut c_void, *mut usize) -> i32) -> Vec<u8> {
    let mut size = 0usize;
    assert_eq!(query(0, core::ptr::null_mut(), &mut size), sys::SUCCESS);
    assert!(size > 0, "scalar queries report their width");
    let mut buf = vec![0u8; size];
    assert_eq!(
        query(size, buf.as_mut_ptr().cast(), core::ptr::null_mut()),
        sys::SUCCESS
    );
    buf
}

fn query_u32(query: impl Fn(usize, *mut c_void, *mut usize) -> i32) -> u32 {
    let bytes = query_scalar(query);
    assert_eq!(bytes.len(), 4);
    u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn query_u64(query: impl Fn(usize, *mut c_void, *mut usize) -> i32) -> u64 {
    let bytes = query_scalar(query);
    assert_eq!(bytes.len(), 8);
    let mut word = [0u8; 8];
    word.copy_from_slice(&bytes);
    u64::from_ne_bytes(word)
}

fn query_usize(query: impl Fn(usize, *mut c_void, *mut usize) -> i32) -> usize {
    let bytes = query_scalar(query);
    let word: [usize; 1] = [usize::from_ne_bytes(
        bytes
            .as_slice()
            .try_into()
            .expect("one native usize of bytes"),
    )];
    word[0]
}

/// Two registered devices: the SMP CPU first, the AMP peer second.
fn both_devices(platform: PlatformHandle) -> [DeviceHandle; 2] {
    let mut devices = [DeviceHandle::from_raw(core::ptr::null_mut()); 2];
    let mut count = 0usize;
    let code = unsafe {
        clGetDeviceIDs(
            platform,
            sys::DEVICE_TYPE_ALL,
            2,
            devices.as_mut_ptr(),
            &mut count,
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(count, 2);
    assert!(!devices[0].is_null());
    assert!(!devices[1].is_null());
    assert_ne!(devices[0], devices[1]);
    devices
}

#[test]
fn concurrent_init_calls_settle_on_the_first_platform() {
    // Eight threads race to publish the identical board; first-call-wins
    // must give every caller the same platform.
    let config = *config();
    let racers: Vec<_> = (0..8)
        .map(|_| std::thread::spawn(move || init(config)))
        .collect();
    for racer in racers {
        racer
            .join()
            .expect("racer thread")
            .expect("init succeeds");
    }
    let state = platform().expect("board published");
    assert_eq!(state.name(), "bare-test-board");
    assert_eq!(state.vendor(), "codevar-tests");

    // A later call with a different description is a no-op: the first
    // platform cannot be replaced, not even with an invalid one.
    let impostor = PlatformConfig {
        name: "impostor",
        vendor: "impostor",
        modules: &[],
        devices: &[],
    };
    init(impostor).expect("idempotent");
    assert_eq!(platform().expect("still published").name(), "bare-test-board");
    assert_eq!(
        platform()
            .expect("still published")
            .devices()
            .len(),
        2
    );
}

#[test]
fn platform_info_round_trips_every_query() {
    let handle = platform_handle();
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetPlatformInfo(handle, sys::PLATFORM_NAME, size, ptr, ret)
        }),
        "bare-test-board"
    );
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetPlatformInfo(handle, sys::PLATFORM_VENDOR, size, ptr, ret)
        }),
        "codevar-tests"
    );
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetPlatformInfo(handle, sys::PLATFORM_PROFILE, size, ptr, ret)
        }),
        "EMBEDDED_PROFILE"
    );
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetPlatformInfo(handle, sys::PLATFORM_VERSION, size, ptr, ret)
        }),
        "OpenCL 1.2 Codevar-bare"
    );
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetPlatformInfo(handle, sys::PLATFORM_EXTENSIONS, size, ptr, ret)
        }),
        ""
    );

    // An undersized buffer is rejected while the required size is still
    // reported, per the `clGet*Info` contract.
    let mut size = 0usize;
    let code = unsafe { clGetPlatformInfo(handle, sys::PLATFORM_NAME, 0, core::ptr::null_mut(), &mut size) };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(size, "bare-test-board".len() + 1);
    let mut small = [0u8; 4];
    let mut again = 0usize;
    let code = unsafe {
        clGetPlatformInfo(
            handle,
            sys::PLATFORM_NAME,
            small.len(),
            small.as_mut_ptr().cast(),
            &mut again,
        )
    };
    assert_eq!(code, sys::INVALID_VALUE);
    assert_eq!(again, size);

    let code = unsafe { clGetPlatformInfo(handle, 0xFFFF, 0, core::ptr::null_mut(), core::ptr::null_mut()) };
    assert_eq!(code, sys::INVALID_VALUE);

    // Forged and null platform handles are rejected without dereference.
    let forged = PlatformHandle::from_raw((core::ptr::null_mut::<core::ffi::c_void>()).wrapping_byte_add(8));
    let code = unsafe {
        clGetPlatformInfo(
            forged,
            sys::PLATFORM_NAME,
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_PLATFORM);
    let code = unsafe {
        clGetPlatformInfo(
            PlatformHandle::from_raw(core::ptr::null_mut()),
            sys::PLATFORM_NAME,
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_PLATFORM);
}

#[test]
fn device_enumeration_honours_type_filters() {
    let handle = platform_handle();
    let devices = both_devices(handle);

    // Count-only queries report the same total.
    let mut count = 0usize;
    let code = unsafe { clGetDeviceIDs(handle, sys::DEVICE_TYPE_ALL, 0, core::ptr::null_mut(), &mut count) };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(count, 2);

    // The SMP device is the platform's default (first) device.
    let mut single = [DeviceHandle::from_raw(core::ptr::null_mut()); 1];
    let mut count = 0usize;
    let code = unsafe {
        clGetDeviceIDs(
            handle,
            sys::DEVICE_TYPE_DEFAULT,
            1,
            single.as_mut_ptr(),
            &mut count,
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(count, 1);
    assert_eq!(single[0], devices[0]);

    let mut count = 0usize;
    let code = unsafe {
        clGetDeviceIDs(
            handle,
            sys::DEVICE_TYPE_ACCELERATOR,
            1,
            single.as_mut_ptr(),
            &mut count,
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(count, 1);
    assert_eq!(single[0], devices[1]);

    // No GPU is registered: the count is zero and the spec's
    // `CL_DEVICE_NOT_FOUND` reports it.
    let mut count = 1usize;
    let mut slots = [DeviceHandle::from_raw(core::ptr::null_mut()); 1];
    let code = unsafe { clGetDeviceIDs(handle, sys::DEVICE_TYPE_GPU, 1, slots.as_mut_ptr(), &mut count) };
    assert_eq!(code, sys::DEVICE_NOT_FOUND);
    assert_eq!(count, 0);

    // Argument validation around the enumeration.
    let code = unsafe {
        clGetDeviceIDs(
            handle,
            sys::DEVICE_TYPE_ALL,
            1,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_VALUE);
    let mut count = 0usize;
    let code = unsafe { clGetDeviceIDs(handle, sys::DEVICE_TYPE_ALL, 0, slots.as_mut_ptr(), &mut count) };
    assert_eq!(code, sys::INVALID_VALUE);
    let code = unsafe { clGetDeviceIDs(handle, 1u64 << 40, 1, slots.as_mut_ptr(), &mut count) };
    assert_eq!(code, sys::INVALID_DEVICE_TYPE);

    // A forged platform handle zeroes the count and is rejected.
    let forged = PlatformHandle::from_raw((core::ptr::null_mut::<core::ffi::c_void>()).wrapping_byte_add(8));
    count = usize::MAX;
    let code = unsafe { clGetDeviceIDs(forged, sys::DEVICE_TYPE_ALL, 1, slots.as_mut_ptr(), &mut count) };
    assert_eq!(code, sys::INVALID_PLATFORM);
    assert_eq!(count, 0);
}

#[test]
fn device_info_reflects_both_registered_back_ends() {
    let handle = platform_handle();
    let [cpu, accel] = both_devices(handle);

    assert_eq!(
        query_string(|size, ptr, ret| unsafe { clGetDeviceInfo(cpu, sys::DEVICE_NAME, size, ptr, ret) }),
        "smp-cpu"
    );
    assert_eq!(
        query_string(|size, ptr, ret| unsafe { clGetDeviceInfo(cpu, sys::DEVICE_VENDOR, size, ptr, ret) }),
        "codevar-tests"
    );
    assert_eq!(
        query_u64(|size, ptr, ret| unsafe { clGetDeviceInfo(cpu, sys::DEVICE_TYPE, size, ptr, ret) }),
        sys::DEVICE_TYPE_CPU
    );
    assert_eq!(
        query_u64(|size, ptr, ret| unsafe { clGetDeviceInfo(accel, sys::DEVICE_TYPE, size, ptr, ret) }),
        sys::DEVICE_TYPE_ACCELERATOR
    );

    // The SMP description mirrors what `SmpDevice::new` was configured
    // with: three cores, a 64-item work-group cap, a 256-byte argument
    // block.
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetDeviceInfo(cpu, sys::DEVICE_MAX_COMPUTE_UNITS, size, ptr, ret)
        }),
        3
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetDeviceInfo(cpu, sys::DEVICE_MAX_WORK_GROUP_SIZE, size, ptr, ret)
        }),
        64
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetDeviceInfo(cpu, sys::DEVICE_MAX_PARAMETER_SIZE, size, ptr, ret)
        }),
        256
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetDeviceInfo(cpu, sys::DEVICE_MAX_WORK_ITEM_DIMENSIONS, size, ptr, ret)
        }),
        3
    );
    let sizes = query_scalar(|size, ptr, ret| unsafe {
        clGetDeviceInfo(cpu, sys::DEVICE_MAX_WORK_ITEM_SIZES, size, ptr, ret)
    });
    assert_eq!(sizes.len(), 3 * 4);
    assert!(
        sizes
            .chunks(4)
            .all(|chunk| { u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) == 64 })
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe { clGetDeviceInfo(cpu, sys::DEVICE_ADDRESS_BITS, size, ptr, ret) }),
        (core::mem::size_of::<usize>() * 8) as u32
    );
    assert_eq!(
        query_u64(|size, ptr, ret| unsafe {
            clGetDeviceInfo(cpu, sys::DEVICE_GLOBAL_MEM_SIZE, size, ptr, ret)
        }),
        1 << 20
    );
    assert_eq!(
        query_u64(|size, ptr, ret| unsafe {
            clGetDeviceInfo(cpu, sys::DEVICE_MAX_MEM_ALLOC_SIZE, size, ptr, ret)
        }),
        (1 << 20) / 4
    );
    assert_eq!(
        query_u64(|size, ptr, ret| unsafe {
            clGetDeviceInfo(cpu, sys::DEVICE_LOCAL_MEM_SIZE, size, ptr, ret)
        }),
        4096
    );

    // The embedded-profile facts every query must agree on.
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetDeviceInfo(cpu, sys::DEVICE_IMAGE_SUPPORT, size, ptr, ret)
        }),
        sys::FALSE
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe { clGetDeviceInfo(cpu, sys::DEVICE_AVAILABLE, size, ptr, ret) }),
        sys::TRUE
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetDeviceInfo(cpu, sys::DEVICE_COMPILER_AVAILABLE, size, ptr, ret)
        }),
        sys::FALSE
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetDeviceInfo(cpu, sys::DEVICE_HOST_UNIFIED_MEMORY, size, ptr, ret)
        }),
        sys::TRUE
    );

    // The registry renders as the `;`-separated builtin-kernel list.
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetDeviceInfo(cpu, sys::DEVICE_BUILTIN_KERNELS, size, ptr, ret)
        }),
        "probe;scale"
    );

    // The device's platform pointer round-trips through the handle.
    assert_eq!(
        query_usize(|size, ptr, ret| unsafe { clGetDeviceInfo(cpu, sys::DEVICE_PLATFORM, size, ptr, ret) }),
        handle.as_raw() as usize
    );

    // Unknown parameters and undersized buffers are rejected.
    let code = unsafe { clGetDeviceInfo(cpu, 0xFFFF, 0, core::ptr::null_mut(), core::ptr::null_mut()) };
    assert_eq!(code, sys::INVALID_VALUE);
    let mut small = [0u8; 2];
    let mut size = 0usize;
    let code = unsafe {
        clGetDeviceInfo(
            cpu,
            sys::DEVICE_NAME,
            small.len(),
            small.as_mut_ptr().cast(),
            &mut size,
        )
    };
    assert_eq!(code, sys::INVALID_VALUE);
    assert_eq!(size, "smp-cpu".len() + 1);
    let code = unsafe {
        clGetDeviceInfo(
            DeviceHandle::from_raw(core::ptr::null_mut()),
            sys::DEVICE_NAME,
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::INVALID_DEVICE);
}
