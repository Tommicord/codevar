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

//! SPIR-V binary integration: a kernel written in the Codevar dialect is
//! lowered through `codevar-ocl-ir`, assembled by `codevar-ocl-asm` into a
//! SPIR-V module, handed to `clCreateProgramWithBinary` and executed on the
//! interpreter back end. The test covers the whole compiler → driver →
//! execution chain, plus the surrounding error surface of binary loading.

use core::ffi::c_void;
use std::sync::{Once, OnceLock};

use codevar_ocl_bare::api::{clCreateBuffer, clCreateCommandQueue, clCreateContext, clCreateKernel, clCreateProgramWithBinary, clCreateProgramWithBuiltInKernels, clEnqueueNDRangeKernel, clEnqueueReadBuffer, clEnqueueWriteBuffer, clFinish, clGetDeviceIDs, clGetEventInfo, clGetPlatformIDs, clGetProgramInfo, clGetKernelInfo, clGetKernelWorkGroupInfo, clReleaseCommandQueue, clReleaseContext, clReleaseEvent, clReleaseKernel, clReleaseMemObject, clReleaseProgram, clSetKernelArg, clWaitForEvents};
use codevar_ocl_bare::sys::{
    self, CommandQueueHandle, ContextHandle, DeviceHandle, EventHandle, KernelHandle, MemHandle,
    ProgramHandle,
};
use codevar_ocl_bare::{ArgKind, Device, DeviceInfo, PlatformConfig, ServiceMode, init};

const VECTOR_ADD: &str = "\
#[kernel]
fn vector_add(a: *const int, b: *const int, out: *mut int, n: int) -> void {
    let i = get_global_id(0);
    if i < n {
        out[i] = a[i] + b[i];
    }
}
";

static AMP_INFO: DeviceInfo = DeviceInfo::new(
    "spirv-amp",
    "codevar-tests",
    sys::DEVICE_TYPE_ACCELERATOR,
    1,
    64,
    1 << 20,
    4096,
    256,
);

static SMP_INFO: DeviceInfo = DeviceInfo::new(
    "spirv-cpu",
    "codevar-tests",
    sys::DEVICE_TYPE_CPU,
    1,
    64,
    1 << 20,
    4096,
    256,
);

fn ensure_init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Keep the registry empty: this file only tests the binary path.
        let smp = codevar_ocl_bare::SmpDevice::new(SMP_INFO.clone(), 1)
            .expect("smp device");
        let amp = codevar_ocl_bare::AmpDevice::simulated(AMP_INFO.clone(), ServiceMode::Cooperative)
            .expect("amp device");
        let devices: &'static [&'static dyn Device] =
            Box::leak(vec![smp as &dyn Device, amp as &dyn Device].into_boxed_slice());
        init(PlatformConfig {
            name: "spirv-board",
            vendor: "codevar-tests",
            modules: &[],
            devices,
        })
        .expect("board registers");
    });
}

fn compiled_binary() -> Vec<u8> {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES
        .get_or_init(|| {
            let module = codevar_ocl_ir::lower::lower_source(VECTOR_ADD).expect("dialect lowers");
            codevar_ocl_asm::assemble_bytes(&module).expect("module assembles")
        })
        .clone()
}

fn device_handles() -> [DeviceHandle; 2] {
    ensure_init();
    let mut platform = sys::PlatformHandle::from_raw(core::ptr::null_mut());
    let mut count = 0usize;
    assert_eq!(
        unsafe { clGetPlatformIDs(1, &mut platform, &mut count) },
        sys::SUCCESS
    );
    let mut devices = [DeviceHandle::from_raw(core::ptr::null_mut()); 2];
    let code = unsafe {
        clGetDeviceIDs(platform, sys::DEVICE_TYPE_ALL, 2, devices.as_mut_ptr(), &mut count)
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(count, 2);
    devices
}

fn make_context(devices: &[DeviceHandle]) -> ContextHandle {
    let mut errcode = sys::SUCCESS;
    let context = unsafe {
        clCreateContext(
            core::ptr::null(),
            devices.len() as u32,
            devices.as_ptr(),
            core::ptr::null(),
            core::ptr::null_mut(),
            &mut errcode,
        )
    };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!context.is_null());
    context
}

fn make_queue(context: ContextHandle, device: DeviceHandle) -> CommandQueueHandle {
    let mut errcode = sys::SUCCESS;
    let queue = unsafe { clCreateCommandQueue(context, device, 0, &mut errcode) };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!queue.is_null());
    queue
}

fn make_buffer(context: ContextHandle, size: usize) -> MemHandle {
    let mut errcode = sys::SUCCESS;
    let buffer = unsafe { clCreateBuffer(context, 0, size, core::ptr::null_mut(), &mut errcode) };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!buffer.is_null());
    buffer
}

fn make_program_from_binary(
    context: ContextHandle,
    device: DeviceHandle,
    bytes: &[u8],
) -> ProgramHandle {
    let length = bytes.len();
    let pointer = bytes.as_ptr();
    let mut binary_status = sys::SUCCESS;
    let mut errcode = sys::SUCCESS;
    let program = unsafe {
        clCreateProgramWithBinary(
            context,
            1,
            &device,
            &length,
            &pointer,
            &mut binary_status,
            &mut errcode,
        )
    };
    assert_eq!(binary_status, sys::SUCCESS);
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!program.is_null());
    program
}

fn make_kernel(program: ProgramHandle, name: &str) -> KernelHandle {
    let mut c_name = name.as_bytes().to_vec();
    c_name.push(0);
    let mut errcode = sys::SUCCESS;
    let kernel = unsafe { clCreateKernel(program, c_name.as_ptr().cast(), &mut errcode) };
    assert_eq!(errcode, sys::SUCCESS);
    assert!(!kernel.is_null());
    kernel
}

fn set_buffer_arg(kernel: KernelHandle, index: u32, buffer: MemHandle) {
    let code = unsafe {
        clSetKernelArg(
            kernel,
            index,
            core::mem::size_of::<MemHandle>(),
            (&buffer as *const MemHandle).cast::<c_void>(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
}

fn set_scalar_arg(kernel: KernelHandle, index: u32, value: u32) {
    let code = unsafe {
        clSetKernelArg(kernel, index, 4, (&value as *const u32).cast::<c_void>())
    };
    assert_eq!(code, sys::SUCCESS);
}

fn enqueue_1d(
    queue: CommandQueueHandle,
    kernel: KernelHandle,
    global: usize,
    event: *mut EventHandle,
) -> i32 {
    let size = [global];
    unsafe {
        clEnqueueNDRangeKernel(
            queue,
            kernel,
            1,
            core::ptr::null(),
            size.as_ptr(),
            core::ptr::null(),
            0,
            core::ptr::null(),
            event,
        )
    }
}

fn write_buffer(queue: CommandQueueHandle, buffer: MemHandle, data: &[u8]) -> i32 {
    unsafe {
        clEnqueueWriteBuffer(
            queue,
            buffer,
            sys::BLOCKING,
            0,
            data.len(),
            data.as_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    }
}

fn read_buffer(queue: CommandQueueHandle, buffer: MemHandle, out: &mut [u8]) -> i32 {
    unsafe {
        clEnqueueReadBuffer(
            queue,
            buffer,
            sys::BLOCKING,
            0,
            out.len(),
            out.as_mut_ptr().cast(),
            0,
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    }
}

fn bytes_mut(words: &mut [u32]) -> &mut [u8] {
    // SAFETY: no padding in i32/u32.
    unsafe { core::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), words.len() * 4) }
}

fn event_status(event: EventHandle) -> i32 {
    let mut size = 0usize;
    let code = unsafe {
        clGetEventInfo(
            event,
            sys::EVENT_COMMAND_EXECUTION_STATUS,
            0,
            core::ptr::null_mut(),
            &mut size,
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(size, 4);
    let mut word = [0u8; 4];
    let code = unsafe {
        clGetEventInfo(
            event,
            sys::EVENT_COMMAND_EXECUTION_STATUS,
            word.len(),
            word.as_mut_ptr().cast(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
    i32::from_ne_bytes(word)
}

fn query_usize(
    query: impl Fn(usize, *mut c_void, *mut usize) -> i32,
) -> usize {
    let mut size = 0usize;
    let code = query(0, core::ptr::null_mut(), &mut size);
    assert_eq!(code, sys::SUCCESS);
    assert!(size >= core::mem::size_of::<usize>());
    let mut buf = vec![0u8; size];
    let code = query(size, buf.as_mut_ptr().cast(), core::ptr::null_mut());
    assert_eq!(code, sys::SUCCESS);
    usize::from_ne_bytes(buf[..core::mem::size_of::<usize>()].try_into().expect("one usize"))
}

fn query_string(
    query: impl Fn(usize, *mut c_void, *mut usize) -> i32,
) -> String {
    let mut size = 0usize;
    let code = query(0, core::ptr::null_mut(), &mut size);
    assert_eq!(code, sys::SUCCESS);
    assert!(size > 0);
    let mut buf = vec![0u8; size];
    let code = query(size, buf.as_mut_ptr().cast(), core::ptr::null_mut());
    assert_eq!(code, sys::SUCCESS);
    let text = String::from_utf8(buf).expect("strings are valid UTF-8");
    text.trim_end_matches('\0').to_string()
}

fn query_u32(
    query: impl Fn(usize, *mut c_void, *mut usize) -> i32,
) -> u32 {
    let mut size = 0usize;
    let code = query(0, core::ptr::null_mut(), &mut size);
    assert_eq!(code, sys::SUCCESS);
    assert!(size >= 4);
    let mut buf = vec![0u8; size];
    let code = query(size, buf.as_mut_ptr().cast(), core::ptr::null_mut());
    assert_eq!(code, sys::SUCCESS);
    u32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]])
}

#[test]
fn dialect_binary_produces_a_program_that_executes_end_to_end() {
    let devices = device_handles();
    let context = make_context(&devices[..1]);
    let queue = make_queue(context, devices[0]);
    let bytes = compiled_binary();
    let program = make_program_from_binary(context, devices[0], &bytes);

    assert_eq!(
        query_usize(|size, ptr, ret| unsafe {
            clGetProgramInfo(program, sys::PROGRAM_NUM_KERNELS, size, ptr, ret)
        }),
        1
    );
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetProgramInfo(program, sys::PROGRAM_KERNEL_NAMES, size, ptr, ret)
        }),
        "vector_add"
    );
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetProgramInfo(program, sys::PROGRAM_SOURCE, size, ptr, ret)
        }),
        ""
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetProgramInfo(program, sys::PROGRAM_BUILD_STATUS, size, ptr, ret)
        }),
        sys::BUILD_SUCCESS as u32
    );

    let options = b"-D SCALE=3\0";
    let code = unsafe {
        codevar_ocl_bare::api::clBuildProgram(
            program,
            0,
            core::ptr::null(),
            options.as_ptr().cast(),
            core::ptr::null(),
            core::ptr::null_mut(),
        )
    };
    assert_eq!(code, sys::SUCCESS);
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetProgramInfo(program, sys::PROGRAM_BUILD_OPTIONS, size, ptr, ret)
        }),
        "-D SCALE=3"
    );

    let kernel = make_kernel(program, "vector_add");
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetKernelInfo(kernel, sys::KERNEL_NUM_ARGS, size, ptr, ret)
        }),
        4
    );
    assert_eq!(
        query_string(|size, ptr, ret| unsafe {
            clGetKernelInfo(kernel, sys::KERNEL_FUNCTION_NAME, size, ptr, ret)
        }),
        "vector_add"
    );
    assert_eq!(
        query_u32(|size, ptr, ret| unsafe {
            clGetKernelWorkGroupInfo(
                kernel,
                devices[0],
                sys::KERNEL_WORK_GROUP_SIZE,
                size,
                ptr,
                ret,
            )
        }),
        64
    );

    let count = 8usize;
    let buffer_a = make_buffer(context, count * 4);
    let buffer_b = make_buffer(context, count * 4);
    let buffer_out = make_buffer(context, count * 4);

    let values_a: Vec<i32> = (0..count).map(|i| (i + 1) as i32).collect();
    let values_b: Vec<i32> = (0..count).map(|i| 2 * (i + 1) as i32).collect();
    let mut seed_a = vec![0u8; count * 4];
    for (i, v) in values_a.iter().enumerate() {
        let bytes = v.to_le_bytes();
        for (j, b) in bytes.iter().enumerate() {
            seed_a[i * 4 + j] = *b;
        }
    }
    let mut seed_b = vec![0u8; count * 4];
    for (i, v) in values_b.iter().enumerate() {
        let bytes = v.to_le_bytes();
        for (j, b) in bytes.iter().enumerate() {
            seed_b[i * 4 + j] = *b;
        }
    }
    assert_eq!(write_buffer(queue, buffer_a, &seed_a), sys::SUCCESS);
    assert_eq!(write_buffer(queue, buffer_b, &seed_b), sys::SUCCESS);

    set_buffer_arg(kernel, 0, buffer_a);
    set_buffer_arg(kernel, 1, buffer_b);
    set_buffer_arg(kernel, 2, buffer_out);
    set_scalar_arg(kernel, 3, count as u32);

    let global = [count];
    let mut event = EventHandle::from_raw(core::ptr::null_mut());
    assert_eq!(enqueue_1d(queue, kernel, count, &mut event), sys::SUCCESS);
    assert_eq!(event_status(event), sys::COMPLETE);
    assert_eq!(unsafe { clWaitForEvents(1, &event) }, sys::SUCCESS);

    let mut out = vec![0u8; count * 4];
    assert_eq!(read_buffer(queue, buffer_out, &mut out), sys::SUCCESS);
    for i in 0..count {
        let word = i32::from_le_bytes([out[i * 4], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3]]);
        assert_eq!(word, 3 * (i + 1) as i32, "index {i}");
    }

    // The `n` guard: a second run only fills the first half and leaves
    // the rest zero.
    let zero = vec![0u8; count * 4];
    let buffer_out2 = make_buffer(context, count * 4);
    assert_eq!(write_buffer(queue, buffer_out2, &zero), sys::SUCCESS);
    set_buffer_arg(kernel, 2, buffer_out2);
    set_scalar_arg(kernel, 3, (count / 2) as u32);
    assert_eq!(
        enqueue_1d(queue, kernel, count, core::ptr::null_mut()),
        sys::SUCCESS
    );
    let mut out2 = vec![0u8; count * 4];
    assert_eq!(read_buffer(queue, buffer_out2, &mut out2), sys::SUCCESS);
    for i in 0..(count / 2) {
        let word = i32::from_le_bytes([
            out2[i * 4],
            out2[i * 4 + 1],
            out2[i * 4 + 2],
            out2[i * 4 + 3],
        ]);
        assert_eq!(word, 3 * (i + 1) as i32, "guarded index {i}");
    }
    for index in (count / 2)..count {
        let word = i32::from_le_bytes([
            out2[index * 4],
            out2[index * 4 + 1],
            out2[index * 4 + 2],
            out2[index * 4 + 3],
        ]);
        assert_eq!(word, 0, "index {index} must stay zero");
    }

    assert_eq!(unsafe { clFinish(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseEvent(event) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseKernel(kernel) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseProgram(program) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(buffer_out2) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(buffer_out) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(buffer_b) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseMemObject(buffer_a) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseCommandQueue(queue) }, sys::SUCCESS);
    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}

#[test]
fn binary_creation_errors_report_exact_codes() {
    let devices = device_handles();
    let context = make_context(&devices[..1]);
    let garbage = [0u8; 16];
    let live = [0xBu8; 4];

    let length = garbage.len();
    let pointer = garbage.as_ptr();
    let mut binary_status = sys::SUCCESS;
    let mut errcode = sys::SUCCESS;
    let program = unsafe {
        clCreateProgramWithBinary(
            context,
            1,
            &devices[0],
            &length,
            &pointer,
            &mut binary_status,
            &mut errcode,
        )
    };
    assert!(program.is_null());
    assert_eq!(binary_status, sys::INVALID_BINARY);
    assert_eq!(errcode, sys::INVALID_BINARY);

    let pointer = live.as_ptr();
    let length = 0usize;
    let mut binary_status = sys::SUCCESS;
    let mut errcode = sys::SUCCESS;
    let program = unsafe {
        clCreateProgramWithBinary(
            context,
            1,
            &devices[0],
            &length,
            &pointer,
            &mut binary_status,
            &mut errcode,
        )
    };
    assert!(program.is_null());
    assert_eq!(binary_status, sys::INVALID_BINARY);
    assert_eq!(errcode, sys::INVALID_BINARY);

    let pointer = live.as_ptr();
    let length = live.len();
    let mut binary_status = sys::SUCCESS;
    let mut errcode = sys::SUCCESS;
    let program = unsafe {
        clCreateProgramWithBinary(
            context,
            0,
            core::ptr::null(),
            &length,
            &pointer,
            &mut binary_status,
            &mut errcode,
        )
    };
    assert!(program.is_null());
    assert_eq!(errcode, sys::INVALID_VALUE);
    // num_devices==0 hands back INVALID_VALUE regardless of the binary
    // status, but the driver still records it when requested.
    let _ = (binary_status, program);

    let mut binary_status = sys::SUCCESS;
    let mut errcode = sys::SUCCESS;
    let program = unsafe {
        clCreateProgramWithBinary(
            context,
            1,
            core::ptr::null(),
            &length,
            &pointer,
            &mut binary_status,
            &mut errcode,
        )
    };
    assert!(program.is_null());
    assert_eq!(errcode, sys::INVALID_VALUE);

    let mut binary_status = sys::SUCCESS;
    let mut errcode = sys::SUCCESS;
    let program = unsafe {
        clCreateProgramWithBinary(
            context,
            1,
            &devices[1],
            &length,
            &pointer,
            &mut binary_status,
            &mut errcode,
        )
    };
    assert!(program.is_null());
    assert_eq!(errcode, sys::INVALID_DEVICE);

    let mut errcode = sys::SUCCESS;
    let program = unsafe {
        clCreateProgramWithBuiltInKernels(
            context,
            1,
            &devices[0],
            b"fill\0".as_ptr().cast(),
            &mut errcode,
        )
    };
    assert!(program.is_null());
    assert_eq!(errcode, sys::INVALID_KERNEL_NAME);

    let mut errcode = sys::SUCCESS;
    let bad_program = unsafe {
        clCreateProgramWithBuiltInKernels(
            context,
            1,
            &devices[0],
            b"\0".as_ptr().cast(),
            &mut errcode,
        )
    };
    assert!(bad_program.is_null());
    assert_eq!(errcode, sys::INVALID_VALUE);

    assert_eq!(unsafe { clReleaseContext(context) }, sys::SUCCESS);
}