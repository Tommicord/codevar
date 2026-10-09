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

//! End-to-end smoke test: a vector add executed on a real OpenCL device.
//!
//! The test skips silently when the host has no usable OpenCL stack
//! (loader, device, or on-device compiler missing); once a device
//! builds the kernel, every following step must succeed.

use core::mem::size_of;

use codevar_ocl::{
    Arg, Buffer, CommandQueue, CommandState, Context, Error, ProfilingTimestamp, Program, QueueProperties,
    Runtime,
};

const KERNEL: &str = r#"
__kernel void vector_add(__global const int* a, __global const int* b,
                         __global int* out, int n) {
    int i = get_global_id(0);
    if (i < n) { out[i] = a[i] + b[i]; }
}
"#;

const SIZE: usize = 4096;

#[test]
fn vector_add_roundtrip_on_real_device() -> Result<(), Error> {
    let Ok(runtime) = Runtime::load() else {
        return Ok(());
    };
    let Some(device) = runtime
        .platforms()
        .iter()
        .flat_map(|platform| platform.all_devices().unwrap_or_default())
        .next()
    else {
        return Ok(());
    };
    let Ok(context) = Context::new(&device) else {
        return Ok(());
    };
    let Ok(queue) = CommandQueue::with_properties(&context, &device, QueueProperties::Profiling) else {
        return Ok(());
    };

    let a: Vec<i32> = (0..SIZE).map(|i| i as i32).collect();
    let b: Vec<i32> = vec![7i32; SIZE];
    let expected: Vec<i32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let a_buffer = Buffer::from_slice(&context, &a)?;
    let b_buffer = Buffer::from_slice(&context, &b)?;
    let out_buffer = Buffer::new(&context, SIZE * size_of::<i32>())?;

    let Ok(program) = Program::from_source(&context, KERNEL) else {
        return Ok(());
    };
    if program.build("").is_err() {
        return Ok(());
    }

    let mut kernel = program.kernel("vector_add")?;
    assert_eq!(kernel.name()?, "vector_add");
    assert_eq!(kernel.num_args()?, 4);
    kernel.set_arg(0, Arg::Buffer(&a_buffer))?;
    kernel.set_arg(1, Arg::Buffer(&b_buffer))?;
    kernel.set_arg(2, Arg::Buffer(&out_buffer))?;
    kernel.set_arg(3, Arg::U32(SIZE as u32))?;

    let event = queue.enqueue_kernel(&kernel, &[SIZE], None)?;
    event.wait()?;
    assert_eq!(event.state()?, CommandState::Complete);
    if let (Ok(start), Ok(end)) = (
        event.profiling_timestamp(ProfilingTimestamp::Started),
        event.profiling_timestamp(ProfilingTimestamp::Finished),
    ) {
        assert!(end >= start, "device timestamps must be monotonic");
    }

    let mut out = vec![0i32; SIZE];
    let _read_event = queue.read_buffer(&out_buffer, 0, &mut out)?;
    assert_eq!(out, expected);
    Ok(())
}
