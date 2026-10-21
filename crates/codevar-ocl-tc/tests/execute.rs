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

//! The native stage, executed.
//!
//! The test compiles a kernel to machine code, copies the code into
//! executable memory, and calls it with the state a driver would hand
//! it: the per-work-item block and the argument block it points at.
//! The buffer the kernel writes is then checked.
//!
//! The test runs on the host, so it only makes sense on an
//! architecture the host back end covers.

#![cfg(target_arch = "x86_64")]

use std::ffi::c_void;

use codevar_ocl_abi::{ARG_SLOT, WorkItem};
use codevar_ocl_tc::compile_source;

/// The kernel under test: it multiplies the element the work-item's
/// global id points at by the factor argument.
const KERNEL: &str = "\
#[kernel]
fn scale(out: *mut int, factor: int) -> void {
    let i = get_global_id(0);
    out[i] = out[i] * factor;
}
";

/// A page of executable memory holding the compiled code.
struct Executable {
    /// The mapping's base address.
    base: *mut u8,
    /// The mapping's length in bytes.
    len: usize,
}

impl Executable {
    /// Maps `len` bytes of readable, writable and executable memory.
    fn new(len: usize) -> Self {
        let page = 4096usize;
        let rounded = len.div_ceil(page) * page;
        // SAFETY: the mapping is created with the protections the test
        // needs, and it is unmapped in `Drop`.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                rounded,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(base, libc::MAP_FAILED, "the executable memory could not be mapped");
        Self {
            base: base.cast::<u8>(),
            len: rounded,
        }
    }

    /// Copies `code` into the mapping and returns the address of its
    /// entry point.
    fn load(&mut self, code: &[u8], entry_offset: u32) -> *mut u8 {
        assert!(code.len() <= self.len, "the code does not fit the mapping");
        // SAFETY: the mapping is at least `code.len()` bytes long and
        // writable.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), self.base, code.len());
        }
        // SAFETY: the entry offset lies inside the code, which was
        // just copied into the mapping.
        unsafe { self.base.add(entry_offset as usize) }
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        // SAFETY: the mapping was created in `new` and is unmapped
        // exactly once.
        unsafe {
            libc::munmap(self.base.cast::<c_void>(), self.len);
        }
    }
}

/// The state a driver hands to a kernel: the work-item block and the
/// argument block it points at.
struct Launch {
    /// The work-item block.
    item: WorkItem,
    /// The argument block.
    args: [u8; 256],
}

impl Launch {
    /// Builds a launch whose first argument is `buffer` and whose
    /// second is the scalar `factor`.
    fn new(buffer: *mut i32, factor: i32) -> Self {
        let mut args = [0u8; 256];
        let pointer = buffer as usize;
        args[..size_of::<usize>()].copy_from_slice(&pointer.to_ne_bytes());
        args[ARG_SLOT..ARG_SLOT + 4].copy_from_slice(&factor.to_ne_bytes());
        Self {
            item: WorkItem {
                global_id: [2, 0, 0],
                local_id: [0; 3],
                group_id: [0; 3],
                global_offset: [0; 3],
                num_groups: [1, 1, 1],
                local_size: [1, 1, 1],
                arg_base: args.as_mut_ptr(),
            },
            args,
        }
    }
}

#[test]
fn the_compiled_kernel_multiplies_the_element() {
    let code = compile_source(KERNEL, "scale").expect("the kernel compiles");
    assert!(!code.bytes().is_empty());

    let mut executable = Executable::new(code.bytes().len());
    let entry = executable.load(code.bytes(), code.entry_offset());

    let mut buffer = [3i32, 5i32, 7i32];
    let mut launch = Launch::new(buffer.as_mut_ptr(), 4);

    // SAFETY: the code was compiled for this architecture and follows
    // the kernel calling convention: it takes the work-item pointer
    // and reads its arguments from the block that pointer names.
    let kernel: unsafe extern "C" fn(*mut WorkItem) = unsafe { std::mem::transmute(entry) };
    unsafe {
        kernel(&mut launch.item);
    }

    assert_eq!(buffer, [12, 20, 28]);
}
