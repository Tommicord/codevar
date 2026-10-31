// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

//! One handle, followed across the border between the two drivers.
//!
//! Every test here starts from the pairing question the drivers themselves
//! answer, so it only runs on a machine whose OpenCL and Vulkan drivers
//! describe the same hardware — a machine where they do not, or where one of
//! the two is missing, has no exchange to check and returns early.

#![cfg(unix)]

mod support;

use std::sync::OnceLock;

use codevar_ocl::{CommandQueue, Context, Device, Runtime};
use codevar_ocl_interop::{MemoryKind, SemaphoreKind, SharedBuffer, SharedSemaphore};
use support::{Machine, close_descriptor};

/// Every OpenCL device the installed drivers report.
///
/// Enumerated once for the whole binary: a platform whose GPU is not present
/// spends seconds probing for it before it answers.
fn opencl_devices() -> &'static [Device] {
    static DEVICES: OnceLock<Vec<Device>> = OnceLock::new();
    DEVICES.get_or_init(|| {
        let Ok(runtime) = Runtime::load() else {
            return Vec::new();
        };
        runtime
            .platforms()
            .iter()
            .filter_map(|platform| platform.all_devices().ok())
            .flatten()
            .collect()
    })
}

/// A Vulkan device that can share and the OpenCL device that is the same
/// hardware, or `None` when nothing on this machine pairs.
fn pair() -> Option<(Machine, Device)> {
    for machine in Machine::all(true) {
        // `identity` is `Copy`, so the endpoint borrow ends with this
        // statement and the machine can still move out of the loop.
        let identity = machine.endpoint().identity();
        for device in opencl_devices() {
            if matches!(identity.pairs_with(device), Ok(true)) {
                return Some((machine, device.clone()));
            }
        }
    }
    None
}

/// An allocation exported by Vulkan becomes an OpenCL buffer over the same
/// memory, and the acquire window of [`Share`] is what keeps the two drivers
/// off it at the same time.
///
/// The window is opened twice in a row: only a release in the first `Drop`
/// lets the second acquire succeed, so this is the test that fails if the
/// guard ever stops giving the allocation back.
#[test]
fn a_shared_allocation_is_acquired_and_released_by_the_other_api() {
    let Some((machine, device)) = pair() else {
        return;
    };
    let endpoint = machine.endpoint();
    let Some(kind) = [MemoryKind::DmaBuf, MemoryKind::OpaqueFd]
        .into_iter()
        .find(|kind| {
            endpoint.can_export_memory(*kind) && matches!(kind.importable_by_opencl(&device), Ok(true))
        })
    else {
        return;
    };
    let Some(allocation) = machine.allocate(kind, 4096) else {
        return;
    };
    let Ok(handle) = endpoint.export_memory(allocation.memory(), kind) else {
        panic!("the device advertised {kind:?} as an exportable handle type");
    };
    let Ok(context) = Context::new(&device) else {
        return;
    };
    let Ok(queue) = CommandQueue::new(&context, &device) else {
        return;
    };
    let Ok(shared) = SharedBuffer::import(&context, 4096, handle) else {
        // SAFETY: the import failed, so the descriptor was never read and
        // still belongs to this process alone.
        unsafe { close_descriptor(handle.raw() as i32) };
        return;
    };
    // SAFETY: `clImportMemoryARM` only reads the descriptor and takes its own
    // reference, which is the rule `SharedBuffer::import` documents.
    unsafe { close_descriptor(handle.raw() as i32) };
    assert_eq!(shared.kind(), kind, "the buffer keeps the kind it crossed as");
    let share = match shared.acquire(&queue) {
        Ok(share) => share,
        Err(error) => panic!("a device that imports the handle must acquire it: {error}"),
    };
    assert_eq!(share.buffers().len(), 1, "a window guards the buffer it names");
    assert!(
        core::ptr::eq(share.queue(), &queue),
        "a window releases on the queue that acquired it"
    );
    drop(share);
    // The only way this second window opens is the first one closing: a
    // driver refuses an allocation another queue still owns, so a `Share`
    // that stopped releasing would fail exactly here.
    if let Err(error) = shared.acquire(&queue) {
        panic!("the guard never gave the allocation back: {error}");
    }
}

/// A Vulkan signal crosses into OpenCL as a semaphore of the very kind the
/// export was made for, which is the half of the exchange a queue then waits
/// on or signals.
#[test]
fn a_shared_signal_imports_as_the_opencl_semaphore_it_crosses_as() {
    let Some((machine, device)) = pair() else {
        return;
    };
    let endpoint = machine.endpoint();
    let Ok(context) = Context::new(&device) else {
        return;
    };
    for kind in [SemaphoreKind::OpaqueFd, SemaphoreKind::SyncFd] {
        if !endpoint.can_export_semaphore(kind) || !matches!(kind.importable_by_opencl(&device), Ok(true)) {
            continue;
        }
        let Some(exporter) = machine.semaphore(Some(kind)) else {
            continue;
        };
        // The payload has to be signaled before it is handed over: an
        // unsignaled binary semaphore leaves the driver's destroy waiting on
        // it forever, which is what a real exporter avoids by signaling first.
        if machine.signal(exporter.handle()).is_none() {
            continue;
        }
        let Ok(handle) = endpoint.export_semaphore(exporter.handle(), kind) else {
            panic!("the device advertised {kind:?} as an exportable semaphore handle");
        };
        match SharedSemaphore::import(&context, handle) {
            Ok(shared) => assert_eq!(shared.kind(), kind, "the semaphore keeps the kind it crossed as"),
            Err(error) => panic!("a device that imports the handle refused it: {error}"),
        }
        // The descriptor is left open on purpose: the semaphore side of
        // `cl_khr_external_semaphore` never says the import only reads it,
        // and a test that closes a descriptor the driver kept open closes
        // whatever file has since taken that slot.
    }
}

/// The other direction, and the reason the export carries its kind: a signal
/// made by OpenCL for export is imported by the Vulkan endpoint, which then
/// owns the descriptor it was handed.
#[test]
fn a_signal_made_by_opencl_is_imported_by_the_vulkan_endpoint() {
    let Some((machine, device)) = pair() else {
        return;
    };
    let endpoint = machine.endpoint();
    let Ok(context) = Context::new(&device) else {
        return;
    };
    let Ok(queue) = CommandQueue::new(&context, &device) else {
        return;
    };
    for kind in [SemaphoreKind::OpaqueFd, SemaphoreKind::SyncFd] {
        if !endpoint.can_import_semaphore(kind) {
            continue;
        }
        let Ok(shared) = SharedSemaphore::exportable(&context, kind) else {
            continue;
        };
        // The payload is signaled before it crosses, so the importing
        // semaphore has something to wait on and the driver's destroy of it
        // returns instead of blocking forever.
        if shared.signal(&queue).is_err() || queue.finish().is_err() {
            continue;
        }
        let Ok(handle) = shared.handle() else {
            continue;
        };
        let Some(importer) = machine.semaphore(None) else {
            // SAFETY: nothing took this descriptor, so it is still ours.
            unsafe { close_descriptor(handle.raw() as i32) };
            continue;
        };
        if let Err(error) = endpoint.import_semaphore_fd(importer.handle(), handle) {
            // SAFETY: a refused import leaves the descriptor with the caller.
            unsafe { close_descriptor(handle.raw() as i32) };
            panic!("the device advertised {kind:?} as importable: {error}");
        }
        // On the success path the driver took the descriptor over, which is
        // why there is no close here.
    }
}
