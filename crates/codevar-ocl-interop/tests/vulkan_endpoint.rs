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

//! The Vulkan endpoint against a driver that is really installed.
//!
//! Each test here asks the loader for a device and returns without asserting
//! when there is none, so a headless runner sees a passing suite; on a
//! machine with a GPU it exercises the three entry points the crate resolves
//! by name, which no unit test can reach.

#![cfg(unix)]

mod support;

use ash::vk;
use codevar_ocl_interop::{Error, MemoryKind, SemaphoreHandle, SemaphoreKind};
use support::{Machine, close_descriptor};

/// `Vulkan::new` resolves the entry points and keeps going when they are
/// absent, because the capability questions and the pairing question have to
/// be answerable on a device that enabled neither extension. A device with
/// neither therefore has to report each share as unsupported — a null
/// function pointer called anyway is the failure this guards against, which
/// is why the crate resolves them by name instead of through a loader that
/// fills the gap with a panic.
#[test]
fn a_device_without_the_extensions_refuses_every_share() {
    let Some(machine) = Machine::first(false) else {
        return;
    };
    let endpoint = machine.endpoint();
    // A null handle is never read: the entry point is looked up first, and
    // a device that enabled nothing answers before the driver is reached.
    let memory = vk::DeviceMemory::default();
    let semaphore = vk::Semaphore::default();
    assert!(
        matches!(
            endpoint.export_memory(memory, MemoryKind::DmaBuf),
            Err(Error::Unsupported { .. })
        ),
        "a device without VK_KHR_external_memory_fd must say so"
    );
    assert!(
        matches!(
            endpoint.export_semaphore(semaphore, SemaphoreKind::OpaqueFd),
            Err(Error::Unsupported { .. })
        ),
        "a device without VK_KHR_external_semaphore_fd must say so"
    );
    assert!(
        matches!(
            endpoint.import_semaphore_fd(semaphore, SemaphoreHandle::OpaqueFd(-1)),
            Err(Error::Unsupported { .. })
        ),
        "an import on the same device must report the same missing extension"
    );
    // The identity query is core Vulkan 1.1, so it answers either way, and a
    // LUID is a Windows concept the loader on this machine never reports.
    assert_eq!(endpoint.identity().luid, None);
}

/// An allocation the device advertises as exportable comes back as a
/// descriptor of the very kind it was made for. The pairing between
/// `VkExportMemoryAllocateInfo` and `vkGetMemoryFdKHR` is what makes that
/// descriptor mean anything to the other API, so a wrong kind here is a
/// wrong handle there.
///
/// A driver that reports no exportable handle type for a storage buffer has
/// nothing to check and the loop body never runs.
#[test]
fn an_exportable_allocation_comes_back_as_the_kind_it_asked_for() {
    let Some(machine) = Machine::first(true) else {
        return;
    };
    let endpoint = machine.endpoint();
    for kind in [MemoryKind::DmaBuf, MemoryKind::OpaqueFd] {
        if !endpoint.can_export_memory(kind) {
            continue;
        }
        let Some(allocation) = machine.allocate(kind, 4096) else {
            continue;
        };
        match endpoint.export_memory(allocation.memory(), kind) {
            Ok(handle) => {
                assert_eq!(
                    handle.kind(),
                    kind,
                    "the descriptor must name the kind the allocation was made for"
                );
                assert!(handle.raw() >= 0, "a negative descriptor is not a file");
                // SAFETY: `vkGetMemoryFdKHR` made this descriptor for this
                // process and nothing else holds it.
                unsafe { close_descriptor(handle.raw() as i32) };
            }
            Err(error) => panic!("the device said {kind:?} exports: {error}"),
        }
    }
}

/// A semaphore exported as a descriptor is imported by a second semaphore of
/// the same device, which is the whole of what the OpenCL side is handed: the
/// export carries the kind, and the import refuses a kind the device cannot
/// take rather than mis-signaling.
///
/// The exporter is signaled first, as a real share's is: the specification
/// only lets `vkGetSemaphoreFdKHR` hand out a `sync_file` for a payload that
/// is already signaled, and a driver's `vkDestroySemaphore` waits for the
/// payload a semaphore still holds. An unsignaled one would never come back.
#[test]
fn a_semaphore_exported_by_the_endpoint_is_imported_by_another_one() {
    let Some(machine) = Machine::first(true) else {
        return;
    };
    let endpoint = machine.endpoint();
    for kind in [SemaphoreKind::OpaqueFd, SemaphoreKind::SyncFd] {
        if !endpoint.can_export_semaphore(kind) || !endpoint.can_import_semaphore(kind) {
            continue;
        }
        let Some(exporter) = machine.semaphore(Some(kind)) else {
            continue;
        };
        if machine.signal(exporter.handle()).is_none() {
            continue;
        }
        let Some(handle) = endpoint
            .export_semaphore(exporter.handle(), kind)
            .ok()
        else {
            panic!("the device said {kind:?} exports a semaphore");
        };
        assert_eq!(handle.kind(), kind, "the descriptor must name its own kind");
        assert!(handle.raw() >= 0, "a negative descriptor is not a file");
        let Some(importer) = machine.semaphore(None) else {
            // SAFETY: as above, the export made this descriptor for us alone.
            unsafe { close_descriptor(handle.raw() as i32) };
            continue;
        };
        match endpoint.import_semaphore_fd(importer.handle(), handle) {
            // Nothing to close: a successful import transfers the descriptor
            // to the driver, which is the one rule here that a `Drop` cannot
            // express for the caller.
            Ok(()) => {}
            Err(error) => {
                // SAFETY: the import failed, so the descriptor is still ours.
                unsafe { close_descriptor(handle.raw() as i32) };
                panic!("a handle the device advertised as importable failed: {error}");
            }
        }
    }
}
