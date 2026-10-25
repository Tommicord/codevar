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

//! The static kernel registry — the embedded profile's answer to the
//! online compiler.
//!
//! A hosted OpenCL stack compiles `__kernel` source at runtime; an
//! OS-less device links its kernels into the image instead. Applications
//! describe the kernels they built in one or more [`KernelModule`]s
//! (typically one per compilation unit of the original `.cl` sources) and
//! hand the list to [`crate::init`]. From then on:
//!
//! * `clCreateProgramWithSource` reports [`sys::COMPILER_NOT_AVAILABLE`] —
//!   there is no compiler in the image, per the embedded profile;
//! * `clCreateProgramWithBuiltInKernels` / `clCreateProgramWithBinary`
//!   resolve names against this registry;
//! * `clGetDeviceInfo(CL_DEVICE_BUILTIN_KERNELS)` and
//!   `clGetProgramInfo(CL_PROGRAM_KERNEL_NAMES)` render it as the
//!   `;`-separated string the C API uses.
//!
//! The registry is read-only after [`crate::init`], so lookups need no
//! locking and cost a linear scan over a handful of static entries.

use alloc::borrow::Cow;
use alloc::string::String;

use crate::exec::{KernelFn, SpirvBinding};

/// How one declared argument of a kernel is bound by `clSetKernelArg`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgKind {
    /// A buffer argument: `arg_value` points at a live `cl_mem` handle and
    /// `arg_size` must be the size of that handle.
    Buffer,
    /// A scalar argument of exactly `usize` bytes (the host copies it into
    /// the launch's argument block).
    Scalar(usize),
}

/// One kernel exported by a [`KernelModule`] — or, for a program created
/// from a SPIR-V binary, one entry point derived from the module at
/// creation time.
///
/// Both flavours are the same type so `clCreateKernel` and every query
/// treat them identically: static registry entries borrow their name and
/// kinds (`Cow::Borrowed`, constructible in a `static`), SPIR-V entries
/// own them (`Cow::Owned`), cloned once into each kernel handle.
#[derive(Debug, Clone)]
pub struct KernelDef {
    /// The OpenCL kernel name — what `clCreateKernel` and the
    /// `CL_DEVICE_BUILTIN_KERNELS` list use, without any `__kernel` prefix.
    pub name: Cow<'static, str>,
    /// The entry point the claim loop calls (never invoked when `binding`
    /// is set — the SPIR-V interpreter runs the module's entry instead).
    pub func: KernelFn,
    /// Declared arguments, in order; the length is `CL_KERNEL_NUM_ARGS`.
    pub kinds: Cow<'static, [ArgKind]>,
    /// SPIR-V entry point for a program-backed kernel, or
    /// [`SpirvBinding::NONE`] for a statically registered one.
    pub binding: SpirvBinding,
}

impl KernelDef {
    /// Number of arguments (`CL_KERNEL_NUM_ARGS`).
    #[must_use]
    #[inline]
    pub fn num_args(&self) -> u32 {
        self.kinds.len() as u32
    }

    /// The `clSetKernelArg` size this argument demands.
    ///
    /// Buffers take the handle size; scalars take their declared size. A
    /// mismatching `arg_size` is reported as `CL_INVALID_ARG_SIZE`.
    #[must_use]
    #[inline]
    pub fn arg_size(&self, index: usize) -> Option<usize> {
        match self.kinds.get(index) {
            Some(ArgKind::Buffer) => Some(core::mem::size_of::<*mut core::ffi::c_void>()),
            Some(ArgKind::Scalar(size)) => Some(*size),
            None => None,
        }
    }
}

/// A group of kernels registered under one module name — normally one per
/// original OpenCL source file (`wredit/block.rs` → `"wredit/block"`).
#[derive(Debug)]
pub struct KernelModule {
    /// Module identifier; informational, shown in diagnostics.
    pub name: &'static str,
    /// The kernels of this module.
    pub kernels: &'static [KernelDef],
}

/// The registry view the platform exposes: every module of the image.
#[derive(Debug)]
pub struct Registry {
    modules: &'static [&'static KernelModule],
}

impl Registry {
    /// Builds a registry over the image's modules.
    #[must_use]
    pub const fn new(modules: &'static [&'static KernelModule]) -> Self {
        Self { modules }
    }

    /// The registered modules.
    #[must_use]
    pub const fn modules(&self) -> &'static [&'static KernelModule] {
        self.modules
    }

    /// Resolves `name` to its definition (first match across modules wins;
    /// duplicate names across modules are a link-time configuration error
    /// the application avoids by construction).
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&'static KernelDef> {
        self.modules
            .iter()
            .flat_map(|module| module.kernels.iter())
            .find(|def| def.name == name)
    }

    /// Renders every registered kernel name as the `;`-separated string
    /// the OpenCL queries use, with a trailing NUL added by the caller's
    /// string writer.
    #[must_use]
    pub fn names_joined(&self) -> String {
        let mut out = String::new();
        for module in self.modules {
            for def in module.kernels {
                if !out.is_empty() {
                    out.push(';');
                }
                out.push_str(&def.name);
            }
        }
        out
    }

    /// Number of registered kernels (`CL_PROGRAM_NUM_KERNELS`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.modules.iter().map(|m| m.kernels.len()).sum()
    }

    /// Whether the registry holds no kernels at all.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::vector_add;

    const KERNEL: KernelDef = KernelDef {
        name: Cow::Borrowed("vector_add"),
        func: vector_add,
        kinds: Cow::Borrowed(&[
            ArgKind::Buffer,
            ArgKind::Buffer,
            ArgKind::Buffer,
            ArgKind::Scalar(4),
        ]),
        binding: SpirvBinding::NONE,
    };
    static MODULE: KernelModule = KernelModule {
        name: "demo",
        kernels: &[KERNEL],
    };
    static MODULES: &[&KernelModule] = &[&MODULE];

    #[test]
    fn find_resolves_registered_names_only() {
        let registry = Registry::new(MODULES);
        let def = registry.find("vector_add").expect("registered");
        assert_eq!(def.num_args(), 4);
        assert_eq!(def.arg_size(3), Some(4));
        assert_eq!(def.arg_size(4), None);
        assert!(registry.find("missing").is_none());
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.names_joined(), "vector_add");
    }
}
