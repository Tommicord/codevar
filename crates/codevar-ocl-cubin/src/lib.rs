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

//! ELF container reader and writer for CUDA cubin binaries.
//!
//! A cubin is an [ELF](https://refspecs.linuxbase.org/elf/gabi4/)-formatted
//! file describing a compiled kernel for one NVIDIA GPU architecture, as
//! produced by `nvcc` (see [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)).
//! This crate parses and writes that container only; the SASS payload that
//! fills a `.text.<kernel>` section is supplied by the caller.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod cubin;
pub mod elf;
pub mod error;

pub use cubin::{
    Cubin, CubinBuilder, KernelImage, KernelSection, NT_CUDA_CUINFO, NT_CUDA_TKINFO, Note, Reloc,
    SHT_CUDA_INFO, cuda_sm_flags, encode_notes, parse as parse_cubin, parse_notes, sm_from_flags,
};
pub use elf::{Elf, Header, Section, Segment, Symbol, parse as parse_elf};
pub use error::{CubinError, ElfError, Error};
