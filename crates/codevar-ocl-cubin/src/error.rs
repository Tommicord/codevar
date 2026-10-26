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

//! Error types shared by the ELF and cubin layers.

use alloc::string::String;
use core::fmt;

/// Every failure this crate can report.
///
/// A single enum is shared by both layers: the cubin reader and writer are
/// built on top of the generic ELF reader and writer, so one error type keeps
/// call sites simple. The aliases [`ElfError`] and [`CubinError`] document
/// intent at the API boundary without duplicating the type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The input ended before `needed` bytes could be read.
    Truncated {
        /// Number of bytes the reader wanted to read.
        needed: usize,
        /// Number of bytes the input actually contained.
        available: usize,
    },
    /// The four-byte ELF magic number `\x7fELF` was absent at offset 0.
    BadMagic,
    /// `e_ident[EI_CLASS]` was not `ELFCLASS64` (2).
    UnsupportedClass(u8),
    /// `e_ident[EI_DATA]` was not `ELFDATA2LSB` (1).
    UnsupportedEndianness(u8),
    /// `e_shentsize` was not 64, which every ELF64 file must use.
    UnsupportedSectionEntrySize(u16),
    /// `e_phentsize` was not 56, which every ELF64 file must use.
    UnsupportedProgramEntrySize(u16),
    /// `e_shnum` was 0, which ELF reserves for extended section numbering.
    UnsupportedSectionCount,
    /// `e_shstrndx` did not index an existing section.
    ShstrndxOutOfRange {
        /// The out-of-range section header string table index.
        index: u16,
        /// The number of section headers present in the file.
        count: u16,
    },
    /// A name offset pointed outside its string table or had no terminator.
    InvalidStringOffset {
        /// The offending offset into the string table.
        offset: u32,
    },
    /// A section's file range fell outside the input.
    SectionDataOutOfBounds {
        /// Index of the section whose `sh_offset`/`sh_size` were invalid.
        index: usize,
    },
    /// A section header `sh_link` did not index a usable section.
    LinkOutOfRange {
        /// Index of the section carrying the out-of-range link.
        section: usize,
        /// The out-of-range link value.
        link: u32,
    },
    /// The symbol table entry size or payload length was invalid.
    InvalidSymbolTable,
    /// A relocation entry was malformed or referenced a missing symbol.
    InvalidRelocation,
    /// A relocation names a symbol that was never declared in the container.
    UnknownRelocationSymbol(String),
    /// `e_machine` did not identify a CUDA binary.
    MachineMismatch {
        /// The machine type found in the file header.
        machine: u16,
    },
    /// Two kernels were registered under the same name.
    DuplicateKernel(String),
    /// A container was requested but contains no kernels.
    NoKernels,
    /// The container has no sections, so it cannot be serialized.
    NoSections,
    /// Sections carry names but there is no `.shstrtab` section to hold them.
    MissingShstrtab,
    /// A program header type other than `PT_PHDR` or `PT_LOAD` cannot be laid
    /// out by the writer.
    UnsupportedSegmentType(u32),
    /// A section or symbol name contains an interior NUL byte.
    InvalidName,
    /// A note entry had inconsistent sizes or no room for its header.
    MalformedNote,
    /// An internal size computation or the writer's output limit was exceeded.
    Overflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { needed, available } => {
                write!(f, "input truncated: need {needed} bytes, have {available}")
            }
            Self::BadMagic => f.write_str("not an ELF file: magic number mismatch"),
            Self::UnsupportedClass(class) => {
                write!(f, "unsupported ELF class {class:#04x}: ELF64 required")
            }
            Self::UnsupportedEndianness(data) => {
                write!(
                    f,
                    "unsupported ELF byte order {data:#04x}: little endian required"
                )
            }
            Self::UnsupportedSectionEntrySize(size) => {
                write!(f, "unsupported e_shentsize {size}: 64 required for ELF64")
            }
            Self::UnsupportedProgramEntrySize(size) => {
                write!(f, "unsupported e_phentsize {size}: 56 required for ELF64")
            }
            Self::UnsupportedSectionCount => {
                f.write_str("e_shnum is 0: extended section numbering is not supported")
            }
            Self::ShstrndxOutOfRange { index, count } => {
                write!(f, "e_shstrndx {index} out of range for {count} sections")
            }
            Self::InvalidStringOffset { offset } => {
                write!(f, "string offset {offset} is not inside its string table")
            }
            Self::SectionDataOutOfBounds { index } => {
                write!(f, "section {index} extends past the end of the file")
            }
            Self::LinkOutOfRange { section, link } => {
                write!(f, "section {section} links to missing section {link}")
            }
            Self::InvalidSymbolTable => f.write_str("symbol table has an invalid entry size"),
            Self::InvalidRelocation => f.write_str("relocation entry is malformed"),
            Self::UnknownRelocationSymbol(name) => {
                write!(f, "relocation references unknown symbol {name:?}")
            }
            Self::MachineMismatch { machine } => {
                write!(f, "e_machine {machine:#06x} is not EM_CUDA (0x00be)")
            }
            Self::DuplicateKernel(name) => write!(f, "kernel {name:?} was added twice"),
            Self::NoKernels => f.write_str("cannot build a container without kernels"),
            Self::NoSections => f.write_str("cannot serialize a container without sections"),
            Self::MissingShstrtab => f.write_str("sections carry names but no .shstrtab section exists"),
            Self::UnsupportedSegmentType(kind) => {
                write!(f, "cannot lay out program header type {kind:#010x}")
            }
            Self::InvalidName => f.write_str("name contains an interior NUL byte"),
            Self::MalformedNote => f.write_str("note entry is malformed"),
            Self::Overflow => f.write_str("size computation overflowed"),
        }
    }
}

/// Error type returned by the functions in [`crate::elf`].
///
/// This is an alias of [`Error`]; it only exists so that API documentation can
/// name the layer that produced the failure.
pub type ElfError = Error;

/// Error type returned by the functions in [`crate::cubin`].
///
/// This is an alias of [`Error`]; it only exists so that API documentation can
/// name the layer that produced the failure.
pub type CubinError = Error;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_describes_each_variant() {
        let cases = [
            Error::Truncated {
                needed: 64,
                available: 3,
            },
            Error::BadMagic,
            Error::UnsupportedClass(1),
            Error::UnsupportedEndianness(2),
            Error::UnsupportedSectionEntrySize(0),
            Error::UnsupportedProgramEntrySize(0),
            Error::UnsupportedSectionCount,
            Error::ShstrndxOutOfRange { index: 9, count: 4 },
            Error::InvalidStringOffset { offset: 7 },
            Error::SectionDataOutOfBounds { index: 2 },
            Error::LinkOutOfRange { section: 3, link: 9 },
            Error::InvalidSymbolTable,
            Error::InvalidRelocation,
            Error::UnknownRelocationSymbol(String::from("missing")),
            Error::MachineMismatch { machine: 62 },
            Error::DuplicateKernel(String::from("k")),
            Error::NoKernels,
            Error::NoSections,
            Error::MissingShstrtab,
            Error::UnsupportedSegmentType(0x6474_e550),
            Error::InvalidName,
            Error::MalformedNote,
            Error::Overflow,
        ];
        for case in cases {
            assert!(!format!("{case}").is_empty());
        }
    }

    #[test]
    fn aliases_match_the_shared_error() {
        let error: ElfError = Error::BadMagic;
        let cubin: CubinError = error.clone();
        assert_eq!(cubin, Error::BadMagic);
    }
}
