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

//! CUDA cubin container built on the ELF layer.
//!
//! A cubin is an [`EM_CUDA`](crate::elf::EM_CUDA) ELF64 image holding one or
//! more compiled kernels. Each kernel lives in a `.text.<name>` section
//! (optionally accompanied by `.rela.text.<name>` relocations and a
//! `.nv.constant0.<name>` constant bank), and each kernel exports one global
//! function symbol. This module provides:
//!
//! - [`parse`], which turns bytes into a [`Cubin`],
//! - [`CubinBuilder`], which assembles bytes from [`KernelImage`] payloads,
//! - [`cuda_sm_flags`] / [`sm_from_flags`], which encode the `e_flags` field,
//! - [`parse_notes`] / [`encode_notes`], which read and write `.note.*`
//!   sections such as `.note.nv.cuver`.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::Error;
use crate::elf::{
    EHDR_SIZE, ELFCLASS64, ELFDATA2LSB, EM_CUDA, ET_EXEC, ET_REL, Elf, Header, PF_R, PF_X, PHDR_SIZE,
    PT_LOAD, PT_PHDR, RELA_SIZE, SHDR_SIZE, SHF_ALLOC, SHF_EXECINSTR, SHF_INFO_LINK, SHN_UNDEF, SHT_NOTE,
    SHT_NULL, SHT_PROGBITS, SHT_RELA, SHT_STRTAB, SHT_SYMTAB, STB_GLOBAL, STT_FUNC, SYM_SIZE, Section,
    Segment, Symbol, le_u64, st_info,
};

/// Section type of NVIDIA's `.nv.info` sections (`SHT_LOPROC` range).
pub const SHT_CUDA_INFO: u32 = 0x7000_0000;

/// `n_type` of the `.note.nv.cuver` (CUDA 12) / `.note.nv.cuinfo` (CUDA 13)
/// note describing the toolchain and target architecture.
pub const NT_CUDA_CUINFO: u32 = 0x03E8;

/// `n_type` of the `.note.nv.tkinfo` note describing the compiling tool.
pub const NT_CUDA_TKINFO: u32 = 0x07D0;

/// High byte NVIDIA writes into the `sh_info` field of `.text` sections, as
/// verified against `ptxas` output; the low bits hold the kernel symbol index.
const TEXT_INFO_HIGH: u32 = 0x06 << 24;

/// Encodes a compute capability into the `e_flags` layout used by `ptxas`.
///
/// Two verified layouts exist. Classic images (SM versions below 100) store
/// the version in both the low byte and bits 16-23 with control byte 0x05 in
/// bits 8-15, for example `0x004b054b` for SM 7.5. Extended images (SM 100 and
/// above) store it in bits 8-15 with control byte 0x06 in bits 24-31, for
/// example `0x06006402` for SM 10.0. The architecture-variant suffix (`90a`,
/// `100a`, ...) cannot be represented and is not encoded. Never panics.
pub fn cuda_sm_flags(sm: u16) -> u32 {
    if sm >= 100 {
        0x02 | (u32::from(sm) << 8) | (0x06 << 24)
    } else {
        u32::from(sm) | (0x05 << 8) | (u32::from(sm) << 16)
    }
}

/// Recovers the compute capability from an `e_flags` word written by
/// [`cuda_sm_flags`].
///
/// The extended layout is detected through its non-zero control byte in bits
/// 24-31; everything else is read as a classic layout. Architecture-variant
/// suffixes are lost, as they are not representable in the flag word.
pub fn sm_from_flags(flags: u32) -> u16 {
    if flags >> 24 != 0 {
        ((flags >> 8) & 0xFF) as u16
    } else {
        (flags & 0xFF) as u16
    }
}

/// One relocation to apply to a kernel section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reloc {
    /// Byte offset inside the section that the relocation applies to.
    pub offset: u64,
    /// Relocation type from the low 32 bits of `r_info`, passed through
    /// uninterpreted because NVIDIA never published the type names.
    pub kind: u32,
    /// Name of the symbol the relocation refers to.
    pub symbol: String,
    /// Sign-extended `r_addend` value.
    pub addend: i64,
}

/// A kernel to place into a new container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelImage {
    /// Kernel name; the section becomes `.text.<name>` and the exported
    /// symbol takes the same name.
    pub name: String,
    /// SASS machine code bytes for the kernel.
    pub code: Vec<u8>,
    /// Raw contents of the `.nv.constant0.<name>` section; skipped entirely
    /// when empty.
    pub constant_data: Vec<u8>,
    /// Relocations for the `.rela.text.<name>` section; skipped when empty.
    /// Every `symbol` must name a kernel in the same container.
    pub relocs: Vec<Reloc>,
}

/// A kernel as recovered from a parsed container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSection {
    /// Kernel name, i.e. the suffix of `.text.<name>`.
    pub name: String,
    /// SASS machine code bytes, i.e. the `.text.<name>` payload.
    pub code: Vec<u8>,
    /// Contents of the matching `.nv.constant0.<name>` section, empty when
    /// the section is absent.
    pub constant_data: Vec<u8>,
    /// Decoded `.rela.text.<name>` relocations, empty when absent.
    pub relocs: Vec<Reloc>,
}

/// One `ELF64_Nhdr`-style note entry with a NUL-terminated name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    /// Note owner name, for example `"NVIDIA Corp"`.
    pub name: String,
    /// Note type, such as [`NT_CUDA_CUINFO`] or [`NT_CUDA_TKINFO`].
    pub n_type: u32,
    /// Raw note descriptor bytes.
    pub desc: Vec<u8>,
}

impl Note {
    /// Serializes the note, including 4-byte padding of name and descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidName`] when the name contains an interior NUL
    /// byte and [`Error::Overflow`] when the sizes do not fit `u32`.
    pub fn encode(&self) -> Result<Vec<u8>, Error> {
        if self.name.as_bytes().contains(&0u8) {
            return Err(Error::InvalidName);
        }
        let name_size = u32::try_from(self.name.len())
            .map_err(|_| Error::Overflow)?
            .checked_add(1)
            .ok_or(Error::Overflow)?;
        let desc_size = u32::try_from(self.desc.len()).map_err(|_| Error::Overflow)?;
        let name_span = pad4(name_size)?;
        let desc_span = pad4(desc_size)?;
        let total = 12_u32
            .checked_add(name_span)
            .and_then(|total| total.checked_add(desc_span))
            .ok_or(Error::Overflow)?;
        let capacity = usize::try_from(total).map_err(|_| Error::Overflow)?;
        let name_end = 12_usize
            .checked_add(usize::try_from(name_span).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(&name_size.to_le_bytes());
        out.extend_from_slice(&desc_size.to_le_bytes());
        out.extend_from_slice(&self.n_type.to_le_bytes());
        out.extend_from_slice(self.name.as_bytes());
        out.push(0u8);
        out.resize(name_end, 0u8);
        out.extend_from_slice(&self.desc);
        out.resize(capacity, 0u8);
        Ok(out)
    }
}

/// Decodes a stream of padded note entries.
///
/// # Errors
///
/// Returns [`Error::MalformedNote`] when the stream ends inside a header,
/// name, or descriptor, or when trailing bytes cannot form another entry.
/// An empty input yields an empty vector.
pub fn parse_notes(bytes: &[u8]) -> Result<Vec<Note>, Error> {
    let mut notes = Vec::new();
    let mut offset = 0_usize;
    while offset < bytes.len() {
        let header_end = offset
            .checked_add(12)
            .ok_or(Error::MalformedNote)?;
        let header = bytes
            .get(offset..header_end)
            .ok_or(Error::MalformedNote)?;
        let name_size = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let desc_size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let n_type = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        let name_size = usize::try_from(name_size).map_err(|_| Error::MalformedNote)?;
        let desc_size = usize::try_from(desc_size).map_err(|_| Error::MalformedNote)?;
        let name_span = name_size
            .checked_add(3)
            .ok_or(Error::MalformedNote)?
            & !3_usize;
        let desc_span = desc_size
            .checked_add(3)
            .ok_or(Error::MalformedNote)?
            & !3_usize;
        let name_stop = header_end
            .checked_add(name_size)
            .ok_or(Error::MalformedNote)?;
        let desc_start = header_end
            .checked_add(name_span)
            .ok_or(Error::MalformedNote)?;
        let desc_end = desc_start
            .checked_add(desc_span)
            .ok_or(Error::MalformedNote)?;
        if desc_end > bytes.len() {
            return Err(Error::MalformedNote);
        }
        let raw_name = bytes
            .get(header_end..name_stop)
            .ok_or(Error::MalformedNote)?;
        let raw_name = raw_name.strip_suffix(&[0u8]).unwrap_or(raw_name);
        let desc = bytes
            .get(desc_start..desc_end)
            .ok_or(Error::MalformedNote)?;
        notes.push(Note {
            name: String::from_utf8_lossy(raw_name).into_owned(),
            n_type,
            desc: desc.to_vec(),
        });
        offset = desc_end;
    }
    Ok(notes)
}

/// Serializes a sequence of notes into one concatenated byte stream.
///
/// # Errors
///
/// Propagates every [`Note::encode`] error.
pub fn encode_notes(notes: &[Note]) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    for note in notes {
        out.extend_from_slice(&note.encode()?);
    }
    Ok(out)
}

fn pad4(value: u32) -> Result<u32, Error> {
    value
        .checked_add(3)
        .ok_or(Error::Overflow)
        .map(|padded| padded & !3)
}

/// A parsed CUDA cubin container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cubin {
    /// Compute capability decoded from `e_flags`, for example 75 for SM 7.5.
    pub sm: u16,
    /// Kernels found in `.text.<name>` sections, in section order.
    pub kernels: Vec<KernelSection>,
    /// The full ELF container, including every section not lifted above.
    pub elf: Elf,
}

impl Cubin {
    /// Returns the kernel with the given `name`, if present.
    pub fn kernel(&self, name: &str) -> Option<&KernelSection> {
        self.kernels
            .iter()
            .find(|kernel| kernel.name == name)
    }

    /// Decodes every `SHT_NOTE` section in the container in section order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MalformedNote`] when a note section is inconsistent.
    pub fn notes(&self) -> Result<Vec<Note>, Error> {
        let mut notes = Vec::new();
        for section in &self.elf.sections {
            if section.sh_type == SHT_NOTE {
                notes.extend(parse_notes(&section.data)?);
            }
        }
        Ok(notes)
    }
}

/// Assembles a cubin image from kernel payloads.
///
/// The builder emits the same section set `ptxas` produces for a linked image:
/// `.shstrtab`, `.strtab`, `.symtab`, `.nv.info`, and per kernel a
/// `.text.<name>` section (with `.rela.text.<name>` and
/// `.nv.constant0.<name>` when supplied), plus `PT_PHDR` and `PT_LOAD`
/// program headers. Relocatable output (`.linked(false)`) matches
/// `ptxas -c` and carries no program headers. Note sections are never
/// synthesized; callers that need them can post-process the bytes.
///
/// # Examples
///
/// ```
/// use codevar_ocl_cubin::{CubinBuilder, Error, KernelImage};
///
/// # fn main() -> Result<(), Error> {
/// let mut builder = CubinBuilder::new(75);
/// builder.add_kernel(KernelImage {
///     name: String::from("vector_add"),
///     code: vec![0x55, 0xAA],
///     constant_data: Vec::new(),
///     relocs: Vec::new(),
/// })?;
/// let image = builder.build()?;
/// let cubin = codevar_ocl_cubin::parse_cubin(&image)?;
/// assert_eq!(cubin.sm, 75);
/// assert_eq!(cubin.kernels.len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CubinBuilder {
    sm: u16,
    linked: bool,
    version: u32,
    kernels: Vec<KernelImage>,
}

impl CubinBuilder {
    /// Creates a builder for the given compute capability, for example 75 for
    /// SM 7.5 or 100 for SM 10.0.
    ///
    /// Linked output and a toolchain version matching the architecture are the
    /// defaults: version 129 (`12.9`) for classic SM versions, version 1 for
    /// extended ones.
    pub fn new(sm: u16) -> Self {
        Self {
            sm,
            linked: true,
            version: if sm >= 100 { 1 } else { 129 },
            kernels: Vec::new(),
        }
    }

    /// Selects linked (`ET_EXEC` with program headers) or relocatable
    /// (`ET_REL` without) output; linked is the default.
    #[must_use = "fluent setters return the modified builder"]
    pub fn linked(mut self, linked: bool) -> Self {
        self.linked = linked;
        self
    }

    /// Overrides the `e_version` field; the default is described in
    /// [`CubinBuilder::new`]. Versions of 130 or higher also switch
    /// `e_ident` to the CUDA 13 ABI bytes for every architecture.
    #[must_use = "fluent setters return the modified builder"]
    pub fn toolkit_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Registers a kernel for inclusion in the image.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidName`] when the name is empty or contains a NUL
    /// byte, and [`Error::DuplicateKernel`] when a kernel with the same name
    /// was already registered.
    pub fn add_kernel(&mut self, image: KernelImage) -> Result<(), Error> {
        if image.name.is_empty() || image.name.as_bytes().contains(&0u8) {
            return Err(Error::InvalidName);
        }
        if self
            .kernels
            .iter()
            .any(|existing| existing.name == image.name)
        {
            return Err(Error::DuplicateKernel(image.name));
        }
        self.kernels.push(image);
        Ok(())
    }

    /// Serializes the container.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoKernels`] when no kernel was registered,
    /// [`Error::UnknownRelocationSymbol`] when a relocation names something
    /// other than a registered kernel, [`Error::InvalidName`] for names with
    /// interior NUL bytes, and [`Error::Overflow`] when the container would
    /// not fit the ELF64 header limits or the writer's size cap.
    pub fn build(self) -> Result<Vec<u8>, Error> {
        let Self {
            sm,
            linked,
            version,
            kernels,
        } = self;
        if kernels.is_empty() {
            return Err(Error::NoKernels);
        }
        let max_kernels = (usize::from(u16::MAX) - 5) / 3;
        if kernels.len() > max_kernels {
            return Err(Error::Overflow);
        }

        let extended_ident = version >= 130 || sm >= 100;
        let mut ident = [0u8; 16];
        ident[0] = 0x7F;
        ident[1] = b'E';
        ident[2] = b'L';
        ident[3] = b'F';
        ident[4] = ELFCLASS64;
        ident[5] = ELFDATA2LSB;
        ident[6] = 1;
        ident[7] = if extended_ident { 0x41 } else { 0x33 };
        ident[8] = if extended_ident { 0x08 } else { 0x07 };

        let header = Header {
            ident,
            e_type: if linked { ET_EXEC } else { ET_REL },
            machine: EM_CUDA,
            version,
            entry: 0,
            phoff: 0,
            shoff: 0,
            flags: cuda_sm_flags(sm),
            ehsize: EHDR_SIZE as u16,
            phentsize: PHDR_SIZE as u16,
            phnum: 0,
            shentsize: SHDR_SIZE as u16,
            shnum: 0,
            shstrndx: 0,
        };

        let names: Vec<String> = kernels
            .iter()
            .map(|image| image.name.clone())
            .collect();

        let mut sections: Vec<Section> = vec![
            Section {
                name: String::new(),
                sh_type: SHT_NULL,
                flags: 0,
                addr: 0,
                align: 0,
                link: 0,
                info: 0,
                entsize: 0,
                data: Vec::new(),
                bss_size: 0,
            },
            unowned_string_table(".shstrtab"),
            unowned_string_table(".strtab"),
            Section {
                name: String::from(".symtab"),
                sh_type: SHT_SYMTAB,
                flags: 0,
                addr: 0,
                align: 8,
                link: 2,
                info: 1,
                entsize: SYM_SIZE as u64,
                data: Vec::new(),
                bss_size: 0,
            },
            Section {
                name: String::from(".nv.info"),
                sh_type: SHT_CUDA_INFO,
                flags: 0,
                addr: 0,
                align: 4,
                link: 3,
                info: 0,
                entsize: 0,
                data: Vec::new(),
                bss_size: 0,
            },
        ];

        let mut symbols: Vec<Symbol> = vec![Symbol {
            name: String::new(),
            info: 0,
            other: 0,
            shndx: SHN_UNDEF,
            value: 0,
            size: 0,
        }];

        let mut segments: Vec<Segment> = Vec::new();
        if linked {
            segments.push(segment_header(PT_PHDR, PF_R | PF_X));
            segments.push(segment_header(PT_LOAD, PF_R | PF_X));
        }

        for image in kernels {
            let code_size = u64::try_from(image.code.len()).map_err(|_| Error::Overflow)?;
            let text_index = u32::try_from(sections.len()).map_err(|_| Error::Overflow)?;
            let symbol_index = u32::try_from(symbols.len()).map_err(|_| Error::Overflow)?;
            sections.push(Section {
                name: format!(".text.{}", image.name),
                sh_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_EXECINSTR,
                addr: 0,
                align: 128,
                link: 3,
                info: TEXT_INFO_HIGH | symbol_index,
                entsize: 0,
                data: image.code,
                bss_size: 0,
            });
            if !image.relocs.is_empty() {
                let mut data = Vec::with_capacity(image.relocs.len() * RELA_SIZE);
                for reloc in &image.relocs {
                    let target = names
                        .iter()
                        .position(|name| *name == reloc.symbol)
                        .ok_or_else(|| Error::UnknownRelocationSymbol(reloc.symbol.clone()))?;
                    let symbol = u32::try_from(target).map_err(|_| Error::Overflow)?;
                    let symbol = symbol.checked_add(1).ok_or(Error::Overflow)?;
                    let info = (u64::from(symbol) << 32) | u64::from(reloc.kind);
                    data.extend_from_slice(&reloc.offset.to_le_bytes());
                    data.extend_from_slice(&info.to_le_bytes());
                    data.extend_from_slice(&reloc.addend.to_le_bytes());
                }
                sections.push(Section {
                    name: format!(".rela.text.{}", image.name),
                    sh_type: SHT_RELA,
                    flags: SHF_INFO_LINK,
                    addr: 0,
                    align: 8,
                    link: 3,
                    info: text_index,
                    entsize: RELA_SIZE as u64,
                    data,
                    bss_size: 0,
                });
            }
            if !image.constant_data.is_empty() {
                sections.push(Section {
                    name: format!(".nv.constant0.{}", image.name),
                    sh_type: SHT_PROGBITS,
                    flags: SHF_ALLOC | SHF_INFO_LINK,
                    addr: 0,
                    align: 4,
                    link: 0,
                    info: text_index,
                    entsize: 0,
                    data: image.constant_data,
                    bss_size: 0,
                });
            }
            symbols.push(Symbol {
                name: image.name,
                info: st_info(STB_GLOBAL, STT_FUNC),
                other: 0,
                shndx: u16::try_from(text_index).map_err(|_| Error::Overflow)?,
                value: 0,
                size: code_size,
            });
        }

        Elf {
            header,
            sections,
            segments,
            symbols,
        }
        .to_bytes()
    }
}

fn unowned_string_table(name: &str) -> Section {
    Section {
        name: String::from(name),
        sh_type: SHT_STRTAB,
        flags: 0,
        addr: 0,
        align: 1,
        link: 0,
        info: 0,
        entsize: 0,
        data: Vec::new(),
        bss_size: 0,
    }
}

fn segment_header(p_type: u32, p_flags: u32) -> Segment {
    Segment {
        p_type,
        p_flags,
        offset: 0,
        vaddr: 0,
        paddr: 0,
        filesz: 0,
        memsz: 0,
        align: 8,
    }
}

/// Parses a CUDA cubin container.
///
/// The bytes must be an ELF64 little-endian file with
/// [`e_machine = EM_CUDA`](crate::elf::EM_CUDA). Every `.text.<name>` section
/// with a non-empty name becomes a [`KernelSection`], joined by the matching
/// `.nv.constant0.<name>` and `.rela.text.<name>` sections when present.
/// Sections without a kernel counterpart stay available through
/// [`Cubin::elf`].
///
/// # Errors
///
/// Returns every error of `elf::parse`, plus [`Error::MachineMismatch`] when
/// the file is a valid ELF image for a different architecture and
/// [`Error::InvalidRelocation`] when a relocation table is malformed or names
/// a missing symbol.
pub fn parse(bytes: &[u8]) -> Result<Cubin, Error> {
    let elf = crate::elf::parse(bytes)?;
    if elf.header.machine != EM_CUDA {
        return Err(Error::MachineMismatch {
            machine: elf.header.machine,
        });
    }
    let sm = sm_from_flags(elf.header.flags);
    let mut kernels = Vec::new();
    for section in &elf.sections {
        let Some(suffix) = section.name.strip_prefix(".text.") else {
            continue;
        };
        if suffix.is_empty() {
            continue;
        }
        let mut constant: Option<&[u8]> = None;
        let mut relocations: Option<&[u8]> = None;
        for candidate in &elf.sections {
            if let Some(rest) = candidate.name.strip_prefix(".nv.constant0.")
                && rest == suffix
            {
                constant = Some(candidate.data.as_slice());
            }
            if let Some(rest) = candidate.name.strip_prefix(".rela.text.")
                && rest == suffix
            {
                relocations = Some(candidate.data.as_slice());
            }
        }
        let relocs = match relocations {
            Some(data) => decode_relocs(data, &elf.symbols)?,
            None => Vec::new(),
        };
        kernels.push(KernelSection {
            name: String::from(suffix),
            code: section.data.clone(),
            constant_data: constant.map(<[u8]>::to_vec).unwrap_or_default(),
            relocs,
        });
    }
    Ok(Cubin { sm, kernels, elf })
}

fn decode_relocs(data: &[u8], symbols: &[Symbol]) -> Result<Vec<Reloc>, Error> {
    if !data.len().is_multiple_of(RELA_SIZE) {
        return Err(Error::InvalidRelocation);
    }
    let mut relocs = Vec::with_capacity(data.len() / RELA_SIZE);
    for entry in data.chunks_exact(RELA_SIZE) {
        let offset = le_u64(entry, 0)?;
        let info = le_u64(entry, 8)?;
        let addend = le_u64(entry, 16)? as i64;
        let kind = (info & 0xFFFF_FFFF) as u32;
        let index = usize::try_from(info >> 32).map_err(|_| Error::InvalidRelocation)?;
        let symbol = symbols
            .get(index)
            .map(|symbol| symbol.name.clone())
            .ok_or(Error::InvalidRelocation)?;
        relocs.push(Reloc {
            offset,
            kind,
            symbol,
            addend,
        });
    }
    Ok(relocs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::{EM_X86_64, st_binding, st_type};

    fn build_with(mut builder: CubinBuilder) -> Vec<u8> {
        builder
            .add_kernel(KernelImage {
                name: String::from("k"),
                code: vec![1],
                constant_data: Vec::new(),
                relocs: Vec::new(),
            })
            .unwrap();
        builder.build().unwrap()
    }

    fn section_position(cubin: &Cubin, name: &str) -> usize {
        cubin
            .elf
            .sections
            .iter()
            .position(|section| section.name == name)
            .unwrap()
    }

    #[test]
    fn builds_and_parses_a_linked_cubin() {
        let mut builder = CubinBuilder::new(75);
        builder
            .add_kernel(KernelImage {
                name: String::from("vector_add"),
                code: vec![0x55, 0xAA, 0x01],
                constant_data: vec![0xDE, 0xAD],
                relocs: Vec::new(),
            })
            .unwrap();
        let bytes = builder.build().unwrap();
        let cubin = parse(&bytes).unwrap();

        assert_eq!(cubin.sm, 75);
        assert_eq!(cubin.elf.header.machine, EM_CUDA);
        assert_eq!(cubin.elf.header.e_type, ET_EXEC);
        assert_eq!(cubin.elf.header.version, 129);
        assert_eq!(cubin.elf.header.flags, 0x004B_054B);
        assert_eq!(cubin.elf.header.ident[7], 0x33);
        assert_eq!(cubin.elf.header.ident[8], 0x07);
        assert_eq!(cubin.elf.header.phnum, 2);
        assert_eq!(cubin.elf.header.ehsize, EHDR_SIZE as u16);

        assert_eq!(cubin.kernels.len(), 1);
        let kernel = cubin.kernel("vector_add").unwrap();
        assert_eq!(kernel.code, [0x55, 0xAA, 0x01]);
        assert_eq!(kernel.constant_data, [0xDE, 0xAD]);
        assert!(kernel.relocs.is_empty());
        assert!(cubin.kernel("missing").is_none());

        let text_index = section_position(&cubin, ".text.vector_add");
        let text = cubin.elf.section(".text.vector_add").unwrap();
        assert_eq!(text.sh_type, SHT_PROGBITS);
        assert_eq!(text.flags, SHF_ALLOC | SHF_EXECINSTR);
        assert_eq!(text.align, 128);
        assert_eq!(text.link, 3);
        assert_eq!(text.info, TEXT_INFO_HIGH | 1);

        let constant = cubin
            .elf
            .section(".nv.constant0.vector_add")
            .unwrap();
        assert_eq!(constant.sh_type, SHT_PROGBITS);
        assert_eq!(constant.flags, SHF_ALLOC | SHF_INFO_LINK);
        assert_eq!(constant.align, 4);
        assert_eq!(constant.info as usize, text_index);

        let symtab = cubin.elf.section(".symtab").unwrap();
        assert_eq!(symtab.sh_type, SHT_SYMTAB);
        assert_eq!(symtab.link, 2);
        assert_eq!(symtab.entsize, SYM_SIZE as u64);
        assert_eq!(symtab.info, 1);
        assert!(cubin.elf.section(".nv.info").is_some());
        assert!(cubin.elf.section(".shstrtab").is_some());
        assert!(cubin.elf.section(".strtab").is_some());

        let symbol = cubin
            .elf
            .symbols
            .iter()
            .find(|symbol| symbol.name == "vector_add")
            .unwrap();
        assert_eq!(symbol.info, st_info(STB_GLOBAL, STT_FUNC));
        assert_eq!(symbol.info, 0x12);
        assert_eq!(st_binding(symbol.info), STB_GLOBAL);
        assert_eq!(st_type(symbol.info), STT_FUNC);
        assert_eq!(symbol.size, 3);
        assert_eq!(symbol.shndx as usize, text_index);

        assert_eq!(cubin.elf.segments.len(), 2);
        assert_eq!(cubin.elf.segments[0].p_type, PT_PHDR);
        assert_eq!(cubin.elf.segments[1].p_type, PT_LOAD);
        assert_eq!(cubin.elf.segments[0].filesz, 2 * PHDR_SIZE as u64);
        assert!(cubin.elf.segments[1].filesz > 0);
        assert_eq!(cubin.elf.segments[1].filesz, cubin.elf.segments[1].memsz);
        assert!(cubin.notes().unwrap().is_empty());
    }

    #[test]
    fn round_trips_through_the_writer() {
        let mut builder = CubinBuilder::new(100);
        builder
            .add_kernel(KernelImage {
                name: String::from("vector_add"),
                code: vec![1, 2, 3, 4, 5],
                constant_data: vec![9, 8, 7],
                relocs: vec![Reloc {
                    offset: 0x20,
                    kind: 0x4B,
                    symbol: String::from("vector_add"),
                    addend: 16,
                }],
            })
            .unwrap();
        builder
            .add_kernel(KernelImage {
                name: String::from("scale"),
                code: vec![6, 7],
                constant_data: Vec::new(),
                relocs: Vec::new(),
            })
            .unwrap();
        let bytes = builder.build().unwrap();
        let first = parse(&bytes).unwrap();
        let reencoded = first.elf.to_bytes().unwrap();
        let second = parse(&reencoded).unwrap();
        assert_eq!(first.sm, 100);
        assert_eq!(first.sm, second.sm);
        assert_eq!(first.elf, second.elf);
        assert_eq!(first.kernels, second.kernels);
    }

    #[test]
    fn encodes_verified_sm_flag_layouts() {
        assert_eq!(cuda_sm_flags(52), 0x0034_0534);
        assert_eq!(cuda_sm_flags(61), 0x003d_053d);
        assert_eq!(cuda_sm_flags(75), 0x004b_054b);
        assert_eq!(cuda_sm_flags(90), 0x005a_055a);
        assert_eq!(cuda_sm_flags(100), 0x0600_6402);
        assert_eq!(cuda_sm_flags(120), 0x0600_7802);
        for sm in [52_u16, 61, 75, 90, 100, 120] {
            assert_eq!(sm_from_flags(cuda_sm_flags(sm)), sm);
        }
        assert_eq!(sm_from_flags(0x005a_0d5a), 90);
    }

    #[test]
    fn selects_ident_bytes_from_architecture_and_toolkit() {
        let classic = parse(&build_with(CubinBuilder::new(75))).unwrap();
        assert_eq!(classic.elf.header.ident[7], 0x33);
        assert_eq!(classic.elf.header.ident[8], 0x07);
        assert_eq!(classic.elf.header.version, 129);

        let extended = parse(&build_with(CubinBuilder::new(100))).unwrap();
        assert_eq!(extended.elf.header.ident[7], 0x41);
        assert_eq!(extended.elf.header.ident[8], 0x08);
        assert_eq!(extended.elf.header.version, 1);

        let cuda13 = parse(&build_with(CubinBuilder::new(75).toolkit_version(130))).unwrap();
        assert_eq!(cuda13.elf.header.ident[7], 0x41);
        assert_eq!(cuda13.elf.header.ident[8], 0x08);
        assert_eq!(cuda13.elf.header.version, 130);
    }

    #[test]
    fn builds_relocatable_images() {
        let bytes = build_with(CubinBuilder::new(75).linked(false));
        let cubin = parse(&bytes).unwrap();
        assert_eq!(cubin.elf.header.e_type, ET_REL);
        assert_eq!(cubin.elf.header.phnum, 0);
        assert_eq!(cubin.elf.header.phoff, 0);
        assert_eq!(cubin.elf.header.phentsize, PHDR_SIZE as u16);
        assert!(cubin.elf.segments.is_empty());
        assert_eq!(cubin.sm, 75);
        assert_eq!(cubin.kernels.len(), 1);
    }

    #[test]
    fn round_trips_relocations_across_kernels() {
        let mut builder = CubinBuilder::new(75);
        builder
            .add_kernel(KernelImage {
                name: String::from("alpha"),
                code: vec![0xDE, 0xAD],
                constant_data: vec![0x01],
                relocs: vec![Reloc {
                    offset: 0x10,
                    kind: 0x38,
                    symbol: String::from("beta"),
                    addend: -4,
                }],
            })
            .unwrap();
        builder
            .add_kernel(KernelImage {
                name: String::from("beta"),
                code: vec![0xBE, 0xEF],
                constant_data: Vec::new(),
                relocs: Vec::new(),
            })
            .unwrap();
        let bytes = builder.build().unwrap();
        let cubin = parse(&bytes).unwrap();

        let alpha = cubin.kernel("alpha").unwrap();
        assert_eq!(
            alpha.relocs,
            vec![Reloc {
                offset: 0x10,
                kind: 0x38,
                symbol: String::from("beta"),
                addend: -4,
            }]
        );
        assert!(cubin.kernel("beta").is_some());

        let text_index = section_position(&cubin, ".text.alpha");
        let rela = cubin.elf.section(".rela.text.alpha").unwrap();
        assert_eq!(rela.sh_type, SHT_RELA);
        assert_eq!(rela.entsize, RELA_SIZE as u64);
        assert_eq!(rela.link, 3);
        assert_eq!(rela.info as usize, text_index);
        assert!(cubin.elf.section(".rela.text.beta").is_none());
        assert!(
            cubin
                .elf
                .symbols
                .iter()
                .any(|symbol| symbol.name == "beta")
        );
    }

    #[test]
    fn rejects_invalid_kernels_and_relocations() {
        let builder = CubinBuilder::new(75);
        assert_eq!(builder.build(), Err(Error::NoKernels));

        let mut builder = CubinBuilder::new(75);
        let empty = KernelImage {
            name: String::new(),
            code: vec![1],
            constant_data: Vec::new(),
            relocs: Vec::new(),
        };
        assert!(matches!(builder.add_kernel(empty), Err(Error::InvalidName)));
        let interior_nul = KernelImage {
            name: String::from("a\0b"),
            code: vec![1],
            constant_data: Vec::new(),
            relocs: Vec::new(),
        };
        assert!(matches!(
            builder.add_kernel(interior_nul),
            Err(Error::InvalidName)
        ));
        let first = KernelImage {
            name: String::from("dup"),
            code: vec![1],
            constant_data: Vec::new(),
            relocs: Vec::new(),
        };
        builder.add_kernel(first).unwrap();
        let second = KernelImage {
            name: String::from("dup"),
            code: vec![2],
            constant_data: Vec::new(),
            relocs: Vec::new(),
        };
        assert!(matches!(
            builder.add_kernel(second),
            Err(Error::DuplicateKernel(name)) if name == "dup"
        ));

        let mut builder = CubinBuilder::new(75);
        builder
            .add_kernel(KernelImage {
                name: String::from("alpha"),
                code: vec![1],
                constant_data: Vec::new(),
                relocs: vec![Reloc {
                    offset: 0,
                    kind: 0x38,
                    symbol: String::from("missing"),
                    addend: 0,
                }],
            })
            .unwrap();
        assert_eq!(
            builder.build(),
            Err(Error::UnknownRelocationSymbol(String::from("missing")))
        );
    }

    #[test]
    fn rejects_relocations_with_missing_symbols() {
        let mut builder = CubinBuilder::new(75);
        builder
            .add_kernel(KernelImage {
                name: String::from("alpha"),
                code: vec![0x10, 0x20],
                constant_data: Vec::new(),
                relocs: vec![Reloc {
                    offset: 0,
                    kind: 0x5A5A_1234,
                    symbol: String::from("alpha"),
                    addend: 0,
                }],
            })
            .unwrap();
        let bytes = builder.build().unwrap();
        assert_eq!(parse(&bytes).unwrap().kernels[0].relocs.len(), 1);

        let needle = [0x34, 0x12, 0x5A, 0x5A, 0x01, 0x00, 0x00, 0x00];
        let mut patched = bytes.clone();
        let at = patched
            .windows(needle.len())
            .position(|window| window == needle)
            .unwrap();
        patched[at + 4..at + 8].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(parse(&patched), Err(Error::InvalidRelocation)));
    }

    #[test]
    fn encodes_and_parses_notes() {
        let cuver_desc = [
            0x01, 0x00, 0x64, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        ];
        let cuver = Note {
            name: String::from("NVIDIA Corp"),
            n_type: NT_CUDA_CUINFO,
            desc: cuver_desc.to_vec(),
        };
        let encoded = cuver.encode().unwrap();
        assert_eq!(encoded.len(), 36);
        assert_eq!(u32::from_le_bytes(encoded[0..4].try_into().unwrap()), 12);
        assert_eq!(u32::from_le_bytes(encoded[4..8].try_into().unwrap()), 12);
        assert_eq!(
            u32::from_le_bytes(encoded[8..12].try_into().unwrap()),
            NT_CUDA_CUINFO
        );
        assert_eq!(&encoded[12..24], &b"NVIDIA Corp\0"[..]);
        assert_eq!(&encoded[24..36], &cuver_desc[..]);

        let tkinfo = Note {
            name: String::from("NVIDIA Corp"),
            n_type: NT_CUDA_TKINFO,
            desc: vec![0u8; 136],
        };
        assert_eq!(tkinfo.encode().unwrap().len(), 160);

        let all = encode_notes(&[cuver.clone(), tkinfo.clone()]).unwrap();
        assert_eq!(all.len(), 36 + 160);
        assert_eq!(parse_notes(&all).unwrap(), vec![cuver, tkinfo]);

        assert!(parse_notes(&[]).unwrap().is_empty());
        assert!(matches!(parse_notes(&encoded[..35]), Err(Error::MalformedNote)));
        assert!(matches!(parse_notes(&[0u8; 8]), Err(Error::MalformedNote)));
        let bad_name = Note {
            name: String::from("a\0b"),
            n_type: 1,
            desc: Vec::new(),
        };
        assert!(matches!(bad_name.encode(), Err(Error::InvalidName)));
    }

    #[test]
    fn reads_notes_from_container_sections() {
        let mut cubin = parse(&build_with(CubinBuilder::new(75))).unwrap();
        assert!(cubin.notes().unwrap().is_empty());
        let note = Note {
            name: String::from("NVIDIA Corp"),
            n_type: NT_CUDA_CUINFO,
            desc: vec![1, 2, 3, 4],
        };
        cubin.elf.sections.push(Section {
            name: String::from(".note.nv.cuver"),
            sh_type: SHT_NOTE,
            flags: 0,
            addr: 0,
            align: 4,
            link: 0,
            info: 0,
            entsize: 0,
            data: note.encode().unwrap(),
            bss_size: 0,
        });
        assert_eq!(cubin.notes().unwrap(), vec![note]);
    }

    #[test]
    fn rejects_non_cuda_elf_images() {
        let bytes = std::fs::read("/bin/ls").unwrap();
        assert!(matches!(
            parse(&bytes),
            Err(Error::MachineMismatch { machine: EM_X86_64 })
        ));
    }
}
