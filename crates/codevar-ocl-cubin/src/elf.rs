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

//! ELF64 little-endian reader and writer.
//!
//! This module understands the subset of the ELF64 format used by CUDA `cubin`
//! files: little-endian encoding, section headers, program headers, and symbol
//! tables. [`parse`] reads a byte slice into an [`Elf`] value and
//! [`Elf::to_bytes`] serializes one again with a canonical layout: section
//! payloads first, each aligned to its `sh_addralign`, then the section header
//! table, then the program header table.
//!
//! Parsing and serializing are both linear, single-pass operations; the parser
//! never reads outside the slice it was given.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::Error;

/// `EI_CLASS` value identifying a 64-bit ELF file.
pub const ELFCLASS64: u8 = 2;
/// `EI_DATA` value identifying little-endian encoding.
pub const ELFDATA2LSB: u8 = 1;
/// Size of the ELF64 file header in bytes.
pub const EHDR_SIZE: usize = 64;
/// Size of one ELF64 section header in bytes.
pub const SHDR_SIZE: usize = 64;
/// Size of one ELF64 program header in bytes.
pub const PHDR_SIZE: usize = 56;
/// Size of one ELF64 symbol table entry in bytes.
pub const SYM_SIZE: usize = 24;
/// Size of one ELF64 `SHT_RELA` relocation entry in bytes.
pub const RELA_SIZE: usize = 24;
/// Size of one ELF64 `SHT_REL` relocation entry in bytes.
pub const REL_SIZE: usize = 16;
/// `e_type` for a relocatable object file, as produced by `ptxas -c`.
pub const ET_REL: u16 = 1;
/// `e_type` for a linked image, as produced by default from `ptxas`.
pub const ET_EXEC: u16 = 2;
/// `e_machine` value identifying an NVIDIA CUDA cubin.
pub const EM_CUDA: u16 = 0x00BE;
/// `e_machine` value identifying an x86-64 System V binary.
pub const EM_X86_64: u16 = 62;
/// Section type for the mandatory section header at index 0.
pub const SHT_NULL: u32 = 0;
/// Section type holding arbitrary keyed bytes, such as kernel code.
pub const SHT_PROGBITS: u32 = 1;
/// Section type for a symbol table linked to a string table.
pub const SHT_SYMTAB: u32 = 2;
/// Section type for a string table.
pub const SHT_STRTAB: u32 = 3;
/// Section type for relocations with explicit addends.
pub const SHT_RELA: u32 = 4;
/// Section type for notes.
pub const SHT_NOTE: u32 = 7;
/// Section type occupying no file space, such as `.bss`.
pub const SHT_NOBITS: u32 = 8;
/// Section type for relocations without explicit addends.
pub const SHT_REL: u32 = 9;
/// Section type for dynamic-linker symbol tables, such as `.dynsym`.
pub const SHT_DYNSYM: u32 = 11;
/// Section flag marking a section writable.
pub const SHF_WRITE: u64 = 0x1;
/// Section flag marking a section loaded into memory.
pub const SHF_ALLOC: u64 = 0x2;
/// Section flag marking a section executable.
pub const SHF_EXECINSTR: u64 = 0x4;
/// Section flag marking `sh_info` as meaningful to linkers.
pub const SHF_INFO_LINK: u64 = 0x40;
/// Symbol binding for symbols visible to the whole image.
pub const STB_GLOBAL: u8 = 1;
/// Symbol type for an executable routine, such as a kernel entry.
pub const STT_FUNC: u8 = 2;
/// Section header index meaning "no section".
pub const SHN_UNDEF: u16 = 0;
/// Program header type for a loadable segment.
pub const PT_LOAD: u32 = 1;
/// Program header type describing the program header table itself.
pub const PT_PHDR: u32 = 6;
/// Segment flag marking a segment executable.
pub const PF_X: u32 = 0x1;
/// Segment flag marking a segment writable.
pub const PF_W: u32 = 0x2;
/// Segment flag marking a segment readable.
pub const PF_R: u32 = 0x4;

/// Largest file [`Elf::to_bytes`] will produce, in bytes.
///
/// The limit keeps a corrupted input (for example an absurd `sh_addralign`)
/// from turning into a huge allocation when the container is written back out.
/// Real cubin images are measured in kilobytes.
const MAX_OUTPUT_SIZE: u64 = 64 * 1024 * 1024;

/// The 64-byte ELF64 file header.
///
/// Multi-byte fields are stored in host order after parsing; the reader only
/// accepts little-endian files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// `e_ident` — magic, class, byte order, and ABI identification bytes.
    pub ident: [u8; 16],
    /// `e_type` — [`ET_REL`] for relocatable objects, [`ET_EXEC`] for images.
    pub e_type: u16,
    /// `e_machine` — [`EM_CUDA`] for cubin files.
    pub machine: u16,
    /// `e_version` — the toolchain version encoded as `major * 10 + minor`.
    pub version: u32,
    /// `e_entry` — entry point, always zero in cubin files.
    pub entry: u64,
    /// `e_phoff` — file offset of the program header table.
    pub phoff: u64,
    /// `e_shoff` — file offset of the section header table.
    pub shoff: u64,
    /// `e_flags` — architecture flags; see `cuda_sm_flags`.
    pub flags: u32,
    /// `e_ehsize` — size of this header, always 64 for ELF64.
    pub ehsize: u16,
    /// `e_phentsize` — size of one program header, always 56 for ELF64.
    pub phentsize: u16,
    /// `e_phnum` — number of program headers.
    pub phnum: u16,
    /// `e_shentsize` — size of one section header, always 64 for ELF64.
    pub shentsize: u16,
    /// `e_shnum` — number of section headers.
    pub shnum: u16,
    /// `e_shstrndx` — index of the section name string table.
    pub shstrndx: u16,
}

/// One entry of the section header table.
///
/// The file offset is intentionally absent: [`parse`] recomputes placement
/// from scratch whenever a container is written, so only the payload and its
/// alignment are kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// Section name as stored in the `.shstrtab` section.
    pub name: String,
    /// `sh_type` — one of the `SHT_*` constants.
    pub sh_type: u32,
    /// `sh_flags` — the `SHF_*` bitmask.
    pub flags: u64,
    /// `sh_addr` — virtual address, always zero in cubin files.
    pub addr: u64,
    /// `sh_addralign` — payload alignment; 0 or 1 means no constraint.
    pub align: u64,
    /// `sh_link` — index of a related section, such as a string table.
    pub link: u32,
    /// `sh_info` — section-specific extra meaning.
    pub info: u32,
    /// `sh_entsize` — size of one entry for table sections, else 0.
    pub entsize: u64,
    /// Section payload; always empty when `sh_type` is [`SHT_NOBITS`].
    pub data: Vec<u8>,
    /// `sh_size` for [`SHT_NOBITS`] sections, which occupy no file space.
    pub bss_size: u64,
}

/// One entry of the program header table.
///
/// The writer recomputes `offset`, `vaddr`, `paddr`, `filesz`, and `memsz`
/// from the sections each segment covers; the remaining fields are preserved
/// exactly as parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// `p_type` — [`PT_LOAD`] or [`PT_PHDR`]; anything else is rejected.
    pub p_type: u32,
    /// `p_flags` — the `PF_*` bitmask.
    pub p_flags: u32,
    /// `p_offset` — file offset of the segment.
    pub offset: u64,
    /// `p_vaddr` — virtual address of the segment.
    pub vaddr: u64,
    /// `p_paddr` — physical address of the segment.
    pub paddr: u64,
    /// `p_filesz` — number of bytes occupied in the file.
    pub filesz: u64,
    /// `p_memsz` — number of bytes occupied in memory.
    pub memsz: u64,
    /// `p_align` — segment alignment.
    pub align: u64,
}

/// One entry of a symbol table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    /// Symbol name as stored in the linked string table.
    pub name: String,
    /// `st_info` — binding and type packed into one byte.
    pub info: u8,
    /// `st_other` — reserved visibility byte, always 0 in cubin files.
    pub other: u8,
    /// `st_shndx` — index of the defining section.
    pub shndx: u16,
    /// `st_value` — symbol value; the section offset for functions.
    pub value: u64,
    /// `st_size` — size in bytes of the symbol's payload.
    pub size: u64,
}

/// A parsed ELF64 container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Elf {
    /// The file header as parsed.
    pub header: Header,
    /// Section headers in table order, including the NULL entry at index 0.
    pub sections: Vec<Section>,
    /// Program headers in table order; empty when the file has none.
    pub segments: Vec<Segment>,
    /// Symbols from the first `SHT_SYMTAB` section, if the file has one.
    pub symbols: Vec<Symbol>,
}

/// Combines an `st_info` binding and symbol type into one byte.
#[inline]
pub fn st_info(binding: u8, kind: u8) -> u8 {
    (binding << 4) | (kind & 0x0F)
}

/// Extracts the `STB_*` binding from an `st_info` byte.
#[inline]
pub fn st_binding(info: u8) -> u8 {
    info >> 4
}

/// Extracts the `STT_*` type from an `st_info` byte.
#[inline]
pub fn st_type(info: u8) -> u8 {
    info & 0x0F
}

/// Reads a little-endian `u16` at `offset`.
pub(crate) fn le_u16(bytes: &[u8], offset: usize) -> Result<u16, Error> {
    let end = offset.checked_add(2).ok_or(Error::Overflow)?;
    let slice = bytes.get(offset..end).ok_or(Error::Truncated {
        needed: 2,
        available: bytes.len().saturating_sub(offset),
    })?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

/// Reads a little-endian `u32` at `offset`.
pub(crate) fn le_u32(bytes: &[u8], offset: usize) -> Result<u32, Error> {
    let end = offset.checked_add(4).ok_or(Error::Overflow)?;
    let slice = bytes.get(offset..end).ok_or(Error::Truncated {
        needed: 4,
        available: bytes.len().saturating_sub(offset),
    })?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Reads a little-endian `u64` at `offset`.
pub(crate) fn le_u64(bytes: &[u8], offset: usize) -> Result<u64, Error> {
    let end = offset.checked_add(8).ok_or(Error::Overflow)?;
    let slice = bytes.get(offset..end).ok_or(Error::Truncated {
        needed: 8,
        available: bytes.len().saturating_sub(offset),
    })?;
    let array: [u8; 8] = slice.try_into().map_err(|_| Error::Truncated {
        needed: 8,
        available: bytes.len().saturating_sub(offset),
    })?;
    Ok(u64::from_le_bytes(array))
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn align_up(value: u64, align: u64) -> Result<u64, Error> {
    if align <= 1 || value.is_multiple_of(align) {
        return Ok(value);
    }
    let padding = align - (value % align);
    value.checked_add(padding).ok_or(Error::Overflow)
}

fn intern<'a>(table: &mut Vec<u8>, seen: &mut BTreeMap<&'a str, u32>, name: &'a str) -> Result<u32, Error> {
    if name.as_bytes().contains(&0u8) {
        return Err(Error::InvalidName);
    }
    if let Some(&offset) = seen.get(name) {
        return Ok(offset);
    }
    let offset = u32::try_from(table.len()).map_err(|_| Error::Overflow)?;
    table.extend_from_slice(name.as_bytes());
    table.push(0u8);
    seen.insert(name, offset);
    Ok(offset)
}

fn cstr(table: &[u8], offset: u32) -> Result<String, Error> {
    if offset == 0 {
        return Ok(String::new());
    }
    let start = usize::try_from(offset).map_err(|_| Error::Overflow)?;
    let rest = table
        .get(start..)
        .ok_or(Error::InvalidStringOffset { offset })?;
    let mut end = 0usize;
    while end < rest.len() && rest[end] != 0u8 {
        end += 1;
    }
    if end == rest.len() {
        return Err(Error::InvalidStringOffset { offset });
    }
    Ok(String::from_utf8_lossy(&rest[..end]).into_owned())
}

fn track_min(slot: &mut Option<u64>, value: u64) {
    *slot = Some((*slot).unwrap_or(value).min(value));
}

fn track_max(slot: &mut Option<u64>, value: u64) {
    *slot = Some((*slot).unwrap_or(value).max(value));
}

/// Parses an ELF64 little-endian container.
///
/// The input may be any byte sequence: every bounds check is explicit, so a
/// malformed file produces an [`Error`] rather than a panic. Only the first
/// `SHT_SYMTAB` section is decoded into [`Elf::symbols`]; other symbol tables
/// (for example `.dynsym`) remain available as raw sections.
///
/// # Errors
///
/// Returns [`Error::Truncated`] when headers run past the end of the input,
/// [`Error::BadMagic`], [`Error::UnsupportedClass`], or
/// [`Error::UnsupportedEndianness`] for foreign file formats, and the
/// section-specific variants when the section table, string table, or symbol
/// table is inconsistent.
pub fn parse(bytes: &[u8]) -> Result<Elf, Error> {
    if bytes.len() < EHDR_SIZE {
        return Err(Error::Truncated {
            needed: EHDR_SIZE,
            available: bytes.len(),
        });
    }
    if bytes[0..4] != [0x7F_u8, b'E', b'L', b'F'] {
        return Err(Error::BadMagic);
    }
    if bytes[4] != ELFCLASS64 {
        return Err(Error::UnsupportedClass(bytes[4]));
    }
    if bytes[5] != ELFDATA2LSB {
        return Err(Error::UnsupportedEndianness(bytes[5]));
    }

    let mut ident = [0u8; 16];
    ident.copy_from_slice(&bytes[0..16]);
    let e_type = le_u16(bytes, 16)?;
    let machine = le_u16(bytes, 18)?;
    let version = le_u32(bytes, 20)?;
    let entry = le_u64(bytes, 24)?;
    let phoff = le_u64(bytes, 32)?;
    let shoff = le_u64(bytes, 40)?;
    let flags = le_u32(bytes, 48)?;
    let ehsize = le_u16(bytes, 52)?;
    let phentsize = le_u16(bytes, 54)?;
    let phnum = le_u16(bytes, 56)?;
    let shentsize = le_u16(bytes, 58)?;
    let shnum = le_u16(bytes, 60)?;
    let shstrndx = le_u16(bytes, 62)?;

    if shnum == 0 {
        return Err(Error::UnsupportedSectionCount);
    }
    if usize::from(shentsize) != SHDR_SIZE {
        return Err(Error::UnsupportedSectionEntrySize(shentsize));
    }
    if phnum > 0 && usize::from(phentsize) != PHDR_SIZE {
        return Err(Error::UnsupportedProgramEntrySize(phentsize));
    }
    if shstrndx >= shnum {
        return Err(Error::ShstrndxOutOfRange {
            index: shstrndx,
            count: shnum,
        });
    }

    let sh_base = usize::try_from(shoff).map_err(|_| Error::Overflow)?;
    let sh_bytes = usize::from(shnum) * SHDR_SIZE;
    let sh_end = sh_base
        .checked_add(sh_bytes)
        .ok_or(Error::Overflow)?;
    if sh_end > bytes.len() {
        return Err(Error::Truncated {
            needed: sh_bytes,
            available: bytes.len().saturating_sub(sh_base),
        });
    }
    if phnum > 0 {
        let phoff = usize::try_from(phoff).map_err(|_| Error::Overflow)?;
        let ph_bytes = usize::from(phnum) * PHDR_SIZE;
        let ph_end = phoff
            .checked_add(ph_bytes)
            .ok_or(Error::Overflow)?;
        if ph_end > bytes.len() {
            return Err(Error::Truncated {
                needed: ph_bytes,
                available: bytes.len().saturating_sub(phoff),
            });
        }
    }

    let str_header = sh_base + usize::from(shstrndx) * SHDR_SIZE;
    let str_type = le_u32(bytes, str_header + 4)?;
    let str_offset = le_u64(bytes, str_header + 24)?;
    let str_size = le_u64(bytes, str_header + 32)?;
    let strtab: &[u8] = if str_type == SHT_NOBITS {
        &[]
    } else {
        let start = usize::try_from(str_offset).map_err(|_| Error::Overflow)?;
        let size = usize::try_from(str_size).map_err(|_| Error::Overflow)?;
        let end = start.checked_add(size).ok_or(Error::Overflow)?;
        bytes
            .get(start..end)
            .ok_or(Error::SectionDataOutOfBounds {
                index: usize::from(shstrndx),
            })?
    };

    let mut sections = Vec::with_capacity(usize::from(shnum));
    for index in 0..usize::from(shnum) {
        let at = sh_base + index * SHDR_SIZE;
        let name_offset = le_u32(bytes, at)?;
        let sh_type = le_u32(bytes, at + 4)?;
        let flags = le_u64(bytes, at + 8)?;
        let addr = le_u64(bytes, at + 16)?;
        let offset = le_u64(bytes, at + 24)?;
        let size = le_u64(bytes, at + 32)?;
        let link = le_u32(bytes, at + 40)?;
        let info = le_u32(bytes, at + 44)?;
        let align = le_u64(bytes, at + 48)?;
        let entsize = le_u64(bytes, at + 56)?;
        let name = cstr(strtab, name_offset)?;
        let (data, bss_size) = if sh_type == SHT_NOBITS {
            (Vec::new(), size)
        } else {
            let start = usize::try_from(offset).map_err(|_| Error::Overflow)?;
            let payload = usize::try_from(size).map_err(|_| Error::Overflow)?;
            let end = start
                .checked_add(payload)
                .ok_or(Error::Overflow)?;
            let slice = bytes
                .get(start..end)
                .ok_or(Error::SectionDataOutOfBounds { index })?;
            (slice.to_vec(), 0)
        };
        sections.push(Section {
            name,
            sh_type,
            flags,
            addr,
            align,
            link,
            info,
            entsize,
            data,
            bss_size,
        });
    }

    let mut segments = Vec::with_capacity(usize::from(phnum));
    if phnum > 0 {
        let base = usize::try_from(phoff).map_err(|_| Error::Overflow)?;
        for index in 0..usize::from(phnum) {
            let at = base + index * PHDR_SIZE;
            segments.push(Segment {
                p_type: le_u32(bytes, at)?,
                p_flags: le_u32(bytes, at + 4)?,
                offset: le_u64(bytes, at + 8)?,
                vaddr: le_u64(bytes, at + 16)?,
                paddr: le_u64(bytes, at + 24)?,
                filesz: le_u64(bytes, at + 32)?,
                memsz: le_u64(bytes, at + 40)?,
                align: le_u64(bytes, at + 48)?,
            });
        }
    }

    let mut symbols = Vec::new();
    if let Some(sym_index) = sections
        .iter()
        .position(|section| section.sh_type == SHT_SYMTAB)
    {
        let link_value = sections[sym_index].link;
        let entsize = sections[sym_index].entsize;
        let payload_len = sections[sym_index].data.len();
        if entsize != SYM_SIZE as u64 || !payload_len.is_multiple_of(SYM_SIZE) {
            return Err(Error::InvalidSymbolTable);
        }
        let link = usize::try_from(link_value).map_err(|_| Error::Overflow)?;
        if link >= sections.len() {
            return Err(Error::LinkOutOfRange {
                section: sym_index,
                link: link_value,
            });
        }
        let str_data = &sections[link].data;
        for entry in sections[sym_index].data.chunks_exact(SYM_SIZE) {
            let name_offset = le_u32(entry, 0)?;
            symbols.push(Symbol {
                name: cstr(str_data, name_offset)?,
                info: entry[4],
                other: entry[5],
                shndx: u16::from_le_bytes([entry[6], entry[7]]),
                value: le_u64(entry, 8)?,
                size: le_u64(entry, 16)?,
            });
        }
    }

    Ok(Elf {
        header: Header {
            ident,
            e_type,
            machine,
            version,
            entry,
            phoff,
            shoff,
            flags,
            ehsize,
            phentsize,
            phnum,
            shentsize,
            shnum,
            shstrndx,
        },
        sections,
        segments,
        symbols,
    })
}

impl Elf {
    /// Creates an empty container carrying only `header`.
    pub fn new(header: Header) -> Self {
        Self {
            header,
            sections: Vec::new(),
            segments: Vec::new(),
            symbols: Vec::new(),
        }
    }

    /// Returns the first section with the given `name`.
    pub fn section(&self, name: &str) -> Option<&Section> {
        self.sections
            .iter()
            .find(|section| section.name == name)
    }

    /// Serializes the container back into an ELF64 byte stream.
    ///
    /// Placement is recomputed from scratch: section payloads are laid out at
    /// their `sh_addralign`, followed by the section header table (8-byte
    /// aligned) and then the program header table (8-byte aligned). The
    /// `.shstrtab` payload and the first `SHT_SYMTAB` payload (with its linked
    /// string table) are rebuilt from the in-memory names, `e_shstrndx`,
    /// `e_shnum`, `e_shoff`, `e_phoff`, `e_phnum`, `e_ehsize`, `e_phentsize`,
    /// and `e_shentsize` are recomputed, and segment spans are re-derived from
    /// the sections they cover. Section 0 never carries a payload: its
    /// `sh_size` is written as 0 unless it is `SHT_NOBITS`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSections`] when there is nothing to write,
    /// [`Error::MissingShstrtab`] when sections carry names but no
    /// `.shstrtab` section exists, [`Error::LinkOutOfRange`] when the symbol
    /// table links to an unusable string table,
    /// [`Error::UnsupportedSegmentType`] for program headers other than
    /// `PT_PHDR` and `PT_LOAD`, [`Error::InvalidName`] when a name contains an
    /// interior NUL byte, and [`Error::Overflow`] when the layout exceeds
    /// 64 MiB or any size computation wraps.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let section_count = self.sections.len();
        if section_count == 0 {
            return Err(Error::NoSections);
        }
        if section_count > usize::from(u16::MAX) {
            return Err(Error::Overflow);
        }
        let segment_count = self.segments.len();
        if segment_count > usize::from(u16::MAX) {
            return Err(Error::Overflow);
        }

        let mut name_offsets = vec![0u32; section_count];
        let mut shstrtab_data: Vec<u8> = Vec::new();
        let mut shstrtab_index: Option<usize> = None;
        if self
            .sections
            .iter()
            .any(|section| !section.name.is_empty())
        {
            let index = self
                .sections
                .iter()
                .position(|section| section.name == ".shstrtab")
                .filter(|&index| index != 0)
                .ok_or(Error::MissingShstrtab)?;
            let mut seen: BTreeMap<&str, u32> = BTreeMap::new();
            intern(&mut shstrtab_data, &mut seen, "")?;
            for (position, section) in self.sections.iter().enumerate() {
                name_offsets[position] = intern(&mut shstrtab_data, &mut seen, &section.name)?;
            }
            shstrtab_index = Some(index);
        }

        let mut symtab_data: Vec<u8> = Vec::new();
        let mut strtab_data: Vec<u8> = Vec::new();
        let mut strtab_index: Option<usize> = None;
        let mut first_global = 0u32;
        let symtab_index = self
            .sections
            .iter()
            .position(|section| section.sh_type == SHT_SYMTAB)
            .filter(|&index| index != 0);
        if let Some(index) = symtab_index {
            let link_value = self.sections[index].link;
            let link_index = usize::try_from(link_value).map_err(|_| Error::Overflow)?;
            if link_index == 0 || link_index >= section_count {
                return Err(Error::LinkOutOfRange {
                    section: index,
                    link: link_value,
                });
            }
            strtab_index = Some(link_index);
            let mut seen: BTreeMap<&str, u32> = BTreeMap::new();
            intern(&mut strtab_data, &mut seen, "")?;
            for symbol in &self.symbols {
                let name_offset = intern(&mut strtab_data, &mut seen, &symbol.name)?;
                symtab_data.extend_from_slice(&name_offset.to_le_bytes());
                symtab_data.push(symbol.info);
                symtab_data.push(symbol.other);
                symtab_data.extend_from_slice(&symbol.shndx.to_le_bytes());
                symtab_data.extend_from_slice(&symbol.value.to_le_bytes());
                symtab_data.extend_from_slice(&symbol.size.to_le_bytes());
            }
            let first = self
                .symbols
                .iter()
                .position(|symbol| st_binding(symbol.info) == STB_GLOBAL)
                .unwrap_or(self.symbols.len());
            first_global = u32::try_from(first).map_err(|_| Error::Overflow)?;
        }

        let mut payloads: Vec<&[u8]> = self
            .sections
            .iter()
            .map(|section| section.data.as_slice())
            .collect();
        if let Some(index) = shstrtab_index {
            payloads[index] = shstrtab_data.as_slice();
        }
        if let Some(index) = symtab_index {
            payloads[index] = symtab_data.as_slice();
        }
        if let Some(index) = strtab_index {
            payloads[index] = strtab_data.as_slice();
        }

        let mut offsets = vec![0u64; section_count];
        let mut cursor = EHDR_SIZE as u64;
        for (index, section) in self.sections.iter().enumerate().skip(1) {
            let position = align_up(cursor, section.align)?;
            offsets[index] = position;
            if section.sh_type != SHT_NOBITS {
                let length = u64::try_from(payloads[index].len()).map_err(|_| Error::Overflow)?;
                cursor = position
                    .checked_add(length)
                    .ok_or(Error::Overflow)?;
            }
        }

        let sh_size = u64::try_from(section_count)
            .map_err(|_| Error::Overflow)?
            .checked_mul(SHDR_SIZE as u64)
            .ok_or(Error::Overflow)?;
        let shoff = align_up(cursor, 8)?;
        let sh_end = shoff
            .checked_add(sh_size)
            .ok_or(Error::Overflow)?;
        let ph_size = u64::try_from(segment_count)
            .map_err(|_| Error::Overflow)?
            .checked_mul(PHDR_SIZE as u64)
            .ok_or(Error::Overflow)?;
        let phoff = if segment_count > 0 {
            align_up(sh_end, 8)?
        } else {
            0
        };
        let planned = cursor
            .checked_add(sh_size)
            .and_then(|total| total.checked_add(ph_size))
            .and_then(|total| total.checked_add(16))
            .ok_or(Error::Overflow)?;
        if planned > MAX_OUTPUT_SIZE {
            return Err(Error::Overflow);
        }

        let mut spans: Vec<(u64, u64, u64, u64)> = Vec::with_capacity(segment_count);
        for segment in &self.segments {
            match segment.p_type {
                PT_PHDR => spans.push((phoff, 0, ph_size, ph_size)),
                PT_LOAD => {
                    let writable = segment.p_flags & PF_W != 0;
                    let mut first_file: Option<u64> = None;
                    let mut last_file: Option<u64> = None;
                    let mut first_mem: Option<u64> = None;
                    let mut last_mem: Option<u64> = None;
                    let mut base_address: Option<u64> = None;
                    for (index, section) in self.sections.iter().enumerate() {
                        if section.flags & SHF_ALLOC == 0 {
                            continue;
                        }
                        if ((section.flags & SHF_WRITE) != 0) != writable {
                            continue;
                        }
                        let start = offsets[index];
                        track_min(&mut first_mem, start);
                        track_min(&mut base_address, section.addr);
                        if section.sh_type == SHT_NOBITS {
                            let end = start
                                .checked_add(section.bss_size)
                                .ok_or(Error::Overflow)?;
                            track_max(&mut last_mem, end);
                        } else {
                            let length = u64::try_from(payloads[index].len()).map_err(|_| Error::Overflow)?;
                            let end = start.checked_add(length).ok_or(Error::Overflow)?;
                            track_max(&mut last_mem, end);
                            track_min(&mut first_file, start);
                            track_max(&mut last_file, end);
                        }
                    }
                    let offset = first_file.or(first_mem).unwrap_or(cursor);
                    let filesz = match (first_file, last_file) {
                        (Some(start), Some(end)) => end.checked_sub(start).ok_or(Error::Overflow)?,
                        _ => 0,
                    };
                    let memsz = match (first_mem, last_mem) {
                        (Some(start), Some(end)) => end.checked_sub(start).ok_or(Error::Overflow)?,
                        _ => 0,
                    };
                    let vaddr = base_address.unwrap_or(0);
                    spans.push((offset, vaddr, filesz, memsz));
                }
                other => return Err(Error::UnsupportedSegmentType(other)),
            }
        }

        let mut ident = self.header.ident;
        ident[4] = ELFCLASS64;
        ident[5] = ELFDATA2LSB;

        let capacity = usize::try_from(planned).map_err(|_| Error::Overflow)?;
        let mut out: Vec<u8> = Vec::with_capacity(capacity);
        out.extend_from_slice(&ident);
        put_u16(&mut out, self.header.e_type);
        put_u16(&mut out, self.header.machine);
        put_u32(&mut out, self.header.version);
        put_u64(&mut out, self.header.entry);
        put_u64(&mut out, phoff);
        put_u64(&mut out, shoff);
        put_u32(&mut out, self.header.flags);
        put_u16(&mut out, EHDR_SIZE as u16);
        put_u16(&mut out, PHDR_SIZE as u16);
        put_u16(&mut out, segment_count as u16);
        put_u16(&mut out, SHDR_SIZE as u16);
        put_u16(&mut out, section_count as u16);
        put_u16(&mut out, shstrtab_index.unwrap_or(0) as u16);

        for index in 1..section_count {
            if self.sections[index].sh_type == SHT_NOBITS {
                continue;
            }
            let target = usize::try_from(offsets[index]).map_err(|_| Error::Overflow)?;
            if out.len() > target {
                return Err(Error::Overflow);
            }
            out.resize(target, 0u8);
            out.extend_from_slice(payloads[index]);
        }
        let sh_target = usize::try_from(shoff).map_err(|_| Error::Overflow)?;
        if out.len() > sh_target {
            return Err(Error::Overflow);
        }
        out.resize(sh_target, 0u8);

        for (index, section) in self.sections.iter().enumerate() {
            put_u32(&mut out, name_offsets[index]);
            put_u32(&mut out, section.sh_type);
            put_u64(&mut out, section.flags);
            put_u64(&mut out, section.addr);
            put_u64(&mut out, offsets[index]);
            let size = if section.sh_type == SHT_NOBITS {
                section.bss_size
            } else if index == 0 {
                0
            } else {
                u64::try_from(payloads[index].len()).map_err(|_| Error::Overflow)?
            };
            put_u64(&mut out, size);
            put_u32(&mut out, section.link);
            let info = if symtab_index == Some(index) {
                first_global
            } else {
                section.info
            };
            put_u32(&mut out, info);
            put_u64(&mut out, section.align);
            let entsize = if symtab_index == Some(index) {
                SYM_SIZE as u64
            } else {
                section.entsize
            };
            put_u64(&mut out, entsize);
        }

        if segment_count > 0 {
            let ph_target = usize::try_from(phoff).map_err(|_| Error::Overflow)?;
            if out.len() > ph_target {
                return Err(Error::Overflow);
            }
            out.resize(ph_target, 0u8);
            for (segment, span) in self.segments.iter().zip(spans) {
                let (offset, vaddr, filesz, memsz) = span;
                put_u32(&mut out, segment.p_type);
                put_u32(&mut out, segment.p_flags);
                put_u64(&mut out, offset);
                put_u64(&mut out, vaddr);
                put_u64(&mut out, vaddr);
                put_u64(&mut out, filesz);
                put_u64(&mut out, memsz);
                put_u64(&mut out, segment.align);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubin::{CubinBuilder, KernelImage, Reloc};

    fn minimal_image() -> Vec<u8> {
        let mut image = vec![0u8; EHDR_SIZE + SHDR_SIZE];
        image[0] = 0x7F;
        image[1] = b'E';
        image[2] = b'L';
        image[3] = b'F';
        image[4] = ELFCLASS64;
        image[5] = ELFDATA2LSB;
        image[6] = 1;
        image[20] = 1;
        image[40..48].copy_from_slice(&64u64.to_le_bytes());
        image[52..54].copy_from_slice(&64u16.to_le_bytes());
        image[54..56].copy_from_slice(&56u16.to_le_bytes());
        image[58..60].copy_from_slice(&64u16.to_le_bytes());
        image[60..62].copy_from_slice(&1u16.to_le_bytes());
        image
    }

    fn named_image(name_offset: u32) -> Vec<u8> {
        let mut image = vec![0u8; 196];
        image[0] = 0x7F;
        image[1] = b'E';
        image[2] = b'L';
        image[3] = b'F';
        image[4] = ELFCLASS64;
        image[5] = ELFDATA2LSB;
        image[6] = 1;
        image[40..48].copy_from_slice(&64u64.to_le_bytes());
        image[52..54].copy_from_slice(&64u16.to_le_bytes());
        image[54..56].copy_from_slice(&56u16.to_le_bytes());
        image[58..60].copy_from_slice(&64u16.to_le_bytes());
        image[60..62].copy_from_slice(&2u16.to_le_bytes());
        image[64 + 4..64 + 8].copy_from_slice(&SHT_STRTAB.to_le_bytes());
        image[64 + 24..64 + 32].copy_from_slice(&192u64.to_le_bytes());
        image[64 + 32..64 + 40].copy_from_slice(&4u64.to_le_bytes());
        image[64 + 48..64 + 56].copy_from_slice(&1u64.to_le_bytes());
        image[128..132].copy_from_slice(&name_offset.to_le_bytes());
        image[192..196].copy_from_slice(b"abc\0");
        image
    }

    fn sample_images() -> Vec<Vec<u8>> {
        let mut linked = CubinBuilder::new(75);
        linked
            .add_kernel(KernelImage {
                name: String::from("vector_add"),
                code: vec![0x55, 0xAA, 0x01, 0x02],
                constant_data: vec![0x11, 0x22],
                relocs: vec![Reloc {
                    offset: 8,
                    kind: 0x38,
                    symbol: String::from("vector_add"),
                    addend: -4,
                }],
            })
            .unwrap();
        linked
            .add_kernel(KernelImage {
                name: String::from("scale"),
                code: vec![0x33, 0x44],
                constant_data: Vec::new(),
                relocs: Vec::new(),
            })
            .unwrap();
        let linked_bytes = linked.build().unwrap();

        let mut relocatable = CubinBuilder::new(75);
        relocatable
            .add_kernel(KernelImage {
                name: String::from("vector_add"),
                code: vec![0x55, 0xAA, 0x01, 0x02],
                constant_data: Vec::new(),
                relocs: Vec::new(),
            })
            .unwrap();
        let relocatable_bytes = relocatable.linked(false).build().unwrap();

        vec![linked_bytes, relocatable_bytes]
    }

    fn patch_symtab_header(image: &mut [u8], field: usize, value: &[u8]) {
        let shoff = u64::from_le_bytes(image[40..48].try_into().unwrap()) as usize;
        let at = shoff + 3 * SHDR_SIZE + field;
        image[at..at + value.len()].copy_from_slice(value);
    }

    #[test]
    fn parses_system_binary() {
        let bytes = std::fs::read("/bin/ls").unwrap();
        let elf = parse(&bytes).unwrap();
        assert_eq!(elf.header.machine, EM_X86_64);
        assert_eq!(elf.header.ident[4], ELFCLASS64);
        assert_eq!(elf.header.ident[5], ELFDATA2LSB);
        assert!(elf.section(".text").is_some());
        assert!(elf.section(".shstrtab").is_some());
        assert!(
            elf.sections
                .iter()
                .any(|section| section.name == ".symtab" || section.name == ".dynsym")
        );
        assert!(!elf.segments.is_empty());
        let bss = elf.section(".bss").unwrap();
        assert_eq!(bss.sh_type, SHT_NOBITS);
        assert!(bss.data.is_empty());
        assert!(bss.bss_size > 0);
        assert!(
            elf.sections
                .iter()
                .all(|section| section.sh_type != SHT_NOBITS || section.data.is_empty())
        );
    }

    #[test]
    fn rejects_truncated_input() {
        let image = minimal_image();
        assert_eq!(
            parse(&image[..32]),
            Err(Error::Truncated {
                needed: EHDR_SIZE,
                available: 32
            })
        );
    }

    #[test]
    fn rejects_wrong_ident() {
        let mut image = minimal_image();
        image[0] = 0;
        assert_eq!(parse(&image), Err(Error::BadMagic));
        let mut image = minimal_image();
        image[4] = 1;
        assert_eq!(parse(&image), Err(Error::UnsupportedClass(1)));
        let mut image = minimal_image();
        image[5] = 2;
        assert_eq!(parse(&image), Err(Error::UnsupportedEndianness(2)));
    }

    #[test]
    fn rejects_bad_section_header_table() {
        let mut image = minimal_image();
        image[58..60].copy_from_slice(&32u16.to_le_bytes());
        assert_eq!(parse(&image), Err(Error::UnsupportedSectionEntrySize(32)));

        let mut image = minimal_image();
        image[60..62].copy_from_slice(&0u16.to_le_bytes());
        assert_eq!(parse(&image), Err(Error::UnsupportedSectionCount));

        let mut image = minimal_image();
        image[62..64].copy_from_slice(&5u16.to_le_bytes());
        assert_eq!(
            parse(&image),
            Err(Error::ShstrndxOutOfRange { index: 5, count: 1 })
        );

        let mut image = minimal_image();
        image[40..48].copy_from_slice(&10_000u64.to_le_bytes());
        assert!(matches!(parse(&image), Err(Error::Truncated { .. })));

        let mut image = minimal_image();
        image[54..56].copy_from_slice(&0u16.to_le_bytes());
        image[56..58].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(parse(&image), Err(Error::UnsupportedProgramEntrySize(0)));
    }

    #[test]
    fn rejects_section_data_beyond_input() {
        let mut image = minimal_image();
        image[64 + 4..64 + 8].copy_from_slice(&SHT_PROGBITS.to_le_bytes());
        image[64 + 24..64 + 32].copy_from_slice(&1_000u64.to_le_bytes());
        image[64 + 32..64 + 40].copy_from_slice(&10u64.to_le_bytes());
        assert_eq!(parse(&image), Err(Error::SectionDataOutOfBounds { index: 0 }));
    }

    #[test]
    fn reads_section_names_from_the_string_table() {
        let elf = parse(&named_image(1)).unwrap();
        assert_eq!(elf.sections[0].name, "");
        assert_eq!(elf.sections[1].name, "bc");

        let elf = parse(&named_image(0)).unwrap();
        assert_eq!(elf.sections[1].name, "");

        assert_eq!(
            parse(&named_image(10)),
            Err(Error::InvalidStringOffset { offset: 10 })
        );
        assert_eq!(
            parse(&named_image(4)),
            Err(Error::InvalidStringOffset { offset: 4 })
        );
    }

    #[test]
    fn decodes_little_endian_header_fields() {
        let mut image = minimal_image();
        image[18..20].copy_from_slice(&EM_CUDA.to_le_bytes());
        image[48..52].copy_from_slice(&0x004B_054B_u32.to_le_bytes());
        let elf = parse(&image).unwrap();
        assert_eq!(elf.header.machine, EM_CUDA);
        assert_eq!(elf.header.flags, 0x004B_054B);
    }

    #[test]
    fn rejects_invalid_symbol_table() {
        let image = sample_images().remove(0);

        let mut wrong_entsize = image.clone();
        patch_symtab_header(&mut wrong_entsize, 56, &16u64.to_le_bytes());
        assert_eq!(parse(&wrong_entsize), Err(Error::InvalidSymbolTable));

        let mut wrong_link = image.clone();
        patch_symtab_header(&mut wrong_link, 40, &99u32.to_le_bytes());
        assert_eq!(
            parse(&wrong_link),
            Err(Error::LinkOutOfRange { section: 3, link: 99 })
        );

        let mut short_payload = image.clone();
        patch_symtab_header(&mut short_payload, 32, &13u64.to_le_bytes());
        assert_eq!(parse(&short_payload), Err(Error::InvalidSymbolTable));
    }

    #[test]
    fn rejects_every_truncated_prefix() {
        for image in sample_images() {
            for end in 0..image.len() {
                assert!(
                    parse(&image[..end]).is_err(),
                    "prefix of {end} bytes parsed unexpectedly"
                );
            }
            assert!(parse(&image).is_ok());
        }
    }

    #[test]
    fn survives_mutated_input() {
        for image in sample_images() {
            let mut mutated = image.clone();
            for index in 0..mutated.len() {
                mutated[index] ^= 0xA5;
                if let Ok(elf) = parse(&mutated)
                    && let Ok(reencoded) = elf.to_bytes()
                {
                    assert!(
                        parse(&reencoded).is_ok(),
                        "reparse failed after flipping byte {index}"
                    );
                }
                mutated[index] ^= 0xA5;
            }
        }
    }

    #[test]
    fn writer_rejects_unwritable_containers() {
        let header = Header {
            ident: [0u8; 16],
            e_type: ET_REL,
            machine: EM_CUDA,
            version: 1,
            entry: 0,
            phoff: 0,
            shoff: 0,
            flags: 0,
            ehsize: EHDR_SIZE as u16,
            phentsize: 0,
            phnum: 0,
            shentsize: SHDR_SIZE as u16,
            shnum: 0,
            shstrndx: 0,
        };
        assert_eq!(Elf::new(header.clone()).to_bytes(), Err(Error::NoSections));

        let mut named = Elf::new(header.clone());
        named.sections.push(Section {
            name: String::from(".text.kernel"),
            sh_type: SHT_PROGBITS,
            flags: SHF_ALLOC,
            addr: 0,
            align: 128,
            link: 0,
            info: 0,
            entsize: 0,
            data: vec![1, 2, 3],
            bss_size: 0,
        });
        assert_eq!(named.to_bytes(), Err(Error::MissingShstrtab));

        let mut foreign = Elf::new(header);
        foreign.sections.push(Section {
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
        });
        foreign.segments.push(Segment {
            p_type: 0x6474_E550,
            p_flags: PF_R,
            offset: 0,
            vaddr: 0,
            paddr: 0,
            filesz: 0,
            memsz: 0,
            align: 1,
        });
        assert_eq!(
            foreign.to_bytes(),
            Err(Error::UnsupportedSegmentType(0x6474_E550))
        );
    }
}
