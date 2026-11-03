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

//! Dump the element streams a `Writable` actually holds in RAM, so a
//! compression measurement runs on Wredit data instead of on proxy text.
//!
//! Each input file is loaded through the engine - staged with `put`,
//! edited with `replace_range`/`overwrite_slice` so the committed view
//! carries the element types a real editing session produces, then
//! flushed - and the committed raw view is written out five ways:
//!
//! | suffix | contents                            | bytes per element |
//! |--------|-------------------------------------|-------------------|
//! | `.w32` | the packed `u32` elements, LE       | 4                 |
//! | `.p0`  | byte 0: the codepoint's low byte    | 1                 |
//! | `.pm`  | bytes 1-2: codepoint bits 8..24     | 2                 |
//! | `.p3`  | byte 3: the element type nibble     | 1                 |
//! | `.lo8` | `packed_char(e) as u8`, the current `save()` output | 1  |
//!
//! Usage:
//!   cargo run -p codevar-wredit --release --example writable_dump -- \
//!     --out DIR [--edits N] [--limit N] FILE...

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use codevar_wredit::compress::compress;
use codevar_wredit::edit::{
    CHAR_MASK, COMMON_MASK, Cursor, MERGE_MASK, StreamWritable, TextEditableBufferQuery, TextEditableDelete,
    TextEditableOverwrite, TextEditablePut, TextEditableReplace, Writable, make_packed, packed_char,
    packed_type,
};

/// The engine instantiation the editor uses for a text document.
type Doc = StreamWritable<u32, u32, 256>;

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            println!("writable_dump: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut out: Option<PathBuf> = None;
    let mut edits = 16usize;
    let mut limit = 0usize;
    let mut files: Vec<PathBuf> = Vec::new();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        if flag == "--out" || flag == "--edits" || flag == "--limit" {
            if index + 1 >= args.len() {
                return Err(format!("{flag} needs a value"));
            }
            let value = args[index + 1].as_str();
            match flag {
                "--out" => out = Some(PathBuf::from(value)),
                "--edits" => {
                    edits = value
                        .parse::<usize>()
                        .map_err(|_| "--edits: not a number")?
                }
                _ => {
                    limit = value
                        .parse::<usize>()
                        .map_err(|_| "--limit: not a number")?
                }
            }
            index += 1;
        } else {
            files.push(PathBuf::from(flag));
        }
        index += 1;
    }
    let dir = out.ok_or_else(|| "missing --out DIR".to_string())?;
    if files.is_empty() {
        return Err("no input files".to_string());
    }
    fs::create_dir_all(&dir).map_err(|err| format!("{}: {err}", dir.display()))?;
    println!(
        "{:<28} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>9} {:>9} codebook p0/w32 B",
        "document", "elements", "w32 B", "p0 B", "pm B", "p3 B", "wide", "escapes", "types"
    );
    for file in &files {
        dump_one(file, &dir, edits, limit)?;
    }
    Ok(())
}

fn dump_one(path: &Path, dir: &Path, edits: usize, limit: usize) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    let mut chars: Vec<u32> = text.chars().map(u32::from).collect();
    // The engine's character field is 28 bits; report what a load loses.
    let dropped = chars
        .iter()
        .copied()
        .filter(|&c| c > CHAR_MASK)
        .count();
    if limit != 0 && chars.len() > limit {
        chars.truncate(limit);
    }
    let mut doc = Doc::with_capacity(chars.len() + 4096);
    let cursor = doc
        .append_cursor(Cursor::new())
        .ok_or("cursor pool is full")?;
    for &ch in &chars {
        doc.put(cursor, make_packed(ch, COMMON_MASK))
            .map_err(|_| format!("{}: put failed", path.display()))?;
    }
    let len = chars.len();
    if len > 64 && edits != 0 {
        let step = len / edits;
        for e in 0..edits {
            let at = e * step + step / 2;
            if e % 2 == 0 {
                let repl = [make_packed(u32::from(b'x'), MERGE_MASK); 4];
                doc.replace_range(at, at + 4, &repl)
                    .map_err(|_| format!("{}: replace_range failed", path.display()))?;
            } else {
                doc.delete_range(at, at + 1)
                    .map_err(|_| format!("{}: delete_range failed", path.display()))?;
                let over = [make_packed(u32::from(b'y'), MERGE_MASK); 2];
                doc.overwrite_slice(at, &over)
                    .map_err(|_| format!("{}: overwrite_slice failed", path.display()))?;
            }
        }
    }
    doc.flush_all();
    let count = doc.raw_length();
    let raw = doc
        .raw_slice(0, count)
        .ok_or_else(|| "raw view is shorter than raw_length".to_string())?;

    let mut w32 = vec![0u8; count * 4];
    let mut p0 = Vec::with_capacity(count);
    let mut pm = Vec::with_capacity(count * 2);
    let mut p3 = Vec::with_capacity(count);
    let mut lo8 = Vec::with_capacity(count);
    let mut types: BTreeMap<u32, usize> = BTreeMap::new();
    let mut wide = 0usize;
    let mut escapes = 0usize;
    for (i, &el) in raw.iter().enumerate() {
        w32[4 * i..4 * i + 4].copy_from_slice(&el.to_le_bytes());
        p0.push(el as u8);
        let mid = [((el >> 8) & 0xFF) as u8, ((el >> 16) & 0xFF) as u8];
        if mid[0] != 0 || mid[1] != 0 {
            escapes += 1;
        }
        pm.extend_from_slice(&mid);
        p3.push((el >> 24) as u8);
        lo8.push(packed_char(el) as u8);
        *types.entry(packed_type(el)).or_insert(0) += 1;
        if packed_char(el) > 0xFF {
            wide += 1;
        }
    }
    let stem = path
        .file_stem()
        .map_or_else(|| "doc".to_string(), |s| s.to_string_lossy().into_owned());
    for (suffix, body) in [
        (".w32", &w32),
        (".p0", &p0),
        (".pm", &pm),
        (".p3", &p3),
        (".lo8", &lo8),
    ] {
        let mut name = stem.clone();
        name.push_str(suffix);
        fs::write(dir.join(&name), body).map_err(|err| format!("{name}: {err}"))?;
    }
    // The engine's own compressor, which today only sees directory names.
    let cb0 = compress(&p0).len();
    let cb32 = compress(&w32).len();
    let type_list: Vec<String> = types
        .iter()
        .map(|(t, n)| format!("{t:X}:{n}"))
        .collect();
    let codebook = format!("{cb0}/{cb32}");
    println!(
        "{:<28} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>9} {:>9} {} (codepoint>28b dropped: {dropped})",
        stem,
        count,
        w32.len(),
        p0.len(),
        pm.len(),
        p3.len(),
        wide,
        escapes,
        type_list.join(","),
        codebook
    );
    Ok(())
}
