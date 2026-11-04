#!/usr/bin/env python3
"""Generate the flat filename codebook used by ``compress.rs``.

The codebook is a single ``&[u8]`` array of *length-prefixed* entries:
each entry is one length byte followed by that many name bytes. Keeping
it flat (instead of a slice-of-slices) makes the whole table one
contiguous, cache-friendly blob and lets the source be produced
mechanically from the word lists below.

Run it from the repository root (or anywhere):

    python3 crates/codevar-wredit/tools/gen_codebook.py

It rewrites ``crates/codevar-wredit/src/compress/codebook.rs``.
"""

import os

# A codebook index is one byte and 0xFE / 0xFF are the literal escapes,
# so at most this many entries can be addressed.
MAX_ENTRIES = 253

# Single characters first: the table is capped, and every single byte we
# list is one that would otherwise cost two bytes as a literal escape.
SINGLE = list("abcdefghijklmnopqrstuvwxyz")
SINGLE += list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SINGLE += list("0123456789")
SINGLE += ["_", "-", ".", "/", " ", ",", ":", ";", "=", "(", ")", "[", "]", "{", "}", "<", ">", "&", "*", "#", "@", "!", "?", "|", '"', "'", "\\", "+"]

# File extensions, most common first.
EXTENSIONS = [
    ".rs", ".toml", ".lock", ".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".py", ".pyi",
    ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt", ".yaml", ".yml",
    ".xml", ".html", ".css", ".scss", ".wasm", ".wat", ".cl", ".spv", ".glsl", ".sh",
    ".bash", ".zsh", ".bat", ".ps1", ".ini", ".cfg", ".conf", ".mk", ".cmake", ".lock",
    ".diff", ".patch", ".svg", ".png", ".jpg", ".gif", ".ico", ".woff", ".woff2", ".ttf",
    ".csv", ".tsv", ".log", ".tmp", ".bak", ".o", ".d", ".a", ".so", ".exe", ".jsonc",
]

# Directory names seen all over a source tree.
DIRECTORIES = [
    "src", "lib", "bin", "test", "tests", "doc", "docs", "example", "examples", "bench",
    "benches", "build", "target", "include", "kernel", "kernels", "shader", "shaders",
    "storage", "engine", "editor", "render", "renderer", "assets", "component",
    "components", "fixture", "fixtures", "migration", "migrations", "plugin", "plugins",
    "scripts", "tools", "utils", "util", "api", "cli", "gui", "ui", "core", "net", "db",
    "cache", "dist", "release", "debug", "vendor", "third_party", "node_modules",
]

# Frequent file names, keywords and fragments.
FILENAMES = [
    "main", "mod", "impl", "pub", "use", "fn", "let", "mut", "self", "crate", "struct",
    "enum", "trait", "type", "const", "static", "readme", "makefile", "config", "index",
    "error", "codevar", "wredit", "cargo", "gitignore", "license", "changelog",
    "dockerfile", "justfile", "gen", "generated", "test_", "_test", "utf8", "utf16",
    "simd", "hash", "block", "buffer", "cursor", "writable", "readonly", "workspace",
]

GROUPS = [SINGLE, EXTENSIONS, DIRECTORIES, FILENAMES]


def collect():
    """Deduplicates the groups in priority order and caps the count."""
    seen = set()
    ordered = []
    for group in GROUPS:
        for text in group:
            data = text.encode("ascii")
            if data in seen:
                continue
            if not 1 <= len(data) <= 255:
                raise ValueError(f"entry out of range: {text!r}")
            seen.add(data)
            ordered.append(data)
    return ordered[:MAX_ENTRIES]


def rust_byte(value):
    """Renders one byte as a Rust byte literal or hex escape."""
    if value == 0x5C:  # backslash
        return "b'\\\\'"
    if value == 0x27:  # single quote
        return "b'\\''"
    if 0x20 <= value < 0x7F:
        return "b'" + chr(value) + "'"
    return f"0x{value:02X}"


def render(entries):
    """Renders the table body with one entry per line.

    Each line carries its *successor's* label in a trailing comment, so
    scanning the table top-to-bottom also scans the dictionary it
    encodes; the final entry therefore needs none.
    """
    lines = [
        "    // Single characters, extensions, directories and file names.",
    ]
    for index, data in enumerate(entries):
        parts = [str(len(data))] + [rust_byte(b) for b in data]
        line = "    " + ", ".join(parts) + ","
        if index + 1 < len(entries):
            line += " // " + entries[index + 1].decode("ascii")
        # A label can be whitespace (` `), which would otherwise survive
        # as trailing whitespace and upset `cargo fmt --check`.
        lines.append(line.rstrip())
    return "\n".join(lines)


HEADER = """//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The filename codebook, generated by `tools/gen_codebook.py`.
//!
//! Do **not** edit this file by hand: change the word lists in the
//! generator and re-run it. Re-running is deterministic, so the
//! committed table always matches the script.
"""


def main():
    entries = collect()
    body = render(entries)
    source = (
        HEADER
        + "\n/// Number of entries packed into [`CODEBOOK`].\n"
        + f"pub const CODEBOOK_ENTRIES: usize = {len(entries)};\n\n"
        + "/// Flat, length-prefixed filename codebook.\n"
        + "///\n"
        + "/// Entry `n` starts at the `n`-th record: one length byte, then\n"
        + "/// that many name bytes. [`crate::compress::compress`] matches\n"
        + "/// against these records longest-first and\n"
        + "/// [`crate::compress::decompress`] indexes them by token byte.\n"
        + "pub const CODEBOOK: &[u8] = &[\n"
        + body
        + "\n];\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.join(root, "src", "codebook.rs")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="ascii") as handle:
        handle.write(source)
    print(f"wrote {target}: {len(entries)} entries, {len(body)} bytes of source")


if __name__ == "__main__":
    main()
