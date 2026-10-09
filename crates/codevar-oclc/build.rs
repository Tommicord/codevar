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

//! Kernel binding generation for `codevar-oclc`.
//!
//! Every `*.cl` file in the crate's `kernels/` directory (override with the
//! `CV_KERNEL_DIR` environment variable) is compiled to SPIR-V with the
//! same pipeline the `codevar-oclc` driver runs for `--emit spirv`
//! (analyze → parse → lower → assemble), and the result is embedded in the
//! executable the way shaders are: the build script writes
//! `OUT_DIR/<name>.spv` plus a copy of the source as `OUT_DIR/<name>.cl`,
//! then generates `OUT_DIR/bindings.rs` holding one `static` per
//! kernel file and a `KERNEL_MODULES` registry. `src/lib.rs` includes that
//! file under the `kernels` module.
//!
//! The script drives the compiler stage crates (`codevar-ocl-sar`,
//! `codevar-ocl-parse`, `codevar-ocl-ir`, `codevar-ocl-asm`) directly:
//! Cargo forbids a package's build script from depending on the package
//! itself, and the driver is a thin shell over exactly those stages.
//!
//! # Failure policy
//!
//! A missing or empty kernel directory is not an error: it produces a
//! `cargo:warning` and an empty binding set, so the build keeps going.
//! Unreadable, non-UTF-8, uncompilable, or name-clashing kernel sources
//! fail the build with a diagnostic, because silently dropping one would
//! surface later as a confusing "static not found" error.
//!
//! # Output
//!
//! All `println!` use in this file is reserved for the `cargo:` build-script
//! protocol on stdout, which Cargo parses as metadata directives — it is not
//! application logging. Human-readable messages go through `cargo:warning`
//! directives (visible in Cargo's output) and `codevar_logger` (stderr).

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use codevar_ocl_sar::{ColorChoice, DeclKind, analyze, emit_stderr};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

/// Environment variable overriding the default `kernels/` source directory.
const KERNEL_DIR_ENV: &str = "CV_KERNEL_DIR";

/// File extension of OpenCL dialect sources (matched case-insensitively).
const KERNEL_EXTENSION: &str = "cl";

/// Prefix fallback for a file stem that sanitizes to nothing.
const FALLBACK_PREFIX: &str = "KERNEL";

/// A discovered kernel source file, validated before compilation.
#[derive(Debug, Clone)]
struct KernelSource {
    /// Path to the `.cl` file as found on disk.
    path: PathBuf,
    /// UTF-8 file name, e.g. `vector_add.cl`.
    file_name: String,
    /// File stem, used as the module name, e.g. `vector_add`.
    stem: String,
    /// Sanitized prefix shared by this file's generated statics, e.g. `VECTOR_ADD`.
    static_prefix: String,
}

/// A kernel source compiled to SPIR-V with its extracted entry points.
#[derive(Debug, Clone)]
struct CompiledKernel {
    /// The source file the artifact came from.
    source: KernelSource,
    /// Names of the `#[kernel]` functions it declares, in source order.
    entry_points: Vec<String>,
}

/// Why binding generation failed; every variant renders as a single line.
#[derive(Debug)]
enum BuildFailure {
    /// A variable Cargo guarantees for build scripts was missing.
    Env {
        /// Name of the missing environment variable.
        name: &'static str,
    },
    /// The kernel directory could not be listed.
    ReadDir {
        /// Directory that could not be read.
        path: PathBuf,
        /// Underlying I/O error.
        source: io::Error,
    },
    /// A kernel source could not be read.
    ReadFile {
        /// File that could not be read.
        path: PathBuf,
        /// Underlying I/O error.
        source: io::Error,
    },
    /// An artifact in `OUT_DIR` could not be written.
    WriteFile {
        /// File that could not be written.
        path: PathBuf,
        /// Underlying I/O error.
        source: io::Error,
    },
    /// A kernel source is not valid UTF-8.
    NotUtf8 {
        /// File that is not UTF-8 encoded.
        path: PathBuf,
    },
    /// Semantic analysis reported errors (already rendered to stderr).
    Analysis {
        /// Source that failed analysis.
        path: PathBuf,
    },
    /// The parser reported errors (already rendered as directives).
    Parse {
        /// Source that failed to parse.
        path: PathBuf,
        /// Number of parse errors.
        count: usize,
    },
    /// Lowering the analyzed program failed.
    Lower {
        /// Source that failed to lower.
        path: PathBuf,
        /// Lowering error message.
        message: String,
    },
    /// SPIR-V assembly failed.
    Assemble {
        /// Source that failed to assemble.
        path: PathBuf,
        /// Assembly error message.
        message: String,
    },
    /// Two file names sanitize to the same static prefix.
    DuplicatePrefix {
        /// The conflicting prefix.
        prefix: String,
        /// First file that claimed the prefix.
        first: PathBuf,
        /// Second file that claimed the prefix.
        second: PathBuf,
    },
}

impl fmt::Display for BuildFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Env { name } => write!(f, "environment variable `{name}` is not set"),
            Self::ReadDir { path, source } => write!(f, "cannot list '{}': {source}", path.display()),
            Self::ReadFile { path, source } => write!(f, "cannot read '{}': {source}", path.display()),
            Self::WriteFile { path, source } => {
                write!(f, "cannot write '{}': {source}", path.display())
            }
            Self::NotUtf8 { path } => {
                write!(
                    f,
                    "'{}' is not valid UTF-8; kernel sources must be UTF-8",
                    path.display()
                )
            }
            Self::Analysis { path } => {
                write!(f, "'{}' failed semantic analysis", path.display())
            }
            Self::Parse { path, count } => {
                write!(f, "'{}' failed to parse ({count} error(s))", path.display())
            }
            Self::Lower { path, message } => write!(f, "lowering '{}': {message}", path.display()),
            Self::Assemble { path, message } => {
                write!(f, "assembling '{}': {message}", path.display())
            }
            Self::DuplicatePrefix {
                prefix,
                first,
                second,
            } => write!(
                f,
                "kernel sources '{}' and '{}' both map to the static prefix `{prefix}`; \
                 rename one of them",
                first.display(),
                second.display()
            ),
        }
    }
}

fn main() -> ExitCode {
    match generate() {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            cargo_warning(&format!(
                "codevar-oclc: kernel binding generation failed: {failure}"
            ));
            codevar_logger::log_error!("kernel binding generation failed: {failure}");
            ExitCode::FAILURE
        }
    }
}

/// Discovers, compiles, and embeds every kernel source; see the module docs.
///
/// Returns the failure that should stop the build, or `Ok(())` when the
/// bindings file (possibly holding an empty registry) was written.
fn generate() -> Result<(), BuildFailure> {
    let out_dir = required_env("OUT_DIR")?;
    let manifest_dir = required_env("CARGO_MANIFEST_DIR")?;
    let kernel_dir = match std::env::var_os(KERNEL_DIR_ENV) {
        Some(value) => PathBuf::from(value),
        None => manifest_dir.join("kernels"),
    };

    cargo_directive(&format!("cargo:rerun-if-env-changed={KERNEL_DIR_ENV}"));
    cargo_directive(&format!("cargo:rerun-if-changed={}", kernel_dir.display()));

    let sources = discover(&kernel_dir)?;
    if sources.is_empty() {
        cargo_warning(&format!(
            "no `{KERNEL_EXTENSION}` kernel sources found in {}; emitting an empty `kernels` module",
            kernel_dir.display()
        ));
        return write_bindings(&out_dir, &[]);
    }

    let mut prefixes: BTreeMap<String, PathBuf> = BTreeMap::new();
    let mut compiled = Vec::with_capacity(sources.len());
    for source in sources {
        cargo_directive(&format!("cargo:rerun-if-changed={}", source.path.display()));
        if let Some(first) = prefixes.insert(source.static_prefix.clone(), source.path.clone()) {
            return Err(BuildFailure::DuplicatePrefix {
                prefix: source.static_prefix,
                first,
                second: source.path,
            });
        }
        compiled.push(compile(source, &out_dir)?);
    }
    write_bindings(&out_dir, &compiled)
}

/// Reads `name` from the environment, mapping its absence to a failure.
fn required_env(name: &'static str) -> Result<PathBuf, BuildFailure> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or(BuildFailure::Env { name })
}

/// Collects the kernel sources in `kernel_dir`, sorted by path.
///
/// A directory that does not exist is treated as an empty one: the build
/// must not fail just because nobody has added kernels yet.
fn discover(kernel_dir: &Path) -> Result<Vec<KernelSource>, BuildFailure> {
    let entries = match std::fs::read_dir(kernel_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(BuildFailure::ReadDir {
                path: kernel_dir.to_path_buf(),
                source,
            });
        }
    };

    let mut sources = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| BuildFailure::ReadDir {
            path: kernel_dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if !path.is_file() || !is_opencl_source(&path) {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            cargo_warning(&format!(
                "skipping kernel source with a non-UTF-8 file name: {}",
                path.display()
            ));
            continue;
        };
        let file_name = String::from(file_name);
        let stem = Path::new(&file_name)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map_or_else(|| file_name.clone(), String::from);
        sources.push(KernelSource {
            static_prefix: derive_static_prefix(&stem),
            path,
            file_name,
            stem,
        });
    }
    sources.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(sources)
}

/// True when `path` has a `.cl` extension (case-insensitive).
fn is_opencl_source(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case(KERNEL_EXTENSION))
}

/// Derives the shared static prefix of a file stem: ASCII upper-case, with
/// every non-alphanumeric character mapped to `_`.
///
/// The mapping keeps names unique for the common case (`vector_add` →
/// `VECTOR_ADD`) and always yields a valid, non-keyword Rust identifier
/// fragment: the result is never empty (empty input becomes
/// [`FALLBACK_PREFIX`]), never starts with a digit (a `_` is prefixed), and
/// contains no lower-case letters, so `format!("{prefix}_SPIRV")` can never
/// spell a Rust keyword. Callers must still detect collisions, because
/// punctuation folding makes the mapping non-injective.
fn derive_static_prefix(stem: &str) -> String {
    let mut prefix = String::with_capacity(stem.len());
    for character in stem.chars() {
        if character.is_ascii_alphanumeric() {
            prefix.push(character.to_ascii_uppercase());
        } else {
            prefix.push('_');
        }
    }
    if prefix.is_empty() {
        return String::from(FALLBACK_PREFIX);
    }
    if prefix.starts_with(|character: char| character.is_ascii_digit()) {
        prefix.insert(0, '_');
    }
    prefix
}

/// Runs the analyze → parse → lower → assemble pipeline for one source and
/// stages its `.cl` and `.spv` artifacts in `out_dir`.
///
/// Mirrors the `--emit spirv` path of the `codevar-oclc` driver: analysis
/// diagnostics are rendered to stderr before the failure is returned, and
/// entry-point names are taken from the analyzer's declarations.
fn compile(source: KernelSource, out_dir: &Path) -> Result<CompiledKernel, BuildFailure> {
    let display = source.path.display().to_string();
    let bytes = std::fs::read(&source.path).map_err(|error| BuildFailure::ReadFile {
        path: source.path.clone(),
        source: error,
    })?;
    let text = String::from_utf8(bytes).map_err(|_| BuildFailure::NotUtf8 {
        path: source.path.clone(),
    })?;

    let analyzed = analyze(&text);
    emit_stderr(&analyzed.diagnostics, &display, &text, ColorChoice::Auto);
    if analyzed.has_errors() {
        return Err(BuildFailure::Analysis {
            path: source.path.clone(),
        });
    }
    let parsed = codevar_ocl_parse::parse(&text);
    if !parsed.errors.is_empty() {
        for error in &parsed.errors {
            let (line, column) = line_column(&text, error.span.offset);
            cargo_warning(&format!("{display}:{line}:{column}: error: {}", error.message));
        }
        return Err(BuildFailure::Parse {
            path: source.path.clone(),
            count: parsed.errors.len(),
        });
    }

    let module =
        codevar_ocl_ir::lower::lower(&parsed.program, &analyzed).map_err(|error| BuildFailure::Lower {
            path: source.path.clone(),
            message: error.to_string(),
        })?;
    let spirv = codevar_ocl_asm::assemble_bytes(&module).map_err(|error| BuildFailure::Assemble {
        path: source.path.clone(),
        message: error.to_string(),
    })?;
    let entry_points = analyzed
        .declarations
        .iter()
        .filter_map(|declaration| match &declaration.kind {
            DeclKind::Function { kernel: true, .. } => Some(declaration.name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if entry_points.is_empty() {
        cargo_warning(&format!(
            "{} declares no `#[kernel]` entry points; embedding it anyway",
            source.file_name
        ));
    }

    let source_copy = out_dir.join(&source.file_name);
    std::fs::write(&source_copy, text.as_bytes()).map_err(|error| BuildFailure::WriteFile {
        path: source_copy,
        source: error,
    })?;
    let spirv_copy = out_dir.join(spirv_file_name(&source.file_name));
    std::fs::write(&spirv_copy, &spirv).map_err(|error| BuildFailure::WriteFile {
        path: spirv_copy,
        source: error,
    })?;

    Ok(CompiledKernel { source, entry_points })
}

/// `vector_add.cl` → `vector_add.spv`, the artifact name staged in `OUT_DIR`.
fn spirv_file_name(file_name: &str) -> PathBuf {
    Path::new(file_name).with_extension("spv")
}

/// Writes `bindings.rs` into `out_dir`, one item per line.
///
/// The item tokens come from [`quote!`], so no Rust code is ever assembled
/// by hand here; `writeln!` only lays the already-tokenized items out.
fn write_bindings(out_dir: &Path, kernels: &[CompiledKernel]) -> Result<(), BuildFailure> {
    let path = out_dir.join("bindings.rs");
    let file = std::fs::File::create(&path).map_err(|source| BuildFailure::WriteFile {
        path: path.clone(),
        source,
    })?;
    let mut writer = io::BufWriter::new(file);
    for item in generated_items(kernels) {
        writeln!(writer, "{item}").map_err(|source| BuildFailure::WriteFile {
            path: path.clone(),
            source,
        })?;
    }
    writer
        .flush()
        .map_err(|source| BuildFailure::WriteFile { path, source })
}

/// Tokenizes every generated item: three statics per kernel plus the
/// registry, in discovery order.
fn generated_items(kernels: &[CompiledKernel]) -> Vec<TokenStream> {
    let mut items = Vec::with_capacity(kernels.len() * 3 + 1);
    for kernel in kernels {
        items.push(spirv_item(kernel));
        items.push(source_item(kernel));
        items.push(entry_points_item(kernel));
    }
    items.push(registry_item(kernels));
    items
}

/// `<PREFIX>_SPIRV`: the embedded SPIR-V image of one kernel file.
fn spirv_item(kernel: &CompiledKernel) -> TokenStream {
    let ident = static_ident(&kernel.source.static_prefix, "SPIRV");
    let path = spirv_file_name(&kernel.source.file_name)
        .to_string_lossy()
        .into_owned();
    let doc = format!("Compiled SPIR-V image of `{}`.", kernel.source.file_name);
    quote! {
        #[doc = #doc]
        pub static #ident: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", #path));
    }
}

/// `<PREFIX>_SOURCE`: the embedded source text of one kernel file.
fn source_item(kernel: &CompiledKernel) -> TokenStream {
    let ident = static_ident(&kernel.source.static_prefix, "SOURCE");
    let file_name = &kernel.source.file_name;
    let doc = format!("Original source text of `{file_name}`.");
    quote! {
        #[doc = #doc]
        pub static #ident: &str = include_str!(concat!(env!("OUT_DIR"), "/", #file_name));
    }
}

/// `<PREFIX>_ENTRY_POINTS`: the `#[kernel]` names declared by one kernel file.
fn entry_points_item(kernel: &CompiledKernel) -> TokenStream {
    let ident = static_ident(&kernel.source.static_prefix, "ENTRY_POINTS");
    let file_name = &kernel.source.file_name;
    let names = &kernel.entry_points;
    let doc = format!("`#[kernel]` entry points declared by `{file_name}`.");
    quote! {
        #[doc = #doc]
        pub static #ident: &[&str] = &[#(#names),*];
    }
}

/// `KERNEL_MODULES`: every compiled kernel, in discovery order.
fn registry_item(kernels: &[CompiledKernel]) -> TokenStream {
    let entries = kernels.iter().map(|kernel| {
        let source_ident = static_ident(&kernel.source.static_prefix, "SOURCE");
        let spirv_ident = static_ident(&kernel.source.static_prefix, "SPIRV");
        let entry_points_ident = static_ident(&kernel.source.static_prefix, "ENTRY_POINTS");
        let name = &kernel.source.stem;
        quote! {
            KernelModule {
                name: #name,
                source: #source_ident,
                spirv: #spirv_ident,
                entry_points: #entry_points_ident,
            }
        }
    });
    let doc = "Every compiled kernel module, ordered by source file name.";
    quote! {
        #[doc = #doc]
        pub static KERNEL_MODULES: &[KernelModule] = &[#(#entries),*];
    }
}

/// Builds one generated static's identifier from a sanitized prefix.
///
/// # Panics
///
/// `proc_macro2::Ident::new` rejects keywords and invalid identifiers;
/// [`derive_static_prefix`] guarantees an upper-case, non-empty, digit-safe
/// prefix, and the suffixes are literals, so the constructed name is always
/// valid and `Ident::new` never panics here.
fn static_ident(prefix: &str, suffix: &str) -> Ident {
    Ident::new(&format!("{prefix}_{suffix}"), Span::call_site())
}

/// Computes the 1-based line and column of byte `offset`.
///
/// An `offset` inside a multi-byte character moves back to the nearest
/// character boundary so the slice below can never panic.
fn line_column(source: &str, offset: u32) -> (u32, u32) {
    let mut offset = (offset as usize).min(source.len());
    while offset > 0 && !source.is_char_boundary(offset) {
        offset -= 1;
    }
    let prefix = &source[..offset];
    let line = prefix
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1;
    let column = match prefix.rfind('\n') {
        Some(position) => prefix[position + 1..].chars().count() + 1,
        None => prefix.chars().count() + 1,
    };
    (line as u32, column as u32)
}

/// Emits one `cargo:` metadata line on stdout, the build-script protocol.
///
/// `println!` is mandatory here — Cargo parses the script's stdout for
/// directives — and is deliberately kept out of the logging paths described
/// in the module docs.
fn cargo_directive(directive: &str) {
    println!("{directive}");
}

/// Reports `message` as a `cargo:warning`, one directive per line so that
/// multi-line messages can never break the directive stream.
fn cargo_warning(message: &str) {
    for line in message.lines() {
        cargo_directive(&format!("cargo:warning={line}"));
    }
}
