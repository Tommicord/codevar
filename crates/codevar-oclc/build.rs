//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
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
//! Every `*.rs` file under the crate's `kernels/` directory (override
//! with the `CV_KERNEL_DIR` environment variable) is compiled to
//! SPIR-V with the same pipeline the `codevar-oclc` driver runs for
//! `--emit spirv` (analyze → parse → lower → assemble), and the
//! result is embedded in the executable the way shaders are. The
//! script writes `OUT_DIR/<artifact>.spv` plus a copy of the source
//! as `OUT_DIR/<artifact>.rs`, then generates `OUT_DIR/bindings.rs`.
//! `src/lib.rs` includes that file under the `kernels` module.
//!
//! # Directory layout → module tree
//!
//! The directory structure below `kernels/` becomes a Rust module
//! tree, so every directory is a submodule and every file is one
//! group of statics inside its directory's module:
//!
//! ```text
//! kernels/
//! ├── truetype/raster.rs   →  kernels::truetype::RASTER_SPIRV
//! ├── wredit/
//! │   ├── block.rs         →  kernels::wredit::BLOCK_SPIRV
//! │   └── gap.rs           →  kernels::wredit::GAP_SPIRV
//! └── util.rs              →  kernels::UTIL_SPIRV
//! ```
//!
//! Each file contributes three statics to its module — `<STEM>_SPIRV`,
//! `<STEM>_SOURCE`, and `<STEM>_ENTRY_POINTS` — plus one
//! [`kernels::KERNEL_MODULES`] registry entry whose `name` is the
//! module-qualified source path (`wredit/block`).
//!
//! The `<artifact>` staged in `OUT_DIR` is the module-qualified
//! relative path with directory separators folded to `_`
//! (`wredit/block.rs` → `wredit_block.rs`), so files with equal
//! stems in different directories never overwrite each other.
//!
//! Directory names are turned into module identifiers by keeping
//! ASCII alphanumerics and `_` and replacing everything else with
//! `_`; a leading digit gains a prefix `_` and a Rust keyword gains a
//! trailing `_`, so the generated `mod` name is always a valid,
//! non-keyword identifier. Callers must still detect collisions,
//! because the sanitization is not injective.
//!
//! The script drives the compiler stage crates (`codevar-ocl-sar`,
//! `codevar-ocl-parse`, `codevar-ocl-ir`, `codevar-ocl-asm`)
//! directly: Cargo forbids a package's build script from depending on
//! the package itself, and the driver is a thin shell over exactly
//! those stages.
//!
//! Before any of that, [`codevar_rt::build::configure_link`] configures the
//! bare-metal link (a no-op on hosted targets): an image that cannot boot is
//! not worth compiling kernels for, so that failure comes first.
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
const KERNEL_EXTENSION: &str = "rs";

/// Prefix fallback for a file stem that sanitizes to nothing.
const FALLBACK_PREFIX: &str = "KERNEL";

/// Module-name fallback for a directory name that sanitizes to nothing.
const FALLBACK_MODULE: &str = "module";

/// Rust keywords a sanitized directory name must not collide with.
///
/// Includes the strict keywords of every edition plus the reserved
/// ones, so a directory called `mod/` or `try/` still yields a valid
/// module identifier (`mod_`, `try_`).
const RUST_KEYWORDS: &[&str] = &[
    "abstract",
    "as",
    "async",
    "await",
    "become",
    "box",
    "break",
    "const",
    "continue",
    "crate",
    "do",
    "dyn",
    "else",
    "enum",
    "extern",
    "false",
    "final",
    "fn",
    "for",
    "gen",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "macro",
    "macro_rules",
    "match",
    "mod",
    "move",
    "mut",
    "override",
    "priv",
    "pub",
    "raw",
    "ref",
    "return",
    "safe",
    "Self",
    "self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "try",
    "type",
    "typeof",
    "unsafe",
    "unsized",
    "use",
    "virtual",
    "where",
    "while",
    "yield",
];

/// A discovered kernel source file, validated before compilation.
#[derive(Debug, Clone)]
struct KernelSource {
    /// Path to the `.rs` file as found on disk.
    path: PathBuf,
    /// UTF-8 file name, e.g. `block.rs`.
    file_name: String,
    /// File stem, source of the static prefix, e.g. `block`.
    stem: String,
    /// Sanitized module path from `kernels/` to this file, e.g. `["wredit"]`.
    module_path: Vec<String>,
    /// Sanitized prefix of this file's statics inside its module, e.g. `BLOCK`.
    static_prefix: String,
    /// Artifact stem staged in `OUT_DIR`, e.g. `wredit_block`.
    artifact_stem: String,
}

impl KernelSource {
    /// `wredit/block` — the module-qualified source path used for
    /// diagnostics and for [`crate::kernels::KernelModule::name`].
    /// Root-level files yield just the stem.
    fn qualified_name(&self) -> String {
        let mut name = self.module_path.join("/");
        if !name.is_empty() {
            name.push('/');
        }
        name.push_str(&self.stem);
        name
    }

    /// `wredit::BLOCK` — the collision key of this file's statics.
    fn static_key(&self) -> String {
        let mut key = self.module_path.join("::");
        if !key.is_empty() {
            key.push_str("::");
        }
        key.push_str(&self.static_prefix);
        key
    }
}

/// A kernel source compiled to SPIR-V with its extracted entry points.
#[derive(Debug, Clone)]
struct CompiledKernel {
    /// The source file the artifact came from.
    source: KernelSource,
    /// Names of the `#[kernel]` functions it declares, in source order.
    entry_points: Vec<String>,
    /// The kernels that were also compiled to native machine code.
    native: Vec<NativeKernel>,
}

/// One `#[kernel]` compiled to native machine code.
#[derive(Debug, Clone)]
struct NativeKernel {
    /// The kernel's name, as it is declared in the source.
    name: String,
    /// The sanitized upper-case name of the generated static.
    static_suffix: String,
    /// The artifact stem staged in `OUT_DIR`.
    artifact_stem: String,
    /// The offset of the entry point inside the machine code.
    entry_offset: u32,
    /// The number of machine instructions the code holds.
    instruction_count: u32,
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
    /// Two files in one module sanitize to the same static prefix.
    DuplicatePrefix {
        /// The conflicting static key, e.g. `wredit::BLOCK`.
        prefix: String,
        /// First file that claimed the key.
        first: PathBuf,
        /// Second file that claimed the key.
        second: PathBuf,
    },
    /// Two kernel sources stage the same artifact name in `OUT_DIR`.
    DuplicateArtifact {
        /// The conflicting artifact stem, e.g. `wredit_block`.
        artifact: String,
        /// First file that claimed the artifact.
        first: PathBuf,
        /// Second file that claimed the artifact.
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
            Self::DuplicateArtifact {
                artifact,
                first,
                second,
            } => write!(
                f,
                "kernel sources '{}' and '{}' both stage the OUT_DIR artifact `{artifact}`; \
                 rename one of them",
                first.display(),
                second.display()
            ),
        }
    }
}

fn main() -> ExitCode {
    // Linker scripts first: a bare-metal image without them cannot boot, so
    // it is not worth spending a kernel compilation cycle to find out later.
    let link = codevar_rt::build::configure_link("codevar-oclc");
    if link != ExitCode::SUCCESS {
        return link;
    }
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
    let mut artifacts: BTreeMap<String, PathBuf> = BTreeMap::new();
    let mut compiled = Vec::with_capacity(sources.len());
    for source in sources {
        cargo_directive(&format!("cargo:rerun-if-changed={}", source.path.display()));
        let prefix = source.static_key();
        if let Some(first) = prefixes.insert(prefix.clone(), source.path.clone()) {
            return Err(BuildFailure::DuplicatePrefix {
                prefix,
                first,
                second: source.path,
            });
        }
        let artifact = source.artifact_stem.clone();
        if let Some(first) = artifacts.insert(artifact.clone(), source.path.clone()) {
            return Err(BuildFailure::DuplicateArtifact {
                artifact,
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

/// Collects the kernel sources under `kernel_dir`, sorted by path.
///
/// The tree is walked recursively: every directory becomes a `kernels`
/// submodule and every `.rs` file a set of statics inside it. A
/// directory that does not exist is treated as an empty one: the build
/// must not fail just because nobody has added kernels yet.
fn discover(kernel_dir: &Path) -> Result<Vec<KernelSource>, BuildFailure> {
    let mut sources = Vec::new();
    walk(kernel_dir, &[], &mut sources)?;
    sources.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(sources)
}

/// Walks one directory of the kernel tree, appending discovered sources.
///
/// Each visited directory registers a `rerun-if-changed` directive so
/// adding or removing a kernel file invalidates the build cache too.
fn walk(
    directory: &Path,
    module_path: &[String],
    sources: &mut Vec<KernelSource>,
) -> Result<(), BuildFailure> {
    cargo_directive(&format!("cargo:rerun-if-changed={}", directory.display()));
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(BuildFailure::ReadDir {
                path: directory.to_path_buf(),
                source,
            });
        }
    };
    for entry in entries {
        let entry = entry.map_err(|source| BuildFailure::ReadDir {
            path: directory.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path.is_dir() {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                cargo_warning(&format!(
                    "skipping kernel directory with a non-UTF-8 name: {}",
                    path.display()
                ));
                continue;
            };
            let mut child_path = module_path.to_vec();
            child_path.push(derive_module_ident(name));
            walk(path.as_path(), &child_path, sources)?;
            continue;
        }
        if !is_opencl_source(&path) {
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
            artifact_stem: derive_artifact_stem(module_path, &stem),
            module_path: module_path.to_vec(),
            path,
            file_name,
            stem,
        });
    }
    Ok(())
}

/// True when `path` has a `.rs` extension (case-insensitive).
fn is_opencl_source(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case(KERNEL_EXTENSION))
}

/// Derives the module identifier of a directory name.
///
/// Keeps ASCII alphanumerics and `_`, maps everything else to `_`,
/// prefixes a leading digit with `_`, and appends `_` to Rust
/// keywords, so the result is always a valid, non-keyword identifier
/// (empty input becomes [`FALLBACK_MODULE`]). The mapping is
/// non-injective, so callers must still detect collisions.
fn derive_module_ident(directory: &str) -> String {
    let mut name = String::with_capacity(directory.len());
    for character in directory.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            name.push(character);
        } else {
            name.push('_');
        }
    }
    if name.is_empty() {
        return String::from(FALLBACK_MODULE);
    }
    if name.starts_with(|character: char| character.is_ascii_digit()) {
        name.insert(0, '_');
    }
    if RUST_KEYWORDS.contains(&name.as_str()) {
        name.push('_');
    }
    name
}

/// Derives the shared static prefix of a file stem: ASCII upper-case, with
/// every non-alphanumeric character mapped to `_`.
///
/// The mapping keeps names unique for the common case (`block` →
/// `BLOCK`) and always yields a valid, non-keyword Rust identifier
/// fragment: the result is never empty (empty input becomes
/// [`FALLBACK_PREFIX`]), never starts with a digit (a `_` is prefixed),
/// and contains no lower-case letters, so `format!("{prefix}_SPIRV")`
/// can never spell a Rust keyword. Callers must still detect
/// collisions, because punctuation folding makes the mapping
/// non-injective.
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

/// Derives the `OUT_DIR` artifact stem of a file: its module path and
/// stem joined with `_` (`wredit` + `block` → `wredit_block`), so
/// same-stem files in different directories never overwrite each
/// other. Root-level files keep just their stem.
fn derive_artifact_stem(module_path: &[String], stem: &str) -> String {
    if module_path.is_empty() {
        return String::from(stem);
    }
    let mut artifact = module_path.join("_");
    artifact.push('_');
    artifact.push_str(stem);
    artifact
}

/// Runs the analysis → parse → lower → assemble pipeline for one source and
/// stages its `.rs` and `.spv` artifacts in `out_dir`.
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

    let source_copy = out_dir.join(artifact_file_name(&source.artifact_stem, KERNEL_EXTENSION));
    std::fs::write(&source_copy, text.as_bytes()).map_err(|error| BuildFailure::WriteFile {
        path: source_copy,
        source: error,
    })?;
    let spirv_copy = out_dir.join(artifact_file_name(&source.artifact_stem, "spv"));
    std::fs::write(&spirv_copy, &spirv).map_err(|error| BuildFailure::WriteFile {
        path: spirv_copy,
        source: error,
    })?;

    let native = compile_native(&module, &entry_points, &source, out_dir)?;

    Ok(CompiledKernel {
        source,
        entry_points,
        native,
    })
}

/// Compiles every kernel of `module` to native machine code and stages the
/// result in `out_dir`.
///
/// The machine code belongs to one architecture, so it is only produced
/// when the crate is built for the architecture the build script runs on;
/// a cross build keeps the SPIR-V path and says so with a warning.  A
/// kernel the native back end cannot express is skipped with a warning
/// as well: the kernel is valid, the back end is simply not there yet.
fn compile_native(
    module: &codevar_ocl_ir::ir::Module,
    entry_points: &[String],
    source: &KernelSource,
    out_dir: &Path,
) -> Result<Vec<NativeKernel>, BuildFailure> {
    if entry_points.is_empty() {
        return Ok(Vec::new());
    }
    if !native_target_matches_host() {
        cargo_warning(&format!(
            "{}: the native machine code is only built for the host architecture; \
             the SPIR-V image is embedded instead",
            source.file_name
        ));
        return Ok(Vec::new());
    }
    let target = match codevar_ocl_tc::host_target() {
        Ok(target) => target,
        Err(error) => {
            cargo_warning(&format!(
                "{}: no native back end for this architecture ({error}); \
                 the SPIR-V image is embedded instead",
                source.file_name
            ));
            return Ok(Vec::new());
        }
    };
    let mut compiled = Vec::with_capacity(entry_points.len());
    for name in entry_points {
        let code = match codevar_ocl_tc::compile_module_for(target.as_target(), module, name) {
            Ok(code) => code,
            Err(error) => {
                cargo_warning(&format!(
                    "{}: the kernel `{name}` has no native machine code ({error}); \
                     it is embedded as SPIR-V only",
                    source.file_name
                ));
                continue;
            }
        };
        let static_suffix = derive_static_prefix(name);
        let artifact_stem = format!("{}_{}", source.artifact_stem, name.to_lowercase());
        let path = out_dir.join(artifact_file_name(&artifact_stem, "bin"));
        std::fs::write(&path, code.bytes())
            .map_err(|error| BuildFailure::WriteFile { path, source: error })?;
        compiled.push(NativeKernel {
            name: name.clone(),
            static_suffix,
            artifact_stem,
            entry_offset: code.entry_offset(),
            instruction_count: code.instruction_count(),
        });
    }
    Ok(compiled)
}

/// True when the crate is built for the architecture the build script runs
/// on.
///
/// Cargo hands the script both triples: `HOST` is the machine the script
/// runs on and `TARGET` the machine the crate is built for.  The machine
/// code of a kernel belongs to one architecture, so it is only embedded
/// when the two agree.
fn native_target_matches_host() -> bool {
    fn architecture(triple: &str) -> &str {
        triple.split('-').next().unwrap_or(triple)
    }
    match (std::env::var("TARGET"), std::env::var("HOST")) {
        (Ok(target), Ok(host)) => architecture(&target) == architecture(&host),
        _ => false,
    }
}

/// The artifact file name staged in `OUT_DIR` for `stem` (`wredit_block`
/// → `wredit_block.rs`).
fn artifact_file_name(stem: &str, extension: &str) -> String {
    format!("{stem}.{extension}")
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

/// The generated statics of the kernel files inside one directory.
#[derive(Default)]
struct ModuleNode {
    /// One item per kernel file directly inside this directory, in
    /// discovery order.
    statics: Vec<TokenStream>,
    /// Child directories keyed by module identifier (sorted, so the
    /// generated module order is deterministic).
    children: BTreeMap<String, ModuleNode>,
}

impl ModuleNode {
    /// Files one directory deeper: recurses until `module_path` is
    /// empty, then stores the statics of one kernel file.
    fn insert(&mut self, module_path: &[String], statics: Vec<TokenStream>) {
        match module_path.split_first() {
            None => self.statics.extend(statics),
            Some((directory, rest)) => {
                self.children
                    .entry(directory.clone())
                    .or_default()
                    .insert(rest, statics);
            }
        }
    }

    /// Emits this directory's contents: its own statics first, then
    /// one `pub mod` item per subdirectory, recursively.
    fn items(&self) -> Vec<TokenStream> {
        let mut items = self.statics.clone();
        for (directory, child) in &self.children {
            let ident = Ident::new(directory, Span::call_site());
            let body = child.items();
            let doc = format!("Embedded kernels from the `{directory}/` directory.");
            items.push(quote! {
                #[doc = #doc]
                pub mod #ident {
                    #(#body)*
                }
            });
        }
        items
    }
}

/// Tokenizes every generated item: the module tree of statics plus the
/// registry, in discovery order.
fn generated_items(kernels: &[CompiledKernel]) -> Vec<TokenStream> {
    let mut root = ModuleNode::default();
    for kernel in kernels {
        let mut statics = vec![spirv_item(kernel), source_item(kernel), entry_points_item(kernel)];
        for native in &kernel.native {
            statics.push(native_item(kernel, native));
        }
        root.insert(&kernel.source.module_path, statics);
    }
    let mut items = root.items();
    items.push(registry_item(kernels));
    items
}

/// `<STEM>_SPIRV`: the embedded SPIR-V image of one kernel file.
fn spirv_item(kernel: &CompiledKernel) -> TokenStream {
    let ident = static_ident(&kernel.source.static_prefix, "SPIRV");
    let artifact = artifact_file_name(&kernel.source.artifact_stem, "spv");
    let doc = format!("Compiled SPIR-V image of `{}`.", kernel.source.qualified_name());
    quote! {
        #[doc = #doc]
        pub static #ident: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", #artifact));
    }
}

/// `<STEM>_SOURCE`: the embedded source text of one kernel file.
fn source_item(kernel: &CompiledKernel) -> TokenStream {
    let ident = static_ident(&kernel.source.static_prefix, "SOURCE");
    let artifact = artifact_file_name(&kernel.source.artifact_stem, KERNEL_EXTENSION);
    let doc = format!("Original source text of `{}`.", kernel.source.qualified_name());
    quote! {
        #[doc = #doc]
        pub static #ident: &str = include_str!(concat!(env!("OUT_DIR"), "/", #artifact));
    }
}

/// `<STEM>_ENTRY_POINTS`: the `#[kernel]` names declared by one kernel file.
fn entry_points_item(kernel: &CompiledKernel) -> TokenStream {
    let ident = static_ident(&kernel.source.static_prefix, "ENTRY_POINTS");
    let doc = format!(
        "`#[kernel]` entry points declared by `{}`.",
        kernel.source.qualified_name()
    );
    let names = &kernel.entry_points;
    quote! {
        #[doc = #doc]
        pub static #ident: &[&str] = &[#(#names),*];
    }
}

/// `<STEM>_<KERNEL>_NATIVE`: the machine code of one kernel.
fn native_item(kernel: &CompiledKernel, native: &NativeKernel) -> TokenStream {
    let ident = static_ident(
        &kernel.source.static_prefix,
        &format!("{}_NATIVE", native.static_suffix),
    );
    let artifact = artifact_file_name(&native.artifact_stem, "bin");
    let doc = format!(
        "Native machine code of the kernel `{}` of `{}`.",
        native.name,
        kernel.source.qualified_name()
    );
    quote! {
        #[doc = #doc]
        pub static #ident: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", #artifact));
    }
}

/// `KERNEL_MODULES`: every compiled kernel, in discovery order.
fn registry_item(kernels: &[CompiledKernel]) -> TokenStream {
    let entries = kernels.iter().map(|kernel| {
        let source_path = static_path(kernel, "SOURCE");
        let spirv_path = static_path(kernel, "SPIRV");
        let entry_points_path = static_path(kernel, "ENTRY_POINTS");
        let name = kernel.source.qualified_name();
        let native = native_registry(kernel);
        quote! {
            KernelModule {
                name: #name,
                source: #source_path,
                spirv: #spirv_path,
                entry_points: #entry_points_path,
                native: #native,
            }
        }
    });
    let doc = "Every compiled kernel module, ordered by source file name.";
    quote! {
        #[doc = #doc]
        pub static KERNEL_MODULES: &[KernelModule] = &[#(#entries),*];
    }
}

/// The `native` field of one kernel module's registry entry.
///
/// Every kernel that was compiled to machine code contributes an entry
/// with its name, its code, the offset of its entry point, and the
/// number of instructions it holds.
fn native_registry(kernel: &CompiledKernel) -> TokenStream {
    let entries = kernel.native.iter().map(|native| {
        let name = native.name.clone();
        let code = native_static_path(kernel, native);
        let entry_offset = native.entry_offset;
        let instruction_count = native.instruction_count;
        quote! {
            NativeKernel {
                name: #name,
                code: #code,
                entry_offset: #entry_offset,
                instruction_count: #instruction_count,
            }
        }
    });
    quote! { &[#(#entries),*] }
}

/// The path of one generated static from the `kernels` module root:
/// `wredit :: BLOCK_SOURCE` for a nested file, `UTIL_SOURCE` for a
/// root-level one.
fn static_path(kernel: &CompiledKernel, suffix: &str) -> TokenStream {
    let mut path = TokenStream::new();
    for segment in &kernel.source.module_path {
        let ident = Ident::new(segment, Span::call_site());
        path.extend(quote! { #ident :: });
    }
    let ident = static_ident(&kernel.source.static_prefix, suffix);
    path.extend(quote! { #ident });
    path
}

/// The path of the static holding one kernel's machine code.
fn native_static_path(kernel: &CompiledKernel, native: &NativeKernel) -> TokenStream {
    let mut path = TokenStream::new();
    for segment in &kernel.source.module_path {
        let ident = Ident::new(segment, Span::call_site());
        path.extend(quote! { #ident :: });
    }
    let ident = static_ident(
        &kernel.source.static_prefix,
        &format!("{}_NATIVE", native.static_suffix),
    );
    path.extend(quote! { #ident });
    path
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
