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

//! Build script for `codevar-kuitui`.
//!
//! Compiles every GLSL source under the crate's `shaders/` directory
//! (override with `CV_SHADER_DIR`) to SPIR-V with the external `glslc`
//! compiler and embeds the result the way `codevar-oclc` embeds its
//! kernels: each source stages `OUT_DIR/<artifact>.spv`, and the script
//! writes `OUT_DIR/shaders.rs`, which `src/lib.rs` includes as the
//! `shaders` module. A backend creates its UI pipelines from those
//! modules instead of shipping private copies of the same programs.
//!
//! # Naming
//!
//! A source is named `<name>.<stage>.glsl` (`ui.vert.glsl`,
//! `text.frag.glsl`); `<stage>` selects the `glslc` shader stage, so it
//! never has to be guessed. The artifact stem folds the dots to
//! underscores (`ui.vert` → `ui_vert`) and the generated static is
//! upper-cased (`ui.vert` → `UI_VERT_SPIRV`).
//!
//! # Failure policy
//!
//! A missing or empty `shaders/` directory is not an error: it produces
//! an empty `shaders` module, so the crate builds before anyone adds a
//! shader. When `glslc` is not on `PATH` the script warns, stages empty
//! `.spv` files so the generated module still compiles, and emits
//! `SHADERS_COMPILED = false`: a build host without the compiler keeps
//! working and a backend can detect the degraded mode. A shader that
//! `glslc` rejects, two sources that stage the same artifact, and an
//! unreadable directory all fail the build with a diagnostic, because
//! silently dropping a shader would surface later as a confusing
//! "empty module" at runtime.
//!
//! # Output
//!
//! `println!` is reserved for the `cargo:` build-script protocol on
//! stdout, which Cargo parses as metadata directives; it is not
//! application logging. Human-readable messages go through
//! `cargo:warning` directives.

use std::fmt;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// Environment variable overriding the default `shaders/` directory.
const SHADER_DIR_ENV: &str = "CV_SHADER_DIR";

/// Default directory, relative to the crate root, holding the GLSL sources.
const DEFAULT_SHADER_DIR: &str = "shaders";

/// Extension of the GLSL sources (matched case-insensitively).
const SHADER_EXTENSION: &str = "glsl";

/// Name of the external compiler invoked for every source.
const COMPILER: &str = "glslc";

/// `glslc` SPIR-V target environment: Vulkan 1.1 covers everything the
/// UI shaders use (push constants, a combined image sampler) and is
/// accepted by every device the backends target.
const TARGET_ENV: &str = "vulkan1.1";

/// A discovered GLSL source together with the names derived from it.
#[derive(Debug, Clone)]
struct ShaderSource {
    /// Path to the `.glsl` file as found on disk.
    path: PathBuf,
    /// Module-qualified name, e.g. `ui.vert` (file name without `.glsl`).
    name: String,
    /// Shader stage passed to `glslc`, e.g. `vert`.
    stage: String,
    /// `OUT_DIR` artifact stem, e.g. `ui_vert`.
    artifact: String,
    /// Generated static prefix, e.g. `UI_VERT`.
    prefix: String,
}

/// Why shader binding generation failed; every variant renders on one line.
#[derive(Debug)]
enum BuildFailure {
    /// A variable Cargo guarantees for build scripts was missing.
    Env {
        /// Name of the missing environment variable.
        name: &'static str,
    },
    /// The shader directory could not be listed.
    ReadDir {
        /// Directory that could not be read.
        path: PathBuf,
        /// Underlying I/O error.
        source: io::Error,
    },
    /// `glslc` could not be spawned.
    Spawn {
        /// Source that could not be compiled.
        path: PathBuf,
        /// Underlying I/O error.
        source: io::Error,
    },
    /// `glslc` rejected a source.
    Compile {
        /// Source that `glslc` rejected.
        path: PathBuf,
        /// Process exit code (`-1` when the process was terminated by a signal).
        status: i32,
        /// The compiler's standard error, trimmed.
        stderr: String,
    },
    /// Two sources stage the same `OUT_DIR` artifact.
    Duplicate {
        /// The conflicting artifact stem, e.g. `ui_vert`.
        artifact: String,
        /// First source that claimed the artifact.
        first: PathBuf,
        /// Second source that claimed the artifact.
        second: PathBuf,
    },
    /// An artifact or the generated bindings could not be written.
    WriteFile {
        /// File that could not be written.
        path: PathBuf,
        /// Underlying I/O error.
        source: io::Error,
    },
}

impl fmt::Display for BuildFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Env { name } => write!(f, "environment variable `{name}` is not set"),
            Self::ReadDir { path, source } => {
                write!(f, "cannot list '{}': {source}", path.display())
            }
            Self::Spawn { path, source } => {
                write!(f, "cannot run `{COMPILER}` on '{}': {source}", path.display())
            }
            Self::Compile { path, status, stderr } => write!(
                f,
                "`{COMPILER}` failed on '{}' (exit {status}): {stderr}",
                path.display()
            ),
            Self::Duplicate {
                artifact,
                first,
                second,
            } => write!(
                f,
                "shader sources '{}' and '{}' both stage `{artifact}.spv`; rename one",
                first.display(),
                second.display()
            ),
            Self::WriteFile { path, source } => {
                write!(f, "cannot write '{}': {source}", path.display())
            }
        }
    }
}

fn main() -> ExitCode {
    match generate() {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            cargo_warning(&format!("codevar-kuitui: shader build failed: {failure}"));
            ExitCode::FAILURE
        }
    }
}

/// Discovers and compiles every shader, then writes the `shaders` module.
fn generate() -> Result<(), BuildFailure> {
    let out_dir = required_env("OUT_DIR")?;
    let manifest_dir = required_env("CARGO_MANIFEST_DIR")?;
    let shader_dir = match std::env::var_os(SHADER_DIR_ENV) {
        Some(value) => PathBuf::from(value),
        None => manifest_dir.join(DEFAULT_SHADER_DIR),
    };
    cargo_directive(&format!("cargo:rerun-if-env-changed={SHADER_DIR_ENV}"));
    cargo_directive(&format!("cargo:rerun-if-changed={}", shader_dir.display()));

    let mut sources = Vec::new();
    walk(&shader_dir, &mut sources)?;
    sources.sort_by(|left, right| left.path.cmp(&right.path));

    let compiled = compiler_available();
    if !compiled {
        cargo_warning(&format!(
            "`{COMPILER}` was not found on PATH; embedding empty SPIR-V modules. \
             Install shaderc/glslc to compile the UI shaders."
        ));
    }

    let mut artifacts: std::collections::BTreeMap<String, PathBuf> = std::collections::BTreeMap::new();
    for source in &sources {
        cargo_directive(&format!("cargo:rerun-if-changed={}", source.path.display()));
        if let Some(first) = artifacts.insert(source.artifact.clone(), source.path.clone()) {
            return Err(BuildFailure::Duplicate {
                artifact: source.artifact.clone(),
                first,
                second: source.path.clone(),
            });
        }
        let out = out_dir.join(format!("{}.spv", source.artifact));
        if compiled {
            compile(source, &out)?;
        } else {
            empty(&out)?;
        }
    }

    write_bindings(&out_dir, &sources, compiled)
}

/// True when the external compiler can be spawned at all.
fn compiler_available() -> bool {
    Command::new(COMPILER)
        .arg("--version")
        .output()
        .is_ok()
}

/// Reads `name` from the environment, mapping its absence to a failure.
fn required_env(name: &'static str) -> Result<PathBuf, BuildFailure> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or(BuildFailure::Env { name })
}

/// Walks `directory` recursively, appending every GLSL source it holds.
///
/// A directory that does not exist is treated as empty: the build must
/// not fail just because nobody has added a shader yet.
fn walk(directory: &Path, sources: &mut Vec<ShaderSource>) -> Result<(), BuildFailure> {
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
            walk(path.as_path(), sources)?;
            continue;
        }
        if let Some(source) = shader_source(&path) {
            sources.push(source);
        }
    }
    Ok(())
}

/// Derives a [`ShaderSource`] from one path, or `None` when it is not a
/// `<name>.<stage>.glsl` file (the stage suffix is mandatory, so an
/// unlabelled `.glsl` is skipped rather than compiled at a guessed stage).
fn shader_source(path: &Path) -> Option<ShaderSource> {
    let file_name = path.file_name()?.to_str()?;
    if !file_name
        .to_ascii_lowercase()
        .ends_with(&format!(".{SHADER_EXTENSION}"))
    {
        return None;
    }
    // `ui.vert.glsl` → stem `ui.vert`, whose own extension is the stage.
    let stem = path.file_stem()?.to_str()?;
    let stage = Path::new(stem)
        .extension()?
        .to_str()?
        .to_ascii_lowercase();
    let artifact = sanitize_ident(stem);
    Some(ShaderSource {
        path: path.to_path_buf(),
        name: stem.into(),
        stage,
        prefix: artifact.to_ascii_uppercase(),
        artifact,
    })
}

/// Folds a file stem into an identifier: ASCII alphanumerics and `_`
/// survive, everything else becomes `_` (`ui.vert` → `ui_vert`). A
/// leading digit gains a `_` prefix so the result is a valid identifier.
fn sanitize_ident(stem: &str) -> String {
    let mut ident = String::with_capacity(stem.len());
    for character in stem.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            ident.push(character);
        } else {
            ident.push('_');
        }
    }
    if ident.is_empty() {
        ident.push_str("shader");
    }
    if ident.starts_with(|character: char| character.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
}

/// Compiles one source to `out` with `glslc`.
fn compile(source: &ShaderSource, out: &Path) -> Result<(), BuildFailure> {
    let output = Command::new(COMPILER)
        .arg(format!("--target-env={TARGET_ENV}"))
        .arg(format!("-fshader-stage={}", source.stage))
        .arg(&source.path)
        .arg("-o")
        .arg(out)
        .output()
        .map_err(|error| BuildFailure::Spawn {
            path: source.path.clone(),
            source: error,
        })?;
    if !output.status.success() {
        return Err(BuildFailure::Compile {
            path: source.path.clone(),
            status: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr)
                .trim()
                .into(),
        });
    }
    Ok(())
}

/// Stages an empty artifact, used when no compiler is available.
fn empty(path: &Path) -> Result<(), BuildFailure> {
    std::fs::write(path, b"").map_err(|source| BuildFailure::WriteFile {
        path: path.to_path_buf(),
        source,
    })
}

/// Writes `OUT_DIR/shaders.rs`: the `ShaderModule` type, the per-shader
/// SPIR-V statics and the `SHADERS` registry.
fn write_bindings(out_dir: &Path, sources: &[ShaderSource], compiled: bool) -> Result<(), BuildFailure> {
    let path = out_dir.join("shaders.rs");
    let file = std::fs::File::create(&path).map_err(|source| BuildFailure::WriteFile {
        path: path.clone(),
        source,
    })?;
    let mut writer = io::BufWriter::new(file);
    let write = |writer: &mut io::BufWriter<std::fs::File>, text: &str| {
        writer
            .write_all(text.as_bytes())
            .map_err(|source| BuildFailure::WriteFile {
                path: path.clone(),
                source,
            })
    };
    write(
        &mut writer,
        "/// One embedded SPIR-V shader stage.\n\
         #[derive(Debug, Clone, Copy)]\n\
         pub struct ShaderModule {\n\
         \x20   /// Module-qualified shader name (`ui.vert`).\n\
         \x20   pub name: &'static str,\n\
         \x20   /// Shader stage (`vert`, `frag`, `comp`).\n\
         \x20   pub stage: &'static str,\n\
         \x20   /// Compiled SPIR-V image, or empty when `glslc` was unavailable.\n\
         \x20   pub spirv: &'static [u8],\n\
         }\n\n",
    )?;
    write(
        &mut writer,
        &format!(
            "/// Whether the shaders were compiled to SPIR-V at build time.\n\
             pub const SHADERS_COMPILED: bool = {compiled};\n\n"
        ),
    )?;
    for source in sources {
        write(
            &mut writer,
            &format!(
                "/// Compiled SPIR-V image of `{}`.\n\
                 pub static {}_SPIRV: &[u8] = \
                 include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{}.spv\"));\n\n",
                source.name, source.prefix, source.artifact
            ),
        )?;
    }
    write(
        &mut writer,
        "/// Every embedded shader stage, sorted by name.\n\
         pub static SHADERS: &[ShaderModule] = &[\n",
    )?;
    for source in sources {
        write(
            &mut writer,
            &format!(
                "    ShaderModule {{ name: \"{}\", stage: \"{}\", spirv: {}_SPIRV }},\n",
                source.name, source.stage, source.prefix
            ),
        )?;
    }
    write(&mut writer, "];\n")?;
    writer
        .flush()
        .map_err(|source| BuildFailure::WriteFile { path, source })
}

/// Emits one `cargo:` metadata line on stdout, the build-script protocol.
fn cargo_directive(directive: &str) {
    println!("{directive}");
}

/// Reports `message` as a `cargo:warning`, one directive per line so a
/// multi-line message can never break the directive stream.
fn cargo_warning(message: &str) {
    for line in message.lines() {
        cargo_directive(&format!("cargo:warning={line}"));
    }
}
