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

//! Compiled programs.
//!
//! A [`Program`] is created from Codevar OpenCL dialect source, which
//! this crate compiles to SPIR-V in-process with the same front end the
//! `codevar-oclc` driver uses, or from a ready-made IL blob with
//! [`Program::from_il`], and is built for the context's device with
//! [`Program::build`]. Source and build options are owned copies; on
//! failure the driver's build log — or the front end's rendered
//! diagnostics for [`Program::from_sources`] — is returned inside
//! [`Error::BuildFailed`](crate::Error::BuildFailed).

use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::fmt;

use codevar_logger::{log_raw, log_warn};
use codevar_ocl_asm::assemble_bytes;
use codevar_ocl_ir::lower::lower;
use codevar_ocl_parse::parse;
use codevar_ocl_sar::{ColorChoice, Diagnostic, analyze, render};

use crate::api::Api;
use crate::context::Context;
use crate::error::{Error, Result, creation, status};
use crate::kernel::Kernel;
use crate::query;
use crate::sys;

/// A program compiled for one context's device.
#[derive(Clone)]
pub struct Program {
    inner: Arc<ProgramInner>,
}

struct ProgramInner {
    api: Arc<Api>,
    context: Context,
    raw: sys::ProgramHandle,
}

impl Drop for ProgramInner {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by a `clCreateProgramWith*`
        // entry point through this same `api`, and this `Drop` runs
        // exactly once for the single shared `ProgramInner`.
        let code = unsafe { (self.api.release_program)(self.raw) };
        if code != sys::SUCCESS {
            log_warn!("clReleaseProgram failed with {}", sys::error_name(code));
        }
    }
}

// SAFETY: `ProgramInner` is an opaque driver handle plus shared
// function pointers; programs are reference-counted by the driver and
// all entry points used here are thread-safe per the specification.
unsafe impl Send for ProgramInner {}
// SAFETY: see the `Send` implementation.
unsafe impl Sync for ProgramInner {}

impl Program {
    /// Creates a program for `context` from one or more source units of
    /// the Codevar OpenCL dialect.
    ///
    /// The units are concatenated with newlines and compiled to SPIR-V
    /// in-process through the same pipeline as the `codevar-oclc`
    /// driver's `--emit spirv` stage (analysis → parse → lowering →
    /// assembly); the resulting module reaches the driver through
    /// `clCreateProgramWithIL`. Every diagnostic the front end reports —
    /// errors *and* warnings — is printed to the console with
    /// [`codevar_logger::log_raw`] as soon as it exists. Like every
    /// other program, the result must be built with [`Program::build`]
    /// before kernels can be created from it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `sources` is empty,
    /// [`Error::BuildFailed`] when the dialect compiler rejects the
    /// source (the rendered diagnostics travel in the error's `log`),
    /// [`Error::Unsupported`] when the driver does not export
    /// `clCreateProgramWithIL`, and [`Error::Status`] /
    /// [`Error::NullHandle`] on driver failures.
    pub fn from_sources<'a, I>(context: &Context, sources: I) -> Result<Self>
    where
        I: IntoIterator<Item = &'a str>,
    {
        Self::from_il(context, &compile_units(sources)?)
    }

    /// Creates a program from a single Codevar OpenCL dialect source
    /// string (see [`Program::from_sources`]).
    ///
    /// # Errors
    ///
    /// Returns [`Error::BuildFailed`] when the dialect compiler rejects
    /// `source`, and [`Error::Unsupported`] / [`Error::Status`] /
    /// [`Error::NullHandle`] on driver failures.
    pub fn from_source(context: &Context, source: &str) -> Result<Self> {
        Self::from_sources(context, core::iter::once(source))
    }

    /// Creates a program from an intermediate-language (IL) blob, usually
    /// SPIR-V, via `clCreateProgramWithIL`.
    ///
    /// `clCreateProgramWithIL` is core since OpenCL 2.1; OpenCL 1.2
    /// implementations may expose it as the `cl_khr_il_program` extension
    /// (`clCreateProgramWithILKHR` — drivers advertising the extension
    /// also export the core symbol name). Like source programs, the
    /// result must be built with [`Program::build`] before kernels can be
    /// created from it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `il` is empty,
    /// [`Error::Unsupported`] when the driver does not export the entry
    /// point, and [`Error::Status`] / [`Error::NullHandle`] when the
    /// driver rejects the blob (a malformed or device-incompatible IL).
    pub fn from_il(context: &Context, il: &[u8]) -> Result<Self> {
        if il.is_empty() {
            return Err(Error::InvalidArgument {
                what: "an IL blob must not be empty",
            });
        }
        let create = Api::optional(context.api().create_program_with_il, "clCreateProgramWithIL")?;
        let api = context.api().clone();
        let mut errcode = sys::SUCCESS;
        // SAFETY: `context` is a live handle of `api`; `il` points at
        // `il.len()` initialized bytes that the driver only reads during
        // the call; `errcode` is a valid out-pointer.
        let raw = unsafe { create(context.raw(), il.as_ptr().cast(), il.len(), &mut errcode) };
        let raw = creation(raw, errcode, "clCreateProgramWithIL")?;
        Ok(Self {
            inner: Arc::new(ProgramInner {
                api,
                context: context.clone(),
                raw,
            }),
        })
    }

    /// Builds the program for the context's device.
    ///
    /// `options` is passed to the driver verbatim (for example
    /// `"-cl-std=CL3.0"` or `"-cl-fast-relaxed-math"`); an empty string
    /// requests the defaults.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BuildFailed`] carrying the driver's build log
    /// when compilation fails, and [`Error::Status`] when the driver
    /// rejects the request outright.
    pub fn build(&self, options: &str) -> Result<()> {
        if options.bytes().any(|byte| byte == 0) {
            return Err(Error::InvalidArgument {
                what: "build options must not contain NUL bytes",
            });
        }
        let mut c_options = options.as_bytes().to_vec();
        c_options.push(0);
        let device = self.inner.context.device().raw();
        // SAFETY: `raw`/`device` are live handles of the same `api`;
        // `c_options` is a valid NUL-terminated string that is only read
        // during the call and `None` selects synchronous building.
        let code = unsafe {
            (self.inner.api.build_program)(
                self.inner.raw,
                1,
                &device,
                core::ffi::CStr::from_bytes_with_nul_unchecked(&c_options).as_ptr(),
                None,
                core::ptr::null_mut(),
            )
        };
        if code == sys::SUCCESS {
            return Ok(());
        }
        let log = self.build_log().unwrap_or_default();
        if !log.is_empty() {
            return Err(Error::BuildFailed { log });
        }
        status(code, "clBuildProgram")
    }

    /// Returns the device build log of this program.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] when the query fails.
    pub fn build_log(&self) -> Result<alloc::string::String> {
        let mut call = |size: usize, value: *mut c_void, size_ret: *mut usize| {
            // SAFETY: `raw` is a live handle of the driver that
            // resolved `api`, and `query` provides valid buffers.
            unsafe {
                (self.inner.api.get_program_build_info)(
                    self.inner.raw,
                    self.inner.context.device().raw(),
                    sys::PROGRAM_BUILD_LOG,
                    size,
                    value,
                    size_ret,
                )
            }
        };
        query::query_string(&mut call, "clGetProgramBuildInfo")
    }

    /// Returns the context the program was created for.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.inner.context
    }

    /// Creates a kernel entry point from this built program.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Status`] / [`Error::NullHandle`] when the
    /// program has no entry point with `name`.
    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        Kernel::new(self, name)
    }

    /// Returns the resolved entry points of the owning driver.
    pub(crate) fn api(&self) -> &Arc<Api> {
        &self.inner.api
    }

    /// Returns the raw driver handle.
    pub(crate) fn raw(&self) -> sys::ProgramHandle {
        self.inner.raw
    }
}

/// Pseudo-file name shown by rendered diagnostics; the sources handed to
/// [`Program::from_sources`] are in-memory strings, not files.
const DIAGNOSTIC_PATH: &str = "<opencl>";

/// Compiles every source unit of the Codevar OpenCL dialect into a
/// SPIR-V module.
///
/// The units are joined with newlines — the concatenation
/// `clCreateProgramWithSource` performs for multi-unit programs — and
/// handed to [`compile_dialect`].
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] when `sources` yields no unit,
/// and whatever [`compile_dialect`] reports for the joined source.
fn compile_units<'a, I>(sources: I) -> Result<Vec<u8>>
where
    I: IntoIterator<Item = &'a str>,
{
    let units: Vec<&str> = sources.into_iter().collect();
    if units.is_empty() {
        return Err(Error::InvalidArgument {
            what: "at least one source unit is required",
        });
    }
    compile_dialect(&units.join("\n"))
}

/// Compiles one Codevar OpenCL dialect source into a SPIR-V module.
///
/// This mirrors the `--emit spirv` pipeline of the `codevar-oclc`
/// driver: [`analyze`], [`parse`], [`lower`], then [`assemble_bytes`].
/// Every diagnostic the front end produces — errors *and* warnings — is
/// written to the console through [`log_raw`] as soon as it exists, and
/// the same rendered text becomes the [`Error::BuildFailed`] log, so
/// callers that never inspect the console still see the diagnostics.
///
/// # Errors
///
/// Returns [`Error::BuildFailed`] when analysis reports an error, the
/// source does not parse, lowering rejects a construct, or the SPIR-V
/// backend cannot assemble the module.
fn compile_dialect(source: &str) -> Result<Vec<u8>> {
    let analyzed = analyze(source);
    let diagnostics = render(&analyzed.diagnostics, DIAGNOSTIC_PATH, source, ColorChoice::Never);
    if !diagnostics.is_empty() {
        let _ = log_raw(diagnostics.as_bytes());
    }
    if analyzed.has_errors() {
        return Err(Error::BuildFailed { log: diagnostics });
    }
    let parsed = parse(source);
    if !parsed.errors.is_empty() {
        // `analyze` already surfaces syntax errors as diagnostics, so
        // this only fires if the two runs ever disagree; the AST must
        // not reach lowering unreported either way.
        let errors: Vec<Diagnostic> = parsed
            .errors
            .iter()
            .map(|error| Diagnostic::error(error.span, error.message.clone()))
            .collect();
        let log = render(&errors, DIAGNOSTIC_PATH, source, ColorChoice::Never);
        let _ = log_raw(log.as_bytes());
        return Err(Error::BuildFailed { log });
    }
    let module = lower(&parsed.program, &analyzed).map_err(|error| {
        let log = format!("lowering failed: {error}");
        let _ = log_raw(log.as_bytes());
        Error::BuildFailed { log }
    })?;
    assemble_bytes(&module).map_err(|error| {
        let log = format!("assembly failed: {error}");
        let _ = log_raw(log.as_bytes());
        Error::BuildFailed { log }
    })
}

impl fmt::Debug for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Program")
            .field("handle", &self.inner.raw)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dialect form of the kernel the end-to-end
    /// `tests/vector_add.rs` test runs: keeping it here proves the
    /// shipped kernel compiles even on hosts without an OpenCL device.
    const KERNEL: &str = "\
#[kernel]
fn vector_add(a: *const int, b: *const int, out: *mut int, n: int) -> void {
    let i = get_global_id(0);
    if i < n {
        out[i] = a[i] + b[i];
    }
}
";

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn program_is_send_and_sync() {
        _assert_send_sync::<Program>();
    }

    /// An empty iterator is rejected before any compilation starts.
    #[test]
    fn empty_sources_are_rejected() {
        let error = compile_units(core::iter::empty::<&str>()).err();
        assert!(
            matches!(error, Some(Error::InvalidArgument { .. })),
            "expected Error::InvalidArgument, got {error:?}"
        );
    }

    /// A well-formed kernel must survive the whole pipeline and come
    /// out as a SPIR-V module (magic word included).
    #[test]
    fn dialect_kernel_compiles_to_a_spirv_module() {
        let bytes = compile_units(core::iter::once(KERNEL)).expect("the dialect kernel compiles");
        assert!(bytes.len() >= 20, "byte length: {}", bytes.len());
        assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07], "SPIR-V magic");
    }

    /// Units are joined with a newline: splitting an identifier across
    /// the boundary must not silently rejoin it into one valid kernel.
    #[test]
    fn source_units_are_separated_by_a_newline() {
        // Without the joining newline this would be `fn abc(...)`, a
        // perfectly compilable kernel.
        let error = compile_units(["#[kernel] fn ab", "c(n: int) -> void {}"]).err();
        assert!(
            matches!(error, Some(Error::BuildFailed { .. })),
            "expected Error::BuildFailed, got {error:?}"
        );
    }

    /// A rejected source reports `BuildFailed`, carrying the rendered
    /// diagnostics.
    #[test]
    fn broken_source_reports_build_failed_with_diagnostics() {
        let error = compile_units(core::iter::once("#[kernel]\nfn broken(n: noway) { }")).err();
        match error {
            Some(Error::BuildFailed { log }) => {
                assert!(log.contains("error"), "unexpected log: {log}");
                assert!(log.contains("noway"), "unexpected log: {log}");
            }
            other => panic!("expected Error::BuildFailed, got {other:?}"),
        }
    }

    /// Everything `log_raw` writes, captured for assertions.
    ///
    /// Installed once for this test binary: later tests keep writing
    /// into the buffer, which they never inspect.
    struct Capture(std::sync::Mutex<alloc::string::String>);

    impl codevar_logger::LogWriter for Capture {
        fn write_stdout(&self, bytes: &[u8]) -> core::result::Result<(), codevar_logger::LogError> {
            if let Ok(text) = core::str::from_utf8(bytes) {
                self.0.lock().unwrap().push_str(text);
            }
            Ok(())
        }

        fn write_stderr(&self, bytes: &[u8]) -> core::result::Result<(), codevar_logger::LogError> {
            self.write_stdout(bytes)
        }

        fn flush(&self) -> core::result::Result<(), codevar_logger::LogError> {
            Ok(())
        }
    }

    static CAPTURE: Capture = Capture(std::sync::Mutex::new(alloc::string::String::new()));

    /// Warnings reach the console through `log_raw` but never fail the
    /// compilation.
    #[test]
    fn warnings_are_logged_but_do_not_fail_compilation() {
        codevar_logger::set_log_writer(&CAPTURE);
        let kernel = "#[kernel]\nfn silent(n: int) -> void { let warn_probe_var = n; }";
        let bytes = compile_units(core::iter::once(kernel)).expect("a warning must not fail the compilation");
        assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07], "SPIR-V magic");
        let logged = CAPTURE.0.lock().unwrap();
        assert!(logged.contains("warn_probe_var"), "logged: {logged}");
    }
}
