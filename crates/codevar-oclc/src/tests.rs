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

//! Driver-level tests: argument sourcing, response files, the emission
//! pipeline, and diagnostic placement.
//!
//! `unwrap` appears only on inputs this file itself constructs (known-good
//! strings and files), which is the sanctioned unit-test exception.

use crate::driver::{expand_response_files, lex_source, line_column, run};
use crate::{Exit, fs};
use alloc::string::String;
use alloc::vec::Vec;

/// Builds a full command line with `argv[0]` in place.
fn args(extra: &[&str]) -> Vec<String> {
    core::iter::once(String::from("codevar-oclc"))
        .chain(
            extra
                .iter()
                .map(|argument| String::from(*argument)),
        )
        .collect()
}

/// A unique path in the system temp directory for this test's tag.
fn temp_path(tag: &str) -> String {
    let path = std::env::temp_dir().join(format!("codevar-oclc-{}-{tag}", std::process::id()));
    path.to_string_lossy().into_owned()
}

/// Removes a temp file if it exists; cleanup failures are irrelevant here.
fn cleanup(path: &str) {
    let _ = std::fs::remove_file(path);
}

#[test]
fn exit_codes_are_stable() {
    assert_eq!(Exit::Success.code(), 0);
    assert_eq!(Exit::Failure.code(), 1);
    assert_eq!(Exit::Usage.code(), 2);
}

#[test]
fn run_requires_input() {
    assert_eq!(run(&args(&[])), Exit::Usage);
}

#[test]
fn run_help_succeeds() {
    assert_eq!(run(&args(&["--help"])), Exit::Success);
    assert_eq!(run(&args(&["-V"])), Exit::Success);
}

#[test]
fn run_unknown_option_is_usage() {
    assert_eq!(run(&args(&["--bogus", "in.cl"])), Exit::Usage);
    assert_eq!(run(&args(&["-z", "in.cl"])), Exit::Usage);
}

#[test]
fn run_rejects_unknown_emit() {
    assert_eq!(run(&args(&["--emit", "wat", "in.cl"])), Exit::Usage);
}

#[test]
fn run_fails_on_missing_input_file() {
    let args = args(&["/nonexistent/codevar-oclc-input.cl"]);
    assert_eq!(run(&args), Exit::Failure);
}

#[test]
fn run_fails_on_unimplemented_stage() {
    let args = args(&["--emit", "ptx", "/nonexistent/codevar-oclc-input.cl"]);
    assert_eq!(run(&args), Exit::Failure);
}

/// `unwrap` is safe: the path and kernel source are constructed here.
#[test]
fn spirv_stage_writes_binary_module() {
    let input = temp_path("spirv-in.cl");
    let output = temp_path("spirv-out.spv");
    let kernel = "\
#[kernel]
fn vector_add(a: *const float, b: *const float, c: *mut float, n: int) -> void {
    let i = get_global_id(0);
    if i < n {
        c[i] = a[i] + b[i];
    }
}
";
    fs::write_file(&input, kernel).unwrap();
    cleanup(&output);

    let args = args(&["--emit", "spirv", "-o", &output, &input]);
    let exit = run(&args);
    let bytes = std::fs::read(&output).unwrap_or_default();
    cleanup(&input);
    cleanup(&output);

    assert_eq!(exit, Exit::Success);
    assert!(bytes.len() >= 20, "byte length: {}", bytes.len());
    assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07], "SPIR-V magic");
}

/// `unwrap` is safe: the path and kernel source are constructed here.
#[test]
fn spirv_stage_reports_analysis_errors() {
    let input = temp_path("spirv-bad.cl");
    fs::write_file(&input, "#[kernel]\nfn broken(n: noway) { }\n").unwrap();

    let args = args(&["--emit", "spirv", &input]);
    let exit = run(&args);
    cleanup(&input);

    assert_eq!(exit, Exit::Failure);
}

/// `unwrap` is safe: every path in this test is constructed here.
#[test]
fn tokens_stage_writes_output_file() {
    let input = temp_path("tokens-in.cl");
    let output = temp_path("tokens-out.txt");
    fs::write_file(&input, "let x = 1;\n").unwrap();
    cleanup(&output);

    let args = args(&["-o", &output, &input]);
    let exit = run(&args);
    let dump = fs::read_file(&output).unwrap_or_default();
    cleanup(&input);
    cleanup(&output);

    assert_eq!(exit, Exit::Success);
    assert!(dump.contains("Ident"), "dump: {dump}");
    assert!(dump.contains("Eq"), "dump: {dump}");
}

/// `unwrap` is safe: the path and contents are constructed here.
#[test]
fn lexical_error_reports_failure() {
    let input = temp_path("lex-error.cl");
    fs::write_file(&input, "let s = \"abc;\n").unwrap();

    let args = args(&[&input]);
    let exit = run(&args);
    cleanup(&input);

    assert_eq!(exit, Exit::Failure);
}

/// `unwrap` is safe: every path and file in this test is constructed here.
#[test]
fn ast_stage_writes_optimized_dump() {
    let input = temp_path("ast-in.cl");
    let output = temp_path("ast-out.txt");
    fs::write_file(&input, "fn f() { 1 + 2 * 3 }\n").unwrap();
    cleanup(&output);

    let args = args(&["--emit", "ast", "-o", &output, &input]);
    let exit = run(&args);
    let dump = fs::read_file(&output).unwrap_or_default();
    cleanup(&input);
    cleanup(&output);

    assert_eq!(exit, Exit::Success);
    assert!(dump.contains("Fn"), "dump: {dump}");
    assert!(dump.contains("Literal"), "dump: {dump}");
    assert!(dump.contains("\"7\""), "folded constant missing: {dump}");
}

/// `unwrap` is safe: the path and contents are constructed here.
#[test]
fn ast_stage_reports_syntax_errors() {
    let input = temp_path("ast-syntax-error.cl");
    fs::write_file(&input, "fn f() { let x = 1 }\n").unwrap();

    let args = args(&["--emit", "ast", &input]);
    let exit = run(&args);
    cleanup(&input);

    assert_eq!(exit, Exit::Failure);
}

/// `unwrap` is safe: the path and contents are constructed here.
#[test]
fn analysis_stage_writes_declarations() {
    let input = temp_path("analysis-in.cl");
    let output = temp_path("analysis-out.txt");
    let source = "\
struct Point { x: float, y: float }
type Coord = Point;
fn scale(p: Point) -> Point { p }
#[kernel]
fn vec_add(a: *mut float, b: *mut float, n: int) { let i = get_global_id(0); }
";
    fs::write_file(&input, source).unwrap();
    cleanup(&output);

    let args = args(&["--emit", "analysis", "-o", &output, &input]);
    let exit = run(&args);
    let dump = fs::read_file(&output).unwrap_or_default();
    cleanup(&input);
    cleanup(&output);

    assert_eq!(exit, Exit::Success);
    assert!(
        dump.contains("struct Point { x: float, y: float }"),
        "dump: {dump}"
    );
    assert!(dump.contains("type Coord = Point"), "dump: {dump}");
    assert!(dump.contains("fn scale(p: Point) -> Point"), "dump: {dump}");
    assert!(
        dump.contains("#[kernel] fn vec_add(a: *mut float, b: *mut float, n: int) -> void"),
        "dump: {dump}"
    );
}

/// `unwrap` is safe: the path and contents are constructed here.
#[test]
fn analysis_stage_reports_semantic_errors() {
    let input = temp_path("analysis-error.cl");
    fs::write_file(&input, "fn f() -> int { true }\n").unwrap();

    let args = args(&["--emit", "analysis", &input]);
    let exit = run(&args);
    cleanup(&input);

    assert_eq!(exit, Exit::Failure);
}

/// `unwrap` is safe: the path and contents are constructed here.
#[test]
fn analysis_stage_treats_warnings_as_success() {
    let input = temp_path("analysis-warn.cl");
    let output = temp_path("analysis-warn-out.txt");
    fs::write_file(&input, "fn f() { let unused = 1; }\n").unwrap();
    cleanup(&output);

    let args = args(&["--emit", "analysis", "-o", &output, &input]);
    let exit = run(&args);
    let dump = fs::read_file(&output).unwrap_or_default();
    cleanup(&input);
    cleanup(&output);

    assert_eq!(exit, Exit::Success);
    assert!(dump.contains("fn f() -> void"), "dump: {dump}");
}

/// `unwrap` is safe: the path and contents are constructed here.
#[test]
fn analysis_stage_gates_on_syntax_errors() {
    let input = temp_path("analysis-syntax.cl");
    fs::write_file(&input, "fn f( { }\n").unwrap();

    let args = args(&["--emit", "analysis", &input]);
    let exit = run(&args);
    cleanup(&input);

    assert_eq!(exit, Exit::Failure);
}

/// `unwrap` is safe: the response file is constructed here.
#[test]
fn response_file_expands_lines() {
    let response = temp_path("args.rsp");
    fs::write_file(&response, "-o\nout.bin\r\nin.cl\n").unwrap();

    let at = format!("@{response}");
    let expanded = expand_response_files(&[at.as_str(), "extra.cl"]).unwrap();
    cleanup(&response);

    assert_eq!(expanded, vec!["-o", "out.bin", "in.cl", "extra.cl"]);
}

#[test]
fn response_file_missing_reports_error() {
    let error = expand_response_files(&["@/nonexistent/codevar-oclc-args"]).unwrap_err();
    assert!(error.contains("failed to load argument file"), "error: {error}");
}

#[test]
fn plain_arguments_pass_through_response_expansion() {
    let expanded = expand_response_files(&["--emit", "tokens", "in.cl"]).unwrap();
    assert_eq!(expanded, vec!["--emit", "tokens", "in.cl"]);
}

#[test]
fn line_column_positions() {
    let source = "a\nbc";
    assert_eq!(line_column(source, 0), (1, 1));
    assert_eq!(line_column(source, 1), (1, 2));
    assert_eq!(line_column(source, 2), (2, 1));
    assert_eq!(line_column(source, 3), (2, 2));
    assert_eq!(line_column(source, 99), (2, 3));
}

#[test]
fn lex_source_reports_clean_tokens() {
    let output = lex_source("fn main() {}");
    assert!(output.diagnostics.is_empty());
    assert!(!output.lines.is_empty());
    assert!(output.lines[0].contains("Ident"));
}

#[test]
fn lex_source_flags_unterminated_string() {
    let output = lex_source("let s = \"abc;");
    assert_eq!(output.diagnostics.len(), 1);
    assert!(
        output.diagnostics[0]
            .message
            .contains("unterminated string literal"),
        "message: {}",
        output.diagnostics[0].message
    );
    assert!(output.diagnostics[0].offset > 0);
}

/// `unwrap` is safe: the path and contents are constructed here.
#[test]
fn fs_roundtrip_preserves_unicode() {
    let path = temp_path("roundtrip.txt");
    let contents = "héllo — kernel\n";
    fs::write_file(&path, contents).unwrap();

    let read_back = fs::read_file(&path).unwrap();
    let via_source = fs::read_source(&path).unwrap();
    cleanup(&path);

    assert_eq!(read_back, contents);
    assert_eq!(via_source, contents);
}

#[test]
fn fs_reports_missing_files() {
    assert!(fs::read_file("/nonexistent/codevar-oclc-file").is_err());
}

/// The font rasterization kernel must be compiled and embedded with its
/// entry point so GPU rasterizers can rely on the registry.
#[test]
fn font_raster_kernel_is_embedded() {
    let module = crate::kernels::KERNEL_MODULES
        .iter()
        .find(|module| module.name == "font_raster");
    let Some(module) = module else {
        panic!("font_raster kernel missing from KERNEL_MODULES");
    };
    assert_eq!(module.entry_points, &["font_raster"]);
    assert_eq!(&module.spirv[0..4], &[0x03, 0x02, 0x23, 0x07]);
    assert!(module.source.contains("fn font_raster"));
}
