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

//! Pretty-printing diagnostics with source excerpts.
//!
//! The layout follows rustc's rendered output:
//!
//! ```text
//! error[E0308]: mismatched types
//!  --> kernel.cl:3:20
//!   |
//! 3 | fn k() -> int { true }
//!   |                    ^^^^ expected `int`, found `bool`
//!   |
//!   = help: cast the value with `as`
//! ```
//!
//! Coloring is opt-in per call through [`ColorChoice`]; the plain-text
//! form is identical to the colored one minus escape codes, so tests and
//! redirected output stay byte-for-byte predictable.

use alloc::format;
use alloc::string::{String, ToString};
use core::fmt::Write;

use codevar_consoleutil::{AnsiColor, Style};

use crate::diagnostic::{Diagnostic, Label, Severity};

/// When ANSI color is emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorChoice {
    /// Color when the terminal reports ANSI support.
    #[default]
    Auto,
    /// Always color, even when piping.
    Always,
    /// Never color.
    Never,
}

impl ColorChoice {
    /// True when this choice resolves to colored output.
    #[must_use]
    pub fn enabled(self) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,
            Self::Auto => codevar_consoleutil::supports_ansi(),
        }
    }
}

/// The style pairs used by the renderer, resolved once per call.
struct Styles {
    error: (String, String),
    warning: (String, String),
    note: (String, String),
    gutter: (String, String),
    secondary: (String, String),
    help: (String, String),
}

impl Styles {
    fn new(color: bool) -> Self {
        Self {
            error: pair(color, Style::new().fg(AnsiColor::BrightRed).bold()),
            warning: pair(color, Style::new().fg(AnsiColor::BrightYellow).bold()),
            note: pair(color, Style::new().fg(AnsiColor::BrightCyan).bold()),
            gutter: pair(color, Style::new().fg(AnsiColor::BrightBlue).bold()),
            secondary: pair(color, Style::new().fg(AnsiColor::BrightBlue)),
            help: pair(color, Style::new().fg(AnsiColor::BrightGreen).bold()),
        }
    }

    fn severity(&self, severity: Severity) -> &(String, String) {
        match severity {
            Severity::Error => &self.error,
            Severity::Warning => &self.warning,
            Severity::Note => &self.note,
        }
    }
}

/// Resolves a style into its `(escape, reset)` pair, or two empty strings.
fn pair(color: bool, style: Style) -> (String, String) {
    if color {
        (style.build(), style.reset())
    } else {
        (String::new(), String::new())
    }
}

/// Paints `text` with `style` when color is on.
fn paint(style: &(String, String), text: &str) -> String {
    if style.0.is_empty() {
        return text.to_string();
    }
    let mut painted = String::with_capacity(text.len() + style.0.len() + style.1.len());
    painted.push_str(&style.0);
    painted.push_str(text);
    painted.push_str(&style.1);
    painted
}

/// Renders every diagnostic as rustc-style source excerpts.
///
/// Diagnostics are printed in the order given; [`crate::analyze`] sorts
/// them by source position first. The returned string always ends with a
/// newline when `diagnostics` is non-empty.
///
/// # Examples
///
/// ```
/// use codevar_ocl_sar::{ColorChoice, Diagnostic, Span, render};
///
/// let diagnostic = Diagnostic::error(Span::new(0, 2), "something is wrong");
/// let text = render(&[diagnostic], "kernel.cl", "fn", ColorChoice::Never);
/// assert!(text.contains("error: something is wrong"));
/// assert!(!text.contains('\u{1b}'));
/// ```
#[must_use]
pub fn render(diagnostics: &[Diagnostic], path: &str, source: &str, choice: ColorChoice) -> String {
    if diagnostics.is_empty() {
        return String::new();
    }
    let styles = Styles::new(choice.enabled());
    let width = line_number_width(diagnostics, source);
    let mut out = String::new();
    for (index, diagnostic) in diagnostics.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        render_diagnostic(&mut out, diagnostic, path, source, width, &styles);
    }
    out
}

/// Renders `diagnostics` to standard error, ignoring a write failure.
///
/// There is nowhere to report a failed stderr write, so the caller's exit
/// status carries the outcome — the same contract the driver uses for its
/// own `report_error`.
pub fn emit_stderr(diagnostics: &[Diagnostic], path: &str, source: &str, choice: ColorChoice) {
    let text = render(diagnostics, path, source, choice);
    let _ = codevar_consoleutil::write_stderr(text.as_bytes());
}

/// Digits of the largest line number any diagnostic points at (min 1).
fn line_number_width(diagnostics: &[Diagnostic], source: &str) -> usize {
    let mut widest = 1usize;
    for diagnostic in diagnostics {
        for label in &diagnostic.labels {
            let line = line_of(source, clamped(label.span.offset as usize, source)).0 + 1;
            widest = widest.max(line);
        }
    }
    widest.to_string().len()
}

/// Renders one diagnostic: header, one block per label, notes, help.
fn render_diagnostic(
    out: &mut String,
    diagnostic: &Diagnostic,
    path: &str,
    source: &str,
    width: usize,
    styles: &Styles,
) {
    let severity_style = styles.severity(diagnostic.severity);
    let severity_word = match diagnostic.code {
        Some(code) => format!("{}[{}]", diagnostic.severity.as_str(), code),
        None => diagnostic.severity.as_str().to_string(),
    };
    let _ = writeln!(
        out,
        "{}: {}",
        paint(severity_style, &severity_word),
        diagnostic.message
    );

    for (index, label) in diagnostic.labels.iter().enumerate() {
        render_label(out, label, index, path, source, width, styles, severity_style);
    }

    let pad = " ".repeat(width + 1);
    for note in &diagnostic.notes {
        let _ = writeln!(
            out,
            "{}{} {}",
            pad,
            paint(&styles.note, "="),
            paint(&styles.note, &format!("note: {note}"))
        );
    }
    if let Some(help) = &diagnostic.help {
        let _ = writeln!(
            out,
            "{}{} {}",
            pad,
            paint(&styles.help, "="),
            paint(&styles.help, &format!("help: {help}"))
        );
    }
}

/// Renders one spanned label as an arrow, a code line, and a marker line.
#[allow(
    clippy::too_many_arguments,
    reason = "render context grouped by role; splitting it would obscure the layout code"
)]
fn render_label(
    out: &mut String,
    label: &Label,
    index: usize,
    path: &str,
    source: &str,
    width: usize,
    styles: &Styles,
    severity_style: &(String, String),
) {
    let offset = clamped(label.span.offset as usize, source);
    let end = clamped(offset + label.span.len as usize, source);
    let (line, line_start) = line_of(source, offset);
    let (end_line, _) = line_of(source, end);
    let line_end = source[line_start..]
        .find('\n')
        .map_or(source.len(), |position| line_start + position);
    let text = &source[line_start..line_end];

    let column = source[line_start..offset].chars().count() + 1;
    let stop = if end_line == line {
        source[line_start..end].chars().count() + 1
    } else {
        text.chars().count() + 1
    };
    let marker_len = stop.saturating_sub(column).max(1);

    let pad = " ".repeat(width);
    let bar = " ".repeat(width + 1);
    let _ = writeln!(
        out,
        "{}{} {}:{}:{}",
        pad,
        paint(&styles.gutter, "-->"),
        path,
        line + 1,
        column
    );
    let _ = writeln!(out, "{}{}", bar, paint(&styles.gutter, "|"));
    let _ = writeln!(
        out,
        "{:>width$} {} {}",
        line + 1,
        paint(&styles.gutter, "|"),
        text.trim_end(),
        width = width
    );

    let marker_style = if index == 0 {
        severity_style
    } else {
        &styles.secondary
    };
    let marker = if index == 0 { '^' } else { '-' };
    let marker = paint(marker_style, &marker.to_string().repeat(marker_len));
    if label.message.is_empty() {
        let _ = writeln!(out, "{}{} {}", bar, paint(&styles.gutter, "|"), marker);
    } else {
        let _ = writeln!(
            out,
            "{}{} {} {}",
            bar,
            paint(&styles.gutter, "|"),
            marker,
            label.message
        );
    }
    let _ = writeln!(out, "{}{}", bar, paint(&styles.gutter, "|"));
}

/// Clamps `offset` to a character boundary inside `source`.
fn clamped(offset: usize, source: &str) -> usize {
    let mut offset = offset.min(source.len());
    while offset > 0 && !source.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

/// Zero-based line index of `offset` plus that line's start byte.
fn line_of(source: &str, offset: usize) -> (usize, usize) {
    let prefix = &source[..offset];
    let line = prefix
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count();
    let start = prefix
        .rfind('\n')
        .map_or(0, |position| position + 1);
    (line, start)
}
