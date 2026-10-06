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

//! Command-line argument parsing and validation.
//!
//! [`ArgParser`] declares boolean flags, value-taking options, and positional
//! arguments with a builder API; [`ArgParser::parse`] then turns an iterator of
//! raw arguments (excluding `argv[0]`) into [`ArgMatches`]. The crate is
//! `no_std` + [`alloc`] and has no external dependencies.
//!
//! Supported syntax follows GNU/getopt conventions:
//!
//! - Long options: `--opt value` and `--opt=value`
//! - Short options: `-o value`, `-ovalue`, and `-o=value`
//! - Clustered short flags: `-abc`
//! - The `--` terminator: every following argument is positional
//! - A lone `-` is a positional (the stdin convention)
//!
//! `-h`/`--help` and `-V`/`--version` are **not** special; define them as
//! ordinary flags and react to them yourself.
//!
//! # Examples
//!
//! ```
//! use codevar_cli_arg_parse::ArgParser;
//!
//! let parser = ArgParser::new("demo", "0.1.0", "A demo tool.")
//!     .flag("verbose", Some('v'), "Enable verbose output")
//!     .option("output", Some('o'), "FILE", "Write output to FILE")
//!     .positional("INPUT", "Input source file", true);
//!
//! let matches = parser.parse(["--verbose", "--output=out.txt", "in.cl"])?;
//! assert!(matches.flag("verbose"));
//! assert_eq!(matches.option("output"), Some("out.txt"));
//! assert_eq!(matches.positional(0), Some("in.cl"));
//! # Ok::<(), codevar_cli_arg_parse::ArgError>(())
//! ```

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Number of spaces between the left help column and the help text.
const HELP_GAP: usize = 6;

/// A boolean flag definition (`--flag` / `-f`).
#[derive(Debug, Clone)]
struct FlagDef {
    long: String,
    short: Option<char>,
    help: String,
}

/// A value-taking option definition (`--opt <VALUE>` / `-o <VALUE>`).
#[derive(Debug, Clone)]
struct OptionDef {
    long: String,
    short: Option<char>,
    value_name: String,
    help: String,
}

/// A positional argument definition.
#[derive(Debug, Clone)]
struct PositionalDef {
    name: String,
    help: String,
    required: bool,
}

/// A command-line interface definition and a parser for its arguments.
///
/// Definitions are recorded with [`Self::flag`], [`Self::option`], and
/// [`Self::positional`]; [`Self::parse`] validates raw arguments against them
/// and [`Self::help`] renders a usage listing.
#[derive(Debug, Clone)]
pub struct ArgParser {
    program: String,
    version: String,
    about: String,
    flags: Vec<FlagDef>,
    options: Vec<OptionDef>,
    positionals: Vec<PositionalDef>,
}

impl ArgParser {
    /// Create a parser for a program with the given name, version, and about text.
    #[must_use]
    pub fn new(program: &str, version: &str, about: &str) -> Self {
        Self {
            program: String::from(program),
            version: String::from(version),
            about: String::from(about),
            flags: Vec::new(),
            options: Vec::new(),
            positionals: Vec::new(),
        }
    }

    /// Define a boolean flag: `--flag` / `-f`.
    ///
    /// Re-defining the same `long` replaces the earlier definition, so repeated
    /// calls are idempotent, and while parsing a flag may appear any number of
    /// times. If `long` is also registered through [`Self::option`], the option
    /// definition takes precedence.
    pub fn flag(mut self, long: &str, short: Option<char>, help: &str) -> Self {
        match self
            .flags
            .iter_mut()
            .find(|flag| flag.long == long)
        {
            Some(flag) => {
                flag.short = short;
                flag.help = String::from(help);
            }
            None => self.flags.push(FlagDef {
                long: String::from(long),
                short,
                help: String::from(help),
            }),
        }
        self
    }

    /// Define an option that takes a value: `--opt value`, `--opt=value`, `-o value`, `-ovalue`.
    ///
    /// A later definition with the same `long` replaces the earlier one. While
    /// parsing, a value-taking option may appear at most once; a second
    /// occurrence yields [`ArgError::DuplicateOption`].
    pub fn option(mut self, long: &str, short: Option<char>, value_name: &str, help: &str) -> Self {
        match self
            .options
            .iter_mut()
            .find(|option| option.long == long)
        {
            Some(option) => {
                option.short = short;
                option.value_name = String::from(value_name);
                option.help = String::from(help);
            }
            None => self.options.push(OptionDef {
                long: String::from(long),
                short,
                value_name: String::from(value_name),
                help: String::from(help),
            }),
        }
        self
    }

    /// Define a positional argument. `required: true` means parsing fails without it.
    ///
    /// Extra positionals beyond the last defined one are rejected with
    /// [`ArgError::UnexpectedPositional`].
    pub fn positional(mut self, name: &str, help: &str, required: bool) -> Self {
        self.positionals.push(PositionalDef {
            name: String::from(name),
            help: String::from(help),
            required,
        });
        self
    }

    /// Parse arguments (NOT including the program name at argv[0]).
    ///
    /// A value-taking option consumes the following argument verbatim, even if
    /// it looks like an option; [`ArgError::MissingValue`] is raised only when
    /// the input ends before a value can be read. Everything after the `--`
    /// terminator is treated as a positional.
    ///
    /// # Errors
    ///
    /// - [`ArgError::UnknownOption`] for an option that was never defined.
    /// - [`ArgError::MissingValue`] when a value-taking option ends the input.
    /// - [`ArgError::DuplicateOption`] when a value-taking option is repeated.
    /// - [`ArgError::UnexpectedValue`] when a boolean flag is given a value.
    /// - [`ArgError::UnexpectedPositional`] for a surplus positional argument.
    /// - [`ArgError::MissingRequired`] when a required positional is absent.
    pub fn parse<I, S>(&self, args: I) -> Result<ArgMatches, ArgError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let owned: Vec<S> = args.into_iter().collect();
        let args: Vec<&str> = owned.iter().map(|arg| arg.as_ref()).collect();
        let mut matches = ArgMatches::default();
        let mut end_of_options = false;
        let mut index = 0;
        while index < args.len() {
            let arg = args[index];
            index += 1;
            if end_of_options {
                self.push_positional(&mut matches, arg)?;
            } else if arg == "--" {
                end_of_options = true;
            } else if let Some(body) = arg.strip_prefix("--") {
                self.parse_long(body, &args, &mut index, &mut matches)?;
            } else if arg.starts_with('-') && arg.len() > 1 {
                self.parse_short(arg, &args, &mut index, &mut matches)?;
            } else {
                self.push_positional(&mut matches, arg)?;
            }
        }
        self.check_required(&matches)?;
        Ok(matches)
    }

    /// Render the help text (usage + flags/options/positionals columns).
    ///
    /// The result begins with `"{program} {version}"` and the about text,
    /// followed by a `USAGE:` line and, for the definitions that exist,
    /// `OPTIONS:` and `POSITIONALS:` sections. All entries share a single
    /// column layout computed from the widest left-hand entry. Every line,
    /// including the last, is newline-terminated.
    pub fn help(&self) -> String {
        let mut options: Vec<(String, String, &str)> = Vec::new();
        for flag in &self.flags {
            let left = match flag.short {
                Some(short) => format!("-{short}, --{}", flag.long),
                None => format!("--{}", flag.long),
            };
            options.push((flag.long.clone(), left, flag.help.as_str()));
        }
        for option in &self.options {
            let base = match option.short {
                Some(short) => format!("-{short}, --{}", option.long),
                None => format!("--{}", option.long),
            };
            let left = format!("{base} <{}>", option.value_name);
            options.push((option.long.clone(), left, option.help.as_str()));
        }
        options.sort_by(|left, right| left.0.cmp(&right.0));
        let mut positionals: Vec<(String, &str)> = Vec::new();
        for definition in &self.positionals {
            let left = if definition.required {
                format!("<{}>", definition.name)
            } else {
                format!("[{}]", definition.name)
            };
            positionals.push((left, definition.help.as_str()));
        }

        let width = options
            .iter()
            .map(|(_, left, _)| left.chars().count())
            .chain(
                positionals
                    .iter()
                    .map(|(left, _)| left.chars().count()),
            )
            .max()
            .unwrap_or(0);

        let mut out = String::new();
        out.push_str(&format!("{} {}\n", self.program, self.version));
        out.push_str(&self.about);
        out.push_str("\n\nUSAGE:\n    ");
        out.push_str(&self.program);
        if !self.flags.is_empty() || !self.options.is_empty() {
            out.push_str(" [OPTIONS]");
        }
        for definition in &self.positionals {
            if definition.required {
                out.push_str(&format!(" <{}>", definition.name));
            } else {
                out.push_str(&format!(" [{}]", definition.name));
            }
        }
        out.push('\n');

        if !options.is_empty() {
            out.push_str("\nOPTIONS:\n");
            for (_, left, help) in &options {
                push_row(&mut out, left, help, width);
            }
        }
        if !positionals.is_empty() {
            out.push_str("\nPOSITIONALS:\n");
            for (left, help) in &positionals {
                push_row(&mut out, left, help, width);
            }
        }
        out
    }

    /// Render `"{program} {version}"`.
    pub fn version(&self) -> String {
        format!("{} {}", self.program, self.version)
    }

    /// Handle a single `--name[=value]` argument.
    fn parse_long(
        &self,
        body: &str,
        args: &[&str],
        index: &mut usize,
        matches: &mut ArgMatches,
    ) -> Result<(), ArgError> {
        let (name, attached) = match body.find('=') {
            Some(position) => (&body[..position], Some(&body[position + 1..])),
            None => (body, None),
        };
        if let Some(option) = self
            .options
            .iter()
            .find(|option| option.long == name)
        {
            let value = match attached {
                Some(attached) => String::from(attached),
                None => take_value(args, index, &option.long)?,
            };
            push_option(matches, &option.long, value)
        } else if let Some(flag) = self.flags.iter().find(|flag| flag.long == name) {
            if attached.is_some() {
                return Err(ArgError::UnexpectedValue {
                    flag: format!("--{}", flag.long),
                });
            }
            push_flag(matches, &flag.long);
            Ok(())
        } else {
            Err(ArgError::UnknownOption {
                name: format!("--{name}"),
            })
        }
    }

    /// Handle a single `-abc` argument: clustered flags until a value-taking option.
    fn parse_short(
        &self,
        arg: &str,
        args: &[&str],
        index: &mut usize,
        matches: &mut ArgMatches,
    ) -> Result<(), ArgError> {
        let letters: Vec<char> = arg[1..].chars().collect();
        let mut position = 0;
        while position < letters.len() {
            let letter = letters[position];
            position += 1;
            if let Some(option) = self
                .options
                .iter()
                .find(|option| option.short == Some(letter))
            {
                let remainder: String = letters[position..].iter().collect();
                let detached = remainder.strip_prefix('=').map(String::from);
                let value = match detached {
                    Some(value) => value,
                    None if remainder.is_empty() => take_value(args, index, &option.long)?,
                    None => remainder,
                };
                push_option(matches, &option.long, value)?;
                return Ok(());
            }
            if let Some(flag) = self
                .flags
                .iter()
                .find(|flag| flag.short == Some(letter))
            {
                push_flag(matches, &flag.long);
                continue;
            }
            return Err(ArgError::UnknownOption {
                name: format!("-{letter}"),
            });
        }
        Ok(())
    }

    /// Record a positional, rejecting values beyond the last definition.
    fn push_positional(&self, matches: &mut ArgMatches, value: &str) -> Result<(), ArgError> {
        if matches.positionals.len() >= self.positionals.len() {
            return Err(ArgError::UnexpectedPositional {
                value: String::from(value),
            });
        }
        matches.positionals.push(String::from(value));
        Ok(())
    }

    /// Fail if the first unfilled required positional is still missing.
    fn check_required(&self, matches: &ArgMatches) -> Result<(), ArgError> {
        for (index, definition) in self.positionals.iter().enumerate() {
            if definition.required && matches.positionals.len() <= index {
                return Err(ArgError::MissingRequired {
                    name: definition.name.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Consume the next raw argument as an option value, or report [`ArgError::MissingValue`].
fn take_value(args: &[&str], index: &mut usize, option: &str) -> Result<String, ArgError> {
    match args.get(*index) {
        Some(value) => {
            *index += 1;
            Ok(String::from(*value))
        }
        None => Err(ArgError::MissingValue {
            option: format!("--{option}"),
        }),
    }
}

/// Record a flag occurrence, ignoring repeats.
fn push_flag(matches: &mut ArgMatches, long: &str) {
    if !matches.flags.iter().any(|flag| flag == long) {
        matches.flags.push(String::from(long));
    }
}

/// Record an option value, rejecting a second occurrence of the same option.
fn push_option(matches: &mut ArgMatches, long: &str, value: String) -> Result<(), ArgError> {
    if matches
        .options
        .iter()
        .any(|(option, _)| option == long)
    {
        return Err(ArgError::DuplicateOption {
            option: format!("--{long}"),
        });
    }
    matches.options.push((String::from(long), value));
    Ok(())
}

/// Append one aligned `    <left><gap><help>` line to a help buffer.
fn push_row(out: &mut String, left: &str, help: &str, width: usize) {
    out.push_str("    ");
    out.push_str(left);
    for _ in left.chars().count()..width {
        out.push(' ');
    }
    for _ in 0..HELP_GAP {
        out.push(' ');
    }
    out.push_str(help);
    out.push('\n');
}

/// The parsed command line, queried by long option name or positional index.
#[derive(Debug, Clone, Default)]
pub struct ArgMatches {
    flags: Vec<String>,
    options: Vec<(String, String)>,
    positionals: Vec<String>,
}

impl ArgMatches {
    /// Returns `true` when the boolean flag `long` was present on the command line.
    #[inline]
    pub fn flag(&self, long: &str) -> bool {
        self.flags.iter().any(|flag| flag == long)
    }

    /// Returns the value given for the option `long`, or `None` if it was absent.
    #[inline]
    pub fn option(&self, long: &str) -> Option<&str> {
        self.options
            .iter()
            .find(|(option, _)| option == long)
            .map(|(_, value)| value.as_str())
    }

    /// Returns the positional argument at zero-based `index`, or `None` if it was absent.
    #[inline]
    pub fn positional(&self, index: usize) -> Option<&str> {
        self.positionals.get(index).map(String::as_str)
    }
}

/// An error produced while parsing or validating command-line arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgError {
    /// An option or flag that was never defined was encountered.
    UnknownOption {
        /// The offending name with its dashes as typed (e.g. `"--output"` or `"-z"`).
        name: String,
    },
    /// A value-taking option reached the end of the input without a value.
    MissingValue {
        /// The canonical long form of the option (e.g. `"--output"`).
        option: String,
    },
    /// A required positional argument was not supplied.
    MissingRequired {
        /// The defined name of the missing positional (e.g. `"INPUT"`).
        name: String,
    },
    /// More positional arguments were supplied than were defined.
    UnexpectedPositional {
        /// The first surplus positional value, exactly as supplied.
        value: String,
    },
    /// An option value did not match the expected set of choices.
    InvalidValue {
        /// The option name as presented to the validator (e.g. `"--emit"`).
        option: String,
        /// The rejected value.
        value: String,
        /// Human-readable description of what was expected.
        expected: &'static str,
    },
    /// A boolean flag was given a value (e.g. `--verbose=true`).
    UnexpectedValue {
        /// The canonical long form of the flag (e.g. `"--verbose"`).
        flag: String,
    },
    /// A value-taking option was supplied more than once.
    DuplicateOption {
        /// The canonical long form of the option (e.g. `"--output"`).
        option: String,
    },
}

impl fmt::Display for ArgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownOption { name } => write!(f, "unknown option '{name}'"),
            Self::MissingValue { option } => {
                write!(f, "a value is required for '{option}' but none was supplied")
            }
            Self::MissingRequired { name } => write!(f, "missing required argument '<{name}>'"),
            Self::UnexpectedPositional { value } => write!(f, "unexpected positional argument '{value}'"),
            Self::InvalidValue {
                option,
                value,
                expected,
            } => write!(f, "invalid value '{value}' for '{option}': {expected}"),
            Self::UnexpectedValue { flag } => write!(f, "option '{flag}' does not take a value"),
            Self::DuplicateOption { option } => write!(f, "option '{option}' was provided more than once"),
        }
    }
}

impl core::error::Error for ArgError {}

/// Validate an option's value against a fixed set of choices.
///
/// Returns `Ok(None)` when `value` is `None`, `Ok(Some(value))` when `value`
/// appears in `choices`, and [`ArgError::InvalidValue`] otherwise. `option` is
/// embedded verbatim in the error message, so include the dashes
/// (e.g. `"--emit"`).
///
/// # Errors
///
/// Returns [`ArgError::InvalidValue`] when `value` is present but is not one
/// of `choices`.
///
/// # Memory
///
/// [`ArgError::InvalidValue::expected`] has type `&'static str`, so the joined
/// choice list (`"one of: a, b"`) is leaked into a `&'static str` on the error
/// path. The leak happens at most once per failed validation, and validation
/// failures are fatal for command-line tools.
pub fn validate_choice<'a>(
    option: &str,
    value: Option<&'a str>,
    choices: &[&'static str],
) -> Result<Option<&'a str>, ArgError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if choices.contains(&value) {
        return Ok(Some(value));
    }
    let expected = describe_choices(choices);
    Err(ArgError::InvalidValue {
        option: String::from(option),
        value: String::from(value),
        expected,
    })
}

/// Build the `"one of: a, b"` expectation string as a `&'static str`.
fn describe_choices(choices: &[&'static str]) -> &'static str {
    if choices.is_empty() {
        return "one of the configured choices";
    }
    let mut joined = String::new();
    for (index, choice) in choices.iter().enumerate() {
        if index > 0 {
            joined.push_str(", ");
        }
        joined.push_str(choice);
    }
    &*Box::leak(format!("one of: {joined}").into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A representative parser used across the parse-rule tests.
    fn sample() -> ArgParser {
        ArgParser::new("codevar-oclc", "1.0.0", "Compile OpenCL sources.")
            .flag("verbose", Some('v'), "Enable verbose output")
            .flag("quiet", Some('q'), "Suppress progress output")
            .option("output", Some('o'), "FILE", "Write output to FILE")
            .option("emit", None, "TYPE", "Select emission kind")
            .positional("INPUT", "Input OpenCL source file", true)
    }

    #[test]
    fn flags_set_via_long_and_short() {
        let matches = sample()
            .parse(["--verbose", "-q", "in.cl"])
            .unwrap();
        assert!(matches.flag("verbose"));
        assert!(matches.flag("quiet"));
    }

    #[test]
    fn clustered_short_flags() {
        let matches = sample().parse(["-vq", "in.cl"]).unwrap();
        assert!(matches.flag("verbose"));
        assert!(matches.flag("quiet"));
    }

    #[test]
    fn flags_are_idempotent() {
        let matches = sample()
            .parse(["-v", "--verbose", "-vv", "in.cl"])
            .unwrap();
        assert!(matches.flag("verbose"));
        assert!(!matches.flag("quiet"));
    }

    #[test]
    fn long_option_space_and_equals_forms() {
        let parser = sample();
        let spaced = parser
            .parse(["--output", "a.txt", "in.cl"])
            .unwrap();
        assert_eq!(spaced.option("output"), Some("a.txt"));
        let attached = parser.parse(["--output=a.txt", "in.cl"]).unwrap();
        assert_eq!(attached.option("output"), Some("a.txt"));
    }

    #[test]
    fn long_option_allows_empty_value() {
        let matches = sample().parse(["--output=", "in.cl"]).unwrap();
        assert_eq!(matches.option("output"), Some(""));
    }

    #[test]
    fn short_option_value_forms() {
        let parser = sample();
        let spaced = parser.parse(["-o", "a.txt", "in.cl"]).unwrap();
        assert_eq!(spaced.option("output"), Some("a.txt"));
        let joined = parser.parse(["-oa.txt", "in.cl"]).unwrap();
        assert_eq!(joined.option("output"), Some("a.txt"));
        let equals = parser.parse(["-o=a.txt", "in.cl"]).unwrap();
        assert_eq!(equals.option("output"), Some("a.txt"));
    }

    #[test]
    fn cluster_stops_at_value_short_with_attached_value() {
        let matches = sample().parse(["-voout.txt", "in.cl"]).unwrap();
        assert!(matches.flag("verbose"));
        assert_eq!(matches.option("output"), Some("out.txt"));
    }

    #[test]
    fn cluster_stops_at_value_short_with_next_arg() {
        let matches = sample()
            .parse(["-vo", "out.txt", "in.cl"])
            .unwrap();
        assert!(matches.flag("verbose"));
        assert_eq!(matches.option("output"), Some("out.txt"));
    }

    #[test]
    fn terminator_passes_options_as_positionals() {
        let parser = ArgParser::new("t", "1", "a")
            .positional("A", "first", false)
            .positional("B", "second", false);
        let matches = parser.parse(["--", "-v", "--output=x"]).unwrap();
        assert_eq!(matches.positional(0), Some("-v"));
        assert_eq!(matches.positional(1), Some("--output=x"));
        assert!(!matches.flag("verbose"));
        assert_eq!(matches.option("output"), None);
    }

    #[test]
    fn lone_dash_is_positional() {
        let matches = sample().parse(["-"]).unwrap();
        assert_eq!(matches.positional(0), Some("-"));
    }

    #[test]
    fn unknown_option_errors() {
        let parser = sample();
        assert_eq!(
            parser.parse(["--nope"]).unwrap_err(),
            ArgError::UnknownOption {
                name: String::from("--nope"),
            }
        );
        assert_eq!(
            parser.parse(["-z"]).unwrap_err(),
            ArgError::UnknownOption {
                name: String::from("-z"),
            }
        );
    }

    #[test]
    fn unknown_short_inside_cluster_error() {
        assert_eq!(
            sample().parse(["-vz"]).unwrap_err(),
            ArgError::UnknownOption {
                name: String::from("-z"),
            }
        );
    }

    #[test]
    fn missing_value_at_end_of_input() {
        let parser = sample();
        assert_eq!(
            parser.parse(["--output"]).unwrap_err(),
            ArgError::MissingValue {
                option: String::from("--output"),
            }
        );
        assert_eq!(
            parser.parse(["-o"]).unwrap_err(),
            ArgError::MissingValue {
                option: String::from("--output"),
            }
        );
    }

    #[test]
    fn duplicate_long_option_error() {
        assert_eq!(
            sample()
                .parse(["--output=a", "--output=b"])
                .unwrap_err(),
            ArgError::DuplicateOption {
                option: String::from("--output"),
            }
        );
    }

    #[test]
    fn duplicate_short_then_long_error() {
        assert_eq!(
            sample()
                .parse(["-o", "a", "--output=b"])
                .unwrap_err(),
            ArgError::DuplicateOption {
                option: String::from("--output"),
            }
        );
    }

    #[test]
    fn flag_rejects_attached_value() {
        assert_eq!(
            sample().parse(["--verbose=true"]).unwrap_err(),
            ArgError::UnexpectedValue {
                flag: String::from("--verbose"),
            }
        );
    }

    #[test]
    fn surplus_positional_error() {
        assert_eq!(
            sample()
                .parse(["a.cl", "b.cl", "c.cl"])
                .unwrap_err(),
            ArgError::UnexpectedPositional {
                value: String::from("b.cl"),
            }
        );
    }

    #[test]
    fn missing_required_reports_first_unfilled() {
        let parser = ArgParser::new("t", "1", "a")
            .positional("SRC", "source", true)
            .positional("DST", "destination", true);
        assert_eq!(
            parser
                .parse(core::iter::empty::<&str>())
                .unwrap_err(),
            ArgError::MissingRequired {
                name: String::from("SRC"),
            }
        );
        assert_eq!(
            parser.parse(["src.cl"]).unwrap_err(),
            ArgError::MissingRequired {
                name: String::from("DST"),
            }
        );
    }

    #[test]
    fn optional_positional_can_be_omitted() {
        let parser = ArgParser::new("t", "1", "a").positional("FILE", "a file", false);
        let matches = parser.parse(core::iter::empty::<&str>()).unwrap();
        assert_eq!(matches.positional(0), None);
    }

    #[test]
    fn option_value_may_look_like_option() {
        let matches = sample()
            .parse(["--output", "--verbose", "in.cl"])
            .unwrap();
        assert_eq!(matches.option("output"), Some("--verbose"));
        assert!(!matches.flag("verbose"));
    }

    #[test]
    fn help_is_not_special_by_default() {
        let parser = ArgParser::new("t", "1", "a").positional("A", "arg", true);
        assert_eq!(
            parser.parse(["--help"]).unwrap_err(),
            ArgError::UnknownOption {
                name: String::from("--help"),
            }
        );
        assert_eq!(
            parser.parse(["-h"]).unwrap_err(),
            ArgError::UnknownOption {
                name: String::from("-h"),
            }
        );
    }

    #[test]
    fn help_and_version_flags_when_defined() {
        let parser = ArgParser::new("t", "1", "a")
            .flag("help", Some('h'), "Print help information")
            .flag("version", Some('V'), "Print version information");
        let matches = parser.parse(["-hV"]).unwrap();
        assert!(matches.flag("help"));
        assert!(matches.flag("version"));
    }

    #[test]
    fn option_definition_is_replaced_by_later_definition() {
        let parser = ArgParser::new("t", "1", "a")
            .option("output", Some('o'), "FILE", "old help")
            .option("output", Some('x'), "PATH", "new help");
        let matches = parser.parse(["--output", "v"]).unwrap();
        assert_eq!(matches.option("output"), Some("v"));
        let help = parser.help();
        assert_eq!(help.matches("--output").count(), 1);
        assert!(help.contains("<PATH>"));
        assert!(!help.contains("old help"));
        assert_eq!(
            parser.parse(["-o", "v"]).unwrap_err(),
            ArgError::UnknownOption {
                name: String::from("-o"),
            }
        );
    }

    #[test]
    fn flag_definition_is_idempotent() {
        let parser = ArgParser::new("t", "1", "a")
            .flag("verbose", Some('v'), "first help")
            .flag("verbose", Some('v'), "second help");
        let help = parser.help();
        assert_eq!(help.matches("--verbose").count(), 1);
        assert!(help.contains("second help"));
        assert!(!help.contains("first help"));
        assert!(parser.parse(["-v"]).unwrap().flag("verbose"));
    }

    #[test]
    fn positionals_and_options_interleave() {
        let matches = sample()
            .parse(["in.cl", "--emit", "ast", "-v", "-o", "out.bc"])
            .unwrap();
        assert_eq!(matches.positional(0), Some("in.cl"));
        assert_eq!(matches.option("emit"), Some("ast"));
        assert_eq!(matches.option("output"), Some("out.bc"));
        assert!(matches.flag("verbose"));
    }

    #[test]
    fn parse_accepts_owned_string_iter() {
        let args = vec![String::from("-v"), String::from("in.cl")];
        let matches = sample().parse(args).unwrap();
        assert!(matches.flag("verbose"));
        assert_eq!(matches.positional(0), Some("in.cl"));
    }

    #[test]
    fn accessors_for_absent_arguments() {
        let matches = sample().parse(["in.cl"]).unwrap();
        assert!(!matches.flag("verbose"));
        assert!(!matches.flag("nope"));
        assert_eq!(matches.option("emit"), None);
        assert_eq!(matches.option("nope"), None);
        assert_eq!(matches.positional(0), Some("in.cl"));
        assert_eq!(matches.positional(1), None);
    }

    #[test]
    fn parser_without_positionals_rejects_any() {
        let parser = ArgParser::new("t", "1", "a").flag("verbose", Some('v'), "verbose");
        assert_eq!(
            parser.parse(["file"]).unwrap_err(),
            ArgError::UnexpectedPositional {
                value: String::from("file"),
            }
        );
        assert_eq!(
            parser.parse(["--", "file"]).unwrap_err(),
            ArgError::UnexpectedPositional {
                value: String::from("file"),
            }
        );
    }

    #[test]
    fn validate_choice_returns_none_for_missing_value() {
        assert_eq!(validate_choice("--emit", None, &["ast", "binary"]), Ok(None));
    }

    #[test]
    fn validate_choice_accepts_listed_value() {
        assert_eq!(
            validate_choice("--emit", Some("ast"), &["ast", "binary"]),
            Ok(Some("ast"))
        );
    }

    #[test]
    fn validate_choice_rejects_unknown_value() {
        let error = validate_choice("--emit", Some("foo"), &["ast", "binary", "llvm"]).unwrap_err();
        assert_eq!(
            error,
            ArgError::InvalidValue {
                option: String::from("--emit"),
                value: String::from("foo"),
                expected: "one of: ast, binary, llvm",
            }
        );
        assert!(
            error
                .to_string()
                .contains("one of: ast, binary, llvm")
        );
    }

    #[test]
    fn help_snapshot_matches_reference_layout() {
        let parser = ArgParser::new("codevar-oclc", "1.0.0", "<about>")
            .flag("help", Some('h'), "Print help information")
            .flag("version", Some('V'), "Print version information")
            .option("output", Some('o'), "FILE", "Write output to FILE")
            .positional("INPUT", "Input OpenCL source file", true);
        let expected = concat!(
            "codevar-oclc 1.0.0\n",
            "<about>\n",
            "\n",
            "USAGE:\n",
            "    codevar-oclc [OPTIONS] <INPUT>\n",
            "\n",
            "OPTIONS:\n",
            "    -h, --help               Print help information\n",
            "    -o, --output <FILE>      Write output to FILE\n",
            "    -V, --version            Print version information\n",
            "\n",
            "POSITIONALS:\n",
            "    <INPUT>                  Input OpenCL source file\n",
        );
        assert_eq!(parser.help(), expected);
    }

    #[test]
    fn help_omits_unused_sections_and_brackets_optional() {
        let parser = ArgParser::new("tool", "2.0.0", "about").positional("FILE", "a file", false);
        let help = parser.help();
        assert!(help.contains("USAGE:\n    tool [FILE]\n"));
        assert!(!help.contains("[OPTIONS]"));
        assert!(!help.contains("OPTIONS:"));
        assert!(help.contains("POSITIONALS:\n    [FILE]      a file\n"));
    }

    #[test]
    fn help_shares_column_across_sections() {
        let parser = ArgParser::new("t", "1", "a")
            .option("x", None, "V", "an option help")
            .positional("SOME-VERY-LONG-POSITIONAL", "a positional help", true);
        let help = parser.help();
        let column_of = |needle: &str| {
            help.lines()
                .find(|line| line.contains(needle))
                .and_then(|line| line.find(needle))
                .unwrap_or(0)
        };
        let option_column = column_of("an option help");
        let positional_column = column_of("a positional help");
        assert_eq!(option_column, positional_column);
        assert_eq!(option_column, 37);
    }

    #[test]
    fn help_renders_option_without_short() {
        let parser = ArgParser::new("t", "1", "a").option("emit", None, "TYPE", "Select emission kind");
        let help = parser.help();
        assert!(help.contains("USAGE:\n    t [OPTIONS]\n"));
        assert!(help.contains("    --emit <TYPE>      Select emission kind\n"));
    }

    #[test]
    fn version_formats_program_and_version() {
        assert_eq!(sample().version(), "codevar-oclc 1.0.0");
    }

    #[test]
    fn display_messages_for_every_error_variant() {
        assert_eq!(
            ArgError::UnknownOption {
                name: String::from("--nope")
            }
            .to_string(),
            "unknown option '--nope'"
        );
        assert_eq!(
            ArgError::MissingValue {
                option: String::from("--output")
            }
            .to_string(),
            "a value is required for '--output' but none was supplied"
        );
        assert_eq!(
            ArgError::MissingRequired {
                name: String::from("INPUT")
            }
            .to_string(),
            "missing required argument '<INPUT>'"
        );
        assert_eq!(
            ArgError::UnexpectedPositional {
                value: String::from("extra")
            }
            .to_string(),
            "unexpected positional argument 'extra'"
        );
        assert_eq!(
            ArgError::InvalidValue {
                option: String::from("--emit"),
                value: String::from("x"),
                expected: "one of: ast, binary",
            }
            .to_string(),
            "invalid value 'x' for '--emit': one of: ast, binary"
        );
        assert_eq!(
            ArgError::UnexpectedValue {
                flag: String::from("--verbose")
            }
            .to_string(),
            "option '--verbose' does not take a value"
        );
        assert_eq!(
            ArgError::DuplicateOption {
                option: String::from("--output")
            }
            .to_string(),
            "option '--output' was provided more than once"
        );
    }

    #[test]
    fn arg_error_implements_core_error_trait() {
        fn assert_error<E: core::error::Error + core::fmt::Debug>() {}
        assert_error::<ArgError>();
        let error: &dyn core::error::Error = &ArgError::UnknownOption {
            name: String::from("--x"),
        };
        assert_eq!(error.to_string(), "unknown option '--x'");
    }
}
