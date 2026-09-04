//! Copyright 2026 Codevar
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

use codevar_core::logtrace::Level;

macro_rules! level_case {
    ($name:ident, $value:expr, $expected:expr) => {
        #[test]
        fn $name() {
            assert_eq!($value.as_str(), $expected);
        }
    };
}

level_case!(trace_as_str, Level::Trace, "trace");
level_case!(debug_as_str, Level::Debug, "debug");
level_case!(info_as_str, Level::Info, "info");
level_case!(warn_as_str, Level::Warn, "warn");
level_case!(error_as_str, Level::Error, "error");
level_case!(fatal_as_str, Level::Fatal, "fatal");

#[test]
fn level_display_uses_as_str() {
    assert_eq!(Level::Info.to_string(), "info");
    assert_eq!(Level::Fatal.to_string(), "fatal");
}

#[test]
fn level_ordering_trace_is_lowest() {
    assert!(Level::Trace < Level::Debug);
    assert!(Level::Debug < Level::Info);
    assert!(Level::Info < Level::Warn);
    assert!(Level::Warn < Level::Error);
    assert!(Level::Error < Level::Fatal);
}

#[test]
fn level_equality_compares_same_values() {
    assert_eq!(Level::Warn, Level::Warn);
    assert_ne!(Level::Warn, Level::Error);
}

#[test]
fn level_copy_is_supported() {
    let a = Level::Info;
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn level_hash_consistency() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut h1 = DefaultHasher::new();
    Level::Debug.hash(&mut h1);
    let h1 = h1.finish();

    let mut h2 = DefaultHasher::new();
    Level::Debug.hash(&mut h2);
    let h2 = h2.finish();

    assert_eq!(h1, h2);
}

#[test]
fn from_str_accepts_debug_lowercase() {
    assert_eq!("debug".parse::<Level>().unwrap(), Level::Debug);
}

#[test]
fn from_str_accepts_trace_lowercase() {
    assert_eq!("trace".parse::<Level>().unwrap(), Level::Trace);
}

#[test]
fn from_str_accepts_info_lowercase() {
    assert_eq!("info".parse::<Level>().unwrap(), Level::Info);
}

#[test]
fn from_str_accepts_warn_lowercase() {
    assert_eq!("warn".parse::<Level>().unwrap(), Level::Warn);
}

#[test]
fn from_str_accepts_error_lowercase() {
    assert_eq!("error".parse::<Level>().unwrap(), Level::Error);
}

#[test]
fn from_str_accepts_fatal_lowercase() {
    assert_eq!("fatal".parse::<Level>().unwrap(), Level::Fatal);
}

#[test]
fn from_str_accepts_uppercase_variants() {
    assert_eq!("INFO".parse::<Level>().unwrap(), Level::Info);
    assert_eq!("ERROR".parse::<Level>().unwrap(), Level::Error);
}

#[test]
fn from_str_accepts_mixed_case_variants() {
    assert_eq!("Warn".parse::<Level>().unwrap(), Level::Warn);
    assert_eq!("FaTaL".parse::<Level>().unwrap(), Level::Fatal);
}

#[test]
fn from_str_rejects_unknown_value() {
    let result = "verbose".parse::<Level>();
    assert!(result.is_err());
}

#[test]
fn from_str_rejects_empty_string() {
    assert!("".parse::<Level>().is_err());
}

#[test]
fn to_log_level_maps_trace() {
    assert_eq!(Level::Trace.to_log_level(), log::Level::Trace);
}

#[test]
fn to_log_level_maps_debug() {
    assert_eq!(Level::Debug.to_log_level(), log::Level::Debug);
}

#[test]
fn to_log_level_maps_info() {
    assert_eq!(Level::Info.to_log_level(), log::Level::Info);
}

#[test]
fn to_log_level_maps_warn() {
    assert_eq!(Level::Warn.to_log_level(), log::Level::Warn);
}

#[test]
fn to_log_level_maps_error() {
    assert_eq!(Level::Error.to_log_level(), log::Level::Error);
}

#[test]
fn to_log_level_maps_fatal_to_error() {
    assert_eq!(Level::Fatal.to_log_level(), log::Level::Error);
}

#[test]
fn from_log_level_maps_trace() {
    assert_eq!(Level::from_log_level(log::Level::Trace), Level::Trace);
}

#[test]
fn from_log_level_maps_debug() {
    assert_eq!(Level::from_log_level(log::Level::Debug), Level::Debug);
}

#[test]
fn from_log_level_maps_info() {
    assert_eq!(Level::from_log_level(log::Level::Info), Level::Info);
}

#[test]
fn from_log_level_maps_warn() {
    assert_eq!(Level::from_log_level(log::Level::Warn), Level::Warn);
}

#[test]
fn from_log_level_maps_error() {
    assert_eq!(Level::from_log_level(log::Level::Error), Level::Error);
}

#[test]
fn from_log_level_maps_error_to_error() {
    assert_eq!(Level::from_log_level(log::Level::Error), Level::Error);
}

#[test]
fn level_is_ord_compatible_with_sorting() {
    let mut values = [
        Level::Fatal,
        Level::Trace,
        Level::Info,
        Level::Warn,
        Level::Debug,
        Level::Error,
    ];
    values.sort();
    assert_eq!(
        values,
        [
            Level::Trace,
            Level::Debug,
            Level::Info,
            Level::Warn,
            Level::Error,
            Level::Fatal
        ]
    );
}

#[test]
fn level_is_hashable_with_sets() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Level::Trace);
    set.insert(Level::Debug);
    set.insert(Level::Info);
    assert_eq!(set.len(), 3);
}
