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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Log level definitions for the logger.

use std::fmt;

/// Log level severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    /// Trace level: extremely detailed logging
    Trace = 0,
    /// Debug level: debugging information
    Debug = 1,
    /// Info level: general informational messages
    Info = 2,
    /// Warn level: warning messages
    Warn = 3,
    /// Error level: error messages
    Error = 4,
    /// Fatal level: critical errors that may cause termination
    Fatal = 5,
}

impl Level {
    /// Convert log level to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Level::Trace => "trace",
            Level::Debug => "debug",
            Level::Info => "info",
            Level::Warn => "warn",
            Level::Error => "error",
            Level::Fatal => "fatal",
        }
    }

    /// Convert log level to platform-specific priority.
    #[cfg(target_os = "android")]
    pub fn to_android_priority(&self) -> i32 {
        match self {
            Level::Trace => android_log_sys::LogPriority::Verbose as i32,
            Level::Debug => android_log_sys::LogPriority::Debug as i32,
            Level::Info => android_log_sys::LogPriority::Info as i32,
            Level::Warn => android_log_sys::LogPriority::Warn as i32,
            Level::Error => android_log_sys::LogPriority::Error as i32,
            Level::Fatal => android_log_sys::LogPriority::Fatal as i32,
        }
    }

    /// Convert log level to syslog priority.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn to_syslog_priority(&self) -> i32 {
        match self {
            Level::Trace => libc::LOG_DEBUG,  // LOG_DEBUG
            Level::Debug => libc::LOG_DEBUG,  // LOG_DEBUG
            Level::Info => libc::LOG_INFO,    // LOG_INFO
            Level::Warn => libc::LOG_WARNING, // LOG_WARNING
            Level::Error => libc::LOG_ERR,    // LOG_ERR
            Level::Fatal => libc::LOG_CRIT,   // LOG_CRIT
        }
    }

    /// Convert from log crate Level.
    pub fn from_log_level(level: log::Level) -> Self {
        match level {
            log::Level::Trace => Level::Trace,
            log::Level::Debug => Level::Debug,
            log::Level::Info => Level::Info,
            log::Level::Warn => Level::Warn,
            log::Level::Error => Level::Error,
        }
    }

    /// Convert to log crate Level.
    pub fn to_log_level(&self) -> log::Level {
        match self {
            Level::Trace => log::Level::Trace,
            Level::Debug => log::Level::Debug,
            Level::Info => log::Level::Info,
            Level::Warn => log::Level::Warn,
            Level::Error => log::Level::Error,
            Level::Fatal => log::Level::Error,
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for Level {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "debug" => Ok(Level::Debug),
            "trace" => Ok(Level::Trace),
            "info" => Ok(Level::Info),
            "warn" => Ok(Level::Warn),
            "error" => Ok(Level::Error),
            "fatal" => Ok(Level::Fatal),
            _ => Err(format!("unknown {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Level;
    use std::collections::HashSet;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn every_level_has_expected_text() {
        let cases = [
            (Level::Trace, "trace"),
            (Level::Debug, "debug"),
            (Level::Info, "info"),
            (Level::Warn, "warn"),
            (Level::Error, "error"),
            (Level::Fatal, "fatal"),
        ];
        for (level, expected) in cases {
            assert_eq!(level.as_str(), expected);
        }
    }

    fn display_and_ordering_are_consistent() {
        assert_eq!(Level::Info.to_string(), "info");
        assert!(Level::Trace < Level::Debug);
        assert!(Level::Debug < Level::Info);
        assert!(Level::Info < Level::Warn);
        assert!(Level::Warn < Level::Error);
        assert!(Level::Error < Level::Fatal);
    }

    fn parsing_accepts_case_variants_and_rejects_unknown_values() {
        assert_eq!("debug".parse::<Level>().unwrap(), Level::Debug);
        assert_eq!("INFO".parse::<Level>().unwrap(), Level::Info);
        assert_eq!("FaTaL".parse::<Level>().unwrap(), Level::Fatal);
        assert!("verbose".parse::<Level>().is_err());
        assert!("".parse::<Level>().is_err());
    }

    fn log_level_conversion_maps_all_values() {
        let cases = [
            (Level::Trace, log::Level::Trace),
            (Level::Debug, log::Level::Debug),
            (Level::Info, log::Level::Info),
            (Level::Warn, log::Level::Warn),
            (Level::Error, log::Level::Error),
            (Level::Fatal, log::Level::Error),
        ];
        for (level, expected) in cases {
            assert_eq!(level.to_log_level(), expected);
            if level != Level::Fatal {
                assert_eq!(Level::from_log_level(expected), level);
            }
        }
    }

    fn copy_hash_and_sorting_work() {
        let copied = Level::Info;
        assert_eq!(copied, Level::Info);

        let mut first = DefaultHasher::new();
        Level::Debug.hash(&mut first);
        let mut second = DefaultHasher::new();
        Level::Debug.hash(&mut second);
        assert_eq!(first.finish(), second.finish());

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

        let mut set = HashSet::new();
        set.extend([Level::Trace, Level::Debug, Level::Info]);
        assert_eq!(set.len(), 3);
    }
}
