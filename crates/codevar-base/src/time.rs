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

//! A modular, syscall-based time infrastructure built around [`SystemTime`].
//!
//! This module re-exports the implementation owned by [`codevar_time_core`],
//! so there is exactly one syscall-backed clock in the workspace. That also
//! fixes the embedded behaviour in one place: on targets without an
//! operating system (`target_os = "none"`) a clock query aborts the
//! application through [`codevar_target::abort_unsupported`] rather than
//! inventing a timestamp.
//!
//! Provides:
//! - [`SystemTime`] – a syscall-driven struct with static methods for
//!   retrieving monotonic and realtime time.
//! - Convenience free functions (`secs_to_nanos`, `millis_to_nanos`) for
//!   backward compatibility with existing callers.

pub use codevar_time_core::{SystemTime, TimeError, TimeVal, Timespec};

/// Converts seconds to nanoseconds.
///
/// Convenience wrapper around [`SystemTime::secs_to_nanos()`].
#[inline]
pub const fn secs_to_nanos(secs: u64) -> u64 {
    SystemTime::secs_to_nanos(secs)
}

/// Converts milliseconds to nanoseconds.
///
/// Convenience wrapper around [`SystemTime::millis_to_nanos()`].
#[inline]
pub const fn millis_to_nanos(millis: u64) -> u64 {
    SystemTime::millis_to_nanos(millis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_convenience_wrappers_match_system_time() {
        assert_eq!(secs_to_nanos(2), SystemTime::secs_to_nanos(2));
        assert_eq!(millis_to_nanos(2), SystemTime::millis_to_nanos(2));
    }

    #[test]
    fn system_time_reexport_reports_monotonic() {
        let now = SystemTime::monotonic();
        assert!(now.is_ok(), "monotonic time should be available");
        let tv = now.expect("checked above");
        assert!(tv.secs > 0 || tv.nsecs > 0, "time should be non-zero");
    }
}
