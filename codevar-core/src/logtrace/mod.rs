//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

pub mod log;
pub mod log_error;
pub(crate) mod log_fmt;
pub mod log_level;
pub(crate) mod log_queue;

pub use log::{
    LogConfigBuilder, LogConfigFlags, LogEntry, LogMessage, Logger, LoggerConfig, debug,
    debug_with_location, default_logger, error, error_with_location, fatal,
    fatal_with_location, info, info_with_location, init_logger,
    log_formatted, logger, remove_logger, set_default_logger, shutdown_all, trace,
    trace_with_location, warn, warn_with_location,
};
pub use log_error::{Error, Result};
pub use log_fmt::{Formattable, Formatter};
pub use log_level::Level;
pub use log_queue::Queue;
