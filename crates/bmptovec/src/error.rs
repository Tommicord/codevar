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

//! Errors produced by the pipeline stages and the pipeline driver.

use core::fmt;

/// Errors produced by the framework stages and the pipeline driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The input image had zero width or height.
    EmptyImage,
    /// A requested feature is recognized but not yet implemented.
    Unsupported(String),
    /// The run was aborted via a [`crate::progress::CancelToken`].
    Cancelled,
    /// Any other failure, carrying a human-readable message.
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyImage => write!(f, "input image is empty"),
            Error::Unsupported(what) => write!(f, "unsupported: {what}"),
            Error::Cancelled => write!(f, "conversion cancelled"),
            Error::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::Other(msg)
    }
}

impl From<&str> for Error {
    fn from(msg: &str) -> Self {
        Error::Other(msg.to_string())
    }
}
