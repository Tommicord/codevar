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

//! The format described in RFC 2822.

/// The format described in [RFC 2822](https://tools.ietf.org/html/rfc2822#section-3.3).
///
/// Example: Fri, 21 Nov 1997 09:55:06 -0600
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rfc2822;
