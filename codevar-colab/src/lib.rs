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

//! User collaboration networking primitives.
//!
//! This module hosts transport and security building blocks used by
//! collaborative editing features, including a self-contained TLS stack
//! and a WebSocket (RFC 6455) protocol implementation.

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(missing_docs)]

pub mod network;
pub mod userclient;
pub mod usersecurity;
pub mod usersession;
pub mod usersystem;
pub mod userview;
