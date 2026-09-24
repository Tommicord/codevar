//! Copyright 2026 Codevar Project
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

//! Wayland client and server.
//!
//! The crate mirrors the API of libwayland ([`wnd_wl_client`] and
//! [`wnd_wl_server`]) over pluggable transports and pollers so it can run
//! without an operating system socket layer.
//!
//! # Example
//!
//! ```
//! use codevar_ui_core::{WlClientDisplay, WlTransport, WlResult};
//! # fn demo<T: WlTransport>(transport: T) -> WlResult<()> {
//! let mut display = WlClientDisplay::connect(transport)?;
//! let _registry = display.get_registry()?;
//! # Ok(())
//! # }
//! ```

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod wnd_wl_client;
pub mod wnd_wl_conn;
pub mod wnd_wl_error;
pub mod wnd_wl_evloop;
pub mod wnd_wl_handle;
pub mod wnd_wl_server;

pub use wnd_wl_client::WlClientDisplay;
pub use wnd_wl_conn::{WlClosure, WlConnection, WlTransport};
pub use wnd_wl_error::{WlError, WlProtocolError, WlResult};
pub use wnd_wl_evloop::{WlClock, WlEventLoop, WlPoller, WlPollEvents};
pub use wnd_wl_handle::{
    WlArgument, WlArray, WlFixed, WlInterface, WlList, WlMap, WlMessage, WlSignal,
};
pub use wnd_wl_server::WlServerDisplay;
