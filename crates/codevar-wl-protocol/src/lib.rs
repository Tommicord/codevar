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

//! Wayland protocol implementation
//!
//! The crate mirrors the API of libwayland ([`wl_client`] and
//! [`wl_server`]) over pluggable transports and pollers so it can run
//! without an operating system socket layer.
//!
//! # Example
//!
//! ```
//! use codevar_wl_protocol::{WlClientDisplay, WlTransport, WlResult};
//! # fn demo<T: WlTransport>(transport: T) -> WlResult<()> {
//! let mut display = WlClientDisplay::connect(transport)?;
//! let _registry = display.get_registry()?;
//! # Ok(())
//! # }
//! ```

#![cfg_attr(not(test), no_std)]

extern crate alloc;

mod wl_client;
mod wl_conn;
mod wl_core;
mod wl_dmabuf;
mod wl_drm_syncobj;
mod wl_error;
mod wl_evloop;
mod wl_handle;
mod wl_server;
#[cfg(unix)]
mod wl_unix;
mod wl_xdg_shell;

pub use wl_client::{WlClientDisplay, WlProxyId, WlRegistryEvent};
pub use wl_conn::{WlClosure, WlConnection, WlTransport, reserve_new_ids};
pub use wl_core::*;
pub use wl_dmabuf::*;
pub use wl_drm_syncobj::*;
pub use wl_error::{WlError, WlProtocolError, WlResult};
pub use wl_evloop::{WlClock, WlEventLoop, WlEventSourceId, WlPollEntry, WlPollEvents, WlPoller};
pub use wl_handle::{
    CALLBACK_DONE, CALLBACK_INTERFACE, MAX_CLOSURE_ARGS, MAX_MESSAGE_WORDS, WlArgument, WlArray,
    WlDisplayError, WlFixed, WlInterface, WlList, WlMap, WlMessage, WlObject, WlSignal,
};
pub use wl_server::{WlClient, WlClientId, WlResource, WlServerDisplay, WlTaskQueue};
#[cfg(unix)]
pub use wl_unix::{WlUnixPoller, WlUnixTransport};
pub use wl_xdg_shell::*;
