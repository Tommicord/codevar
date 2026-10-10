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
//! The crate mirrors the API of libwayland ([`client`] and
//! [`server`]) over pluggable transports and pollers so it can run
//! without an operating system socket layer.
//!
//! # Example
//!
//! ```
//! use codevar_wayland::{WlClientDisplay, WlTransport, WlResult};
//! # fn demo<T: WlTransport>(transport: T) -> WlResult<()> {
//! let mut display = WlClientDisplay::connect(transport)?;
//! let _registry = display.get_registry()?;
//! # Ok(())
//! # }
//! ```

#![cfg_attr(not(test), no_std)]

extern crate alloc;

mod client;
mod conn;
mod core;
mod dmabuf;
mod drm_syncobj;
mod error;
mod evloop;
mod handle;
mod server;
#[cfg(unix)]
mod unix;
mod xdg_shell;

pub use client::{WlClientDisplay, WlProxyId, WlRegistryEvent};
pub use conn::{WlClosure, WlConnection, WlHandle, WlTransport, reserve_new_ids};
pub use core::*;
pub use dmabuf::*;
pub use drm_syncobj::*;
pub use error::{WlError, WlProtocolError, WlResult};
pub use evloop::{WlClock, WlEventLoop, WlEventSourceId, WlPollEntry, WlPollEvents, WlPoller};
pub use handle::{
    CALLBACK_DONE, CALLBACK_INTERFACE, MAX_CLOSURE_ARGS, MAX_MESSAGE_WORDS, WlArgument, WlArray,
    WlDisplayError, WlFd, WlFixed, WlInterface, WlList, WlMap, WlMessage, WlObject, WlSignal,
};
pub use server::{WlClient, WlClientId, WlResource, WlServerDisplay, WlTaskQueue};
#[cfg(unix)]
pub use unix::{WlUnixPoller, WlUnixTransport};
pub use xdg_shell::*;
