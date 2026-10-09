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

pub mod client;
pub mod connection;
pub mod error;
pub mod frame;
pub mod handshake;
pub mod ids;
pub mod message;
pub mod server;
pub mod stream;
pub mod utf8;

pub use crate::client::ClientConnection as WsClientConnection;
pub use crate::connection::{
    ConnectionConfig as WsConnectionConfig, ConnectionState as WsConnectionState, IoState as WsIoState,
};
pub use crate::error::{WsError, WsResult};
pub use crate::frame::{WsFrame, WsFrameHeader};
pub use crate::handshake::{
    HandshakeRequest, WsClientHandshake, WsHandshakeResponse, WsServerHandshake, accept_key_from_nonce,
    compute_accept_key, generate_key_nonce,
};
pub use crate::ids::{
    DEFAULT_MAX_FRAME_SIZE, DEFAULT_MAX_MESSAGE_SIZE, GUID, Role, VERSION, WsCloseCode, WsOpcode,
};
pub use crate::message::WsMessage;
pub use crate::server::ServerConnection as WsServerConnection;
pub use crate::stream::WebSocketStream as WsWebSocketStream;
