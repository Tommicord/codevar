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

pub mod ws_client;
pub mod ws_connection;
pub mod ws_error;
pub mod ws_frame;
pub mod ws_handshake;
pub mod ws_ids;
pub mod ws_message;
pub mod ws_server;
pub mod ws_stream;
pub mod ws_utf8;

pub use crate::ws_client::ClientConnection as WsClientConnection;
pub use crate::ws_connection::{
    ConnectionConfig as WsConnectionConfig, ConnectionState as WsConnectionState, IoState as WsIoState,
};
pub use crate::ws_error::{WsError, WsResult};
pub use crate::ws_frame::{WsFrame, WsFrameHeader};
pub use crate::ws_handshake::{
    HandshakeRequest, WsClientHandshake, WsHandshakeResponse, WsServerHandshake, accept_key_from_nonce,
    compute_accept_key, generate_key_nonce,
};
pub use crate::ws_ids::{
    DEFAULT_MAX_FRAME_SIZE, DEFAULT_MAX_MESSAGE_SIZE, GUID, Role, VERSION, WsCloseCode, WsOpcode,
};
pub use crate::ws_message::WsMessage;
pub use crate::ws_server::ServerConnection as WsServerConnection;
pub use crate::ws_stream::WebSocketStream as WsWebSocketStream;
