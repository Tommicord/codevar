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

#![cfg_attr(not(test), no_std)]
extern crate alloc;

pub mod xdp_account;
pub mod xdp_clipboard;
pub mod xdp_dynamic_launcher;
pub mod xdp_flatpak;
pub mod xdp_memory_monitor;
pub mod xdp_network_monitor;
pub mod xdp_notification;
pub mod xdp_proxy_resolver;
pub mod xdp_registry;
pub mod xdp_remote_desktop;
pub mod xdp_screenshot;
pub mod xdp_trash;
pub mod xdp_uri;
pub mod xdp_usb;
pub mod xdp_wallpaper;

pub mod xdp_app_info;
pub mod xdp_documents;
pub mod xdp_error;
pub mod xdp_method_info;
pub mod xdp_permissions;
pub mod xdp_portal_config;
pub mod xdp_utils;
pub mod xdp_validate;
