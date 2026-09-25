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

pub mod portal_clipboard;
pub mod portal_account;
pub mod portal_network_monitor;
pub mod portal_memory_monitor;
pub mod portal_proxy_resolver;
pub mod portal_remote_desktop;
pub mod portal_usb;
pub mod portal_trash;
pub mod portal_wallpaper;
pub mod portal_registry;
pub mod portal_flatpak;
pub mod portal_dynamic_launcher;
pub mod portal_notification;
pub mod portal_uri;
pub mod portal_screenshot;