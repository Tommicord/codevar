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

pub mod account;
pub mod clipboard;
pub mod dynamic_launcher;
pub mod flatpak;
pub mod memory_monitor;
pub mod network_monitor;
pub mod notification;
pub mod proxy_resolver;
pub mod registry;
pub mod remote_desktop;
pub mod screenshot;
pub mod uri;
pub mod usb;
pub mod wallpaper;

pub mod app_info;
pub mod context;
pub mod documents;
pub mod error;
pub mod method_info;
pub mod permissions;
pub mod portal_config;
pub mod request;
pub mod session;
pub mod trash;
pub mod utils;
pub mod validate;
