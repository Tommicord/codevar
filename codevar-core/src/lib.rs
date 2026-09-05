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

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(unsafe_code)]

extern crate alloc;
extern crate core;

#[cfg(target_os = "android")]
pub mod android;
pub mod base;
pub mod edit;
pub mod fcware;
pub mod logtrace;
pub mod timeutil;

#[cfg(target_os = "android")]
pub use android::{
    CallbackManager, InputCallback, InputEvent, KeyAction, KeyEvent, MouseEvent,
    MouseEventType,
};

pub trait NamedEnvDir {
    fn get(&self) -> &str;
}

#[derive(Clone, Debug, Copy)]
pub struct EnvDir(&'static str);

#[derive(Clone, Debug, Copy)]
pub struct LazyEnvDir<F>(F);

impl NamedEnvDir for EnvDir {
    fn get(&self) -> &str {
        self.0
    }
}
impl<F> NamedEnvDir for LazyEnvDir<F>
where
    F: Fn() -> &'static str + Sync,
{
    fn get(&self) -> &'static str {
        (self.0)()
    }
}

fn env_dir_concat<L, R>(left_dir: L, right_dir: R) -> &'static str
where
    R: NamedEnvDir + 'static,
    L: NamedEnvDir + 'static,
{
    let left_bytes: &[u8] = left_dir.get().as_bytes();
    let right_bytes: &[u8] = right_dir.get().as_bytes();
    let byte_count: usize = left_bytes.len() + right_bytes.len() + 1;

    unsafe {
        let mut buffer: Vec<_> = Vec::with_capacity(byte_count);
        std::ptr::copy_nonoverlapping(
            left_bytes.as_ptr(),
            buffer.as_mut_ptr(),
            left_bytes.len(),
        );
        buffer[left_bytes.len()] = std::path::MAIN_SEPARATOR as u8;

        std::ptr::copy_nonoverlapping(
            right_bytes.as_ptr(),
            buffer.as_mut_ptr().add(buffer.len()),
            right_bytes.len(),
        );
        let dir_str = str::from_utf8_unchecked(buffer.as_slice());
        let dir: &'static str = Box::leak(dir_str.to_string().into_boxed_str());
        dir
    }
}

#[cfg(target_os = "linux")]
const fn get_bin_dir() -> &'static str {
    "/usr/bin/"
}

#[cfg(target_os = "windows")]
const fn get_bin_dir() -> &'static str {
    match std::env::var_os("ProgramFiles") {
        Some(val) => val,
        None => {
            use std::ffi::OsString;
            use std::os::windows::ffi::OsStringExt;
            use windows::Win32::Foundation::System::Com::{
                COINIT_APARTMENTTHREADED, CoInitializeEx, CoTaskMemFree,
            };
            use windows::Win32::UI::Shell::{
                FOLDERID_ProgramFiles, KF_FLAG_DEFAULT, SHGetKnownFolderPath,
            };
            use windows::core::PWSTR;

            unsafe {
                let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
                let mut path: PWSTR = PWSTR::null();
                SHGetKnownFolderPath(
                    &FOLDERID_ProgramFiles,
                    KF_FLAG_DEFAULT,
                    None,
                    &mut path,
                );
                let path_wide = unsafe { path.as_wide() };
                let path_str = OsString::from_wide(path_wide)
                    .to_string_lossy()
                    .into_owned();
                CoTaskMemFree(Some(path.as_ptr() as *const _));
                Ok(&path_str)
            }
        }
    }
}

#[cfg(target_os = "macos")]
const fn get_bin_dir() -> &'static str {
    "/usr/local/bin/"
}

pub static ENV_NAME: &'static str = "codevar";

#[cfg(any(target_os = "windows", target_os = "unix"))]
pub static ENV_SYSTEM_BIN_DIR: &(dyn NamedEnvDir + Sync) = { &EnvDir(get_bin_dir()) };

#[cfg(any(target_os = "windows", target_os = "unix"))]
pub static ENV_DIR: &(dyn NamedEnvDir + Sync) = {
    let env_bin = EnvDir(ENV_SYSTEM_BIN_DIR.get());
    let env_suffix = EnvDir(ENV_NAME);
    &LazyEnvDir(move || env_dir_concat(env_bin, env_suffix))
};

pub static ENV_EXE_BIN_DIR: &(dyn NamedEnvDir + Sync) = {
    let env_prefix = EnvDir(ENV_NAME);
    let env_bin = EnvDir("bin");
    &LazyEnvDir(move || env_dir_concat(env_prefix, env_bin))
};
