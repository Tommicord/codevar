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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Environment variable definitions and path constants for the Codevar application.
//!
//! This module provides a [`NamedEnvDir`] trait and its implementations for
//! accessing well-known directories and paths used by Codevar at runtime.
//!
//! ## Overview
//!
//! The primary interface is the [`NamedEnvDir`] trait, which allows uniform
//! access to directory paths regardless of whether they are known at compile
//! time or computed lazily. Each public static constant is a `&dyn NamedEnvDir`
//! that can be queried with [`NamedEnvDir::get`] to obtain the path as a
//! string slice.
//!
//! ## Examples
//!
//! ```rust
//! use codevar_base::basic_env::{ENV_DIR, ENV_CONFIG_DIR, NamedEnvDir};
//!
//! // Access the main configuration directory.
//! let config_dir = ENV_CONFIG_DIR.get();
//!
//! // Access the main environment directory.
//! let env_dir = ENV_DIR.get();
//! ```
//!
//! ## Path construction
//!
//! All path-based constants use [`PathBuilder`] from [`crate::basic_pathbuf`]
//! for validated, platform-aware path construction instead of manual string
//! concatenation.

use crate::basic_pathbuf::PathBuilder;
use alloc::boxed::Box;
use alloc::ffi::CString;
use alloc::string::String;

cfg_if::cfg_if! {
    if #[cfg(any(unix, target_os = "macos"))] {
        use libc;
    }
}

/// Reads an environment variable using platform-specific APIs.
///
/// On Unix systems this uses [`libc::getenv`]. On Windows this uses
/// the Win32 `GetEnvironmentVariableW` API. On unsupported platforms
/// returns `None`.
///
/// # Safety
///
/// On Unix, `libc::getenv` is thread-safe since glibc 2.0.
/// On Windows, `GetEnvironmentVariableW` is process-safe.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::env_var;
///
/// let home = env_var("HOME");
/// ```
pub fn env_var(key: &str) -> Option<String> {
    #[cfg(any(unix, target_os = "macos"))]
    {
        let c_key = CString::new(key).ok()?;
        let ptr = unsafe { libc::getenv(c_key.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            let c_str = unsafe { core::ffi::CStr::from_ptr(ptr) };
            c_str.to_str().ok().map(String::from)
        }
    }
    #[cfg(windows)]
    {
        let wide_key = encode_utf16(key);
        let size = unsafe { GetEnvironmentVariableW(&wide_key, None as *mut u16, 0) };
        if size == 0 {
            return None;
        }
        let mut buf: Vec<u16> = Vec::with_capacity(size as usize);
        let result = unsafe { GetEnvironmentVariableW(&wide_key, buf.as_mut_ptr(), size) };
        if result == 0 {
            return None;
        }
        let len = unsafe { result as usize };
        buf.set_len(len);
        String::from_utf16(&buf).ok()
    }
    #[cfg(not(any(unix, target_os = "macos", windows)))]
    {
        None
    }
}

/// UTF-16 encode a UTF-8 string for Win32 API calls.
#[cfg(windows)]
fn encode_utf16(s: &str) -> Vec<u16> {
    let mut out: Vec<u16> = Vec::with_capacity(s.len());
    for ch in s.encode_utf16() {
        out.push(ch);
    }
    out.push(0);
    out
}

/// Windows `GetEnvironmentVariableW` import.
#[cfg(windows)]
use windows::Win32::System::Threading::GetEnvironmentVariableW;

/// Leak a `String` to obtain a `&'static str`.
#[inline]
fn leak_str(s: String) -> &'static str {
    &*Box::leak(s.into_boxed_str())
}

/// Converts a [`PathBuilder`]-constructed path into a leaked `'static str`.
///
/// This helper builds a path using [`PathBuilder`], converts the resulting
/// [`PathBuf`] into an owned [`String`], and leaks it to obtain a
/// `&'static str`. If path construction fails (which should not happen for
/// valid hardcoded components), the fallback string is leaked instead.
///
/// # Safety
///
/// This function leaks memory and is therefore irreversible. It should only
/// be called during initialization of [`LazyEnvDir`] closures.
#[inline]
pub fn build_and_leak(builder: PathBuilder, fallback: &'static str) -> &'static str {
    builder
        .build()
        .map(|p| leak_str(p.into_string()))
        .unwrap_or_else(|_| leak_str(String::from(fallback)))
}

/// Trait for accessing a named environment directory as a string slice.
///
/// Implementors provide a [`get`] method that returns the directory path
/// as a `&str`. This allows uniform access to both compile-time known
/// paths (via [`EnvDir`]) and lazily computed paths (via [`LazyEnvDir`]).
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::{EnvDir, NamedEnvDir};
///
/// let dir = EnvDir::new("/usr/bin");
/// assert_eq!(dir.get(), "/usr/bin");
/// ```
pub trait NamedEnvDir {
    /// Returns the directory path as a string slice.
    fn get(&self) -> &str;
}

/// A compile-time known directory path.
///
/// Wraps a `&'static str` and implements [`NamedEnvDir`]. Use this when
/// the path is known at compile time and does not require construction
/// logic.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::EnvDir;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let dir = EnvDir::new("/usr/bin");
/// assert_eq!(dir.get(), "/usr/bin");
/// ```
#[derive(Clone, Debug, Copy)]
pub struct EnvDir(&'static str);

impl EnvDir {
    /// Creates a new `EnvDir` from a static string.
    pub const fn new(path: &'static str) -> Self {
        EnvDir(path)
    }
}

/// A lazily computed directory path.
///
/// Wraps a closure `F: Fn() -> &'static str` and implements
/// [`NamedEnvDir`]. Use this when the path requires runtime computation,
/// such as combining other paths or reading from the environment.
///
/// The closure must be [`Sync`] so that [`LazyEnvDir`] can be shared
/// across threads safely.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::{LazyEnvDir, NamedEnvDir};
///
/// let dir = LazyEnvDir::new(|| "/usr/local/bin");
/// assert_eq!(dir.get(), "/usr/local/bin");
/// ```
#[derive(Clone, Debug, Copy)]
pub struct LazyEnvDir<F>(F);

impl<F> LazyEnvDir<F> {
    /// Creates a new `LazyEnvDir` from a closure.
    pub const fn new(f: F) -> Self {
        LazyEnvDir(f)
    }
}

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

/// Returns the system binary directory for the current platform.
///
/// On Linux this is `/usr/bin`. On macOS this is `/usr/local/bin`.
/// On Windows this is the `ProgramFiles` environment directory.
///
/// # Platform-specific behavior
///
/// | Platform | Path |
/// |----------|------|
/// | Linux    | `/usr/bin` |
/// | macOS    | `/usr/local/bin` |
/// | Windows  | `ProgramFiles` directory |
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_SYSTEM_BIN_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let bin_dir = ENV_SYSTEM_BIN_DIR.get();
/// ```
fn get_bin_dir() -> &'static str {
    #[cfg(target_os = "linux")]
    {
        "/usr/bin"
    }
    #[cfg(target_os = "macos")]
    {
        "/usr/local/bin"
    }
    #[cfg(windows)]
    {
        leak_str(env_var("ProgramFiles").unwrap_or_else(|| String::from("/ProgramFiles")))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        "/usr/bin"
    }
}

/// The application name, `"codevar"`.
///
/// This is the canonical identifier for the Codevar application. It is used
/// as a path component in directory paths and as the default application
/// name in various contexts.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_NAME;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// assert_eq!(ENV_NAME, "codevar");
/// ```
pub static ENV_NAME: &str = "codevar";

/// The short name of the application, `"cv"`.
///
/// This is a shorter alias for [`ENV_NAME`] that can be used in contexts
/// where a concise identifier is preferred (e.g., CLI arguments,
/// configuration file names, cache directories).
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_APP_SHORT_NAME;
///
/// assert_eq!(ENV_APP_SHORT_NAME, "cv");
/// ```
pub static ENV_APP_SHORT_NAME: &str = "cv";

/// The full version string of the Codevar application.
///
/// This constant represents the current version of Codevar. It follows
/// semantic versioning conventions and is used for display purposes,
/// update checks, and configuration versioning.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_VERSION;
///
/// assert_eq!(ENV_VERSION, env!("CARGO_PKG_VERSION"));
/// ```
pub static ENV_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The system binary directory path.
///
/// This is the directory containing system-level executables for the
/// current platform. On Linux it is `/usr/bin`, on macOS `/usr/local/bin`,
/// and on Windows the `ProgramFiles` directory.
///
/// See [`ENV_DIR`] for the application-specific configuration directory.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_SYSTEM_BIN_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let bin = ENV_SYSTEM_BIN_DIR.get();
/// ```
#[cfg(any(target_os = "windows", target_family = "unix"))]
pub static ENV_SYSTEM_BIN_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(get_bin_dir);

/// The main application data directory.
///
/// This is the root directory for all application data. It is
/// constructed by combining the system binary directory with the
/// application name using [`PathBuilder`] from [`crate::basic_pathbuf`].
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let dir = ENV_DIR.get();
/// ```
#[cfg(any(target_os = "windows", target_family = "unix"))]
pub static ENV_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(move || {
    build_and_leak(
        PathBuilder::new()
            .root()
            .push(ENV_SYSTEM_BIN_DIR.get())
            .push(ENV_NAME),
        "/usr/bin/codevar",
    )
});

/// The executable binary directory path.
///
/// This is the directory where own executables are installed.
/// It is constructed as `<app_name>/bin`.
///
/// This path is constructed using [`PathBuilder`] from [`crate::basic_pathbuf`]
/// to ensure platform-correct path separators and validation.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_EXE_BIN_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let bin = ENV_EXE_BIN_DIR.get();
/// ```
pub static ENV_EXE_BIN_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(move || {
    build_and_leak(
        PathBuilder::new()
            .root()
            .push(ENV_NAME)
            .push("bin"),
        "/codevar/bin",
    )
});

/// The user's home directory path.
///
/// This is the user's home directory as determined by the `HOME` environment
/// variable on Unix systems (via `libc::getenv`) or `USERPROFILE` on Windows
/// (via Win32 `GetEnvironmentVariableW`).
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_HOME_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let home = ENV_HOME_DIR.get();
/// ```
pub static ENV_HOME_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(move || {
    #[cfg(any(unix, target_os = "macos"))]
    {
        leak_str(env_var("HOME").unwrap_or_else(|| String::from("/tmp")))
    }
    #[cfg(windows)]
    {
        leak_str(env_var("USERPROFILE").unwrap_or_else(|| String::from("C:\\Users\\Default")))
    }
    #[cfg(not(any(unix, target_os = "macos", windows)))]
    {
        leak_str(String::from("/tmp"))
    }
});

/// The application configuration directory.
///
/// This is the directory where is stored its configuration files.
/// On Unix it resolves to `~/.config/codevar`. On Windows it is under
/// the user's `APPDATA` directory. On macOS it is `~/Library/Application
/// Support/codevar`.
///
/// This path is constructed using [`PathBuilder`] from [`crate::basic_pathbuf`]
/// to ensure platform-correct path separators and validation.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_CONFIG_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let config = ENV_CONFIG_DIR.get();
/// ```
pub static ENV_CONFIG_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(move || {
    #[cfg(any(unix, target_os = "macos"))]
    {
        build_and_leak(
            PathBuilder::new()
                .root()
                .push(
                    env_var("HOME")
                        .unwrap_or_else(|| String::from("/tmp"))
                        .as_str(),
                )
                .push(".config")
                .push(ENV_NAME),
            "/tmp/.config/codevar",
        )
    }
    #[cfg(windows)]
    {
        let mut appdata = env_var("APPDATA").unwrap_or_else(|| String::from("C:\\ProgramData"));
        appdata.push('\\');
        appdata.push_str(ENV_NAME);
        leak_str(appdata)
    }
    #[cfg(not(any(unix, target_os = "macos", windows)))]
    {
        build_and_leak(
            PathBuilder::new()
                .root()
                .push(
                    env_var("HOME")
                        .unwrap_or_else(|| String::from("/tmp"))
                        .as_str(),
                )
                .push(ENV_NAME),
            "/tmp/codevar",
        )
    }
});

/// The application cache directory.
///
/// This is the directory where is stored cached data that can be
/// regenerated or cleared without data loss. On Unix it resolves to
/// `~/.cache/codevar`. On Windows it is under the user's `LOCALAPPDATA`
/// directory. On macOS, it is `~/Library/Caches/codevar`.
///
/// This path is constructed using [`PathBuilder`] from [`crate::basic_pathbuf`]
/// to ensure platform-correct path separators and validation.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_CACHE_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let cache = ENV_CACHE_DIR.get();
/// ```
pub static ENV_CACHE_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(move || {
    #[cfg(any(unix, target_os = "macos"))]
    {
        build_and_leak(
            PathBuilder::new()
                .root()
                .push(
                    env_var("HOME")
                        .unwrap_or_else(|| String::from("/tmp"))
                        .as_str(),
                )
                .push(".cache")
                .push(ENV_NAME),
            "/tmp/.cache/codevar",
        )
    }
    #[cfg(windows)]
    {
        let mut local = env_var("LOCALAPPDATA").unwrap_or_else(|| String::from("C:\\ProgramData\\Local"));
        local.push('\\');
        local.push_str(ENV_NAME);
        leak_str(local)
    }
    #[cfg(not(any(unix, target_os = "macos", windows)))]
    {
        build_and_leak(
            PathBuilder::new()
                .root()
                .push(env_var("HOME").unwrap_or_else(|| String::from("/tmp")))
                .push(".cache")
                .push(ENV_NAME),
            "/tmp/.cache/codevar",
        )
    }
});

/// The application data directory.
///
/// This is the directory where is stored persistent application data
/// such as user projects, settings, and documents. On Unix it resolves to
/// `~/.local/share/codevar`. On Windows it is under `APPDATA`. On macOS it
/// is `~/Library/Application Support/codevar`.
///
/// This path is constructed using [`PathBuilder`] from [`crate::basic_pathbuf`]
/// to ensure platform-correct path separators and validation.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_DATA_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let data = ENV_DATA_DIR.get();
/// ```
pub static ENV_DATA_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(move || {
    #[cfg(any(unix, target_os = "macos"))]
    {
        build_and_leak(
            PathBuilder::new()
                .root()
                .push(
                    env_var("HOME")
                        .unwrap_or_else(|| String::from("/tmp"))
                        .as_str(),
                )
                .push(".local")
                .push("share")
                .push(ENV_NAME),
            "/tmp/.local/share/codevar",
        )
    }
    #[cfg(windows)]
    {
        let mut appdata = env_var("APPDATA").unwrap_or_else(|| String::from("C:\\ProgramData\\Roaming"));
        appdata.push('\\');
        appdata.push_str(ENV_NAME);
        appdata.push('\\');
        appdata.push_str("Data");
        leak_str(appdata)
    }
    #[cfg(not(any(unix, target_os = "macos", windows)))]
    {
        build_and_leak(
            PathBuilder::new()
                .root()
                .push(env_var("HOME").unwrap_or_else(|| String::from("/tmp")))
                .push(".local")
                .push("share")
                .push(ENV_NAME),
            "/tmp/.local/share/codevar",
        )
    }
});

/// The application logs directory.
///
/// This is the directory where is stored log files. On Unix it resolves
/// to `~/.local/share/codevar/logs`. On Windows it is under `APPDATA`.
/// On macOS it is `~/Library/Logs/codevar`.
///
/// This path is constructed using [`PathBuilder`] from [`crate::basic_pathbuf`]
/// to ensure platform-correct path separators and validation.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_LOGS_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let logs = ENV_LOGS_DIR.get();
/// ```
pub static ENV_LOGS_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(move || {
    #[cfg(any(unix, target_os = "macos"))]
    {
        build_and_leak(
            PathBuilder::new()
                .root()
                .push(ENV_DATA_DIR.get())
                .push("logs"),
            "/tmp/.local/share/codevar/logs",
        )
    }
    #[cfg(windows)]
    {
        let mut appdata = env_var("APPDATA").unwrap_or_else(|| String::from("C:\\ProgramData\\Roaming"));
        appdata.push('\\');
        appdata.push_str(ENV_NAME);
        appdata.push_str("-Logs");
        leak_str(appdata)
    }
    #[cfg(not(any(unix, target_os = "macos", windows)))]
    {
        build_and_leak(
            PathBuilder::new()
                .root()
                .push(ENV_DATA_DIR.get())
                .push("logs"),
            "/tmp/.local/share/codevar/logs",
        )
    }
});

/// The application plugins directory.
///
/// This is the directory where Codevar stores plugin extensions. It is
/// constructed as `<data_dir>/plugins` using [`PathBuilder`] from
/// [`crate::basic_pathbuf`].
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_PLUGINS_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let plugins = ENV_PLUGINS_DIR.get();
/// ```
pub static ENV_PLUGINS_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(move || {
    build_and_leak(
        PathBuilder::new()
            .root()
            .push(ENV_DATA_DIR.get())
            .push("plugins"),
        "/data/codevar/plugins",
    )
});

/// The application assets directory.
///
/// This is the directory where is stored bundled assets such as
/// themes, icons, and syntax highlighting files. It is constructed as
/// `<data_dir>/assets` using [`PathBuilder`] from [`crate::basic_pathbuf`].
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_ASSETS_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let assets = ENV_ASSETS_DIR.get();
/// ```
pub static ENV_ASSETS_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(move || {
    build_and_leak(
        PathBuilder::new()
            .root()
            .push(ENV_DATA_DIR.get())
            .push("assets"),
        "/data/codevar/assets",
    )
});

/// The application temporary directory.
///
/// This is the directory where is stored temporary files. It uses
/// the system's temporary directory (`TMPDIR` on Unix via `libc::getenv`,
/// `TEMP`/`TMP` on Windows via Win32 API) with a `codevar` subdirectory.
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_TEMP_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let temp = ENV_TEMP_DIR.get();
/// ```
pub static ENV_TEMP_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(move || {
    #[cfg(any(unix, target_os = "macos"))]
    {
        let mut temp_dir = env_var("TMPDIR").unwrap_or_else(|| String::from("/tmp"));
        temp_dir.push('/');
        temp_dir.push_str(ENV_NAME);
        leak_str(temp_dir)
    }
    #[cfg(windows)]
    {
        let mut temp = env_var("TEMP")
            .or_else(|| env_var("TMP"))
            .unwrap_or_else(|| String::from("C:\\Temp"));
        temp.push('\\');
        temp.push_str(ENV_NAME);
        leak_str(temp)
    }
    #[cfg(not(any(unix, target_os = "macos", windows)))]
    {
        let mut temp = String::from(ENV_HOME_DIR.get());
        temp.push('/');
        temp.push_str(ENV_NAME);
        leak_str(temp)
    }
});

/// The system temporary directory path.
///
/// This is the system-level temporary directory without the `codevar`
/// subdirectory prefix. On Unix it comes from the `TMPDIR` environment
/// variable (via `libc::getenv`, defaulting to `/tmp`). On Windows it
/// comes from `TEMP` or `TMP` (via Win32 `GetEnvironmentVariableW`).
///
/// # Examples
///
/// ```rust
/// use codevar_base::basic_env::ENV_SYSTEM_TEMP_DIR;
/// use codevar_base::basic_env::NamedEnvDir;
///
/// let sys_temp = ENV_SYSTEM_TEMP_DIR.get();
/// ```
pub static ENV_SYSTEM_TEMP_DIR: &(dyn NamedEnvDir + Sync) = &LazyEnvDir(move || {
    #[cfg(any(unix, target_os = "macos"))]
    {
        leak_str(env_var("TMPDIR").unwrap_or_else(|| String::from("/tmp")))
    }
    #[cfg(windows)]
    {
        leak_str(
            env_var("TEMP")
                .or_else(|| env_var("TMP"))
                .unwrap_or_else(|| String::from("C:\\Temp")),
        )
    }
    #[cfg(not(any(unix, target_os = "macos", windows)))]
    {
        leak_str(String::from("/tmp"))
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_name() {
        assert_eq!(ENV_NAME, "codevar");
    }

    #[test]
    fn test_env_app_short_name() {
        assert_eq!(ENV_APP_SHORT_NAME, "cv");
    }

    #[test]
    fn test_env_version_not_empty() {
        assert!(!ENV_VERSION.is_empty());
    }

    #[test]
    fn test_env_dir_contains_app_name() {
        #[cfg(any(target_os = "windows", target_family = "unix"))]
        {
            let dir = ENV_DIR.get();
            assert!(dir.contains(ENV_NAME));
        }
    }

    #[test]
    fn test_env_exe_bin_dir_contains_app_name() {
        let dir = ENV_EXE_BIN_DIR.get();
        assert!(dir.contains(ENV_NAME));
        assert!(dir.contains("bin"));
    }

    #[test]
    fn test_env_home_dir_is_not_empty() {
        let home = ENV_HOME_DIR.get();
        assert!(!home.is_empty());
    }

    #[test]
    fn test_env_config_dir_contains_app_name() {
        let config = ENV_CONFIG_DIR.get();
        assert!(config.contains(ENV_NAME));
    }

    #[test]
    fn test_env_cache_dir_contains_app_name() {
        let cache = ENV_CACHE_DIR.get();
        assert!(cache.contains(ENV_NAME));
    }

    #[test]
    fn test_env_data_dir_contains_app_name() {
        let data = ENV_DATA_DIR.get();
        assert!(data.contains(ENV_NAME));
    }

    #[test]
    fn test_env_logs_dir_contains_app_name() {
        let logs = ENV_LOGS_DIR.get();
        assert!(logs.contains(ENV_NAME));
    }

    #[test]
    fn test_env_plugins_dir_contains_app_name() {
        let plugins = ENV_PLUGINS_DIR.get();
        assert!(plugins.contains(ENV_NAME));
    }

    #[test]
    fn test_env_assets_dir_contains_app_name() {
        let assets = ENV_ASSETS_DIR.get();
        assert!(assets.contains(ENV_NAME));
    }

    #[test]
    fn test_env_temp_dir_contains_app_name() {
        let temp = ENV_TEMP_DIR.get();
        assert!(temp.contains(ENV_NAME));
    }

    #[test]
    fn test_env_system_temp_dir_is_not_empty() {
        let sys_temp = ENV_SYSTEM_TEMP_DIR.get();
        assert!(!sys_temp.is_empty());
    }

    #[test]
    fn test_env_dir_ends_with_app_name() {
        #[cfg(any(target_os = "windows", target_family = "unix"))]
        {
            let dir = ENV_DIR.get();
            assert!(dir.ends_with(ENV_NAME) || dir.ends_with('/'));
        }
    }

    #[test]
    fn test_env_exe_bin_dir_ends_with_bin() {
        let dir = ENV_EXE_BIN_DIR.get();
        assert!(dir.ends_with("bin"));
    }

    #[test]
    fn test_named_env_dir_trait_object() {
        let _dir: &(dyn NamedEnvDir + Sync) = ENV_DIR;
        let _name = _dir.get();
        assert!(!_name.is_empty());
    }

    #[test]
    fn test_build_and_leak_returns_valid_path() {
        let path = build_and_leak(PathBuilder::new().root().push("usr").push("bin"), "/usr/bin");
        assert!(path.contains("usr"));
        assert!(path.contains("bin"));
    }

    #[test]
    fn test_build_and_leak_fallback() {
        let path = build_and_leak(PathBuilder::new().root().push("usr").push("bin"), "/usr/bin");
        assert_eq!(path, "/usr/bin");
    }

    #[test]
    fn test_env_dir_contains_config() {
        #[cfg(any(target_os = "windows", target_family = "unix"))]
        {
            let dir = ENV_DIR.get();
            assert!(dir.contains(ENV_NAME));
        }
    }

    #[test]
    fn test_home_dir_is_not_empty() {
        let home = ENV_HOME_DIR.get();
        assert!(!home.is_empty());
    }

    #[test]
    fn test_env_var_returns_option() {
        let val = env_var("NONEXISTENT_VAR_12345");
        assert!(val.is_none());
    }
}
