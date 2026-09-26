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

//! Flatpak instance metadata parsing.
//!
//! This module provides utilities to read Flatpak instance information
//! from `~/.var/app/.../.flatpak-instance` metadata directories. It is
//! not a D-Bus portal and does not register on the bus.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::{KeyFile, env_var};

/// Group name for application metadata.
const FLATPAK_METADATA_GROUP_APPLICATION: &str = "Application";
/// Group name for runtime metadata.
const FLATPAK_METADATA_GROUP_RUNTIME: &str = "Runtime";
/// Group name for instance metadata.
const FLATPAK_METADATA_GROUP_INSTANCE: &str = "Instance";

/// Key for instance path.
const FLATPAK_METADATA_KEY_INSTANCE_PATH: &str = "instance-path";
/// Key for instance ID.
const FLATPAK_METADATA_KEY_INSTANCE_ID: &str = "instance-id";
/// Key for app commit.
const FLATPAK_METADATA_KEY_APP_COMMIT: &str = "app-commit";
/// Key for architecture.
const FLATPAK_METADATA_KEY_ARCH: &str = "arch";
/// Key for branch.
const FLATPAK_METADATA_KEY_BRANCH: &str = "branch";
/// Key for application name.
const FLATPAK_METADATA_KEY_NAME: &str = "name";
/// Key for runtime ref.
const FLATPAK_METADATA_KEY_RUNTIME: &str = "runtime";

/// Represents a Flatpak instance with its metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatpakInstance {
    /// The instance ID (directory name).
    pub id: String,
    /// The application ID (or runtime ID if no application).
    pub app_id: String,
    /// The architecture of the application/runtime.
    pub arch: String,
    /// The branch of the application/runtime.
    pub branch: String,
    /// The commit of the application.
    pub commit: String,
    /// The filesystem path to the instance directory.
    pub instance_path: String,
    /// The ref type: "app" or "runtime".
    pub ref_type: String,
}

impl FlatpakInstance {
    /// Creates a new FlatpakInstance by reading metadata from `dir`.
    pub fn new(dir: &str) -> XdpResult<Self> {
        let info_path = format!("{}/info", dir);
        let info_bytes = read_file_bytes(&info_path)?;
        let info_str = core::str::from_utf8(&info_bytes)
            .map_err(|_| PortalError::Failed(format!("Invalid UTF-8 in {}", info_path)))?;

        let key_file = KeyFile::parse(info_str)
            .map_err(|e| PortalError::Failed(format!("Failed to parse {}: {}", info_path, e.message())))?;

        let _instance_id = key_file
            .get(FLATPAK_METADATA_GROUP_INSTANCE, FLATPAK_METADATA_KEY_INSTANCE_ID)
            .ok_or_else(|| {
                PortalError::NotFound(format!(
                    "Missing {} in {}",
                    FLATPAK_METADATA_KEY_INSTANCE_ID, info_path
                ))
            })?;

        let id = dir.rsplit('/').next().unwrap_or("").to_string();

        let (app_id, ref_type) = if key_file.has_group(FLATPAK_METADATA_GROUP_APPLICATION) {
            let name = key_file
                .get(FLATPAK_METADATA_GROUP_APPLICATION, FLATPAK_METADATA_KEY_NAME)
                .ok_or_else(|| {
                    PortalError::NotFound(format!("Missing {} in {}", FLATPAK_METADATA_KEY_NAME, info_path))
                })?;
            (name, "app".to_string())
        } else {
            let runtime = key_file
                .get(FLATPAK_METADATA_GROUP_RUNTIME, FLATPAK_METADATA_KEY_RUNTIME)
                .ok_or_else(|| {
                    PortalError::NotFound(format!(
                        "Missing {} in {}",
                        FLATPAK_METADATA_KEY_RUNTIME, info_path
                    ))
                })?;
            (runtime, "runtime".to_string())
        };

        let arch = key_file
            .get(FLATPAK_METADATA_GROUP_INSTANCE, FLATPAK_METADATA_KEY_ARCH)
            .unwrap_or_default();

        let branch = key_file
            .get(FLATPAK_METADATA_GROUP_INSTANCE, FLATPAK_METADATA_KEY_BRANCH)
            .unwrap_or_default();

        let commit = key_file
            .get(FLATPAK_METADATA_GROUP_INSTANCE, FLATPAK_METADATA_KEY_APP_COMMIT)
            .unwrap_or_default();

        let instance_path = key_file
            .get(
                FLATPAK_METADATA_GROUP_INSTANCE,
                FLATPAK_METADATA_KEY_INSTANCE_PATH,
            )
            .unwrap_or_else(|| dir.to_string());

        Ok(Self {
            id,
            app_id,
            arch,
            branch,
            commit,
            instance_path,
            ref_type,
        })
    }
}

/// Lists all Flatpak instances in the user's runtime directory.
///
/// Scans `$XDG_RUNTIME_DIR/.flatpak/` (or `$HOME/.local/share/flatpak/instances/`
/// as fallback) for instance directories and reads their metadata.
pub fn list_instances() -> XdpResult<Vec<FlatpakInstance>> {
    let mut instances = Vec::new();
    let base_dirs = get_instance_base_dirs();

    for base_dir in base_dirs {
        if let Ok(entries) = read_dir_entries(&base_dir) {
            for entry in entries {
                let instance_dir = format!("{}/{}", base_dir, entry);
                if is_directory(&instance_dir)
                    && let Ok(instance) = FlatpakInstance::new(&instance_dir)
                {
                    instances.push(instance);
                }
            }
        }
    }

    Ok(instances)
}

/// Finds a Flatpak instance by application ID.
pub fn find_by_app_id(app_id: &str) -> XdpResult<Option<FlatpakInstance>> {
    let instances = list_instances()?;
    Ok(instances
        .into_iter()
        .find(|inst| inst.app_id == app_id))
}

/// Returns the base directories to scan for Flatpak instances.
fn get_instance_base_dirs() -> Vec<String> {
    let mut dirs = Vec::new();

    // Primary: $XDG_RUNTIME_DIR/.flatpak/
    if let Some(runtime_dir) = env_var("XDG_RUNTIME_DIR") {
        dirs.push(format!("{}/.flatpak", runtime_dir));
    }

    // Fallback: $HOME/.local/share/flatpak/instances/
    if let Some(home) = env_var("HOME") {
        dirs.push(format!("{}/.local/share/flatpak/instances", home));
    }

    // System-wide: /var/lib/flatpak/instances/
    dirs.push(String::from("/var/lib/flatpak/instances"));

    dirs
}

/// Reads directory entries (filenames) from `path`.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn read_dir_entries(path: &str) -> Result<Vec<String>, PortalError> {
    use alloc::ffi::CString;

    let c_path = CString::new(path)
        .map_err(|_| PortalError::InvalidArgument(format!("Path contains nul byte: {}", path)))?;

    // SAFETY: c_path is a valid C string
    let dir = unsafe { libc::opendir(c_path.as_ptr()) };
    if dir.is_null() {
        let errno = unsafe { *libc::__errno_location() };
        if errno == libc::ENOENT {
            return Ok(Vec::new());
        }
        return Err(PortalError::Failed(format!(
            "Failed to open directory {}: errno {}",
            path, errno
        )));
    }

    let mut entries = Vec::new();
    loop {
        // SAFETY: dir is a valid DIR* from opendir
        let entry = unsafe { libc::readdir(dir) };
        if entry.is_null() {
            break;
        }
        // SAFETY: entry is a valid dirent*
        let name = unsafe { (*entry).d_name.as_ptr() };
        // SAFETY: name is a valid C string
        let c_str = unsafe { core::ffi::CStr::from_ptr(name) };
        if let Ok(name_str) = c_str.to_str()
            && name_str != "."
            && name_str != ".."
        {
            entries.push(name_str.to_string());
        }
    }

    // SAFETY: dir is a valid DIR*
    unsafe { libc::closedir(dir) };

    Ok(entries)
}

/// Reads directory entries on non-Unix or wasm targets (returns empty).
#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
fn read_dir_entries(_path: &str) -> Result<Vec<String>, PortalError> {
    Ok(Vec::new())
}

/// Checks if `path` is a directory.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn is_directory(path: &str) -> bool {
    use alloc::ffi::CString;

    let Ok(c_path) = CString::new(path) else {
        return false;
    };
    let mut stat_buf: libc::stat = unsafe { core::mem::zeroed() };
    // SAFETY: c_path is a valid C string and stat_buf is writable
    let result = unsafe { libc::stat(c_path.as_ptr(), &mut stat_buf) };
    if result != 0 {
        return false;
    }
    (stat_buf.st_mode & libc::S_IFMT) == libc::S_IFDIR
}

/// Checks if `path` is a directory on non-Unix or wasm targets (returns false).
#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
fn is_directory(_path: &str) -> bool {
    false
}

/// Reads a file into a byte vector.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn read_file_bytes(path: &str) -> Result<Vec<u8>, PortalError> {
    use alloc::ffi::CString;

    let c_path = CString::new(path)
        .map_err(|_| PortalError::InvalidArgument(format!("Path contains nul byte: {}", path)))?;

    // SAFETY: c_path is a valid C string
    let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        let errno = unsafe { *libc::__errno_location() };
        return Err(PortalError::Failed(format!(
            "Failed to open {}: errno {}",
            path, errno
        )));
    }

    let mut bytes = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        // SAFETY: fd is valid, buffer is writable
        let count = unsafe { libc::read(fd, buffer.as_mut_ptr().cast(), buffer.len()) };
        if count < 0 {
            let errno = unsafe { *libc::__errno_location() };
            if errno == libc::EINTR {
                continue;
            }
            unsafe { libc::close(fd) };
            return Err(PortalError::Failed(format!(
                "Failed to read {}: errno {}",
                path, errno
            )));
        }
        if count == 0 {
            break;
        }
        let count = count as usize;
        bytes.extend_from_slice(&buffer[..count]);
    }

    // SAFETY: fd is valid
    unsafe { libc::close(fd) };
    Ok(bytes)
}

/// Reads a file on non-Unix or wasm targets (returns error).
#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
fn read_file_bytes(_path: &str) -> Result<Vec<u8>, PortalError> {
    Err(PortalError::Failed(String::from(
        "File reading not supported on this platform",
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatpak_instance_fields() {
        let instance = FlatpakInstance {
            id: "inst123".to_string(),
            app_id: "org.example.App".to_string(),
            arch: "x86_64".to_string(),
            branch: "stable".to_string(),
            commit: "abc123".to_string(),
            instance_path: "/path/to/instance".to_string(),
            ref_type: "app".to_string(),
        };
        assert_eq!(instance.id, "inst123");
        assert_eq!(instance.app_id, "org.example.App");
        assert_eq!(instance.ref_type, "app");
    }

    #[test]
    fn test_get_instance_base_dirs() {
        let dirs = get_instance_base_dirs();
        assert!(!dirs.is_empty());
        assert!(dirs.iter().any(|d| d.contains(".flatpak")));
    }
}
