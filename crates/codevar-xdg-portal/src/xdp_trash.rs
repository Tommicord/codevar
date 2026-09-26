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

//! Trash portal implementation
//!
//! Ported from `desktop-portal/trash.c`. Provides the
//! `org.freedesktop.portal.Trash` interface with `TrashFile`.
//!
//! Unlike other portals, this does not use an implementation backend.
//! It directly implements the freedesktop.org trash specification
//! (https://specifications.freedesktop.org/trash-spec/latest/).

use codevar_base::basic_xml::XmlBuilder;

use crate::xdp_context::{MethodInvocation, PortalContext, PortalFn};
use crate::xdp_error::{PortalError, XdpResult};
use crate::xdp_utils::env_var;
use alloc::ffi::CString;
use alloc::format;
use alloc::string::{String, ToString};
use codevar_base::basic_pathbuf::PathBuf;

const TRASH_INTERFACE: &str = "org.freedesktop.portal.Trash";
const TRASH_VERSION: u32 = 1;

#[cfg(all(unix, not(target_arch = "wasm32")))]
mod unix_trash {
    use super::*;
    use alloc::vec;

    const TRASH_DIR_NAME: &str = "Trash";
    const TRASH_INFO_DIR: &str = "info";
    const TRASH_FILES_DIR: &str = "files";
    const MODE_0700: libc::mode_t = 0o700;

    fn make_dir(path: &str) -> Result<(), PortalError> {
        let c_path = CString::new(path.as_bytes())
            .map_err(|_| PortalError::InvalidArgument("path contains nul byte".to_string()))?;
        let result = unsafe { libc::mkdir(c_path.as_ptr(), MODE_0700) };
        if result == 0 {
            Ok(())
        } else {
            let errno = unsafe { *libc::__errno_location() };
            if errno == libc::EEXIST {
                Ok(())
            } else {
                Err(PortalError::Failed(format!("mkdir failed: {}", errno)))
            }
        }
    }

    pub(crate) fn get_trash_dirs() -> Result<(String, String), PortalError> {
        let data_dir = match env_var("XDG_DATA_HOME") {
            Some(dir) => dir,
            None => {
                let home = env_var("HOME").ok_or_else(|| PortalError::Failed("HOME not set".to_string()))?;
                format!("{}/.local/share", home)
            }
        };
        let trash_dir = format!("{}/{}", data_dir, TRASH_DIR_NAME);
        make_dir(&trash_dir)?;
        let info_dir = format!("{}/{}", trash_dir, TRASH_INFO_DIR);
        make_dir(&info_dir)?;
        let files_dir = format!("{}/{}", trash_dir, TRASH_FILES_DIR);
        make_dir(&files_dir)?;
        Ok((info_dir, files_dir))
    }

    fn path_exists(path: &str) -> bool {
        let c_path = match CString::new(path) {
            Ok(p) => p,
            Err(_) => return false,
        };
        unsafe { libc::access(c_path.as_ptr(), libc::F_OK) == 0 }
    }

    pub(crate) fn get_unique_name(base_name: &str, files_dir: &str) -> String {
        let mut name = base_name.to_string();
        let mut counter = 0;
        while path_exists(&format!("{}/{}", files_dir, name)) {
            counter += 1;
            name = format!("{}.{}", base_name, counter);
        }
        name
    }

    pub(crate) fn write_trash_info(
        info_dir: &str,
        info_name: &str,
        original_path: &str,
        deletion_time: &str,
    ) -> Result<(), PortalError> {
        let info_path = format!("{}/{}", info_dir, info_name);
        let content = format!(
            "[Trash Info]\nPath={}\nDeletionDate={}\n",
            original_path, deletion_time
        );
        // Write using libc
        let c_path = CString::new(info_path.as_bytes())
            .map_err(|_| PortalError::InvalidArgument("path contains nul byte".to_string()))?;
        let fd = unsafe {
            libc::open(
                c_path.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC | libc::O_CLOEXEC,
                0o644,
            )
        };
        if fd < 0 {
            return Err(PortalError::Failed("failed to open trash info file".to_string()));
        }
        let bytes = content.as_bytes();
        let mut written = 0;
        while written < bytes.len() {
            let n = unsafe {
                libc::write(
                    fd,
                    bytes[written..].as_ptr() as *const libc::c_void,
                    bytes.len() - written,
                )
            };
            if n < 0 {
                let errno = unsafe { *libc::__errno_location() };
                unsafe { libc::close(fd) };
                return Err(PortalError::Failed(format!("write failed: {}", errno)));
            }
            written += n as usize;
        }
        unsafe { libc::close(fd) };
        Ok(())
    }

    pub(crate) fn get_timestamp() -> String {
        let mut ts: libc::time_t = 0;
        unsafe { libc::time(&mut ts) };
        let mut tm: libc::tm = unsafe { core::mem::zeroed() };
        unsafe { libc::localtime_r(&ts, &mut tm) };
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec
        )
    }

    pub(crate) fn get_original_path_from_fd(fd: u32) -> Result<String, PortalError> {
        let proc_path = format!("/proc/self/fd/{}", fd);
        let c_path = CString::new(proc_path.as_bytes())
            .map_err(|_| PortalError::InvalidArgument("path contains nul byte".to_string()))?;
        let mut buf = vec![0u8; 4096];
        let n = unsafe { libc::readlink(c_path.as_ptr(), buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
        if n < 0 {
            return Err(PortalError::Failed(
                "Failed to resolve file descriptor".to_string(),
            ));
        }
        buf.truncate(n as usize);
        String::from_utf8(buf).map_err(|_| PortalError::Failed("Invalid UTF-8 in path".to_string()))
    }
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
fn handle_trash_file<T: codevar_dbus::DbusTransport + 'static>(
    ctx: &mut PortalContext<T>,
    inv: &MethodInvocation,
) -> XdpResult<()> {
    let mut reader = inv.body_reader();
    let fd = reader.read_fd()?;
    let _options = crate::xdp_utils::decode_options(&mut reader)?;
    let _app_info = crate::xdp_app_info::AppInfo::host(&inv.sender);

    let original_path = unix_trash::get_original_path_from_fd(fd)?;
    let (info_dir, files_dir) = unix_trash::get_trash_dirs()?;
    let base_name = PathBuf::from_string(original_path.clone())
        .file_name()
        .map(ToString::to_string)
        .ok_or_else(|| PortalError::InvalidArgument("Invalid file path".to_string()))?;
    let unique_name = unix_trash::get_unique_name(&base_name, &files_dir);
    let info_name = format!("{}.trashinfo", unique_name);
    let dest_path = format!("{}/{}", files_dir, unique_name);
    let c_src = CString::new(original_path.as_bytes())
        .map_err(|_| PortalError::InvalidArgument("path contains nul byte".to_string()))?;
    let c_dest = CString::new(dest_path.as_bytes())
        .map_err(|_| PortalError::InvalidArgument("path contains nul byte".to_string()))?;
    let result = unsafe { libc::rename(c_src.as_ptr(), c_dest.as_ptr()) };
    if result != 0 {
        let errno = unsafe { *libc::__errno_location() };
        return Err(PortalError::Failed(format!("rename failed: {}", errno)));
    }
    let deletion_time = unix_trash::get_timestamp();
    unix_trash::write_trash_info(&info_dir, &info_name, &original_path, &deletion_time)?;
    let result = 0u32;
    ctx.reply(inv, |bw| bw.write_u32(result))
}

#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
fn handle_trash_file<T: codevar_dbus::DbusTransport + 'static>(
    _ctx: &mut PortalContext<T>,
    _inv: &MethodInvocation,
) -> XdpResult<()> {
    Err(PortalError::NotSupported(String::from(
        "Trash portal not implemented for this platform",
    )))
}

pub fn register<T: codevar_dbus::DbusTransport + 'static>(ctx: &mut PortalContext<T>) -> XdpResult<()> {
    let methods: &[(&str, PortalFn<T>)] = &[("TrashFile", handle_trash_file)];

    let iface_xml = XmlBuilder::new("interface")
        .attr("name", "org.freedesktop.portal.Trash")
        .child("method")
        .attr("name", "TrashFile")
        .child("annotation")
        .attr("name", "org.gtk.GDBus.C.UnixFD")
        .attr("value", "true")
        .end()
        .child("arg")
        .attr("type", "h")
        .attr("name", "fd")
        .attr("direction", "in")
        .end()
        .child("arg")
        .attr("type", "u")
        .attr("name", "result")
        .attr("direction", "out")
        .end()
        .end()
        .child("property")
        .attr("name", "version")
        .attr("type", "u")
        .attr("access", "read")
        .end()
        .build();

    let iface = crate::xdp_context::PortalInterface {
        name: TRASH_INTERFACE,
        version: TRASH_VERSION,
        introspect_xml: iface_xml,
        methods,
    };

    ctx.register_interface(iface);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::xdp_utils::{generate_token, is_valid_token};

    #[test]
    fn generates_valid_token() {
        let token = generate_token().unwrap();
        assert!(is_valid_token(&token));
    }
}
