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

//! Application identity for portal callers
//!
//! An [`AppInfo`] describes the process behind a portal call: which
//! sandbox engine (if any) hosts it, its application id, and the
//! policies derived from that (network access, `O_PATH` support,
//! USB queries, dynamic launcher rewriting).
//!
//! Notes:
//! * Linyaps app infos are not detected (out of scope) and the
//!   systemd unit lookup for host apps is skipped, so host apps get
//!   the empty id unless registered explicitly.
//! * Pidfds, pid namespaces and `GDesktopAppInfo` lookups are not
//!   ported; the `REQUIRE_GAPPINFO` flag is recorded but does not
//!   gate construction.
//! * Snap detection reads `SNAP_NAME` from `/proc/<pid>/environ`
//!   instead of consulting the cgroup and spawning
//!   `snap routine portal-info`; `SNAP_DESKTOP_FILE` and
//!   `SNAP_HAS_NETWORK_STATUS` are read from the environment when
//!   present.
//! * Flatpak `bwrapinfo.json` pidfd resolution is skipped, so no
//!   JSON parser is needed.
//! * The `XDG_DESKTOP_PORTAL_TEST_*` environment hooks are not
//!   ported.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::xdp_error::PortalError;
use crate::xdp_utils::{
    KeyFile, documents_mountpoint, env_var, get_alternate_document_path, is_valid_app_id, maybe_quote,
    shell_parse_argv, shell_quote,
};

/// Engine recorded for flatpak applications, like `FLATPAK_ENGINE_ID`.
const FLATPAK_ENGINE_ID: &str = "org.flatpak";

/// Engine recorded for snap applications, like `SNAP_ENGINE_ID`.
const SNAP_ENGINE_ID: &str = "io.snapcraft";

/// Desktop entry group of a `.desktop` file, like
/// `G_KEY_FILE_DESKTOP_GROUP`.
const DESKTOP_GROUP: &str = "Desktop Entry";

/// Desktop entry key holding the command line, like
/// `G_KEY_FILE_DESKTOP_KEY_EXEC`.
const DESKTOP_KEY_EXEC: &str = "Exec";

/// Capability flags of an [`AppInfo`], mirroring `XdpAppInfoFlags`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AppInfoFlags(u8);

impl AppInfoFlags {
    /// The application has network access inside its sandbox.
    pub const HAS_NETWORK: Self = Self(1 << 0);
    /// The application may pass `O_PATH` file descriptors.
    pub const SUPPORTS_OPATH: Self = Self(1 << 1);
    /// A desktop entry must exist for the application id; recorded
    /// but not enforced without `GDesktopAppInfo`.
    pub const REQUIRE_GAPPINFO: Self = Self(1 << 2);

    /// Returns whether every bit of `other` is present in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl core::ops::BitOr for AppInfoFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// The sandbox engine hosting an application, mirroring the C
/// subclasses of `XdpAppInfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppInfoKind {
    /// No sandbox; the application runs directly on the host.
    Host,
    /// A flatpak sandbox.
    Flatpak,
    /// A snap sandbox.
    Snap,
}

impl AppInfoKind {
    /// Returns the `GType` name the C implementation reports for the
    /// matching subclass.
    const fn type_name(self) -> &'static str {
        match self {
            Self::Host => "XdpAppInfoHost",
            Self::Flatpak => "XdpAppInfoFlatpak",
            Self::Snap => "XdpAppInfoSnap",
        }
    }
}

/// Whether a [`UsbQuery`] matches enumerable or hidden devices,
/// mirroring `XdpUsbQueryType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsbQueryType {
    /// Devices the application may see (`enumerable-devices`).
    Enumerable,
    /// Devices the application must not see (`hidden-devices`).
    Hidden,
}

/// One rule of a USB device query, mirroring `XdpUsbRule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsbRule {
    /// `all` — every device.
    All,
    /// `cls:<class>:<subclass>` — device class match; `subclass` is
    /// `None` for the `*` wildcard.
    Class {
        /// USB device class in hex.
        class: u16,
        /// USB subclass in hex, or `None` when wildcarded.
        subclass: Option<u16>,
    },
    /// `dev:<product>` — product id match.
    Device {
        /// USB product id in hex.
        product: u16,
    },
    /// `vnd:<vendor>` — vendor id match; the C code stores this in
    /// the same union slot as [`UsbRule::Device`].
    Vendor {
        /// USB vendor id in hex.
        vendor: u16,
    },
}

/// A parsed USB device query of `+`-joined rules, mirroring
/// `XdpUsbQuery`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsbQuery {
    /// Whether the query lists enumerable or hidden devices.
    pub query_type: UsbQueryType,
    /// The rules that make up the query; never empty.
    pub rules: Vec<UsbRule>,
}

/// Parses `value` as a non-zero `u16` of exactly `expected_length`
/// hexadecimal digits, like `xdp_validate_hex_uint16`.
#[must_use]
pub fn validate_hex_uint16(value: &str, expected_length: usize) -> Option<u16> {
    if value.len() != expected_length || expected_length == 0 || !value.is_ascii() {
        return None;
    }
    if !value.bytes().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let parsed = u16::from_str_radix(value, 16).ok()?;
    if parsed == 0 {
        return None;
    }
    Some(parsed)
}

/// Parses one `all`/`cls`/`dev`/`vnd` rule string, like
/// `xdp_usb_rule_from_string`.
#[must_use]
pub fn usb_rule_from_string(string: &str) -> Option<UsbRule> {
    let parts: Vec<&str> = string.split(':').collect();
    if parts.len() > 3 {
        return None;
    }
    match parts[0] {
        "all" if parts.len() == 1 => Some(UsbRule::All),
        "cls" if parts.len() == 3 => {
            let class = validate_hex_uint16(parts[1], 2)?;
            let subclass = if parts[2] == "*" {
                None
            } else {
                Some(validate_hex_uint16(parts[2], 2)?)
            };
            Some(UsbRule::Class { class, subclass })
        }
        "dev" if parts.len() == 2 => Some(UsbRule::Device {
            product: validate_hex_uint16(parts[1], 4)?,
        }),
        "vnd" if parts.len() == 2 => Some(UsbRule::Vendor {
            vendor: validate_hex_uint16(parts[1], 4)?,
        }),
        _ => None,
    }
}

/// Parses a `+`-joined query string, like
/// `xdp_usb_query_from_string`; `None` when any rule is invalid or
/// no rule is present.
#[must_use]
pub fn usb_query_from_string(query_type: UsbQueryType, string: &str) -> Option<UsbQuery> {
    let mut rules = Vec::new();
    for rule in string.split('+') {
        rules.push(usb_rule_from_string(rule)?);
    }
    if rules.is_empty() {
        return None;
    }
    Some(UsbQuery { query_type, rules })
}

/// The resolved path of a file descriptor passed to a portal,
/// produced by [`AppInfo::path_for_fd`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FdPath {
    /// Absolute path of the file, remapped from the sandbox view to
    /// the host where needed.
    pub path: String,
    /// Whether the application may write to the file.
    pub writable: bool,
}

/// Identity of the process calling a portal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppInfo {
    kind: AppInfoKind,
    sender: String,
    id: String,
    instance: Option<String>,
    engine: Option<String>,
    flags: AppInfoFlags,
    desktop_file: Option<String>,
    flatpak_info: Option<KeyFile>,
    usb_queries: Option<Vec<UsbQuery>>,
}

impl AppInfo {
    /// Detects the kind of application behind `pid`, trying flatpak
    /// and snap before falling back to a host app, like
    /// `xdp_app_info_new`.
    ///
    /// # Errors
    ///
    /// Returns the underlying failure when `/proc` cannot be read or
    /// the `.flatpak-info` file is unreadable; "not this sandbox
    /// kind" outcomes fall through to the next candidate instead.
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    pub fn new(sender: &str, pid: u32) -> Result<Self, PortalError> {
        if let Some(app_info) = Self::try_flatpak(sender, pid)? {
            return Ok(app_info);
        }
        if let Some(app_info) = Self::try_snap(sender, pid) {
            return Ok(app_info);
        }
        Ok(Self::host(sender))
    }

    /// Builds a host app info with the empty id, like
    /// `xdp_app_info_host_new` without the systemd unit lookup.
    #[must_use]
    pub fn host(sender: &str) -> Self {
        Self {
            kind: AppInfoKind::Host,
            sender: String::from(sender),
            id: String::new(),
            instance: None,
            engine: None,
            flags: AppInfoFlags::HAS_NETWORK | AppInfoFlags::SUPPORTS_OPATH,
            desktop_file: None,
            flatpak_info: None,
            usb_queries: Some(Vec::from([UsbQuery {
                query_type: UsbQueryType::Enumerable,
                rules: Vec::from([UsbRule::All]),
            }])),
        }
    }

    /// Builds a host app info for a registered application id, like
    /// `xdp_app_info_host_new_registered`.
    #[must_use]
    pub fn host_registered(sender: &str, app_id: &str) -> Self {
        let mut app_info = Self::host(sender);
        app_info.id = String::from(app_id);
        app_info.flags = app_info.flags | AppInfoFlags::REQUIRE_GAPPINFO;
        app_info
    }

    /// Builds a flatpak app info from the contents of a
    /// `/proc/<pid>/root/.flatpak-info` file, the parsing step of
    /// [`AppInfo::new`].
    ///
    /// # Errors
    ///
    /// Returns [`PortalError::Failed`] when the file cannot be
    /// parsed, [`PortalError::NotFound`] when a required key is
    /// missing, and [`PortalError::InvalidArgument`] when the
    /// `name` key does not hold a valid application id.
    pub fn from_flatpak_info(sender: &str, info: &str) -> Result<Self, PortalError> {
        let key_file = KeyFile::parse(info).map_err(|error| {
            PortalError::Failed(format!("Can't load .flatpak-info file: {}", error.message()))
        })?;
        let group = if key_file.has_group("Runtime") {
            "Runtime"
        } else {
            "Application"
        };
        let id = key_file
            .get(group, "name")
            .ok_or_else(|| PortalError::NotFound(format!(".flatpak-info has no {group} name key")))?;
        if !is_valid_app_id(&id) {
            return Err(PortalError::InvalidArgument(format!(
                ".flatpak-info name '{id}' is not a valid app id"
            )));
        }
        let instance = key_file
            .get("Instance", "instance-id")
            .ok_or_else(|| {
                PortalError::NotFound(String::from(".flatpak-info has no Instance instance-id key"))
            })?;
        let has_network = key_file
            .list("Context", "shared")
            .is_some_and(|shared| shared.iter().any(|entry| entry == "network"));

        let mut flags = AppInfoFlags::SUPPORTS_OPATH;
        if has_network {
            flags = flags | AppInfoFlags::HAS_NETWORK;
        }
        let usb_queries = usb_queries_from_info(&key_file, &id);
        log::debug!("Found {} USB queries for app {id}", usb_queries.len());

        Ok(Self {
            kind: AppInfoKind::Flatpak,
            sender: String::from(sender),
            id,
            instance: Some(instance),
            engine: Some(String::from(FLATPAK_ENGINE_ID)),
            flags,
            desktop_file: None,
            flatpak_info: Some(key_file),
            usb_queries: Some(usb_queries),
        })
    }

    /// Builds a snap app info when `environ` (the contents of
    /// `/proc/<pid>/environ`) carries a non-empty `SNAP_NAME`,
    /// returning `None` otherwise. This replaces the cgroup check
    /// and `snap routine portal-info` call of the C implementation.
    #[must_use]
    pub fn from_snap_environ(sender: &str, environ: &[u8]) -> Option<Self> {
        let name = environ_value(environ, b"SNAP_NAME=")?;
        if name.is_empty() {
            return None;
        }
        let name = core::str::from_utf8(name).ok()?;
        let id = format!("snap.{name}");
        let desktop_file = environ_value(environ, b"SNAP_DESKTOP_FILE=")
            .and_then(|value| core::str::from_utf8(value).ok())
            .filter(|value| !value.is_empty())
            .map(String::from);
        let has_network = environ_value(environ, b"SNAP_HAS_NETWORK_STATUS=")
            .and_then(|value| core::str::from_utf8(value).ok())
            == Some("true");

        let mut flags = AppInfoFlags::default();
        if has_network {
            flags = flags | AppInfoFlags::HAS_NETWORK;
        }
        Some(Self {
            kind: AppInfoKind::Snap,
            sender: String::from(sender),
            id,
            instance: None,
            engine: Some(String::from(SNAP_ENGINE_ID)),
            flags,
            desktop_file,
            flatpak_info: None,
            usb_queries: None,
        })
    }

    /// Returns the sandbox engine of the application.
    #[must_use]
    pub const fn kind(&self) -> AppInfoKind {
        self.kind
    }

    /// Returns the application id; the empty string for unidentified
    /// host applications.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the sandbox instance id (flatpak instance id).
    #[must_use]
    pub fn instance(&self) -> Option<&str> {
        self.instance.as_deref()
    }

    /// Returns the engine id (`org.flatpak`, `io.snapcraft`) or
    /// `None` for host applications.
    #[must_use]
    pub fn engine(&self) -> Option<&str> {
        self.engine.as_deref()
    }

    /// Returns the D-Bus name the application called from.
    #[must_use]
    pub fn sender(&self) -> &str {
        &self.sender
    }

    /// Returns the capability flags of the application.
    #[must_use]
    pub const fn flags(&self) -> AppInfoFlags {
        self.flags
    }

    /// Returns whether the application has network access, like
    /// `xdp_app_info_has_network`.
    #[must_use]
    pub fn has_network(&self) -> bool {
        self.flags.contains(AppInfoFlags::HAS_NETWORK)
    }

    /// Returns whether the application runs unconfined, like
    /// `xdp_app_info_is_host`.
    #[must_use]
    pub fn is_host(&self) -> bool {
        self.engine.is_none()
    }

    /// Returns the snap desktop file id when the environment
    /// provided `SNAP_DESKTOP_FILE`.
    #[must_use]
    pub fn desktop_file(&self) -> Option<&str> {
        self.desktop_file.as_deref()
    }

    /// Returns a human readable application name, like
    /// `xdp_app_info_get_app_display_name`.
    ///
    /// The C code prefers the `GDesktopAppInfo` display name, which
    /// is unavailable here, so the application id is used instead.
    #[must_use]
    pub fn app_display_name(&self) -> Option<&str> {
        if self.id.is_empty() { None } else { Some(&self.id) }
    }

    /// Returns a human readable engine name, like
    /// `xdp_app_info_get_engine_display_name`.
    #[must_use]
    pub fn engine_display_name(&self) -> &str {
        match self.engine.as_deref() {
            Some(engine) if !engine.is_empty() => engine,
            _ => self.kind.type_name(),
        }
    }

    /// Returns the USB queries of the application, like
    /// `xdp_app_info_get_usb_queries`; `None` when the app kind
    /// does not define any (snap) or the id is unset.
    #[must_use]
    pub fn usb_queries(&self) -> Option<&[UsbQuery]> {
        self.usb_queries.as_deref()
    }

    /// Returns whether `sub_app_id` may act as `id`'s sub
    /// application, like `xdp_app_info_is_valid_sub_app_id`.
    ///
    /// Host applications accept every id, snap applications accept
    /// none (the C class defines no override), and flatpak
    /// applications require `id` plus a dot and a valid flatpak
    /// name.
    #[must_use]
    pub fn is_valid_sub_app_id(&self, sub_app_id: &str) -> bool {
        match self.kind {
            AppInfoKind::Host => true,
            AppInfoKind::Snap => false,
            AppInfoKind::Flatpak => {
                if !sub_app_id.starts_with(&self.id) {
                    return false;
                }
                if sub_app_id.as_bytes().get(self.id.len()) != Some(&b'.') {
                    return false;
                }
                is_flatpak_name(sub_app_id)
            }
        }
    }

    /// Maps `path` from the sandbox view to the host path, like the
    /// `remap_path` virtual method (flatpak rewrites, identity
    /// otherwise).
    #[must_use]
    pub fn remap_path(&self, path: &str) -> String {
        match self.kind {
            AppInfoKind::Flatpak => {
                let info = self.flatpak_info.as_ref();
                flatpak_remap_path(info, &self.id, path)
            }
            _ => String::from(path),
        }
    }

    /// Rewrites the desktop entry in `key_file` for installation by
    /// the dynamic launcher portal, like
    /// `xdp_app_info_validate_dynamic_launcher`.
    ///
    /// # Errors
    ///
    /// Returns [`PortalError::InvalidArgument`] when the entry has
    /// no usable `Exec` line or uses `--file-forwarding`, and
    /// [`PortalError::NotAllowed`] when the application kind does
    /// not support dynamic launchers (snap).
    pub fn validate_dynamic_launcher(&self, key_file: &mut KeyFile) -> Result<(), PortalError> {
        match self.kind {
            AppInfoKind::Host => Ok(()),
            AppInfoKind::Flatpak => self.flatpak_validate_dynamic_launcher(key_file),
            AppInfoKind::Snap => Err(PortalError::NotAllowed(format!(
                "DynamicLauncher install not supported for: {}",
                self.id
            ))),
        }
    }

    /// Resolves the host path behind the file descriptor `fd`, like
    /// `xdp_app_info_get_path_for_fd`.
    ///
    /// `require_st_mode` restricts the file type (`0` accepts any);
    /// pass [`libc::S_IFREG`] or [`libc::S_IFDIR`] to require a
    /// regular file or directory.
    ///
    /// # Errors
    ///
    /// Returns [`PortalError::InvalidArgument`] for an invalid
    /// descriptor or an unusable open mode, [`PortalError::NotAllowed`]
    /// when the sandbox refuses access, and [`PortalError::Failed`]
    /// for the remaining `fstat`/`readlink`/identity failures of the
    /// C implementation.
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    pub fn path_for_fd(&self, fd: i32, require_st_mode: u32) -> Result<FdPath, PortalError> {
        if fd == -1 {
            return Err(PortalError::InvalidArgument(String::from(
                "Invalid file descriptor",
            )));
        }

        // SAFETY: `fd` is a caller supplied descriptor; a negative
        // result only means `errno` was set.
        let fd_flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if fd_flags < 0 {
            // SAFETY: the call failed, so `errno` is live.
            let errno = unsafe { *libc::__errno_location() };
            return Err(PortalError::Failed(format!(
                "Cannot get file descriptor flags (fcntl F_GETFL: errno {errno})"
            )));
        }

        let mut stat_buf: libc::stat = unsafe { core::mem::zeroed() };
        // SAFETY: `fd` is open and `stat_buf` is writable storage.
        if unsafe { libc::fstat(fd, &mut stat_buf) } < 0 {
            // SAFETY: the call failed, so `errno` is live.
            let errno = unsafe { *libc::__errno_location() };
            return Err(PortalError::Failed(format!(
                "Cannot get file information (fstat: errno {errno})"
            )));
        }

        let file_type = stat_buf.st_mode & libc::S_IFMT;
        if require_st_mode != 0 && file_type != require_st_mode {
            return Err(match require_st_mode {
                libc::S_IFDIR => PortalError::Failed(format!("File type 0o{file_type:o} is not a directory")),
                libc::S_IFREG => {
                    PortalError::Failed(format!("File type 0o{file_type:o} is not a regular file"))
                }
                _ => PortalError::Failed(format!(
                    "File type 0o{file_type:o} does not match expected 0o{require_st_mode:o}"
                )),
            });
        }

        let proc_path = format!("/proc/self/fd/{fd}");
        let path = verify_proc_self_fd(&proc_path)?;
        let path = if fd_flags & libc::O_PATH == libc::O_PATH {
            if fd_flags & libc::O_NOFOLLOW == libc::O_NOFOLLOW {
                return Err(PortalError::InvalidArgument(String::from(
                    "O_PATH fd was opened O_NOFOLLOW",
                )));
            }
            if !self.flags.contains(AppInfoFlags::SUPPORTS_OPATH) {
                return Err(PortalError::NotAllowed(format!(
                    "App \"{}\" of type {} does not support O_PATH fd passing",
                    self.id,
                    self.engine_display_name()
                )));
            }
            let mut read_access_mode = libc::R_OK;
            if file_type == libc::S_IFDIR {
                read_access_mode |= libc::X_OK;
            }
            if !access(&proc_path, read_access_mode) {
                return Err(PortalError::NotAllowed(format!(
                    "\"{path}\" not available for read access via \"{proc_path}\""
                )));
            }
            let writable = self.is_host() || access(&proc_path, libc::W_OK);
            (path, writable)
        } else {
            let accmode = fd_flags & libc::O_ACCMODE;
            if accmode != libc::O_RDONLY && accmode != libc::O_RDWR {
                return Err(PortalError::InvalidArgument(String::from(
                    "File descriptor is not open for reading",
                )));
            }
            let writable = self.is_host() || accmode == libc::O_RDWR;
            (path, writable)
        };
        let (path, writable) = path;

        if let Err(identity_error) = check_same_file(&path, &stat_buf) {
            let Some(alt_path) = get_alternate_document_path(&path, &self.id) else {
                return Err(identity_error);
            };
            check_same_file(&alt_path, &stat_buf)?;
        }

        Ok(FdPath { path, writable })
    }

    /// Tries to open the flatpak info of `pid`; `Ok(None)` means
    /// the process is not a flatpak.
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    fn try_flatpak(sender: &str, pid: u32) -> Result<Option<Self>, PortalError> {
        let Some(info) = open_flatpak_info(pid)? else {
            return Ok(None);
        };
        let info = String::from_utf8(info).map_err(|_| {
            PortalError::Failed(String::from("Can't load .flatpak-info file: not valid UTF-8"))
        })?;
        Self::from_flatpak_info(sender, &info).map(Some)
    }

    /// Tries to detect a snap from `pid`'s environment; `Ok(None)`
    /// means the process is not a snap.
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    fn try_snap(sender: &str, pid: u32) -> Option<Self> {
        let environ = read_proc_file(&format!("/proc/{pid}/environ")).ok()?;
        Self::from_snap_environ(sender, &environ)
    }

    /// Applies the flatpak `Exec` rewriting of
    /// `xdp_app_info_flatpak_validate_dynamic_launcher`.
    fn flatpak_validate_dynamic_launcher(&self, key_file: &mut KeyFile) -> Result<(), PortalError> {
        let exec = key_file
            .get(DESKTOP_GROUP, DESKTOP_KEY_EXEC)
            .ok_or_else(|| {
                PortalError::InvalidArgument(String::from(
                    "Desktop entry given to Install() has no Exec line",
                ))
            })?;
        let exec_strv = shell_parse_argv(&exec).map_err(|_| {
            PortalError::InvalidArgument(String::from(
                "Desktop entry given to Install() has invalid Exec line",
            ))
        })?;
        if exec_strv
            .iter()
            .any(|arg| arg == "--file-forwarding")
        {
            return Err(PortalError::InvalidArgument(String::from(
                "Desktop entry given to Install() must not use --file-forwarding",
            )));
        }

        let rewritten = rewrite_commandline(&self.id, &exec_strv, true);
        key_file.set(DESKTOP_GROUP, DESKTOP_KEY_EXEC, &join_args(&rewritten));

        if let Some(tryexec) = self.flatpak_tryexec_path() {
            key_file.set(DESKTOP_GROUP, "TryExec", &tryexec);
        }
        // Flatpak checks for this key.
        key_file.set(DESKTOP_GROUP, "X-Flatpak", &self.id);
        // Flatpak removes this one for security.
        key_file.remove(DESKTOP_GROUP, "X-GNOME-Bugzilla-ExtraInfoScript");
        Ok(())
    }

    /// Returns the exported wrapper script usable as `TryExec`,
    /// like `get_tryexec_path`; `None` when it cannot be derived
    /// or is not executable.
    fn flatpak_tryexec_path(&self) -> Option<String> {
        let info = self.flatpak_info.as_ref()?;
        let original_app_path = info.get("Instance", "original-app-path");
        let app_path = info.get("Instance", "app-path");
        let path = original_app_path.or(app_path)?;
        if path.is_empty() {
            return None;
        }
        let app_slash = format!("app/{}", self.id);
        let index = path.find(&app_slash)?;
        let prefix = &path[..index];
        let tryexec_path = format!("{prefix}exports/bin/{}", self.id);
        if !access(&tryexec_path, libc::X_OK) {
            log::debug!("Wrapper script unexpectedly not executable or nonexistent: {tryexec_path}");
            return None;
        }
        Some(tryexec_path)
    }
}

/// Returns whether `string` is a valid flatpak application name, the
/// rules of `flatpak_is_valid_name` in the C reference: at most 255
/// bytes, three or more dot separated elements that do not start
/// with a digit, with dashes only in the final element.
fn is_flatpak_name(string: &str) -> bool {
    let len = string.len();
    if len == 0 || len > 255 {
        return false;
    }
    let bytes = string.as_bytes();
    if bytes[0] == b'.' {
        return false;
    }
    let last_dot = bytes.iter().rposition(|&b| b == b'.');
    let mut dot_count = 0u32;
    let mut last_element = false;
    let mut index = 0;
    if !is_initial_name_character(bytes[0], last_element) {
        return false;
    }
    index += 1;
    while index < len {
        if bytes[index] == b'.' {
            last_element = Some(index) == last_dot;
            index += 1;
            if index == len {
                return false;
            }
            if !is_initial_name_character(bytes[index], last_element) {
                return false;
            }
            dot_count += 1;
        } else if !is_name_character(bytes[index], last_element) {
            return false;
        }
        index += 1;
    }
    dot_count >= 2
}

/// Returns whether `c` may start an app name element.
const fn is_initial_name_character(c: u8, allow_dash: bool) -> bool {
    c.is_ascii_alphabetic() || c == b'_' || (allow_dash && c == b'-')
}

/// Returns whether `c` may appear inside an app name element.
const fn is_name_character(c: u8, allow_dash: bool) -> bool {
    is_initial_name_character(c, allow_dash) || c.is_ascii_digit()
}

/// Rewrites `commandline` into a `flatpak run` invocation, like
/// `rewrite_commandline` in the C reference.
fn rewrite_commandline(app_id: &str, commandline: &[String], quote_escape: bool) -> Vec<String> {
    let mut args = Vec::from([String::from("flatpak"), String::from("run")]);
    if let Some(command) = commandline.first() {
        let quoted_command = maybe_quote(command, quote_escape);
        args.push(format!("--command={quoted_command}"));
        // Always quote the app ID if quote_escape is enabled to make
        // rewriting the file simpler in case the app is renamed.
        if quote_escape {
            args.push(shell_quote(app_id));
        } else {
            args.push(String::from(app_id));
        }
        for argument in &commandline[1..] {
            args.push(maybe_quote(argument, quote_escape));
        }
    } else if quote_escape {
        args.push(shell_quote(app_id));
    } else {
        args.push(String::from(app_id));
    }
    args
}

/// Joins rewritten command line arguments with spaces.
fn join_args(args: &[String]) -> String {
    let mut out = String::new();
    for (index, arg) in args.iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        out.push_str(arg);
    }
    out
}

/// Joins `base` and `rest` like `g_build_filename` for two parts.
fn join_path(base: &str, rest: &str) -> String {
    if base.is_empty() {
        return String::from(rest);
    }
    if base.ends_with('/') {
        format!("{base}{rest}")
    } else {
        format!("{base}/{rest}")
    }
}

/// Maps a sandbox path to its host location using the flatpak
/// metadata, like `xdp_app_info_flatpak_remap_path`.
fn flatpak_remap_path(info: Option<&KeyFile>, app_id: &str, path: &str) -> String {
    let mut path = path;
    // Drop the /newroot prefix added by bubblewrap for other files
    // to work; see https://github.com/projectatomic/bubblewrap/pull/172.
    if let Some(rest) = path.strip_prefix("/newroot")
        && rest.starts_with('/')
    {
        path = rest;
    }

    let app_path = info.and_then(|info| info.get("Instance", "app-path"));
    let runtime_path = info.and_then(|info| info.get("Instance", "runtime-path"));

    if let (Some(app_path), Some(rest)) = (app_path.as_deref(), path.strip_prefix("/app/")) {
        return join_path(app_path, rest);
    }
    if let (Some(runtime_path), Some(rest)) = (runtime_path.as_deref(), path.strip_prefix("/usr/")) {
        return join_path(runtime_path, rest);
    }
    if let Some(rest) = path.strip_prefix("/run/host/usr/") {
        return join_path("/usr", rest);
    }
    if let Some(rest) = path.strip_prefix("/run/host/etc/") {
        return join_path("/etc", rest);
    }
    if let Some(rest) = path.strip_prefix("/run/flatpak/app/") {
        let runtime_dir = env_var("XDG_RUNTIME_DIR").unwrap_or_default();
        return join_path(&join_path(&runtime_dir, "app"), rest);
    }
    if let Some(rest) = path.strip_prefix("/run/flatpak/doc/") {
        let runtime_dir = env_var("XDG_RUNTIME_DIR").unwrap_or_default();
        return join_path(&join_path(&runtime_dir, "doc"), rest);
    }
    if let Some(rest) = path.strip_prefix("/var/config/") {
        let base = join_path(
            &join_path(
                &join_path(&join_path(&env_var("HOME").unwrap_or_default(), ".var"), "app"),
                app_id,
            ),
            "config",
        );
        return join_path(&base, rest);
    }
    if let Some(rest) = path.strip_prefix("/var/data/") {
        let base = join_path(
            &join_path(
                &join_path(&join_path(&env_var("HOME").unwrap_or_default(), ".var"), "app"),
                app_id,
            ),
            "data",
        );
        return join_path(&base, rest);
    }
    String::from(path)
}

/// Collects the USB queries stored in a flatpak info file, like
/// `xdp_app_info_flaptak_get_usb_queries`; entries that do not
/// parse are skipped.
fn usb_queries_from_info(info: &KeyFile, app_id: &str) -> Vec<UsbQuery> {
    let enumerable = info
        .list("USB Devices", "enumerable-devices")
        .unwrap_or_default();
    let hidden = info
        .list("USB Devices", "hidden-devices")
        .unwrap_or_default();
    let mut queries = Vec::new();
    for entry in &enumerable {
        if let Some(query) = usb_query_from_string(UsbQueryType::Enumerable, entry) {
            queries.push(query);
        }
    }
    for entry in &hidden {
        if let Some(query) = usb_query_from_string(UsbQueryType::Hidden, entry) {
            queries.push(query);
        }
    }
    log::debug!(
        "Found {} enumerable and {} hidden for app {app_id}",
        enumerable.len(),
        hidden.len()
    );
    queries
}

/// Looks up `key=value` inside NUL separated `environ` bytes.
fn environ_value<'a>(environ: &'a [u8], key: &[u8]) -> Option<&'a [u8]> {
    environ
        .split(|&byte| byte == 0)
        .find_map(|entry| entry.strip_prefix(key))
}

/// Verifies the `/proc/self/fd/<n>` symlink of a descriptor and
/// remaps it, like `verify_proc_self_fd`.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn verify_proc_self_fd(proc_path: &str) -> Result<String, PortalError> {
    use alloc::ffi::CString;

    let c_path = CString::new(proc_path)
        .map_err(|_| PortalError::InvalidArgument(String::from("path contains a nul byte")))?;
    let mut buffer = [0u8; 4096];
    // SAFETY: `c_path` is a valid C string and `buffer` is writable
    // for its full length.
    let size = unsafe { libc::readlink(c_path.as_ptr(), buffer.as_mut_ptr().cast(), buffer.len()) };
    if size < 0 {
        // SAFETY: the call failed, so `errno` is live.
        let errno = unsafe { *libc::__errno_location() };
        return Err(PortalError::Failed(format!(
            "readlink {proc_path}: errno {errno}"
        )));
    }
    let Ok(size) = usize::try_from(size) else {
        return Err(PortalError::Failed(format!("readlink {proc_path}")));
    };
    let link = String::from_utf8(buffer[..size].to_vec())
        .map_err(|_| PortalError::Failed(format!("readlink {proc_path}: not valid UTF-8")))?;

    // All normal paths start with /, but some weird things don't,
    // such as socket:[27345] or anon_inode:[eventfd].
    if !link.starts_with('/') {
        return Err(PortalError::Failed(format!(
            "Not a regular file or directory: {link}"
        )));
    }

    // Files marked as deleted carry a " (deleted)" suffix; the
    // document portal FUSE mount is allowed and rewritten because
    // the identity check below still pins the real file.
    if let Some(stripped) = link.strip_suffix(" (deleted)") {
        let allowed = documents_mountpoint()
            .as_deref()
            .is_some_and(|mountpoint| link.starts_with(mountpoint));
        if allowed {
            return Ok(String::from(stripped));
        }
        return Err(PortalError::Failed(format!("Cannot share deleted file: {link}")));
    }
    Ok(link)
}

/// Returns whether `path` is the same file as `expected`.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn check_same_file(path: &str, expected: &libc::stat) -> Result<(), PortalError> {
    use alloc::ffi::CString;

    let c_path = CString::new(path)
        .map_err(|_| PortalError::InvalidArgument(String::from("path contains a nul byte")))?;
    let mut real: libc::stat = unsafe { core::mem::zeroed() };
    // SAFETY: `c_path` is a valid path and `real` is writable.
    if unsafe { libc::stat(c_path.as_ptr(), &mut real) } < 0 {
        // SAFETY: the call failed, so `errno` is live.
        let errno = unsafe { *libc::__errno_location() };
        return Err(PortalError::Failed(format!("stat {path}: errno {errno}")));
    }
    if expected.st_dev != real.st_dev || expected.st_ino != real.st_ino {
        return Err(PortalError::Failed(format!(
            "\"{path}\" identity ({}, {}) does not match expected ({}, {})",
            expected.st_dev, expected.st_ino, real.st_dev, real.st_ino
        )));
    }
    Ok(())
}

/// Calls `access` on `path`.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn access(path: &str, mode: libc::c_int) -> bool {
    use alloc::ffi::CString;

    let Ok(c_path) = CString::new(path) else {
        return false;
    };
    // SAFETY: `c_path` is a valid C string.
    let result = unsafe { libc::access(c_path.as_ptr(), mode) };
    result == 0
}

/// Opens `/proc/<pid>/root/.flatpak-info`; `Ok(None)` when the file
/// or the process root cannot be read in a way that means "not a
/// flatpak".
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn open_flatpak_info(pid: u32) -> Result<Option<Vec<u8>>, PortalError> {
    use alloc::ffi::CString;

    let root_path = format!("/proc/{pid}/root");
    let c_root = CString::new(root_path.clone())
        .map_err(|_| PortalError::InvalidArgument(String::from("path contains a nul byte")))?;
    // SAFETY: `c_root` is a valid C string; the descriptor is
    // checked against negative values before use.
    let root_fd = unsafe {
        libc::open(
            c_root.as_ptr(),
            libc::O_RDONLY | libc::O_NONBLOCK | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOCTTY,
        )
    };
    if root_fd < 0 {
        // SAFETY: the call failed, so `errno` is live.
        let errno = unsafe { *libc::__errno_location() };
        if errno == libc::EACCES && is_fuse_root(&root_path) {
            // Access to the root dir isn't allowed and the root is
            // on a FUSE filesystem (a toolbox container, never a
            // flatpak); detect other kinds instead.
            return Ok(None);
        }
        return Err(PortalError::Failed(format!(
            "Unable to open {root_path}: errno {errno}"
        )));
    }

    let c_info = CString::new(".flatpak-info")
        .map_err(|_| PortalError::InvalidArgument(String::from("path contains a nul byte")))?;
    // SAFETY: `root_fd` is an open directory and `c_info` is a
    // valid relative name.
    let info_fd = unsafe {
        libc::openat(
            root_fd,
            c_info.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOCTTY,
        )
    };
    // SAFETY: the descriptor is open and will be closed.
    unsafe { libc::close(root_fd) };
    if info_fd < 0 {
        // SAFETY: the call failed, so `errno` is live.
        let errno = unsafe { *libc::__errno_location() };
        if errno == libc::ENOENT {
            // No file => on the host.
            return Ok(None);
        }
        return Err(PortalError::Failed(format!(
            "Unable to open application info file: errno {errno}"
        )));
    }

    let bytes = read_fd_all(info_fd, ".flatpak-info")?;
    // SAFETY: the descriptor is open and will be closed.
    unsafe { libc::close(info_fd) };
    Ok(Some(bytes))
}

/// Returns whether `path` is a FUSE mount point (Linux only; other
/// platforms never match, so `EACCES` becomes a hard failure).
#[cfg(all(unix, not(target_arch = "wasm32"), target_os = "linux"))]
fn is_fuse_root(path: &str) -> bool {
    use alloc::ffi::CString;

    const FUSE_SUPER_MAGIC: u64 = 0x6573_5546;
    let Ok(c_path) = CString::new(path) else {
        return false;
    };
    let mut buf: libc::statfs = unsafe { core::mem::zeroed() };
    // SAFETY: `c_path` is a valid path and `buf` is writable.
    let result = unsafe { libc::statfs(c_path.as_ptr(), &mut buf) };
    result == 0 && (buf.f_type as u64) == FUSE_SUPER_MAGIC
}

/// Non-Linux Unix variant of [`is_fuse_root`].
#[cfg(all(unix, not(target_arch = "wasm32"), not(target_os = "linux")))]
fn is_fuse_root(_path: &str) -> bool {
    false
}

/// Reads a whole file into bytes.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn read_proc_file(path: &str) -> Result<Vec<u8>, PortalError> {
    use alloc::ffi::CString;

    let c_path = CString::new(path)
        .map_err(|_| PortalError::InvalidArgument(String::from("path contains a nul byte")))?;
    // SAFETY: `c_path` is a valid path; the descriptor is checked
    // before use.
    let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        // SAFETY: the call failed, so `errno` is live.
        let errno = unsafe { *libc::__errno_location() };
        return Err(PortalError::Failed(format!("cannot open {path}: errno {errno}")));
    }
    let bytes = read_fd_all(fd, path);
    // SAFETY: the descriptor is open and will be closed.
    unsafe { libc::close(fd) };
    bytes
}

/// Reads `fd` to end of file.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn read_fd_all(fd: i32, label: &str) -> Result<Vec<u8>, PortalError> {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        // SAFETY: `fd` is open and `buffer` is writable.
        let count = unsafe { libc::read(fd, buffer.as_mut_ptr().cast(), buffer.len()) };
        if count < 0 {
            // SAFETY: the call failed, so `errno` is live.
            let errno = unsafe { *libc::__errno_location() };
            if errno == libc::EINTR {
                continue;
            }
            return Err(PortalError::Failed(format!("cannot read {label}: errno {errno}")));
        }
        if count == 0 {
            return Ok(bytes);
        }
        let Ok(count) = usize::try_from(count) else {
            return Err(PortalError::Failed(format!("cannot read {label}")));
        };
        bytes.extend_from_slice(&buffer[..count]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // unwrap()/expect() in tests is permitted by AGENTS.md; every
    // call below asserts a value this test itself constructed.

    fn flatpak_info() -> &'static str {
        "[Application]\nname=org.example.App\n[Instance]\ninstance-id=inst1\n\
         [Context]\nshared=network;screens;\n"
    }

    #[test]
    fn detects_flatpak_metadata() {
        let app = AppInfo::from_flatpak_info("org.test.Sender", flatpak_info()).unwrap();
        assert_eq!(app.kind(), AppInfoKind::Flatpak);
        assert_eq!(app.id(), "org.example.App");
        assert_eq!(app.instance(), Some("inst1"));
        assert_eq!(app.engine(), Some("org.flatpak"));
        assert_eq!(app.sender(), "org.test.Sender");
        assert!(!app.is_host());
        assert!(app.has_network());
        assert!(app.flags().contains(AppInfoFlags::SUPPORTS_OPATH));
        assert!(
            !app.flags()
                .contains(AppInfoFlags::REQUIRE_GAPPINFO)
        );
        assert_eq!(app.app_display_name(), Some("org.example.App"));
        assert_eq!(app.engine_display_name(), "org.flatpak");

        let no_network = AppInfo::from_flatpak_info(
            "s",
            "[Application]\nname=org.example.App\n[Instance]\ninstance-id=i\n",
        )
        .unwrap();
        assert!(!no_network.has_network());

        let runtime = AppInfo::from_flatpak_info(
            "s",
            "[Runtime]\nname=org.example.Runtime\n[Instance]\ninstance-id=i\n",
        )
        .unwrap();
        assert_eq!(runtime.id(), "org.example.Runtime");

        assert!(
            AppInfo::from_flatpak_info("s", "not a key file").is_err(),
            "malformed metadata must fail"
        );
        assert!(
            AppInfo::from_flatpak_info("s", "[Application]\n[Instance]\ninstance-id=i\n").is_err(),
            "missing name must fail"
        );
        assert!(
            AppInfo::from_flatpak_info("s", "[Application]\nname=nodots\n[Instance]\ninstance-id=i\n")
                .is_err(),
            "invalid app id must fail"
        );
        assert!(
            AppInfo::from_flatpak_info("s", "[Application]\nname=org.example.App\n").is_err(),
            "missing instance id must fail"
        );
    }

    #[test]
    fn rewrites_flatpak_paths() {
        let info =
            KeyFile::parse("[Instance]\napp-path=/flatpak/app\nruntime-path=/flatpak/runtime\n").unwrap();
        let app_id = "org.example.App";

        assert_eq!(
            flatpak_remap_path(Some(&info), app_id, "/app/bin/tool"),
            "/flatpak/app/bin/tool"
        );
        assert_eq!(
            flatpak_remap_path(Some(&info), app_id, "/usr/lib/lib.so"),
            "/flatpak/runtime/lib/lib.so"
        );
        assert_eq!(
            flatpak_remap_path(Some(&info), app_id, "/run/host/usr/bin/x"),
            "/usr/bin/x"
        );
        assert_eq!(
            flatpak_remap_path(Some(&info), app_id, "/run/host/etc/os-release"),
            "/etc/os-release"
        );
        assert_eq!(
            flatpak_remap_path(Some(&info), app_id, "/newroot/app/bin/tool"),
            "/flatpak/app/bin/tool"
        );
        assert_eq!(
            flatpak_remap_path(Some(&info), app_id, "/home/user/file"),
            "/home/user/file"
        );

        let host = AppInfo::host("s");
        assert_eq!(host.remap_path("/app/bin/tool"), "/app/bin/tool");
    }

    #[test]
    fn validates_flatpak_sub_app_ids() {
        let app = AppInfo::from_flatpak_info("s", flatpak_info()).unwrap();
        assert!(app.is_valid_sub_app_id("org.example.App.Sub"));
        assert!(app.is_valid_sub_app_id("org.example.App.sub-app"));
        assert!(app.is_valid_sub_app_id("org.example.App.SubApp.Sub"));
        assert!(!app.is_valid_sub_app_id("org.example.App"));
        assert!(!app.is_valid_sub_app_id("org.example.AppX.Sub"));
        assert!(!app.is_valid_sub_app_id("org.example.App.1bad"));
        assert!(!app.is_valid_sub_app_id("org.example.Other.Sub"));

        let host = AppInfo::host_registered("s", "org.example.App");
        assert!(host.is_valid_sub_app_id("anything"));

        let snap = AppInfo::from_snap_environ("s", b"SNAP_NAME=foo\0").unwrap();
        assert!(!snap.is_valid_sub_app_id("snap.foo.bar"));
    }

    #[test]
    fn builds_host_and_registered_app_infos() {
        let host = AppInfo::host("s");
        assert_eq!(host.kind(), AppInfoKind::Host);
        assert!(host.is_host());
        assert_eq!(host.id(), "");
        assert_eq!(host.app_display_name(), None);
        assert_eq!(host.engine_display_name(), "XdpAppInfoHost");
        assert!(host.has_network());
        assert!(
            host.flags()
                .contains(AppInfoFlags::SUPPORTS_OPATH)
        );
        assert!(
            !host
                .flags()
                .contains(AppInfoFlags::REQUIRE_GAPPINFO)
        );
        let queries = host.usb_queries().unwrap();
        assert_eq!(queries.len(), 1);
        assert_eq!(queries[0].query_type, UsbQueryType::Enumerable);
        assert_eq!(queries[0].rules, Vec::from([UsbRule::All]));

        let registered = AppInfo::host_registered("s", "org.example.App");
        assert_eq!(registered.id(), "org.example.App");
        assert_eq!(registered.app_display_name(), Some("org.example.App"));
        assert!(
            registered
                .flags()
                .contains(AppInfoFlags::REQUIRE_GAPPINFO)
        );
    }

    #[test]
    fn builds_snap_app_infos_from_environ() {
        let snap =
            AppInfo::from_snap_environ("s", b"SNAP_NAME=calibre\0SNAP_DESKTOP_FILE=x.desktop\0").unwrap();
        assert_eq!(snap.kind(), AppInfoKind::Snap);
        assert_eq!(snap.id(), "snap.calibre");
        assert_eq!(snap.engine(), Some("io.snapcraft"));
        assert_eq!(snap.engine_display_name(), "io.snapcraft");
        assert_eq!(snap.desktop_file(), Some("x.desktop"));
        assert!(!snap.has_network());
        assert!(
            !snap
                .flags()
                .contains(AppInfoFlags::SUPPORTS_OPATH)
        );
        assert_eq!(snap.usb_queries(), None);

        let networked =
            AppInfo::from_snap_environ("s", b"SNAP_NAME=calibre\0SNAP_HAS_NETWORK_STATUS=true\0").unwrap();
        assert!(networked.has_network());

        assert_eq!(AppInfo::from_snap_environ("s", b"PATH=/usr/bin\0"), None);
        assert_eq!(AppInfo::from_snap_environ("s", b"SNAP_NAME=\0"), None);
        assert_eq!(AppInfo::from_snap_environ("s", b""), None);
    }

    #[test]
    fn parses_usb_queries() {
        assert_eq!(validate_hex_uint16("03", 2), Some(3));
        assert_eq!(validate_hex_uint16("ffff", 4), Some(0xffff));
        assert_eq!(validate_hex_uint16("FFFF", 4), Some(0xffff));
        assert_eq!(validate_hex_uint16("0000", 4), None, "zero is invalid");
        assert_eq!(validate_hex_uint16("10000", 4), None, "wrong length");
        assert_eq!(validate_hex_uint16("12", 4), None, "wrong length");
        assert_eq!(validate_hex_uint16("gg", 2), None, "not hex");
        assert_eq!(validate_hex_uint16("", 2), None);

        assert_eq!(usb_rule_from_string("all"), Some(UsbRule::All));
        assert_eq!(usb_rule_from_string("all:x"), None);
        assert_eq!(
            usb_rule_from_string("cls:03:01"),
            Some(UsbRule::Class {
                class: 3,
                subclass: Some(1)
            })
        );
        assert_eq!(
            usb_rule_from_string("cls:03:*"),
            Some(UsbRule::Class {
                class: 3,
                subclass: None
            })
        );
        assert_eq!(usb_rule_from_string("cls:zz:01"), None);
        assert_eq!(usb_rule_from_string("cls:03"), None);
        assert_eq!(
            usb_rule_from_string("dev:1234"),
            Some(UsbRule::Device { product: 0x1234 })
        );
        assert_eq!(
            usb_rule_from_string("vnd:abcd"),
            Some(UsbRule::Vendor { vendor: 0xabcd })
        );
        assert_eq!(usb_rule_from_string("dev:12"), None, "needs four digits");
        assert_eq!(usb_rule_from_string("dev:1234:extra"), None, "too long");
        assert_eq!(usb_rule_from_string("bogus"), None);
        assert_eq!(usb_rule_from_string(""), None);

        let query = usb_query_from_string(UsbQueryType::Enumerable, "all+dev:0001").unwrap();
        assert_eq!(
            query.rules,
            Vec::from([UsbRule::All, UsbRule::Device { product: 1 }])
        );
        assert_eq!(usb_query_from_string(UsbQueryType::Hidden, "dev:0001+"), None);
        assert_eq!(usb_query_from_string(UsbQueryType::Hidden, ""), None);
    }

    #[test]
    fn builds_flatpak_usb_queries_from_metadata() {
        let info = KeyFile::parse(
            "[Application]\nname=org.example.App\n\
             [USB Devices]\nenumerable-devices=dev:1234+bogus;dev:0001;\nhidden-devices=cls:03:*;\n",
        )
        .unwrap();
        let queries = usb_queries_from_info(&info, "org.example.App");
        assert_eq!(queries.len(), 2, "queries with invalid rules are skipped");
        assert_eq!(queries[0].query_type, UsbQueryType::Enumerable);
        assert_eq!(queries[0].rules, Vec::from([UsbRule::Device { product: 1 }]));
        assert_eq!(queries[1].query_type, UsbQueryType::Hidden);
        assert_eq!(
            queries[1].rules,
            Vec::from([UsbRule::Class {
                class: 3,
                subclass: None
            }])
        );
    }

    #[test]
    fn validates_dynamic_launcher_entries() {
        let host = AppInfo::host("s");
        let mut entry = KeyFile::parse("[Desktop Entry]\nExec=true\n").unwrap();
        host.validate_dynamic_launcher(&mut entry)
            .unwrap();

        let flatpak = AppInfo::from_flatpak_info("s", flatpak_info()).unwrap();
        let mut entry = KeyFile::parse("[Desktop Entry]\nName=Calc\nExec=gnome-calculator --open\n").unwrap();
        flatpak
            .validate_dynamic_launcher(&mut entry)
            .unwrap();
        assert_eq!(
            entry.get("Desktop Entry", "Exec").unwrap(),
            "flatpak run --command=gnome-calculator 'org.example.App' --open"
        );
        assert_eq!(
            entry.get("Desktop Entry", "X-Flatpak").unwrap(),
            "org.example.App"
        );
        assert_eq!(entry.get("Desktop Entry", "TryExec"), None);

        let mut no_exec = KeyFile::parse("[Desktop Entry]\nName=Calc\n").unwrap();
        assert!(
            matches!(
                flatpak.validate_dynamic_launcher(&mut no_exec),
                Err(PortalError::InvalidArgument(_))
            ),
            "missing Exec must be rejected"
        );

        let mut bad_exec = KeyFile::parse("[Desktop Entry]\nExec='unterminated\n").unwrap();
        assert!(
            matches!(
                flatpak.validate_dynamic_launcher(&mut bad_exec),
                Err(PortalError::InvalidArgument(_))
            ),
            "unparseable Exec must be rejected"
        );

        let mut forwarding = KeyFile::parse("[Desktop Entry]\nExec=tool --file-forwarding\n").unwrap();
        assert!(
            matches!(
                flatpak.validate_dynamic_launcher(&mut forwarding),
                Err(PortalError::InvalidArgument(_))
            ),
            "--file-forwarding must be rejected"
        );

        let mut security_key =
            KeyFile::parse("[Desktop Entry]\nExec=true\nX-GNOME-Bugzilla-ExtraInfoScript=evil\n").unwrap();
        flatpak
            .validate_dynamic_launcher(&mut security_key)
            .unwrap();
        assert_eq!(
            security_key.get("Desktop Entry", "X-GNOME-Bugzilla-ExtraInfoScript"),
            None
        );

        let snap = AppInfo::from_snap_environ("s", b"SNAP_NAME=foo\0").unwrap();
        let mut entry = KeyFile::parse("[Desktop Entry]\nExec=true\n").unwrap();
        assert!(matches!(
            snap.validate_dynamic_launcher(&mut entry),
            Err(PortalError::NotAllowed(_))
        ));
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn resolves_paths_for_file_descriptors() {
        use alloc::ffi::CString;

        let host = AppInfo::host("s");

        assert!(matches!(
            host.path_for_fd(-1, 0),
            Err(PortalError::InvalidArgument(_))
        ));

        let path = CString::new("/dev/null").unwrap();
        // SAFETY: `path` is a valid C string and the flags describe
        // a plain read-only open.
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        assert!(fd >= 0, "opening /dev/null must succeed");
        let resolved = host.path_for_fd(fd, 0).unwrap();
        assert_eq!(resolved.path, "/dev/null");
        assert!(resolved.writable, "host apps may write");
        assert!(
            matches!(host.path_for_fd(fd, libc::S_IFDIR), Err(PortalError::Failed(_))),
            "mode mismatch must fail"
        );
        assert!(
            matches!(host.path_for_fd(fd, libc::S_IFREG), Err(PortalError::Failed(_))),
            "/dev/null is a character device, not a regular file"
        );
        // SAFETY: `fd` is open and will be closed.
        unsafe { libc::close(fd) };

        // SAFETY: `path` is a valid C string; O_PATH|O_NOFOLLOW is
        // rejected by the portal.
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_PATH | libc::O_NOFOLLOW | libc::O_CLOEXEC) };
        assert!(fd >= 0, "opening an O_PATH fd must succeed");
        assert!(matches!(
            host.path_for_fd(fd, 0),
            Err(PortalError::InvalidArgument(_))
        ));
        // SAFETY: `fd` is open and will be closed.
        unsafe { libc::close(fd) };

        // SAFETY: `path` is a valid C string.
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) };
        assert!(fd >= 0, "opening an O_PATH fd must succeed");
        let resolved = host.path_for_fd(fd, 0).unwrap();
        assert_eq!(resolved.path, "/dev/null");
        assert!(resolved.writable);
        // SAFETY: `fd` is open and will be closed.
        unsafe { libc::close(fd) };

        // A real regular file satisfies the S_IFREG requirement.
        let temp = std::env::temp_dir().join(format!("codevar-xdp-app-info-{}", std::process::id()));
        std::fs::write(&temp, b"codevar").unwrap();
        let temp_c = CString::new(temp.to_str().unwrap()).unwrap();
        // SAFETY: `temp_c` is a valid C string; O_RDWR matches the
        // "writable" expectation tested below.
        let fd = unsafe { libc::open(temp_c.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        assert!(fd >= 0, "opening the temp file must succeed");
        let resolved = host.path_for_fd(fd, libc::S_IFREG).unwrap();
        assert_eq!(resolved.path, temp.to_str().unwrap());
        assert!(resolved.writable, "O_RDWR fd is writable");
        // SAFETY: `fd` is open and will be closed.
        unsafe { libc::close(fd) };
        std::fs::remove_file(&temp).unwrap();
    }
}
