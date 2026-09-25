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

//! Document store registration and path resolution
//!
//! host files are offered to the `org.freedesktop.portal.Documents` service, the FUSE paths the
//! service hands back to sandboxed applications are resolved to their
//! host locations again, and the mount point the service announces is
//! cached for the rest of the portal.
//!
//! Notes:
//!
//! * [`plan_register_document`] builds a [`DocumentAddPlan`] instead of
//!   performing the `Add*` call itself, because [`Connection::call`]
//!   cannot attach file descriptors. The plan carries everything the C
//!   function does step by step: [`DocumentAddPlan::open_register_fd`],
//!   [`DocumentAddPlan::build_add_message`],
//!   [`DocumentAddPlan::read_doc_id`], [`grant_permissions`] and
//!   [`DocumentAddPlan::result_uri`].
//! * The FUSE mount point of a registration is derived from
//!   [`AppInfo::kind`] (flatpak implies `/run/flatpak/doc`) instead of
//!   the `X-Flatpak` key of a `GDesktopAppInfo`, falling back to
//!   `XDG_RUNTIME_DIR/doc` and then to the cached [`get_mount_point`]
//!   value where GLib's `g_get_user_runtime_dir()` always succeeds.
//! * There is no proxy singleton: the connection is passed explicitly
//!   and the interface version is read with an explicit
//!   `org.freedesktop.DBus.Properties.Get` rather than from the proxy's
//!   cached property.
//! * Paths are UTF-8 [`String`]s, so a percent encoded non UTF-8
//!   sequence in a `file:` URI is rejected where GLib keeps raw bytes,
//!   and a leading `//` in the URI path is collapsed where GLib keeps
//!   it.
//! * [`init_document_proxy`] logs the C warning and reports the failure
//!   as an error; proxy creation and the `GetMountPoint` call are a
//!   single call here.
//! * Failures to open the registration target map `errno` onto
//!   [`PortalError`] variants the way `g_io_error_from_errno()` feeds
//!   `G_IO_ERROR`, keeping the C message text.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

use crate::xdp_app_info::{AppInfo, AppInfoKind};
use crate::xdp_error::PortalError;
use crate::xdp_utils::{documents_mountpoint, env_var, set_documents_mountpoint};
use codevar_dbus::{BodyWriter, Connection, DbusError, DbusMessage, DbusResult, DbusTransport};

/// Bus name of the document portal service.
pub const DOCUMENTS_DBUS_NAME: &str = "org.freedesktop.portal.Documents";

/// Object path of the document portal service.
pub const DOCUMENTS_DBUS_PATH: &str = "/org/freedesktop/portal/documents";

/// Interface implemented by the document portal service.
pub const DOCUMENTS_INTERFACE: &str = "org.freedesktop.portal.Documents";

/// Interface of the bus properties used to read the portal version.
const PROPERTIES_INTERFACE: &str = "org.freedesktop.DBus.Properties";

/// Call timeout used by GDBus proxies created with default settings
/// (25 seconds).
const CALL_TIMEOUT: Duration = Duration::from_secs(25);

/// Location the FUSE mount of the document store is announced at when
/// the application runs inside flatpak, like the `X-Flatpak` branch of
/// `xdp_desktop_app_info_get_doc_mountpoint`.
const FLATPAK_DOCUMENTS_MOUNTPOINT: &str = "/run/flatpak/doc";

/// Hexadecimal digits used when percent encoding a path, like
/// `g_uri_escape_string()`.
const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

/// What the caller wants the document portal to export, mirroring
/// `XdpDocumentFlags`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DocumentFlags(u8);

impl DocumentFlags {
    /// Export an existing file as it is today.
    pub const NONE: Self = Self(0);
    /// Reserve a name for a file that does not exist yet; the parent
    /// directory is exported instead of the file.
    pub const FOR_SAVE: Self = Self(1 << 0);
    /// The application may write to the file.
    pub const WRITABLE: Self = Self(1 << 1);
    /// The export target is a directory.
    pub const DIRECTORY: Self = Self(1 << 2);
    /// The application may remove the document store entry.
    pub const DELETABLE: Self = Self(1 << 3);

    /// Returns whether every bit of `other` is present in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the raw flag bits.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }
}

impl core::ops::BitOr for DocumentFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Flags of the `AddFull` and `AddNamedFull` methods, mirroring
/// `DocumentAddFullFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocumentAddFullFlags(u8);

impl DocumentAddFullFlags {
    /// Reuse an existing document store entry for the same file.
    pub const REUSE_EXISTING: Self = Self(1 << 0);
    /// Keep the entry beyond the lifetime of the portal.
    pub const PERSISTENT: Self = Self(1 << 1);
    /// Export the file only when the application cannot reach it yet.
    pub const AS_NEEDED_BY_APP: Self = Self(1 << 2);
    /// Export a directory.
    pub const DIRECTORY: Self = Self(1 << 3);
    /// Every defined flag, `DOCUMENT_ADD_FLAGS_FLAGS_ALL`.
    pub const ALL: Self = Self((1 << 4) - 1);

    /// Returns the flag bits as sent on the wire (`u`).
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }
}

impl core::ops::BitOr for DocumentAddFullFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// How a FUSE path maps back onto the host file system, mirroring
/// `XdpResolveDocumentStrategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveDocumentStrategy {
    /// Keep the file the path points at
    /// (`XDP_RESOLVE_DOCUMENT_TO_FILE`).
    File,
    /// Reduce the path to the directory holding it
    /// (`XDP_RESOLVE_DOCUMENT_TO_DIRECTORY`).
    Directory,
}

/// The `org.freedesktop.portal.Documents` method a registration uses,
/// chosen from the interface version and [`DocumentFlags`] exactly like
/// `xdp_register_document` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentAddMethod {
    /// `Add(h, b, b)` — versions before 2.
    Add,
    /// `AddNamed(h, ay, b, b)` — versions before 3, for saves.
    AddNamed,
    /// `AddFull(ah, u, s, as)` — version 2 and later.
    AddFull,
    /// `AddNamedFull(h, ay, u, s, as)` — version 3 and later, for
    /// saves.
    AddNamedFull,
}

/// One `xdp_register_document` registration, prepared but not sent.
///
/// Building the plan is pure; the dispatch layer opens
/// [`open_path`](Self::open_path) with
/// [`open_register_fd`](Self::open_register_fd), sends the message
/// built by [`build_add_message`](Self::build_add_message), reads the
/// document id with [`read_doc_id`](Self::read_doc_id), calls
/// [`grant_permissions`] when
/// [`needs_grant_permissions`](Self::needs_grant_permissions) and turns
/// the id into the URI the application gets back with
/// [`result_uri`](Self::result_uri).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentAddPlan {
    uri: String,
    path: String,
    basename: String,
    open_path: String,
    app_id: String,
    permissions: Vec<String>,
    full_flags: DocumentAddFullFlags,
    method: DocumentAddMethod,
    needs_grant_permissions: bool,
    mountpoint: Option<String>,
}

impl DocumentAddPlan {
    /// Returns the URI the caller asked to register.
    #[must_use]
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Returns the host path decoded from [`uri`](Self::uri).
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the final path component, sent as the `filename`
    /// argument of the named methods.
    #[must_use]
    pub fn basename(&self) -> &str {
        &self.basename
    }

    /// Returns the directory the registration opens: the parent
    /// directory for [`DocumentFlags::FOR_SAVE`], the file itself
    /// otherwise.
    #[must_use]
    pub fn open_path(&self) -> &str {
        &self.open_path
    }

    /// Returns the application id the permissions are granted to.
    #[must_use]
    pub fn app_id(&self) -> &str {
        &self.app_id
    }

    /// Returns the permission list sent with the call, in the order
    /// `xdp_register_document` builds it.
    #[must_use]
    pub fn permissions(&self) -> &[String] {
        &self.permissions
    }

    /// Returns the `AddFull`/`AddNamedFull` flags.
    #[must_use]
    pub const fn full_flags(&self) -> DocumentAddFullFlags {
        self.full_flags
    }

    /// Returns the method the registration calls.
    #[must_use]
    pub const fn method(&self) -> DocumentAddMethod {
        self.method
    }

    /// Returns whether [`grant_permissions`] still has to be called
    /// after the `Add*` call, which only the versioned `*Full` methods
    /// do on the portal's side.
    #[must_use]
    pub const fn needs_grant_permissions(&self) -> bool {
        self.needs_grant_permissions
    }

    /// Returns the FUSE mount point the document id resolves under, or
    /// `None` when it cannot be determined yet.
    #[must_use]
    pub fn mountpoint(&self) -> Option<&str> {
        self.mountpoint.as_deref()
    }

    /// Returns the D-Bus method name of the registration.
    #[must_use]
    pub const fn method_name(&self) -> &'static str {
        match self.method {
            DocumentAddMethod::Add => "Add",
            DocumentAddMethod::AddNamed => "AddNamed",
            DocumentAddMethod::AddFull => "AddFull",
            DocumentAddMethod::AddNamedFull => "AddNamedFull",
        }
    }

    /// Returns the argument signature of the registration, excluding
    /// the file descriptor it indexes.
    #[must_use]
    pub const fn arg_signature(&self) -> &'static str {
        match self.method {
            DocumentAddMethod::Add => "hbb",
            DocumentAddMethod::AddNamed => "haybb",
            DocumentAddMethod::AddFull => "ahusas",
            DocumentAddMethod::AddNamedFull => "hayusas",
        }
    }

    /// Marshals the arguments of the registration, passing `fd_index`
    /// as the file descriptor argument.
    ///
    /// # Errors
    ///
    /// Returns a [`DbusError`] when the arguments do not fit the
    /// signature, as reported by [`BodyWriter`].
    pub fn write_args(&self, writer: &mut BodyWriter, fd_index: u32) -> DbusResult<()> {
        match self.method {
            DocumentAddMethod::Add => {
                writer.write_fd(fd_index)?;
                // `xdp_register_document` hardcodes both flags on the
                // pre-version-2 methods.
                writer.write_bool(true)?;
                writer.write_bool(true)?;
            }
            DocumentAddMethod::AddNamed => {
                writer.write_fd(fd_index)?;
                write_byte_string(writer, &self.basename)?;
                writer.write_bool(true)?;
                writer.write_bool(true)?;
            }
            DocumentAddMethod::AddFull => {
                writer.write_array("h", |writer| writer.write_fd(fd_index))?;
                writer.write_u32(u32::from(self.full_flags.bits()))?;
                writer.write_str(&self.app_id)?;
                write_string_array(writer, &self.permissions)?;
            }
            DocumentAddMethod::AddNamedFull => {
                writer.write_fd(fd_index)?;
                write_byte_string(writer, &self.basename)?;
                writer.write_u32(u32::from(self.full_flags.bits()))?;
                writer.write_str(&self.app_id)?;
                write_string_array(writer, &self.permissions)?;
            }
        }
        Ok(())
    }

    /// Reads the document id out of the reply of the registration.
    ///
    /// An empty `doc_ids` array of `AddFull` yields the empty string,
    /// which [`result_uri`](Self::result_uri) resolves to the original
    /// path; the C code would keep a `NULL` id in that unreachable
    /// case.
    ///
    /// # Errors
    ///
    /// Returns [`PortalError::Failed`] when the reply cannot be
    /// decoded.
    pub fn read_doc_id(&self, reply: &DbusMessage) -> Result<String, PortalError> {
        decode_doc_id(self.method, reply)
    }

    /// Turns the document id of a successful registration into the URI
    /// the application receives, like the tail of
    /// `xdp_register_document`.
    ///
    /// An empty document id means the file was not exported and the
    /// original path is returned as a URI.
    ///
    /// # Errors
    ///
    /// Returns [`PortalError::Failed`] when the FUSE mount point is
    /// needed but unknown, and [`PortalError::InvalidArgument`] when
    /// the resulting path is not absolute.
    pub fn result_uri(&self, doc_id: &str) -> Result<String, PortalError> {
        if doc_id.is_empty() {
            return filename_to_uri(&self.path);
        }
        let mountpoint = self.mountpoint.as_deref().ok_or_else(|| {
            PortalError::Failed(String::from("Cannot determine the document portal mount point"))
        })?;
        filename_to_uri(&build_filename(&[mountpoint, doc_id, &self.basename]))
    }

    /// Opens the registration target the way `xdp_register_document`
    /// does: the parent directory for [`DocumentFlags::FOR_SAVE`], the
    /// file itself otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`PortalError::NotAllowed`] when access is denied,
    /// [`PortalError::NotFound`] when the target is missing and
    /// [`PortalError::Failed`] for any other `errno`, all with the C
    /// message `Failed to open <uri>`.
    ///
    /// The returned descriptor belongs to the caller; hand it to
    /// [`build_add_message`](Self::build_add_message) or close it.
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    pub fn open_register_fd(&self) -> Result<i32, PortalError> {
        use alloc::ffi::CString;

        let target = CString::new(self.open_path.as_str())
            .map_err(|_| PortalError::InvalidArgument(String::from("path contains a nul byte")))?;
        // SAFETY: `target` is a valid NUL-terminated path string and
        // the returned descriptor is checked for failure before use.
        let raw_fd = unsafe { libc::open(target.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        if raw_fd < 0 {
            // SAFETY: the call failed, so `errno` is live.
            let errno = unsafe { *libc::__errno_location() };
            let message = format!("Failed to open {}", self.uri);
            return Err(match errno {
                libc::EACCES | libc::EPERM => PortalError::NotAllowed(message),
                libc::ENOENT => PortalError::NotFound(message),
                _ => PortalError::Failed(message),
            });
        }
        Ok(raw_fd)
    }

    /// Builds the method call for the registration with `fd` attached
    /// as its file descriptor.
    ///
    /// Ownership of `fd` moves into the message: the connection closes
    /// it once the call has been handed to the kernel. A message that
    /// is dropped without being sent keeps the descriptor, which the
    /// caller then takes back with [`DbusMessage::take_fds`].
    ///
    /// # Errors
    ///
    /// Returns a [`DbusError`] when the method call or its body cannot
    /// be built.
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    pub fn build_add_message(&self, fd: i32) -> DbusResult<DbusMessage> {
        let mut message = DbusMessage::method_call(
            DOCUMENTS_DBUS_NAME,
            DOCUMENTS_DBUS_PATH,
            DOCUMENTS_INTERFACE,
            self.method_name(),
        )?;
        message.build_body(|writer| self.write_args(writer, 0))?;
        message.set_fds(alloc::vec![fd]);
        Ok(message)
    }
}

/// Returns the FUSE mount point documents of `app_info` appear under,
/// like `xdp_desktop_app_info_get_doc_mountpoint`.
///
/// Flatpak applications use `/run/flatpak/doc`; every other kind uses
/// `$XDG_RUNTIME_DIR/doc`, falling back to the mount point reported by
/// [`get_mount_point`] when the environment does not say.
#[must_use]
pub fn doc_mountpoint(app_info: &AppInfo) -> Option<String> {
    if app_info.kind() == AppInfoKind::Flatpak {
        return Some(String::from(FLATPAK_DOCUMENTS_MOUNTPOINT));
    }
    if let Some(runtime_dir) = user_runtime_dir() {
        return Some(format!("{runtime_dir}/doc"));
    }
    documents_mountpoint()
}

/// Prepares the registration `xdp_register_document` performs for
/// `uri`.
///
/// The plan describes which method to call with which arguments; see
/// [`DocumentAddPlan`] for the steps the dispatch layer runs with it.
///
/// # Errors
///
/// Returns [`PortalError::InvalidArgument`] when `app_id` is empty or
/// when `uri` is not a `file:` URI the document portal can export.
pub fn plan_register_document(
    uri: &str,
    app_id: &str,
    app_info: &AppInfo,
    flags: DocumentFlags,
    version: u32,
) -> Result<DocumentAddPlan, PortalError> {
    if app_id.is_empty() {
        return Err(PortalError::InvalidArgument(String::from(
            "app id must not be empty",
        )));
    }
    let path = uri_to_path(uri).ok_or_else(|| {
        PortalError::InvalidArgument(format!("URI {uri} not supported by the document portal"))
    })?;
    let basename = path_basename(&path);
    let open_path = if flags.contains(DocumentFlags::FOR_SAVE) {
        path_dirname(&path)
    } else {
        path.clone()
    };

    let method = match (flags.contains(DocumentFlags::FOR_SAVE), version) {
        (true, 3..) => DocumentAddMethod::AddNamedFull,
        (true, _) => DocumentAddMethod::AddNamed,
        (false, 2..) => DocumentAddMethod::AddFull,
        (false, _) => DocumentAddMethod::Add,
    };
    let needs_grant_permissions = matches!(method, DocumentAddMethod::Add | DocumentAddMethod::AddNamed);

    Ok(DocumentAddPlan {
        uri: String::from(uri),
        path,
        basename,
        open_path,
        app_id: String::from(app_id),
        permissions: permissions_for(flags),
        full_flags: add_full_flags(flags),
        method,
        needs_grant_permissions,
        mountpoint: doc_mountpoint(app_info),
    })
}

/// Converts the `file:` URI `uri` into a host path, like
/// `g_file_new_for_uri()` followed by `g_file_get_path()`.
///
/// Returns `None` for every other scheme, for a URI without a path and
/// for percent escapes GLib refuses or cannot represent in a UTF-8
/// path. Query and fragment are dropped and `.`/`..` segments are
/// resolved, as GLib does.
#[must_use]
pub fn uri_to_path(uri: &str) -> Option<String> {
    let (scheme, rest) = uri.split_once(':')?;
    if !scheme.eq_ignore_ascii_case("file") {
        return None;
    }
    let raw_path = match rest.strip_prefix("//") {
        Some(after_authority) if after_authority.starts_with('/') => after_authority,
        Some(after_authority) => &after_authority[after_authority.find('/')?..],
        None if rest.starts_with('/') => rest,
        None => return None,
    };
    let end = raw_path.find(['?', '#']).unwrap_or(raw_path.len());
    let decoded = decode_percent_encoding(&raw_path[..end])?;
    Some(normalize_path(&decoded))
}

/// Converts the absolute host `path` into a `file:` URI, like
/// `g_filename_to_uri()`.
///
/// # Errors
///
/// Returns [`PortalError::InvalidArgument`] when `path` is not
/// absolute.
pub fn filename_to_uri(path: &str) -> Result<String, PortalError> {
    if !path.starts_with('/') {
        return Err(PortalError::InvalidArgument(format!(
            "Path {path} is not an absolute path"
        )));
    }
    let mut uri = String::with_capacity(path.len() + "file://".len());
    uri.push_str("file://");
    for &byte in path.as_bytes() {
        if is_uri_path_safe(byte) {
            uri.push(char::from(byte));
        } else {
            uri.push('%');
            uri.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
            uri.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
        }
    }
    Ok(uri)
}

/// Splits `path` into the document id it points at and the part of the
/// path below the directory name the FUSE mount inserts, like
/// `xdp_looks_like_document_portal_path`.
///
/// Returns `None` when `path` does not live under `runtime_dir`, does
/// not contain `/doc/` or names an empty document id. An empty
/// `runtime_dir` never matches, so a missing `XDG_RUNTIME_DIR` cannot
/// turn every path into a document path.
#[must_use]
pub fn parse_document_portal_path<'a>(path: &'a str, runtime_dir: &str) -> Option<(&'a str, &'a str)> {
    if runtime_dir.is_empty() || !path.starts_with(runtime_dir) {
        return None;
    }
    let rest = path.split_once("/doc/")?.1;
    let (doc_id, tail) = match rest.split_once('/') {
        Some((doc_id, tail)) => (doc_id, tail),
        None => (rest, ""),
    };
    if doc_id.is_empty() {
        return None;
    }
    // The mapping to the host path already provides the directory
    // name right after the document id, so it is dropped here.
    let suffix = match tail.find('/') {
        Some(index) => &tail[index..],
        None => "",
    };
    Some((doc_id, suffix))
}

/// Resolves a FUSE path to its host path with a caller supplied
/// lookup, like `xdp_resolve_document_portal_path`.
///
/// `lookup` answers [`get_real_path_for_doc_id`]-style questions about
/// a document id; when it returns `None`, or when `path` is not a
/// document path, `path` is returned unchanged.
#[must_use]
pub fn resolve_document_portal_path_with<F>(
    path: &str,
    strategy: ResolveDocumentStrategy,
    runtime_dir: &str,
    lookup: F,
) -> String
where
    F: FnOnce(&str) -> Option<String>,
{
    let Some((doc_id, suffix)) = parse_document_portal_path(path, runtime_dir) else {
        return String::from(path);
    };
    let Some(host_path) = lookup(doc_id) else {
        return String::from(path);
    };
    let mut resolved = format!("{host_path}{suffix}");
    if strategy == ResolveDocumentStrategy::Directory
        && let Some(index) = resolved.rfind('/')
    {
        resolved.truncate(index);
    }
    resolved
}

/// Resolves a FUSE path to its host path, like
/// `xdp_resolve_document_portal_path`.
///
/// Never fails: when the document portal does not know the document
/// (or the environment carries no `XDG_RUNTIME_DIR`) the input path is
/// returned unchanged.
#[must_use]
pub fn resolve_document_portal_path<T: DbusTransport>(
    connection: &mut Connection<T>,
    path: &str,
    strategy: ResolveDocumentStrategy,
) -> String {
    let Some(runtime_dir) = user_runtime_dir() else {
        return String::from(path);
    };
    resolve_document_portal_path_with(path, strategy, &runtime_dir, |doc_id| {
        get_real_path_for_doc_id(connection, doc_id).ok()
    })
}

/// Creates the document portal proxy and caches the FUSE mount point
/// it reports, like `xdp_init_document_proxy`.
///
/// The C code logs a warning and keeps running without a mount point
/// when `GetMountPoint` fails; here the failure is reported after the
/// same warning is logged and the cache is cleared, so the caller
/// decides whether that is fatal.
///
/// # Errors
///
/// Returns the underlying [`PortalError`] when the document portal
/// cannot be reached or replies with an unusable path.
pub fn init_document_proxy<T: DbusTransport>(connection: &mut Connection<T>) -> Result<String, PortalError> {
    match get_mount_point(connection) {
        Ok(path) => {
            set_documents_mountpoint(Some(&path));
            Ok(path)
        }
        Err(error) => {
            log::warn!("Document portal fuse mount point unknown: {}", error.message());
            set_documents_mountpoint(None);
            Err(error)
        }
    }
}

/// Reads the FUSE mount point of the document store, the
/// `GetMountPoint` method.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when the call or the `ay`
/// bytestring reply cannot be decoded.
pub fn get_mount_point<T: DbusTransport>(connection: &mut Connection<T>) -> Result<String, PortalError> {
    let reply = connection.call(
        DOCUMENTS_DBUS_NAME,
        DOCUMENTS_DBUS_PATH,
        DOCUMENTS_INTERFACE,
        "GetMountPoint",
        |_writer| Ok(()),
        CALL_TIMEOUT,
    )?;
    decode_path_reply(&reply, "GetMountPoint")
}

/// Reads the `version` property of the document portal interface, the
/// value `xdp_dbus_documents_get_version()` returns from the proxy's
/// property cache.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when the call fails or the reply is
/// not a `u` variant.
pub fn get_documents_version<T: DbusTransport>(connection: &mut Connection<T>) -> Result<u32, PortalError> {
    let reply = connection.call(
        DOCUMENTS_DBUS_NAME,
        DOCUMENTS_DBUS_PATH,
        PROPERTIES_INTERFACE,
        "Get",
        |writer| {
            writer.write_str(DOCUMENTS_INTERFACE)?;
            writer.write_str("version")?;
            Ok(())
        },
        CALL_TIMEOUT,
    )?;
    decode_version_reply(&reply)
}

/// Looks up the host path of `doc_id`, the `Info` method, like
/// `xdp_get_real_path_for_doc_id`.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when the call fails or the reply
/// cannot be decoded.
pub fn get_real_path_for_doc_id<T: DbusTransport>(
    connection: &mut Connection<T>,
    doc_id: &str,
) -> Result<String, PortalError> {
    let outcome: Result<String, PortalError> = connection
        .call(
            DOCUMENTS_DBUS_NAME,
            DOCUMENTS_DBUS_PATH,
            DOCUMENTS_INTERFACE,
            "Info",
            |writer| writer.write_str(doc_id),
            CALL_TIMEOUT,
        )
        .map_err(PortalError::from)
        .and_then(|reply| decode_path_reply(&reply, "Info"));
    if let Err(error) = &outcome {
        log::debug!("document portal error for doc id '{doc_id}': {}", error.message());
    }
    outcome
}

/// Grants `permissions` on `doc_id` to `app_id`, the
/// `GrantPermissions` method.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when the document portal rejects
/// the call or the transport fails.
pub fn grant_permissions<T: DbusTransport>(
    connection: &mut Connection<T>,
    doc_id: &str,
    app_id: &str,
    permissions: &[String],
) -> Result<(), PortalError> {
    connection.call(
        DOCUMENTS_DBUS_NAME,
        DOCUMENTS_DBUS_PATH,
        DOCUMENTS_INTERFACE,
        "GrantPermissions",
        |writer| write_grant_permissions_args(writer, doc_id, app_id, permissions),
        CALL_TIMEOUT,
    )?;
    Ok(())
}

/// Returns `$XDG_RUNTIME_DIR`, the directory GLib's
/// `g_get_user_runtime_dir()` reports on a session bus system.
fn user_runtime_dir() -> Option<String> {
    env_var("XDG_RUNTIME_DIR").filter(|value| !value.is_empty())
}

/// Builds the permission list `xdp_register_document` sends: `read`,
/// then `write` for writable targets and saves, always
/// `grant-permissions`, and `delete` for deletable ones.
fn permissions_for(flags: DocumentFlags) -> Vec<String> {
    let mut permissions = Vec::new();
    permissions.push(String::from("read"));
    if flags.contains(DocumentFlags::WRITABLE) || flags.contains(DocumentFlags::FOR_SAVE) {
        permissions.push(String::from("write"));
    }
    permissions.push(String::from("grant-permissions"));
    if flags.contains(DocumentFlags::DELETABLE) {
        permissions.push(String::from("delete"));
    }
    permissions
}

/// Builds the `AddFull` flags `xdp_register_document` uses.
fn add_full_flags(flags: DocumentFlags) -> DocumentAddFullFlags {
    let mut full_flags = DocumentAddFullFlags::REUSE_EXISTING
        | DocumentAddFullFlags::PERSISTENT
        | DocumentAddFullFlags::AS_NEEDED_BY_APP;
    if flags.contains(DocumentFlags::DIRECTORY) {
        full_flags = full_flags | DocumentAddFullFlags::DIRECTORY;
    }
    full_flags
}

/// Marshals the `GrantPermissions(s, s, as)` arguments.
fn write_grant_permissions_args(
    writer: &mut BodyWriter,
    doc_id: &str,
    app_id: &str,
    permissions: &[String],
) -> DbusResult<()> {
    writer.write_str(doc_id)?;
    writer.write_str(app_id)?;
    write_string_array(writer, permissions)
}

/// Marshals an `as` array of strings.
fn write_string_array(writer: &mut BodyWriter, values: &[String]) -> DbusResult<()> {
    writer.write_array("s", |writer| {
        for value in values {
            writer.write_str(value)?;
        }
        Ok(())
    })
}

/// Marshals a `GVariant` bytestring (`ay`), including the trailing
/// NUL byte `g_variant_new_bytestring()` adds.
fn write_byte_string(writer: &mut BodyWriter, value: &str) -> DbusResult<()> {
    writer.write_array("y", |writer| {
        for byte in value.as_bytes().iter().chain(core::iter::once(&0u8)) {
            writer.write_u8(*byte)?;
        }
        Ok(())
    })
}

/// Reads the `ay` bytestring that starts the `GetMountPoint` and
/// `Info` replies and drops its trailing NUL terminator.
fn decode_path_reply(reply: &DbusMessage, context: &str) -> Result<String, PortalError> {
    let mut reader = reply.body_reader();
    let mut bytes = reader
        .read_array(1)
        .map_err(|error| decode_failed(context, error))?;
    let mut value = Vec::new();
    while !bytes.is_empty() {
        let byte = bytes
            .read_u8()
            .map_err(|error| decode_failed(context, error))?;
        value.push(byte);
    }
    if value.last() == Some(&0u8) {
        value.pop();
    }
    String::from_utf8(value)
        .map_err(|_| PortalError::Failed(format!("The {context} reply is not valid UTF-8")))
}

/// Reads the `u` variant the `Properties.Get` reply wraps the
/// interface version in.
fn decode_version_reply(reply: &DbusMessage) -> Result<u32, PortalError> {
    let context = "version";
    let mut reader = reply.body_reader();
    let signature = reader
        .read_variant_signature()
        .map_err(|error| decode_failed(context, error))?;
    if signature != "u" {
        return Err(PortalError::Failed(format!(
            "Expected a 'u' version, got '{signature}'"
        )));
    }
    reader
        .read_u32()
        .map_err(|error| decode_failed(context, error))
}

/// Reads the document id out of an `Add*` reply; `AddFull` answers
/// with an array and takes its first entry, as the C code does.
fn decode_doc_id(method: DocumentAddMethod, reply: &DbusMessage) -> Result<String, PortalError> {
    let context = match method {
        DocumentAddMethod::AddFull => "AddFull",
        DocumentAddMethod::AddNamedFull => "AddNamedFull",
        DocumentAddMethod::Add => "Add",
        DocumentAddMethod::AddNamed => "AddNamed",
    };
    let mut reader = reply.body_reader();
    if method == DocumentAddMethod::AddFull {
        let mut doc_ids = reader
            .read_array(4)
            .map_err(|error| decode_failed(context, error))?;
        if doc_ids.is_empty() {
            return Ok(String::new());
        }
        let doc_id = doc_ids
            .read_str()
            .map_err(|error| decode_failed(context, error))?;
        return Ok(String::from(doc_id));
    }
    let doc_id = reader
        .read_str()
        .map_err(|error| decode_failed(context, error))?;
    Ok(String::from(doc_id))
}

/// Wraps a decoding failure with the reply it came from.
fn decode_failed(context: &str, error: DbusError) -> PortalError {
    PortalError::Failed(format!("Cannot decode the {context} reply: {error}"))
}

/// Returns the final path component of `path`, like
/// `g_path_get_basename()`.
fn path_basename(path: &str) -> String {
    if path.is_empty() {
        return String::from(".");
    }
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        return String::from("/");
    }
    match trimmed.rfind('/') {
        Some(index) => String::from(&trimmed[index + 1..]),
        None => String::from(trimmed),
    }
}

/// Returns the directory part of `path`, like
/// `g_path_get_dirname()`.
fn path_dirname(path: &str) -> String {
    let Some(index) = path.rfind('/') else {
        return String::from(".");
    };
    let dir = path[..index].trim_end_matches('/');
    if dir.is_empty() {
        String::from("/")
    } else {
        String::from(dir)
    }
}

/// Joins the non-empty `parts` with `/`, keeping the separators the
/// parts already carry, like `g_build_filename()`.
fn build_filename(parts: &[&str]) -> String {
    let mut out = String::new();
    for part in parts.iter().filter(|part| !part.is_empty()) {
        if out.is_empty() {
            out.push_str(part);
        } else {
            if !out.ends_with('/') && !part.starts_with('/') {
                out.push('/');
            }
            out.push_str(part);
        }
    }
    out
}

/// Resolves `.` and `..` segments, collapses repeated separators and
/// drops the trailing one, the path handling `g_file_get_path()`
/// applies to a `file:` URI.
fn normalize_path(path: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    if segments.is_empty() {
        return String::from("/");
    }
    let mut normalized = String::with_capacity(path.len());
    for segment in segments {
        normalized.push('/');
        normalized.push_str(segment);
    }
    normalized
}

/// Percent decodes `value` the way GLib unescapes a `file:` URI path:
/// a bad escape, an encoded NUL or an encoded separator rejects the
/// whole path, and the result must be UTF-8.
fn decode_percent_encoding(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        let high = hex_digit(*bytes.get(index + 1)?)?;
        let low = hex_digit(*bytes.get(index + 2)?)?;
        let byte = high << 4 | low;
        if byte == 0 || byte == b'/' {
            return None;
        }
        decoded.push(byte);
        index += 3;
    }
    String::from_utf8(decoded).ok()
}

/// Returns the value of a hexadecimal digit byte.
fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Returns whether `byte` stays unescaped in a `file:` URI path, the
/// set `g_filename_to_uri()` keeps.
const fn is_uri_path_safe(byte: u8) -> bool {
    matches!(
        byte,
        b'/'
            | b'!'
            | b'$'
            | b'&'
            | b'\''
            | b'('
            | b')'
            | b'*'
            | b'+'
            | b','
            | b'-'
            | b'.'
            | b'0'..=b'9'
            | b':'
            | b'='
            | b'@'
            | b'A'..=b'Z'
            | b'_'
            | b'a'..=b'z'
            | b'~'
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_dbus::{ByteOrder, DbusReader};

    // Every unwrap() below operates on a value this test constructed
    // itself with known-good inputs, which AGENTS.md allows.

    fn flatpak_app_info() -> AppInfo {
        AppInfo::from_flatpak_info(
            "sender",
            "[Application]\nname=org.example.App\n[Instance]\ninstance-id=i\n",
        )
        // Justified: the metadata above is valid for the parser.
        .unwrap()
    }

    fn plan_with(uri: &str, flags: DocumentFlags, version: u32) -> DocumentAddPlan {
        plan_register_document(uri, "org.example.App", &flatpak_app_info(), flags, version)
            // Justified: a file URI and a valid app id cannot fail.
            .unwrap()
    }

    #[test]
    fn converts_file_uris_to_paths_like_glib() {
        assert_eq!(uri_to_path("file:///tmp/a%20b").as_deref(), Some("/tmp/a b"));
        assert_eq!(uri_to_path("file://localhost/tmp/x").as_deref(), Some("/tmp/x"));
        assert_eq!(uri_to_path("file://otherhost/tmp/x").as_deref(), Some("/tmp/x"));
        assert_eq!(uri_to_path("file:/tmp/x").as_deref(), Some("/tmp/x"));
        assert_eq!(uri_to_path("FILE:///tmp/x").as_deref(), Some("/tmp/x"));
        assert_eq!(uri_to_path("file:///").as_deref(), Some("/"));
        assert_eq!(uri_to_path("file:///tmp/x?a=1#f").as_deref(), Some("/tmp/x"));
        assert_eq!(
            uri_to_path("file:///tmp/uni%C3%BCcode.txt").as_deref(),
            Some("/tmp/uni\u{fc}code.txt")
        );

        // Dot segments are resolved and the trailing separator drops.
        assert_eq!(uri_to_path("file:///tmp/x/../y").as_deref(), Some("/tmp/y"));
        assert_eq!(uri_to_path("file:///tmp/./").as_deref(), Some("/tmp"));
        assert_eq!(uri_to_path("file:///tmp/x/").as_deref(), Some("/tmp/x"));
        assert_eq!(uri_to_path("file:///tmp//x").as_deref(), Some("/tmp/x"));
        assert_eq!(uri_to_path("file:///../x").as_deref(), Some("/x"));
        assert_eq!(uri_to_path("file:///tmp/a//b/../c").as_deref(), Some("/tmp/a/c"));

        // Not a file URI, not a path, or an escape GLib rejects.
        assert_eq!(uri_to_path("http://h/p"), None);
        assert_eq!(uri_to_path("/tmp/plain"), None);
        assert_eq!(uri_to_path("file:relative"), None);
        assert_eq!(uri_to_path("file://"), None);
        assert_eq!(uri_to_path("file:///tmp/a%2"), None);
        assert_eq!(uri_to_path("file:///tmp/a%GG"), None);
        assert_eq!(uri_to_path("file:///tmp/a%00b"), None);
        assert_eq!(uri_to_path("file:///tmp/x%2Fy"), None);
        assert_eq!(uri_to_path("file:///tmp/a%FFb"), None);
    }

    #[test]
    fn converts_paths_to_file_uris_like_glib() {
        assert_eq!(
            // Justified: an absolute path cannot be rejected.
            filename_to_uri("/tmp/a b").unwrap(),
            "file:///tmp/a%20b"
        );
        assert_eq!(filename_to_uri("/tmp/a#b").unwrap(), "file:///tmp/a%23b");
        assert_eq!(filename_to_uri("/tmp/a?b").unwrap(), "file:///tmp/a%3Fb");
        assert_eq!(filename_to_uri("/tmp/a%b").unwrap(), "file:///tmp/a%25b");
        assert_eq!(filename_to_uri("/tmp/a;b").unwrap(), "file:///tmp/a%3Bb");
        assert_eq!(filename_to_uri("/tmp/a+b").unwrap(), "file:///tmp/a+b");
        assert_eq!(filename_to_uri("/tmp/a~b.txt").unwrap(), "file:///tmp/a~b.txt");
        assert_eq!(
            filename_to_uri("/tmp/\u{fc}n\u{ef}code.txt").unwrap(),
            "file:///tmp/%C3%BCn%C3%AFcode.txt"
        );

        for path in ["/tmp/a b", "/tmp/a#b", "/tmp/\u{fc}n\u{ef}code.txt"] {
            let uri = filename_to_uri(path).unwrap();
            assert_eq!(uri_to_path(&uri).as_deref(), Some(path));
        }

        assert!(matches!(
            filename_to_uri("relative/path"),
            Err(PortalError::InvalidArgument(_))
        ));
    }

    #[test]
    fn splits_paths_into_basename_and_dirname_like_glib() {
        for (path, basename, dirname) in [
            ("/tmp/x", "x", "/tmp"),
            ("/tmp/x/", "x", "/tmp/x"),
            ("/tmp/", "tmp", "/tmp"),
            ("/", "/", "/"),
            ("", ".", "."),
            ("abc", "abc", "."),
            ("a/b", "b", "a"),
            ("/tmp/a//b", "b", "/tmp/a"),
            ("//x", "x", "/"),
            ("//x//y", "y", "//x"),
            ("a/", "a", "a"),
            (".", ".", "."),
            ("..", "..", "."),
        ] {
            assert_eq!(path_basename(path), basename, "basename of {path}");
            assert_eq!(path_dirname(path), dirname, "dirname of {path}");
        }
    }

    #[test]
    fn selects_the_add_method_from_version_and_flags() {
        let save_new = plan_with("file:///tmp/dir/file.txt", DocumentFlags::FOR_SAVE, 5);
        assert_eq!(save_new.method(), DocumentAddMethod::AddNamedFull);
        assert!(!save_new.needs_grant_permissions());
        assert_eq!(save_new.method_name(), "AddNamedFull");
        assert_eq!(save_new.arg_signature(), "hayusas");
        assert_eq!(save_new.open_path(), "/tmp/dir");

        let save_old = plan_with("file:///tmp/dir/file.txt", DocumentFlags::FOR_SAVE, 2);
        assert_eq!(save_old.method(), DocumentAddMethod::AddNamed);
        assert!(save_old.needs_grant_permissions());
        assert_eq!(save_old.arg_signature(), "haybb");

        let open_new = plan_with("file:///tmp/dir/file.txt", DocumentFlags::NONE, 4);
        assert_eq!(open_new.method(), DocumentAddMethod::AddFull);
        assert!(!open_new.needs_grant_permissions());
        assert_eq!(open_new.arg_signature(), "ahusas");
        assert_eq!(open_new.open_path(), "/tmp/dir/file.txt");

        let open_old = plan_with("file:///tmp/dir/file.txt", DocumentFlags::NONE, 1);
        assert_eq!(open_old.method(), DocumentAddMethod::Add);
        assert!(open_old.needs_grant_permissions());
        assert_eq!(open_old.arg_signature(), "hbb");
    }

    #[test]
    fn collects_permissions_and_full_flags_from_flags() {
        let plain = plan_with("file:///tmp/f", DocumentFlags::NONE, 5);
        assert_eq!(
            plain
                .permissions()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["read", "grant-permissions"]
        );
        assert_eq!(
            plain.full_flags(),
            DocumentAddFullFlags::REUSE_EXISTING
                | DocumentAddFullFlags::PERSISTENT
                | DocumentAddFullFlags::AS_NEEDED_BY_APP
        );

        let save = plan_with("file:///tmp/f", DocumentFlags::FOR_SAVE, 5);
        assert_eq!(
            save.permissions()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["read", "write", "grant-permissions"]
        );

        let all = plan_with(
            "file:///tmp/f",
            DocumentFlags::WRITABLE
                | DocumentFlags::DELETABLE
                | DocumentFlags::DIRECTORY
                | DocumentFlags::FOR_SAVE,
            5,
        );
        assert_eq!(
            all.permissions()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["read", "write", "grant-permissions", "delete"]
        );
        assert_eq!(
            all.full_flags(),
            DocumentAddFullFlags::ALL,
            "every add flag is set"
        );
        assert_eq!(DocumentAddFullFlags::ALL.bits(), 0x0f);
    }

    #[test]
    fn plans_result_uris_and_rejects_bad_input() {
        let flatpak = flatpak_app_info();
        let plan = plan_register_document(
            "file:///tmp/my%20file.txt",
            "org.example.App",
            &flatpak,
            DocumentFlags::NONE,
            5,
        )
        .unwrap();
        assert_eq!(plan.uri(), "file:///tmp/my%20file.txt");
        assert_eq!(plan.path(), "/tmp/my file.txt");
        assert_eq!(plan.basename(), "my file.txt");
        assert_eq!(plan.app_id(), "org.example.App");
        assert_eq!(plan.mountpoint(), Some("/run/flatpak/doc"));
        assert_eq!(plan.result_uri("").unwrap(), "file:///tmp/my%20file.txt");
        assert_eq!(
            plan.result_uri("abc123").unwrap(),
            "file:///run/flatpak/doc/abc123/my%20file.txt"
        );

        // The host mount point follows $XDG_RUNTIME_DIR when the
        // environment provides it.
        if let Some(runtime_dir) = env_var("XDG_RUNTIME_DIR") {
            assert_eq!(
                doc_mountpoint(&AppInfo::host("sender")),
                Some(format!("{runtime_dir}/doc"))
            );
        }

        assert!(matches!(
            plan_register_document("file:///tmp/f", "", &flatpak, DocumentFlags::NONE, 5),
            Err(PortalError::InvalidArgument(_))
        ));
        assert!(matches!(
            plan_register_document(
                "http://example.com/f",
                "org.example.App",
                &flatpak,
                DocumentFlags::NONE,
                5
            ),
            Err(PortalError::InvalidArgument(_))
        ));

        // Without a mount point the document id cannot be mapped.
        let mut orphaned = plan;
        orphaned.mountpoint = None;
        assert!(matches!(
            orphaned.result_uri("abc123"),
            Err(PortalError::Failed(_))
        ));
        assert!(orphaned.result_uri("").is_ok());
    }

    #[test]
    fn marshals_the_add_calls() {
        let plan = plan_with("file:///tmp/dir/file.txt", DocumentFlags::NONE, 1);
        let mut body = BodyWriter::new(ByteOrder::Little);
        plan.write_args(&mut body, 3).unwrap();
        let (bytes, signature) = body.into_parts();
        assert_eq!(signature, plan.arg_signature());
        assert_eq!(signature, "hbb");
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert_eq!(reader.read_fd().unwrap(), 3);
        assert!(reader.read_bool().unwrap());
        assert!(reader.read_bool().unwrap());
        assert!(reader.is_empty());

        let plan = plan_with("file:///tmp/dir/file.txt", DocumentFlags::FOR_SAVE, 2);
        let mut body = BodyWriter::new(ByteOrder::Little);
        plan.write_args(&mut body, 0).unwrap();
        let (bytes, signature) = body.into_parts();
        assert_eq!(signature, "haybb");
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert_eq!(reader.read_fd().unwrap(), 0);
        let mut filename = reader.read_array(1).unwrap();
        let mut name = Vec::new();
        while !filename.is_empty() {
            name.push(filename.read_u8().unwrap());
        }
        // The filename travels as a NUL terminated bytestring.
        assert_eq!(name, b"file.txt\0");
        assert!(reader.read_bool().unwrap());
        assert!(reader.read_bool().unwrap());
        assert!(reader.is_empty());

        let plan = plan_with("file:///tmp/dir/file.txt", DocumentFlags::DIRECTORY, 4);
        let mut body = BodyWriter::new(ByteOrder::Little);
        plan.write_args(&mut body, 2).unwrap();
        let (bytes, signature) = body.into_parts();
        assert_eq!(signature, "ahusas");
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        let mut handles = reader.read_array(4).unwrap();
        assert_eq!(handles.read_fd().unwrap(), 2);
        assert!(handles.is_empty());
        assert_eq!(reader.read_u32().unwrap(), u32::from(plan.full_flags().bits()));
        assert_eq!(reader.read_str().unwrap(), "org.example.App");
        let mut permissions = reader.read_array(4).unwrap();
        assert_eq!(permissions.read_str().unwrap(), "read");
        assert_eq!(permissions.read_str().unwrap(), "grant-permissions");
        assert!(permissions.is_empty());
        assert!(reader.is_empty());

        let plan = plan_with("file:///tmp/dir/file.txt", DocumentFlags::FOR_SAVE, 3);
        let mut body = BodyWriter::new(ByteOrder::Little);
        plan.write_args(&mut body, 1).unwrap();
        let (bytes, signature) = body.into_parts();
        assert_eq!(signature, "hayusas");
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert_eq!(reader.read_fd().unwrap(), 1);
        let mut filename = reader.read_array(1).unwrap();
        assert_eq!(filename.read_u8().unwrap(), b'f');
        let mut rest = Vec::new();
        while !filename.is_empty() {
            rest.push(filename.read_u8().unwrap());
        }
        assert_eq!(rest, b"ile.txt\0");
        assert_eq!(reader.read_u32().unwrap(), u32::from(plan.full_flags().bits()));
        assert_eq!(reader.read_str().unwrap(), "org.example.App");
        let mut permissions = reader.read_array(4).unwrap();
        assert_eq!(permissions.read_str().unwrap(), "read");
        assert_eq!(permissions.read_str().unwrap(), "write");
        assert_eq!(permissions.read_str().unwrap(), "grant-permissions");
        assert!(permissions.is_empty());
        assert!(reader.is_empty());
    }

    #[test]
    fn encodes_grant_permissions_arguments() {
        let permissions = Vec::from([String::from("read"), String::from("delete")]);
        let mut body = BodyWriter::new(ByteOrder::Little);
        write_grant_permissions_args(&mut body, "abc123", "org.example.App", &permissions).unwrap();
        let (bytes, signature) = body.into_parts();
        assert_eq!(signature, "ssas");
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert_eq!(reader.read_str().unwrap(), "abc123");
        assert_eq!(reader.read_str().unwrap(), "org.example.App");
        let mut granted = reader.read_array(4).unwrap();
        assert_eq!(granted.read_str().unwrap(), "read");
        assert_eq!(granted.read_str().unwrap(), "delete");
        assert!(granted.is_empty());
        assert!(reader.is_empty());
    }

    fn reply_with<F>(signature: &str, body: F) -> DbusMessage
    where
        F: FnOnce(&mut BodyWriter) -> DbusResult<()>,
    {
        let mut reply = DbusMessage::method_return(7);
        reply.set_serial(1000).unwrap();
        reply.build_body(body).unwrap();
        assert_eq!(reply.signature(), signature);
        reply
    }

    #[test]
    fn decodes_documents_replies() {
        let mount_point = reply_with("ay", |writer| write_byte_string(writer, "/run/user/1000/doc"));
        assert_eq!(
            decode_path_reply(&mount_point, "GetMountPoint").unwrap(),
            "/run/user/1000/doc"
        );

        let info = reply_with("aya{sas}", |writer| {
            write_byte_string(writer, "/home/user/file.txt")?;
            writer.write_array("{sas}", |writer| {
                writer.write_struct("sas", |writer| {
                    writer.write_str("org.example.App")?;
                    writer.write_array("s", |writer| writer.write_str("read"))
                })
            })
        });
        assert_eq!(decode_path_reply(&info, "Info").unwrap(), "/home/user/file.txt");

        let version = reply_with("v", |writer| {
            writer.write_variant("u", |writer| writer.write_u32(5))
        });
        assert_eq!(decode_version_reply(&version).unwrap(), 5);

        let wrong = reply_with("s", |writer| writer.write_str("org.example"));
        assert!(matches!(
            decode_version_reply(&wrong),
            Err(PortalError::Failed(_))
        ));

        let single = reply_with("s", |writer| writer.write_str("abc123"));
        for method in [
            DocumentAddMethod::Add,
            DocumentAddMethod::AddNamed,
            DocumentAddMethod::AddNamedFull,
        ] {
            assert_eq!(decode_doc_id(method, &single).unwrap(), "abc123");
        }

        let many = reply_with("asa{sv}", |writer| {
            writer.write_array("s", |writer| {
                writer.write_str("first")?;
                writer.write_str("second")
            })?;
            writer.write_array("{sv}", |_writer| Ok(()))
        });
        assert_eq!(decode_doc_id(DocumentAddMethod::AddFull, &many).unwrap(), "first");

        let empty = reply_with("asa{sv}", |writer| {
            writer.write_array("s", |_writer| Ok(()))?;
            writer.write_array("{sv}", |_writer| Ok(()))
        });
        assert_eq!(decode_doc_id(DocumentAddMethod::AddFull, &empty).unwrap(), "");
    }

    #[test]
    fn resolves_document_portal_paths() {
        let runtime_dir = "/run/user/1000";
        let file_path = "/run/user/1000/doc/abc123/file.txt";

        assert_eq!(
            parse_document_portal_path(file_path, runtime_dir),
            Some(("abc123", ""))
        );
        assert_eq!(
            parse_document_portal_path("/run/user/1000/doc/abc123/mydir/sub/file.txt", runtime_dir),
            Some(("abc123", "/sub/file.txt"))
        );
        assert_eq!(
            parse_document_portal_path("/run/user/1000/doc/abc123", runtime_dir),
            Some(("abc123", ""))
        );
        assert_eq!(
            parse_document_portal_path("/run/user/1000/doc/", runtime_dir),
            None
        );
        assert_eq!(
            parse_document_portal_path("/run/user/1000/documentary/x", runtime_dir),
            None
        );
        assert_eq!(parse_document_portal_path("/etc/passwd", runtime_dir), None);
        assert_eq!(parse_document_portal_path(file_path, ""), None);

        assert_eq!(
            resolve_document_portal_path_with(file_path, ResolveDocumentStrategy::File, runtime_dir, |_| {
                Some(String::from("/home/user/file.txt"))
            }),
            "/home/user/file.txt"
        );

        // The directory name right after the id comes from the host
        // path, so the suffix starts below it.
        assert_eq!(
            resolve_document_portal_path_with(
                "/run/user/1000/doc/abc123/mydir/sub/file.txt",
                ResolveDocumentStrategy::File,
                runtime_dir,
                |_| Some(String::from("/home/user/mydir"))
            ),
            "/home/user/mydir/sub/file.txt"
        );
        assert_eq!(
            resolve_document_portal_path_with(
                "/run/user/1000/doc/abc123/mydir/sub",
                ResolveDocumentStrategy::Directory,
                runtime_dir,
                |_| Some(String::from("/home/user/mydir"))
            ),
            "/home/user/mydir"
        );

        // Unknown documents and unrelated paths stay untouched.
        assert_eq!(
            resolve_document_portal_path_with(file_path, ResolveDocumentStrategy::File, runtime_dir, |_| {
                None
            }),
            file_path
        );
        assert_eq!(
            resolve_document_portal_path_with(
                "/tmp/plain",
                ResolveDocumentStrategy::File,
                runtime_dir,
                |_| Some(String::from("/should/not/be/used"))
            ),
            "/tmp/plain"
        );
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn opens_the_registration_target() {
        let directory = std::env::temp_dir();
        let path = directory.join(format!("codevar-documents-{}.txt", std::process::id()));
        std::fs::write(&path, b"x").unwrap();
        let uri = format!("file://{}", path.display());

        // Without FOR_SAVE the file itself is opened, and the
        // pre-version-2 portal still gets a plain `Add` call.
        let plan = plan_register_document(
            &uri,
            "org.example.App",
            &AppInfo::host("sender"),
            DocumentFlags::NONE,
            1,
        )
        .unwrap();
        let mut message = plan
            .build_add_message(plan.open_register_fd().unwrap())
            .unwrap();
        assert_eq!(message.member().unwrap(), "Add");
        assert_eq!(message.signature(), "hbb");
        assert_eq!(message.fds().len(), 1);
        // The connection assigns the serial when it sends; give the
        // message one here so the wire encoding can be checked.
        message.set_serial(1).unwrap();
        assert!(!message.encode().unwrap().is_empty());
        // Close the descriptor the message still owns.
        for fd in message.take_fds() {
            // SAFETY: the descriptor came out of `open_register_fd`.
            unsafe { libc::close(fd) };
        }

        // FOR_SAVE opens the parent directory, which exists even
        // though the file name is only reserved.
        let plan = plan_register_document(
            &uri,
            "org.example.App",
            &AppInfo::host("sender"),
            DocumentFlags::FOR_SAVE,
            5,
        )
        .unwrap();
        assert_eq!(plan.open_path(), path.parent().unwrap().to_str().unwrap());
        let fd = plan.open_register_fd().unwrap();
        // SAFETY: the descriptor came out of `open_register_fd`.
        unsafe { libc::close(fd) };

        // A missing target reports the C message and a not-found
        // style error.
        let missing = format!(
            "file://{}/codevar-documents-missing-{}.txt",
            directory.display(),
            std::process::id()
        );
        let plan = plan_register_document(
            &missing,
            "org.example.App",
            &AppInfo::host("sender"),
            DocumentFlags::NONE,
            5,
        )
        .unwrap();
        let error = plan.open_register_fd().unwrap_err();
        assert!(error.message().starts_with("Failed to open "));
        assert!(matches!(error, PortalError::NotFound(_)));

        std::fs::remove_file(&path).unwrap();
    }
}
