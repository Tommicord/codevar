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

//! Portal backend selection, ported from `desktop-portal/xdp-portal-config.c`.
//!
//! Loads the installed `*.portal` files and the `portals.conf(5)`
//! preference files and resolves which backend implements a given
//! `org.freedesktop.impl.portal.*` interface.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use codevar_dbus::{is_valid_bus_name, is_valid_interface_name};

use crate::xdp_error::PortalError;
use crate::xdp_utils::{KeyFile, env_var};

/// Directory name shared by configuration and data files.
const XDP_SUBDIR: &str = "xdg-desktop-portal";

/// Prefix every backend interface of a `.portal` file must carry.
const IMPL_IFACE_PREFIX: &str = "org.freedesktop.impl.portal.";

/// Bus name of the historical GTK fallback backend.
const GTK_FALLBACK_DBUS_NAME: &str = "org.freedesktop.impl.portal.desktop.gtk";

/// Compiled-in `${datadir}` (meson default; distributions override it
/// at build time).
const DATADIR: &str = "/usr/local/share";

/// Compiled-in `${sysconfdir}` (meson default; distributions override
/// it at build time).
const SYSCONFDIR: &str = "/usr/local/etc";

/// Warns at most once about the preferred `portals.conf(5)` mechanism.
static WARNED_PORTALS_CONF: AtomicBool = AtomicBool::new(false);

/// One registered `*.portal` implementation file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortalImpl {
    /// File name of the portal file without the `.portal` suffix.
    pub source: String,
    /// Well-known bus name the implementation owns.
    pub dbus_name: String,
    /// Backend interfaces (`org.freedesktop.impl.portal.*`) the file
    /// registers.
    pub interfaces: Vec<String>,
    /// Desktop environments listed under the deprecated `UseIn` key;
    /// empty when the key is absent.
    pub use_in: Vec<String>,
}

impl PortalImpl {
    /// Returns whether the implementation registers `interface`.
    #[must_use]
    pub fn supports(&self, interface: &str) -> bool {
        self.interfaces.iter().any(|entry| entry == interface)
    }
}

/// One key of a `[preferred]` group: an interface name (or the
/// special key `default`) mapped to an ordered backend list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortalPreference {
    /// Interface name the entry applies to, or `default`.
    pub interface: String,
    /// Ordered backend sources; `none` disables the interface and `*`
    /// selects any implementation.
    pub portals: Vec<String>,
}

/// A loaded `portals.conf` or `{desktop}-portals.conf` file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PreferredConfig {
    /// Interface specific preferences in file order.
    pub interfaces: Vec<PortalPreference>,
    /// The `default` entry; the first one wins when duplicates exist.
    pub default_portal: Option<PortalPreference>,
}

impl PreferredConfig {
    /// Returns the preference list for `interface`.
    #[must_use]
    pub fn preference(&self, interface: &str) -> Option<&PortalPreference> {
        self.interfaces
            .iter()
            .find(|entry| entry.interface == interface)
    }

    /// Returns whether the configuration asks for `interface` (or the
    /// default entry) to be disabled with `none`.
    #[must_use]
    pub fn prefers_none(&self, interface: &str) -> bool {
        match self.preference(interface) {
            Some(preference) => preference.portals.iter().any(|portal| portal == "none"),
            None => self.default_portal.as_ref().is_some_and(|default| {
                default.portals.iter().any(|portal| portal == "none")
            }),
        }
    }
}

/// Loaded portal configuration: registered implementations plus the
/// ordered list of preference files.
#[derive(Debug, Clone, Default)]
pub struct PortalConfig {
    current_desktops: Vec<String>,
    impls: Vec<PortalImpl>,
    configs: Vec<PreferredConfig>,
}

impl PortalConfig {
    /// Builds a configuration from explicit inputs, sorting the
    /// implementations like the loader does. Intended for tests and
    /// embedders that provide their own file contents.
    #[must_use]
    pub fn from_parts(
        current_desktops: Vec<String>,
        impls: Vec<PortalImpl>,
        configs: Vec<PreferredConfig>,
    ) -> Self {
        let mut impls = impls;
        sort_impls(&mut impls, &current_desktops);
        Self {
            current_desktops,
            impls,
            configs,
        }
    }

    /// Loads the configuration from the environment and the file
    /// system, following the directory search of the C reference.
    ///
    /// When `$XDG_DESKTOP_PORTAL_DIR` is set only that directory is
    /// consulted, both for `.portal` files and preference files. On
    /// non-unix targets an empty configuration is returned.
    #[must_use]
    pub fn load() -> Self {
        #[cfg(all(unix, not(target_arch = "wasm32")))]
        {
            let current_desktops = current_lowercase_desktops();
            if let Some(dir) = env_nonempty("XDG_DESKTOP_PORTAL_DIR") {
                let mut map = BTreeMap::new();
                load_portals_dir(&mut map, &dir);
                let mut impls: Vec<PortalImpl> = map.into_values().collect();
                sort_impls(&mut impls, &current_desktops);
                let mut configs = Vec::new();
                if let Some(config) = load_config_directory(&dir, &current_desktops) {
                    configs.push(config);
                }
                return Self {
                    current_desktops,
                    impls,
                    configs,
                };
            }

            let mut map = BTreeMap::new();
            for base in data_home_dirs() {
                load_portals_dir(&mut map, &format!("{base}/{XDP_SUBDIR}/portals"));
            }
            for base in data_dirs() {
                load_portals_dir(&mut map, &format!("{base}/{XDP_SUBDIR}/portals"));
            }
            load_portals_dir(&mut map, &format!("{DATADIR}/{XDP_SUBDIR}/portals"));
            let mut impls: Vec<PortalImpl> = map.into_values().collect();
            sort_impls(&mut impls, &current_desktops);

            let mut configs = Vec::new();
            for base in config_search_dirs() {
                if let Some(config) = load_config_directory(&base, &current_desktops) {
                    configs.push(config);
                }
            }
            Self {
                current_desktops,
                impls,
                configs,
            }
        }
        #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
        {
            Self::default()
        }
    }

    /// Returns the lowercased, validated `$XDG_CURRENT_DESKTOP`
    /// entries in order.
    #[must_use]
    pub fn current_desktops(&self) -> &[String] {
        &self.current_desktops
    }

    /// Returns all registered implementations in resolution order.
    #[must_use]
    pub fn impls(&self) -> &[PortalImpl] {
        &self.impls
    }

    /// Returns every loaded preference file in search order.
    #[must_use]
    pub fn configs(&self) -> &[PreferredConfig] {
        &self.configs
    }

    /// Resolves the backend for `interface` exactly like
    /// `xdp_portal_config_find`: preference files first, then the
    /// deprecated `UseIn` key, then the GTK fallback.
    #[must_use]
    pub fn find(&self, interface: &str) -> Option<&PortalImpl> {
        for config in &self.configs {
            if config.prefers_none(interface) {
                log::debug!("Found 'none' in configuration for {interface}");
                return None;
            }
            if let Some(impl_config) =
                self.impl_for_preference(config.preference(interface), interface)
            {
                log::debug!(
                    "Using {} for {interface} (interface specific config)",
                    impl_config.source
                );
                return Some(impl_config);
            }
            if let Some(impl_config) =
                self.impl_for_preference(config.default_portal.as_ref(), interface)
            {
                log::debug!(
                    "Using {} for {interface} (default config)",
                    impl_config.source
                );
                return Some(impl_config);
            }
        }

        for desktop in &self.current_desktops {
            for impl_config in &self.impls {
                if impl_config.supports(interface)
                    && contains_ci(&impl_config.use_in, desktop)
                {
                    warn_use_in(&impl_config.source, interface, desktop);
                    return Some(impl_config);
                }
            }
        }

        self.gtk_fallback(interface)
    }

    /// Returns every backend that should serve `interface`, mirroring
    /// `xdp_portal_config_find_all`.
    #[must_use]
    pub fn find_all(&self, interface: &str) -> Vec<&PortalImpl> {
        let mut out: Vec<&PortalImpl> = Vec::new();
        for config in &self.configs {
            if config.prefers_none(interface) {
                return out;
            }
            self.collect_impls(config.preference(interface), interface, &mut out);
            if !out.is_empty() {
                break;
            }
            self.collect_impls(config.default_portal.as_ref(), interface, &mut out);
            if !out.is_empty() {
                break;
            }
        }
        if !out.is_empty() {
            return out;
        }

        for desktop in &self.current_desktops {
            for impl_config in &self.impls {
                if impl_config.supports(interface)
                    && contains_ci(&impl_config.use_in, desktop)
                {
                    warn_use_in(&impl_config.source, interface, desktop);
                    out.push(impl_config);
                }
            }
        }
        if !out.is_empty() {
            return out;
        }

        if let Some(impl_config) = self.gtk_fallback(interface) {
            out.push(impl_config);
        }
        out
    }

    fn impl_for_preference(
        &self,
        preference: Option<&PortalPreference>,
        interface: &str,
    ) -> Option<&PortalImpl> {
        let preference = preference?;
        for portal in &preference.portals {
            log::debug!("Found '{portal}' in configuration for {interface}");
            if portal == "*" {
                return self
                    .impls
                    .iter()
                    .find(|candidate| candidate.supports(interface));
            }
            let impl_config = self
                .impls
                .iter()
                .find(|candidate| candidate.source == *portal);
            let Some(impl_config) = impl_config else {
                log::info!("Requested backend {portal} does not exist. Skipping...");
                continue;
            };
            if !impl_config.supports(interface) {
                log::info!(
                    "Requested backend {}.portal does not support {interface}. Skipping...",
                    impl_config.source
                );
                continue;
            }
            return Some(impl_config);
        }
        None
    }

    fn collect_impls<'a>(
        &'a self,
        preference: Option<&PortalPreference>,
        interface: &str,
        out: &mut Vec<&'a PortalImpl>,
    ) {
        let Some(preference) = preference else {
            return;
        };
        let portals = preference.portals.join(";");
        log::debug!(
            "Found '{portals}' in configuration for {}",
            preference.interface
        );
        for portal in &preference.portals {
            for candidate in &self.impls {
                if candidate.source != *portal && portal != "*" {
                    continue;
                }
                if out.iter().any(|entry| entry.source == candidate.source) {
                    log::info!(
                        "Duplicate backend {}.portal. Skipping...",
                        candidate.source
                    );
                    continue;
                }
                if !candidate.supports(interface) {
                    log::info!(
                        "Requested backend {}.portal does not support {interface}. Skipping...",
                        candidate.source
                    );
                    continue;
                }
                log::debug!("Using {}.portal for {interface} (config)", candidate.source);
                out.push(candidate);
            }
        }
    }

    fn gtk_fallback(&self, interface: &str) -> Option<&PortalImpl> {
        let impl_config = self.impls.iter().find(|candidate| {
            candidate.dbus_name == GTK_FALLBACK_DBUS_NAME && candidate.supports(interface)
        })?;
        log::warn!(
            "Choosing {} for {interface} as a last-resort fallback",
            impl_config.source
        );
        Some(impl_config)
    }
}

/// Warns about the deprecated `UseIn` fallback, repeating the
/// selection but only hinting at `portals.conf(5)` once per process.
fn warn_use_in(source: &str, interface: &str, desktop: &str) {
    log::warn!("Choosing {source} for {interface} via the deprecated UseIn key");
    if !WARNED_PORTALS_CONF.swap(true, Ordering::Relaxed) {
        log::warn!(
            "The preferred method to match portal implementations to desktop \
             environments is to use the portals.conf(5) configuration file"
        );
    }
    log::debug!("Using {source} for {interface} in {desktop} (fallback)");
}

/// Parses the contents of one `.portal` file into a [`PortalImpl`].
///
/// `source` is the file name without the `.portal` suffix.
///
/// # Errors
///
/// Returns [`PortalError::InvalidArgument`] when the `[portal]` group
/// is missing, `DBusName` is absent or not a valid bus name, or an
/// `Interfaces` entry is not a valid
/// `org.freedesktop.impl.portal.*` interface name.
pub fn parse_portal_file(
    source: &str,
    contents: &str,
) -> Result<PortalImpl, PortalError> {
    let key_file = KeyFile::parse(contents)?;
    let dbus_name = key_file
        .get("portal", "DBusName")
        .ok_or_else(|| PortalError::InvalidArgument(String::from("missing DBusName")))?;
    if !is_valid_bus_name(&dbus_name) {
        return Err(PortalError::InvalidArgument(format!(
            "Not a valid bus name: {dbus_name}"
        )));
    }
    let interfaces = key_file.list("portal", "Interfaces").ok_or_else(|| {
        PortalError::InvalidArgument(String::from("missing Interfaces"))
    })?;
    for interface in &interfaces {
        if !is_valid_interface_name(interface) {
            return Err(PortalError::InvalidArgument(format!(
                "Not a valid interface name: {interface}"
            )));
        }
        if !interface.starts_with(IMPL_IFACE_PREFIX) {
            return Err(PortalError::InvalidArgument(format!(
                "Not a portal backend interface: {interface}"
            )));
        }
        log::debug!("portal implementation supports {interface}");
    }
    let use_in = key_file.list("portal", "UseIn").unwrap_or_default();
    Ok(PortalImpl {
        source: String::from(source),
        dbus_name,
        interfaces,
        use_in,
    })
}

/// Parses the contents of a `portals.conf`-style file.
///
/// Returns `Ok(None)` when the file holds no `[preferred]` group.
///
/// # Errors
///
/// Returns [`PortalError::InvalidArgument`] when the file is not a
/// valid key file or a preference key has no value list.
pub fn parse_preferred_config(
    contents: &str,
) -> Result<Option<PreferredConfig>, PortalError> {
    let key_file = KeyFile::parse(contents)?;
    let Some(entries) = key_file.entries("preferred") else {
        return Ok(None);
    };
    let mut config = PreferredConfig::default();
    for (key, _) in entries {
        let portals = key_file.list("preferred", key).ok_or_else(|| {
            PortalError::InvalidArgument(format!("Invalid portals for interface '{key}'"))
        })?;
        let preference = PortalPreference {
            interface: key.clone(),
            portals,
        };
        if preference.interface == "default" {
            if config.default_portal.is_none() {
                config.default_portal = Some(preference);
            } else {
                log::warn!("Duplicate default key will get ignored");
            }
        } else {
            config.interfaces.push(preference);
        }
    }
    Ok(Some(config))
}

/// Returns whether `needle` equals any entry of `haystack`
/// case-insensitively, like `g_strv_case_contains`.
fn contains_ci(haystack: &[String], needle: &str) -> bool {
    haystack
        .iter()
        .any(|entry| entry.eq_ignore_ascii_case(needle))
}

/// Sorts implementations like `sort_impl_by_use_in_and_name`: the
/// first current desktop with a `UseIn` match wins, ties fall back to
/// the source name.
fn sort_impls(impls: &mut [PortalImpl], desktops: &[String]) {
    impls.sort_by(|left, right| {
        for desktop in desktops {
            let use_left = contains_ci(&left.use_in, desktop);
            let use_right = contains_ci(&right.use_in, desktop);
            if use_left != use_right {
                return if use_left {
                    core::cmp::Ordering::Less
                } else {
                    core::cmp::Ordering::Greater
                };
            }
            if use_left {
                break;
            }
        }
        left.source.cmp(&right.source)
    });
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
/// Validates one `$XDG_CURRENT_DESKTOP` element (alphanumeric plus
/// `-` and `_`, non-empty), like `validate_xdg_desktop`.
fn validate_xdg_desktop(desktop: &str) -> bool {
    !desktop.is_empty()
        && desktop
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
/// Reads `$XDG_CURRENT_DESKTOP`, keeps valid entries and lowercases
/// them like `get_current_lowercase_desktops`.
fn current_lowercase_desktops() -> Vec<String> {
    let value = env_nonempty("XDG_CURRENT_DESKTOP").unwrap_or_default();
    value
        .split(':')
        .filter(|entry| validate_xdg_desktop(entry))
        .map(|entry| entry.to_ascii_lowercase())
        .collect()
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
/// Reads a non-empty environment variable.
fn env_nonempty(name: &str) -> Option<String> {
    env_var(name).filter(|value| !value.is_empty())
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
/// Returns the primary user configuration directory
/// (`$XDG_CONFIG_HOME` or `$HOME/.config`).
fn user_config_home() -> Option<String> {
    if let Some(dir) = env_nonempty("XDG_CONFIG_HOME") {
        return Some(dir);
    }
    let home = env_nonempty("HOME")?;
    Some(format!("{home}/.config"))
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
/// Returns the primary user data directory
/// (`$XDG_DATA_HOME` or `$HOME/.local/share`).
fn user_data_home() -> Option<String> {
    if let Some(dir) = env_nonempty("XDG_DATA_HOME") {
        return Some(dir);
    }
    let home = env_nonempty("HOME")?;
    Some(format!("{home}/.local/share"))
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
/// Returns the user data directories used to find `.portal` files.
fn data_home_dirs() -> Vec<String> {
    user_data_home().into_iter().collect()
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
/// Returns the system data directories (`$XDG_DATA_DIRS` or the
/// specification default).
fn data_dirs() -> Vec<String> {
    match env_nonempty("XDG_DATA_DIRS") {
        Some(value) => value
            .split(':')
            .filter(|e| !e.is_empty())
            .map(String::from)
            .collect(),
        None => Vec::from([String::from("/usr/local/share"), String::from("/usr/share")]),
    }
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
/// Returns the system configuration directories (`$XDG_CONFIG_DIRS`
/// or the specification default).
fn config_dirs() -> Vec<String> {
    match env_nonempty("XDG_CONFIG_DIRS") {
        Some(value) => value
            .split(':')
            .filter(|e| !e.is_empty())
            .map(String::from)
            .collect(),
        None => Vec::from([String::from("/etc/xdg")]),
    }
}

/// Returns every directory searched for preference files, in the
/// order used by `load_portal_configurations`.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn config_search_dirs() -> Vec<String> {
    let mut dirs = Vec::new();
    if let Some(dir) = user_config_home() {
        dirs.push(format!("{dir}/{XDP_SUBDIR}"));
    }
    for dir in config_dirs() {
        dirs.push(format!("{dir}/{XDP_SUBDIR}"));
    }
    dirs.push(format!("{SYSCONFDIR}/{XDP_SUBDIR}"));
    if let Some(dir) = user_data_home() {
        dirs.push(format!("{dir}/{XDP_SUBDIR}"));
    }
    for dir in data_dirs() {
        dirs.push(format!("{dir}/{XDP_SUBDIR}"));
    }
    dirs.push(format!("{DATADIR}/{XDP_SUBDIR}"));
    dirs
}

/// Reads a whole file into a string.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when the file cannot be opened,
/// read or is not valid UTF-8.
#[cfg(all(unix, not(target_arch = "wasm32")))]
pub fn read_file(path: &str) -> Result<String, PortalError> {
    use alloc::ffi::CString;

    let c_path = CString::new(path).map_err(|_| {
        PortalError::InvalidArgument(String::from("path contains a nul byte"))
    })?;
    // SAFETY: `c_path` is a valid NUL-terminated path; the returned
    // descriptor is checked against -1 before use.
    let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        // SAFETY: `fd` is negative, so `errno` describes the failure.
        let errno = unsafe { *libc::__errno_location() };
        return Err(PortalError::Failed(format!(
            "cannot open {path}: errno {errno}"
        )));
    }
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        // SAFETY: `buffer` is writable for its full length and `fd`
        // refers to an open file.
        let count = unsafe { libc::read(fd, buffer.as_mut_ptr().cast(), buffer.len()) };
        if count < 0 {
            // SAFETY: a negative return value means `errno` is set.
            let errno = unsafe { *libc::__errno_location() };
            // SAFETY: the descriptor is open and will be closed.
            unsafe { libc::close(fd) };
            if errno == libc::EINTR {
                continue;
            }
            return Err(PortalError::Failed(format!(
                "cannot read {path}: errno {errno}"
            )));
        }
        if count == 0 {
            break;
        }
        let Ok(count) = usize::try_from(count) else {
            // SAFETY: the descriptor is open and will be closed.
            unsafe { libc::close(fd) };
            return Err(PortalError::Failed(format!("cannot read {path}")));
        };
        if bytes.len() + count > 16 * 1024 * 1024 {
            // SAFETY: the descriptor is open and will be closed.
            unsafe { libc::close(fd) };
            return Err(PortalError::Failed(format!("{path} exceeds 16 MiB")));
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    // SAFETY: the descriptor is open and will be closed.
    unsafe { libc::close(fd) };
    String::from_utf8(bytes)
        .map_err(|_| PortalError::InvalidArgument(format!("{path} is not valid UTF-8")))
}

/// Lists the names of the directory entries in `path`, skipping `.`
/// and `..`. Returns an empty vector when the directory cannot be
/// opened.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn list_dir(path: &str) -> Vec<String> {
    use alloc::ffi::CString;
    use core::ffi::CStr;

    let Ok(c_path) = CString::new(path) else {
        return Vec::new();
    };
    // SAFETY: `c_path` is a valid NUL-terminated directory path; a
    // null result is handled below.
    let dir = unsafe { libc::opendir(c_path.as_ptr()) };
    if dir.is_null() {
        return Vec::new();
    }
    let mut names = Vec::new();
    loop {
        // SAFETY: `dir` is an open, valid stream; the result is
        // either null (end of stream) or valid until the next
        // `readdir`, and the reference cannot outlive this loop body.
        let entry = unsafe { libc::readdir(dir).as_ref() };
        let Some(entry) = entry else {
            break;
        };
        // SAFETY: `d_name` is a NUL-terminated byte array inside the
        // entry, which is alive while we borrow it.
        let name = unsafe { CStr::from_ptr(entry.d_name.as_ptr()) };
        let Ok(name) = name.to_str() else {
            continue;
        };
        if name == "." || name == ".." {
            continue;
        }
        names.push(String::from(name));
    }
    // SAFETY: `dir` was opened above and is not used afterward.
    unsafe { libc::closedir(dir) };
    names
}

/// Loads every `*.portal` file of `dir` into `portals`, keeping the
/// first file registered per source name.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn load_portals_dir(portals: &mut BTreeMap<String, PortalImpl>, dir: &str) {
    log::debug!("load portals from {dir}");
    for name in list_dir(dir) {
        let Some(source) = name.strip_suffix(".portal") else {
            continue;
        };
        if portals.contains_key(source) {
            log::debug!("Skipping duplicate source {source}");
            continue;
        }
        let path = format!("{dir}/{name}");
        match read_file(&path).and_then(|contents| parse_portal_file(source, &contents)) {
            Ok(impl_config) => {
                portals.insert(impl_config.source.clone(), impl_config);
            }
            Err(error) => log::warn!("Error loading {path}: {error}"),
        }
    }
}

/// Tries to load one preference file; returns `None` when the file is
/// absent or unusable.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn load_portal_configuration_for_dir(
    base_directory: &str,
    portal_file: &str,
) -> Option<PreferredConfig> {
    let path = format!("{base_directory}/{portal_file}");
    log::debug!("Looking for portals configuration in '{path}'");
    let contents = read_file(&path).ok()?;
    parse_preferred_config(&contents).unwrap_or_else(|error| {
        log::warn!("Error loading {path}: {error}");
        None
    })
}

/// Loads `{desktop}-portals.conf` for every current desktop, then
/// `portals.conf`, from `dir`.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn load_config_directory(dir: &str, desktops: &[String]) -> Option<PreferredConfig> {
    for desktop in desktops {
        let file = format!("{desktop}-portals.conf");
        if let Some(config) = load_portal_configuration_for_dir(dir, &file) {
            log::debug!(
                "Using portal configuration file '{dir}/{file}' for desktop '{desktop}'"
            );
            return Some(config);
        }
    }
    let config = load_portal_configuration_for_dir(dir, "portals.conf")?;
    log::debug!(
        "Using portal configuration file '{dir}/portals.conf' for non-specific desktop"
    );
    Some(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    // unwrap() calls in tests are safe: they operate on inputs
    // constructed inside the test with known shape.

    const IFACE: &str = "org.freedesktop.impl.portal.FileChooser";

    fn impl_with(source: &str, interfaces: &[&str], use_in: &[&str]) -> PortalImpl {
        PortalImpl {
            source: String::from(source),
            dbus_name: format!("org.freedesktop.impl.portal.desktop.{source}"),
            interfaces: interfaces.iter().map(|i| String::from(*i)).collect(),
            use_in: use_in.iter().map(|i| String::from(*i)).collect(),
        }
    }

    fn preference(interface: &str, portals: &[&str]) -> PortalPreference {
        PortalPreference {
            interface: String::from(interface),
            portals: portals.iter().map(|p| String::from(*p)).collect(),
        }
    }

    #[test]
    fn parses_portal_files() {
        let impl_config = parse_portal_file(
            "gtk",
            "[portal]\nDBusName=org.freedesktop.impl.portal.desktop.gtk\n\
             Interfaces=org.freedesktop.impl.portal.FileChooser;org.freedesktop.impl.portal.Settings;\n\
             UseIn=GNOME;X-Cinnamon;\n",
        )
        .unwrap();
        assert_eq!(impl_config.source, "gtk");
        assert_eq!(impl_config.interfaces.len(), 2);
        assert_eq!(
            impl_config.use_in,
            Vec::from(["GNOME", "X-Cinnamon"].map(String::from))
        );
        assert!(impl_config.supports(IFACE));
        assert!(!impl_config.supports("org.freedesktop.impl.portal.Email"));
    }

    #[test]
    fn rejects_invalid_portal_files() {
        let missing = parse_portal_file("x", "[other]\nDBusName=a.b\n").unwrap_err();
        assert!(matches!(missing, PortalError::InvalidArgument(_)));

        let bad_name = parse_portal_file(
            "x",
            "[portal]\nDBusName=Not A Name\nInterfaces=org.freedesktop.impl.portal.A;\n",
        )
        .unwrap_err();
        assert_eq!(bad_name.message(), "Not a valid bus name: Not A Name");

        let bad_iface = parse_portal_file(
            "x",
            "[portal]\nDBusName=a.b\nInterfaces=org.freedesktop.NotAPortal;\n",
        )
        .unwrap_err();
        assert_eq!(
            bad_iface.message(),
            "Not a portal backend interface: org.freedesktop.NotAPortal"
        );

        let missing_use_in = parse_portal_file(
            "x",
            "[portal]\nDBusName=a.b\nInterfaces=org.freedesktop.impl.portal.A;\n",
        )
        .unwrap();
        assert!(missing_use_in.use_in.is_empty());
    }

    #[test]
    fn parses_preferred_configs() {
        let config = parse_preferred_config(
            "[preferred]\ndefault=gtk;wlr;\norg.freedesktop.impl.portal.FileChooser=gnome;\norg.freedesktop.impl.portal.Email=none;\n",
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            config.default_portal.as_ref().unwrap().portals,
            Vec::from([String::from("gtk"), String::from("wlr")])
        );
        assert_eq!(config.interfaces.len(), 2);
        assert_eq!(
            config.preference(IFACE).unwrap().portals,
            Vec::from([String::from("gnome")])
        );
        assert!(config.prefers_none("org.freedesktop.impl.portal.Email"));
        assert!(!config.prefers_none(IFACE));
        assert!(!config.prefers_none("org.freedesktop.impl.portal.Camera"));

        // The second occurrence of the key is ignored (with a
        // warning), but reads return the last duplicate value: the C
        // code calls g_key_file_get_string_list for every key
        // returned by g_key_file_get_keys, which resolves "default" to
        // its final entry.
        let duplicate_default =
            parse_preferred_config("[preferred]\ndefault=first;\ndefault=second;\n")
                .unwrap()
                .unwrap();
        assert_eq!(
            duplicate_default.default_portal.unwrap().portals,
            Vec::from([String::from("second")])
        );

        let no_group = parse_preferred_config("[other]\ndefault=gtk;\n").unwrap();
        assert!(no_group.is_none());
    }

    #[test]
    fn sorts_impls_by_use_in_then_source() {
        let mut impls = Vec::from([
            impl_with("aaa", &[IFACE], &[]),
            impl_with("zzz", &[IFACE], &["GNOME"]),
            impl_with("mmm", &[IFACE], &["gnome"]),
        ]);
        sort_impls(&mut impls, &[String::from("gnome")]);
        assert_eq!(impls[0].source, "mmm");
        assert_eq!(impls[1].source, "zzz");
        assert_eq!(impls[2].source, "aaa");

        let mut impls = Vec::from([impl_with("b", &[], &[]), impl_with("a", &[], &[])]);
        sort_impls(&mut impls, &[String::from("gnome")]);
        assert_eq!(impls[0].source, "a");
    }

    #[test]
    fn find_follows_the_config_chain() {
        let impls = Vec::from([
            impl_with(
                "gtk",
                &[IFACE, "org.freedesktop.impl.portal.Settings"],
                &["GNOME"],
            ),
            impl_with(
                "wlr",
                &["org.freedesktop.impl.portal.Screenshot"],
                &["wlroots"],
            ),
            impl_with("other", &[IFACE], &[]),
        ]);

        let config = PortalConfig::from_parts(
            Vec::from([String::from("gnome")]),
            impls.clone(),
            Vec::from([PreferredConfig {
                interfaces: Vec::from([
                    preference(IFACE, &["other"]),
                    preference("org.freedesktop.impl.portal.Screenshot", &["wlr"]),
                ]),
                default_portal: Some(preference("default", &["gtk"])),
            }]),
        );
        assert_eq!(config.find(IFACE).unwrap().source, "other");
        assert_eq!(
            config
                .find("org.freedesktop.impl.portal.Screenshot")
                .unwrap()
                .source,
            "wlr"
        );
        assert_eq!(
            config
                .find("org.freedesktop.impl.portal.Settings")
                .unwrap()
                .source,
            "gtk"
        );

        let none_config = PortalConfig::from_parts(
            Vec::from([String::from("gnome")]),
            impls.clone(),
            Vec::from([PreferredConfig {
                interfaces: Vec::from([preference(IFACE, &["none"])]),
                default_portal: None,
            }]),
        );
        assert!(none_config.find(IFACE).is_none());

        let wildcard = PortalConfig::from_parts(
            Vec::from([String::from("gnome")]),
            impls.clone(),
            Vec::from([PreferredConfig {
                interfaces: Vec::from([preference(IFACE, &["missing", "*"])]),
                default_portal: None,
            }]),
        );
        assert_eq!(wildcard.find(IFACE).unwrap().source, "gtk");

        let no_config =
            PortalConfig::from_parts(Vec::from([String::from("gnome")]), impls, vec![]);
        assert_eq!(no_config.find(IFACE).unwrap().source, "gtk");
    }

    #[test]
    fn find_falls_back_to_use_in_and_gtk() {
        let impls = Vec::from([
            impl_with("kde", &[IFACE], &["KDE"]),
            impl_with("gtk", &[IFACE], &[]),
        ]);
        let config =
            PortalConfig::from_parts(Vec::from([String::from("kde")]), impls, vec![]);
        assert_eq!(config.find(IFACE).unwrap().source, "kde");

        let impls = Vec::from([
            impl_with("wlr", &["org.freedesktop.impl.portal.Screenshot"], &[]),
            impl_with("gtk", &[IFACE], &[]),
        ]);
        let config =
            PortalConfig::from_parts(Vec::from([String::from("gnome")]), impls, vec![]);
        assert_eq!(config.find(IFACE).unwrap().source, "gtk");
        assert!(config.find("org.freedesktop.impl.portal.Email").is_none());
    }

    #[test]
    fn find_all_collects_backends_and_deduplicates() {
        let impls = Vec::from([
            impl_with("gtk", &[IFACE], &["GNOME"]),
            impl_with("wlr", &[IFACE], &[]),
            impl_with("third", &[IFACE], &[]),
        ]);
        let config = PortalConfig::from_parts(
            Vec::from([String::from("gnome")]),
            impls.clone(),
            Vec::from([PreferredConfig {
                interfaces: Vec::from([preference(IFACE, &["wlr", "*", "wlr"])]),
                default_portal: None,
            }]),
        );
        let all = config.find_all(IFACE);
        let sources: Vec<&str> = all.iter().map(|entry| entry.source.as_str()).collect();
        assert_eq!(sources, Vec::from(["wlr", "gtk", "third"]));

        let config = PortalConfig::from_parts(
            Vec::from([String::from("gnome")]),
            impls,
            Vec::from([PreferredConfig {
                interfaces: Vec::from([preference(IFACE, &["none"])]),
                default_portal: None,
            }]),
        );
        assert!(config.find_all(IFACE).is_empty());
    }

    #[test]
    fn validates_desktop_elements() {
        assert!(validate_xdg_desktop("GNOME"));
        assert!(validate_xdg_desktop("X-Cinnamon"));
        assert!(validate_xdg_desktop("ubuntu"));
        assert!(!validate_xdg_desktop(""));
        assert!(!validate_xdg_desktop("has space"));
        assert!(!validate_xdg_desktop("has/slash"));
    }
}
