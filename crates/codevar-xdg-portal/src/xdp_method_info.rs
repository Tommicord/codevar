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

//! Generated metadata for every portal method.
//!
//! The table is produced with `desktop-portal/generate-method-info.py` over the
//! `org.freedesktop.portal.*` XML descriptions plus the host
//! `Registry` interface, keeping the XML document order so entries of
//! one interface stay contiguous.

/// Metadata describing a single portal method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MethodInfo {
    /// Full D-Bus interface name the method belongs to.
    pub interface: &'static str,
    /// Name of the method.
    pub method: &'static str,
    /// Whether the method returns a request object path (an `out`
    /// argument named `handle` or `request_handle` of type `o`).
    pub uses_request: bool,
    /// Zero-based index of the `options` argument (`a{sv}`, `in`)
    /// counting every argument of the method, or `-1` when the method
    /// takes no options.
    pub option_arg: i32,
}

/// Looks up the metadata of `interface.method`, mirroring
/// `xdp_method_info_find`: the scan stops at the end of the matching
/// interface's block, so interfaces must stay contiguous in
/// [`METHOD_INFO`].
#[must_use]
pub fn find(interface: &str, method: &str) -> Option<&'static MethodInfo> {
    let mut interface_found = false;
    for info in METHOD_INFO {
        if info.interface == interface {
            interface_found = true;
            if info.method == method {
                return Some(info);
            }
        } else if interface_found {
            break;
        }
    }
    None
}

/// Returns the whole generated table.
#[must_use]
pub const fn all() -> &'static [MethodInfo] {
    METHOD_INFO
}

/// Returns the number of entries in the generated table.
#[must_use]
pub const fn count() -> usize {
    METHOD_INFO.len()
}

/// The generated method table in XML document order.
pub static METHOD_INFO: &[MethodInfo] = &[
    MethodInfo {
        interface: "org.freedesktop.portal.Account",
        method: "GetUserInformation",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Background",
        method: "RequestBackground",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Background",
        method: "SetStatus",
        uses_request: false,
        option_arg: 0,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Camera",
        method: "AccessCamera",
        uses_request: true,
        option_arg: 0,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Camera",
        method: "OpenPipeWireRemote",
        uses_request: false,
        option_arg: 0,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Clipboard",
        method: "RequestClipboard",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Clipboard",
        method: "SetSelection",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Clipboard",
        method: "SelectionWrite",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Clipboard",
        method: "SelectionWriteDone",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Clipboard",
        method: "SelectionRead",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "GetMountPoint",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "Add",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "AddNamed",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "AddFull",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "AddNamedFull",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "GrantPermissions",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "RevokePermissions",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "Delete",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "Lookup",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "Info",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "List",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Documents",
        method: "GetHostPaths",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.DynamicLauncher",
        method: "Install",
        uses_request: false,
        option_arg: 3,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.DynamicLauncher",
        method: "PrepareInstall",
        uses_request: true,
        option_arg: 3,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.DynamicLauncher",
        method: "RequestInstallToken",
        uses_request: false,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.DynamicLauncher",
        method: "Uninstall",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.DynamicLauncher",
        method: "GetDesktopEntry",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.DynamicLauncher",
        method: "GetIcon",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.DynamicLauncher",
        method: "Launch",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Email",
        method: "ComposeEmail",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.FileChooser",
        method: "OpenFile",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.FileChooser",
        method: "SaveFile",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.FileChooser",
        method: "SaveFiles",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.FileTransfer",
        method: "StartTransfer",
        uses_request: false,
        option_arg: 0,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.FileTransfer",
        method: "AddFiles",
        uses_request: false,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.FileTransfer",
        method: "RetrieveFiles",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.FileTransfer",
        method: "StopTransfer",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GameMode",
        method: "QueryStatus",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GameMode",
        method: "RegisterGame",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GameMode",
        method: "UnregisterGame",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GameMode",
        method: "QueryStatusByPid",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GameMode",
        method: "RegisterGameByPid",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GameMode",
        method: "UnregisterGameByPid",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GameMode",
        method: "QueryStatusByPIDFd",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GameMode",
        method: "RegisterGameByPIDFd",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GameMode",
        method: "UnregisterGameByPIDFd",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GlobalShortcuts",
        method: "CreateSession",
        uses_request: true,
        option_arg: 0,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GlobalShortcuts",
        method: "BindShortcuts",
        uses_request: true,
        option_arg: 3,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GlobalShortcuts",
        method: "ListShortcuts",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.GlobalShortcuts",
        method: "ConfigureShortcuts",
        uses_request: false,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Inhibit",
        method: "Inhibit",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Inhibit",
        method: "CreateMonitor",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Inhibit",
        method: "QueryEndResponse",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.InputCapture",
        method: "CreateSession",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.InputCapture",
        method: "CreateSession2",
        uses_request: false,
        option_arg: 0,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.InputCapture",
        method: "Start",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.InputCapture",
        method: "GetZones",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.InputCapture",
        method: "SetPointerBarriers",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.InputCapture",
        method: "Enable",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.InputCapture",
        method: "Disable",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.InputCapture",
        method: "Release",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.InputCapture",
        method: "ConnectToEIS",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Location",
        method: "CreateSession",
        uses_request: true,
        option_arg: 0,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Location",
        method: "Start",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.NetworkMonitor",
        method: "GetAvailable",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.NetworkMonitor",
        method: "GetMetered",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.NetworkMonitor",
        method: "GetConnectivity",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.NetworkMonitor",
        method: "GetStatus",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.NetworkMonitor",
        method: "CanReach",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Notification",
        method: "AddNotification",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Notification",
        method: "RemoveNotification",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.OpenURI",
        method: "OpenURI",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.OpenURI",
        method: "OpenFile",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.OpenURI",
        method: "OpenDirectory",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.OpenURI",
        method: "SchemeSupported",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Print",
        method: "PreparePrint",
        uses_request: true,
        option_arg: 4,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Print",
        method: "Print",
        uses_request: true,
        option_arg: 3,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.ProxyResolver",
        method: "Lookup",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Realtime",
        method: "MakeThreadRealtimeWithPID",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Realtime",
        method: "MakeThreadHighPriorityWithPID",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "CreateSession",
        uses_request: true,
        option_arg: 0,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "SelectDevices",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "Start",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "NotifyPointerMotion",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "NotifyPointerMotionAbsolute",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "NotifyPointerButton",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "NotifyPointerAxis",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "NotifyPointerAxisDiscrete",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "NotifyKeyboardKeycode",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "NotifyKeyboardKeysym",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "NotifyTouchDown",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "NotifyTouchMotion",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "NotifyTouchUp",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.RemoteDesktop",
        method: "ConnectToEIS",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Request",
        method: "Close",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.ScreenCast",
        method: "CreateSession",
        uses_request: true,
        option_arg: 0,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.ScreenCast",
        method: "SelectSources",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.ScreenCast",
        method: "Start",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.ScreenCast",
        method: "OpenPipeWireRemote",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Screenshot",
        method: "Screenshot",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Screenshot",
        method: "PickColor",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Secret",
        method: "RetrieveSecret",
        uses_request: true,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Session",
        method: "Close",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Settings",
        method: "ReadAll",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Settings",
        method: "Read",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Settings",
        method: "ReadOne",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Trash",
        method: "TrashFile",
        uses_request: false,
        option_arg: -1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Usb",
        method: "CreateSession",
        uses_request: false,
        option_arg: 0,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Usb",
        method: "EnumerateDevices",
        uses_request: false,
        option_arg: 0,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Usb",
        method: "AcquireDevices",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Usb",
        method: "FinishAcquireDevices",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Usb",
        method: "ReleaseDevices",
        uses_request: false,
        option_arg: 1,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Wallpaper",
        method: "SetWallpaperURI",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.portal.Wallpaper",
        method: "SetWallpaperFile",
        uses_request: true,
        option_arg: 2,
    },
    MethodInfo {
        interface: "org.freedesktop.host.portal.Registry",
        method: "Register",
        uses_request: false,
        option_arg: 1,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    // unwrap() is safe here: the tests look up entries that are
    // pinned by the generated table.

    #[test]
    fn table_is_grouped_by_interface() {
        assert_eq!(count(), METHOD_INFO.len());
        assert!(!METHOD_INFO.is_empty());
        let mut seen: alloc::collections::BTreeSet<&str> =
            alloc::collections::BTreeSet::new();
        let mut previous: Option<&str> = None;
        for info in METHOD_INFO {
            if previous != Some(info.interface) {
                assert!(
                    seen.insert(info.interface),
                    "interface {} is not contiguous",
                    info.interface
                );
            }
            previous = Some(info.interface);
        }
        assert!(seen.contains("org.freedesktop.portal.Account"));
        assert!(seen.contains("org.freedesktop.host.portal.Registry"));
    }

    #[test]
    fn finds_request_bearing_and_option_methods() {
        let info = find("org.freedesktop.portal.Account", "GetUserInformation").unwrap();
        assert!(info.uses_request);
        assert_eq!(info.option_arg, 1);

        let info =
            find("org.freedesktop.portal.DynamicLauncher", "PrepareInstall").unwrap();
        assert!(info.uses_request);
        assert_eq!(info.option_arg, 3);

        let info =
            find("org.freedesktop.portal.Notification", "AddNotification").unwrap();
        assert!(!info.uses_request);
        assert_eq!(info.option_arg, -1);

        let info = find("org.freedesktop.host.portal.Registry", "Register").unwrap();
        assert!(!info.uses_request);
        assert_eq!(info.option_arg, 1);
    }

    #[test]
    fn returns_none_for_unknown_pairs() {
        assert!(find("org.freedesktop.portal.Account", "NoSuchMethod").is_none());
        assert!(find("org.freedesktop.portal.NoSuch", "OpenFile").is_none());
        assert!(find("", "").is_none());
    }
}
