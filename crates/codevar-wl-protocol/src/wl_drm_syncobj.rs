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

//! `linux-drm-syncobj` explicit synchronization for surfaces.
//!
//! The tables mirror
//! `staging/linux-drm-syncobj/linux-drm-syncobj-v1.xml` of
//! `wayland-protocols` message by message: signatures, argument
//! interfaces, enum error codes and the opcode order a real compositor
//! expects on the wire. The protocol is a staging protocol, so the three
//! interfaces are version 1 and carry no events.
//!
//! The handshake a client performs is: bind
//! `wp_linux_drm_syncobj_manager_v1`, `import_timeline` with a DRM
//! syncobj timeline file descriptor, `get_surface` on the
//! [`wl_surface`](crate::SURFACE_INTERFACE) to synchronize, then
//! `set_acquire_point` and `set_release_point` on the resulting
//! `wp_linux_drm_syncobj_surface_v1` before every `wl_surface.commit`
//! that attaches a buffer.

use core::fmt;

use crate::wl_core::SURFACE_INTERFACE;
use crate::wl_handle::{WlInterface, WlMessage};

/// `wp_linux_drm_syncobj_manager_v1.destroy` request opcode.
pub const DRM_SYNCOBJ_MANAGER_DESTROY: u32 = 0;
/// `wp_linux_drm_syncobj_manager_v1.get_surface` request opcode.
pub const DRM_SYNCOBJ_MANAGER_GET_SURFACE: u32 = 1;
/// `wp_linux_drm_syncobj_manager_v1.import_timeline` request opcode.
pub const DRM_SYNCOBJ_MANAGER_IMPORT_TIMELINE: u32 = 2;

/// `wp_linux_drm_syncobj_timeline_v1.destroy` request opcode.
pub const DRM_SYNCOBJ_TIMELINE_DESTROY: u32 = 0;

/// `wp_linux_drm_syncobj_surface_v1.destroy` request opcode.
pub const DRM_SYNCOBJ_SURFACE_DESTROY: u32 = 0;
/// `wp_linux_drm_syncobj_surface_v1.set_acquire_point` request opcode.
pub const DRM_SYNCOBJ_SURFACE_SET_ACQUIRE_POINT: u32 = 1;
/// `wp_linux_drm_syncobj_surface_v1.set_release_point` request opcode.
pub const DRM_SYNCOBJ_SURFACE_SET_RELEASE_POINT: u32 = 2;

/// `wp_linux_drm_syncobj_manager_v1` explicit synchronization factory.
pub static DRM_SYNCOBJ_MANAGER_INTERFACE: WlInterface = WlInterface {
    name: "wp_linux_drm_syncobj_manager_v1",
    version: 1,
    requests: &[
        WlMessage::new("destroy", "", &[]),
        WlMessage::new(
            "get_surface",
            "no",
            &[Some(&DRM_SYNCOBJ_SURFACE_INTERFACE), Some(&SURFACE_INTERFACE)],
        ),
        WlMessage::new(
            "import_timeline",
            "nh",
            &[Some(&DRM_SYNCOBJ_TIMELINE_INTERFACE), None],
        ),
    ],
    events: &[],
};

/// `wp_linux_drm_syncobj_timeline_v1` imported syncobj timeline.
pub static DRM_SYNCOBJ_TIMELINE_INTERFACE: WlInterface = WlInterface {
    name: "wp_linux_drm_syncobj_timeline_v1",
    version: 1,
    requests: &[WlMessage::new("destroy", "", &[])],
    events: &[],
};

/// `wp_linux_drm_syncobj_surface_v1` per-surface synchronization object.
pub static DRM_SYNCOBJ_SURFACE_INTERFACE: WlInterface = WlInterface {
    name: "wp_linux_drm_syncobj_surface_v1",
    version: 1,
    requests: &[
        WlMessage::new("destroy", "", &[]),
        WlMessage::new(
            "set_acquire_point",
            "ouu",
            &[Some(&DRM_SYNCOBJ_TIMELINE_INTERFACE), None, None],
        ),
        WlMessage::new(
            "set_release_point",
            "ouu",
            &[Some(&DRM_SYNCOBJ_TIMELINE_INTERFACE), None, None],
        ),
    ],
    events: &[],
};

/// Protocol error codes of `enum wp_linux_drm_syncobj_manager_v1.error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DrmSyncobjManagerError {
    /// The `wl_surface` already has a synchronization object associated.
    SurfaceExists = 0,
    /// The timeline file descriptor could not be imported.
    InvalidTimeline = 1,
}

impl DrmSyncobjManagerError {
    /// Returns the wire code of the error.
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a wire code, returning `None` for unknown codes.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::SurfaceExists),
            1 => Some(Self::InvalidTimeline),
            _ => None,
        }
    }

    /// Returns the symbolic name of the error.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SurfaceExists => "surface_exists",
            Self::InvalidTimeline => "invalid_timeline",
        }
    }
}

impl fmt::Display for DrmSyncobjManagerError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl core::error::Error for DrmSyncobjManagerError {}

/// Protocol error codes of `enum wp_linux_drm_syncobj_surface_v1.error`.
///
/// The enum has no entry at code 0, so [`Self::from_code`] returns
/// `None` for that value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DrmSyncobjSurfaceError {
    /// The associated `wl_surface` was destroyed.
    NoSurface = 1,
    /// The attached buffer does not support explicit synchronization.
    UnsupportedBuffer = 2,
    /// No buffer was attached at commit time.
    NoBuffer = 3,
    /// No acquire timeline point was set at commit time.
    NoAcquirePoint = 4,
    /// No release timeline point was set at commit time.
    NoReleasePoint = 5,
    /// The acquire and release points of one timeline are in conflict.
    ConflictingPoints = 6,
}

impl DrmSyncobjSurfaceError {
    /// Returns the wire code of the error.
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a wire code, returning `None` for unknown codes.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::NoSurface),
            2 => Some(Self::UnsupportedBuffer),
            3 => Some(Self::NoBuffer),
            4 => Some(Self::NoAcquirePoint),
            5 => Some(Self::NoReleasePoint),
            6 => Some(Self::ConflictingPoints),
            _ => None,
        }
    }

    /// Returns the symbolic name of the error.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NoSurface => "no_surface",
            Self::UnsupportedBuffer => "unsupported_buffer",
            Self::NoBuffer => "no_buffer",
            Self::NoAcquirePoint => "no_acquire_point",
            Self::NoReleasePoint => "no_release_point",
            Self::ConflictingPoints => "conflicting_points",
        }
    }
}

impl fmt::Display for DrmSyncobjSurfaceError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl core::error::Error for DrmSyncobjSurfaceError {}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[&WlInterface] = &[
        &DRM_SYNCOBJ_MANAGER_INTERFACE,
        &DRM_SYNCOBJ_TIMELINE_INTERFACE,
        &DRM_SYNCOBJ_SURFACE_INTERFACE,
    ];

    const REQUEST_OPCODES: &[(&WlInterface, u32, &str)] = &[
        (
            &DRM_SYNCOBJ_MANAGER_INTERFACE,
            DRM_SYNCOBJ_MANAGER_DESTROY,
            "destroy",
        ),
        (
            &DRM_SYNCOBJ_MANAGER_INTERFACE,
            DRM_SYNCOBJ_MANAGER_GET_SURFACE,
            "get_surface",
        ),
        (
            &DRM_SYNCOBJ_MANAGER_INTERFACE,
            DRM_SYNCOBJ_MANAGER_IMPORT_TIMELINE,
            "import_timeline",
        ),
        (
            &DRM_SYNCOBJ_TIMELINE_INTERFACE,
            DRM_SYNCOBJ_TIMELINE_DESTROY,
            "destroy",
        ),
        (
            &DRM_SYNCOBJ_SURFACE_INTERFACE,
            DRM_SYNCOBJ_SURFACE_DESTROY,
            "destroy",
        ),
        (
            &DRM_SYNCOBJ_SURFACE_INTERFACE,
            DRM_SYNCOBJ_SURFACE_SET_ACQUIRE_POINT,
            "set_acquire_point",
        ),
        (
            &DRM_SYNCOBJ_SURFACE_INTERFACE,
            DRM_SYNCOBJ_SURFACE_SET_RELEASE_POINT,
            "set_release_point",
        ),
    ];

    /// The protocol defines no events on any of its interfaces.
    const EVENT_OPCODES: &[(&WlInterface, u32, &str)] = &[];

    #[test]
    fn every_message_slot_matches_its_signature() {
        for interface in ALL {
            for message in interface
                .requests
                .iter()
                .chain(interface.events.iter())
            {
                assert!(!message.name.is_empty(), "unnamed message");
                assert_eq!(
                    message.arg_count(),
                    message.types.len(),
                    "{}.{} signature and types table disagree",
                    interface.name,
                    message.name
                );
            }
        }
    }

    #[test]
    fn no_message_needs_a_newer_version_than_its_interface() {
        for interface in ALL {
            for message in interface
                .requests
                .iter()
                .chain(interface.events.iter())
            {
                assert!(
                    message.since() <= interface.version,
                    "{}.{} is since {} but the interface is version {}",
                    interface.name,
                    message.name,
                    message.since(),
                    interface.version
                );
            }
        }
    }

    #[test]
    fn opcodes_index_the_reference_tables() {
        for (interface, opcode, expected) in REQUEST_OPCODES {
            let message = interface
                .request_at(*opcode)
                .unwrap_or_else(|| panic!("{} has no request {opcode}", interface.name));
            assert_eq!(message.name, *expected, "{}.{}", interface.name, expected);
        }
        for (interface, opcode, expected) in EVENT_OPCODES {
            let message = interface
                .event_at(*opcode)
                .unwrap_or_else(|| panic!("{} has no event {opcode}", interface.name));
            assert_eq!(message.name, *expected, "{}.{}", interface.name, expected);
        }
    }

    #[test]
    fn signatures_and_versions_match_the_reference_protocol() {
        for interface in ALL {
            assert_eq!(interface.version, 1, "{}", interface.name);
            assert!(interface.events.is_empty(), "{} has events", interface.name);
        }

        let destroy = *DRM_SYNCOBJ_MANAGER_INTERFACE
            .request_at(DRM_SYNCOBJ_MANAGER_DESTROY)
            .unwrap_or_else(|| panic!("the manager has no destroy"));
        assert_eq!(destroy.signature, "");
        assert_eq!(destroy.arg_count(), 0);
        assert_eq!(destroy.since(), 1);

        let get_surface = DRM_SYNCOBJ_MANAGER_INTERFACE
            .request_at(DRM_SYNCOBJ_MANAGER_GET_SURFACE)
            .unwrap_or_else(|| panic!("the manager has no get_surface"));
        assert_eq!(get_surface.signature, "no");
        assert_eq!(get_surface.arg_count(), 2);
        assert_eq!(get_surface.types.len(), 2);
        assert_eq!(
            get_surface
                .type_at(0)
                .map(|interface| interface.name),
            Some("wp_linux_drm_syncobj_surface_v1")
        );
        assert_eq!(
            get_surface
                .type_at(1)
                .map(|interface| interface.name),
            Some("wl_surface")
        );

        let import_timeline = DRM_SYNCOBJ_MANAGER_INTERFACE
            .request_at(DRM_SYNCOBJ_MANAGER_IMPORT_TIMELINE)
            .unwrap_or_else(|| panic!("the manager has no import_timeline"));
        assert_eq!(import_timeline.signature, "nh");
        assert_eq!(import_timeline.arg_count(), 2);
        assert_eq!(import_timeline.types.len(), 2);
        assert_eq!(import_timeline.fd_count(), 1);
        assert_eq!(
            import_timeline
                .type_at(0)
                .map(|interface| interface.name),
            Some("wp_linux_drm_syncobj_timeline_v1")
        );
        assert!(import_timeline.type_at(1).is_none());

        let timeline_destroy = DRM_SYNCOBJ_TIMELINE_INTERFACE
            .request_at(DRM_SYNCOBJ_TIMELINE_DESTROY)
            .unwrap_or_else(|| panic!("the timeline has no destroy"));
        assert_eq!(timeline_destroy.signature, "");
        assert_eq!(timeline_destroy.arg_count(), 0);

        for opcode in [
            DRM_SYNCOBJ_SURFACE_SET_ACQUIRE_POINT,
            DRM_SYNCOBJ_SURFACE_SET_RELEASE_POINT,
        ] {
            let point = DRM_SYNCOBJ_SURFACE_INTERFACE
                .request_at(opcode)
                .unwrap_or_else(|| panic!("the surface has no request {opcode}"));
            assert_eq!(point.signature, "ouu");
            assert_eq!(point.arg_count(), 3);
            assert_eq!(point.types.len(), 3);
            assert_eq!(
                point.type_at(0).map(|interface| interface.name),
                Some("wp_linux_drm_syncobj_timeline_v1")
            );
            assert!(point.type_at(1).is_none());
            assert!(point.type_at(2).is_none());
        }

        let surface_destroy = DRM_SYNCOBJ_SURFACE_INTERFACE
            .request_at(DRM_SYNCOBJ_SURFACE_DESTROY)
            .unwrap_or_else(|| panic!("the surface has no destroy"));
        assert_eq!(surface_destroy.signature, "");
        assert_eq!(surface_destroy.arg_count(), 0);
    }

    #[test]
    fn error_codes_round_trip() {
        assert_eq!(DrmSyncobjManagerError::SurfaceExists.code(), 0);
        assert_eq!(
            DrmSyncobjManagerError::from_code(0),
            Some(DrmSyncobjManagerError::SurfaceExists)
        );
        assert_eq!(
            DrmSyncobjManagerError::from_code(1),
            Some(DrmSyncobjManagerError::InvalidTimeline)
        );
        assert_eq!(DrmSyncobjManagerError::from_code(2), None);
        assert_eq!(DrmSyncobjManagerError::InvalidTimeline.name(), "invalid_timeline");
        assert_eq!(
            DrmSyncobjManagerError::SurfaceExists.to_string(),
            "surface_exists"
        );

        // wp_linux_drm_syncobj_surface_v1.error has no entry at code 0.
        assert_eq!(DrmSyncobjSurfaceError::from_code(0), None);
        for error in [
            DrmSyncobjSurfaceError::NoSurface,
            DrmSyncobjSurfaceError::UnsupportedBuffer,
            DrmSyncobjSurfaceError::NoBuffer,
            DrmSyncobjSurfaceError::NoAcquirePoint,
            DrmSyncobjSurfaceError::NoReleasePoint,
            DrmSyncobjSurfaceError::ConflictingPoints,
        ] {
            assert_eq!(DrmSyncobjSurfaceError::from_code(error.code()), Some(error));
        }
        assert_eq!(DrmSyncobjSurfaceError::NoBuffer.code(), 3);
        assert_eq!(DrmSyncobjSurfaceError::from_code(7), None);
        assert_eq!(DrmSyncobjSurfaceError::NoReleasePoint.name(), "no_release_point");
        assert_eq!(
            DrmSyncobjSurfaceError::ConflictingPoints.to_string(),
            "conflicting_points"
        );
    }
}
