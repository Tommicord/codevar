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

//! `linux-dmabuf` dmabuf based `wl_buffer` creation and feedback.
//!
//! The tables mirror `stable/linux-dmabuf/linux-dmabuf-v1.xml` of
//! `wayland-protocols` message by message: signatures, `since`
//! versions, argument interfaces, enum codes and the opcode order a
//! real compositor expects on the wire. The protocol is stable at
//! version 5.
//!
//! The handshake a client performs is: bind `zwp_linux_dmabuf_v1`,
//! `create_params`, one `add` per plane carrying a dmabuf file
//! descriptor, then `create_immed` (or `create` followed by the
//! `created` event) to obtain a [`wl_buffer`](crate::BUFFER_INTERFACE).
//! Since version 4 the `format` and `modifier` advertisements are
//! deprecated: `get_default_feedback` and `get_surface_feedback`
//! deliver a format table and preference tranches on
//! `zwp_linux_dmabuf_feedback_v1` instead.

use core::fmt;

use crate::wl_core::{BUFFER_INTERFACE, SURFACE_INTERFACE};
use crate::wl_handle::{WlInterface, WlMessage};

/// `zwp_linux_dmabuf_v1.destroy` request opcode.
pub const DMABUF_DESTROY: u32 = 0;
/// `zwp_linux_dmabuf_v1.create_params` request opcode.
pub const DMABUF_CREATE_PARAMS: u32 = 1;
/// `zwp_linux_dmabuf_v1.get_default_feedback` request opcode (since 4).
pub const DMABUF_GET_DEFAULT_FEEDBACK: u32 = 2;
/// `zwp_linux_dmabuf_v1.get_surface_feedback` request opcode (since 4).
pub const DMABUF_GET_SURFACE_FEEDBACK: u32 = 3;
/// `zwp_linux_dmabuf_v1.format` event opcode (deprecated since 4).
pub const DMABUF_FORMAT: u32 = 0;
/// `zwp_linux_dmabuf_v1.modifier` event opcode (since 3, deprecated since 4).
pub const DMABUF_MODIFIER: u32 = 1;

/// `zwp_linux_buffer_params_v1.destroy` request opcode.
pub const BUFFER_PARAMS_DESTROY: u32 = 0;
/// `zwp_linux_buffer_params_v1.add` request opcode.
pub const BUFFER_PARAMS_ADD: u32 = 1;
/// `zwp_linux_buffer_params_v1.create` request opcode.
pub const BUFFER_PARAMS_CREATE: u32 = 2;
/// `zwp_linux_buffer_params_v1.create_immed` request opcode (since 2).
pub const BUFFER_PARAMS_CREATE_IMMED: u32 = 3;
/// `zwp_linux_buffer_params_v1.created` event opcode.
pub const BUFFER_PARAMS_CREATED: u32 = 0;
/// `zwp_linux_buffer_params_v1.failed` event opcode.
pub const BUFFER_PARAMS_FAILED: u32 = 1;

/// `zwp_linux_dmabuf_feedback_v1.destroy` request opcode.
pub const FEEDBACK_DESTROY: u32 = 0;
/// `zwp_linux_dmabuf_feedback_v1.done` event opcode.
pub const FEEDBACK_DONE: u32 = 0;
/// `zwp_linux_dmabuf_feedback_v1.format_table` event opcode.
pub const FEEDBACK_FORMAT_TABLE: u32 = 1;
/// `zwp_linux_dmabuf_feedback_v1.main_device` event opcode.
pub const FEEDBACK_MAIN_DEVICE: u32 = 2;
/// `zwp_linux_dmabuf_feedback_v1.tranche_done` event opcode.
pub const FEEDBACK_TRANCHE_DONE: u32 = 3;
/// `zwp_linux_dmabuf_feedback_v1.tranche_target_device` event opcode.
pub const FEEDBACK_TRANCHE_TARGET_DEVICE: u32 = 4;
/// `zwp_linux_dmabuf_feedback_v1.tranche_formats` event opcode.
pub const FEEDBACK_TRANCHE_FORMATS: u32 = 5;
/// `zwp_linux_dmabuf_feedback_v1.tranche_flags` event opcode.
pub const FEEDBACK_TRANCHE_FLAGS: u32 = 6;

/// `zwp_linux_dmabuf_v1` factory for dmabuf based `wl_buffer`s.
pub static DMABUF_INTERFACE: WlInterface = WlInterface {
    name: "zwp_linux_dmabuf_v1",
    version: 5,
    requests: &[
        WlMessage::new("destroy", "", &[]),
        WlMessage::new("create_params", "n", &[Some(&BUFFER_PARAMS_INTERFACE)]),
        WlMessage::new("get_default_feedback", "4n", &[Some(&DMABUF_FEEDBACK_INTERFACE)]),
        WlMessage::new(
            "get_surface_feedback",
            "4no",
            &[Some(&DMABUF_FEEDBACK_INTERFACE), Some(&SURFACE_INTERFACE)],
        ),
    ],
    events: &[
        WlMessage::new("format", "u", &[None]),
        WlMessage::new("modifier", "3uuu", &[None, None, None]),
    ],
};

/// `zwp_linux_buffer_params_v1` collector of dmabufs for one `wl_buffer`.
pub static BUFFER_PARAMS_INTERFACE: WlInterface = WlInterface {
    name: "zwp_linux_buffer_params_v1",
    version: 5,
    requests: &[
        WlMessage::new("destroy", "", &[]),
        WlMessage::new("add", "huuuuu", &[None, None, None, None, None, None]),
        WlMessage::new("create", "iiuu", &[None, None, None, None]),
        WlMessage::new(
            "create_immed",
            "2niiuu",
            &[Some(&BUFFER_INTERFACE), None, None, None, None],
        ),
    ],
    events: &[
        WlMessage::new("created", "n", &[Some(&BUFFER_INTERFACE)]),
        WlMessage::new("failed", "", &[]),
    ],
};

/// `zwp_linux_dmabuf_feedback_v1` advertised dmabuf parameters.
pub static DMABUF_FEEDBACK_INTERFACE: WlInterface = WlInterface {
    name: "zwp_linux_dmabuf_feedback_v1",
    version: 5,
    requests: &[WlMessage::new("destroy", "", &[])],
    events: &[
        WlMessage::new("done", "", &[]),
        WlMessage::new("format_table", "hu", &[None, None]),
        WlMessage::new("main_device", "a", &[None]),
        WlMessage::new("tranche_done", "", &[]),
        WlMessage::new("tranche_target_device", "a", &[None]),
        WlMessage::new("tranche_formats", "a", &[None]),
        WlMessage::new("tranche_flags", "u", &[None]),
    ],
};

/// `zwp_linux_buffer_params_v1.create` format value `xrgb8888`, the
/// DRM fourcc code `XR24` of `drm_fourcc.h`.
pub const DRM_FORMAT_XRGB8888: u32 = 0x3432_5258;

/// Protocol error codes of `enum zwp_linux_buffer_params_v1.error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum LinuxBufferParamsError {
    /// The params object already created a `wl_buffer`.
    AlreadyUsed = 0,
    /// The plane index is out of bounds for the format.
    PlaneIdx = 1,
    /// The plane index was already set on this object.
    PlaneSet = 2,
    /// Planes are missing or there are too many for the format.
    Incomplete = 3,
    /// The format is not supported by the compositor.
    InvalidFormat = 4,
    /// The width or height of the buffer is invalid.
    InvalidDimensions = 5,
    /// `offset + stride * height` reaches outside the dmabuf.
    OutOfBounds = 6,
    /// `create_immed` produced an invalid `wl_buffer`.
    InvalidWlBuffer = 7,
}

impl LinuxBufferParamsError {
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
            0 => Some(Self::AlreadyUsed),
            1 => Some(Self::PlaneIdx),
            2 => Some(Self::PlaneSet),
            3 => Some(Self::Incomplete),
            4 => Some(Self::InvalidFormat),
            5 => Some(Self::InvalidDimensions),
            6 => Some(Self::OutOfBounds),
            7 => Some(Self::InvalidWlBuffer),
            _ => None,
        }
    }

    /// Returns the symbolic name of the error.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::AlreadyUsed => "already_used",
            Self::PlaneIdx => "plane_idx",
            Self::PlaneSet => "plane_set",
            Self::Incomplete => "incomplete",
            Self::InvalidFormat => "invalid_format",
            Self::InvalidDimensions => "invalid_dimensions",
            Self::OutOfBounds => "out_of_bounds",
            Self::InvalidWlBuffer => "invalid_wl_buffer",
        }
    }
}

impl fmt::Display for LinuxBufferParamsError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl core::error::Error for LinuxBufferParamsError {}

/// Bits of `enum zwp_linux_buffer_params_v1.flags`, a bitfield passed
/// as the `flags` argument of `create` and `create_immed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum LinuxBufferParamsFlags {
    /// The contents are y-inverted.
    YInvert = 1,
    /// The content is interlaced.
    Interlaced = 2,
    /// The bottom field comes first.
    BottomFirst = 4,
}

impl LinuxBufferParamsFlags {
    /// Returns the wire bit of the flag.
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a wire bit, returning `None` for unknown bits.
    ///
    /// Combined masks are not decoded; test individual bits with
    /// [`Self::code`] instead.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::YInvert),
            2 => Some(Self::Interlaced),
            4 => Some(Self::BottomFirst),
            _ => None,
        }
    }

    /// Returns the symbolic name of the flag.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::YInvert => "y_invert",
            Self::Interlaced => "interlaced",
            Self::BottomFirst => "bottom_first",
        }
    }
}

impl fmt::Display for LinuxBufferParamsFlags {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl core::error::Error for LinuxBufferParamsFlags {}

/// Bits of `enum zwp_linux_dmabuf_feedback_v1.tranche_flags`, a
/// bitfield passed as the `flags` argument of the `tranche_flags`
/// event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum LinuxDmabufFeedbackTrancheFlags {
    /// Direct scan-out may be attempted on the target device.
    Scanout = 1,
}

impl LinuxDmabufFeedbackTrancheFlags {
    /// Returns the wire bit of the flag.
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a wire bit, returning `None` for unknown bits.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::Scanout),
            _ => None,
        }
    }

    /// Returns the symbolic name of the flag.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scanout => "scanout",
        }
    }
}

impl fmt::Display for LinuxDmabufFeedbackTrancheFlags {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl core::error::Error for LinuxDmabufFeedbackTrancheFlags {}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[&WlInterface] = &[
        &DMABUF_INTERFACE,
        &BUFFER_PARAMS_INTERFACE,
        &DMABUF_FEEDBACK_INTERFACE,
    ];

    /// Every opcode constant must index the message it names; a
    /// transposed entry would silently corrupt the wire protocol.
    const REQUEST_OPCODES: &[(&WlInterface, u32, &str)] = &[
        (&DMABUF_INTERFACE, DMABUF_DESTROY, "destroy"),
        (&DMABUF_INTERFACE, DMABUF_CREATE_PARAMS, "create_params"),
        (
            &DMABUF_INTERFACE,
            DMABUF_GET_DEFAULT_FEEDBACK,
            "get_default_feedback",
        ),
        (
            &DMABUF_INTERFACE,
            DMABUF_GET_SURFACE_FEEDBACK,
            "get_surface_feedback",
        ),
        (&BUFFER_PARAMS_INTERFACE, BUFFER_PARAMS_DESTROY, "destroy"),
        (&BUFFER_PARAMS_INTERFACE, BUFFER_PARAMS_ADD, "add"),
        (&BUFFER_PARAMS_INTERFACE, BUFFER_PARAMS_CREATE, "create"),
        (
            &BUFFER_PARAMS_INTERFACE,
            BUFFER_PARAMS_CREATE_IMMED,
            "create_immed",
        ),
        (&DMABUF_FEEDBACK_INTERFACE, FEEDBACK_DESTROY, "destroy"),
    ];

    const EVENT_OPCODES: &[(&WlInterface, u32, &str)] = &[
        (&DMABUF_INTERFACE, DMABUF_FORMAT, "format"),
        (&DMABUF_INTERFACE, DMABUF_MODIFIER, "modifier"),
        (&BUFFER_PARAMS_INTERFACE, BUFFER_PARAMS_CREATED, "created"),
        (&BUFFER_PARAMS_INTERFACE, BUFFER_PARAMS_FAILED, "failed"),
        (&DMABUF_FEEDBACK_INTERFACE, FEEDBACK_DONE, "done"),
        (&DMABUF_FEEDBACK_INTERFACE, FEEDBACK_FORMAT_TABLE, "format_table"),
        (&DMABUF_FEEDBACK_INTERFACE, FEEDBACK_MAIN_DEVICE, "main_device"),
        (&DMABUF_FEEDBACK_INTERFACE, FEEDBACK_TRANCHE_DONE, "tranche_done"),
        (
            &DMABUF_FEEDBACK_INTERFACE,
            FEEDBACK_TRANCHE_TARGET_DEVICE,
            "tranche_target_device",
        ),
        (
            &DMABUF_FEEDBACK_INTERFACE,
            FEEDBACK_TRANCHE_FORMATS,
            "tranche_formats",
        ),
        (
            &DMABUF_FEEDBACK_INTERFACE,
            FEEDBACK_TRANCHE_FLAGS,
            "tranche_flags",
        ),
    ];

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
            assert_eq!(interface.version, 5, "{}", interface.name);
        }

        let create_params = DMABUF_INTERFACE
            .request_at(DMABUF_CREATE_PARAMS)
            .unwrap_or_else(|| panic!("zwp_linux_dmabuf_v1 has no create_params"));
        assert_eq!(create_params.signature, "n");
        assert_eq!(
            create_params
                .type_at(0)
                .map(|interface| interface.name),
            Some("zwp_linux_buffer_params_v1")
        );

        let get_default_feedback = DMABUF_INTERFACE
            .request_at(DMABUF_GET_DEFAULT_FEEDBACK)
            .unwrap_or_else(|| panic!("zwp_linux_dmabuf_v1 has no get_default_feedback"));
        assert_eq!(get_default_feedback.signature, "4n");
        assert_eq!(get_default_feedback.since(), 4);
        assert_eq!(
            get_default_feedback
                .type_at(0)
                .map(|interface| interface.name),
            Some("zwp_linux_dmabuf_feedback_v1")
        );

        let get_surface_feedback = DMABUF_INTERFACE
            .request_at(DMABUF_GET_SURFACE_FEEDBACK)
            .unwrap_or_else(|| panic!("zwp_linux_dmabuf_v1 has no get_surface_feedback"));
        assert_eq!(get_surface_feedback.signature, "4no");
        assert_eq!(get_surface_feedback.since(), 4);
        assert_eq!(
            get_surface_feedback
                .type_at(1)
                .map(|interface| interface.name),
            Some("wl_surface")
        );

        let modifier = *DMABUF_INTERFACE
            .event_at(DMABUF_MODIFIER)
            .unwrap_or_else(|| panic!("zwp_linux_dmabuf_v1 has no modifier"));
        assert_eq!(modifier.signature, "3uuu");
        assert_eq!(modifier.since(), 3);

        let add = BUFFER_PARAMS_INTERFACE
            .request_at(BUFFER_PARAMS_ADD)
            .unwrap_or_else(|| panic!("zwp_linux_buffer_params_v1 has no add"));
        assert_eq!(add.signature, "huuuuu");
        assert_eq!(add.fd_count(), 1);
        assert_eq!(add.arg_count(), 6);
        assert!(add.type_at(0).is_none());

        let create_immed = BUFFER_PARAMS_INTERFACE
            .request_at(BUFFER_PARAMS_CREATE_IMMED)
            .unwrap_or_else(|| panic!("zwp_linux_buffer_params_v1 has no create_immed"));
        assert_eq!(create_immed.signature, "2niiuu");
        assert_eq!(create_immed.since(), 2);
        assert_eq!(
            create_immed
                .type_at(0)
                .map(|interface| interface.name),
            Some("wl_buffer")
        );

        let created = *BUFFER_PARAMS_INTERFACE
            .event_at(BUFFER_PARAMS_CREATED)
            .unwrap_or_else(|| panic!("zwp_linux_buffer_params_v1 has no created"));
        assert_eq!(created.signature, "n");
        assert_eq!(
            created.type_at(0).map(|interface| interface.name),
            Some("wl_buffer")
        );

        let format_table = *DMABUF_FEEDBACK_INTERFACE
            .event_at(FEEDBACK_FORMAT_TABLE)
            .unwrap_or_else(|| panic!("zwp_linux_dmabuf_feedback_v1 has no format_table"));
        assert_eq!(format_table.signature, "hu");
        assert_eq!(format_table.fd_count(), 1);
        assert_eq!(format_table.arg_count(), 2);

        let tranche_formats = *DMABUF_FEEDBACK_INTERFACE
            .event_at(FEEDBACK_TRANCHE_FORMATS)
            .unwrap_or_else(|| panic!("zwp_linux_dmabuf_feedback_v1 has no tranche_formats"));
        assert_eq!(tranche_formats.signature, "a");
        assert_eq!(tranche_formats.array_count(), 1);

        let tranche_flags = *DMABUF_FEEDBACK_INTERFACE
            .event_at(FEEDBACK_TRANCHE_FLAGS)
            .unwrap_or_else(|| panic!("zwp_linux_dmabuf_feedback_v1 has no tranche_flags"));
        assert_eq!(tranche_flags.signature, "u");

        let feedback_destroy = DMABUF_FEEDBACK_INTERFACE
            .request_at(FEEDBACK_DESTROY)
            .unwrap_or_else(|| panic!("zwp_linux_dmabuf_feedback_v1 has no destroy"));
        assert_eq!(feedback_destroy.signature, "");
        assert_eq!(feedback_destroy.arg_count(), 0);
    }

    #[test]
    fn error_and_flag_codes_round_trip() {
        assert_eq!(LinuxBufferParamsError::AlreadyUsed.code(), 0);
        assert_eq!(
            LinuxBufferParamsError::from_code(0),
            Some(LinuxBufferParamsError::AlreadyUsed)
        );
        for error in [
            LinuxBufferParamsError::PlaneIdx,
            LinuxBufferParamsError::PlaneSet,
            LinuxBufferParamsError::Incomplete,
            LinuxBufferParamsError::InvalidFormat,
            LinuxBufferParamsError::InvalidDimensions,
            LinuxBufferParamsError::OutOfBounds,
            LinuxBufferParamsError::InvalidWlBuffer,
        ] {
            assert_eq!(LinuxBufferParamsError::from_code(error.code()), Some(error));
        }
        assert_eq!(LinuxBufferParamsError::from_code(8), None);
        assert_eq!(LinuxBufferParamsError::PlaneIdx.name(), "plane_idx");
        assert_eq!(
            LinuxBufferParamsError::InvalidWlBuffer.to_string(),
            "invalid_wl_buffer"
        );

        assert_eq!(LinuxBufferParamsFlags::YInvert.code(), 1);
        assert_eq!(LinuxBufferParamsFlags::Interlaced.code(), 2);
        assert_eq!(LinuxBufferParamsFlags::BottomFirst.code(), 4);
        assert_eq!(
            LinuxBufferParamsFlags::from_code(4),
            Some(LinuxBufferParamsFlags::BottomFirst)
        );
        // Combined masks are not a single flag.
        assert_eq!(LinuxBufferParamsFlags::from_code(3), None);
        assert_eq!(LinuxBufferParamsFlags::YInvert.to_string(), "y_invert");

        assert_eq!(LinuxDmabufFeedbackTrancheFlags::Scanout.code(), 1);
        assert_eq!(
            LinuxDmabufFeedbackTrancheFlags::from_code(1),
            Some(LinuxDmabufFeedbackTrancheFlags::Scanout)
        );
        assert_eq!(LinuxDmabufFeedbackTrancheFlags::from_code(2), None);
        assert_eq!(LinuxDmabufFeedbackTrancheFlags::Scanout.name(), "scanout");

        assert_eq!(DRM_FORMAT_XRGB8888, 0x3432_5258);
    }
}
