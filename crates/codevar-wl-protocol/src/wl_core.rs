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

//! Core interfaces for drawing and displaying surfaces.
//!
//! The tables mirror `protocol/wayland.xml` message by
//! message: signatures, `since` versions, argument interfaces and the
//! opcode order a real compositor expects on the wire.
//!
//! Besides the three requested families, `wl_compositor`, `wl_surface`
//! and `wl_shm`, the module defines every object those interfaces can
//! name — `wl_region`, `wl_shm_pool`, `wl_buffer` and `wl_output`, so
//! argument validation never falls back to an untyped slot. `wl_callback`
//! comes from [`crate::wl_handle`].

use core::fmt;

use crate::wl_handle::{CALLBACK_INTERFACE, WlInterface, WlMessage};

/// `wl_buffer.destroy` request opcode.
pub const BUFFER_DESTROY: u32 = 0;
/// `wl_buffer.release` event opcode.
pub const BUFFER_RELEASE: u32 = 0;

/// `wl_region.destroy` request opcode.
pub const REGION_DESTROY: u32 = 0;
/// `wl_region.add` request opcode.
pub const REGION_ADD: u32 = 1;
/// `wl_region.subtract` request opcode.
pub const REGION_SUBTRACT: u32 = 2;

/// `wl_shm_pool.create_buffer` request opcode.
pub const SHM_POOL_CREATE_BUFFER: u32 = 0;
/// `wl_shm_pool.destroy` request opcode.
pub const SHM_POOL_DESTROY: u32 = 1;
/// `wl_shm_pool.resize` request opcode.
pub const SHM_POOL_RESIZE: u32 = 2;

/// `wl_shm.create_pool` request opcode.
pub const SHM_CREATE_POOL: u32 = 0;
/// `wl_shm.release` request opcode.
pub const SHM_RELEASE: u32 = 1;
/// `wl_shm.format` event opcode.
pub const SHM_FORMAT: u32 = 0;

/// `wl_compositor.create_surface` request opcode.
pub const COMPOSITOR_CREATE_SURFACE: u32 = 0;
/// `wl_compositor.create_region` request opcode.
pub const COMPOSITOR_CREATE_REGION: u32 = 1;
/// `wl_compositor.release` request opcode.
pub const COMPOSITOR_RELEASE: u32 = 2;

/// `wl_surface.destroy` request opcode.
pub const SURFACE_DESTROY: u32 = 0;
/// `wl_surface.attach` request opcode.
pub const SURFACE_ATTACH: u32 = 1;
/// `wl_surface.damage` request opcode.
pub const SURFACE_DAMAGE: u32 = 2;
/// `wl_surface.frame` request opcode.
pub const SURFACE_FRAME: u32 = 3;
/// `wl_surface.set_opaque_region` request opcode.
pub const SURFACE_SET_OPAQUE_REGION: u32 = 4;
/// `wl_surface.set_input_region` request opcode.
pub const SURFACE_SET_INPUT_REGION: u32 = 5;
/// `wl_surface.commit` request opcode.
pub const SURFACE_COMMIT: u32 = 6;
/// `wl_surface.set_buffer_transform` request opcode.
pub const SURFACE_SET_BUFFER_TRANSFORM: u32 = 7;
/// `wl_surface.set_buffer_scale` request opcode.
pub const SURFACE_SET_BUFFER_SCALE: u32 = 8;
/// `wl_surface.damage_buffer` request opcode.
pub const SURFACE_DAMAGE_BUFFER: u32 = 9;
/// `wl_surface.offset` request opcode.
pub const SURFACE_OFFSET: u32 = 10;
/// `wl_surface.get_release` request opcode.
pub const SURFACE_GET_RELEASE: u32 = 11;
/// `wl_surface.enter` event opcode.
pub const SURFACE_ENTER: u32 = 0;
/// `wl_surface.leave` event opcode.
pub const SURFACE_LEAVE: u32 = 1;
/// `wl_surface.preferred_buffer_scale` event opcode.
pub const SURFACE_PREFERRED_BUFFER_SCALE: u32 = 2;
/// `wl_surface.preferred_buffer_transform` event opcode.
pub const SURFACE_PREFERRED_BUFFER_TRANSFORM: u32 = 3;

/// `wl_output.release` request opcode.
pub const OUTPUT_RELEASE: u32 = 0;
/// `wl_output.geometry` event opcode.
pub const OUTPUT_GEOMETRY: u32 = 0;
/// `wl_output.mode` event opcode.
pub const OUTPUT_MODE: u32 = 1;
/// `wl_output.done` event opcode.
pub const OUTPUT_DONE: u32 = 2;
/// `wl_output.scale` event opcode.
pub const OUTPUT_SCALE: u32 = 3;
/// `wl_output.name` event opcode.
pub const OUTPUT_NAME: u32 = 4;
/// `wl_output.description` event opcode.
pub const OUTPUT_DESCRIPTION: u32 = 5;

/// `wl_seat.get_pointer` request opcode.
pub const SEAT_GET_POINTER: u32 = 0;
/// `wl_seat.get_keyboard` request opcode.
pub const SEAT_GET_KEYBOARD: u32 = 1;
/// `wl_seat.get_touch` request opcode.
pub const SEAT_GET_TOUCH: u32 = 2;
/// `wl_seat.release` request opcode.
pub const SEAT_RELEASE: u32 = 3;
/// `wl_seat.capabilities` event opcode.
pub const SEAT_CAPABILITIES: u32 = 0;
/// `wl_seat.name` event opcode.
pub const SEAT_NAME: u32 = 1;

/// `wl_buffer` core interface.
pub static BUFFER_INTERFACE: WlInterface = WlInterface {
    name: "wl_buffer",
    version: 1,
    requests: &[WlMessage::new("destroy", "", &[])],
    events: &[WlMessage::new("release", "", &[])],
};

/// `wl_region` core interface.
pub static REGION_INTERFACE: WlInterface = WlInterface {
    name: "wl_region",
    version: 7,
    requests: &[
        WlMessage::new("destroy", "", &[]),
        WlMessage::new("add", "iiii", &[None, None, None, None]),
        WlMessage::new("subtract", "iiii", &[None, None, None, None]),
    ],
    events: &[],
};

/// `wl_shm_pool` core interface.
pub static SHM_POOL_INTERFACE: WlInterface = WlInterface {
    name: "wl_shm_pool",
    version: 3,
    requests: &[
        WlMessage::new(
            "create_buffer",
            "niiiiu",
            &[Some(&BUFFER_INTERFACE), None, None, None, None, None],
        ),
        WlMessage::new("destroy", "", &[]),
        WlMessage::new("resize", "i", &[None]),
    ],
    events: &[],
};

/// `wl_shm` core interface.
pub static SHM_INTERFACE: WlInterface = WlInterface {
    name: "wl_shm",
    version: 3,
    requests: &[
        WlMessage::new("create_pool", "nhi", &[Some(&SHM_POOL_INTERFACE), None, None]),
        WlMessage::new("release", "2", &[]),
    ],
    events: &[WlMessage::new("format", "u", &[None])],
};

/// `wl_output` core interface.
pub static OUTPUT_INTERFACE: WlInterface = WlInterface {
    name: "wl_output",
    version: 4,
    requests: &[WlMessage::new("release", "3", &[])],
    events: &[
        WlMessage::new(
            "geometry",
            "iiiiissi",
            &[None, None, None, None, None, None, None, None],
        ),
        WlMessage::new("mode", "uiii", &[None, None, None, None]),
        WlMessage::new("done", "2", &[]),
        WlMessage::new("scale", "2i", &[None]),
        WlMessage::new("name", "4s", &[None]),
        WlMessage::new("description", "4s", &[None]),
    ],
};

/// `wl_seat` core interface.
///
/// The pointer, keyboard and touch objects the `get_*` requests create
/// have no tables yet, so their new ids carry no interface and
/// `WlClientDisplay::marshal_new_id` refuses to create them until
/// those interfaces exist.
pub static SEAT_INTERFACE: WlInterface = WlInterface {
    name: "wl_seat",
    version: 11,
    requests: &[
        WlMessage::new("get_pointer", "n", &[None]),
        WlMessage::new("get_keyboard", "n", &[None]),
        WlMessage::new("get_touch", "n", &[None]),
        WlMessage::new("release", "5", &[]),
    ],
    events: &[
        WlMessage::new("capabilities", "u", &[None]),
        WlMessage::new("name", "2s", &[None]),
    ],
};

/// `wl_surface` core interface.
pub static SURFACE_INTERFACE: WlInterface = WlInterface {
    name: "wl_surface",
    version: 7,
    requests: &[
        WlMessage::new("destroy", "", &[]),
        WlMessage::new("attach", "?oii", &[Some(&BUFFER_INTERFACE), None, None]),
        WlMessage::new("damage", "iiii", &[None, None, None, None]),
        WlMessage::new("frame", "n", &[Some(&CALLBACK_INTERFACE)]),
        WlMessage::new("set_opaque_region", "?o", &[Some(&REGION_INTERFACE)]),
        WlMessage::new("set_input_region", "?o", &[Some(&REGION_INTERFACE)]),
        WlMessage::new("commit", "", &[]),
        WlMessage::new("set_buffer_transform", "2i", &[None]),
        WlMessage::new("set_buffer_scale", "3i", &[None]),
        WlMessage::new("damage_buffer", "4iiii", &[None, None, None, None]),
        WlMessage::new("offset", "5ii", &[None, None]),
        WlMessage::new("get_release", "7n", &[Some(&CALLBACK_INTERFACE)]),
    ],
    events: &[
        WlMessage::new("enter", "o", &[Some(&OUTPUT_INTERFACE)]),
        WlMessage::new("leave", "o", &[Some(&OUTPUT_INTERFACE)]),
        WlMessage::new("preferred_buffer_scale", "6i", &[None]),
        WlMessage::new("preferred_buffer_transform", "6u", &[None]),
    ],
};

/// `wl_compositor` core interface.
pub static COMPOSITOR_INTERFACE: WlInterface = WlInterface {
    name: "wl_compositor",
    version: 7,
    requests: &[
        WlMessage::new("create_surface", "n", &[Some(&SURFACE_INTERFACE)]),
        WlMessage::new("create_region", "n", &[Some(&REGION_INTERFACE)]),
        WlMessage::new("release", "7", &[]),
    ],
    events: &[],
};

/// `wl_shm.format` value `argb8888`, supported by every compositor.
pub const SHM_FORMAT_ARGB8888: u32 = 0;
/// `wl_shm.format` value `xrgb8888`, supported by every compositor.
pub const SHM_FORMAT_XRGB8888: u32 = 1;

/// Protocol error codes of `enum wl_shm.error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum WlShmError {
    /// Pixel format is unknown to the compositor.
    InvalidFormat = 0,
    /// Buffer stride is not a multiple of 4 bytes.
    InvalidStride = 1,
    /// Pool descriptor cannot be mapped into memory.
    InvalidFd = 2,
}

impl WlShmError {
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
            0 => Some(Self::InvalidFormat),
            1 => Some(Self::InvalidStride),
            2 => Some(Self::InvalidFd),
            _ => None,
        }
    }

    /// Returns the symbolic name of the error.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InvalidFormat => "invalid_format",
            Self::InvalidStride => "invalid_stride",
            Self::InvalidFd => "invalid_fd",
        }
    }
}

impl fmt::Display for WlShmError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wl_handle::WlArgType;

    const ALL: &[&WlInterface] = &[
        &BUFFER_INTERFACE,
        &REGION_INTERFACE,
        &SHM_POOL_INTERFACE,
        &SHM_INTERFACE,
        &OUTPUT_INTERFACE,
        &SEAT_INTERFACE,
        &SURFACE_INTERFACE,
        &COMPOSITOR_INTERFACE,
    ];

    /// Every opcode constant must index the message it names; a
    /// transposed entry would silently corrupt the wire protocol.
    const REQUEST_OPCODES: &[(&WlInterface, u32, &str)] = &[
        (&BUFFER_INTERFACE, BUFFER_DESTROY, "destroy"),
        (&REGION_INTERFACE, REGION_DESTROY, "destroy"),
        (&REGION_INTERFACE, REGION_ADD, "add"),
        (&REGION_INTERFACE, REGION_SUBTRACT, "subtract"),
        (&SHM_POOL_INTERFACE, SHM_POOL_CREATE_BUFFER, "create_buffer"),
        (&SHM_POOL_INTERFACE, SHM_POOL_DESTROY, "destroy"),
        (&SHM_POOL_INTERFACE, SHM_POOL_RESIZE, "resize"),
        (&SHM_INTERFACE, SHM_CREATE_POOL, "create_pool"),
        (&SHM_INTERFACE, SHM_RELEASE, "release"),
        (&COMPOSITOR_INTERFACE, COMPOSITOR_CREATE_SURFACE, "create_surface"),
        (&COMPOSITOR_INTERFACE, COMPOSITOR_CREATE_REGION, "create_region"),
        (&COMPOSITOR_INTERFACE, COMPOSITOR_RELEASE, "release"),
        (&SURFACE_INTERFACE, SURFACE_DESTROY, "destroy"),
        (&SURFACE_INTERFACE, SURFACE_ATTACH, "attach"),
        (&SURFACE_INTERFACE, SURFACE_DAMAGE, "damage"),
        (&SURFACE_INTERFACE, SURFACE_FRAME, "frame"),
        (&SURFACE_INTERFACE, SURFACE_SET_OPAQUE_REGION, "set_opaque_region"),
        (&SURFACE_INTERFACE, SURFACE_SET_INPUT_REGION, "set_input_region"),
        (&SURFACE_INTERFACE, SURFACE_COMMIT, "commit"),
        (
            &SURFACE_INTERFACE,
            SURFACE_SET_BUFFER_TRANSFORM,
            "set_buffer_transform",
        ),
        (&SURFACE_INTERFACE, SURFACE_SET_BUFFER_SCALE, "set_buffer_scale"),
        (&SURFACE_INTERFACE, SURFACE_DAMAGE_BUFFER, "damage_buffer"),
        (&SURFACE_INTERFACE, SURFACE_OFFSET, "offset"),
        (&SURFACE_INTERFACE, SURFACE_GET_RELEASE, "get_release"),
        (&OUTPUT_INTERFACE, OUTPUT_RELEASE, "release"),
        (&SEAT_INTERFACE, SEAT_GET_POINTER, "get_pointer"),
        (&SEAT_INTERFACE, SEAT_GET_KEYBOARD, "get_keyboard"),
        (&SEAT_INTERFACE, SEAT_GET_TOUCH, "get_touch"),
        (&SEAT_INTERFACE, SEAT_RELEASE, "release"),
    ];

    const EVENT_OPCODES: &[(&WlInterface, u32, &str)] = &[
        (&BUFFER_INTERFACE, BUFFER_RELEASE, "release"),
        (&SHM_INTERFACE, SHM_FORMAT, "format"),
        (&SURFACE_INTERFACE, SURFACE_ENTER, "enter"),
        (&SURFACE_INTERFACE, SURFACE_LEAVE, "leave"),
        (
            &SURFACE_INTERFACE,
            SURFACE_PREFERRED_BUFFER_SCALE,
            "preferred_buffer_scale",
        ),
        (
            &SURFACE_INTERFACE,
            SURFACE_PREFERRED_BUFFER_TRANSFORM,
            "preferred_buffer_transform",
        ),
        (&OUTPUT_INTERFACE, OUTPUT_GEOMETRY, "geometry"),
        (&OUTPUT_INTERFACE, OUTPUT_MODE, "mode"),
        (&OUTPUT_INTERFACE, OUTPUT_DONE, "done"),
        (&OUTPUT_INTERFACE, OUTPUT_SCALE, "scale"),
        (&OUTPUT_INTERFACE, OUTPUT_NAME, "name"),
        (&OUTPUT_INTERFACE, OUTPUT_DESCRIPTION, "description"),
        (&SEAT_INTERFACE, SEAT_CAPABILITIES, "capabilities"),
        (&SEAT_INTERFACE, SEAT_NAME, "name"),
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
        assert_eq!(SURFACE_INTERFACE.version, 7);
        assert_eq!(SHM_INTERFACE.version, 3);
        assert_eq!(SHM_POOL_INTERFACE.version, 3);
        assert_eq!(OUTPUT_INTERFACE.version, 4);
        assert_eq!(BUFFER_INTERFACE.version, 1);

        let attach = SURFACE_INTERFACE
            .request_at(SURFACE_ATTACH)
            .unwrap_or_else(|| panic!("wl_surface has no attach"));
        assert_eq!(attach.signature, "?oii");
        assert_eq!(attach.since(), 1);

        let offset = SURFACE_INTERFACE
            .request_at(SURFACE_OFFSET)
            .unwrap_or_else(|| panic!("wl_surface has no offset"));
        assert_eq!(offset.signature, "5ii");
        assert_eq!(offset.since(), 5);

        let get_release = SURFACE_INTERFACE
            .request_at(SURFACE_GET_RELEASE)
            .unwrap_or_else(|| panic!("wl_surface has no get_release"));
        assert_eq!(get_release.signature, "7n");
        assert_eq!(get_release.since(), 7);

        let create_pool = SHM_INTERFACE
            .request_at(SHM_CREATE_POOL)
            .unwrap_or_else(|| panic!("wl_shm has no create_pool"));
        assert_eq!(create_pool.signature, "nhi");
        assert_eq!(create_pool.fd_count(), 1);

        let create_buffer = SHM_POOL_INTERFACE
            .request_at(SHM_POOL_CREATE_BUFFER)
            .unwrap_or_else(|| panic!("wl_shm_pool has no create_buffer"));
        assert_eq!(create_buffer.signature, "niiiiu");

        let release = SHM_INTERFACE
            .request_at(SHM_RELEASE)
            .unwrap_or_else(|| panic!("wl_shm has no release"));
        assert_eq!(release.signature, "2");
        assert_eq!(release.since(), 2);

        let geometry = OUTPUT_INTERFACE
            .event_at(OUTPUT_GEOMETRY)
            .unwrap_or_else(|| panic!("wl_output has no geometry"));
        assert_eq!(geometry.signature, "iiiiissi");

        let done = OUTPUT_INTERFACE
            .event_at(OUTPUT_DONE)
            .unwrap_or_else(|| panic!("wl_output has no done"));
        assert_eq!(done.signature, "2");
        assert_eq!(done.since(), 2);
    }

    #[test]
    fn object_arguments_carry_interfaces_and_nullability() {
        let attach = *SURFACE_INTERFACE
            .request_at(SURFACE_ATTACH)
            .unwrap_or_else(|| panic!("wl_surface has no attach"));
        let mut attach_args = attach.args();
        let buffer = attach_args
            .next()
            .unwrap_or_else(|| panic!("attach has no arguments"));
        assert_eq!(buffer.details.ty, WlArgType::Object);
        assert!(buffer.details.nullable);
        assert_eq!(
            buffer.interface.map(|interface| interface.name),
            Some("wl_buffer")
        );

        let create_surface = *COMPOSITOR_INTERFACE
            .request_at(COMPOSITOR_CREATE_SURFACE)
            .unwrap_or_else(|| panic!("wl_compositor has no create_surface"));
        let mut create_args = create_surface.args();
        let surface = create_args
            .next()
            .unwrap_or_else(|| panic!("create_surface has no arguments"));
        assert_eq!(surface.details.ty, WlArgType::NewId);
        assert_eq!(
            surface.interface.map(|interface| interface.name),
            Some("wl_surface")
        );

        let create_pool = *SHM_INTERFACE
            .request_at(SHM_CREATE_POOL)
            .unwrap_or_else(|| panic!("wl_shm has no create_pool"));
        let pool_args: Vec<_> = create_pool.args().collect();
        assert_eq!(pool_args.len(), 3);
        assert_eq!(pool_args[0].details.ty, WlArgType::NewId);
        assert_eq!(pool_args[1].details.ty, WlArgType::Fd);
        assert_eq!(pool_args[2].details.ty, WlArgType::Int);

        let enter = *SURFACE_INTERFACE
            .event_at(SURFACE_ENTER)
            .unwrap_or_else(|| panic!("wl_surface has no enter"));
        let mut enter_args = enter.args();
        let output = enter_args
            .next()
            .unwrap_or_else(|| panic!("enter has no arguments"));
        assert!(!output.details.nullable);
        assert_eq!(
            output.interface.map(|interface| interface.name),
            Some("wl_output")
        );
    }

    #[test]
    fn shm_error_codes_round_trip() {
        for error in [
            WlShmError::InvalidFormat,
            WlShmError::InvalidStride,
            WlShmError::InvalidFd,
        ] {
            assert_eq!(WlShmError::from_code(error.code()), Some(error));
        }
        assert_eq!(WlShmError::from_code(3), None);
        assert_eq!(WlShmError::InvalidFormat.code(), 0);
        assert_eq!(WlShmError::InvalidFd.code(), 2);
        assert_eq!(WlShmError::InvalidStride.name(), "invalid_stride");
        assert_eq!(WlShmError::InvalidFormat.to_string(), "invalid_format");
        assert_eq!(SHM_FORMAT_ARGB8888, 0);
        assert_eq!(SHM_FORMAT_XRGB8888, 1);
    }
}
