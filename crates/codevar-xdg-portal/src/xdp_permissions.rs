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

//! Permission store access, ported from `desktop-portal/xdp-permissions.c`.
//!
//! Reads and writes per-application permissions through the
//! `org.freedesktop.impl.portal.PermissionStore` service and converts
//! them to and from the `yes`/`no`/`ask` tristate used by portals.
//!
//! Deviations from the C implementation: there is no process-wide
//! proxy singleton (`xdp_init_permission_store`/`xdp_get_permission_store`)
//! — the functions take the connection explicitly instead — the
//! application is identified by its `app_id` string rather than an
//! `XdpAppInfo` pointer, and the Dex future variants are left to the
//! dispatch layer, which owns the asynchronous call machinery.

use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

use codevar_dbus::{BodyWriter, Connection, DbusReader, DbusResult, DbusTransport};

use crate::xdp_error::PortalError;

/// Bus name of the permission store service.
pub const PERMISSION_STORE_DBUS_NAME: &str =
    "org.freedesktop.impl.portal.PermissionStore";

/// Object path of the permission store service.
pub const PERMISSION_STORE_DBUS_PATH: &str =
    "/org/freedesktop/impl/portal/PermissionStore";

/// Interface implemented by the permission store service.
pub const PERMISSION_STORE_INTERFACE: &str =
    "org.freedesktop.impl.portal.PermissionStore";

/// Call timeout used by GDBus proxies created with default settings
/// (25 seconds).
const CALL_TIMEOUT: Duration = Duration::from_secs(25);

/// The tristate permission of an application for one resource,
/// mirroring `XdpPermission`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Permission {
    /// No stored permission, or a stored entry with an unusable
    /// format; portals treat it like an unset preference.
    #[default]
    Unset,
    /// The permission is denied.
    No,
    /// The permission is granted.
    Yes,
    /// The user is asked before granting the permission.
    Ask,
}

/// Converts a stored permission list to a [`Permission`], like
/// `xdp_permissions_to_tristate`.
///
/// Anything other than exactly one `yes`, `no` or `ask` entry yields
/// [`Permission::Unset`] with a warning, as in the C code.
#[must_use]
pub fn to_tristate(permissions: &[String]) -> Permission {
    if permissions.len() != 1 {
        log::warn!(
            "Wrong permission format, ignoring ({})",
            permissions.join(" ")
        );
        return Permission::Unset;
    }
    match permissions[0].as_str() {
        "yes" => Permission::Yes,
        "no" => Permission::No,
        "ask" => Permission::Ask,
        _ => {
            log::warn!(
                "Wrong permission format, ignoring ({})",
                permissions.join(" ")
            );
            Permission::Unset
        }
    }
}

/// Converts a [`Permission`] back to the stored permission list,
/// like `xdp_permissions_from_tristate`.
///
/// Returns `None` for [`Permission::Unset`], which callers pass on as
/// an empty permission array.
#[must_use]
pub fn from_tristate(permission: Permission) -> Option<Vec<String>> {
    match permission {
        Permission::Unset => None,
        Permission::No => Some(Vec::from([String::from("no")])),
        Permission::Yes => Some(Vec::from([String::from("yes")])),
        Permission::Ask => Some(Vec::from([String::from("ask")])),
    }
}

/// Looks up the permissions stored for `app_id` under `table`/`id`,
/// like `xdp_get_permissions_sync`.
///
/// Mirroring the C behaviour, any failure to reach the permission
/// store or a missing entry for `app_id` yields `Ok(None)`; only a
/// malformed reply is reported as an error.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when the `Lookup` reply cannot be
/// decoded.
pub fn get_permissions<T: DbusTransport>(
    connection: &mut Connection<T>,
    app_id: &str,
    table: &str,
    id: &str,
) -> Result<Option<Vec<String>>, PortalError> {
    let reply = match connection.call(
        PERMISSION_STORE_DBUS_NAME,
        PERMISSION_STORE_DBUS_PATH,
        PERMISSION_STORE_INTERFACE,
        "Lookup",
        |writer| write_lookup_args(writer, table, id),
        CALL_TIMEOUT,
    ) {
        Ok(reply) => reply,
        Err(error) => {
            log::debug!("No '{table}' permissions found: {error}");
            return Ok(None);
        }
    };
    let mut reader = reply.body_reader();
    let permissions = decode_permissions(&mut reader, app_id)?;
    if permissions.is_none() {
        log::debug!("No permissions stored for: {table} {id}, app {app_id}");
    }
    Ok(permissions)
}

/// Returns the tristate permission of `app_id` under `table`/`id`,
/// like `xdp_get_permission_sync`.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when the `Lookup` reply cannot be
/// decoded.
pub fn get_permission<T: DbusTransport>(
    connection: &mut Connection<T>,
    app_id: &str,
    table: &str,
    id: &str,
) -> Result<Permission, PortalError> {
    let permissions = get_permissions(connection, app_id, table, id)?;
    Ok(match permissions {
        Some(permissions) => to_tristate(&permissions),
        None => Permission::Unset,
    })
}

/// Writes `permissions` for `app_id` under `table`/`id`, like
/// `xdp_set_permissions_sync`.
///
/// Unlike the C function, which only logs a warning, transport
/// failures are propagated so the caller decides whether to continue.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when the permission store rejects
/// the call or the transport fails.
pub fn set_permissions<T: DbusTransport>(
    connection: &mut Connection<T>,
    app_id: &str,
    table: &str,
    id: &str,
    permissions: &[String],
) -> Result<(), PortalError> {
    match connection.call(
        PERMISSION_STORE_DBUS_NAME,
        PERMISSION_STORE_DBUS_PATH,
        PERMISSION_STORE_INTERFACE,
        "SetPermission",
        |writer| write_set_permission_args(writer, table, id, app_id, permissions),
        CALL_TIMEOUT,
    ) {
        Ok(_) => Ok(()),
        Err(error) => {
            log::warn!("Error updating permission store for {app_id}: {error}");
            Err(PortalError::from(error))
        }
    }
}

/// Writes `permission` for `app_id` under `table`/`id`, like
/// `xdp_set_permission_sync`.
///
/// [`Permission::Unset`] writes an empty permission array, matching
/// the `NULL` string vector the C code passes to `SetPermission`.
///
/// # Errors
///
/// Returns [`PortalError::Failed`] when the permission store rejects
/// the call or the transport fails.
pub fn set_permission<T: DbusTransport>(
    connection: &mut Connection<T>,
    app_id: &str,
    table: &str,
    id: &str,
    permission: Permission,
) -> Result<(), PortalError> {
    let permissions = from_tristate(permission).unwrap_or_default();
    set_permissions(connection, app_id, table, id, &permissions)
}

/// Reads the `a{sas}` permission map at the start of a `Lookup`
/// reply and returns the entry for `app_id`.
fn decode_permissions(
    reader: &mut DbusReader<'_>,
    app_id: &str,
) -> Result<Option<Vec<String>>, PortalError> {
    let mut entries = reader.read_array(8)?;
    while !entries.is_empty() {
        entries.read_struct()?;
        let key = entries.read_str()?;
        let mut values = entries.read_array(4)?;
        let mut permissions = Vec::new();
        while !values.is_empty() {
            permissions.push(String::from(values.read_str()?));
        }
        if key == app_id {
            return Ok(Some(permissions));
        }
    }
    Ok(None)
}

/// Marshals the `Lookup(s, s)` arguments.
fn write_lookup_args(writer: &mut BodyWriter, table: &str, id: &str) -> DbusResult<()> {
    writer.write_str(table)?;
    writer.write_str(id)?;
    Ok(())
}

/// Marshals the `SetPermission(s, b, s, s, as)` arguments with
/// `create` set to `TRUE`, as the C code always does.
fn write_set_permission_args(
    writer: &mut BodyWriter,
    table: &str,
    id: &str,
    app_id: &str,
    permissions: &[String],
) -> DbusResult<()> {
    writer.write_str(table)?;
    writer.write_bool(true)?;
    writer.write_str(id)?;
    writer.write_str(app_id)?;
    writer.write_array("s", |writer| {
        for permission in permissions {
            writer.write_str(permission)?;
        }
        Ok(())
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use codevar_dbus::{ByteOrder, DbusMessage};

    // unwrap() in tests is permitted by AGENTS.md; every call below
    // asserts a value this test itself constructed.

    fn build_lookup_body(entries: &[(&str, &[&str])]) -> (Vec<u8>, String) {
        let mut body = BodyWriter::new(ByteOrder::Little);
        body.write_array("{sas}", |writer| {
            for (app_id, permissions) in entries {
                writer.write_struct("sas", |writer| {
                    writer.write_str(app_id)?;
                    writer.write_array("s", |writer| {
                        for permission in *permissions {
                            writer.write_str(permission)?;
                        }
                        Ok(())
                    })
                })?;
            }
            Ok(())
        })
        .unwrap();
        body.into_parts()
    }

    fn decode_lookup(bytes: &[u8], app_id: &str) -> Option<Vec<String>> {
        let mut reader = DbusReader::new(bytes, ByteOrder::Little);
        decode_permissions(&mut reader, app_id).unwrap()
    }

    #[test]
    fn converts_between_permission_tristates() {
        assert_eq!(to_tristate(&[String::from("yes")]), Permission::Yes);
        assert_eq!(to_tristate(&[String::from("no")]), Permission::No);
        assert_eq!(to_tristate(&[String::from("ask")]), Permission::Ask);
        assert_eq!(to_tristate(&[]), Permission::Unset);
        assert_eq!(
            to_tristate(&[String::from("yes"), String::from("no")]),
            Permission::Unset
        );
        assert_eq!(to_tristate(&[String::from("YES")]), Permission::Unset);
        assert_eq!(to_tristate(&[String::from("")]), Permission::Unset);

        assert_eq!(from_tristate(Permission::Unset), None);
        for permission in [Permission::Yes, Permission::No, Permission::Ask] {
            let stored = from_tristate(permission).unwrap();
            assert_eq!(to_tristate(&stored), permission);
        }
    }

    #[test]
    fn decodes_permission_store_lookup_replies() {
        let (bytes, signature) = build_lookup_body(&[
            ("org.example.App", &["yes"]),
            ("org.other.App", &["no", "ask"]),
        ]);
        assert_eq!(signature, "a{sas}");

        assert_eq!(
            decode_lookup(&bytes, "org.example.App").unwrap(),
            Vec::from([String::from("yes")])
        );
        assert_eq!(
            decode_lookup(&bytes, "org.other.App").unwrap(),
            Vec::from([String::from("no"), String::from("ask")])
        );
        assert_eq!(decode_lookup(&bytes, "org.missing.App"), None);

        let (empty, _) = build_lookup_body(&[]);
        assert_eq!(decode_lookup(&empty, "org.example.App"), None);
    }

    #[test]
    fn encodes_permission_store_arguments() {
        let mut body = BodyWriter::new(ByteOrder::Little);
        write_lookup_args(&mut body, "notifications", "remind").unwrap();
        let (bytes, signature) = body.into_parts();
        assert_eq!(signature, "ss");
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert_eq!(reader.read_str().unwrap(), "notifications");
        assert_eq!(reader.read_str().unwrap(), "remind");
        assert!(reader.is_empty());

        let permissions = Vec::from([String::from("yes")]);
        let mut body = BodyWriter::new(ByteOrder::Little);
        write_set_permission_args(
            &mut body,
            "notifications",
            "remind",
            "org.example.App",
            &permissions,
        )
        .unwrap();
        let (bytes, signature) = body.into_parts();
        assert_eq!(signature, "sbssas");
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert_eq!(reader.read_str().unwrap(), "notifications");
        assert!(reader.read_bool().unwrap());
        assert_eq!(reader.read_str().unwrap(), "remind");
        assert_eq!(reader.read_str().unwrap(), "org.example.App");
        let mut array = reader.read_array(4).unwrap();
        let mut stored = Vec::new();
        while !array.is_empty() {
            stored.push(String::from(array.read_str().unwrap()));
        }
        assert_eq!(stored, permissions);
        assert!(reader.is_empty());

        let mut body = BodyWriter::new(ByteOrder::Little);
        write_set_permission_args(&mut body, "t", "id", "app", &[]).unwrap();
        let (bytes, signature) = body.into_parts();
        assert_eq!(signature, "sbssas");
        let mut reader = DbusReader::new(&bytes, ByteOrder::Little);
        assert_eq!(reader.read_str().unwrap(), "t");
        assert!(reader.read_bool().unwrap());
        assert_eq!(reader.read_str().unwrap(), "id");
        assert_eq!(reader.read_str().unwrap(), "app");
        assert!(reader.read_array(4).unwrap().is_empty());
        assert!(reader.is_empty());

        let reply = DbusMessage::method_call(
            PERMISSION_STORE_DBUS_NAME,
            PERMISSION_STORE_DBUS_PATH,
            PERMISSION_STORE_INTERFACE,
            "Lookup",
        )
        .unwrap();
        assert_eq!(reply.interface().unwrap(), PERMISSION_STORE_INTERFACE);
    }
}
