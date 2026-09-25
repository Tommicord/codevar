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

//! Server side of the wayland protocol
//!
//! [`WlServerDisplay`] owns an [`WlEventLoop`], every connected [`WlClient`]
//! and the globals published through `wl_registry`. Clients are attached with
//! [`WlServerDisplay::create_client`] over an arbitrary [`WlTransport`]; the
//! transport handle is watched by the event loop, which turns readiness into a
//! task on an internal queue drained by [`WlServerDisplay::dispatch`].
//!
//! Requests of the core `wl_display` and `wl_registry` interfaces are handled
//! by the display itself. Every other resource is dispatched through the
//! closure registered with [`WlServerDisplay::add_request_handler`], the
//! counterpart of libwayland's `void **` implementation tables.
//!
//! Listening sockets, global filters and resource destroy callbacks are not
//! provided: the embedder supplies connected transports and composes its own
//! teardown from the resource map of each client.
//!
//! # Example
//!
//! ```
//! use core::time::Duration;
//!
//! use codevar_wl_protocol::{
//!     WlClock, WlInterface, WlPollEntry, WlPoller, WlResult, WlServerDisplay, WlTransport,
//! };
//!
//! struct Never;
//!
//! impl WlPoller for Never {
//!     fn poll(
//!         &mut self,
//!         _entries: &mut [WlPollEntry],
//!         _timeout: Option<Duration>,
//!     ) -> WlResult<usize> {
//!         Ok(0)
//!     }
//! }
//!
//! struct Ticks;
//!
//! impl WlClock for Ticks {
//!     fn now_ms(&self) -> u64 {
//!         0
//!     }
//! }
//!
//! static TEST_INTERFACE: WlInterface = WlInterface::new("wl_test", 1, &[], &[]);
//!
//! # fn demo<T: WlTransport + 'static>(transport: T) -> WlResult<()> {
//! let mut server = WlServerDisplay::new(Never, Ticks);
//! let name = server.add_global(&TEST_INTERFACE, 1)?;
//! assert_eq!(name, 1);
//! assert_eq!(server.client_count(), 0);
//! server.terminate();
//! let _client = server.create_client(transport)?;
//! # Ok(())
//! # }
//! ```

use core::cell::RefCell;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::wl_conn::{WlClosure, WlConnection, WlTransport, lookup_objects};
use crate::wl_error::{WlError, WlResult};
use crate::wl_evloop::{WlClock, WlEventLoop, WlEventSourceId, WlPoller};
use crate::wl_handle::{
    CALLBACK_DONE, CALLBACK_INTERFACE, DISPLAY_DELETE_ID, DISPLAY_ERROR,
    DISPLAY_GET_REGISTRY, DISPLAY_INTERFACE, DISPLAY_SYNC, MAX_MESSAGE_SIZE,
    REGISTRY_BIND, REGISTRY_GLOBAL, REGISTRY_GLOBAL_REMOVE, REGISTRY_INTERFACE,
    SERVER_ID_START, WlArgument, WlDisplayError, WlInterface, WlMap, WlMapIter,
    WlMapSide, WlObject, WlPollEvents,
};

/// Id of the display resource, which is always `1`.
pub const DISPLAY_RESOURCE_ID: u32 = 1;

type BindFn<T> = Box<dyn FnMut(&mut WlClient<T>, WlClientId, u32, u32)>;
type RequestFn<T> =
    Box<dyn FnMut(&mut WlClient<T>, WlClientId, u32, u32, &mut [WlArgument])>;
type ScheduledTask<T, P, C> = Box<dyn FnOnce(&mut WlServerDisplay<T, P, C>)>;

/// Handle to one connected client of a [`WlServerDisplay`].
///
/// Identifiers stay valid until the client is destroyed; slots may be reused
/// for later connections, which invalidates the previous handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WlClientId {
    slot: usize,
    generation: u32,
}

/// Server side resource, the counterpart of the client proxy.
#[derive(Debug, Clone, Copy)]
pub struct WlResource {
    /// Id of the resource on the wire.
    pub id: u32,
    /// Interface of the resource.
    pub interface: &'static WlInterface,
    /// Protocol version negotiated when the resource was created.
    pub version: u32,
}

impl WlObject for WlResource {
    #[inline]
    fn id(&self) -> u32 {
        self.id
    }

    #[inline]
    fn interface(&self) -> &'static WlInterface {
        self.interface
    }

    #[inline]
    fn version(&self) -> u32 {
        self.version
    }
}

/// Connected client, mirroring `struct wl_client`.
///
/// The value owns the [`WlConnection`] over the client's transport and the
/// map of every resource the client created. During request dispatch the
/// display lends the client to the handler, so handlers send events with
/// [`WlClient::post_event`] instead of going through the display.
pub struct WlClient<T: WlTransport> {
    id: WlClientId,
    connection: WlConnection<T>,
    resources: WlMap<WlResource>,
    error: bool,
    source: WlEventSourceId,
}

impl<T: WlTransport> WlClient<T> {
    /// Returns the handle identifying this client in its display.
    #[inline]
    #[must_use]
    pub const fn id(&self) -> WlClientId {
        self.id
    }

    /// Returns the underlying connection.
    #[inline]
    #[must_use]
    pub fn connection(&self) -> &WlConnection<T> {
        &self.connection
    }

    /// Mutably returns the underlying connection.
    #[inline]
    #[must_use]
    pub fn connection_mut(&mut self) -> &mut WlConnection<T> {
        &mut self.connection
    }

    /// Returns `true` after a protocol error was posted on this client.
    #[inline]
    #[must_use]
    pub const fn is_in_error(&self) -> bool {
        self.error
    }

    /// Returns the live resources of the client with their ids.
    #[inline]
    #[must_use]
    pub fn resources(&self) -> WlMapIter<'_, WlResource> {
        self.resources.iter()
    }

    /// Returns the number of live resources of the client.
    #[must_use]
    pub fn resource_count(&self) -> usize {
        self.resources.iter().count()
    }

    /// Returns the interface of the resource behind `id`.
    #[must_use]
    pub fn resource_interface(&self, id: u32) -> Option<&'static WlInterface> {
        self.resources.lookup(id).map(WlResource::interface)
    }

    /// Returns the version of the resource behind `id`.
    #[must_use]
    pub fn resource_version(&self, id: u32) -> Option<u32> {
        self.resources.lookup(id).map(|resource| resource.version)
    }

    /// Creates a resource for `interface` at `id`.
    ///
    /// An `id` of zero asks the server id space for a fresh id, which is
    /// returned; any other id is used verbatim and the same value is
    /// returned.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidArgument`] when the id belongs to the peer
    /// id space or is already in use, [`WlError::InvalidObject`] when the id
    /// skips ahead of the map and [`WlError::TooManyObjects`] when the id
    /// space is exhausted.
    pub fn create_resource(
        &mut self,
        id: u32,
        interface: &'static WlInterface,
        version: u32,
    ) -> WlResult<u32> {
        if id == 0 {
            let allocated = self.resources.insert_new(WlResource {
                id: 0,
                interface,
                version,
            })?;
            if let Some(resource) = self.resources.lookup_mut(allocated) {
                resource.id = allocated;
            }
            return Ok(allocated);
        }
        if id < SERVER_ID_START {
            self.resources.reserve_new(id)?;
            self.resources.insert_at(
                id,
                WlResource {
                    id,
                    interface,
                    version,
                },
            )?;
            return Ok(id);
        }
        if self.resources.contains(id) {
            return Err(WlError::invalid_argument(format!(
                "id {id} is already in use"
            )));
        }
        self.resources.insert_at(
            id,
            WlResource {
                id,
                interface,
                version,
            },
        )?;
        Ok(id)
    }

    /// Destroys the resource behind `id`.
    ///
    /// Client created ids vacate their slot and are announced with
    /// `wl_display.delete_id`; ids from the server id space are freed for
    /// reuse immediately.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidState`] for the display resource and
    /// [`WlError::InvalidObject`] when the id has no live resource.
    pub fn destroy_resource(&mut self, id: u32) -> WlResult<()> {
        if id == DISPLAY_RESOURCE_ID {
            return Err(WlError::invalid_state(
                "the display resource cannot be destroyed",
            ));
        }
        if self.resources.lookup(id).is_none() {
            return Err(WlError::InvalidObject(id));
        }
        if id < SERVER_ID_START {
            if self.resources.lookup(DISPLAY_RESOURCE_ID).is_some()
                && self
                    .post_event(
                        DISPLAY_RESOURCE_ID,
                        DISPLAY_DELETE_ID,
                        vec![WlArgument::Uint(id)],
                    )
                    .is_err()
            {
                self.error = true;
            }
            self.resources.vacate_at(id)?;
        } else {
            self.resources.remove(id);
        }
        Ok(())
    }

    /// Queues an event on `resource`.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidObject`] when the resource is unknown,
    /// [`WlError::InvalidMethod`] when the interface has no event at
    /// `opcode` and [`WlError::InvalidArgument`] when `args` do not match
    /// the message signature.
    pub fn post_event(
        &mut self,
        resource: u32,
        opcode: u32,
        args: Vec<WlArgument>,
    ) -> WlResult<()> {
        let interface = self
            .resources
            .lookup(resource)
            .map(WlResource::interface)
            .ok_or(WlError::InvalidObject(resource))?;
        let message = interface.event_at(opcode).ok_or(WlError::InvalidMethod {
            interface: interface.name,
            opcode,
        })?;
        let mut closure = WlClosure::new(resource, opcode, message, args)?;
        self.connection.queue_closure(&mut closure)
    }

    /// Reports a protocol error with `wl_display.error` and marks the
    /// client for destruction.
    ///
    /// The event is attributed to `object_id`; the display posts errors of
    /// the request dispatch itself. Like in libwayland, repeat calls after
    /// the first error are ignored.
    pub fn post_error(&mut self, object_id: u32, code: u32, message: impl Into<String>) {
        if self.error {
            return;
        }
        let args = vec![
            WlArgument::Object(object_id),
            WlArgument::Uint(code),
            WlArgument::Str(Some(message.into())),
        ];
        if self
            .post_event(DISPLAY_RESOURCE_ID, DISPLAY_ERROR, args)
            .is_err()
        {
            return;
        }
        self.error = true;
    }

    /// Reports an out of memory condition with `wl_display.error`.
    pub fn post_no_memory(&mut self) {
        self.post_error(
            DISPLAY_RESOURCE_ID,
            WlDisplayError::NoMemory.code(),
            "no memory",
        );
    }

    /// Writes buffered events to the transport.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::WouldBlock`] when the transport accepted only
    /// part of the buffer and [`WlError::Disconnected`] when the peer is
    /// gone.
    pub fn flush(&mut self) -> WlResult<usize> {
        self.connection.flush()
    }
}

struct WlGlobal<T: WlTransport> {
    name: u32,
    interface: &'static WlInterface,
    version: u32,
    bind: Option<BindFn<T>>,
}

struct WlHandler<T: WlTransport> {
    interface: &'static WlInterface,
    callback: Option<RequestFn<T>>,
}

/// Cloneable handle to the task queue of a [`WlServerDisplay`].
///
/// Tasks queued through the handle run at the start of the next
/// [`WlServerDisplay::dispatch`] with exclusive access to the display,
/// which is how timer and idle callbacks of the event loop reach server
/// state.
pub struct WlTaskQueue<T, P, C>
where
    T: WlTransport + 'static,
    P: WlPoller + 'static,
    C: WlClock + 'static,
{
    tasks: Rc<RefCell<VecDeque<ScheduledTask<T, P, C>>>>,
}

impl<T, P, C> Clone for WlTaskQueue<T, P, C>
where
    T: WlTransport + 'static,
    P: WlPoller + 'static,
    C: WlClock + 'static,
{
    fn clone(&self) -> Self {
        Self {
            tasks: Rc::clone(&self.tasks),
        }
    }
}

impl<T, P, C> WlTaskQueue<T, P, C>
where
    T: WlTransport + 'static,
    P: WlPoller + 'static,
    C: WlClock + 'static,
{
    /// Queues `task` for the next dispatch of the display.
    pub fn push(&self, task: impl FnOnce(&mut WlServerDisplay<T, P, C>) + 'static) {
        self.tasks.borrow_mut().push_back(Box::new(task));
    }
}

/// Server display mirroring `struct wl_display`.
///
/// The display owns the event loop, the connected clients and the published
/// globals. [`WlServerDisplay::dispatch`] waits on the event loop, turns the
/// resulting readiness into client pumps, drains compositor tasks and flushes
/// every client.
pub struct WlServerDisplay<T, P, C>
where
    T: WlTransport + 'static,
    P: WlPoller + 'static,
    C: WlClock + 'static,
{
    event_loop: WlEventLoop<P, C>,
    clients: Vec<Option<WlClient<T>>>,
    generations: Vec<u32>,
    free: Vec<usize>,
    globals: Vec<WlGlobal<T>>,
    handlers: Vec<WlHandler<T>>,
    next_global_name: u32,
    next_serial: u64,
    tasks: Rc<RefCell<VecDeque<ScheduledTask<T, P, C>>>>,
    running: bool,
}

impl<T, P, C> WlServerDisplay<T, P, C>
where
    T: WlTransport + 'static,
    P: WlPoller + 'static,
    C: WlClock + 'static,
{
    /// Creates a display over `poller` and `clock`.
    #[must_use]
    pub fn new(poller: P, clock: C) -> Self {
        Self {
            event_loop: WlEventLoop::new(poller, clock),
            clients: Vec::new(),
            generations: Vec::new(),
            free: Vec::new(),
            globals: Vec::new(),
            handlers: Vec::new(),
            next_global_name: 1,
            next_serial: 1,
            tasks: Rc::new(RefCell::new(VecDeque::new())),
            running: true,
        }
    }

    /// Returns the event loop.
    #[inline]
    #[must_use]
    pub fn event_loop(&self) -> &WlEventLoop<P, C> {
        &self.event_loop
    }

    /// Mutably returns the event loop.
    #[inline]
    #[must_use]
    pub fn event_loop_mut(&mut self) -> &mut WlEventLoop<P, C> {
        &mut self.event_loop
    }

    /// Returns a handle to the task queue of the display.
    #[must_use]
    pub fn tasks(&self) -> WlTaskQueue<T, P, C> {
        WlTaskQueue {
            tasks: Rc::clone(&self.tasks),
        }
    }

    /// Queues `task` for the next dispatch of the display.
    pub fn schedule(&self, task: impl FnOnce(&mut Self) + 'static) {
        self.tasks.borrow_mut().push_back(Box::new(task));
    }

    /// Announces `interface` at `version` on every `wl_registry`.
    ///
    /// Binding the returned name creates a bare resource; use
    /// [`WlServerDisplay::add_global_with`] to run compositor code at bind
    /// time.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidArgument`] when `version` is zero or
    /// exceeds the version of `interface` and [`WlError::InvalidState`]
    /// when no more global names are available.
    pub fn add_global(
        &mut self,
        interface: &'static WlInterface,
        version: u32,
    ) -> WlResult<u32> {
        self.add_global_inner(interface, version, None)
    }

    /// Announces `interface` at `version` and invokes `bind` for every
    /// `wl_registry.bind` of the global.
    ///
    /// The callback receives the client, its handle, the negotiated version
    /// and the client allocated id; it typically creates the resource with
    /// [`WlClient::create_resource`]. Replacing the resources of the global
    /// is up to the callback.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidArgument`] when `version` is zero or
    /// exceeds the version of `interface` and [`WlError::InvalidState`]
    /// when no more global names are available.
    pub fn add_global_with<F>(
        &mut self,
        interface: &'static WlInterface,
        version: u32,
        bind: F,
    ) -> WlResult<u32>
    where
        F: FnMut(&mut WlClient<T>, WlClientId, u32, u32) + 'static,
    {
        self.add_global_inner(interface, version, Some(Box::new(bind)))
    }

    /// Withdraws the global called `name` and announces the removal to
    /// every registry.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidArgument`] when no global uses `name`.
    pub fn remove_global(&mut self, name: u32) -> WlResult<()> {
        let Some(index) = self.globals.iter().position(|global| global.name == name)
        else {
            return Err(WlError::invalid_argument(format!(
                "global {name} does not exist"
            )));
        };
        self.globals.remove(index);
        for slot in &mut self.clients {
            let Some(client) = slot.as_mut() else {
                continue;
            };
            let registries: Vec<u32> = client
                .resources()
                .filter(|(_, resource)| resource.interface.equal(&REGISTRY_INTERFACE))
                .map(|(id, _)| id)
                .collect();
            for id in registries {
                if client
                    .post_event(id, REGISTRY_GLOBAL_REMOVE, vec![WlArgument::Uint(name)])
                    .is_err()
                {
                    client.error = true;
                }
            }
        }
        Ok(())
    }

    /// Registers the request handler of every resource of `interface`.
    ///
    /// The handler receives the client, its handle, the target resource,
    /// the opcode and the decoded arguments. Core `wl_display` and
    /// `wl_registry` requests are handled by the display itself; their
    /// interfaces cannot be overridden.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidState`] when `interface` already has a
    /// handler.
    pub fn add_request_handler<F>(
        &mut self,
        interface: &'static WlInterface,
        handler: F,
    ) -> WlResult<()>
    where
        F: FnMut(&mut WlClient<T>, WlClientId, u32, u32, &mut [WlArgument]) + 'static,
    {
        if self
            .handlers
            .iter()
            .any(|existing| existing.interface.equal(interface))
        {
            return Err(WlError::invalid_state(format!(
                "{interface} already has a request handler"
            )));
        }
        self.handlers.push(WlHandler {
            interface,
            callback: Some(Box::new(handler)),
        });
        Ok(())
    }

    /// Attaches `transport` as a new client and installs its display
    /// resource.
    ///
    /// The transport handle is watched for reads; readiness is turned into
    /// a pump task during dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::TooManyObjects`] when the id map cannot install
    /// the display resource.
    pub fn create_client(&mut self, transport: T) -> WlResult<WlClientId> {
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => {
                self.clients.push(None);
                self.generations.push(0);
                self.clients.len() - 1
            }
        };
        let generation = {
            let current = self.generations.get(slot).copied().unwrap_or(0);
            let generation = current.wrapping_add(1);
            if self.generations.len() <= slot {
                self.generations.resize(slot + 1, 0);
            }
            self.generations[slot] = generation;
            generation
        };
        let id = WlClientId { slot, generation };
        let connection = WlConnection::new(transport);
        let handle = connection.handle();
        let mut resources = WlMap::new(WlMapSide::Server);
        if let Err(err) = resources.insert_at(
            DISPLAY_RESOURCE_ID,
            WlResource {
                id: DISPLAY_RESOURCE_ID,
                interface: &DISPLAY_INTERFACE,
                version: DISPLAY_INTERFACE.version,
            },
        ) {
            self.free.push(slot);
            return Err(err);
        }
        let tasks = Rc::clone(&self.tasks);
        let source = self.event_loop.add_fd(
            handle,
            WlPollEvents::READABLE,
            move |_, _, events| {
                tasks.borrow_mut().push_back(Box::new(
                    move |display: &mut WlServerDisplay<T, P, C>| {
                        display.pump(id, events)
                    },
                ));
                0
            },
        );
        self.clients[slot] = Some(WlClient {
            id,
            connection,
            resources,
            error: false,
            source,
        });
        Ok(id)
    }

    /// Destroys `id`, removing its event source and dropping the
    /// transport.
    pub fn destroy_client(&mut self, id: WlClientId) {
        if let Some(client) = self.take_client(id) {
            self.retire_client(id.slot, client);
        }
    }

    /// Returns the client behind `id`.
    #[must_use]
    pub fn client(&self, id: WlClientId) -> Option<&WlClient<T>> {
        let client = self.clients.get(id.slot)?.as_ref()?;
        (client.id.generation == id.generation).then_some(client)
    }

    /// Mutably returns the client behind `id`.
    ///
    /// The client of the request currently being dispatched is not in its
    /// slot; handlers receive it as their argument instead.
    #[must_use]
    pub fn client_mut(&mut self, id: WlClientId) -> Option<&mut WlClient<T>> {
        let client = self.clients.get_mut(id.slot)?.as_mut()?;
        (client.id.generation == id.generation).then_some(client)
    }

    /// Returns the number of connected clients.
    #[must_use]
    pub fn client_count(&self) -> usize {
        self.clients.iter().filter(|slot| slot.is_some()).count()
    }

    /// Iterates over the connected clients with their handles.
    pub fn clients(&self) -> impl Iterator<Item = (WlClientId, &WlClient<T>)> {
        self.clients
            .iter()
            .filter_map(|slot| slot.as_ref().map(|client| (client.id, client)))
    }

    /// Iterates over the published globals with name, interface and
    /// version.
    pub fn globals(&self) -> impl Iterator<Item = (u32, &'static WlInterface, u32)> {
        self.globals
            .iter()
            .map(|global| (global.name, global.interface, global.version))
    }

    /// Queues an event on `resource` of `id`.
    ///
    /// This form addresses clients that are not dispatching a request; a
    /// handler of [`WlServerDisplay::add_request_handler`] sends events
    /// with [`WlClient::post_event`] on its client argument instead.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Disconnected`] when `id` is stale or its client
    /// is currently dispatching and the errors of [`WlClient::post_event`].
    pub fn post_event(
        &mut self,
        id: WlClientId,
        resource: u32,
        opcode: u32,
        args: Vec<WlArgument>,
    ) -> WlResult<()> {
        self.client_mut(id)
            .ok_or(WlError::Disconnected)?
            .post_event(resource, opcode, args)
    }

    /// Returns the next event serial of the display.
    pub fn alloc_serial(&mut self) -> u64 {
        let serial = self.next_serial;
        self.next_serial = serial.wrapping_add(1);
        serial
    }

    /// Writes every pending client buffer to its transport.
    ///
    /// Clients whose transport fails beyond a blocked write are destroyed.
    /// Matches `wl_display_flush_clients`.
    pub fn flush_clients(&mut self) {
        let mut dead = Vec::new();
        for slot in 0..self.clients.len() {
            let Some(client) = self.clients[slot].as_mut() else {
                continue;
            };
            if client
                .connection
                .flush()
                .is_err_and(|error| !matches!(error, WlError::WouldBlock))
            {
                dead.push(slot);
                continue;
            }
            let interest = if client.connection.wants_write() {
                WlPollEvents::READABLE.union(WlPollEvents::WRITABLE)
            } else {
                WlPollEvents::READABLE
            };
            let _ = self.event_loop.fd_update(client.source, interest);
        }
        for slot in dead {
            let client = match self.clients.get_mut(slot) {
                Some(entry) => entry.take(),
                None => None,
            };
            if let Some(client) = client {
                self.retire_client(slot, client);
            }
        }
    }

    /// Runs compositor tasks, waits on the event loop up to `timeout`,
    /// pumps the ready clients, runs the resulting tasks again and flushes
    /// every client.
    ///
    /// A `timeout` of `None` blocks until a source or an armed timer is
    /// ready.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Io`] when the poller fails.
    pub fn dispatch(&mut self, timeout: Option<Duration>) -> WlResult<()> {
        self.drain_tasks();
        self.event_loop.dispatch(timeout)?;
        self.drain_tasks();
        self.flush_clients();
        Ok(())
    }

    /// Dispatches until [`WlServerDisplay::terminate`] is called.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Io`] when the poller fails.
    pub fn run(&mut self) -> WlResult<()> {
        while self.running {
            self.dispatch(None)?;
        }
        Ok(())
    }

    /// Asks [`WlServerDisplay::run`] to return after the next dispatch.
    pub fn terminate(&mut self) {
        self.running = false;
    }

    fn add_global_inner(
        &mut self,
        interface: &'static WlInterface,
        version: u32,
        bind: Option<BindFn<T>>,
    ) -> WlResult<u32> {
        if version == 0 {
            return Err(WlError::invalid_argument(
                "version 0 is not a valid global version",
            ));
        }
        if version > interface.version {
            return Err(WlError::invalid_argument(format!(
                "version {version} exceeds the maximum version {} of {}",
                interface.version, interface.name
            )));
        }
        let name = self.next_global_name;
        self.next_global_name = name
            .checked_add(1)
            .ok_or_else(|| WlError::invalid_state("global names are exhausted"))?;
        self.globals.push(WlGlobal {
            name,
            interface,
            version,
            bind,
        });
        Ok(name)
    }

    fn drain_tasks(&mut self) {
        loop {
            let task = self.tasks.borrow_mut().pop_front();
            let Some(task) = task else {
                break;
            };
            task(self);
        }
    }

    fn take_client(&mut self, id: WlClientId) -> Option<WlClient<T>> {
        let current = self.clients.get(id.slot)?.as_ref()?;
        if current.id.generation != id.generation {
            return None;
        }
        self.clients[id.slot].take()
    }

    fn retire_client(&mut self, slot: usize, client: WlClient<T>) {
        let _ = self.event_loop.remove_source(client.source);
        if let Some(entry) = self.clients.get_mut(slot) {
            *entry = None;
        }
        if !self.free.contains(&slot) {
            self.free.push(slot);
        }
    }

    /// Handles the readiness of the transport of `id`.
    ///
    /// Mirrors `wl_client_connection_data`: hangs up destroy the client,
    /// writable flushes and readable pumps every complete request.
    fn pump(&mut self, id: WlClientId, events: WlPollEvents) {
        let Some(mut client) = self.take_client(id) else {
            return;
        };
        if events.contains(WlPollEvents::HANGUP) || events.contains(WlPollEvents::ERROR) {
            self.retire_client(id.slot, client);
            return;
        }
        if events.contains(WlPollEvents::WRITABLE)
            && client
                .connection
                .flush()
                .is_err_and(|error| !matches!(error, WlError::WouldBlock))
        {
            self.retire_client(id.slot, client);
            return;
        }
        if events.contains(WlPollEvents::READABLE) {
            loop {
                match client.connection.read() {
                    Ok(_) => {
                        self.process_requests(&mut client);
                        if client.error {
                            break;
                        }
                    }
                    Err(WlError::WouldBlock) => break,
                    Err(_) => {
                        self.retire_client(id.slot, client);
                        return;
                    }
                }
            }
        }
        if client.error {
            let _ = client.connection.flush();
            self.retire_client(id.slot, client);
            return;
        }
        self.put_client(id, client);
    }

    fn put_client(&mut self, id: WlClientId, client: WlClient<T>) {
        if let Some(entry) = self.clients.get_mut(id.slot)
            && entry.is_none()
        {
            *entry = Some(client);
        }
    }

    /// Decodes and dispatches every complete request of the client.
    fn process_requests(&mut self, client: &mut WlClient<T>) {
        loop {
            let Some((sender, opcode, size)) = client.connection.peek() else {
                return;
            };
            let size = size as usize;
            if size > MAX_MESSAGE_SIZE {
                client.post_error(
                    DISPLAY_RESOURCE_ID,
                    WlDisplayError::InvalidMethod.code(),
                    format!("message length {size} exceeds {MAX_MESSAGE_SIZE}"),
                );
                return;
            }
            if size < 8 {
                client.post_error(
                    DISPLAY_RESOURCE_ID,
                    WlDisplayError::InvalidMethod.code(),
                    "message shorter than its header",
                );
                return;
            }
            if client.connection.pending_input() < size {
                return;
            }
            let Some((interface, version)) = client
                .resources
                .lookup(sender)
                .map(|resource| (resource.interface, resource.version))
            else {
                client.post_error(
                    DISPLAY_RESOURCE_ID,
                    WlDisplayError::InvalidObject.code(),
                    format!("invalid object {sender}"),
                );
                return;
            };
            let Some(message) = interface.request_at(opcode) else {
                client.post_error(
                    DISPLAY_RESOURCE_ID,
                    WlDisplayError::InvalidMethod.code(),
                    format!("invalid method {opcode}, object {interface}#{sender}"),
                );
                return;
            };
            if version > 0 && version < message.since() {
                client.post_error(
                    DISPLAY_RESOURCE_ID,
                    WlDisplayError::InvalidMethod.code(),
                    format!(
                        "invalid version for {interface}#{sender}.{} ({version}, need at least {})",
                        message.name,
                        message.since()
                    ),
                );
                return;
            }
            let mut closure = match client.connection.demarshal(message) {
                Ok(closure) => closure,
                Err(WlError::WouldBlock) => return,
                Err(_) => {
                    client.post_error(
                        DISPLAY_RESOURCE_ID,
                        WlDisplayError::InvalidMethod.code(),
                        format!(
                            "invalid arguments for {interface}#{sender}.{}",
                            message.name
                        ),
                    );
                    return;
                }
            };
            if lookup_objects(&mut closure, &client.resources).is_err() {
                client.connection.release_argument_fds(&mut closure.args);
                client.post_error(
                    DISPLAY_RESOURCE_ID,
                    WlDisplayError::InvalidMethod.code(),
                    format!(
                        "invalid arguments for {interface}#{sender}.{}",
                        message.name
                    ),
                );
                return;
            }
            let opcode = closure.opcode;
            let mut args = closure.args;
            self.invoke_request(client, sender, opcode, &mut args);
            client.connection.release_argument_fds(&mut args);
            if client.error {
                return;
            }
        }
    }

    /// Dispatches one demarshalled request to the built-in implementations
    /// or to the registered handler of the interface.
    fn invoke_request(
        &mut self,
        client: &mut WlClient<T>,
        sender: u32,
        opcode: u32,
        args: &mut [WlArgument],
    ) {
        let Some(interface) = client.resource_interface(sender) else {
            client.post_error(
                DISPLAY_RESOURCE_ID,
                WlDisplayError::InvalidObject.code(),
                format!("invalid object {sender}"),
            );
            return;
        };
        if interface.equal(&DISPLAY_INTERFACE) {
            match opcode {
                DISPLAY_SYNC | DISPLAY_GET_REGISTRY => {
                    let Some(new_id) = args
                        .first()
                        .and_then(|arg0: &WlArgument| new_id_arg(Option::from(arg0)))
                    else {
                        post_invalid_arguments(client, sender, opcode, interface);
                        return;
                    };
                    if opcode == DISPLAY_SYNC {
                        self.handle_sync(client, new_id);
                    } else {
                        self.handle_get_registry(client, new_id);
                    }
                }
                _ => post_invalid_arguments(client, sender, opcode, interface),
            }
            return;
        }
        if interface.equal(&REGISTRY_INTERFACE) && opcode == REGISTRY_BIND {
            self.handle_bind(client, sender, args);
            return;
        }
        let Some(index) = self
            .handlers
            .iter()
            .position(|handler| handler.interface.equal(interface))
        else {
            client.post_error(
                DISPLAY_RESOURCE_ID,
                WlDisplayError::InvalidMethod.code(),
                format!("invalid method {opcode}, object {interface}#{sender}"),
            );
            return;
        };
        let Some(mut handler) = self.handlers[index].callback.take() else {
            return;
        };
        handler(client, client.id, sender, opcode, args);
        if let Some(slot) = self.handlers.get_mut(index)
            && slot.interface.equal(interface)
            && slot.callback.is_none()
        {
            slot.callback = Some(handler);
        }
    }

    /// Answers `wl_display.sync` with a completed callback.
    fn handle_sync(&mut self, client: &mut WlClient<T>, new_id: u32) {
        if let Err(error) = client.create_resource(
            new_id,
            &CALLBACK_INTERFACE,
            CALLBACK_INTERFACE.version,
        ) {
            post_create_failure(client, new_id, error);
            return;
        }
        let serial = self.alloc_serial();
        if client
            .post_event(new_id, CALLBACK_DONE, vec![WlArgument::Uint(serial as u32)])
            .is_err()
            || client.destroy_resource(new_id).is_err()
        {
            client.post_no_memory();
        }
    }

    /// Answers `wl_display.get_registry` and publishes every global.
    fn handle_get_registry(&mut self, client: &mut WlClient<T>, new_id: u32) {
        if let Err(error) = client.create_resource(
            new_id,
            &REGISTRY_INTERFACE,
            REGISTRY_INTERFACE.version,
        ) {
            post_create_failure(client, new_id, error);
            return;
        }
        for global in &self.globals {
            let args = vec![
                WlArgument::Uint(global.name),
                WlArgument::Str(Some(String::from(global.interface.name))),
                WlArgument::Uint(global.version),
            ];
            if client.post_event(new_id, REGISTRY_GLOBAL, args).is_err() {
                client.post_no_memory();
                return;
            }
        }
    }

    /// Validates `wl_registry.bind` and runs the bind of the global.
    fn handle_bind(
        &mut self,
        client: &mut WlClient<T>,
        registry: u32,
        args: &mut [WlArgument],
    ) {
        let interface_name = match args.get_mut(1) {
            Some(WlArgument::Str(slot)) => core::mem::take(slot),
            _ => None,
        };
        let (
            Some(WlArgument::Uint(name)),
            Some(WlArgument::Uint(version)),
            Some(WlArgument::NewId(new_id)),
        ) = (args.first(), args.get(2), args.get(3))
        else {
            post_invalid_arguments(client, registry, REGISTRY_BIND, &REGISTRY_INTERFACE);
            return;
        };
        let (name, version, new_id) = (*name, *version, *new_id);
        let requested = interface_name.as_deref().unwrap_or("");
        let Some(index) = self.globals.iter().position(|global| global.name == name)
        else {
            client.post_error(
                registry,
                WlDisplayError::InvalidObject.code(),
                format!("global {requested} ({name}) is unavailable"),
            );
            return;
        };
        let (global_interface, global_version) =
            (self.globals[index].interface, self.globals[index].version);
        if !global_interface.name.eq(requested) {
            client.post_error(
                registry,
                WlDisplayError::InvalidObject.code(),
                format!(
                    "invalid interface for global {name}: expected {}, got {requested}",
                    global_interface.name
                ),
            );
            return;
        }
        if version == 0 {
            client.post_error(
                registry,
                WlDisplayError::InvalidObject.code(),
                format!("invalid version for global {requested} ({name}): 0 is not a valid version"),
            );
            return;
        }
        if version > global_version {
            client.post_error(
                registry,
                WlDisplayError::InvalidObject.code(),
                format!(
                    "invalid version for global {requested} ({name}): expected at most {global_version}, got {version}"
                ),
            );
            return;
        }
        let client_id = client.id;
        if let Some(mut bind) = self.globals[index].bind.take() {
            bind(client, client_id, version, new_id);
            if let Some(position) =
                self.globals.iter().position(|global| global.name == name)
                && self.globals[position].bind.is_none()
            {
                self.globals[position].bind = Some(bind);
            }
        } else if let Err(error) =
            client.create_resource(new_id, global_interface, version)
        {
            post_create_failure(client, new_id, error);
        }
    }
}

fn new_id_arg(arg: Option<&WlArgument>) -> Option<u32> {
    match arg {
        Some(WlArgument::NewId(id)) => Some(*id),
        _ => None,
    }
}

fn post_invalid_arguments<T: WlTransport>(
    client: &mut WlClient<T>,
    sender: u32,
    opcode: u32,
    interface: &'static WlInterface,
) {
    let name = interface
        .request_at(opcode)
        .map_or("request", |message| message.name);
    client.post_error(
        DISPLAY_RESOURCE_ID,
        WlDisplayError::InvalidMethod.code(),
        format!("invalid arguments for {interface}#{sender}.{name}"),
    );
}

fn post_create_failure<T: WlTransport>(
    client: &mut WlClient<T>,
    id: u32,
    error: WlError,
) {
    match error {
        WlError::TooManyObjects => client.post_no_memory(),
        _ => client.post_error(
            DISPLAY_RESOURCE_ID,
            WlDisplayError::InvalidObject.code(),
            format!("invalid new id {id}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use crate::wl_conn::WlHandle;
    use crate::wl_evloop::WlPollEntry;
    use crate::wl_handle::{WlFd, WlMessage};

    use super::*;

    static POKE: WlMessage = WlMessage::new("poke", "u", &[None]);
    static PONG: WlMessage = WlMessage::new("pong", "u", &[None]);
    static TEST_INTERFACE: WlInterface = WlInterface::new("wl_test", 3, &[POKE], &[PONG]);
    static OTHER_INTERFACE: WlInterface = WlInterface::new("wl_other", 2, &[], &[]);

    /// Opcode of `wl_test.poke`, the only request of [`TEST_INTERFACE`].
    const POKE_OPCODE: u32 = 0;
    /// Opcode of `wl_test.pong`, the only event of [`TEST_INTERFACE`].
    const PONG_OPCODE: u32 = 0;
    /// Handle reported by the single test pipe.
    const PIPE_HANDLE: WlHandle = 7;

    /// State shared by the server transport, the poller and the test.
    #[derive(Default)]
    struct PipeState {
        input: Vec<u8>,
        input_pos: usize,
        output: Vec<u8>,
        output_pos: usize,
        peer_closed: bool,
        task_ran: bool,
        polls_after_task: usize,
    }

    /// Server end of an in-memory pipe.
    struct ServerPipe {
        state: Rc<RefCell<PipeState>>,
    }

    impl WlTransport for ServerPipe {
        fn recv(&mut self, buf: &mut [u8], _fds: &mut Vec<WlFd>) -> WlResult<usize> {
            let mut state = self.state.borrow_mut();
            if state.peer_closed {
                return Ok(0);
            }
            if state.input_pos >= state.input.len() {
                return Err(WlError::WouldBlock);
            }
            let available = state.input.len() - state.input_pos;
            let count = available.min(buf.len());
            let start = state.input_pos;
            buf[..count].copy_from_slice(&state.input[start..start + count]);
            state.input_pos += count;
            Ok(count)
        }

        fn send(&mut self, data: &[u8], _fds: &[WlFd]) -> WlResult<usize> {
            self.state.borrow_mut().output.extend_from_slice(data);
            Ok(data.len())
        }

        fn wait(
            &mut self,
            _timeout: Option<Duration>,
            mask: WlPollEvents,
        ) -> WlResult<WlPollEvents> {
            let state = self.state.borrow();
            let mut events = WlPollEvents::EMPTY;
            if state.peer_closed {
                events.insert(WlPollEvents::HANGUP);
            }
            if state.input_pos < state.input.len() {
                events.insert(WlPollEvents::READABLE);
            }
            Ok(events.intersection(mask))
        }

        fn handle(&self) -> WlHandle {
            PIPE_HANDLE
        }
    }

    /// Poller reporting the readiness of the single test pipe.
    ///
    /// Hangup is reported outside of the interest mask, like `poll(2)`,
    /// so the event loop can never miss a closed peer.
    struct PipePoller {
        state: Rc<RefCell<PipeState>>,
    }

    impl WlPoller for PipePoller {
        fn poll(
            &mut self,
            entries: &mut [WlPollEntry],
            _timeout: Option<Duration>,
        ) -> WlResult<usize> {
            let mut state = self.state.borrow_mut();
            if state.task_ran {
                state.polls_after_task += 1;
            }
            let mut ready = 0;
            for entry in entries.iter_mut() {
                let mut events = WlPollEvents::EMPTY;
                if state.peer_closed {
                    events.insert(WlPollEvents::HANGUP);
                }
                if state.input_pos < state.input.len() {
                    events.insert(WlPollEvents::READABLE);
                }
                if entry.interest.contains(WlPollEvents::WRITABLE) {
                    events.insert(WlPollEvents::WRITABLE);
                }
                entry.revents = events;
                if !events.is_empty() {
                    ready += 1;
                }
            }
            Ok(ready)
        }
    }

    /// Clock standing still at zero.
    struct TestClock;

    impl WlClock for TestClock {
        fn now_ms(&self) -> u64 {
            0
        }
    }

    type TestDisplay = WlServerDisplay<ServerPipe, PipePoller, TestClock>;

    /// Display with one connected client over an in-memory pipe.
    struct Fixture {
        display: TestDisplay,
        client_id: WlClientId,
        state: Rc<RefCell<PipeState>>,
    }

    impl Fixture {
        fn new() -> Self {
            Self::with_setup(|_| {})
        }

        fn with_setup(setup: impl FnOnce(&mut TestDisplay)) -> Self {
            let state = Rc::new(RefCell::new(PipeState::default()));
            let mut display = WlServerDisplay::new(
                PipePoller {
                    state: Rc::clone(&state),
                },
                TestClock,
            );
            setup(&mut display);
            let client_id = display
                .create_client(ServerPipe {
                    state: Rc::clone(&state),
                })
                .unwrap();
            Self {
                display,
                client_id,
                state,
            }
        }

        /// Queues raw request bytes for the server to read.
        fn send(&self, bytes: &[u8]) {
            self.state.borrow_mut().input.extend_from_slice(bytes);
        }

        /// Runs one non-blocking dispatch of the display.
        fn dispatch(&mut self) {
            self.display.dispatch(Some(Duration::ZERO)).unwrap();
        }

        /// Returns the events flushed since the previous call.
        fn take_messages(&self) -> Vec<u8> {
            let mut state = self.state.borrow_mut();
            let bytes = state.output[state.output_pos..].to_vec();
            state.output_pos = state.output.len();
            bytes
        }

        /// Closes the test end of the pipe.
        fn close_peer(&self) {
            self.state.borrow_mut().peer_closed = true;
        }
    }

    /// Builds a request whose arguments are 32 bit words only.
    fn fixed_request(sender: u32, opcode: u32, words: &[u32]) -> Vec<u8> {
        let size = 8 + 4 * words.len();
        let mut bytes = Vec::with_capacity(size);
        bytes.extend_from_slice(&sender.to_le_bytes());
        bytes.extend_from_slice(&(((size as u32) << 16) | opcode).to_le_bytes());
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// Builds a `wl_registry.bind` request.
    fn bind_request(
        registry: u32,
        name: u32,
        interface: &str,
        version: u32,
        id: u32,
    ) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&name.to_le_bytes());
        payload.extend_from_slice(&((interface.len() + 1) as u32).to_le_bytes());
        payload.extend_from_slice(interface.as_bytes());
        payload.push(0);
        while payload.len() % 4 != 0 {
            payload.push(0);
        }
        payload.extend_from_slice(&version.to_le_bytes());
        payload.extend_from_slice(&id.to_le_bytes());
        let size = 8 + payload.len();
        let mut bytes = Vec::with_capacity(size);
        bytes.extend_from_slice(&registry.to_le_bytes());
        bytes.extend_from_slice(&(((size as u32) << 16) | REGISTRY_BIND).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    /// Splits raw event bytes into sender, opcode and payload.
    fn split_messages(bytes: &[u8]) -> Vec<(u32, u32, Vec<u8>)> {
        let mut messages = Vec::new();
        let mut pos = 0;
        while pos + 8 <= bytes.len() {
            let sender =
                u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap_or([0; 4]));
            let header =
                u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap_or([0; 4]));
            let size = (header >> 16) as usize;
            let opcode = header & 0xffff;
            if size < 8 || pos + size > bytes.len() {
                break;
            }
            messages.push((sender, opcode, bytes[pos + 8..pos + size].to_vec()));
            pos += size;
        }
        messages
    }

    /// Reads a 32 bit word out of a message payload.
    fn word(payload: &[u8], at: usize) -> u32 {
        let end = at + 4;
        if end > payload.len() {
            return 0;
        }
        u32::from_le_bytes(payload[at..end].try_into().unwrap_or([0; 4]))
    }

    /// Decodes a `wl_registry.global` payload.
    fn parse_global(payload: &[u8]) -> (u32, String, u32) {
        let name = word(payload, 0);
        let length = word(payload, 4) as usize;
        let text_end = (8 + length).min(payload.len());
        let interface = if length > 0 && 8 < text_end {
            String::from_utf8_lossy(&payload[8..text_end - 1]).into_owned()
        } else {
            String::new()
        };
        let pos = (8 + length + 3) & !3;
        (name, interface, word(payload, pos))
    }

    /// Decodes a `wl_display.error` payload.
    fn parse_error(payload: &[u8]) -> (u32, u32, String) {
        let length = word(payload, 8) as usize;
        let text_end = (12 + length).min(payload.len());
        let message = if length > 0 && 12 < text_end {
            String::from_utf8_lossy(&payload[12..text_end - 1]).into_owned()
        } else {
            String::new()
        };
        (word(payload, 0), word(payload, 4), message)
    }

    #[test]
    fn sync_replies_with_done_and_delete_id() {
        let mut fixture = Fixture::new();
        fixture.send(&fixed_request(DISPLAY_RESOURCE_ID, DISPLAY_SYNC, &[2]));
        fixture.dispatch();

        let messages = split_messages(&fixture.take_messages());
        assert_eq!(messages.len(), 2);
        assert_eq!((messages[0].0, messages[0].1), (2, CALLBACK_DONE));
        assert_eq!(word(&messages[0].2, 0), 1);
        assert_eq!(
            (messages[1].0, messages[1].1),
            (DISPLAY_RESOURCE_ID, DISPLAY_DELETE_ID)
        );
        assert_eq!(word(&messages[1].2, 0), 2);

        let client = fixture.display.client(fixture.client_id).unwrap();
        assert_eq!(client.resource_count(), 1);
        assert!(client.resource_interface(2).is_none());
    }

    #[test]
    fn get_registry_publishes_globals() {
        let mut fixture = Fixture::with_setup(|display| {
            display.add_global(&TEST_INTERFACE, 3).unwrap();
            display.add_global(&OTHER_INTERFACE, 2).unwrap();
        });
        fixture.send(&fixed_request(
            DISPLAY_RESOURCE_ID,
            DISPLAY_GET_REGISTRY,
            &[2],
        ));
        fixture.dispatch();

        let messages = split_messages(&fixture.take_messages());
        assert_eq!(messages.len(), 2);
        assert_eq!((messages[0].0, messages[0].1), (2, REGISTRY_GLOBAL));
        assert_eq!(
            parse_global(&messages[0].2),
            (1, String::from("wl_test"), 3)
        );
        assert_eq!((messages[1].0, messages[1].1), (2, REGISTRY_GLOBAL));
        assert_eq!(
            parse_global(&messages[1].2),
            (2, String::from("wl_other"), 2)
        );
    }

    #[test]
    fn bind_invokes_the_bind_callback_and_creates_the_resource() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut fixture = Fixture::with_setup({
            let calls = Rc::clone(&calls);
            move |display| {
                display
                    .add_global_with(&TEST_INTERFACE, 3, move |client, _, version, id| {
                        calls.borrow_mut().push((version, id));
                        let created =
                            client.create_resource(id, &TEST_INTERFACE, version);
                        assert!(created.is_ok());
                    })
                    .unwrap();
            }
        });
        fixture.send(&fixed_request(
            DISPLAY_RESOURCE_ID,
            DISPLAY_GET_REGISTRY,
            &[2],
        ));
        fixture.send(&bind_request(2, 1, "wl_test", 2, 3));
        fixture.dispatch();

        assert_eq!(*calls.borrow(), [(2u32, 3u32)]);
        let messages = split_messages(&fixture.take_messages());
        assert_eq!(messages.len(), 1);
        assert_eq!((messages[0].0, messages[0].1), (2, REGISTRY_GLOBAL));
        assert_eq!(fixture.display.client_count(), 1);
        let client = fixture.display.client(fixture.client_id).unwrap();
        assert_eq!(
            client.resource_interface(3).map(|interface| interface.name),
            Some("wl_test")
        );
        assert_eq!(client.resource_version(3), Some(2));
    }

    #[test]
    fn bind_rejects_versions_above_the_global() {
        let mut fixture = Fixture::with_setup(|display| {
            display.add_global(&TEST_INTERFACE, 1).unwrap();
        });
        fixture.send(&fixed_request(
            DISPLAY_RESOURCE_ID,
            DISPLAY_GET_REGISTRY,
            &[2],
        ));
        fixture.send(&bind_request(2, 1, "wl_test", 2, 3));
        fixture.dispatch();

        let messages = split_messages(&fixture.take_messages());
        assert_eq!(messages.len(), 2);
        assert_eq!(
            (messages[1].0, messages[1].1),
            (DISPLAY_RESOURCE_ID, DISPLAY_ERROR)
        );
        let (object, code, message) = parse_error(&messages[1].2);
        assert_eq!(object, 2);
        assert_eq!(code, WlDisplayError::InvalidObject.code());
        assert!(message.contains("expected at most 1, got 2"), "{message}");
        assert_eq!(fixture.display.client_count(), 0);
    }

    #[test]
    fn unknown_object_posts_an_error_and_destroys_the_client() {
        let mut fixture = Fixture::new();
        fixture.send(&fixed_request(42, DISPLAY_SYNC, &[0]));
        fixture.dispatch();

        let messages = split_messages(&fixture.take_messages());
        assert_eq!(messages.len(), 1);
        assert_eq!(
            (messages[0].0, messages[0].1),
            (DISPLAY_RESOURCE_ID, DISPLAY_ERROR)
        );
        let (object, code, message) = parse_error(&messages[0].2);
        assert_eq!(object, DISPLAY_RESOURCE_ID);
        assert_eq!(code, WlDisplayError::InvalidObject.code());
        assert_eq!(message, "invalid object 42");
        assert_eq!(fixture.display.client_count(), 0);
    }

    #[test]
    fn request_handler_receives_requests_and_replies() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut fixture = Fixture::with_setup({
            let calls = Rc::clone(&calls);
            move |display| {
                display
                    .add_global_with(&TEST_INTERFACE, 1, |client, _, version, id| {
                        let created =
                            client.create_resource(id, &TEST_INTERFACE, version);
                        assert!(created.is_ok());
                    })
                    .unwrap();
                display
                    .add_request_handler(
                        &TEST_INTERFACE,
                        move |client, _, sender, opcode, args| {
                            let value = match args.first() {
                                Some(WlArgument::Uint(value)) => *value,
                                _ => 0,
                            };
                            calls.borrow_mut().push((sender, opcode, value));
                            let reply = vec![WlArgument::Uint(9)];
                            let sent = client.post_event(sender, PONG_OPCODE, reply);
                            assert!(sent.is_ok());
                        },
                    )
                    .unwrap();
            }
        });
        fixture.send(&fixed_request(
            DISPLAY_RESOURCE_ID,
            DISPLAY_GET_REGISTRY,
            &[2],
        ));
        fixture.send(&bind_request(2, 1, "wl_test", 1, 3));
        fixture.send(&fixed_request(3, POKE_OPCODE, &[7]));
        fixture.dispatch();

        assert_eq!(*calls.borrow(), [(3u32, POKE_OPCODE, 7u32)]);
        let messages = split_messages(&fixture.take_messages());
        assert_eq!(messages.len(), 2);
        assert_eq!((messages[0].0, messages[0].1), (2, REGISTRY_GLOBAL));
        assert_eq!((messages[1].0, messages[1].1), (3, PONG_OPCODE));
        assert_eq!(word(&messages[1].2, 0), 9);
    }

    #[test]
    fn remove_global_announces_global_remove() {
        let mut fixture = Fixture::with_setup(|display| {
            display.add_global(&TEST_INTERFACE, 3).unwrap();
        });
        fixture.send(&fixed_request(
            DISPLAY_RESOURCE_ID,
            DISPLAY_GET_REGISTRY,
            &[2],
        ));
        fixture.dispatch();
        let globals = split_messages(&fixture.take_messages());
        assert_eq!(globals.len(), 1);

        fixture.display.remove_global(1).unwrap();
        fixture.display.flush_clients();
        let messages = split_messages(&fixture.take_messages());
        assert_eq!(messages.len(), 1);
        assert_eq!((messages[0].0, messages[0].1), (2, REGISTRY_GLOBAL_REMOVE));
        assert_eq!(word(&messages[0].2, 0), 1);
    }

    #[test]
    fn closed_peer_removes_the_client() {
        let mut fixture = Fixture::new();
        assert_eq!(fixture.display.client_count(), 1);

        fixture.close_peer();
        fixture.dispatch();
        assert_eq!(fixture.display.client_count(), 0);
        fixture.dispatch();
    }

    #[test]
    fn scheduled_tasks_run_before_the_poll() {
        let mut fixture = Fixture::new();
        let state = Rc::clone(&fixture.state);
        fixture.display.tasks().push(move |_| {
            state.borrow_mut().task_ran = true;
        });
        fixture.dispatch();

        let state = fixture.state.borrow();
        assert!(state.task_ran);
        assert_eq!(state.polls_after_task, 1);
    }
}
