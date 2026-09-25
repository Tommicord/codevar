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

//! Client side of the wayland protocol
//!
//! [`WlClientDisplay`] owns a [`WlConnection`] over an arbitrary
//! [`WlTransport`], tracks every proxy in a [`WlMap`] and queues incoming
//! events until they are dispatched. Objects are addressed through
//! [`WlProxyId`] handles; listeners are closures instead of libwayland's
//! `void **` implementation tables.
//!
//! The threaded `wl_display_prepare_read` / `wl_display_read_events`
//! dance is not provided: dispatching always reads from the transport on
//! behalf of the caller.
//!
//! # Example
//!
//! ```
//! use codevar_wl_protocol::{WlClientDisplay, WlTransport, WlResult};
//! # fn demo<T: WlTransport>(transport: T) -> WlResult<()> {
//! let mut display = WlClientDisplay::connect(transport)?;
//! let _registry = display.get_registry()?;
//! display.roundtrip()?;
//! # Ok(())
//! # }
//! ```

use core::cell::Cell;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::wl_conn::{WlClosure, WlConnection, WlTransport, lookup_objects};
use crate::wl_error::{WlError, WlProtocolError, WlResult};
use crate::wl_handle::{
    CALLBACK_DONE, CALLBACK_INTERFACE, DISPLAY_DELETE_ID, DISPLAY_ERROR, DISPLAY_GET_REGISTRY,
    DISPLAY_INTERFACE, DISPLAY_SYNC, MAX_MESSAGE_SIZE, REGISTRY_BIND, REGISTRY_GLOBAL,
    REGISTRY_GLOBAL_REMOVE, REGISTRY_INTERFACE, SERVER_ID_START, WlArgType, WlArgument, WlInterface, WlMap,
    WlMapSide, WlMessage, WlObject, WlPollEvents,
};

/// Id of the display proxy, which is always `1`.
pub const DISPLAY_PROXY_ID: u32 = 1;

type ProxyListener<T> = Box<dyn FnMut(&mut WlClientDisplay<T>, u32, &mut [WlArgument]) -> i32>;

/// Handle of a proxy object owned by a [`WlClientDisplay`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WlProxyId(u32);

impl WlProxyId {
    /// Returns the protocol id of the proxy.
    #[inline]
    #[must_use]
    pub const fn id(self) -> u32 {
        self.0
    }
}

/// Event sent by the compositor on a `wl_registry` object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WlRegistryEvent {
    /// A new global was announced.
    Global {
        /// Numeric name of the global.
        name: u32,
        /// Interface name such as `wl_compositor`.
        interface: String,
        /// Highest supported version of the interface.
        version: u32,
    },
    /// A previously announced global disappeared.
    GlobalRemove {
        /// Numeric name of the removed global.
        name: u32,
    },
}

struct WlProxy<T: WlTransport> {
    id: u32,
    interface: &'static WlInterface,
    version: u32,
    serial: u64,
    id_deleted: bool,
    listener: Option<ProxyListener<T>>,
}

impl<T: WlTransport> WlObject for WlProxy<T> {
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

struct WlQueuedEvent {
    id: u32,
    serial: u64,
    closure: WlClosure,
}

/// Client connection mirroring `struct wl_display`.
///
/// The display tracks two queues: the internal display queue for events of
/// the display proxy itself (`wl_display.delete_id` and
/// `wl_display.error`) and the default queue for every other proxy.
pub struct WlClientDisplay<T: WlTransport> {
    connection: WlConnection<T>,
    proxies: WlMap<WlProxy<T>>,
    next_serial: u64,
    display_queue: VecDeque<WlQueuedEvent>,
    default_queue: VecDeque<WlQueuedEvent>,
    protocol_error: Option<WlProtocolError>,
}

impl<T: WlTransport> WlClientDisplay<T> {
    /// Creates a client display over `transport` and installs the display
    /// proxy with id [`DISPLAY_PROXY_ID`].
    ///
    /// # Errors
    ///
    /// Returns [`WlError::TooManyObjects`] when the object map cannot
    /// allocate the display id.
    pub fn connect(transport: T) -> WlResult<Self> {
        let mut display = Self {
            connection: WlConnection::new(transport),
            proxies: WlMap::new(WlMapSide::Client),
            next_serial: 1,
            display_queue: VecDeque::new(),
            default_queue: VecDeque::new(),
            protocol_error: None,
        };
        let serial = display.alloc_serial();
        display
            .proxies
            .insert_at(
                DISPLAY_PROXY_ID,
                WlProxy {
                    id: DISPLAY_PROXY_ID,
                    interface: &DISPLAY_INTERFACE,
                    version: DISPLAY_INTERFACE.version,
                    serial,
                    id_deleted: false,
                    listener: None,
                },
            )?;
        Ok(display)
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

    /// Unwraps the display, returning the connection.
    #[must_use]
    pub fn into_connection(self) -> WlConnection<T> {
        self.connection
    }

    /// Returns the protocol error reported by the compositor, if any.
    #[must_use]
    pub fn protocol_error(&self) -> Option<&WlProtocolError> {
        self.protocol_error
            .as_ref()
    }

    /// Returns the number of events waiting to be dispatched.
    #[must_use]
    pub fn pending_events(&self) -> usize {
        self.display_queue
            .len()
            + self
                .default_queue
                .len()
    }

    /// Returns `true` while `id` refers to a live proxy.
    #[must_use]
    pub fn is_alive(&self, id: WlProxyId) -> bool {
        self.proxies
            .lookup(id.0)
            .is_some()
    }

    /// Returns the interface of the proxy behind `id`.
    #[must_use]
    pub fn proxy_interface(&self, id: WlProxyId) -> Option<&'static WlInterface> {
        self.proxies
            .lookup(id.0)
            .map(WlProxy::interface)
    }

    /// Returns the version of the proxy behind `id`.
    #[must_use]
    pub fn proxy_version(&self, id: WlProxyId) -> Option<u32> {
        self.proxies
            .lookup(id.0)
            .map(|proxy| proxy.version)
    }

    /// Creates a `wl_registry` proxy and sends `wl_display.get_registry`.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::TooManyObjects`] when the id space is exhausted
    /// and the marshalling errors of the underlying connection.
    pub fn get_registry(&mut self) -> WlResult<WlProxyId> {
        let id = self.create_proxy(&REGISTRY_INTERFACE, REGISTRY_INTERFACE.version)?;
        let args = vec![WlArgument::NewId(id)];
        if let Err(error) = self.marshal(
            DISPLAY_PROXY_ID,
            DISPLAY_GET_REGISTRY,
            &DISPLAY_INTERFACE.requests[DISPLAY_GET_REGISTRY as usize],
            args,
        ) {
            self.proxies
                .remove(id);
            return Err(error);
        }
        Ok(WlProxyId(id))
    }

    /// Creates a `wl_callback` proxy and sends `wl_display.sync`.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::TooManyObjects`] when the id space is exhausted
    /// and the marshalling errors of the underlying connection.
    pub fn sync(&mut self) -> WlResult<WlProxyId> {
        let id = self.create_proxy(&CALLBACK_INTERFACE, CALLBACK_INTERFACE.version)?;
        let args = vec![WlArgument::NewId(id)];
        if let Err(error) = self.marshal(
            DISPLAY_PROXY_ID,
            DISPLAY_SYNC,
            &DISPLAY_INTERFACE.requests[DISPLAY_SYNC as usize],
            args,
        ) {
            self.proxies
                .remove(id);
            return Err(error);
        }
        Ok(WlProxyId(id))
    }

    /// Binds a global announced by `registry` and sends
    /// `wl_registry.bind`.
    ///
    /// The proxy behind the returned handle carries `interface` and
    /// `version`; `interface.name` is also the string sent on the wire.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidObject`] when `registry` is stale,
    /// [`WlError::InvalidArgument`] when it is not a registry and the
    /// marshalling errors of the underlying connection.
    pub fn registry_bind(
        &mut self,
        registry: WlProxyId,
        name: u32,
        interface: &'static WlInterface,
        version: u32,
    ) -> WlResult<WlProxyId> {
        self.require_interface(registry, REGISTRY_INTERFACE.name)?;
        let id = self.create_proxy(interface, version)?;
        let args = vec![
            WlArgument::Uint(name),
            WlArgument::Str(Some(String::from(interface.name))),
            WlArgument::Uint(version),
            WlArgument::NewId(id),
        ];
        if let Err(error) = self.marshal(
            registry.0,
            REGISTRY_BIND,
            &REGISTRY_INTERFACE.requests[REGISTRY_BIND as usize],
            args,
        ) {
            self.proxies
                .remove(id);
            return Err(error);
        }
        Ok(WlProxyId(id))
    }

    /// Installs a raw listener on `id`.
    ///
    /// The listener receives the opcode and the decoded arguments of every
    /// event of the proxy and returns a status code that is ignored, just
    /// like in libwayland. Arguments referring to objects destroyed since
    /// the event was queued arrive as the null object.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidObject`] when `id` is stale,
    /// [`WlError::InvalidState`] when `id` is the display proxy or already
    /// has a listener.
    pub fn add_listener<F>(&mut self, id: WlProxyId, listener: F) -> WlResult<()>
    where
        F: FnMut(&mut Self, u32, &mut [WlArgument]) -> i32 + 'static,
    {
        if id.0 == DISPLAY_PROXY_ID {
            return Err(WlError::invalid_state("the display proxy has no public listener"));
        }
        let proxy = self
            .proxies
            .lookup_mut(id.0)
            .ok_or(WlError::InvalidObject(id.0))?;
        if proxy
            .listener
            .is_some()
        {
            return Err(WlError::invalid_state("proxy already has a listener"));
        }
        proxy.listener = Some(Box::new(listener));
        Ok(())
    }

    /// Installs a typed listener on a `wl_registry` proxy.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidObject`] when `id` is stale,
    /// [`WlError::InvalidArgument`] when `id` is not a registry and
    /// [`WlError::InvalidState`] when it already has a listener.
    pub fn add_registry_listener<F>(&mut self, registry: WlProxyId, mut listener: F) -> WlResult<()>
    where
        F: FnMut(&mut Self, WlRegistryEvent) -> i32 + 'static,
    {
        self.require_interface(registry, REGISTRY_INTERFACE.name)?;
        self.add_listener(registry, move |display, opcode, args| match opcode {
            REGISTRY_GLOBAL => {
                let (Some(name), Some(version)) = (uint_arg(args, 0), uint_arg(args, 2)) else {
                    return -1;
                };
                let Some(interface) = str_arg(args, 1) else {
                    return -1;
                };
                listener(
                    display,
                    WlRegistryEvent::Global {
                        name,
                        interface,
                        version,
                    },
                )
            }
            REGISTRY_GLOBAL_REMOVE => match uint_arg(args, 0) {
                Some(name) => listener(display, WlRegistryEvent::GlobalRemove { name }),
                None => -1,
            },
            _ => -1,
        })
    }

    /// Installs a typed listener on a `wl_callback` proxy.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidObject`] when `id` is stale,
    /// [`WlError::InvalidArgument`] when `id` is not a callback and
    /// [`WlError::InvalidState`] when it already has a listener.
    pub fn add_callback_listener<F>(&mut self, callback: WlProxyId, mut listener: F) -> WlResult<()>
    where
        F: FnMut(&mut Self, u32) -> i32 + 'static,
    {
        self.require_interface(callback, CALLBACK_INTERFACE.name)?;
        self.add_listener(callback, move |display, opcode, args| match opcode {
            CALLBACK_DONE => match uint_arg(args, 0) {
                Some(data) => listener(display, data),
                None => -1,
            },
            _ => -1,
        })
    }

    /// Destroys the proxy behind `id` without sending a request.
    ///
    /// Client ids become zombies until `wl_display.delete_id` arrives,
    /// ids created by the compositor are dropped immediately.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::InvalidObject`] when `id` is stale and
    /// [`WlError::InvalidState`] when `id` is the display proxy.
    pub fn proxy_destroy(&mut self, id: WlProxyId) -> WlResult<()> {
        if id.0 == DISPLAY_PROXY_ID {
            return Err(WlError::invalid_state("the display proxy cannot be destroyed"));
        }
        let proxy = self
            .proxies
            .lookup(id.0)
            .ok_or(WlError::InvalidObject(id.0))?;
        let (interface, id_deleted) = (proxy.interface, proxy.id_deleted);
        if id.0 < SERVER_ID_START {
            if id_deleted {
                if !self
                    .proxies
                    .remove(id.0)
                {
                    return Err(WlError::InvalidObject(id.0));
                }
            } else {
                self.proxies
                    .make_zombie(id.0, interface)?;
            }
        } else {
            self.proxies
                .vacate_at(id.0)?;
        }
        Ok(())
    }

    /// Writes buffered requests to the transport.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::WouldBlock`] when the transport accepted only
    /// part of the buffer and [`WlError::Disconnected`] when the peer is
    /// gone.
    pub fn flush(&mut self) -> WlResult<usize> {
        let written = self
            .connection
            .flush()?;
        if self
            .connection
            .wants_write()
        {
            Err(WlError::WouldBlock)
        } else {
            Ok(written)
        }
    }

    /// Dispatches queued events, reading from the transport when none are
    /// pending.
    ///
    /// Buffered requests are flushed first. When no event is queued the
    /// call waits up to `timeout` for the compositor, reads every complete
    /// message and dispatches the display queue before the default queue.
    /// A `timeout` of `None` blocks indefinitely.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Protocol`] when the compositor reported a
    /// protocol error, [`WlError::Disconnected`] when the peer closed the
    /// connection and [`WlError::InvalidMethod`] when an event does not
    /// exist on the receiving interface.
    pub fn dispatch(&mut self, timeout: Option<Duration>) -> WlResult<usize> {
        if let Some(error) = &self.protocol_error {
            return Err(WlError::Protocol(error.clone()));
        }
        if self.pending_events() > 0 {
            return self.dispatch_pending();
        }
        if let Err(error) = self
            .connection
            .flush()
            && !matches!(error, WlError::WouldBlock)
        {
            return Err(error);
        }
        let wake = WlPollEvents::READABLE
            .union(WlPollEvents::HANGUP)
            .union(WlPollEvents::ERROR);
        loop {
            let events = self
                .connection
                .wait(timeout, WlPollEvents::READABLE)?;
            if events
                .intersection(wake)
                .is_empty()
            {
                return Ok(0);
            }
            match self
                .connection
                .read()
            {
                Ok(_) => {}
                Err(WlError::WouldBlock) => continue,
                Err(error) => return Err(error),
            }
            self.queue_events()?;
            if self.pending_events() > 0 {
                return self.dispatch_pending();
            }
        }
    }

    /// Dispatches the events that are already queued without reading from
    /// the transport.
    ///
    /// The display queue is drained before the default queue, matching
    /// `wl_display_dispatch_queue_pending`.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Protocol`] when the compositor reported a
    /// protocol error, also when it arrived during this call.
    pub fn dispatch_pending(&mut self) -> WlResult<usize> {
        if let Some(error) = &self.protocol_error {
            return Err(WlError::Protocol(error.clone()));
        }
        let mut count = 0usize;
        while let Some(event) = self
            .display_queue
            .pop_front()
        {
            self.invoke_event(event);
            count += 1;
        }
        while let Some(event) = self
            .default_queue
            .pop_front()
        {
            self.invoke_event(event);
            count += 1;
        }
        if let Some(error) = &self.protocol_error {
            return Err(WlError::Protocol(error.clone()));
        }
        Ok(count)
    }

    /// Blocks until the compositor processed all previously sent requests.
    ///
    /// A `wl_display.sync` round trip is issued and the call dispatches
    /// events until its callback arrives.
    ///
    /// # Errors
    ///
    /// Returns [`WlError::Protocol`] when the compositor reported a
    /// protocol error and [`WlError::Disconnected`] when the peer is gone.
    pub fn roundtrip(&mut self) -> WlResult<usize> {
        let done = Rc::new(Cell::new(false));
        let flag = Rc::clone(&done);
        let callback = self.sync()?;
        if let Err(error) = self.add_callback_listener(callback, move |_, _| {
            flag.set(true);
            0
        }) {
            self.proxies
                .remove(callback.0);
            return Err(error);
        }
        let mut total = 0usize;
        while !done.get() {
            match self.dispatch(None) {
                Ok(count) => total += count,
                Err(error) => {
                    let _ = self.proxy_destroy(callback);
                    return Err(error);
                }
            }
        }
        self.proxy_destroy(callback)?;
        Ok(total)
    }

    fn alloc_serial(&mut self) -> u64 {
        let serial = self.next_serial;
        self.next_serial = serial.wrapping_add(1);
        serial
    }

    fn create_proxy(&mut self, interface: &'static WlInterface, version: u32) -> WlResult<u32> {
        let serial = self.alloc_serial();
        let id = self
            .proxies
            .insert_new(WlProxy {
                id: 0,
                interface,
                version,
                serial,
                id_deleted: false,
                listener: None,
            })?;
        if let Some(proxy) = self
            .proxies
            .lookup_mut(id)
        {
            proxy.id = id;
        }
        Ok(id)
    }

    fn marshal(
        &mut self,
        sender: u32,
        opcode: u32,
        message: &'static WlMessage,
        args: Vec<WlArgument>,
    ) -> WlResult<()> {
        let mut closure = WlClosure::new(sender, opcode, message, args)?;
        self.connection
            .queue_closure(&mut closure)
    }

    fn require_interface(&self, id: WlProxyId, name: &str) -> WlResult<()> {
        let proxy = self
            .proxies
            .lookup(id.0)
            .ok_or(WlError::InvalidObject(id.0))?;
        if proxy
            .interface
            .name
            == name
        {
            Ok(())
        } else {
            Err(WlError::invalid_argument(format!(
                "expected a {} proxy, got {}",
                name,
                proxy
                    .interface
                    .name
            )))
        }
    }

    /// Reads every complete message of the input buffer into a queue.
    fn queue_events(&mut self) -> WlResult<()> {
        while let Some((sender, opcode, size)) = self
            .connection
            .peek()
        {
            let size = size as usize;
            if size > MAX_MESSAGE_SIZE {
                return Err(WlError::MessageTooBig(size));
            }
            if size < 8 {
                return Err(WlError::invalid_argument("message shorter than its header"));
            }
            if self
                .connection
                .pending_input()
                < size
            {
                return Ok(());
            }
            let interface = match self
                .proxies
                .lookup(sender)
            {
                Some(proxy) => proxy.interface,
                None => {
                    let fds = self
                        .proxies
                        .zombie_interface(sender)
                        .and_then(|interface| interface.event_at(opcode))
                        .map_or(0, WlMessage::fd_count);
                    self.connection
                        .skip_message(size, fds);
                    continue;
                }
            };
            let Some(message) = interface.event_at(opcode) else {
                return Err(WlError::InvalidMethod {
                    interface: interface.name,
                    opcode,
                });
            };
            let mut closure = match self
                .connection
                .demarshal(message)
            {
                Ok(closure) => closure,
                Err(WlError::WouldBlock) => return Ok(()),
                Err(error) => return Err(error),
            };
            self.create_event_proxies(&closure)?;
            lookup_objects(&mut closure, &self.proxies)?;
            let serial = self
                .proxies
                .lookup(sender)
                .map_or(0, |proxy| proxy.serial);
            let event = WlQueuedEvent {
                id: sender,
                serial,
                closure,
            };
            if sender == DISPLAY_PROXY_ID {
                self.display_queue
                    .push_back(event);
            } else {
                self.default_queue
                    .push_back(event);
            }
        }
        Ok(())
    }

    /// Creates proxies for `new_id` arguments of an incoming event.
    ///
    /// The new proxy inherits the version and the queue of the sender,
    /// like `wl_proxy_create_for_id` does.
    fn create_event_proxies(&mut self, closure: &WlClosure) -> WlResult<()> {
        for (arg, details) in closure
            .args
            .iter()
            .zip(
                closure
                    .message
                    .args(),
            )
        {
            if details
                .details
                .ty
                != WlArgType::NewId
            {
                continue;
            }
            let WlArgument::NewId(id) = arg else {
                continue;
            };
            if *id == 0 {
                continue;
            }
            let interface = details
                .interface
                .ok_or_else(|| WlError::invalid_argument("new id event without an interface"))?;
            let version = self
                .proxies
                .lookup(closure.sender_id)
                .map_or(1, |proxy| proxy.version);
            let serial = self.alloc_serial();
            self.proxies
                .insert_at(
                    *id,
                    WlProxy {
                        id: *id,
                        interface,
                        version,
                        serial,
                        id_deleted: false,
                        listener: None,
                    },
                )?;
        }
        Ok(())
    }

    /// Dispatches a single queued event to its listener.
    fn invoke_event(&mut self, event: WlQueuedEvent) {
        let WlQueuedEvent {
            id,
            serial,
            mut closure,
        } = event;
        let current = match self
            .proxies
            .lookup(id)
        {
            None => None,
            Some(proxy) if proxy.serial == serial => Some(proxy.serial),
            Some(_) => None,
        };
        if current.is_none() {
            self.connection
                .release_argument_fds(&mut closure.args);
            return;
        }
        self.refresh_object_args(&mut closure.args);
        let opcode = closure.opcode;
        let mut args = closure.args;
        if id == DISPLAY_PROXY_ID {
            match opcode {
                DISPLAY_ERROR => self.handle_display_error(&mut args),
                DISPLAY_DELETE_ID => {
                    if let Some(WlArgument::Uint(id)) = args.first() {
                        self.handle_delete_id(*id);
                    }
                }
                _ => {}
            }
        } else {
            let listener = self
                .proxies
                .lookup_mut(id)
                .and_then(WlProxy::take_listener);
            if let Some(mut listener) = listener {
                let _ = listener(self, opcode, &mut args);
                if let Some(proxy) = self
                    .proxies
                    .lookup_mut(id)
                    && proxy.serial == serial
                    && proxy
                        .listener
                        .is_none()
                {
                    proxy.listener = Some(listener);
                }
            }
        }
        self.connection
            .release_argument_fds(&mut args);
    }

    /// Rewrites arguments of objects destroyed after the event was queued.
    fn refresh_object_args(&mut self, args: &mut [WlArgument]) {
        for arg in args {
            if let WlArgument::Object(id) = arg
                && *id != 0
                && self
                    .proxies
                    .lookup(*id)
                    .is_none()
            {
                *id = 0;
            }
        }
    }

    /// Handles `wl_display.delete_id`.
    fn handle_delete_id(&mut self, id: u32) {
        if self
            .proxies
            .is_zombie(id)
        {
            self.proxies
                .remove(id);
        } else if let Some(proxy) = self
            .proxies
            .lookup_mut(id)
        {
            proxy.id_deleted = true;
        }
    }

    /// Handles `wl_display.error` by recording the protocol error.
    fn handle_display_error(&mut self, args: &mut [WlArgument]) {
        // The first argument is an object; unknown ids already arrived as
        // the null object, matching `display_handle_error` of libwayland.
        let object_id = match args.first() {
            Some(WlArgument::Object(id)) => *id,
            _ => uint_arg(args, 0).unwrap_or(0),
        };
        let code = uint_arg(args, 1).unwrap_or(0);
        let message = str_arg(args, 2).unwrap_or_default();
        let interface = self
            .proxies
            .lookup(object_id)
            .map(|proxy| {
                proxy
                    .interface
                    .name
            })
            .or_else(|| {
                self.proxies
                    .zombie_interface(object_id)
                    .map(|interface| interface.name)
            })
            .unwrap_or("wl_unknown");
        self.protocol_error = Some(WlProtocolError::new(code, object_id, interface, message));
    }
}

impl<T: WlTransport> WlProxy<T> {
    fn take_listener(&mut self) -> Option<ProxyListener<T>> {
        self.listener
            .take()
    }
}

fn uint_arg(args: &[WlArgument], index: usize) -> Option<u32> {
    match args.get(index) {
        Some(WlArgument::Uint(value)) => Some(*value),
        Some(WlArgument::NewId(value)) => Some(*value),
        _ => None,
    }
}

fn str_arg(args: &mut [WlArgument], index: usize) -> Option<String> {
    match args.get_mut(index)? {
        WlArgument::Str(slot) => core::mem::take(slot),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wl_conn::WlHandle;
    use crate::wl_handle::WlFd;

    /// In-memory transport that answers `wl_display.sync` automatically.
    #[derive(Default)]
    struct MemoryTransport {
        input: Vec<u8>,
        input_fds: Vec<WlFd>,
        input_pos: usize,
        output: Vec<u8>,
        output_pos: usize,
    }

    impl MemoryTransport {
        fn with_input(bytes: &[u8]) -> Self {
            Self {
                input: bytes.to_vec(),
                ..Self::default()
            }
        }

        /// Answers every `wl_display.sync` request found in the output.
        fn respond(&mut self) {
            while self.output_pos + 12
                <= self
                    .output
                    .len()
            {
                let bytes = &self.output[self.output_pos..];
                let sender = u32::from_le_bytes(
                    bytes[0..4]
                        .try_into()
                        .unwrap_or([0; 4]),
                );
                let header = u32::from_le_bytes(
                    bytes[4..8]
                        .try_into()
                        .unwrap_or([0; 4]),
                );
                let size = (header >> 16) as usize;
                let opcode = header & 0xffff;
                if size < 12 || size > bytes.len() {
                    break;
                }
                if sender == DISPLAY_PROXY_ID && opcode == DISPLAY_SYNC {
                    let callback = u32::from_le_bytes(
                        bytes[8..12]
                            .try_into()
                            .unwrap_or([0; 4]),
                    );
                    self.input
                        .extend_from_slice(&callback.to_le_bytes());
                    self.input
                        .extend_from_slice(&((12u32 << 16) | CALLBACK_DONE).to_le_bytes());
                    self.input
                        .extend_from_slice(&0u32.to_le_bytes());
                }
                self.output_pos += size;
            }
        }
    }

    impl WlTransport for MemoryTransport {
        fn recv(&mut self, buf: &mut [u8], fds: &mut Vec<WlFd>) -> WlResult<usize> {
            if self.input_pos
                >= self
                    .input
                    .len()
            {
                return Err(WlError::WouldBlock);
            }
            let available = self
                .input
                .len()
                - self.input_pos;
            let count = available.min(buf.len());
            let start = self.input_pos;
            buf[..count].copy_from_slice(&self.input[start..start + count]);
            self.input_pos += count;
            fds.append(&mut self.input_fds);
            Ok(count)
        }

        fn send(&mut self, data: &[u8], _fds: &[WlFd]) -> WlResult<usize> {
            self.output
                .extend_from_slice(data);
            self.respond();
            Ok(data.len())
        }

        fn wait(&mut self, _timeout: Option<Duration>, mask: WlPollEvents) -> WlResult<WlPollEvents> {
            let mut events = WlPollEvents::EMPTY;
            if self.input_pos
                < self
                    .input
                    .len()
            {
                events.insert(WlPollEvents::READABLE);
            }
            if self.output_pos
                < self
                    .output
                    .len()
            {
                events.insert(WlPollEvents::WRITABLE);
            }
            Ok(events.intersection(mask))
        }

        fn handle(&self) -> WlHandle {
            0
        }
    }

    /// Builds the bytes of a `wl_registry.global` event.
    fn global_event(name: u32, interface: &str, version: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_le_bytes());
        let payload = 4 + 4 + 4 + interface.len() + 1;
        let padded = (payload + 3) & !3;
        let size = 8 + padded;
        bytes.extend_from_slice(&((size as u32) << 16).to_le_bytes());
        bytes.extend_from_slice(&name.to_le_bytes());
        bytes.extend_from_slice(&((interface.len() + 1) as u32).to_le_bytes());
        bytes.extend_from_slice(interface.as_bytes());
        bytes.push(0);
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes
    }

    /// Builds the bytes of a `wl_display.delete_id` event.
    fn delete_id_event(id: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&DISPLAY_PROXY_ID.to_le_bytes());
        bytes.extend_from_slice(&((12u32 << 16) | DISPLAY_DELETE_ID).to_le_bytes());
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes
    }

    /// Builds the bytes of a `wl_display.error` event.
    fn display_error_event(object: u32, code: u32, message: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&DISPLAY_PROXY_ID.to_le_bytes());
        let payload = 4 + 4 + 4 + message.len() + 1;
        let padded = (payload + 3) & !3;
        let size = 8 + padded;
        bytes.extend_from_slice(&((size as u32) << 16).to_le_bytes());
        bytes.extend_from_slice(&object.to_le_bytes());
        bytes.extend_from_slice(&code.to_le_bytes());
        bytes.extend_from_slice(&((message.len() + 1) as u32).to_le_bytes());
        bytes.extend_from_slice(message.as_bytes());
        bytes.push(0);
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        bytes
    }

    #[test]
    fn creates_registry_and_encodes_requests() {
        let mut display = WlClientDisplay::connect(MemoryTransport::default()).unwrap();
        let registry = display
            .get_registry()
            .unwrap();
        assert_eq!(registry.id(), 2);
        assert_eq!(
            display
                .proxy_interface(registry)
                .map(|interface| interface.name),
            Some("wl_registry")
        );
        assert!(
            display
                .connection()
                .pending_output()
                > 0
        );
        assert!(
            display
                .flush()
                .is_ok()
        );

        let callback = display
            .sync()
            .unwrap();
        assert_eq!(callback.id(), 3);
        assert!(
            display
                .connection()
                .pending_output()
                > 0
        );
    }

    #[test]
    fn rejects_listener_misuse() {
        let mut display = WlClientDisplay::connect(MemoryTransport::default()).unwrap();
        let registry = display
            .get_registry()
            .unwrap();
        display
            .add_registry_listener(registry, |_, _| 0)
            .unwrap();
        assert!(
            display
                .add_registry_listener(registry, |_, _| 0)
                .is_err()
        );
        assert!(
            display
                .add_listener(WlProxyId(DISPLAY_PROXY_ID), |_, _, _| 0)
                .is_err()
        );
        assert!(
            display
                .add_registry_listener(WlProxyId(9), |_, _| 0)
                .is_err()
        );
        assert!(
            display
                .proxy_destroy(WlProxyId(DISPLAY_PROXY_ID))
                .is_err()
        );
        assert!(
            display
                .proxy_destroy(WlProxyId(9))
                .is_err()
        );
    }

    #[test]
    fn dispatches_registry_globals() {
        let event = global_event(1, "wl_seat", 3);
        let transport = MemoryTransport::with_input(&event);
        let mut display = WlClientDisplay::connect(transport).unwrap();
        let registry = display
            .get_registry()
            .unwrap();
        let seen = Rc::new(Cell::new(None));
        let slot = Rc::clone(&seen);
        display
            .add_registry_listener(registry, move |_, event| {
                slot.set(Some(event));
                0
            })
            .unwrap();

        let dispatched = display
            .dispatch(Some(Duration::ZERO))
            .unwrap();
        assert_eq!(dispatched, 1);
        assert_eq!(
            seen.take(),
            Some(WlRegistryEvent::Global {
                name: 1,
                interface: String::from("wl_seat"),
                version: 3,
            })
        );
        assert_eq!(display.pending_events(), 0);
    }

    #[test]
    fn delete_id_frees_zombies() {
        let transport = MemoryTransport::with_input(&delete_id_event(2));
        let mut display = WlClientDisplay::connect(transport).unwrap();
        let registry = display
            .get_registry()
            .unwrap();
        display
            .proxy_destroy(registry)
            .unwrap();
        assert!(!display.is_alive(registry));

        // The zombie still owns id 2, so the next proxy takes id 3.
        let second = display
            .get_registry()
            .unwrap();
        assert_eq!(second.id(), 3);

        display
            .dispatch(Some(Duration::ZERO))
            .unwrap();
        let third = display
            .get_registry()
            .unwrap();
        assert_eq!(third.id(), 2);
    }

    #[test]
    fn reports_protocol_errors() {
        // The error refers to the display object, the only object a fresh
        // client knows.
        let event = display_error_event(DISPLAY_PROXY_ID, 2, "no memory");
        let transport = MemoryTransport::with_input(&event);
        let mut display = WlClientDisplay::connect(transport).unwrap();

        let error = display
            .dispatch(Some(Duration::ZERO))
            .unwrap_err();
        let WlError::Protocol(protocol) = &error else {
            panic!("expected a protocol error, got {error:?}");
        };
        assert_eq!(protocol.code, 2);
        assert_eq!(protocol.message, "no memory");
        assert!(
            display
                .protocol_error()
                .is_some()
        );
        assert!(matches!(display.dispatch(None), Err(WlError::Protocol(_))));
    }

    #[test]
    fn roundtrip_waits_for_the_sync_callback() {
        let mut display = WlClientDisplay::connect(MemoryTransport::default()).unwrap();
        let dispatched = display
            .roundtrip()
            .unwrap();
        assert!(dispatched >= 1);
        assert!(!display.is_alive(WlProxyId(2)));
        assert!(
            !display
                .connection()
                .is_disconnected()
        );
    }

    #[test]
    fn binds_globals_through_the_registry() {
        static TEST_INTERFACE: WlInterface = WlInterface::new("wl_test", 1, &[], &[]);

        let mut display = WlClientDisplay::connect(MemoryTransport::default()).unwrap();
        let registry = display
            .get_registry()
            .unwrap();
        let bound = display
            .registry_bind(registry, 7, &TEST_INTERFACE, 1)
            .unwrap();
        assert_eq!(bound.id(), 3);
        assert_eq!(display.proxy_version(bound), Some(1));
        assert_eq!(
            display
                .proxy_interface(bound)
                .map(|interface| interface.name),
            Some("wl_test")
        );
        assert!(
            display
                .registry_bind(WlProxyId(9), 1, &TEST_INTERFACE, 1)
                .is_err()
        );
        assert!(
            display
                .flush()
                .is_ok()
        );
    }
}
