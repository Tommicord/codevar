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

//! Win32 window manager for the launcher.
//!
//! The backend drives the classic Win32 message loop. The window
//! class is registered once per process, the window is created on the
//! thread that owns it, and [`Win32Window::poll`] pumps the message
//! queue of that thread. The messages the launcher cares about are
//! translated into [`WndEvent`]s: `WM_SIZE` becomes
//! [`WndEvent::Resized`], `WM_CLOSE` becomes
//! [`WndEvent::CloseRequested`] and `WM_PAINT` becomes
//! [`WndEvent::RedrawRequested`].
//!
//! The window has no drawing surface yet: the redraw event asks the
//! driver for a new frame instead of painting into the device
//! context, and the window procedure only validates the update region
//! so the system stops asking for a repaint.
//!
//! The class outlives the windows that use it, so it is registered
//! when the first window opens and released again when the last one
//! closes. A process-exit hook is not available in this `no_std`
//! runtime, so releasing the class with the last window is the
//! closest equivalent.
//!
//! The application id of [`WndConfig`] is a Wayland concept and is
//! ignored here: Win32 identifies a window by its handle.

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::sync::atomic::{AtomicU16, AtomicU32, AtomicUsize, Ordering};

use codevar_time_core::TimeDuration;
use windows::Win32::Foundation::{
    HINSTANCE, HWND, LPARAM, LRESULT, RECT, WAIT_OBJECT_0, WAIT_TIMEOUT, WPARAM,
};
use windows::Win32::Graphics::Gdi::{
    BeginPaint, EndPaint, GET_STOCK_OBJECT_FLAGS, GetStockObject, HBRUSH, PAINTSTRUCT,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRectEx, CS_HREDRAW, CS_VREDRAW, CW_USEDEFAULT, CreateWindowExW, DefWindowProcW,
    DestroyWindow, DispatchMessageW, GetClientRect, HCURSOR, HICON, IDC_ARROW, IsWindow, LoadCursorW, MSG,
    MWMO_INPUTAVAILABLE, MsgWaitForMultipleObjectsEx, PM_REMOVE, PeekMessageW, QS_ALLINPUT, RegisterClassExW,
    SW_SHOW, SetWindowTextW, ShowWindow, TranslateMessage, UnregisterClassW, WM_CLOSE, WM_PAINT, WM_QUIT,
    WM_SIZE, WNDCLASSEXW, WS_CLIPCHILDREN, WS_EX_APPWINDOW, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};
use windows::core::PCWSTR;

use crate::wnd::{Window, WndBackend, WndBackendId, WndConfig, WndError, WndEvent, WndSize};

/// The name of the window class, in UTF-16 with a terminating zero.
///
/// The system keeps the name of a registered class, so it lives in a
/// static instead of on the stack.
const CLASS_NAME: &[u16] = &[
    0x0043, 0x006f, 0x0064, 0x0065, 0x0076, 0x0061, 0x0072, 0x004c, 0x0061, 0x0075, 0x006e, 0x0063, 0x0068,
    0x0065, 0x0072, 0x0000,
];

/// The stock object index of `DEFAULT_BRUSH` (`WHITE_BRUSH + 1`).
const DEFAULT_BRUSH: i32 = 1;

/// The wait that blocks until a message arrives.
const INFINITE_WAIT: u32 = u32::MAX;

/// The largest wait the message pump accepts, in milliseconds.
///
/// One millisecond below [`INFINITE_WAIT`], so a long timeout is never
/// mistaken for "wait forever".
const MAX_WAIT_MS: u32 = u32::MAX - 1;

/// The window class shared by every window of the process.
struct ClassSlot {
    /// Serializes the registration against its release.
    lock: spin::Mutex<()>,
    /// The handle of the running image, zero when it was not found.
    instance: AtomicUsize,
    /// The error the image lookup failed with, zero when it worked.
    instance_error: AtomicU32,
    /// The atom of the registered class, zero while it is released.
    atom: AtomicU16,
    /// The number of windows that use the class.
    live: AtomicUsize,
}

impl ClassSlot {
    /// Prepares the slot and resolves the handle of the running image.
    fn new() -> Self {
        // Safety: a null module name asks for the handle of the
        // running image, which the system always knows.
        let (instance, error) = match unsafe { GetModuleHandleW(PCWSTR::null()) } {
            Ok(module) => (module.0 as usize, 0),
            Err(error) => (0, error.code().0 as u32),
        };
        Self {
            lock: spin::Mutex::new(()),
            instance: AtomicUsize::new(instance),
            instance_error: AtomicU32::new(error),
            atom: AtomicU16::new(0),
            live: AtomicUsize::new(0),
        }
    }

    /// Returns the handle of the running image.
    fn instance(&self) -> HINSTANCE {
        HINSTANCE(self.instance.load(Ordering::Acquire) as *mut c_void)
    }

    /// Returns the error the image lookup failed with.
    fn instance_error(&self) -> u32 {
        self.instance_error.load(Ordering::Acquire)
    }
}

/// The window class of the process, prepared once.
static CLASS: spin::Once<ClassSlot> = spin::Once::new();

/// Returns the window class of the process.
fn class_slot() -> &'static ClassSlot {
    CLASS.call_once(ClassSlot::new)
}

/// Registers the window class when it is not registered yet.
///
/// # Errors
///
/// Returns a [`WndError`] when the handle of the running image could
/// not be resolved or when the system refused the class.
fn ensure_class() -> Result<u16, WndError> {
    let slot = class_slot();
    let _guard = slot.lock.lock();
    let atom = slot.atom.load(Ordering::Acquire);
    if atom != 0 {
        return Ok(atom);
    }
    let instance = slot.instance();
    if instance.is_invalid() {
        return Err(WndError::backend(
            WndBackendId::Win32,
            format!(
                "the running image could not be found (error {})",
                slot.instance_error()
            ),
        ));
    }
    let class = WNDCLASSEXW {
        cbSize: core::mem::size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(wnd_proc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: instance,
        hIcon: HICON::default(),
        hCursor: cursor(),
        hbrBackground: background_brush(),
        lpszMenuName: PCWSTR::null(),
        lpszClassName: PCWSTR(CLASS_NAME.as_ptr()),
        hIconSm: HICON::default(),
    };
    // Safety: the class is fully initialized, its name points at a
    // static buffer and the procedure outlives the class.
    let atom = unsafe { RegisterClassExW(&class) };
    if atom == 0 {
        return Err(WndError::backend(
            WndBackendId::Win32,
            "the window class was refused",
        ));
    }
    slot.atom.store(atom, Ordering::Release);
    Ok(atom)
}

/// Releases the window class once the last window using it closed.
fn release_class() {
    let slot = class_slot();
    let _guard = slot.lock.lock();
    let atom = slot.atom.swap(0, Ordering::AcqRel);
    if atom == 0 {
        return;
    }
    // Safety: the atom was registered by `ensure_class` with the
    // handle of the running image and no window uses the class any
    // more.
    let _ = unsafe { UnregisterClassW(PCWSTR(CLASS_NAME.as_ptr()), Some(slot.instance())) };
}

/// Loads the arrow cursor of the system.
fn cursor() -> HCURSOR {
    // Safety: `IDC_ARROW` is a system resource identifier and not a
    // pointer, and a null instance asks for the shared cursor.
    unsafe { LoadCursorW(None, IDC_ARROW) }.unwrap_or_default()
}

/// Returns the stock brush the window background is painted with.
fn background_brush() -> HBRUSH {
    // Safety: stock objects are always available and need no cleanup.
    let stock = unsafe { GetStockObject(GET_STOCK_OBJECT_FLAGS(DEFAULT_BRUSH)) };
    HBRUSH(stock.0)
}

/// Encodes `text` as a null terminated UTF-16 buffer.
fn wide_string(text: &str) -> Vec<u16> {
    let mut buffer: Vec<u16> = text.encode_utf16().collect();
    buffer.push(0);
    buffer
}

/// Converts a timeout into the milliseconds the message pump waits for.
fn wait_millis(timeout: TimeDuration) -> u32 {
    let millis = timeout.as_millis();
    if millis >= u128::from(MAX_WAIT_MS) {
        return MAX_WAIT_MS;
    }
    millis as u32
}

/// The lifecycle state of a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WinState {
    /// The window is alive and can be polled.
    Open,
    /// The window was closed but not destroyed yet.
    Closed,
    /// The window was destroyed and released.
    Destroyed,
}

/// A window on the Win32 desktop.
///
/// The window owns its `HWND` and destroys it when it is dropped.
pub struct Win32Window {
    /// The handle of the window.
    hwnd: HWND,
    /// The size the caller asked for, used until the first `WM_SIZE`.
    requested: WndSize,
    /// The size the window reported, `None` before the first `WM_SIZE`.
    reported: Option<WndSize>,
    /// The lifecycle state of the window.
    state: WinState,
}

impl Win32Window {
    /// Opens a window matching `config`.
    ///
    /// The requested geometry is the size of the client area; the
    /// frame of the window is added on top of it. The application id
    /// of the configuration is ignored.
    ///
    /// # Errors
    ///
    /// Returns a [`WndError`] when the window class could not be
    /// registered, when the frame of the window could not be measured
    /// or when the window could not be created.
    pub fn open(config: &WndConfig) -> Result<Self, WndError> {
        ensure_class()?;
        let slot = class_slot();
        let instance = slot.instance();
        let mut frame = RECT {
            left: 0,
            top: 0,
            right: config.geom.width,
            bottom: config.geom.height,
        };
        // Safety: `frame` is a valid, writable rectangle.
        unsafe { AdjustWindowRectEx(&mut frame, WS_OVERLAPPEDWINDOW, false, WS_EX_APPWINDOW) }.map_err(
            |error| {
                WndError::backend(
                    WndBackendId::Win32,
                    format!("the window frame could not be measured: {error}"),
                )
            },
        )?;
        let title = wide_string(&config.title);
        // Safety: the class is registered, the title is a null
        // terminated UTF-16 buffer that outlives the call and the
        // instance handle is the one the class was registered with.
        let hwnd = unsafe {
            CreateWindowExW(
                WS_EX_APPWINDOW,
                PCWSTR(CLASS_NAME.as_ptr()),
                PCWSTR(title.as_ptr()),
                WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_VISIBLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                frame.right - frame.left,
                frame.bottom - frame.top,
                None,
                None,
                Some(instance),
                None,
            )
        }
        .map_err(|error| {
            WndError::backend(
                WndBackendId::Win32,
                format!("the window could not be created: {error}"),
            )
        })?;
        // Safety: the window was just created, so it can be shown.
        let _ = unsafe { ShowWindow(hwnd, SW_SHOW) };
        slot.live.fetch_add(1, Ordering::AcqRel);
        Ok(Self {
            hwnd,
            requested: config.geom,
            reported: None,
            state: WinState::Open,
        })
    }

    /// Returns `true` while the window is alive.
    #[must_use]
    pub fn is_open(&self) -> bool {
        matches!(self.state, WinState::Open) && unsafe { IsWindow(Some(self.hwnd)) }.as_bool()
    }

    /// Destroys the window; later polls do nothing.
    pub fn close(&mut self) {
        self.destroy();
    }

    /// Returns the size of the client area in pixels.
    #[must_use]
    pub fn size(&self) -> WndSize {
        self.client_size()
            .or(self.reported)
            .unwrap_or(self.requested)
    }

    /// Returns the size of the client area, or `None` when the window
    /// does not answer any more.
    fn client_size(&self) -> Option<WndSize> {
        let mut rect = RECT::default();
        // Safety: `rect` is a valid, writable rectangle and the handle
        // is alive as long as the window was not destroyed.
        unsafe { GetClientRect(self.hwnd, &mut rect) }.ok()?;
        Some(WndSize::new(rect.right - rect.left, rect.bottom - rect.top))
    }

    /// Destroys the window handle and releases the class when it was
    /// the last window of the process.
    fn destroy(&mut self) {
        if matches!(self.state, WinState::Destroyed) {
            return;
        }
        self.state = WinState::Destroyed;
        // Safety: the handle was created by `open` and is destroyed
        // exactly once.
        let _ = unsafe { DestroyWindow(self.hwnd) };
        let slot = class_slot();
        if slot.live.fetch_sub(1, Ordering::AcqRel) == 1 {
            release_class();
        }
    }

    /// Pumps the message queue of the thread for at most `timeout` and
    /// reports the window events to `sink`.
    ///
    /// A `timeout` of `None` waits until at least one message was
    /// dispatched.
    ///
    /// # Errors
    ///
    /// Returns a [`WndError`] when the message pump failed.
    fn pump(
        &mut self,
        timeout: Option<TimeDuration>,
        sink: &mut dyn FnMut(WndEvent),
    ) -> Result<usize, WndError> {
        if !matches!(self.state, WinState::Open) {
            return Ok(0);
        }
        let blocking = timeout.is_none();
        let wait = timeout.map_or(INFINITE_WAIT, wait_millis);
        let mut dispatched = 0usize;
        loop {
            // Safety: the call only waits on the message queue of the
            // calling thread and reports whether it holds a message.
            let event = unsafe { MsgWaitForMultipleObjectsEx(None, wait, QS_ALLINPUT, MWMO_INPUTAVAILABLE) };
            if event == WAIT_TIMEOUT {
                break;
            }
            if event != WAIT_OBJECT_0 {
                return Err(WndError::backend(
                    WndBackendId::Win32,
                    format!("the message pump failed with the wait event {}", event.0),
                ));
            }
            let drained = self.drain(sink);
            dispatched += drained;
            if drained > 0 || !blocking {
                break;
            }
            // The queue reported input that was not posted yet: wait
            // for the message itself.
        }
        Ok(dispatched)
    }

    /// Removes every posted message and reports the window events.
    fn drain(&mut self, sink: &mut dyn FnMut(WndEvent)) -> usize {
        let mut dispatched = 0usize;
        loop {
            let mut message = MSG::default();
            // Safety: `message` is a valid, writable message.
            if !unsafe { PeekMessageW(&mut message, None, 0, 0, PM_REMOVE) }.as_bool() {
                break;
            }
            dispatched += 1;
            // Safety: the message was filled by `PeekMessageW` and the
            // window procedure of the launcher copes with it.
            unsafe {
                let _ = TranslateMessage(&message);
                DispatchMessageW(&message);
            }
            match message.message {
                WM_SIZE => {
                    let size = self.client_size().unwrap_or(self.requested);
                    // A minimized window has no client area to draw in.
                    if size.width > 0 && size.height > 0 {
                        self.reported = Some(size);
                        sink(WndEvent::Resized(size));
                    }
                }
                WM_CLOSE => {
                    self.state = WinState::Closed;
                    sink(WndEvent::CloseRequested);
                }
                WM_PAINT => sink(WndEvent::RedrawRequested),
                WM_QUIT => {
                    self.state = WinState::Closed;
                    sink(WndEvent::CloseRequested);
                }
                _ => {}
            }
            if !matches!(self.state, WinState::Open) {
                break;
            }
        }
        dispatched
    }
}

impl Drop for Win32Window {
    fn drop(&mut self) {
        self.destroy();
    }
}

impl Window for Win32Window {
    fn backend(&self) -> WndBackendId {
        WndBackendId::Win32
    }

    fn is_open(&self) -> bool {
        Win32Window::is_open(self)
    }

    fn close(&mut self) {
        Win32Window::close(self);
    }

    fn size(&self) -> WndSize {
        Win32Window::size(self)
    }

    fn set_title(&mut self, title: &str) -> Result<(), WndError> {
        if !matches!(self.state, WinState::Open) {
            return Ok(());
        }
        let wide = wide_string(title);
        // Safety: `wide` is a null terminated UTF-16 buffer that
        // outlives the call and the handle is alive.
        unsafe { SetWindowTextW(self.hwnd, PCWSTR(wide.as_ptr())) }.map_err(|error| {
            WndError::backend(
                WndBackendId::Win32,
                format!("the title could not be set: {error}"),
            )
        })
    }

    fn poll(
        &mut self,
        timeout: Option<TimeDuration>,
        sink: &mut dyn FnMut(WndEvent),
    ) -> Result<usize, WndError> {
        self.pump(timeout, sink)
    }
}

/// The window procedure of the launcher class.
unsafe extern "system" fn wnd_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match message {
        WM_PAINT => {
            let mut paint = PAINTSTRUCT::default();
            // Safety: `paint` is a valid, writable paint structure and
            // the handle belongs to the window that sent the message.
            unsafe {
                let _ = BeginPaint(hwnd, &mut paint);
                let _ = EndPaint(hwnd, &paint);
            }
            LRESULT(0)
        }
        // The launcher decides when to close: the request is reported
        // to the driver instead of destroying the window.
        WM_CLOSE => LRESULT(0),
        _ => unsafe { DefWindowProcW(hwnd, message, wparam, lparam) },
    }
}

/// The compiled-in windowing backends, in probe order.
pub(crate) const BACKENDS: &[&dyn WndBackend] = &[&Win32Backend];

/// Windowing backend that opens [`Win32Window`]s on the desktop.
pub(crate) struct Win32Backend;

impl WndBackend for Win32Backend {
    fn id(&self) -> WndBackendId {
        WndBackendId::Win32
    }

    fn is_available(&self) -> bool {
        true
    }

    fn open(&self, config: &WndConfig) -> Result<Box<dyn Window>, WndError> {
        Ok(Box::new(Win32Window::open(config)?))
    }
}
