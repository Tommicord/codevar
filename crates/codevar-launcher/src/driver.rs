//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The driver: handles the loading of the binary.

use crate::wnd;
use crate::wnd::{Window, WndConfig, WndError, WndEvent};
use alloc::boxed::Box;
use alloc::string::String;
use codevar_base::sleep::sleep;
use codevar_logger::{log_debug, log_error, log_info, log_success};
use codevar_time_core::TimeDuration;

/// Poll interval of the window event loop (about 60 Hz).
const WND_POLL_INTERVAL: TimeDuration = TimeDuration::from_millis(16);
/// Pause between retries of a transient window failure.
const RETRY_DELAY: TimeDuration = TimeDuration::from_millis(128);
/// Retries granted per transient failure.
const RETRY_LIMIT: u32 = 0x10;

/// Process exit status produced by the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// Compilation (or `--help`/`--version`) completed successfully.
    Success,
    /// A compilation, I/O, or internal failure occurred.
    Failure,
    /// The command line was invalid (bad flags, missing `INPUT`).
    Usage,
}

impl Exit {
    /// The POSIX-style numeric exit code for this status.
    #[must_use]
    #[inline]
    pub const fn code(self) -> i32 {
        match self {
            Self::Success => 0,
            Self::Failure => 1,
            Self::Usage => 2,
        }
    }
}

/// Opens the launcher window and polls it until it closes.
///
/// Drains the events once per frame and stops when the user or the
/// compositor asked the window to close. Transient open and poll
/// failures are retried through [`retry_transient`]; once a permanent
/// error surfaces or the retry budget is spent, the loop ends with
/// [`Exit::Failure`] instead of exiting the process itself.
fn loop_handler() -> Exit {
    let mut window = match open_window(&WndConfig::default()) {
        Ok(window) => window,
        Err(error) => {
            log_error!("window could not be opened: {}", error);
            return Exit::Failure;
        }
    };
    let mut on_event = |event: WndEvent| match event {
        WndEvent::CloseRequested => log_info!("wnd close requested"),
        WndEvent::Resized(size) => log_info!("wnd resized to {}", size),
    };
    while window.is_open() {
        if let Err(error) = poll_window(window.as_mut(), &mut on_event) {
            log_error!("wnd loop gave up on polling: {}", error);
            window.close();
            return Exit::Failure;
        }
    }
    window.close();
    Exit::Success
}

/// Runs a window operation until it succeeds or the failure is final.
///
/// [`WndError::is_transient`] failures may clear on their own — the
/// compositor is still starting up, a resource frees up — so each one
/// is retried after a [`RETRY_DELAY`] pause until [`RETRY_LIMIT`]
/// attempts are spent. Permanent failures ([`WndError::Unsupported`],
/// [`WndError::Protocol`], [`WndError::Disconnected`]) and a failed
/// delay are returned at once, so the loop always terminates.
///
/// # Errors
///
/// Returns the last classified failure once the budget is exhausted.
fn retry_transient<T>(what: &str, mut attempt: impl FnMut() -> Result<T, WndError>) -> Result<T, WndError> {
    let mut attempts = 0;
    loop {
        match attempt() {
            Ok(value) => return Ok(value),
            Err(error) if error.is_transient() => {
                if attempts >= RETRY_LIMIT {
                    return Err(error);
                }
                attempts += 1;
                log_debug!(
                    "{} attempt {} of {} failed: {}; retrying",
                    what,
                    attempts,
                    RETRY_LIMIT,
                    error
                );
                if sleep(RETRY_DELAY).is_err() {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
    }
}

/// Opens a window with [`WndConfig::default`], retrying transient
/// failures through [`retry_transient`].
///
/// # Errors
///
/// Returns the classified failure the backend kept reporting once the
/// retry budget is spent.
fn open_window(config: &WndConfig) -> Result<Box<dyn Window>, WndError> {
    retry_transient("window open", || wnd::open(config))
}

/// Polls `window` once per frame, retrying transient failures until
/// the poll succeeds.
///
/// # Errors
///
/// Returns the classified poll failure once the retry budget is spent
/// or a permanent error proved the connection is gone.
fn poll_window(window: &mut dyn Window, sink: &mut dyn FnMut(WndEvent)) -> Result<(), WndError> {
    retry_transient("wnd poll", || {
        window
            .poll(Some(WND_POLL_INTERVAL), sink)
            .map(|_| ())
    })
}

/// The driver: a full command line plus the pipeline that consumes it.
#[derive(Debug, Clone, Copy)]
pub struct Driver<'a> {
    args: &'a [String],
}

impl<'a> Driver<'a> {
    /// Creates a driver over the complete command line, `argv[0]` included.
    #[must_use]
    pub const fn new(args: &'a [String]) -> Self {
        Self { args }
    }

    /// Runs the compile pipeline and returns the exit status.
    ///
    /// Never panics on malformed input: every failure path reports a
    /// diagnostic to stderr and returns a non-[`Exit::Success`] status.
    pub fn run(self) -> Exit {
        codevar_sig_module_base::init();
        if let Err(e) = codevar_sig_handler::install() {
            log_error!("error installing signal handler: {}", e);
        } else {
            log_success!("installed signal handler");
        }
        loop_handler()
    }
}

/// Runs a full command line through [`Driver`] (the `run_compiler` free
/// function shape; `args` includes `argv[0]`).
///
/// # Examples
///
/// ```no_run
/// let args = vec![String::from("codevar"), String::from("--version")];
/// assert_eq!(codevar_launcher::driver::run(&args).code(), 0);
/// ```
#[must_use]
pub fn run(args: &[String]) -> Exit {
    Driver::new(args).run()
}
