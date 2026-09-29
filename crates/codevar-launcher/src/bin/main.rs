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

//! The codevar launcher entry point.

use codevar_env::{ENV_APP_ID, ENV_NAME};
use codevar_logger::{log_error, log_info, log_warn};
use codevar_ui_core::ui_display::{UiDisplay, WindowInit};
use std::ffi::CStr;

fn main() {
    codevar_sig_module_base::init();
    if let Err(e) = codevar_sig_handler::install() {
        log_info!("error installing signal handler: {}", e);
    } else {
        log_info!("installed signal handler");
    }
    let init = WindowInit {
        title: unsafe { CStr::from_ptr(ENV_NAME.as_ptr() as *const libc::c_char) },
        app_id: unsafe { CStr::from_ptr(ENV_APP_ID.as_ptr() as *const libc::c_char) },
        width: 640,
        height: 480,
    };
    let mut display = match UiDisplay::new(init) {
        Ok(display) => display,
        Err(e) => {
            log_error!("codevar: failed to initialize display: {e}");
            return;
        }
    };
    if let Err(e) = display.run() {
        log_error!("codevar: render loop error: {e}");
    }
    if let Err(e) = codevar_sig_handler::uninstall() {
        log_warn!("codevar: failed to uninstall signal handlers: {e}");
    }
}
