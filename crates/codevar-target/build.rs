//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Build-script hook: publishes the core clock to the crate's compile-time
//! configuration.
//!
//! A board with no debug agent cannot be measured at run time, so the
//! fallback SysTick frequency has to be known at build time instead.
//! `CODEVAR_CPU_HZ` supplies it; when it is absent or implausible the
//! library keeps its documented default. The value is echoed back with
//! `cargo:rustc-env` so `option_env!("CODEVAR_CPU_HZ")` resolves while
//! compiling `codevar-target` itself — build-script `rustc-env` entries
//! apply to the package being built, which is exactly this crate.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CODEVAR_CPU_HZ");
    let Some(value) = env::var_os("CODEVAR_CPU_HZ") else {
        return;
    };
    let value = value.to_string_lossy();
    if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) {
        println!("cargo:rustc-env=CODEVAR_CPU_HZ={value}");
    } else {
        // A typo in a frequency must not silently become a wrong clock: it
        // falls through to the default *and* warns, so the build stays
        // correct while the mistake stays visible. Newlines are flattened so
        // the message stays a single cargo directive.
        let value = value.replace(['\n', '\r'], " ");
        println!("cargo:warning=CODEVAR_CPU_HZ must be a positive integer, ignoring: {value}");
    }
}
