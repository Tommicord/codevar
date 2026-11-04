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

//! Use `Trampoline::new` + `Trampoline::call` directly with raw machine
//! code, following the Negex ABI (a single `*mut MatchItem` argument).
//!
//! This is illustrative: the blob below is just a `ret`, so it does not
//! scan input or write a match result. A real compiled regex blob would
//! read `item->input`/`item->input_len` and write `item->result`/`start`/`end`.
//!
//! Run with: cargo run --example trampoline_direct -p codevar-negex

use codevar_negex::{Trampoline, TrampolineError};
use codevar_negex_abi::MatchItem;

fn main() {
    // Raw position-independent machine code. Entry offset 0 means the blob
    // starts at its entry point.
    //
    // x86-64: 0xC3 = `ret`
    // aarch64: 0xC0 0x03 0x5F 0xD6 = `ret`
    let code: &[u8] = if cfg!(target_arch = "x86_64") {
        &[0xC3]
    } else if cfg!(target_arch = "aarch64") {
        &[0xC0, 0x03, 0x5F, 0xD6]
    } else {
        &[]
    };

    println!("trampoline direct call\n");
    println!("code bytes: {:02x?}", code);

    let trampoline = match Trampoline::new(code, 0) {
        Ok(trampoline) => trampoline,
        Err(TrampolineError::Unsupported) => {
            println!("unsupported target: no native trampoline back end");
            return;
        }
        Err(error) => {
            println!("trampoline creation failed: {error}");
            return;
        }
    };

    // Per-match state following the Negex ABI. `input` must point at
    // readable memory for the duration of the call.
    let input = b"hello";
    let mut item = MatchItem::new(input);

    println!("before call:");
    println!("  input_len: {}", item.input_len);
    println!("  result:    {}", item.result);
    println!("  start:     {}", item.start);
    println!("  end:       {}", item.end);

    // SAFETY: `item` is a fully initialized MatchItem on the stack, and
    // `input` outlives the call. The blob follows the Negex ABI: its
    // entry point consumes `*mut MatchItem` and nothing else.
    trampoline.call(&mut item as *mut MatchItem);

    println!("after call:");
    println!("  input_len: {}", item.input_len);
    println!("  result:    {}", item.result);
    println!("  start:     {}", item.start);
    println!("  end:       {}", item.end);
}
