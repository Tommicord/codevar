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

//! Loading a kernel program through `clCreateProgramWithIL`.
//!
//! The test skips silently when the host has no usable OpenCL stack or
//! no driver exporting `clCreateProgramWithIL`; every other outcome
//! must match the documented error surface of `Program::from_il`.

use codevar_ocl::{Context, Error, Program, Runtime};

/// A structurally plausible but empty SPIR-V module: valid magic word,
/// version 1.0, zero generator/bound/schema. Enough to reach
/// driver-side IL validation without describing any real code.
const SPIRV_PROBE: &[u8] = &[
    0x03, 0x02, 0x23, 0x07, // magic 0x07230203 (little-endian)
    0x00, 0x00, 0x01, 0x00, // version 1.0
    0x00, 0x00, 0x00, 0x00, // generator
    0x00, 0x00, 0x00, 0x00, // bound
    0x00, 0x00, 0x00, 0x00, // schema
];

#[test]
fn loads_kernel_program_from_il_when_supported() -> Result<(), Error> {
    let Ok(runtime) = Runtime::load() else {
        return Ok(());
    };
    let Some(device) = runtime
        .platforms()
        .iter()
        .flat_map(|platform| platform.all_devices().unwrap_or_default())
        .next()
    else {
        return Ok(());
    };
    let Ok(context) = Context::new(&device) else {
        return Ok(());
    };

    // Validation fires before any driver call.
    assert!(matches!(
        Program::from_il(&context, &[]),
        Err(Error::InvalidArgument { .. })
    ));

    match Program::from_il(&context, SPIRV_PROBE) {
        // No `clCreateProgramWithIL` in this implementation: the
        // optional entry point resolves to `Unsupported` as documented.
        Err(Error::Unsupported { symbol }) => {
            assert_eq!(symbol, "clCreateProgramWithIL");
            Ok(())
        }
        // The entry point exists and the driver rejected the probe IL
        // (`CL_INVALID_IL` and friends surface as `Status`; a null
        // handle without an error code as `NullHandle`).
        Err(Error::Status { .. }) | Err(Error::NullHandle { .. }) => Ok(()),
        // Any other failure kind contradicts the documented surface.
        Err(unexpected) => Err(unexpected),
        // A driver willing to accept the probe must also survive a
        // build attempt on the IL program.
        Ok(program) => {
            let _ = program.build("");
            Ok(())
        }
    }
}
