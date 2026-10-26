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

//! SPIR-V binary layer: opcode tables, module-to-binary emission, and
//! binary validation.
//!
//! [`ops`] holds the machine-readable tables generated from the SPIR-V
//! core grammar (opcode numbers, enumerant names, extended-instruction
//! numbers), so the builder and the validator agree on the binary format
//! by construction rather than by parallel hand-maintained constants.

pub mod ops;
