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

//! A scratch test that prints the optimized form of a kernel.

use codevar_ocl_native_codegen::opt::{OptConfig, optimize};

#[test]
fn print_the_optimized_module() {
    let path = std::env::var("CV_KERNEL").unwrap_or_default();
    if path.is_empty() {
        return;
    }
    let source = std::fs::read_to_string(&path).expect("the kernel is readable");
    let mut module = codevar_ocl_ir::lower::lower_source(&source).expect("the kernel lowers");
    // One pass at a time, so the pass that breaks the module is named.
    let passes: Vec<(&str, fn(&mut codevar_ocl_ir::ir::Module) -> Result<(), codevar_ocl_native_codegen::error::CodegenError>)> = vec![
        ("mem2reg", codevar_ocl_native_codegen::opt::passes::mem2reg),
        ("constant_folding", codevar_ocl_native_codegen::opt::passes::constant_folding),
        ("sccp", codevar_ocl_native_codegen::opt::passes::sccp),
        ("instcombine", codevar_ocl_native_codegen::opt::passes::instcombine),
        ("gvn", codevar_ocl_native_codegen::opt::passes::gvn),
        ("simplify_cfg", codevar_ocl_native_codegen::opt::passes::simplify_cfg),
        ("dce", codevar_ocl_native_codegen::opt::passes::dce),
        ("inline", |module| codevar_ocl_native_codegen::opt::passes::inline(module, 48)),
        ("licm", codevar_ocl_native_codegen::opt::passes::licm),
        ("unroll", |module| codevar_ocl_native_codegen::opt::passes::unroll(module, 8)),
    ];
    for (name, pass) in passes {
        if let Err(error) = pass(&mut module) {
            panic!("the {name} pass failed: {error}");
        }
        if let Err(error) = codevar_ocl_ir::verify::verify(&module) {
            panic!("after {name}: {error}\n{}", codevar_ocl_ir::print::print(&module));
        }
        if std::env::var("CV_DUMP").is_ok() {
            println!("=== after {name}");
            println!("{}", codevar_ocl_ir::print::print(&module));
        }
    }
    let config = OptConfig::default();
    if let Err(error) = optimize(&mut module, &config) {
        panic!("the optimizer failed: {error}");
    }
    if let Err(error) = codevar_ocl_ir::verify::verify(&module) {
        panic!("the optimized module does not verify: {error}\n{}", codevar_ocl_ir::print::print(&module));
    }
    println!("{}", codevar_ocl_ir::print::print(&module));
}
