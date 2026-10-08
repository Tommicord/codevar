//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License.  You may obtain a copy of
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

//! Textual printer for [`Module`].
//!
//! [`print`] renders a module in the SPIR-V-assembly-shaped IR text
//! described at the crate level.  Section order follows the logical layout
//! of the SPIR-V specification: header comment, `target` line,
//! extended-instruction imports, entry points, decorations, types,
//! constants, globals, and finally functions with declarations before
//! definitions.
//!
//! Identifiers are `%`-prefixed.  Named entities print their name;
//! unnamed entities print a generated identifier made from their arena
//! index (`%t0`, `%v7`, `%bb1`, `%x0`).  Names and generated identifiers
//! share one namespace, exactly as in SPIR-V, so a name that would collide
//! with an identifier already emitted is disambiguated with a `.1`, `.2`,
//! … suffix.  Disambiguation happens at print time only; the module is not
//! modified, and [`crate::parse`] records the printed identifier as the
//! entity's name, which makes printing idempotent.
//!
//! Linkage is printed as a `Linkage Attributes` decoration derived from
//! the [`Linkage`] field of a function or global, because that field is
//! the single source of truth for linkage.  Linkage decorations present in
//! [`Module::decorations`] are therefore not printed.
//!
//! [`Module`]: crate::ir::Module
//! [`Module::decorations`]: crate::ir::Module#structfield.decorations
//! [`Linkage`]: crate::ir::Linkage

use crate::ir::{
    AddressingModel, ConstValue, Decor, ExecutionModel, FunctionControl, Inst, Linkage, MemoryModel, Module,
    Op, Storage, Type, TypeId, ValueId, ValueKind,
};
use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Renders `module` as textual IR.
#[must_use]
pub fn print(module: &Module) -> String {
    let mut ids = Ids::new(module);
    let mut out = String::with_capacity(1024);
    out.push_str("; Codevar IR 0.1\n");
    out.push_str(&target_line(module));

    push_section(&mut out, &ext_inst_lines(module, &ids));
    push_section(&mut out, &entry_point_lines(module, &ids));
    push_section(&mut out, &decoration_lines(module, &ids));
    push_section(&mut out, &type_lines(module, &ids));
    push_section(&mut out, &constant_lines(module, &ids));
    push_section(&mut out, &global_lines(module, &ids));
    push_section(&mut out, &function_lines(module, &mut ids));
    out
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&print(self))
    }
}

/// Display identifiers, assigned once in section order.
struct Ids {
    /// Every identifier already handed out.
    used: BTreeSet<String>,
    /// Identifier per extended-instruction set, by arena index.
    ext_sets: Vec<String>,
    /// Identifier per type, by arena index.
    types: Vec<String>,
    /// Identifier per value, by arena index.
    values: Vec<String>,
}

impl Ids {
    fn new(module: &Module) -> Self {
        let mut used = BTreeSet::new();
        let ext_sets = module
            .ext_inst_sets
            .iter()
            .enumerate()
            .map(|(index, set)| claim(&mut used, &set.name, &format!("x{index}")))
            .collect();
        let types = (0..module.types().len())
            .map(|index| {
                let name = module.type_name(TypeId(u32::try_from(index).unwrap_or(u32::MAX)));
                claim(&mut used, name, &format!("t{index}"))
            })
            .collect();
        let values = module
            .values()
            .iter()
            .enumerate()
            .map(|(index, value)| claim(&mut used, &value.name, &format!("v{index}")))
            .collect();
        Self {
            used,
            ext_sets,
            types,
            values,
        }
    }

    fn ty(&self, id: TypeId) -> &str {
        debug_assert!(id.index() < self.types.len(), "unknown type handle");
        &self.types[id.index()]
    }

    fn val(&self, id: ValueId) -> &str {
        debug_assert!(id.index() < self.values.len(), "unknown value handle");
        &self.values[id.index()]
    }

    fn ext_set(&self, index: usize) -> &str {
        &self.ext_sets[index]
    }

    fn claim_block(&mut self, name: &str, index: usize) -> String {
        claim(&mut self.used, name, &format!("bb{index}"))
    }
}

/// Returns the `%`-prefixed identifier for `base`, disambiguated against
/// the identifiers already used.
fn claim(used: &mut BTreeSet<String>, base: &str, generated: &str) -> String {
    let base = if base.is_empty() { generated } else { base };
    let mut candidate = format!("%{base}");
    let mut suffix = 1u32;
    while used.contains(&candidate) {
        candidate = format!("%{base}.{suffix}");
        suffix += 1;
    }
    used.insert(candidate.clone());
    candidate
}

fn push_section(out: &mut String, lines: &str) {
    if lines.is_empty() {
        return;
    }
    out.push('\n');
    out.push_str(lines);
}

fn target_line(module: &Module) -> String {
    let address = match module.target.addressing {
        AddressingModel::Logical => "logical",
        AddressingModel::Physical32 => "physical32",
        AddressingModel::Physical64 => "physical64",
    };
    let memory = match module.target.memory {
        MemoryModel::Simple => "simple",
        MemoryModel::GLSL450 => "glsl450",
        MemoryModel::OpenCL => "opencl",
    };
    format!(
        "target {} address {address} memory {memory}\n",
        module.target.name
    )
}

fn ext_inst_lines(module: &Module, ids: &Ids) -> String {
    let mut out = String::new();
    for (index, set) in module.ext_inst_sets.iter().enumerate() {
        out.push_str(ids.ext_set(index));
        out.push_str(" = OpExtInstImport ");
        out.push_str(&quoted(&set.name));
        out.push('\n');
    }
    out
}

fn entry_point_lines(module: &Module, ids: &Ids) -> String {
    let mut out = String::new();
    for entry in &module.entry_points {
        let model = match entry.model {
            ExecutionModel::Kernel => "Kernel",
            ExecutionModel::GLCompute => "GLCompute",
        };
        out.push_str(&format!(
            "OpEntryPoint {model} {} {}\n",
            ids.val(entry.func),
            quoted(&entry.name)
        ));
    }
    out
}

fn decoration_lines(module: &Module, ids: &Ids) -> String {
    let mut out = String::new();
    for decor in &module.decorations {
        if let Decor::BuiltIn { name } = &decor.kind {
            out.push_str(&format!("OpDecorate {} BuiltIn {name}\n", ids.val(decor.target)));
        }
    }
    for id in module
        .globals()
        .iter()
        .copied()
        .chain(module.functions().iter().copied())
    {
        let linkage = match module.value_kind(id) {
            ValueKind::Global(global) => global.linkage,
            ValueKind::Function(function) => function.linkage,
            _ => continue,
        };
        let word = match linkage {
            Linkage::External => continue,
            Linkage::Internal => "Internal",
            Linkage::Import => "Import",
            Linkage::Export => "Export",
        };
        let id = ids.val(id);
        let name = id.strip_prefix('%').unwrap_or(id);
        out.push_str(&format!(
            "OpDecorate {id} LinkageAttributes {} {word}\n",
            quoted(name)
        ));
    }
    out
}

fn type_lines(module: &Module, ids: &Ids) -> String {
    let mut out = String::new();
    for (index, def) in module.types().iter().enumerate() {
        let id = TypeId(u32::try_from(index).unwrap_or(u32::MAX));
        let name = ids.ty(id);
        let body = match &def.ty {
            Type::Void => String::from("OpTypeVoid"),
            Type::Bool => String::from("OpTypeBool"),
            Type::Int { bits, signed } => format!("OpTypeInt {bits} {}", u8::from(*signed)),
            Type::Float { bits } => format!("OpTypeFloat {bits}"),
            Type::Vector { elem, len } => format!("OpTypeVector {} {len}", ids.ty(*elem)),
            Type::Array { elem, len } => format!("OpTypeArray {} {len}", ids.ty(*elem)),
            Type::Struct { fields } => {
                let mut line = String::from("OpTypeStruct");
                for field in fields {
                    line.push(' ');
                    line.push_str(ids.ty(*field));
                }
                line
            }
            Type::Pointer { storage, pointee } => {
                format!("OpTypePointer {} {}", storage_name(*storage), ids.ty(*pointee))
            }
            Type::Function { ret, params } => {
                let mut line = format!("OpTypeFunction {}", ids.ty(*ret));
                for param in params {
                    line.push(' ');
                    line.push_str(ids.ty(*param));
                }
                line
            }
        };
        out.push_str(name);
        out.push_str(" = ");
        out.push_str(&body);
        out.push('\n');
    }
    out
}

fn constant_lines(module: &Module, ids: &Ids) -> String {
    let mut out = String::new();
    for (index, value) in module.values().iter().enumerate() {
        let payload = match value.kind {
            ValueKind::Constant(payload) => payload,
            _ => continue,
        };
        let id = ValueId(u32::try_from(index).unwrap_or(u32::MAX));
        let name = ids.val(id);
        let ty = ids.ty(value.ty);
        match payload {
            ConstValue::Int(raw) => {
                out.push_str(&format!(
                    "{name} = OpConstant {ty} {}\n",
                    int_literal(module.ty(value.ty), raw)
                ));
            }
            ConstValue::Float32(bits) => {
                out.push_str(&format!("{name} = OpConstant {ty} {}\n", f32::from_bits(bits)));
            }
            ConstValue::Float64(bits) => {
                out.push_str(&format!("{name} = OpConstant {ty} {}\n", f64::from_bits(bits)));
            }
            ConstValue::Bool(true) => {
                out.push_str(&format!("{name} = OpConstantTrue {ty}\n"));
            }
            ConstValue::Bool(false) => {
                out.push_str(&format!("{name} = OpConstantFalse {ty}\n"));
            }
            ConstValue::Null => {
                out.push_str(&format!("{name} = OpConstantNull {ty}\n"));
            }
            ConstValue::Undef => {
                out.push_str(&format!("{name} = OpUndef {ty}\n"));
            }
        }
    }
    out
}

fn global_lines(module: &Module, ids: &Ids) -> String {
    let mut out = String::new();
    for &id in module.globals() {
        let name = ids.val(id);
        let ty = ids.ty(module.type_of(id));
        let storage = module
            .pointer_parts(module.type_of(id))
            .map(|(storage, _)| storage)
            .unwrap_or(Storage::Function);
        out.push_str(&format!("{name} = OpVariable {ty} {}", storage_name(storage)));
        if let Some(init) = module.global(id).and_then(|global| global.init) {
            out.push(' ');
            out.push_str(ids.val(init));
        }
        out.push('\n');
    }
    out
}

fn function_lines(module: &Module, ids: &mut Ids) -> String {
    let mut out = String::new();
    let declarations = module.functions().iter().copied().filter(|&id| {
        module
            .function(id)
            .is_some_and(|function| function.is_declaration())
    });
    let definitions = module.functions().iter().copied().filter(|&id| {
        module
            .function(id)
            .is_some_and(|function| !function.is_declaration())
    });
    for id in declarations.chain(definitions) {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&function_text(module, ids, id));
    }
    out
}

fn function_text(module: &Module, ids: &mut Ids, id: ValueId) -> String {
    let function = match module.function(id) {
        Some(function) => function,
        None => return String::new(),
    };
    let ret = match module.ty(function.sig) {
        Type::Function { ret, .. } => *ret,
        _ => return String::new(),
    };
    let mut out = format!(
        "{} = OpFunction {} {} {}\n",
        ids.val(id),
        ids.ty(ret),
        function_control_text(function.control),
        ids.ty(function.sig)
    );
    for &arg in &function.args {
        out.push_str(&format!(
            "    {} = OpFunctionParameter {}\n",
            ids.val(arg),
            ids.ty(module.type_of(arg))
        ));
    }
    let body = match &function.body {
        Some(body) => body,
        None => {
            out.push_str("OpFunctionEnd\n");
            return out;
        }
    };
    let labels: Vec<String> = body
        .iter()
        .enumerate()
        .map(|(index, block)| ids.claim_block(&block.name, index))
        .collect();
    for (index, block) in body.iter().enumerate() {
        out.push_str(&format!("    {} = OpLabel\n", labels[index]));
        for inst in &block.insts {
            out.push_str("    ");
            out.push_str(&inst_text(module, ids, &labels, inst));
            out.push('\n');
        }
    }
    out.push_str("OpFunctionEnd\n");
    out
}

fn inst_text(module: &Module, ids: &Ids, labels: &[String], inst: &Inst) -> String {
    let head = |opcode: &str| -> String {
        match (inst.result, inst.ty) {
            (Some(result), Some(ty)) => format!("{} = {opcode} {}", ids.val(result), ids.ty(ty)),
            _ => String::from(opcode),
        }
    };
    let label = |index: usize| -> &str {
        labels
            .get(index)
            .map(String::as_str)
            .unwrap_or("bb?")
    };
    match &inst.op {
        Op::Variable { init } => {
            let storage = inst
                .ty
                .and_then(|ty| module.pointer_parts(ty))
                .map(|(storage, _)| storage)
                .unwrap_or(Storage::Function);
            let mut out = format!("{} {}", head("OpVariable"), storage_name(storage));
            if let Some(init) = init {
                out.push(' ');
                out.push_str(ids.val(*init));
            }
            out
        }
        Op::Load { ptr } => format!("{} {}", head("OpLoad"), ids.val(*ptr)),
        Op::Store { ptr, value } => format!("OpStore {} {}", ids.val(*ptr), ids.val(*value)),
        Op::AccessChain { base, indices } => {
            let mut out = format!("{} {}", head("OpAccessChain"), ids.val(*base));
            for index in indices {
                out.push(' ');
                out.push_str(ids.val(*index));
            }
            out
        }
        Op::PtrAccessChain { base, indices } => {
            let mut out = format!("{} {}", head("OpPtrAccessChain"), ids.val(*base));
            for index in indices {
                out.push(' ');
                out.push_str(ids.val(*index));
            }
            out
        }
        Op::CopyObject { operand } => format!("{} {}", head("OpCopyObject"), ids.val(*operand)),
        Op::Unary { op, operand } => format!("{} {}", head(&format!("Op{}", op.as_str())), ids.val(*operand)),
        Op::Binary { op, lhs, rhs } => format!(
            "{} {} {}",
            head(&format!("Op{}", op.as_str())),
            ids.val(*lhs),
            ids.val(*rhs)
        ),
        Op::Compare { op, lhs, rhs } => format!(
            "{} {} {}",
            head(&format!("Op{}", op.as_str())),
            ids.val(*lhs),
            ids.val(*rhs)
        ),
        Op::Select { cond, a, b } => format!(
            "{} {} {} {}",
            head("OpSelect"),
            ids.val(*cond),
            ids.val(*a),
            ids.val(*b)
        ),
        Op::Convert { op, operand } => {
            format!("{} {}", head(&format!("Op{}", op.as_str())), ids.val(*operand))
        }
        Op::Call { callee, args } => {
            let mut out = format!("{} {}", head("OpFunctionCall"), ids.val(*callee));
            for arg in args {
                out.push(' ');
                out.push_str(ids.val(*arg));
            }
            out
        }
        Op::ExtInst {
            set,
            inst: number,
            args,
        } => {
            let mut out = format!("{} {} {number}", head("OpExtInst"), ids.ext_set(set.index()));
            for arg in args {
                out.push(' ');
                out.push_str(ids.val(*arg));
            }
            out
        }
        Op::CompositeExtract { composite, indices } => {
            let mut out = format!("{} {}", head("OpCompositeExtract"), ids.val(*composite));
            for index in indices {
                out.push(' ');
                out.push_str(&index.to_string());
            }
            out
        }
        Op::CompositeConstruct { constituents } => {
            let mut out = head("OpCompositeConstruct");
            for value in constituents {
                out.push(' ');
                out.push_str(ids.val(*value));
            }
            out
        }
        Op::Phi { incomings } => {
            let mut out = head("OpPhi");
            for (value, block) in incomings {
                out.push(' ');
                out.push_str(ids.val(*value));
                out.push(' ');
                out.push_str(label(block.index()));
            }
            out
        }
        Op::ControlBarrier { exec, mem, semantics } => format!(
            "OpControlBarrier {} {} {}",
            ids.val(*exec),
            ids.val(*mem),
            ids.val(*semantics)
        ),
        Op::MemoryBarrier { mem, semantics } => {
            format!("OpMemoryBarrier {} {}", ids.val(*mem), ids.val(*semantics))
        }
        Op::SelectionMerge { target, control } => format!(
            "OpSelectionMerge {} {}",
            label(target.index()),
            mask_text(*control)
        ),
        Op::LoopMerge { merge, cont, control } => format!(
            "OpLoopMerge {} {} {}",
            label(merge.index()),
            label(cont.index()),
            mask_text(*control)
        ),
        Op::Branch { target } => format!("OpBranch {}", label(target.index())),
        Op::BranchConditional { cond, then, other } => format!(
            "OpBranchConditional {} {} {}",
            ids.val(*cond),
            label(then.index()),
            label(other.index())
        ),
        Op::Return => String::from("OpReturn"),
        Op::ReturnValue { value } => format!("OpReturnValue {}", ids.val(*value)),
        Op::Unreachable => String::from("OpUnreachable"),
    }
}

/// The integer literal for a constant payload in the module's text form.
///
/// Signed types print the sign-extended value so `-1` reads as `-1`;
/// unsigned types print the raw zero-extended bits.
fn int_literal(ty: &Type, raw: u64) -> String {
    match ty {
        Type::Int { bits: 0, .. } => raw.to_string(),
        Type::Int { bits, signed: true } if *bits < 64 => {
            let shift = 64 - *bits;
            ((raw << shift) as i64 >> shift).to_string()
        }
        Type::Int {
            bits: 64,
            signed: true,
        } => (raw as i64).to_string(),
        _ => raw.to_string(),
    }
}

fn storage_name(storage: Storage) -> &'static str {
    match storage {
        Storage::Function => "Function",
        Storage::Workgroup => "Workgroup",
        Storage::CrossWorkgroup => "CrossWorkgroup",
        Storage::UniformConstant => "UniformConstant",
        Storage::Input => "Input",
        Storage::Output => "Output",
        Storage::Private => "Private",
    }
}

fn function_control_text(control: FunctionControl) -> String {
    let bits = control.bits();
    if bits == 0 {
        return String::from("None");
    }
    const FLAGS: [(u32, &str); 4] = [(1, "Inline"), (2, "DontInline"), (4, "Pure"), (8, "Const")];
    let mut rest = bits;
    let mut parts: Vec<String> = Vec::new();
    for (bit, name) in FLAGS {
        if rest & bit != 0 {
            parts.push(name.to_string());
            rest &= !bit;
        }
    }
    if rest != 0 {
        parts.push(format!("0x{rest:x}"));
    }
    parts.join(" ")
}

fn mask_text(mask: u32) -> String {
    if mask == 0 {
        String::from("None")
    } else {
        format!("0x{mask:x}")
    }
}

fn quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ConvOp, Target};
    use alloc::vec;

    fn golden_module() -> Module {
        let mut module = Module::new(Target::opencl());
        let set = module.add_ext_inst_set("OpenCL.std");
        let void = module.intern_named_type(Type::Void, "void");
        let uint = module.intern_named_type(
            Type::Int {
                bits: 32,
                signed: false,
            },
            "uint",
        );
        let float = module.intern_named_type(Type::Float { bits: 32 }, "float");
        let ulong = module.intern_named_type(
            Type::Int {
                bits: 64,
                signed: false,
            },
            "ulong",
        );
        let ptr = module.intern_named_type(
            Type::Pointer {
                storage: Storage::CrossWorkgroup,
                pointee: float,
            },
            "ptr_cw_float",
        );
        let fn_void = module.intern_named_type(
            Type::Function {
                ret: void,
                params: vec![ptr, ptr],
            },
            "fn_void",
        );
        let fn_gid = module.intern_named_type(
            Type::Function {
                ret: ulong,
                params: vec![uint],
            },
            "fn_ulong_uint",
        );
        let c0 = module
            .intern_const(uint, ConstValue::Int(0))
            .expect("zero fits");
        let gid = module
            .add_function("get_global_id", fn_gid, Linkage::Import)
            .expect("fresh name");
        module.set_value_name(module.function(gid).expect("function").args[0], "dim");

        let kernel = module
            .add_function("vector_add", fn_void, Linkage::External)
            .expect("fresh name");
        module
            .set_entry_point(ExecutionModel::Kernel, kernel, "vector_add")
            .expect("function");
        let args = module
            .function(kernel)
            .expect("function")
            .args
            .clone();
        module.set_value_name(args[0], "a");
        module.set_value_name(args[1], "b");
        module.begin_body(kernel).expect("declaration");
        let entry = module.push_block(kernel, "entry").expect("body");

        let idx = module.new_inst_value(ulong);
        module.name_value(idx, "idx");
        module
            .emit(
                kernel,
                entry,
                Inst::def(
                    idx,
                    ulong,
                    Op::Call {
                        callee: gid,
                        args: vec![c0],
                    },
                ),
            )
            .expect("block");
        let idx32 = module.new_inst_value(uint);
        module.name_value(idx32, "idx32");
        module
            .emit(
                kernel,
                entry,
                Inst::def(
                    idx32,
                    uint,
                    Op::Convert {
                        op: ConvOp::UConvert,
                        operand: idx,
                    },
                ),
            )
            .expect("block");
        let slot_ty = module.ptr_ty(Storage::Function, float);
        let slot = module.new_inst_value(slot_ty);
        module.name_value(slot, "slot");
        module
            .emit(
                kernel,
                entry,
                Inst::def(slot, slot_ty, Op::Variable { init: None }),
            )
            .expect("block");
        let loaded = module.new_inst_value(float);
        module.name_value(loaded, "loaded");
        module
            .emit(kernel, entry, Inst::def(loaded, float, Op::Load { ptr: args[0] }))
            .expect("block");
        let scaled = module.new_inst_value(float);
        module.name_value(scaled, "scaled");
        module
            .emit(
                kernel,
                entry,
                Inst::def(
                    scaled,
                    float,
                    Op::ExtInst {
                        set,
                        inst: 61,
                        args: vec![loaded],
                    },
                ),
            )
            .expect("block");
        module
            .emit(
                kernel,
                entry,
                Inst::none(Op::Store {
                    ptr: args[1],
                    value: scaled,
                }),
            )
            .expect("block");
        module
            .emit(kernel, entry, Inst::none(Op::Return))
            .expect("block");
        module
    }

    #[test]
    fn prints_a_complete_kernel_module() {
        let text = print(&golden_module());
        let expected = "\
; Codevar IR 0.1
target opencl address physical64 memory opencl

%OpenCL.std = OpExtInstImport \"OpenCL.std\"

OpEntryPoint Kernel %vector_add \"vector_add\"

OpDecorate %get_global_id LinkageAttributes \"get_global_id\" Import

%void = OpTypeVoid
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%ulong = OpTypeInt 64 0
%ptr_cw_float = OpTypePointer CrossWorkgroup %float
%fn_void = OpTypeFunction %void %ptr_cw_float %ptr_cw_float
%fn_ulong_uint = OpTypeFunction %ulong %uint
%t7 = OpTypePointer Function %float
";
        assert!(text.contains(expected), "missing header in:\n{text}");
        let expected = "\
%v0 = OpConstant %uint 0

%get_global_id = OpFunction %ulong None %fn_ulong_uint
    %dim = OpFunctionParameter %uint
OpFunctionEnd

%vector_add = OpFunction %void None %fn_void
    %a = OpFunctionParameter %ptr_cw_float
    %b = OpFunctionParameter %ptr_cw_float
    %entry = OpLabel
    %idx = OpFunctionCall %ulong %get_global_id %v0
    %idx32 = OpUConvert %uint %idx
    %slot = OpVariable %t7 Function
    %loaded = OpLoad %float %a
    %scaled = OpExtInst %float %OpenCL.std 61 %loaded
    OpStore %b %scaled
    OpReturn
OpFunctionEnd
";
        assert!(text.ends_with(expected), "missing body in:\n{text}");
    }

    #[test]
    fn display_matches_print() {
        let module = golden_module();
        assert_eq!(module.to_string(), print(&module));
    }

    #[test]
    fn signed_constants_print_sign_extended_and_unsigned_raw() {
        let mut module = Module::new(Target::opencl());
        let int = module.intern_named_type(
            Type::Int {
                bits: 32,
                signed: true,
            },
            "int",
        );
        let uint = module.intern_named_type(
            Type::Int {
                bits: 32,
                signed: false,
            },
            "uint",
        );
        module
            .intern_const(int, ConstValue::Int(0xFFFF_FFFF))
            .expect("fits");
        module
            .intern_const(uint, ConstValue::Int(0xFFFF_FFFF))
            .expect("fits");
        let text = print(&module);
        assert!(text.contains("%int = OpTypeInt 32 1\n"), "{text}");
        assert!(text.contains("= OpConstant %int -1\n"), "{text}");
        assert!(text.contains("= OpConstant %uint 4294967295\n"), "{text}");
    }

    #[test]
    fn float_constants_print_round_trippable_literals() {
        let mut module = Module::new(Target::opencl());
        let float = module.intern_named_type(Type::Float { bits: 32 }, "float");
        module
            .intern_const(float, ConstValue::from_f32_bits(0xBFC0_0000))
            .expect("fits");
        module
            .intern_const(float, ConstValue::from_f32_bits(0x7F80_0000))
            .expect("fits");
        module
            .intern_const(float, ConstValue::from_f32_bits(0x7FC0_0001))
            .expect("fits");
        module
            .intern_const(float, ConstValue::from_f32_bits(0x8000_0000))
            .expect("fits");
        let text = print(&module);
        assert!(text.contains("= OpConstant %float -1.5\n"), "{text}");
        assert!(text.contains("= OpConstant %float inf\n"), "{text}");
        assert!(text.contains("= OpConstant %float NaN\n"), "{text}");
        assert!(text.contains("= OpConstant %float -0\n"), "{text}");
    }

    #[test]
    fn unnamed_entities_get_generated_ids_and_collisions_are_renamed() {
        let mut module = Module::new(Target::opencl());
        let unnamed = module.intern_type(Type::Void);
        let sig = module.fn_ty(unnamed, Vec::new());
        let function = module
            .add_function("t0", sig, Linkage::External)
            .expect("fresh name");
        assert!(module.function(function).is_some());
        let text = print(&module);
        assert!(text.contains("%t0 = OpTypeVoid\n"), "{text}");
        assert!(text.contains("%t0.1 = OpFunction"), "{text}");
    }
}
