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

//! Enumerant tables extracted from SPIRV-Headers.
//!
//! Source: `spirv.core.grammar.json` (SPIR-V 1.6) and
//! `extinst.opencl.std.100.grammar.json` in the KhronosGroup/SPIRV-Headers
//! repository. Values are authoritative; regenerate rather than edit.
//!
//! Contains the opcode, enumerant, and extended-instruction constants the
//! Codevar backend emits, plus name/number lookups used by the textual IR
//! printer and parser.

#![allow(dead_code)]
/// `OpUndef`.
pub const OP_UNDEF: u16 = 1;
/// `OpSource`.
pub const OP_SOURCE: u16 = 3;
/// `OpName`.
pub const OP_NAME: u16 = 5;
/// `OpString`.
pub const OP_STRING: u16 = 7;
/// `OpExtInstImport`.
pub const OP_EXTINSTIMPORT: u16 = 11;
/// `OpExtInst`.
pub const OP_EXTINST: u16 = 12;
/// `OpMemoryModel`.
pub const OP_MEMORYMODEL: u16 = 14;
/// `OpEntryPoint`.
pub const OP_ENTRYPOINT: u16 = 15;
/// `OpExecutionMode`.
pub const OP_EXECUTIONMODE: u16 = 16;
/// `OpCapability`.
pub const OP_CAPABILITY: u16 = 17;
/// `OpTypeVoid`.
pub const OP_TYPEVOID: u16 = 19;
/// `OpTypeBool`.
pub const OP_TYPEBOOL: u16 = 20;
/// `OpTypeInt`.
pub const OP_TYPEINT: u16 = 21;
/// `OpTypeFloat`.
pub const OP_TYPEFLOAT: u16 = 22;
/// `OpTypeVector`.
pub const OP_TYPEVECTOR: u16 = 23;
/// `OpTypeArray`.
pub const OP_TYPEARRAY: u16 = 28;
/// `OpTypeStruct`.
pub const OP_TYPESTRUCT: u16 = 30;
/// `OpTypePointer`.
pub const OP_TYPEPOINTER: u16 = 32;
/// `OpTypeFunction`.
pub const OP_TYPEFUNCTION: u16 = 33;
/// `OpConstantTrue`.
pub const OP_CONSTANTTRUE: u16 = 41;
/// `OpConstantFalse`.
pub const OP_CONSTANTFALSE: u16 = 42;
/// `OpConstant`.
pub const OP_CONSTANT: u16 = 43;
/// `OpFunction`.
pub const OP_FUNCTION: u16 = 54;
/// `OpFunctionParameter`.
pub const OP_FUNCTIONPARAMETER: u16 = 55;
/// `OpFunctionEnd`.
pub const OP_FUNCTIONEND: u16 = 56;
/// `OpFunctionCall`.
pub const OP_FUNCTIONCALL: u16 = 57;
/// `OpVariable`.
pub const OP_VARIABLE: u16 = 59;
/// `OpLoad`.
pub const OP_LOAD: u16 = 61;
/// `OpStore`.
pub const OP_STORE: u16 = 62;
/// `OpAccessChain`.
pub const OP_ACCESSCHAIN: u16 = 65;
/// `OpInBoundsAccessChain`.
pub const OP_INBOUNDSACCESSCHAIN: u16 = 66;
/// `OpDecorate`.
pub const OP_DECORATE: u16 = 71;
/// `OpCompositeConstruct`.
pub const OP_COMPOSITECONSTRUCT: u16 = 80;
/// `OpCompositeExtract`.
pub const OP_COMPOSITEEXTRACT: u16 = 81;
/// `OpCopyObject`.
pub const OP_COPYOBJECT: u16 = 83;
/// `OpConvertFToU`.
pub const OP_CONVERTFTOU: u16 = 109;
/// `OpConvertFToS`.
pub const OP_CONVERTFTOS: u16 = 110;
/// `OpConvertSToF`.
pub const OP_CONVERTSTOF: u16 = 111;
/// `OpConvertUToF`.
pub const OP_CONVERTUTOF: u16 = 112;
/// `OpUConvert`.
pub const OP_UCONVERT: u16 = 113;
/// `OpSConvert`.
pub const OP_SCONVERT: u16 = 114;
/// `OpFConvert`.
pub const OP_FCONVERT: u16 = 115;
/// `OpBitcast`.
pub const OP_BITCAST: u16 = 124;
/// `OpSNegate`.
pub const OP_SNEGATE: u16 = 126;
/// `OpFNegate`.
pub const OP_FNEGATE: u16 = 127;
/// `OpIAdd`.
pub const OP_IADD: u16 = 128;
/// `OpFAdd`.
pub const OP_FADD: u16 = 129;
/// `OpISub`.
pub const OP_ISUB: u16 = 130;
/// `OpFSub`.
pub const OP_FSUB: u16 = 131;
/// `OpIMul`.
pub const OP_IMUL: u16 = 132;
/// `OpFMul`.
pub const OP_FMUL: u16 = 133;
/// `OpUDiv`.
pub const OP_UDIV: u16 = 134;
/// `OpSDiv`.
pub const OP_SDIV: u16 = 135;
/// `OpFDiv`.
pub const OP_FDIV: u16 = 136;
/// `OpUMod`.
pub const OP_UMOD: u16 = 137;
/// `OpSRem`.
pub const OP_SREM: u16 = 138;
/// `OpFRem`.
pub const OP_FREM: u16 = 140;
/// `OpLogicalOr`.
pub const OP_LOGICALOR: u16 = 166;
/// `OpLogicalAnd`.
pub const OP_LOGICALAND: u16 = 167;
/// `OpLogicalNot`.
pub const OP_LOGICALNOT: u16 = 168;
/// `OpSelect`.
pub const OP_SELECT: u16 = 169;
/// `OpIEqual`.
pub const OP_IEQUAL: u16 = 170;
/// `OpINotEqual`.
pub const OP_INOTEQUAL: u16 = 171;
/// `OpUGreaterThan`.
pub const OP_UGREATERTHAN: u16 = 172;
/// `OpSGreaterThan`.
pub const OP_SGREATERTHAN: u16 = 173;
/// `OpUGreaterThanEqual`.
pub const OP_UGREATERTHANEQUAL: u16 = 174;
/// `OpSGreaterThanEqual`.
pub const OP_SGREATERTHANEQUAL: u16 = 175;
/// `OpULessThan`.
pub const OP_ULESSTHAN: u16 = 176;
/// `OpSLessThan`.
pub const OP_SLESSTHAN: u16 = 177;
/// `OpULessThanEqual`.
pub const OP_ULESSTHANEQUAL: u16 = 178;
/// `OpSLessThanEqual`.
pub const OP_SLESSTHANEQUAL: u16 = 179;
/// `OpFOrdEqual`.
pub const OP_FORDEQUAL: u16 = 180;
/// `OpFOrdNotEqual`.
pub const OP_FORDNOTEQUAL: u16 = 182;
/// `OpFOrdLessThan`.
pub const OP_FORDLESSTHAN: u16 = 184;
/// `OpFOrdGreaterThan`.
pub const OP_FORDGREATERTHAN: u16 = 186;
/// `OpFOrdLessThanEqual`.
pub const OP_FORDLESSTHANEQUAL: u16 = 188;
/// `OpFOrdGreaterThanEqual`.
pub const OP_FORDGREATERTHANEQUAL: u16 = 190;
/// `OpShiftRightLogical`.
pub const OP_SHIFTRIGHTLOGICAL: u16 = 194;
/// `OpShiftRightArithmetic`.
pub const OP_SHIFTRIGHTARITHMETIC: u16 = 195;
/// `OpShiftLeftLogical`.
pub const OP_SHIFTLEFTLOGICAL: u16 = 196;
/// `OpBitwiseOr`.
pub const OP_BITWISEOR: u16 = 197;
/// `OpBitwiseXor`.
pub const OP_BITWISEXOR: u16 = 198;
/// `OpBitwiseAnd`.
pub const OP_BITWISEAND: u16 = 199;
/// `OpNot`.
pub const OP_NOT: u16 = 200;
/// `OpControlBarrier`.
pub const OP_CONTROLBARRIER: u16 = 224;
/// `OpMemoryBarrier`.
pub const OP_MEMORYBARRIER: u16 = 225;
/// `OpPhi`.
pub const OP_PHI: u16 = 245;
/// `OpLoopMerge`.
pub const OP_LOOPMERGE: u16 = 246;
/// `OpSelectionMerge`.
pub const OP_SELECTIONMERGE: u16 = 247;
/// `OpLabel`.
pub const OP_LABEL: u16 = 248;
/// `OpBranch`.
pub const OP_BRANCH: u16 = 249;
/// `OpBranchConditional`.
pub const OP_BRANCHCONDITIONAL: u16 = 250;
/// `OpReturn`.
pub const OP_RETURN: u16 = 253;
/// `OpReturnValue`.
pub const OP_RETURNVALUE: u16 = 254;
/// `OpUnreachable`.
pub const OP_UNREACHABLE: u16 = 255;

/// Human-readable name of a core opcode known to the backend.
#[must_use]
pub const fn op_name(opcode: u16) -> &'static str {
    match opcode {
        1 => "OpUndef",
        3 => "OpSource",
        5 => "OpName",
        7 => "OpString",
        11 => "OpExtInstImport",
        12 => "OpExtInst",
        14 => "OpMemoryModel",
        15 => "OpEntryPoint",
        16 => "OpExecutionMode",
        17 => "OpCapability",
        19 => "OpTypeVoid",
        20 => "OpTypeBool",
        21 => "OpTypeInt",
        22 => "OpTypeFloat",
        23 => "OpTypeVector",
        28 => "OpTypeArray",
        30 => "OpTypeStruct",
        32 => "OpTypePointer",
        33 => "OpTypeFunction",
        41 => "OpConstantTrue",
        42 => "OpConstantFalse",
        43 => "OpConstant",
        54 => "OpFunction",
        55 => "OpFunctionParameter",
        56 => "OpFunctionEnd",
        57 => "OpFunctionCall",
        59 => "OpVariable",
        61 => "OpLoad",
        62 => "OpStore",
        65 => "OpAccessChain",
        66 => "OpInBoundsAccessChain",
        71 => "OpDecorate",
        80 => "OpCompositeConstruct",
        81 => "OpCompositeExtract",
        83 => "OpCopyObject",
        109 => "OpConvertFToU",
        110 => "OpConvertFToS",
        111 => "OpConvertSToF",
        112 => "OpConvertUToF",
        113 => "OpUConvert",
        114 => "OpSConvert",
        115 => "OpFConvert",
        124 => "OpBitcast",
        126 => "OpSNegate",
        127 => "OpFNegate",
        128 => "OpIAdd",
        129 => "OpFAdd",
        130 => "OpISub",
        131 => "OpFSub",
        132 => "OpIMul",
        133 => "OpFMul",
        134 => "OpUDiv",
        135 => "OpSDiv",
        136 => "OpFDiv",
        137 => "OpUMod",
        138 => "OpSRem",
        140 => "OpFRem",
        166 => "OpLogicalOr",
        167 => "OpLogicalAnd",
        168 => "OpLogicalNot",
        169 => "OpSelect",
        170 => "OpIEqual",
        171 => "OpINotEqual",
        172 => "OpUGreaterThan",
        173 => "OpSGreaterThan",
        174 => "OpUGreaterThanEqual",
        175 => "OpSGreaterThanEqual",
        176 => "OpULessThan",
        177 => "OpSLessThan",
        178 => "OpULessThanEqual",
        179 => "OpSLessThanEqual",
        180 => "OpFOrdEqual",
        182 => "OpFOrdNotEqual",
        184 => "OpFOrdLessThan",
        186 => "OpFOrdGreaterThan",
        188 => "OpFOrdLessThanEqual",
        190 => "OpFOrdGreaterThanEqual",
        194 => "OpShiftRightLogical",
        195 => "OpShiftRightArithmetic",
        196 => "OpShiftLeftLogical",
        197 => "OpBitwiseOr",
        198 => "OpBitwiseXor",
        199 => "OpBitwiseAnd",
        200 => "OpNot",
        224 => "OpControlBarrier",
        225 => "OpMemoryBarrier",
        245 => "OpPhi",
        246 => "OpLoopMerge",
        247 => "OpSelectionMerge",
        248 => "OpLabel",
        249 => "OpBranch",
        250 => "OpBranchConditional",
        253 => "OpReturn",
        254 => "OpReturnValue",
        255 => "OpUnreachable",
        _ => "",
    }
}

/// Core opcode number for a name known to the backend.
#[must_use]
pub fn op_opcode(name: &str) -> Option<u16> {
    let name = name.strip_prefix("Op")?;
    let opcode = match name {
        "AccessChain" => 65,
        "Bitcast" => 124,
        "BitwiseAnd" => 199,
        "BitwiseOr" => 197,
        "BitwiseXor" => 198,
        "Branch" => 249,
        "BranchConditional" => 250,
        "Capability" => 17,
        "CompositeConstruct" => 80,
        "CompositeExtract" => 81,
        "Constant" => 43,
        "ConstantFalse" => 42,
        "ConstantTrue" => 41,
        "ControlBarrier" => 224,
        "ConvertFToS" => 110,
        "ConvertFToU" => 109,
        "ConvertSToF" => 111,
        "ConvertUToF" => 112,
        "CopyObject" => 83,
        "Decorate" => 71,
        "EntryPoint" => 15,
        "ExecutionMode" => 16,
        "ExtInst" => 12,
        "ExtInstImport" => 11,
        "FAdd" => 129,
        "FConvert" => 115,
        "FDiv" => 136,
        "FMul" => 133,
        "FNegate" => 127,
        "FOrdEqual" => 180,
        "FOrdGreaterThan" => 186,
        "FOrdGreaterThanEqual" => 190,
        "FOrdLessThan" => 184,
        "FOrdLessThanEqual" => 188,
        "FOrdNotEqual" => 182,
        "FRem" => 140,
        "FSub" => 131,
        "Function" => 54,
        "FunctionCall" => 57,
        "FunctionEnd" => 56,
        "FunctionParameter" => 55,
        "IAdd" => 128,
        "IEqual" => 170,
        "IMul" => 132,
        "INotEqual" => 171,
        "ISub" => 130,
        "InBoundsAccessChain" => 66,
        "Label" => 248,
        "Load" => 61,
        "LogicalAnd" => 167,
        "LogicalNot" => 168,
        "LogicalOr" => 166,
        "LoopMerge" => 246,
        "MemoryBarrier" => 225,
        "MemoryModel" => 14,
        "Name" => 5,
        "Not" => 200,
        "Phi" => 245,
        "Return" => 253,
        "ReturnValue" => 254,
        "SConvert" => 114,
        "SDiv" => 135,
        "SGreaterThan" => 173,
        "SGreaterThanEqual" => 175,
        "SLessThan" => 177,
        "SLessThanEqual" => 179,
        "SNegate" => 126,
        "SRem" => 138,
        "Select" => 169,
        "SelectionMerge" => 247,
        "ShiftLeftLogical" => 196,
        "ShiftRightArithmetic" => 195,
        "ShiftRightLogical" => 194,
        "Source" => 3,
        "Store" => 62,
        "String" => 7,
        "TypeArray" => 28,
        "TypeBool" => 20,
        "TypeFloat" => 22,
        "TypeFunction" => 33,
        "TypeInt" => 21,
        "TypePointer" => 32,
        "TypeStruct" => 30,
        "TypeVector" => 23,
        "TypeVoid" => 19,
        "UConvert" => 113,
        "UDiv" => 134,
        "UGreaterThan" => 172,
        "UGreaterThanEqual" => 174,
        "ULessThan" => 176,
        "ULessThanEqual" => 178,
        "UMod" => 137,
        "Undef" => 1,
        "Unreachable" => 255,
        "Variable" => 59,
        _ => return None,
    };
    Some(opcode)
}

pub const ADDRESSINGMODEL_LOGICAL: u32 = 0;
pub const ADDRESSINGMODEL_PHYSICAL32: u32 = 1;
pub const ADDRESSINGMODEL_PHYSICAL64: u32 = 2;
pub const ADDRESSINGMODEL_PHYSICALSTORAGEBUFFER64: u32 = 5348;
/// `AddressingModel` enumerant value for a textual name.
#[must_use]
pub fn addressingmodel_value(name: &str) -> Option<u32> {
    let value = match name {
        "Logical" => 0,
        "Physical32" => 1,
        "Physical64" => 2,
        "PhysicalStorageBuffer64" => 5348,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `AddressingModel` value known to the backend.
#[must_use]
pub const fn addressingmodel_name(value: u32) -> &'static str {
    match value {
        0 => "Logical",
        1 => "Physical32",
        2 => "Physical64",
        5348 => "PhysicalStorageBuffer64",
        _ => "",
    }
}

pub const MEMORYMODEL_SIMPLE: u32 = 0;
pub const MEMORYMODEL_GLSL450: u32 = 1;
pub const MEMORYMODEL_OPENCL: u32 = 2;
pub const MEMORYMODEL_VULKAN: u32 = 3;
/// `MemoryModel` enumerant value for a textual name.
#[must_use]
pub fn memorymodel_value(name: &str) -> Option<u32> {
    let value = match name {
        "Simple" => 0,
        "GLSL450" => 1,
        "OpenCL" => 2,
        "Vulkan" => 3,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `MemoryModel` value known to the backend.
#[must_use]
pub const fn memorymodel_name(value: u32) -> &'static str {
    match value {
        0 => "Simple",
        1 => "GLSL450",
        2 => "OpenCL",
        3 => "Vulkan",
        _ => "",
    }
}

pub const EXECUTIONMODEL_VERTEX: u32 = 0;
pub const EXECUTIONMODEL_TESSELLATIONCONTROL: u32 = 1;
pub const EXECUTIONMODEL_TESSELLATIONEVALUATION: u32 = 2;
pub const EXECUTIONMODEL_GEOMETRY: u32 = 3;
pub const EXECUTIONMODEL_FRAGMENT: u32 = 4;
pub const EXECUTIONMODEL_GLCOMPUTE: u32 = 5;
pub const EXECUTIONMODEL_KERNEL: u32 = 6;
pub const EXECUTIONMODEL_TASKNV: u32 = 5267;
pub const EXECUTIONMODEL_MESHNV: u32 = 5268;
pub const EXECUTIONMODEL_RAYGENERATIONKHR: u32 = 5313;
pub const EXECUTIONMODEL_INTERSECTIONKHR: u32 = 5314;
pub const EXECUTIONMODEL_ANYHITKHR: u32 = 5315;
pub const EXECUTIONMODEL_CLOSESTHITKHR: u32 = 5316;
pub const EXECUTIONMODEL_MISSKHR: u32 = 5317;
pub const EXECUTIONMODEL_CALLABLEKHR: u32 = 5318;
pub const EXECUTIONMODEL_TASKEXT: u32 = 5364;
pub const EXECUTIONMODEL_MESHEXT: u32 = 5365;
/// `ExecutionModel` enumerant value for a textual name.
#[must_use]
pub fn executionmodel_value(name: &str) -> Option<u32> {
    let value = match name {
        "Vertex" => 0,
        "TessellationControl" => 1,
        "TessellationEvaluation" => 2,
        "Geometry" => 3,
        "Fragment" => 4,
        "GLCompute" => 5,
        "Kernel" => 6,
        "TaskNV" => 5267,
        "MeshNV" => 5268,
        "RayGenerationKHR" => 5313,
        "IntersectionKHR" => 5314,
        "AnyHitKHR" => 5315,
        "ClosestHitKHR" => 5316,
        "MissKHR" => 5317,
        "CallableKHR" => 5318,
        "TaskEXT" => 5364,
        "MeshEXT" => 5365,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `ExecutionModel` value known to the backend.
#[must_use]
pub const fn executionmodel_name(value: u32) -> &'static str {
    match value {
        0 => "Vertex",
        1 => "TessellationControl",
        2 => "TessellationEvaluation",
        3 => "Geometry",
        4 => "Fragment",
        5 => "GLCompute",
        6 => "Kernel",
        5267 => "TaskNV",
        5268 => "MeshNV",
        5313 => "RayGenerationKHR",
        5314 => "IntersectionKHR",
        5315 => "AnyHitKHR",
        5316 => "ClosestHitKHR",
        5317 => "MissKHR",
        5318 => "CallableKHR",
        5364 => "TaskEXT",
        5365 => "MeshEXT",
        _ => "",
    }
}

// ---- StorageClass ----
pub const STORAGECLASS_UNIFORMCONSTANT: u32 = 0;
pub const STORAGECLASS_INPUT: u32 = 1;
pub const STORAGECLASS_UNIFORM: u32 = 2;
pub const STORAGECLASS_OUTPUT: u32 = 3;
pub const STORAGECLASS_WORKGROUP: u32 = 4;
pub const STORAGECLASS_CROSSWORKGROUP: u32 = 5;
pub const STORAGECLASS_PRIVATE: u32 = 6;
pub const STORAGECLASS_FUNCTION: u32 = 7;
pub const STORAGECLASS_GENERIC: u32 = 8;
pub const STORAGECLASS_PUSHCONSTANT: u32 = 9;
pub const STORAGECLASS_ATOMICCOUNTER: u32 = 10;
pub const STORAGECLASS_IMAGE: u32 = 11;
pub const STORAGECLASS_STORAGEBUFFER: u32 = 12;
pub const STORAGECLASS_TILEIMAGEEXT: u32 = 4172;
pub const STORAGECLASS_TILEATTACHMENTQCOM: u32 = 4491;
pub const STORAGECLASS_NODEPAYLOADAMDX: u32 = 5068;
pub const STORAGECLASS_CALLABLEDATAKHR: u32 = 5328;
pub const STORAGECLASS_INCOMINGCALLABLEDATAKHR: u32 = 5329;
pub const STORAGECLASS_RAYPAYLOADKHR: u32 = 5338;
pub const STORAGECLASS_HITATTRIBUTEKHR: u32 = 5339;
pub const STORAGECLASS_INCOMINGRAYPAYLOADKHR: u32 = 5342;
pub const STORAGECLASS_SHADERRECORDBUFFERKHR: u32 = 5343;
pub const STORAGECLASS_PHYSICALSTORAGEBUFFER: u32 = 5349;
pub const STORAGECLASS_HITOBJECTATTRIBUTENV: u32 = 5385;
pub const STORAGECLASS_TASKPAYLOADWORKGROUPEXT: u32 = 5402;
pub const STORAGECLASS_HITOBJECTATTRIBUTEEXT: u32 = 5411;
pub const STORAGECLASS_CODESECTIONINTEL: u32 = 5605;
pub const STORAGECLASS_DEVICEONLYALTERA: u32 = 5936;
pub const STORAGECLASS_HOSTONLYALTERA: u32 = 5937;
/// `StorageClass` enumerant value for a textual name.
#[must_use]
pub fn storageclass_value(name: &str) -> Option<u32> {
    let value = match name {
        "UniformConstant" => 0,
        "Input" => 1,
        "Uniform" => 2,
        "Output" => 3,
        "Workgroup" => 4,
        "CrossWorkgroup" => 5,
        "Private" => 6,
        "Function" => 7,
        "Generic" => 8,
        "PushConstant" => 9,
        "AtomicCounter" => 10,
        "Image" => 11,
        "StorageBuffer" => 12,
        "TileImageEXT" => 4172,
        "TileAttachmentQCOM" => 4491,
        "NodePayloadAMDX" => 5068,
        "CallableDataKHR" => 5328,
        "IncomingCallableDataKHR" => 5329,
        "RayPayloadKHR" => 5338,
        "HitAttributeKHR" => 5339,
        "IncomingRayPayloadKHR" => 5342,
        "ShaderRecordBufferKHR" => 5343,
        "PhysicalStorageBuffer" => 5349,
        "HitObjectAttributeNV" => 5385,
        "TaskPayloadWorkgroupEXT" => 5402,
        "HitObjectAttributeEXT" => 5411,
        "CodeSectionINTEL" => 5605,
        "DeviceOnlyALTERA" => 5936,
        "HostOnlyALTERA" => 5937,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `StorageClass` value known to the backend.
#[must_use]
pub const fn storageclass_name(value: u32) -> &'static str {
    match value {
        0 => "UniformConstant",
        1 => "Input",
        2 => "Uniform",
        3 => "Output",
        4 => "Workgroup",
        5 => "CrossWorkgroup",
        6 => "Private",
        7 => "Function",
        8 => "Generic",
        9 => "PushConstant",
        10 => "AtomicCounter",
        11 => "Image",
        12 => "StorageBuffer",
        4172 => "TileImageEXT",
        4491 => "TileAttachmentQCOM",
        5068 => "NodePayloadAMDX",
        5328 => "CallableDataKHR",
        5329 => "IncomingCallableDataKHR",
        5338 => "RayPayloadKHR",
        5339 => "HitAttributeKHR",
        5342 => "IncomingRayPayloadKHR",
        5343 => "ShaderRecordBufferKHR",
        5349 => "PhysicalStorageBuffer",
        5385 => "HitObjectAttributeNV",
        5402 => "TaskPayloadWorkgroupEXT",
        5411 => "HitObjectAttributeEXT",
        5605 => "CodeSectionINTEL",
        5936 => "DeviceOnlyALTERA",
        5937 => "HostOnlyALTERA",
        _ => "",
    }
}

pub const CAPABILITY_MATRIX: u32 = 0;
pub const CAPABILITY_SHADER: u32 = 1;
pub const CAPABILITY_GEOMETRY: u32 = 2;
pub const CAPABILITY_TESSELLATION: u32 = 3;
pub const CAPABILITY_ADDRESSES: u32 = 4;
pub const CAPABILITY_LINKAGE: u32 = 5;
pub const CAPABILITY_KERNEL: u32 = 6;
pub const CAPABILITY_VECTOR16: u32 = 7;
pub const CAPABILITY_FLOAT16BUFFER: u32 = 8;
pub const CAPABILITY_FLOAT16: u32 = 9;
pub const CAPABILITY_FLOAT64: u32 = 10;
pub const CAPABILITY_INT64: u32 = 11;
pub const CAPABILITY_INT64ATOMICS: u32 = 12;
pub const CAPABILITY_IMAGEBASIC: u32 = 13;
pub const CAPABILITY_IMAGEREADWRITE: u32 = 14;
pub const CAPABILITY_IMAGEMIPMAP: u32 = 15;
pub const CAPABILITY_PIPES: u32 = 17;
pub const CAPABILITY_GROUPS: u32 = 18;
pub const CAPABILITY_DEVICEENQUEUE: u32 = 19;
pub const CAPABILITY_LITERALSAMPLER: u32 = 20;
pub const CAPABILITY_ATOMICSTORAGE: u32 = 21;
pub const CAPABILITY_INT16: u32 = 22;
pub const CAPABILITY_TESSELLATIONPOINTSIZE: u32 = 23;
pub const CAPABILITY_GEOMETRYPOINTSIZE: u32 = 24;
pub const CAPABILITY_IMAGEGATHEREXTENDED: u32 = 25;
pub const CAPABILITY_STORAGEIMAGEMULTISAMPLE: u32 = 27;
pub const CAPABILITY_UNIFORMBUFFERARRAYDYNAMICINDEXING: u32 = 28;
pub const CAPABILITY_SAMPLEDIMAGEARRAYDYNAMICINDEXING: u32 = 29;
pub const CAPABILITY_STORAGEBUFFERARRAYDYNAMICINDEXING: u32 = 30;
pub const CAPABILITY_STORAGEIMAGEARRAYDYNAMICINDEXING: u32 = 31;
pub const CAPABILITY_CLIPDISTANCE: u32 = 32;
pub const CAPABILITY_CULLDISTANCE: u32 = 33;
pub const CAPABILITY_IMAGECUBEARRAY: u32 = 34;
pub const CAPABILITY_SAMPLERATESHADING: u32 = 35;
pub const CAPABILITY_IMAGERECT: u32 = 36;
pub const CAPABILITY_SAMPLEDRECT: u32 = 37;
pub const CAPABILITY_GENERICPOINTER: u32 = 38;
pub const CAPABILITY_INT8: u32 = 39;
pub const CAPABILITY_INPUTATTACHMENT: u32 = 40;
pub const CAPABILITY_SPARSERESIDENCY: u32 = 41;
pub const CAPABILITY_MINLOD: u32 = 42;
pub const CAPABILITY_SAMPLED1D: u32 = 43;
pub const CAPABILITY_IMAGE1D: u32 = 44;
pub const CAPABILITY_SAMPLEDCUBEARRAY: u32 = 45;
pub const CAPABILITY_SAMPLEDBUFFER: u32 = 46;
pub const CAPABILITY_IMAGEBUFFER: u32 = 47;
pub const CAPABILITY_IMAGEMSARRAY: u32 = 48;
pub const CAPABILITY_STORAGEIMAGEEXTENDEDFORMATS: u32 = 49;
pub const CAPABILITY_IMAGEQUERY: u32 = 50;
pub const CAPABILITY_DERIVATIVECONTROL: u32 = 51;
pub const CAPABILITY_INTERPOLATIONFUNCTION: u32 = 52;
pub const CAPABILITY_TRANSFORMFEEDBACK: u32 = 53;
pub const CAPABILITY_GEOMETRYSTREAMS: u32 = 54;
pub const CAPABILITY_STORAGEIMAGEREADWITHOUTFORMAT: u32 = 55;
pub const CAPABILITY_STORAGEIMAGEWRITEWITHOUTFORMAT: u32 = 56;
pub const CAPABILITY_MULTIVIEWPORT: u32 = 57;
pub const CAPABILITY_SUBGROUPDISPATCH: u32 = 58;
pub const CAPABILITY_NAMEDBARRIER: u32 = 59;
pub const CAPABILITY_PIPESTORAGE: u32 = 60;
pub const CAPABILITY_GROUPNONUNIFORM: u32 = 61;
pub const CAPABILITY_GROUPNONUNIFORMVOTE: u32 = 62;
pub const CAPABILITY_GROUPNONUNIFORMARITHMETIC: u32 = 63;
pub const CAPABILITY_GROUPNONUNIFORMBALLOT: u32 = 64;
pub const CAPABILITY_GROUPNONUNIFORMSHUFFLE: u32 = 65;
pub const CAPABILITY_GROUPNONUNIFORMSHUFFLERELATIVE: u32 = 66;
pub const CAPABILITY_GROUPNONUNIFORMCLUSTERED: u32 = 67;
pub const CAPABILITY_GROUPNONUNIFORMQUAD: u32 = 68;
pub const CAPABILITY_SHADERLAYER: u32 = 69;
pub const CAPABILITY_SHADERVIEWPORTINDEX: u32 = 70;
pub const CAPABILITY_UNIFORMDECORATION: u32 = 71;
pub const CAPABILITY_COREBUILTINSARM: u32 = 4165;
pub const CAPABILITY_TILEIMAGECOLORREADACCESSEXT: u32 = 4166;
pub const CAPABILITY_TILEIMAGEDEPTHREADACCESSEXT: u32 = 4167;
pub const CAPABILITY_TILEIMAGESTENCILREADACCESSEXT: u32 = 4168;
pub const CAPABILITY_TENSORSARM: u32 = 4174;
pub const CAPABILITY_STORAGETENSORARRAYDYNAMICINDEXINGARM: u32 = 4175;
pub const CAPABILITY_STORAGETENSORARRAYNONUNIFORMINDEXINGARM: u32 = 4176;
pub const CAPABILITY_GRAPHARM: u32 = 4191;
pub const CAPABILITY_COOPERATIVEMATRIXLAYOUTSARM: u32 = 4201;
pub const CAPABILITY_FLOAT8EXT: u32 = 4212;
pub const CAPABILITY_FLOAT8COOPERATIVEMATRIXEXT: u32 = 4213;
pub const CAPABILITY_FLOAT6EXT: u32 = 4228;
pub const CAPABILITY_FLOAT4EXT: u32 = 4229;
pub const CAPABILITY_FLOAT8UNSIGNEDE8M0EXT: u32 = 4230;
pub const CAPABILITY_MXINT8EXT: u32 = 4231;
pub const CAPABILITY_BITCASTEXTRACTEXT: u32 = 4232;
pub const CAPABILITY_FRAGMENTSHADINGRATEKHR: u32 = 4422;
pub const CAPABILITY_SUBGROUPBALLOTKHR: u32 = 4423;
pub const CAPABILITY_DRAWPARAMETERS: u32 = 4427;
pub const CAPABILITY_WORKGROUPMEMORYEXPLICITLAYOUTKHR: u32 = 4428;
pub const CAPABILITY_WORKGROUPMEMORYEXPLICITLAYOUT8BITACCESSKHR: u32 = 4429;
pub const CAPABILITY_WORKGROUPMEMORYEXPLICITLAYOUT16BITACCESSKHR: u32 = 4430;
pub const CAPABILITY_SUBGROUPVOTEKHR: u32 = 4431;
pub const CAPABILITY_STORAGEBUFFER16BITACCESS: u32 = 4433;
pub const CAPABILITY_UNIFORMANDSTORAGEBUFFER16BITACCESS: u32 = 4434;
pub const CAPABILITY_STORAGEPUSHCONSTANT16: u32 = 4435;
pub const CAPABILITY_STORAGEINPUTOUTPUT16: u32 = 4436;
pub const CAPABILITY_DEVICEGROUP: u32 = 4437;
pub const CAPABILITY_MULTIVIEW: u32 = 4439;
pub const CAPABILITY_VARIABLEPOINTERSSTORAGEBUFFER: u32 = 4441;
pub const CAPABILITY_VARIABLEPOINTERS: u32 = 4442;
pub const CAPABILITY_ATOMICSTORAGEOPS: u32 = 4445;
pub const CAPABILITY_SAMPLEMASKPOSTDEPTHCOVERAGE: u32 = 4447;
pub const CAPABILITY_STORAGEBUFFER8BITACCESS: u32 = 4448;
pub const CAPABILITY_UNIFORMANDSTORAGEBUFFER8BITACCESS: u32 = 4449;
pub const CAPABILITY_STORAGEPUSHCONSTANT8: u32 = 4450;
pub const CAPABILITY_DENORMPRESERVE: u32 = 4464;
pub const CAPABILITY_DENORMFLUSHTOZERO: u32 = 4465;
pub const CAPABILITY_SIGNEDZEROINFNANPRESERVE: u32 = 4466;
pub const CAPABILITY_ROUNDINGMODERTE: u32 = 4467;
pub const CAPABILITY_ROUNDINGMODERTZ: u32 = 4468;
pub const CAPABILITY_RAYQUERYPROVISIONALKHR: u32 = 4471;
pub const CAPABILITY_RAYQUERYKHR: u32 = 4472;
pub const CAPABILITY_UNTYPEDPOINTERSKHR: u32 = 4473;
pub const CAPABILITY_RAYTRAVERSALPRIMITIVECULLINGKHR: u32 = 4478;
pub const CAPABILITY_RAYTRACINGKHR: u32 = 4479;
pub const CAPABILITY_TEXTURESAMPLEWEIGHTEDQCOM: u32 = 4484;
pub const CAPABILITY_TEXTUREBOXFILTERQCOM: u32 = 4485;
pub const CAPABILITY_TEXTUREBLOCKMATCHQCOM: u32 = 4486;
pub const CAPABILITY_TILESHADINGQCOM: u32 = 4495;
pub const CAPABILITY_COOPERATIVEMATRIXCONVERSIONQCOM: u32 = 4496;
pub const CAPABILITY_TEXTUREBLOCKMATCH2QCOM: u32 = 4498;
pub const CAPABILITY_BFLOAT16MULADDQCOM: u32 = 4504;
pub const CAPABILITY_SUBGROUPSIZEQCOM: u32 = 4506;
pub const CAPABILITY_MULTIPLEWAITQUEUESQCOM: u32 = 4539;
pub const CAPABILITY_IMAGEGATHERLINEARQCOM: u32 = 4543;
pub const CAPABILITY_IMAGEGATHEREXTENDEDMODESQCOM: u32 = 4544;
pub const CAPABILITY_FLOAT16IMAGEAMD: u32 = 5008;
pub const CAPABILITY_IMAGEGATHERBIASLODAMD: u32 = 5009;
pub const CAPABILITY_FRAGMENTMASKAMD: u32 = 5010;
pub const CAPABILITY_STENCILEXPORTEXT: u32 = 5013;
pub const CAPABILITY_IMAGEREADWRITELODAMD: u32 = 5015;
pub const CAPABILITY_INT64IMAGEEXT: u32 = 5016;
pub const CAPABILITY_SHADERCLOCKKHR: u32 = 5055;
pub const CAPABILITY_SHADERENQUEUEAMDX: u32 = 5067;
pub const CAPABILITY_QUADCONTROLKHR: u32 = 5087;
pub const CAPABILITY_INT4TYPEINTEL: u32 = 5112;
pub const CAPABILITY_INT4COOPERATIVEMATRIXINTEL: u32 = 5114;
pub const CAPABILITY_BFLOAT16TYPEKHR: u32 = 5116;
pub const CAPABILITY_BFLOAT16DOTPRODUCTKHR: u32 = 5117;
pub const CAPABILITY_BFLOAT16COOPERATIVEMATRIXKHR: u32 = 5118;
pub const CAPABILITY_ABORTKHR: u32 = 5120;
pub const CAPABILITY_DESCRIPTORHEAPEXT: u32 = 5128;
pub const CAPABILITY_CONSTANTDATAKHR: u32 = 5146;
pub const CAPABILITY_POISONFREEZEKHR: u32 = 5156;
pub const CAPABILITY_WEAKLINKAGEAMD: u32 = 5181;
pub const CAPABILITY_SAMPLEMASKOVERRIDECOVERAGENV: u32 = 5249;
pub const CAPABILITY_GEOMETRYSHADERPASSTHROUGHNV: u32 = 5251;
pub const CAPABILITY_SHADERVIEWPORTINDEXLAYEREXT: u32 = 5254;
pub const CAPABILITY_SHADERVIEWPORTMASKNV: u32 = 5255;
pub const CAPABILITY_SHADERSTEREOVIEWNV: u32 = 5259;
pub const CAPABILITY_PERVIEWATTRIBUTESNV: u32 = 5260;
pub const CAPABILITY_FRAGMENTFULLYCOVEREDEXT: u32 = 5265;
pub const CAPABILITY_MESHSHADINGNV: u32 = 5266;
pub const CAPABILITY_IMAGEFOOTPRINTNV: u32 = 5282;
pub const CAPABILITY_MESHSHADINGEXT: u32 = 5283;
pub const CAPABILITY_FRAGMENTBARYCENTRICKHR: u32 = 5284;
pub const CAPABILITY_COMPUTEDERIVATIVEGROUPQUADSKHR: u32 = 5288;
pub const CAPABILITY_FRAGMENTDENSITYEXT: u32 = 5291;
pub const CAPABILITY_GROUPNONUNIFORMPARTITIONEDEXT: u32 = 5297;
pub const CAPABILITY_SHADERNONUNIFORM: u32 = 5301;
pub const CAPABILITY_RUNTIMEDESCRIPTORARRAY: u32 = 5302;
pub const CAPABILITY_INPUTATTACHMENTARRAYDYNAMICINDEXING: u32 = 5303;
pub const CAPABILITY_UNIFORMTEXELBUFFERARRAYDYNAMICINDEXING: u32 = 5304;
pub const CAPABILITY_STORAGETEXELBUFFERARRAYDYNAMICINDEXING: u32 = 5305;
pub const CAPABILITY_UNIFORMBUFFERARRAYNONUNIFORMINDEXING: u32 = 5306;
pub const CAPABILITY_SAMPLEDIMAGEARRAYNONUNIFORMINDEXING: u32 = 5307;
pub const CAPABILITY_STORAGEBUFFERARRAYNONUNIFORMINDEXING: u32 = 5308;
pub const CAPABILITY_STORAGEIMAGEARRAYNONUNIFORMINDEXING: u32 = 5309;
pub const CAPABILITY_INPUTATTACHMENTARRAYNONUNIFORMINDEXING: u32 = 5310;
pub const CAPABILITY_UNIFORMTEXELBUFFERARRAYNONUNIFORMINDEXING: u32 = 5311;
pub const CAPABILITY_STORAGETEXELBUFFERARRAYNONUNIFORMINDEXING: u32 = 5312;
pub const CAPABILITY_RAYTRACINGPOSITIONFETCHKHR: u32 = 5336;
pub const CAPABILITY_RAYTRACINGNV: u32 = 5340;
pub const CAPABILITY_RAYTRACINGMOTIONBLURNV: u32 = 5341;
pub const CAPABILITY_VULKANMEMORYMODEL: u32 = 5345;
pub const CAPABILITY_VULKANMEMORYMODELDEVICESCOPE: u32 = 5346;
pub const CAPABILITY_PHYSICALSTORAGEBUFFERADDRESSES: u32 = 5347;
pub const CAPABILITY_COMPUTEDERIVATIVEGROUPLINEARKHR: u32 = 5350;
pub const CAPABILITY_RAYTRACINGPROVISIONALKHR: u32 = 5353;
pub const CAPABILITY_COOPERATIVEMATRIXNV: u32 = 5357;
pub const CAPABILITY_FRAGMENTSHADERSAMPLEINTERLOCKEXT: u32 = 5363;
pub const CAPABILITY_FRAGMENTSHADERSHADINGRATEINTERLOCKEXT: u32 = 5372;
pub const CAPABILITY_SHADERSMBUILTINSNV: u32 = 5373;
pub const CAPABILITY_FRAGMENTSHADERPIXELINTERLOCKEXT: u32 = 5378;
pub const CAPABILITY_DEMOTETOHELPERINVOCATION: u32 = 5379;
pub const CAPABILITY_DISPLACEMENTMICROMAPNV: u32 = 5380;
pub const CAPABILITY_RAYTRACINGOPACITYMICROMAPKHR: u32 = 5381;
pub const CAPABILITY_SHADERINVOCATIONREORDERNV: u32 = 5383;
pub const CAPABILITY_SHADERINVOCATIONREORDEREXT: u32 = 5388;
pub const CAPABILITY_BINDLESSTEXTURENV: u32 = 5390;
pub const CAPABILITY_RAYQUERYPOSITIONFETCHKHR: u32 = 5391;
pub const CAPABILITY_COOPERATIVEVECTORNV: u32 = 5394;
pub const CAPABILITY_ATOMICFLOAT16VECTORNV: u32 = 5404;
pub const CAPABILITY_RAYTRACINGDISPLACEMENTMICROMAPNV: u32 = 5409;
pub const CAPABILITY_RAWACCESSCHAINSNV: u32 = 5414;
pub const CAPABILITY_RAYTRACINGSPHERESGEOMETRYNV: u32 = 5418;
pub const CAPABILITY_RAYTRACINGLINEARSWEPTSPHERESGEOMETRYNV: u32 = 5419;
pub const CAPABILITY_PUSHCONSTANTBANKSNV: u32 = 5423;
pub const CAPABILITY_LONGVECTOREXT: u32 = 5425;
pub const CAPABILITY_SHADER64BITINDEXINGEXT: u32 = 5426;
pub const CAPABILITY_COOPERATIVEMATRIXCONVERSIONSEXT: u32 = 5429;
pub const CAPABILITY_COOPERATIVEMATRIXREDUCTIONSEXT: u32 = 5430;
pub const CAPABILITY_COOPERATIVEMATRIXCONVERSIONSNV: u32 = 5431;
pub const CAPABILITY_COOPERATIVEMATRIXPERELEMENTOPERATIONSEXT: u32 = 5432;
pub const CAPABILITY_COOPERATIVEMATRIXTENSORADDRESSINGNV: u32 = 5433;
pub const CAPABILITY_COOPERATIVEMATRIXBLOCKLOADSNV: u32 = 5434;
pub const CAPABILITY_COOPERATIVEVECTORTRAININGNV: u32 = 5435;
pub const CAPABILITY_RAYTRACINGCLUSTERACCELERATIONSTRUCTURENV: u32 = 5437;
pub const CAPABILITY_COOPERATIVEMATRIXGETCOORDINATEEXT: u32 = 5438;
pub const CAPABILITY_TENSORADDRESSINGNV: u32 = 5439;
pub const CAPABILITY_COOPERATIVEMATRIXDECODEVECTORNV: u32 = 5447;
pub const CAPABILITY_SUBGROUPSHUFFLEINTEL: u32 = 5568;
pub const CAPABILITY_SUBGROUPBUFFERBLOCKIOINTEL: u32 = 5569;
pub const CAPABILITY_SUBGROUPIMAGEBLOCKIOINTEL: u32 = 5570;
pub const CAPABILITY_SUBGROUPIMAGEMEDIABLOCKIOINTEL: u32 = 5579;
pub const CAPABILITY_ROUNDTOINFINITYINTEL: u32 = 5582;
pub const CAPABILITY_FLOATINGPOINTMODEINTEL: u32 = 5583;
pub const CAPABILITY_INTEGERFUNCTIONS2INTEL: u32 = 5584;
pub const CAPABILITY_FUNCTIONPOINTERSINTEL: u32 = 5603;
pub const CAPABILITY_INDIRECTREFERENCESINTEL: u32 = 5604;
pub const CAPABILITY_ASMINTEL: u32 = 5606;
pub const CAPABILITY_ATOMICFLOAT32MINMAXEXT: u32 = 5612;
pub const CAPABILITY_ATOMICFLOAT64MINMAXEXT: u32 = 5613;
pub const CAPABILITY_ATOMICFLOAT16MINMAXEXT: u32 = 5616;
pub const CAPABILITY_VECTORCOMPUTEINTEL: u32 = 5617;
pub const CAPABILITY_VECTORANYINTEL: u32 = 5619;
pub const CAPABILITY_EXPECTASSUMEKHR: u32 = 5629;
pub const CAPABILITY_SUBGROUPAVCMOTIONESTIMATIONINTEL: u32 = 5696;
pub const CAPABILITY_SUBGROUPAVCMOTIONESTIMATIONINTRAINTEL: u32 = 5697;
pub const CAPABILITY_SUBGROUPAVCMOTIONESTIMATIONCHROMAINTEL: u32 = 5698;
pub const CAPABILITY_VARIABLELENGTHARRAYINTEL: u32 = 5817;
pub const CAPABILITY_FUNCTIONFLOATCONTROLINTEL: u32 = 5821;
pub const CAPABILITY_FPGAMEMORYATTRIBUTESALTERA: u32 = 5824;
pub const CAPABILITY_FPFASTMATHMODEINTEL: u32 = 5837;
pub const CAPABILITY_ARBITRARYPRECISIONINTEGERSALTERA: u32 = 5844;
pub const CAPABILITY_ARBITRARYPRECISIONFLOATINGPOINTALTERA: u32 = 5845;
pub const CAPABILITY_UNSTRUCTUREDLOOPCONTROLSINTEL: u32 = 5886;
pub const CAPABILITY_FPGALOOPCONTROLSALTERA: u32 = 5888;
pub const CAPABILITY_KERNELATTRIBUTESINTEL: u32 = 5892;
pub const CAPABILITY_FPGAKERNELATTRIBUTESINTEL: u32 = 5897;
pub const CAPABILITY_FPGAMEMORYACCESSESALTERA: u32 = 5898;
pub const CAPABILITY_FPGACLUSTERATTRIBUTESALTERA: u32 = 5904;
pub const CAPABILITY_LOOPFUSEALTERA: u32 = 5906;
pub const CAPABILITY_FPGADSPCONTROLALTERA: u32 = 5908;
pub const CAPABILITY_MEMORYACCESSALIASINGINTEL: u32 = 5910;
pub const CAPABILITY_FPGAINVOCATIONPIPELININGATTRIBUTESALTERA: u32 = 5916;
pub const CAPABILITY_FPGABUFFERLOCATIONALTERA: u32 = 5920;
pub const CAPABILITY_ARBITRARYPRECISIONFIXEDPOINTALTERA: u32 = 5922;
pub const CAPABILITY_USMSTORAGECLASSESALTERA: u32 = 5935;
pub const CAPABILITY_RUNTIMEALIGNEDATTRIBUTEALTERA: u32 = 5939;
pub const CAPABILITY_IOPIPESALTERA: u32 = 5943;
pub const CAPABILITY_BLOCKINGPIPESALTERA: u32 = 5945;
pub const CAPABILITY_FPGAREGALTERA: u32 = 5948;
pub const CAPABILITY_DOTPRODUCTINPUTALL: u32 = 6016;
pub const CAPABILITY_DOTPRODUCTINPUT4X8BIT: u32 = 6017;
pub const CAPABILITY_DOTPRODUCTINPUT4X8BITPACKED: u32 = 6018;
pub const CAPABILITY_DOTPRODUCT: u32 = 6019;
pub const CAPABILITY_RAYCULLMASKKHR: u32 = 6020;
pub const CAPABILITY_COOPERATIVEMATRIXKHR: u32 = 6022;
pub const CAPABILITY_REPLICATEDCOMPOSITESEXT: u32 = 6024;
pub const CAPABILITY_BITINSTRUCTIONS: u32 = 6025;
pub const CAPABILITY_GROUPNONUNIFORMROTATEKHR: u32 = 6026;
pub const CAPABILITY_FLOATCONTROLS2: u32 = 6029;
pub const CAPABILITY_FMAKHR: u32 = 6030;
pub const CAPABILITY_RAYTRACINGOPACITYMICROMAPEXECUTIONMODEKHR: u32 = 6032;
pub const CAPABILITY_ATOMICFLOAT32ADDEXT: u32 = 6033;
pub const CAPABILITY_ATOMICFLOAT64ADDEXT: u32 = 6034;
pub const CAPABILITY_LONGCOMPOSITESINTEL: u32 = 6089;
pub const CAPABILITY_OPTNONEEXT: u32 = 6094;
pub const CAPABILITY_ATOMICFLOAT16ADDEXT: u32 = 6095;
pub const CAPABILITY_DEBUGINFOMODULEINTEL: u32 = 6114;
pub const CAPABILITY_BFLOAT16CONVERSIONINTEL: u32 = 6115;
pub const CAPABILITY_SPLITBARRIEREXT: u32 = 6141;
pub const CAPABILITY_ARITHMETICFENCEEXT: u32 = 6144;
pub const CAPABILITY_FPGACLUSTERATTRIBUTESV2ALTERA: u32 = 6150;
pub const CAPABILITY_FPGAKERNELATTRIBUTESV2INTEL: u32 = 6161;
pub const CAPABILITY_TASKSEQUENCEALTERA: u32 = 6162;
pub const CAPABILITY_FPMAXERRORINTEL: u32 = 6169;
pub const CAPABILITY_FPGALATENCYCONTROLALTERA: u32 = 6171;
pub const CAPABILITY_FPGAARGUMENTINTERFACESALTERA: u32 = 6174;
pub const CAPABILITY_DEVICEBARRIERINTEL: u32 = 6185;
pub const CAPABILITY_GLOBALVARIABLEHOSTACCESSINTEL: u32 = 6187;
pub const CAPABILITY_GLOBALVARIABLEFPGADECORATIONSALTERA: u32 = 6189;
pub const CAPABILITY_SUBGROUPBITCASTSHUFFLEINTEL: u32 = 6207;
pub const CAPABILITY_SUBGROUPBUFFERPREFETCHINTEL: u32 = 6220;
pub const CAPABILITY_SUBGROUP2DBLOCKIOINTEL: u32 = 6228;
pub const CAPABILITY_SUBGROUP2DBLOCKTRANSFORMINTEL: u32 = 6229;
pub const CAPABILITY_SUBGROUP2DBLOCKTRANSPOSEINTEL: u32 = 6230;
pub const CAPABILITY_SUBGROUPMATRIXMULTIPLYACCUMULATEINTEL: u32 = 6236;
pub const CAPABILITY_TERNARYBITWISEFUNCTIONINTEL: u32 = 6241;
pub const CAPABILITY_UNTYPEDVARIABLELENGTHARRAYINTEL: u32 = 6243;
pub const CAPABILITY_SPECCONDITIONALINTEL: u32 = 6245;
pub const CAPABILITY_FUNCTIONVARIANTSINTEL: u32 = 6246;
pub const CAPABILITY_PREDICATEDIOINTEL: u32 = 6257;
pub const CAPABILITY_ROUNDEDDIVIDESQRTINTEL: u32 = 6265;
pub const CAPABILITY_GROUPUNIFORMARITHMETICKHR: u32 = 6400;
pub const CAPABILITY_TENSORFLOAT32ROUNDINGINTEL: u32 = 6425;
pub const CAPABILITY_MASKEDGATHERSCATTERINTEL: u32 = 6427;
pub const CAPABILITY_CACHECONTROLSINTEL: u32 = 6441;
pub const CAPABILITY_REGISTERLIMITSINTEL: u32 = 6460;
pub const CAPABILITY_BINDLESSIMAGESINTEL: u32 = 6528;
pub const CAPABILITY_DOTPRODUCTFLOAT16ACCFLOAT32VALVE: u32 = 6912;
pub const CAPABILITY_DOTPRODUCTFLOAT16ACCFLOAT16VALVE: u32 = 6913;
pub const CAPABILITY_DOTPRODUCTBFLOAT16ACCVALVE: u32 = 6914;
pub const CAPABILITY_DOTPRODUCTFLOAT8ACCFLOAT32VALVE: u32 = 6915;
pub const CAPABILITY_INTRINSICSAMSUNG: u32 = 7041;
/// `Capability` enumerant value for a textual name.
#[must_use]
pub fn capability_value(name: &str) -> Option<u32> {
    let value = match name {
        "Matrix" => 0,
        "Shader" => 1,
        "Geometry" => 2,
        "Tessellation" => 3,
        "Addresses" => 4,
        "Linkage" => 5,
        "Kernel" => 6,
        "Vector16" => 7,
        "Float16Buffer" => 8,
        "Float16" => 9,
        "Float64" => 10,
        "Int64" => 11,
        "Int64Atomics" => 12,
        "ImageBasic" => 13,
        "ImageReadWrite" => 14,
        "ImageMipmap" => 15,
        "Pipes" => 17,
        "Groups" => 18,
        "DeviceEnqueue" => 19,
        "LiteralSampler" => 20,
        "AtomicStorage" => 21,
        "Int16" => 22,
        "TessellationPointSize" => 23,
        "GeometryPointSize" => 24,
        "ImageGatherExtended" => 25,
        "StorageImageMultisample" => 27,
        "UniformBufferArrayDynamicIndexing" => 28,
        "SampledImageArrayDynamicIndexing" => 29,
        "StorageBufferArrayDynamicIndexing" => 30,
        "StorageImageArrayDynamicIndexing" => 31,
        "ClipDistance" => 32,
        "CullDistance" => 33,
        "ImageCubeArray" => 34,
        "SampleRateShading" => 35,
        "ImageRect" => 36,
        "SampledRect" => 37,
        "GenericPointer" => 38,
        "Int8" => 39,
        "InputAttachment" => 40,
        "SparseResidency" => 41,
        "MinLod" => 42,
        "Sampled1D" => 43,
        "Image1D" => 44,
        "SampledCubeArray" => 45,
        "SampledBuffer" => 46,
        "ImageBuffer" => 47,
        "ImageMSArray" => 48,
        "StorageImageExtendedFormats" => 49,
        "ImageQuery" => 50,
        "DerivativeControl" => 51,
        "InterpolationFunction" => 52,
        "TransformFeedback" => 53,
        "GeometryStreams" => 54,
        "StorageImageReadWithoutFormat" => 55,
        "StorageImageWriteWithoutFormat" => 56,
        "MultiViewport" => 57,
        "SubgroupDispatch" => 58,
        "NamedBarrier" => 59,
        "PipeStorage" => 60,
        "GroupNonUniform" => 61,
        "GroupNonUniformVote" => 62,
        "GroupNonUniformArithmetic" => 63,
        "GroupNonUniformBallot" => 64,
        "GroupNonUniformShuffle" => 65,
        "GroupNonUniformShuffleRelative" => 66,
        "GroupNonUniformClustered" => 67,
        "GroupNonUniformQuad" => 68,
        "ShaderLayer" => 69,
        "ShaderViewportIndex" => 70,
        "UniformDecoration" => 71,
        "CoreBuiltinsARM" => 4165,
        "TileImageColorReadAccessEXT" => 4166,
        "TileImageDepthReadAccessEXT" => 4167,
        "TileImageStencilReadAccessEXT" => 4168,
        "TensorsARM" => 4174,
        "StorageTensorArrayDynamicIndexingARM" => 4175,
        "StorageTensorArrayNonUniformIndexingARM" => 4176,
        "GraphARM" => 4191,
        "CooperativeMatrixLayoutsARM" => 4201,
        "Float8EXT" => 4212,
        "Float8CooperativeMatrixEXT" => 4213,
        "Float6EXT" => 4228,
        "Float4EXT" => 4229,
        "Float8UnsignedE8M0EXT" => 4230,
        "MXInt8EXT" => 4231,
        "BitcastExtractEXT" => 4232,
        "FragmentShadingRateKHR" => 4422,
        "SubgroupBallotKHR" => 4423,
        "DrawParameters" => 4427,
        "WorkgroupMemoryExplicitLayoutKHR" => 4428,
        "WorkgroupMemoryExplicitLayout8BitAccessKHR" => 4429,
        "WorkgroupMemoryExplicitLayout16BitAccessKHR" => 4430,
        "SubgroupVoteKHR" => 4431,
        "StorageBuffer16BitAccess" => 4433,
        "UniformAndStorageBuffer16BitAccess" => 4434,
        "StoragePushConstant16" => 4435,
        "StorageInputOutput16" => 4436,
        "DeviceGroup" => 4437,
        "MultiView" => 4439,
        "VariablePointersStorageBuffer" => 4441,
        "VariablePointers" => 4442,
        "AtomicStorageOps" => 4445,
        "SampleMaskPostDepthCoverage" => 4447,
        "StorageBuffer8BitAccess" => 4448,
        "UniformAndStorageBuffer8BitAccess" => 4449,
        "StoragePushConstant8" => 4450,
        "DenormPreserve" => 4464,
        "DenormFlushToZero" => 4465,
        "SignedZeroInfNanPreserve" => 4466,
        "RoundingModeRTE" => 4467,
        "RoundingModeRTZ" => 4468,
        "RayQueryProvisionalKHR" => 4471,
        "RayQueryKHR" => 4472,
        "UntypedPointersKHR" => 4473,
        "RayTraversalPrimitiveCullingKHR" => 4478,
        "RayTracingKHR" => 4479,
        "TextureSampleWeightedQCOM" => 4484,
        "TextureBoxFilterQCOM" => 4485,
        "TextureBlockMatchQCOM" => 4486,
        "TileShadingQCOM" => 4495,
        "CooperativeMatrixConversionQCOM" => 4496,
        "TextureBlockMatch2QCOM" => 4498,
        "BFloat16MulAddQCOM" => 4504,
        "SubgroupSizeQCOM" => 4506,
        "MultipleWaitQueuesQCOM" => 4539,
        "ImageGatherLinearQCOM" => 4543,
        "ImageGatherExtendedModesQCOM" => 4544,
        "Float16ImageAMD" => 5008,
        "ImageGatherBiasLodAMD" => 5009,
        "FragmentMaskAMD" => 5010,
        "StencilExportEXT" => 5013,
        "ImageReadWriteLodAMD" => 5015,
        "Int64ImageEXT" => 5016,
        "ShaderClockKHR" => 5055,
        "ShaderEnqueueAMDX" => 5067,
        "QuadControlKHR" => 5087,
        "Int4TypeINTEL" => 5112,
        "Int4CooperativeMatrixINTEL" => 5114,
        "BFloat16TypeKHR" => 5116,
        "BFloat16DotProductKHR" => 5117,
        "BFloat16CooperativeMatrixKHR" => 5118,
        "AbortKHR" => 5120,
        "DescriptorHeapEXT" => 5128,
        "ConstantDataKHR" => 5146,
        "PoisonFreezeKHR" => 5156,
        "WeakLinkageAMD" => 5181,
        "SampleMaskOverrideCoverageNV" => 5249,
        "GeometryShaderPassthroughNV" => 5251,
        "ShaderViewportIndexLayerEXT" => 5254,
        "ShaderViewportMaskNV" => 5255,
        "ShaderStereoViewNV" => 5259,
        "PerViewAttributesNV" => 5260,
        "FragmentFullyCoveredEXT" => 5265,
        "MeshShadingNV" => 5266,
        "ImageFootprintNV" => 5282,
        "MeshShadingEXT" => 5283,
        "FragmentBarycentricKHR" => 5284,
        "ComputeDerivativeGroupQuadsKHR" => 5288,
        "FragmentDensityEXT" => 5291,
        "GroupNonUniformPartitionedEXT" => 5297,
        "ShaderNonUniform" => 5301,
        "RuntimeDescriptorArray" => 5302,
        "InputAttachmentArrayDynamicIndexing" => 5303,
        "UniformTexelBufferArrayDynamicIndexing" => 5304,
        "StorageTexelBufferArrayDynamicIndexing" => 5305,
        "UniformBufferArrayNonUniformIndexing" => 5306,
        "SampledImageArrayNonUniformIndexing" => 5307,
        "StorageBufferArrayNonUniformIndexing" => 5308,
        "StorageImageArrayNonUniformIndexing" => 5309,
        "InputAttachmentArrayNonUniformIndexing" => 5310,
        "UniformTexelBufferArrayNonUniformIndexing" => 5311,
        "StorageTexelBufferArrayNonUniformIndexing" => 5312,
        "RayTracingPositionFetchKHR" => 5336,
        "RayTracingNV" => 5340,
        "RayTracingMotionBlurNV" => 5341,
        "VulkanMemoryModel" => 5345,
        "VulkanMemoryModelDeviceScope" => 5346,
        "PhysicalStorageBufferAddresses" => 5347,
        "ComputeDerivativeGroupLinearKHR" => 5350,
        "RayTracingProvisionalKHR" => 5353,
        "CooperativeMatrixNV" => 5357,
        "FragmentShaderSampleInterlockEXT" => 5363,
        "FragmentShaderShadingRateInterlockEXT" => 5372,
        "ShaderSMBuiltinsNV" => 5373,
        "FragmentShaderPixelInterlockEXT" => 5378,
        "DemoteToHelperInvocation" => 5379,
        "DisplacementMicromapNV" => 5380,
        "RayTracingOpacityMicromapKHR" => 5381,
        "ShaderInvocationReorderNV" => 5383,
        "ShaderInvocationReorderEXT" => 5388,
        "BindlessTextureNV" => 5390,
        "RayQueryPositionFetchKHR" => 5391,
        "CooperativeVectorNV" => 5394,
        "AtomicFloat16VectorNV" => 5404,
        "RayTracingDisplacementMicromapNV" => 5409,
        "RawAccessChainsNV" => 5414,
        "RayTracingSpheresGeometryNV" => 5418,
        "RayTracingLinearSweptSpheresGeometryNV" => 5419,
        "PushConstantBanksNV" => 5423,
        "LongVectorEXT" => 5425,
        "Shader64BitIndexingEXT" => 5426,
        "CooperativeMatrixConversionsEXT" => 5429,
        "CooperativeMatrixReductionsEXT" => 5430,
        "CooperativeMatrixConversionsNV" => 5431,
        "CooperativeMatrixPerElementOperationsEXT" => 5432,
        "CooperativeMatrixTensorAddressingNV" => 5433,
        "CooperativeMatrixBlockLoadsNV" => 5434,
        "CooperativeVectorTrainingNV" => 5435,
        "RayTracingClusterAccelerationStructureNV" => 5437,
        "CooperativeMatrixGetCoordinateEXT" => 5438,
        "TensorAddressingNV" => 5439,
        "CooperativeMatrixDecodeVectorNV" => 5447,
        "SubgroupShuffleINTEL" => 5568,
        "SubgroupBufferBlockIOINTEL" => 5569,
        "SubgroupImageBlockIOINTEL" => 5570,
        "SubgroupImageMediaBlockIOINTEL" => 5579,
        "RoundToInfinityINTEL" => 5582,
        "FloatingPointModeINTEL" => 5583,
        "IntegerFunctions2INTEL" => 5584,
        "FunctionPointersINTEL" => 5603,
        "IndirectReferencesINTEL" => 5604,
        "AsmINTEL" => 5606,
        "AtomicFloat32MinMaxEXT" => 5612,
        "AtomicFloat64MinMaxEXT" => 5613,
        "AtomicFloat16MinMaxEXT" => 5616,
        "VectorComputeINTEL" => 5617,
        "VectorAnyINTEL" => 5619,
        "ExpectAssumeKHR" => 5629,
        "SubgroupAvcMotionEstimationINTEL" => 5696,
        "SubgroupAvcMotionEstimationIntraINTEL" => 5697,
        "SubgroupAvcMotionEstimationChromaINTEL" => 5698,
        "VariableLengthArrayINTEL" => 5817,
        "FunctionFloatControlINTEL" => 5821,
        "FPGAMemoryAttributesALTERA" => 5824,
        "FPFastMathModeINTEL" => 5837,
        "ArbitraryPrecisionIntegersALTERA" => 5844,
        "ArbitraryPrecisionFloatingPointALTERA" => 5845,
        "UnstructuredLoopControlsINTEL" => 5886,
        "FPGALoopControlsALTERA" => 5888,
        "KernelAttributesINTEL" => 5892,
        "FPGAKernelAttributesINTEL" => 5897,
        "FPGAMemoryAccessesALTERA" => 5898,
        "FPGAClusterAttributesALTERA" => 5904,
        "LoopFuseALTERA" => 5906,
        "FPGADSPControlALTERA" => 5908,
        "MemoryAccessAliasingINTEL" => 5910,
        "FPGAInvocationPipeliningAttributesALTERA" => 5916,
        "FPGABufferLocationALTERA" => 5920,
        "ArbitraryPrecisionFixedPointALTERA" => 5922,
        "USMStorageClassesALTERA" => 5935,
        "RuntimeAlignedAttributeALTERA" => 5939,
        "IOPipesALTERA" => 5943,
        "BlockingPipesALTERA" => 5945,
        "FPGARegALTERA" => 5948,
        "DotProductInputAll" => 6016,
        "DotProductInput4x8Bit" => 6017,
        "DotProductInput4x8BitPacked" => 6018,
        "DotProduct" => 6019,
        "RayCullMaskKHR" => 6020,
        "CooperativeMatrixKHR" => 6022,
        "ReplicatedCompositesEXT" => 6024,
        "BitInstructions" => 6025,
        "GroupNonUniformRotateKHR" => 6026,
        "FloatControls2" => 6029,
        "FMAKHR" => 6030,
        "RayTracingOpacityMicromapExecutionModeKHR" => 6032,
        "AtomicFloat32AddEXT" => 6033,
        "AtomicFloat64AddEXT" => 6034,
        "LongCompositesINTEL" => 6089,
        "OptNoneEXT" => 6094,
        "AtomicFloat16AddEXT" => 6095,
        "DebugInfoModuleINTEL" => 6114,
        "BFloat16ConversionINTEL" => 6115,
        "SplitBarrierEXT" => 6141,
        "ArithmeticFenceEXT" => 6144,
        "FPGAClusterAttributesV2ALTERA" => 6150,
        "FPGAKernelAttributesv2INTEL" => 6161,
        "TaskSequenceALTERA" => 6162,
        "FPMaxErrorINTEL" => 6169,
        "FPGALatencyControlALTERA" => 6171,
        "FPGAArgumentInterfacesALTERA" => 6174,
        "DeviceBarrierINTEL" => 6185,
        "GlobalVariableHostAccessINTEL" => 6187,
        "GlobalVariableFPGADecorationsALTERA" => 6189,
        "SubgroupBitcastShuffleINTEL" => 6207,
        "SubgroupBufferPrefetchINTEL" => 6220,
        "Subgroup2DBlockIOINTEL" => 6228,
        "Subgroup2DBlockTransformINTEL" => 6229,
        "Subgroup2DBlockTransposeINTEL" => 6230,
        "SubgroupMatrixMultiplyAccumulateINTEL" => 6236,
        "TernaryBitwiseFunctionINTEL" => 6241,
        "UntypedVariableLengthArrayINTEL" => 6243,
        "SpecConditionalINTEL" => 6245,
        "FunctionVariantsINTEL" => 6246,
        "PredicatedIOINTEL" => 6257,
        "RoundedDivideSqrtINTEL" => 6265,
        "GroupUniformArithmeticKHR" => 6400,
        "TensorFloat32RoundingINTEL" => 6425,
        "MaskedGatherScatterINTEL" => 6427,
        "CacheControlsINTEL" => 6441,
        "RegisterLimitsINTEL" => 6460,
        "BindlessImagesINTEL" => 6528,
        "DotProductFloat16AccFloat32VALVE" => 6912,
        "DotProductFloat16AccFloat16VALVE" => 6913,
        "DotProductBFloat16AccVALVE" => 6914,
        "DotProductFloat8AccFloat32VALVE" => 6915,
        "IntrinsicSAMSUNG" => 7041,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `Capability` value known to the backend.
#[must_use]
pub const fn capability_name(value: u32) -> &'static str {
    match value {
        0 => "Matrix",
        1 => "Shader",
        2 => "Geometry",
        3 => "Tessellation",
        4 => "Addresses",
        5 => "Linkage",
        6 => "Kernel",
        7 => "Vector16",
        8 => "Float16Buffer",
        9 => "Float16",
        10 => "Float64",
        11 => "Int64",
        12 => "Int64Atomics",
        13 => "ImageBasic",
        14 => "ImageReadWrite",
        15 => "ImageMipmap",
        17 => "Pipes",
        18 => "Groups",
        19 => "DeviceEnqueue",
        20 => "LiteralSampler",
        21 => "AtomicStorage",
        22 => "Int16",
        23 => "TessellationPointSize",
        24 => "GeometryPointSize",
        25 => "ImageGatherExtended",
        27 => "StorageImageMultisample",
        28 => "UniformBufferArrayDynamicIndexing",
        29 => "SampledImageArrayDynamicIndexing",
        30 => "StorageBufferArrayDynamicIndexing",
        31 => "StorageImageArrayDynamicIndexing",
        32 => "ClipDistance",
        33 => "CullDistance",
        34 => "ImageCubeArray",
        35 => "SampleRateShading",
        36 => "ImageRect",
        37 => "SampledRect",
        38 => "GenericPointer",
        39 => "Int8",
        40 => "InputAttachment",
        41 => "SparseResidency",
        42 => "MinLod",
        43 => "Sampled1D",
        44 => "Image1D",
        45 => "SampledCubeArray",
        46 => "SampledBuffer",
        47 => "ImageBuffer",
        48 => "ImageMSArray",
        49 => "StorageImageExtendedFormats",
        50 => "ImageQuery",
        51 => "DerivativeControl",
        52 => "InterpolationFunction",
        53 => "TransformFeedback",
        54 => "GeometryStreams",
        55 => "StorageImageReadWithoutFormat",
        56 => "StorageImageWriteWithoutFormat",
        57 => "MultiViewport",
        58 => "SubgroupDispatch",
        59 => "NamedBarrier",
        60 => "PipeStorage",
        61 => "GroupNonUniform",
        62 => "GroupNonUniformVote",
        63 => "GroupNonUniformArithmetic",
        64 => "GroupNonUniformBallot",
        65 => "GroupNonUniformShuffle",
        66 => "GroupNonUniformShuffleRelative",
        67 => "GroupNonUniformClustered",
        68 => "GroupNonUniformQuad",
        69 => "ShaderLayer",
        70 => "ShaderViewportIndex",
        71 => "UniformDecoration",
        4165 => "CoreBuiltinsARM",
        4166 => "TileImageColorReadAccessEXT",
        4167 => "TileImageDepthReadAccessEXT",
        4168 => "TileImageStencilReadAccessEXT",
        4174 => "TensorsARM",
        4175 => "StorageTensorArrayDynamicIndexingARM",
        4176 => "StorageTensorArrayNonUniformIndexingARM",
        4191 => "GraphARM",
        4201 => "CooperativeMatrixLayoutsARM",
        4212 => "Float8EXT",
        4213 => "Float8CooperativeMatrixEXT",
        4228 => "Float6EXT",
        4229 => "Float4EXT",
        4230 => "Float8UnsignedE8M0EXT",
        4231 => "MXInt8EXT",
        4232 => "BitcastExtractEXT",
        4422 => "FragmentShadingRateKHR",
        4423 => "SubgroupBallotKHR",
        4427 => "DrawParameters",
        4428 => "WorkgroupMemoryExplicitLayoutKHR",
        4429 => "WorkgroupMemoryExplicitLayout8BitAccessKHR",
        4430 => "WorkgroupMemoryExplicitLayout16BitAccessKHR",
        4431 => "SubgroupVoteKHR",
        4433 => "StorageBuffer16BitAccess",
        4434 => "UniformAndStorageBuffer16BitAccess",
        4435 => "StoragePushConstant16",
        4436 => "StorageInputOutput16",
        4437 => "DeviceGroup",
        4439 => "MultiView",
        4441 => "VariablePointersStorageBuffer",
        4442 => "VariablePointers",
        4445 => "AtomicStorageOps",
        4447 => "SampleMaskPostDepthCoverage",
        4448 => "StorageBuffer8BitAccess",
        4449 => "UniformAndStorageBuffer8BitAccess",
        4450 => "StoragePushConstant8",
        4464 => "DenormPreserve",
        4465 => "DenormFlushToZero",
        4466 => "SignedZeroInfNanPreserve",
        4467 => "RoundingModeRTE",
        4468 => "RoundingModeRTZ",
        4471 => "RayQueryProvisionalKHR",
        4472 => "RayQueryKHR",
        4473 => "UntypedPointersKHR",
        4478 => "RayTraversalPrimitiveCullingKHR",
        4479 => "RayTracingKHR",
        4484 => "TextureSampleWeightedQCOM",
        4485 => "TextureBoxFilterQCOM",
        4486 => "TextureBlockMatchQCOM",
        4495 => "TileShadingQCOM",
        4496 => "CooperativeMatrixConversionQCOM",
        4498 => "TextureBlockMatch2QCOM",
        4504 => "BFloat16MulAddQCOM",
        4506 => "SubgroupSizeQCOM",
        4539 => "MultipleWaitQueuesQCOM",
        4543 => "ImageGatherLinearQCOM",
        4544 => "ImageGatherExtendedModesQCOM",
        5008 => "Float16ImageAMD",
        5009 => "ImageGatherBiasLodAMD",
        5010 => "FragmentMaskAMD",
        5013 => "StencilExportEXT",
        5015 => "ImageReadWriteLodAMD",
        5016 => "Int64ImageEXT",
        5055 => "ShaderClockKHR",
        5067 => "ShaderEnqueueAMDX",
        5087 => "QuadControlKHR",
        5112 => "Int4TypeINTEL",
        5114 => "Int4CooperativeMatrixINTEL",
        5116 => "BFloat16TypeKHR",
        5117 => "BFloat16DotProductKHR",
        5118 => "BFloat16CooperativeMatrixKHR",
        5120 => "AbortKHR",
        5128 => "DescriptorHeapEXT",
        5146 => "ConstantDataKHR",
        5156 => "PoisonFreezeKHR",
        5181 => "WeakLinkageAMD",
        5249 => "SampleMaskOverrideCoverageNV",
        5251 => "GeometryShaderPassthroughNV",
        5254 => "ShaderViewportIndexLayerEXT",
        5255 => "ShaderViewportMaskNV",
        5259 => "ShaderStereoViewNV",
        5260 => "PerViewAttributesNV",
        5265 => "FragmentFullyCoveredEXT",
        5266 => "MeshShadingNV",
        5282 => "ImageFootprintNV",
        5283 => "MeshShadingEXT",
        5284 => "FragmentBarycentricKHR",
        5288 => "ComputeDerivativeGroupQuadsKHR",
        5291 => "FragmentDensityEXT",
        5297 => "GroupNonUniformPartitionedEXT",
        5301 => "ShaderNonUniform",
        5302 => "RuntimeDescriptorArray",
        5303 => "InputAttachmentArrayDynamicIndexing",
        5304 => "UniformTexelBufferArrayDynamicIndexing",
        5305 => "StorageTexelBufferArrayDynamicIndexing",
        5306 => "UniformBufferArrayNonUniformIndexing",
        5307 => "SampledImageArrayNonUniformIndexing",
        5308 => "StorageBufferArrayNonUniformIndexing",
        5309 => "StorageImageArrayNonUniformIndexing",
        5310 => "InputAttachmentArrayNonUniformIndexing",
        5311 => "UniformTexelBufferArrayNonUniformIndexing",
        5312 => "StorageTexelBufferArrayNonUniformIndexing",
        5336 => "RayTracingPositionFetchKHR",
        5340 => "RayTracingNV",
        5341 => "RayTracingMotionBlurNV",
        5345 => "VulkanMemoryModel",
        5346 => "VulkanMemoryModelDeviceScope",
        5347 => "PhysicalStorageBufferAddresses",
        5350 => "ComputeDerivativeGroupLinearKHR",
        5353 => "RayTracingProvisionalKHR",
        5357 => "CooperativeMatrixNV",
        5363 => "FragmentShaderSampleInterlockEXT",
        5372 => "FragmentShaderShadingRateInterlockEXT",
        5373 => "ShaderSMBuiltinsNV",
        5378 => "FragmentShaderPixelInterlockEXT",
        5379 => "DemoteToHelperInvocation",
        5380 => "DisplacementMicromapNV",
        5381 => "RayTracingOpacityMicromapKHR",
        5383 => "ShaderInvocationReorderNV",
        5388 => "ShaderInvocationReorderEXT",
        5390 => "BindlessTextureNV",
        5391 => "RayQueryPositionFetchKHR",
        5394 => "CooperativeVectorNV",
        5404 => "AtomicFloat16VectorNV",
        5409 => "RayTracingDisplacementMicromapNV",
        5414 => "RawAccessChainsNV",
        5418 => "RayTracingSpheresGeometryNV",
        5419 => "RayTracingLinearSweptSpheresGeometryNV",
        5423 => "PushConstantBanksNV",
        5425 => "LongVectorEXT",
        5426 => "Shader64BitIndexingEXT",
        5429 => "CooperativeMatrixConversionsEXT",
        5430 => "CooperativeMatrixReductionsEXT",
        5431 => "CooperativeMatrixConversionsNV",
        5432 => "CooperativeMatrixPerElementOperationsEXT",
        5433 => "CooperativeMatrixTensorAddressingNV",
        5434 => "CooperativeMatrixBlockLoadsNV",
        5435 => "CooperativeVectorTrainingNV",
        5437 => "RayTracingClusterAccelerationStructureNV",
        5438 => "CooperativeMatrixGetCoordinateEXT",
        5439 => "TensorAddressingNV",
        5447 => "CooperativeMatrixDecodeVectorNV",
        5568 => "SubgroupShuffleINTEL",
        5569 => "SubgroupBufferBlockIOINTEL",
        5570 => "SubgroupImageBlockIOINTEL",
        5579 => "SubgroupImageMediaBlockIOINTEL",
        5582 => "RoundToInfinityINTEL",
        5583 => "FloatingPointModeINTEL",
        5584 => "IntegerFunctions2INTEL",
        5603 => "FunctionPointersINTEL",
        5604 => "IndirectReferencesINTEL",
        5606 => "AsmINTEL",
        5612 => "AtomicFloat32MinMaxEXT",
        5613 => "AtomicFloat64MinMaxEXT",
        5616 => "AtomicFloat16MinMaxEXT",
        5617 => "VectorComputeINTEL",
        5619 => "VectorAnyINTEL",
        5629 => "ExpectAssumeKHR",
        5696 => "SubgroupAvcMotionEstimationINTEL",
        5697 => "SubgroupAvcMotionEstimationIntraINTEL",
        5698 => "SubgroupAvcMotionEstimationChromaINTEL",
        5817 => "VariableLengthArrayINTEL",
        5821 => "FunctionFloatControlINTEL",
        5824 => "FPGAMemoryAttributesALTERA",
        5837 => "FPFastMathModeINTEL",
        5844 => "ArbitraryPrecisionIntegersALTERA",
        5845 => "ArbitraryPrecisionFloatingPointALTERA",
        5886 => "UnstructuredLoopControlsINTEL",
        5888 => "FPGALoopControlsALTERA",
        5892 => "KernelAttributesINTEL",
        5897 => "FPGAKernelAttributesINTEL",
        5898 => "FPGAMemoryAccessesALTERA",
        5904 => "FPGAClusterAttributesALTERA",
        5906 => "LoopFuseALTERA",
        5908 => "FPGADSPControlALTERA",
        5910 => "MemoryAccessAliasingINTEL",
        5916 => "FPGAInvocationPipeliningAttributesALTERA",
        5920 => "FPGABufferLocationALTERA",
        5922 => "ArbitraryPrecisionFixedPointALTERA",
        5935 => "USMStorageClassesALTERA",
        5939 => "RuntimeAlignedAttributeALTERA",
        5943 => "IOPipesALTERA",
        5945 => "BlockingPipesALTERA",
        5948 => "FPGARegALTERA",
        6016 => "DotProductInputAll",
        6017 => "DotProductInput4x8Bit",
        6018 => "DotProductInput4x8BitPacked",
        6019 => "DotProduct",
        6020 => "RayCullMaskKHR",
        6022 => "CooperativeMatrixKHR",
        6024 => "ReplicatedCompositesEXT",
        6025 => "BitInstructions",
        6026 => "GroupNonUniformRotateKHR",
        6029 => "FloatControls2",
        6030 => "FMAKHR",
        6032 => "RayTracingOpacityMicromapExecutionModeKHR",
        6033 => "AtomicFloat32AddEXT",
        6034 => "AtomicFloat64AddEXT",
        6089 => "LongCompositesINTEL",
        6094 => "OptNoneEXT",
        6095 => "AtomicFloat16AddEXT",
        6114 => "DebugInfoModuleINTEL",
        6115 => "BFloat16ConversionINTEL",
        6141 => "SplitBarrierEXT",
        6144 => "ArithmeticFenceEXT",
        6150 => "FPGAClusterAttributesV2ALTERA",
        6161 => "FPGAKernelAttributesv2INTEL",
        6162 => "TaskSequenceALTERA",
        6169 => "FPMaxErrorINTEL",
        6171 => "FPGALatencyControlALTERA",
        6174 => "FPGAArgumentInterfacesALTERA",
        6185 => "DeviceBarrierINTEL",
        6187 => "GlobalVariableHostAccessINTEL",
        6189 => "GlobalVariableFPGADecorationsALTERA",
        6207 => "SubgroupBitcastShuffleINTEL",
        6220 => "SubgroupBufferPrefetchINTEL",
        6228 => "Subgroup2DBlockIOINTEL",
        6229 => "Subgroup2DBlockTransformINTEL",
        6230 => "Subgroup2DBlockTransposeINTEL",
        6236 => "SubgroupMatrixMultiplyAccumulateINTEL",
        6241 => "TernaryBitwiseFunctionINTEL",
        6243 => "UntypedVariableLengthArrayINTEL",
        6245 => "SpecConditionalINTEL",
        6246 => "FunctionVariantsINTEL",
        6257 => "PredicatedIOINTEL",
        6265 => "RoundedDivideSqrtINTEL",
        6400 => "GroupUniformArithmeticKHR",
        6425 => "TensorFloat32RoundingINTEL",
        6427 => "MaskedGatherScatterINTEL",
        6441 => "CacheControlsINTEL",
        6460 => "RegisterLimitsINTEL",
        6528 => "BindlessImagesINTEL",
        6912 => "DotProductFloat16AccFloat32VALVE",
        6913 => "DotProductFloat16AccFloat16VALVE",
        6914 => "DotProductBFloat16AccVALVE",
        6915 => "DotProductFloat8AccFloat32VALVE",
        7041 => "IntrinsicSAMSUNG",
        _ => "",
    }
}

// ---- Decoration ----
pub const DECORATION_RELAXEDPRECISION: u32 = 0;
pub const DECORATION_SPECID: u32 = 1;
pub const DECORATION_BLOCK: u32 = 2;
pub const DECORATION_BUFFERBLOCK: u32 = 3;
pub const DECORATION_ROWMAJOR: u32 = 4;
pub const DECORATION_COLMAJOR: u32 = 5;
pub const DECORATION_ARRAYSTRIDE: u32 = 6;
pub const DECORATION_MATRIXSTRIDE: u32 = 7;
pub const DECORATION_GLSLSHARED: u32 = 8;
pub const DECORATION_GLSLPACKED: u32 = 9;
pub const DECORATION_CPACKED: u32 = 10;
pub const DECORATION_BUILTIN: u32 = 11;
pub const DECORATION_NOPERSPECTIVE: u32 = 13;
pub const DECORATION_FLAT: u32 = 14;
pub const DECORATION_PATCH: u32 = 15;
pub const DECORATION_CENTROID: u32 = 16;
pub const DECORATION_SAMPLE: u32 = 17;
pub const DECORATION_INVARIANT: u32 = 18;
pub const DECORATION_RESTRICT: u32 = 19;
pub const DECORATION_ALIASED: u32 = 20;
pub const DECORATION_VOLATILE: u32 = 21;
pub const DECORATION_CONSTANT: u32 = 22;
pub const DECORATION_COHERENT: u32 = 23;
pub const DECORATION_NONWRITABLE: u32 = 24;
pub const DECORATION_NONREADABLE: u32 = 25;
pub const DECORATION_UNIFORM: u32 = 26;
pub const DECORATION_UNIFORMID: u32 = 27;
pub const DECORATION_SATURATEDCONVERSION: u32 = 28;
pub const DECORATION_STREAM: u32 = 29;
pub const DECORATION_LOCATION: u32 = 30;
pub const DECORATION_COMPONENT: u32 = 31;
pub const DECORATION_INDEX: u32 = 32;
pub const DECORATION_BINDING: u32 = 33;
pub const DECORATION_DESCRIPTORSET: u32 = 34;
pub const DECORATION_OFFSET: u32 = 35;
pub const DECORATION_XFBBUFFER: u32 = 36;
pub const DECORATION_XFBSTRIDE: u32 = 37;
pub const DECORATION_FUNCPARAMATTR: u32 = 38;
pub const DECORATION_FPROUNDINGMODE: u32 = 39;
pub const DECORATION_FPFASTMATHMODE: u32 = 40;
pub const DECORATION_LINKAGEATTRIBUTES: u32 = 41;
pub const DECORATION_NOCONTRACTION: u32 = 42;
pub const DECORATION_INPUTATTACHMENTINDEX: u32 = 43;
pub const DECORATION_ALIGNMENT: u32 = 44;
pub const DECORATION_MAXBYTEOFFSET: u32 = 45;
pub const DECORATION_ALIGNMENTID: u32 = 46;
pub const DECORATION_MAXBYTEOFFSETID: u32 = 47;
pub const DECORATION_SATURATEDTOLARGESTFLOAT8NORMALCONVERSIONEXT: u32 = 4216;
pub const DECORATION_NOSIGNEDWRAP: u32 = 4469;
pub const DECORATION_NOUNSIGNEDWRAP: u32 = 4470;
pub const DECORATION_WEIGHTTEXTUREQCOM: u32 = 4487;
pub const DECORATION_BLOCKMATCHTEXTUREQCOM: u32 = 4488;
pub const DECORATION_BLOCKMATCHSAMPLERQCOM: u32 = 4499;
pub const DECORATION_EXPLICITINTERPAMD: u32 = 4999;
pub const DECORATION_NODESHARESPAYLOADLIMITSWITHAMDX: u32 = 5019;
pub const DECORATION_NODEMAXPAYLOADSAMDX: u32 = 5020;
pub const DECORATION_TRACKFINISHWRITINGAMDX: u32 = 5078;
pub const DECORATION_PAYLOADNODENAMEAMDX: u32 = 5091;
pub const DECORATION_PAYLOADNODEBASEINDEXAMDX: u32 = 5098;
pub const DECORATION_PAYLOADNODESPARSEARRAYAMDX: u32 = 5099;
pub const DECORATION_PAYLOADNODEARRAYSIZEAMDX: u32 = 5100;
pub const DECORATION_PAYLOADDISPATCHINDIRECTAMDX: u32 = 5105;
pub const DECORATION_ARRAYSTRIDEIDEXT: u32 = 5124;
pub const DECORATION_OFFSETIDEXT: u32 = 5125;
pub const DECORATION_UTFENCODEDKHR: u32 = 5145;
pub const DECORATION_OVERRIDECOVERAGENV: u32 = 5248;
pub const DECORATION_PASSTHROUGHNV: u32 = 5250;
pub const DECORATION_VIEWPORTRELATIVENV: u32 = 5252;
pub const DECORATION_SECONDARYVIEWPORTRELATIVENV: u32 = 5256;
pub const DECORATION_PERPRIMITIVEEXT: u32 = 5271;
pub const DECORATION_PERVIEWNV: u32 = 5272;
pub const DECORATION_PERTASKNV: u32 = 5273;
pub const DECORATION_PERVERTEXKHR: u32 = 5285;
pub const DECORATION_NONUNIFORM: u32 = 5300;
pub const DECORATION_RESTRICTPOINTER: u32 = 5355;
pub const DECORATION_ALIASEDPOINTER: u32 = 5356;
pub const DECORATION_MEMBEROFFSETNV: u32 = 5358;
pub const DECORATION_HITOBJECTSHADERRECORDBUFFERNV: u32 = 5386;
pub const DECORATION_HITOBJECTSHADERRECORDBUFFEREXT: u32 = 5389;
pub const DECORATION_BANKNV: u32 = 5397;
pub const DECORATION_BINDLESSSAMPLERNV: u32 = 5398;
pub const DECORATION_BINDLESSIMAGENV: u32 = 5399;
pub const DECORATION_BOUNDSAMPLERNV: u32 = 5400;
pub const DECORATION_BOUNDIMAGENV: u32 = 5401;
pub const DECORATION_COOPERATIVEMATRIXTRANSPOSEEXT: u32 = 5440;
pub const DECORATION_SIMTCALLINTEL: u32 = 5599;
pub const DECORATION_REFERENCEDINDIRECTLYINTEL: u32 = 5602;
pub const DECORATION_CLOBBERINTEL: u32 = 5607;
pub const DECORATION_SIDEEFFECTSINTEL: u32 = 5608;
pub const DECORATION_VECTORCOMPUTEVARIABLEINTEL: u32 = 5624;
pub const DECORATION_FUNCPARAMIOKINDINTEL: u32 = 5625;
pub const DECORATION_VECTORCOMPUTEFUNCTIONINTEL: u32 = 5626;
pub const DECORATION_STACKCALLINTEL: u32 = 5627;
pub const DECORATION_GLOBALVARIABLEOFFSETINTEL: u32 = 5628;
pub const DECORATION_COUNTERBUFFER: u32 = 5634;
pub const DECORATION_USERSEMANTIC: u32 = 5635;
pub const DECORATION_USERTYPEGOOGLE: u32 = 5636;
pub const DECORATION_FUNCTIONROUNDINGMODEINTEL: u32 = 5822;
pub const DECORATION_FUNCTIONDENORMMODEINTEL: u32 = 5823;
pub const DECORATION_REGISTERALTERA: u32 = 5825;
pub const DECORATION_MEMORYALTERA: u32 = 5826;
pub const DECORATION_NUMBANKSALTERA: u32 = 5827;
pub const DECORATION_BANKWIDTHALTERA: u32 = 5828;
pub const DECORATION_MAXPRIVATECOPIESALTERA: u32 = 5829;
pub const DECORATION_SINGLEPUMPALTERA: u32 = 5830;
pub const DECORATION_DOUBLEPUMPALTERA: u32 = 5831;
pub const DECORATION_MAXREPLICATESALTERA: u32 = 5832;
pub const DECORATION_SIMPLEDUALPORTALTERA: u32 = 5833;
pub const DECORATION_MERGEALTERA: u32 = 5834;
pub const DECORATION_BANKBITSALTERA: u32 = 5835;
pub const DECORATION_FORCEPOW2DEPTHALTERA: u32 = 5836;
pub const DECORATION_STRIDESIZEALTERA: u32 = 5883;
pub const DECORATION_WORDSIZEALTERA: u32 = 5884;
pub const DECORATION_TRUEDUALPORTALTERA: u32 = 5885;
pub const DECORATION_BURSTCOALESCEALTERA: u32 = 5899;
pub const DECORATION_CACHESIZEALTERA: u32 = 5900;
pub const DECORATION_DONTSTATICALLYCOALESCEALTERA: u32 = 5901;
pub const DECORATION_PREFETCHALTERA: u32 = 5902;
pub const DECORATION_STALLENABLEALTERA: u32 = 5905;
pub const DECORATION_FUSELOOPSINFUNCTIONALTERA: u32 = 5907;
pub const DECORATION_MATHOPDSPMODEALTERA: u32 = 5909;
pub const DECORATION_ALIASSCOPEINTEL: u32 = 5914;
pub const DECORATION_NOALIASINTEL: u32 = 5915;
pub const DECORATION_INITIATIONINTERVALALTERA: u32 = 5917;
pub const DECORATION_MAXCONCURRENCYALTERA: u32 = 5918;
pub const DECORATION_PIPELINEENABLEALTERA: u32 = 5919;
pub const DECORATION_BUFFERLOCATIONALTERA: u32 = 5921;
pub const DECORATION_IOPIPESTORAGEALTERA: u32 = 5944;
pub const DECORATION_FUNCTIONFLOATINGPOINTMODEINTEL: u32 = 6080;
pub const DECORATION_SINGLEELEMENTVECTORINTEL: u32 = 6085;
pub const DECORATION_VECTORCOMPUTECALLABLEFUNCTIONINTEL: u32 = 6087;
pub const DECORATION_MEDIABLOCKIOINTEL: u32 = 6140;
pub const DECORATION_STALLFREEALTERA: u32 = 6151;
pub const DECORATION_FPMAXERRORDECORATIONINTEL: u32 = 6170;
pub const DECORATION_LATENCYCONTROLLABELALTERA: u32 = 6172;
pub const DECORATION_LATENCYCONTROLCONSTRAINTALTERA: u32 = 6173;
pub const DECORATION_CONDUITKERNELARGUMENTALTERA: u32 = 6175;
pub const DECORATION_REGISTERMAPKERNELARGUMENTALTERA: u32 = 6176;
pub const DECORATION_MMHOSTINTERFACEADDRESSWIDTHALTERA: u32 = 6177;
pub const DECORATION_MMHOSTINTERFACEDATAWIDTHALTERA: u32 = 6178;
pub const DECORATION_MMHOSTINTERFACELATENCYALTERA: u32 = 6179;
pub const DECORATION_MMHOSTINTERFACEREADWRITEMODEALTERA: u32 = 6180;
pub const DECORATION_MMHOSTINTERFACEMAXBURSTALTERA: u32 = 6181;
pub const DECORATION_MMHOSTINTERFACEWAITREQUESTALTERA: u32 = 6182;
pub const DECORATION_STABLEKERNELARGUMENTALTERA: u32 = 6183;
pub const DECORATION_HOSTACCESSINTEL: u32 = 6188;
pub const DECORATION_INITMODEALTERA: u32 = 6190;
pub const DECORATION_IMPLEMENTINREGISTERMAPALTERA: u32 = 6191;
pub const DECORATION_CONDITIONALINTEL: u32 = 6247;
pub const DECORATION_CACHECONTROLLOADINTEL: u32 = 6442;
pub const DECORATION_CACHECONTROLSTOREINTEL: u32 = 6443;
pub const DECORATION_INTRINSICSAMSUNG: u32 = 7040;
/// `Decoration` enumerant value for a textual name.
#[must_use]
pub fn decoration_value(name: &str) -> Option<u32> {
    let value = match name {
        "RelaxedPrecision" => 0,
        "SpecId" => 1,
        "Block" => 2,
        "BufferBlock" => 3,
        "RowMajor" => 4,
        "ColMajor" => 5,
        "ArrayStride" => 6,
        "MatrixStride" => 7,
        "GLSLShared" => 8,
        "GLSLPacked" => 9,
        "CPacked" => 10,
        "BuiltIn" => 11,
        "NoPerspective" => 13,
        "Flat" => 14,
        "Patch" => 15,
        "Centroid" => 16,
        "Sample" => 17,
        "Invariant" => 18,
        "Restrict" => 19,
        "Aliased" => 20,
        "Volatile" => 21,
        "Constant" => 22,
        "Coherent" => 23,
        "NonWritable" => 24,
        "NonReadable" => 25,
        "Uniform" => 26,
        "UniformId" => 27,
        "SaturatedConversion" => 28,
        "Stream" => 29,
        "Location" => 30,
        "Component" => 31,
        "Index" => 32,
        "Binding" => 33,
        "DescriptorSet" => 34,
        "Offset" => 35,
        "XfbBuffer" => 36,
        "XfbStride" => 37,
        "FuncParamAttr" => 38,
        "FPRoundingMode" => 39,
        "FPFastMathMode" => 40,
        "LinkageAttributes" => 41,
        "NoContraction" => 42,
        "InputAttachmentIndex" => 43,
        "Alignment" => 44,
        "MaxByteOffset" => 45,
        "AlignmentId" => 46,
        "MaxByteOffsetId" => 47,
        "SaturatedToLargestFloat8NormalConversionEXT" => 4216,
        "NoSignedWrap" => 4469,
        "NoUnsignedWrap" => 4470,
        "WeightTextureQCOM" => 4487,
        "BlockMatchTextureQCOM" => 4488,
        "BlockMatchSamplerQCOM" => 4499,
        "ExplicitInterpAMD" => 4999,
        "NodeSharesPayloadLimitsWithAMDX" => 5019,
        "NodeMaxPayloadsAMDX" => 5020,
        "TrackFinishWritingAMDX" => 5078,
        "PayloadNodeNameAMDX" => 5091,
        "PayloadNodeBaseIndexAMDX" => 5098,
        "PayloadNodeSparseArrayAMDX" => 5099,
        "PayloadNodeArraySizeAMDX" => 5100,
        "PayloadDispatchIndirectAMDX" => 5105,
        "ArrayStrideIdEXT" => 5124,
        "OffsetIdEXT" => 5125,
        "UTFEncodedKHR" => 5145,
        "OverrideCoverageNV" => 5248,
        "PassthroughNV" => 5250,
        "ViewportRelativeNV" => 5252,
        "SecondaryViewportRelativeNV" => 5256,
        "PerPrimitiveEXT" => 5271,
        "PerViewNV" => 5272,
        "PerTaskNV" => 5273,
        "PerVertexKHR" => 5285,
        "NonUniform" => 5300,
        "RestrictPointer" => 5355,
        "AliasedPointer" => 5356,
        "MemberOffsetNV" => 5358,
        "HitObjectShaderRecordBufferNV" => 5386,
        "HitObjectShaderRecordBufferEXT" => 5389,
        "BankNV" => 5397,
        "BindlessSamplerNV" => 5398,
        "BindlessImageNV" => 5399,
        "BoundSamplerNV" => 5400,
        "BoundImageNV" => 5401,
        "CooperativeMatrixTransposeEXT" => 5440,
        "SIMTCallINTEL" => 5599,
        "ReferencedIndirectlyINTEL" => 5602,
        "ClobberINTEL" => 5607,
        "SideEffectsINTEL" => 5608,
        "VectorComputeVariableINTEL" => 5624,
        "FuncParamIOKindINTEL" => 5625,
        "VectorComputeFunctionINTEL" => 5626,
        "StackCallINTEL" => 5627,
        "GlobalVariableOffsetINTEL" => 5628,
        "CounterBuffer" => 5634,
        "UserSemantic" => 5635,
        "UserTypeGOOGLE" => 5636,
        "FunctionRoundingModeINTEL" => 5822,
        "FunctionDenormModeINTEL" => 5823,
        "RegisterALTERA" => 5825,
        "MemoryALTERA" => 5826,
        "NumbanksALTERA" => 5827,
        "BankwidthALTERA" => 5828,
        "MaxPrivateCopiesALTERA" => 5829,
        "SinglepumpALTERA" => 5830,
        "DoublepumpALTERA" => 5831,
        "MaxReplicatesALTERA" => 5832,
        "SimpleDualPortALTERA" => 5833,
        "MergeALTERA" => 5834,
        "BankBitsALTERA" => 5835,
        "ForcePow2DepthALTERA" => 5836,
        "StridesizeALTERA" => 5883,
        "WordsizeALTERA" => 5884,
        "TrueDualPortALTERA" => 5885,
        "BurstCoalesceALTERA" => 5899,
        "CacheSizeALTERA" => 5900,
        "DontStaticallyCoalesceALTERA" => 5901,
        "PrefetchALTERA" => 5902,
        "StallEnableALTERA" => 5905,
        "FuseLoopsInFunctionALTERA" => 5907,
        "MathOpDSPModeALTERA" => 5909,
        "AliasScopeINTEL" => 5914,
        "NoAliasINTEL" => 5915,
        "InitiationIntervalALTERA" => 5917,
        "MaxConcurrencyALTERA" => 5918,
        "PipelineEnableALTERA" => 5919,
        "BufferLocationALTERA" => 5921,
        "IOPipeStorageALTERA" => 5944,
        "FunctionFloatingPointModeINTEL" => 6080,
        "SingleElementVectorINTEL" => 6085,
        "VectorComputeCallableFunctionINTEL" => 6087,
        "MediaBlockIOINTEL" => 6140,
        "StallFreeALTERA" => 6151,
        "FPMaxErrorDecorationINTEL" => 6170,
        "LatencyControlLabelALTERA" => 6172,
        "LatencyControlConstraintALTERA" => 6173,
        "ConduitKernelArgumentALTERA" => 6175,
        "RegisterMapKernelArgumentALTERA" => 6176,
        "MMHostInterfaceAddressWidthALTERA" => 6177,
        "MMHostInterfaceDataWidthALTERA" => 6178,
        "MMHostInterfaceLatencyALTERA" => 6179,
        "MMHostInterfaceReadWriteModeALTERA" => 6180,
        "MMHostInterfaceMaxBurstALTERA" => 6181,
        "MMHostInterfaceWaitRequestALTERA" => 6182,
        "StableKernelArgumentALTERA" => 6183,
        "HostAccessINTEL" => 6188,
        "InitModeALTERA" => 6190,
        "ImplementInRegisterMapALTERA" => 6191,
        "ConditionalINTEL" => 6247,
        "CacheControlLoadINTEL" => 6442,
        "CacheControlStoreINTEL" => 6443,
        "IntrinsicSAMSUNG" => 7040,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `Decoration` value known to the backend.
#[must_use]
pub const fn decoration_name(value: u32) -> &'static str {
    match value {
        0 => "RelaxedPrecision",
        1 => "SpecId",
        2 => "Block",
        3 => "BufferBlock",
        4 => "RowMajor",
        5 => "ColMajor",
        6 => "ArrayStride",
        7 => "MatrixStride",
        8 => "GLSLShared",
        9 => "GLSLPacked",
        10 => "CPacked",
        11 => "BuiltIn",
        13 => "NoPerspective",
        14 => "Flat",
        15 => "Patch",
        16 => "Centroid",
        17 => "Sample",
        18 => "Invariant",
        19 => "Restrict",
        20 => "Aliased",
        21 => "Volatile",
        22 => "Constant",
        23 => "Coherent",
        24 => "NonWritable",
        25 => "NonReadable",
        26 => "Uniform",
        27 => "UniformId",
        28 => "SaturatedConversion",
        29 => "Stream",
        30 => "Location",
        31 => "Component",
        32 => "Index",
        33 => "Binding",
        34 => "DescriptorSet",
        35 => "Offset",
        36 => "XfbBuffer",
        37 => "XfbStride",
        38 => "FuncParamAttr",
        39 => "FPRoundingMode",
        40 => "FPFastMathMode",
        41 => "LinkageAttributes",
        42 => "NoContraction",
        43 => "InputAttachmentIndex",
        44 => "Alignment",
        45 => "MaxByteOffset",
        46 => "AlignmentId",
        47 => "MaxByteOffsetId",
        4216 => "SaturatedToLargestFloat8NormalConversionEXT",
        4469 => "NoSignedWrap",
        4470 => "NoUnsignedWrap",
        4487 => "WeightTextureQCOM",
        4488 => "BlockMatchTextureQCOM",
        4499 => "BlockMatchSamplerQCOM",
        4999 => "ExplicitInterpAMD",
        5019 => "NodeSharesPayloadLimitsWithAMDX",
        5020 => "NodeMaxPayloadsAMDX",
        5078 => "TrackFinishWritingAMDX",
        5091 => "PayloadNodeNameAMDX",
        5098 => "PayloadNodeBaseIndexAMDX",
        5099 => "PayloadNodeSparseArrayAMDX",
        5100 => "PayloadNodeArraySizeAMDX",
        5105 => "PayloadDispatchIndirectAMDX",
        5124 => "ArrayStrideIdEXT",
        5125 => "OffsetIdEXT",
        5145 => "UTFEncodedKHR",
        5248 => "OverrideCoverageNV",
        5250 => "PassthroughNV",
        5252 => "ViewportRelativeNV",
        5256 => "SecondaryViewportRelativeNV",
        5271 => "PerPrimitiveEXT",
        5272 => "PerViewNV",
        5273 => "PerTaskNV",
        5285 => "PerVertexKHR",
        5300 => "NonUniform",
        5355 => "RestrictPointer",
        5356 => "AliasedPointer",
        5358 => "MemberOffsetNV",
        5386 => "HitObjectShaderRecordBufferNV",
        5389 => "HitObjectShaderRecordBufferEXT",
        5397 => "BankNV",
        5398 => "BindlessSamplerNV",
        5399 => "BindlessImageNV",
        5400 => "BoundSamplerNV",
        5401 => "BoundImageNV",
        5440 => "CooperativeMatrixTransposeEXT",
        5599 => "SIMTCallINTEL",
        5602 => "ReferencedIndirectlyINTEL",
        5607 => "ClobberINTEL",
        5608 => "SideEffectsINTEL",
        5624 => "VectorComputeVariableINTEL",
        5625 => "FuncParamIOKindINTEL",
        5626 => "VectorComputeFunctionINTEL",
        5627 => "StackCallINTEL",
        5628 => "GlobalVariableOffsetINTEL",
        5634 => "CounterBuffer",
        5635 => "UserSemantic",
        5636 => "UserTypeGOOGLE",
        5822 => "FunctionRoundingModeINTEL",
        5823 => "FunctionDenormModeINTEL",
        5825 => "RegisterALTERA",
        5826 => "MemoryALTERA",
        5827 => "NumbanksALTERA",
        5828 => "BankwidthALTERA",
        5829 => "MaxPrivateCopiesALTERA",
        5830 => "SinglepumpALTERA",
        5831 => "DoublepumpALTERA",
        5832 => "MaxReplicatesALTERA",
        5833 => "SimpleDualPortALTERA",
        5834 => "MergeALTERA",
        5835 => "BankBitsALTERA",
        5836 => "ForcePow2DepthALTERA",
        5883 => "StridesizeALTERA",
        5884 => "WordsizeALTERA",
        5885 => "TrueDualPortALTERA",
        5899 => "BurstCoalesceALTERA",
        5900 => "CacheSizeALTERA",
        5901 => "DontStaticallyCoalesceALTERA",
        5902 => "PrefetchALTERA",
        5905 => "StallEnableALTERA",
        5907 => "FuseLoopsInFunctionALTERA",
        5909 => "MathOpDSPModeALTERA",
        5914 => "AliasScopeINTEL",
        5915 => "NoAliasINTEL",
        5917 => "InitiationIntervalALTERA",
        5918 => "MaxConcurrencyALTERA",
        5919 => "PipelineEnableALTERA",
        5921 => "BufferLocationALTERA",
        5944 => "IOPipeStorageALTERA",
        6080 => "FunctionFloatingPointModeINTEL",
        6085 => "SingleElementVectorINTEL",
        6087 => "VectorComputeCallableFunctionINTEL",
        6140 => "MediaBlockIOINTEL",
        6151 => "StallFreeALTERA",
        6170 => "FPMaxErrorDecorationINTEL",
        6172 => "LatencyControlLabelALTERA",
        6173 => "LatencyControlConstraintALTERA",
        6175 => "ConduitKernelArgumentALTERA",
        6176 => "RegisterMapKernelArgumentALTERA",
        6177 => "MMHostInterfaceAddressWidthALTERA",
        6178 => "MMHostInterfaceDataWidthALTERA",
        6179 => "MMHostInterfaceLatencyALTERA",
        6180 => "MMHostInterfaceReadWriteModeALTERA",
        6181 => "MMHostInterfaceMaxBurstALTERA",
        6182 => "MMHostInterfaceWaitRequestALTERA",
        6183 => "StableKernelArgumentALTERA",
        6188 => "HostAccessINTEL",
        6190 => "InitModeALTERA",
        6191 => "ImplementInRegisterMapALTERA",
        6247 => "ConditionalINTEL",
        6442 => "CacheControlLoadINTEL",
        6443 => "CacheControlStoreINTEL",
        7040 => "IntrinsicSAMSUNG",
        _ => "",
    }
}

pub const FUNCTIONCONTROL_NONE: u32 = 0;
pub const FUNCTIONCONTROL_INLINE: u32 = 1;
pub const FUNCTIONCONTROL_DONTINLINE: u32 = 2;
pub const FUNCTIONCONTROL_PURE: u32 = 4;
pub const FUNCTIONCONTROL_CONST: u32 = 8;
pub const FUNCTIONCONTROL_OPTNONEEXT: u32 = 65536;
/// `FunctionControl` enumerant value for a textual name.
#[must_use]
pub fn functioncontrol_value(name: &str) -> Option<u32> {
    let value = match name {
        "None" => 0,
        "Inline" => 1,
        "DontInline" => 2,
        "Pure" => 4,
        "Const" => 8,
        "OptNoneEXT" => 65536,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `FunctionControl` value known to the backend.
#[must_use]
pub const fn functioncontrol_name(value: u32) -> &'static str {
    match value {
        0 => "None",
        1 => "Inline",
        2 => "DontInline",
        4 => "Pure",
        8 => "Const",
        65536 => "OptNoneEXT",
        _ => "",
    }
}

pub const SELECTIONCONTROL_NONE: u32 = 0;
pub const SELECTIONCONTROL_FLATTEN: u32 = 1;
pub const SELECTIONCONTROL_DONTFLATTEN: u32 = 2;
/// `SelectionControl` enumerant value for a textual name.
#[must_use]
pub fn selectioncontrol_value(name: &str) -> Option<u32> {
    let value = match name {
        "None" => 0,
        "Flatten" => 1,
        "DontFlatten" => 2,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `SelectionControl` value known to the backend.
#[must_use]
pub const fn selectioncontrol_name(value: u32) -> &'static str {
    match value {
        0 => "None",
        1 => "Flatten",
        2 => "DontFlatten",
        _ => "",
    }
}

// ---- LoopControl ----
pub const LOOPCONTROL_NONE: u32 = 0;
pub const LOOPCONTROL_UNROLL: u32 = 1;
pub const LOOPCONTROL_DONTUNROLL: u32 = 2;
pub const LOOPCONTROL_DEPENDENCYINFINITE: u32 = 4;
pub const LOOPCONTROL_DEPENDENCYLENGTH: u32 = 8;
pub const LOOPCONTROL_MINITERATIONS: u32 = 16;
pub const LOOPCONTROL_MAXITERATIONS: u32 = 32;
pub const LOOPCONTROL_ITERATIONMULTIPLE: u32 = 64;
pub const LOOPCONTROL_PEELCOUNT: u32 = 128;
pub const LOOPCONTROL_PARTIALCOUNT: u32 = 256;
pub const LOOPCONTROL_INITIATIONINTERVALALTERA: u32 = 65536;
pub const LOOPCONTROL_MAXCONCURRENCYALTERA: u32 = 131072;
pub const LOOPCONTROL_DEPENDENCYARRAYALTERA: u32 = 262144;
pub const LOOPCONTROL_PIPELINEENABLEALTERA: u32 = 524288;
pub const LOOPCONTROL_LOOPCOALESCEALTERA: u32 = 1048576;
pub const LOOPCONTROL_MAXINTERLEAVINGALTERA: u32 = 2097152;
pub const LOOPCONTROL_SPECULATEDITERATIONSALTERA: u32 = 4194304;
pub const LOOPCONTROL_NOFUSIONALTERA: u32 = 8388608;
pub const LOOPCONTROL_LOOPCOUNTALTERA: u32 = 16777216;
pub const LOOPCONTROL_MAXREINVOCATIONDELAYALTERA: u32 = 33554432;
pub const LOOPCONTROL_MULTIPLEWAITQUEUESQCOM: u32 = 268435456;
/// `LoopControl` enumerant value for a textual name.
#[must_use]
pub fn loopcontrol_value(name: &str) -> Option<u32> {
    let value = match name {
        "None" => 0,
        "Unroll" => 1,
        "DontUnroll" => 2,
        "DependencyInfinite" => 4,
        "DependencyLength" => 8,
        "MinIterations" => 16,
        "MaxIterations" => 32,
        "IterationMultiple" => 64,
        "PeelCount" => 128,
        "PartialCount" => 256,
        "InitiationIntervalALTERA" => 65536,
        "MaxConcurrencyALTERA" => 131072,
        "DependencyArrayALTERA" => 262144,
        "PipelineEnableALTERA" => 524288,
        "LoopCoalesceALTERA" => 1048576,
        "MaxInterleavingALTERA" => 2097152,
        "SpeculatedIterationsALTERA" => 4194304,
        "NoFusionALTERA" => 8388608,
        "LoopCountALTERA" => 16777216,
        "MaxReinvocationDelayALTERA" => 33554432,
        "MultipleWaitQueuesQCOM" => 268435456,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `LoopControl` value known to the backend.
#[must_use]
pub const fn loopcontrol_name(value: u32) -> &'static str {
    match value {
        0 => "None",
        1 => "Unroll",
        2 => "DontUnroll",
        4 => "DependencyInfinite",
        8 => "DependencyLength",
        16 => "MinIterations",
        32 => "MaxIterations",
        64 => "IterationMultiple",
        128 => "PeelCount",
        256 => "PartialCount",
        65536 => "InitiationIntervalALTERA",
        131072 => "MaxConcurrencyALTERA",
        262144 => "DependencyArrayALTERA",
        524288 => "PipelineEnableALTERA",
        1048576 => "LoopCoalesceALTERA",
        2097152 => "MaxInterleavingALTERA",
        4194304 => "SpeculatedIterationsALTERA",
        8388608 => "NoFusionALTERA",
        16777216 => "LoopCountALTERA",
        33554432 => "MaxReinvocationDelayALTERA",
        268435456 => "MultipleWaitQueuesQCOM",
        _ => "",
    }
}

pub const SCOPE_CROSSDEVICE: u32 = 0;
pub const SCOPE_DEVICE: u32 = 1;
pub const SCOPE_WORKGROUP: u32 = 2;
pub const SCOPE_SUBGROUP: u32 = 3;
pub const SCOPE_INVOCATION: u32 = 4;
pub const SCOPE_QUEUEFAMILY: u32 = 5;
pub const SCOPE_SHADERCALLKHR: u32 = 6;
/// `Scope` enumerant value for a textual name.
#[must_use]
pub fn scope_value(name: &str) -> Option<u32> {
    let value = match name {
        "CrossDevice" => 0,
        "Device" => 1,
        "Workgroup" => 2,
        "Subgroup" => 3,
        "Invocation" => 4,
        "QueueFamily" => 5,
        "ShaderCallKHR" => 6,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `Scope` value known to the backend.
#[must_use]
pub const fn scope_name(value: u32) -> &'static str {
    match value {
        0 => "CrossDevice",
        1 => "Device",
        2 => "Workgroup",
        3 => "Subgroup",
        4 => "Invocation",
        5 => "QueueFamily",
        6 => "ShaderCallKHR",
        _ => "",
    }
}

pub const MEMORYSEMANTICS_RELAXED: u32 = 0;
pub const MEMORYSEMANTICS_ACQUIRE: u32 = 2;
pub const MEMORYSEMANTICS_RELEASE: u32 = 4;
pub const MEMORYSEMANTICS_ACQUIRERELEASE: u32 = 8;
pub const MEMORYSEMANTICS_SEQUENTIALLYCONSISTENT: u32 = 16;
pub const MEMORYSEMANTICS_UNIFORMMEMORY: u32 = 64;
pub const MEMORYSEMANTICS_SUBGROUPMEMORY: u32 = 128;
pub const MEMORYSEMANTICS_WORKGROUPMEMORY: u32 = 256;
pub const MEMORYSEMANTICS_CROSSWORKGROUPMEMORY: u32 = 512;
pub const MEMORYSEMANTICS_ATOMICCOUNTERMEMORY: u32 = 1024;
pub const MEMORYSEMANTICS_IMAGEMEMORY: u32 = 2048;
pub const MEMORYSEMANTICS_OUTPUTMEMORY: u32 = 4096;
pub const MEMORYSEMANTICS_MAKEAVAILABLE: u32 = 8192;
pub const MEMORYSEMANTICS_MAKEVISIBLE: u32 = 16384;
pub const MEMORYSEMANTICS_VOLATILE: u32 = 32768;
/// `MemorySemantics` enumerant value for a textual name.
#[must_use]
pub fn memorysemantics_value(name: &str) -> Option<u32> {
    let value = match name {
        "Relaxed" => 0,
        "Acquire" => 2,
        "Release" => 4,
        "AcquireRelease" => 8,
        "SequentiallyConsistent" => 16,
        "UniformMemory" => 64,
        "SubgroupMemory" => 128,
        "WorkgroupMemory" => 256,
        "CrossWorkgroupMemory" => 512,
        "AtomicCounterMemory" => 1024,
        "ImageMemory" => 2048,
        "OutputMemory" => 4096,
        "MakeAvailable" => 8192,
        "MakeVisible" => 16384,
        "Volatile" => 32768,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `MemorySemantics` value known to the backend.
#[must_use]
pub const fn memorysemantics_name(value: u32) -> &'static str {
    match value {
        0 => "Relaxed",
        2 => "Acquire",
        4 => "Release",
        8 => "AcquireRelease",
        16 => "SequentiallyConsistent",
        64 => "UniformMemory",
        128 => "SubgroupMemory",
        256 => "WorkgroupMemory",
        512 => "CrossWorkgroupMemory",
        1024 => "AtomicCounterMemory",
        2048 => "ImageMemory",
        4096 => "OutputMemory",
        8192 => "MakeAvailable",
        16384 => "MakeVisible",
        32768 => "Volatile",
        _ => "",
    }
}

pub const LINKAGETYPE_EXPORT: u32 = 0;
pub const LINKAGETYPE_IMPORT: u32 = 1;
pub const LINKAGETYPE_LINKONCEODR: u32 = 2;
pub const LINKAGETYPE_WEAKAMD: u32 = 3;
/// `LinkageType` enumerant value for a textual name.
#[must_use]
pub fn linkagetype_value(name: &str) -> Option<u32> {
    let value = match name {
        "Export" => 0,
        "Import" => 1,
        "LinkOnceODR" => 2,
        "WeakAMD" => 3,
        _ => return None,
    };
    Some(value)
}

/// Textual name of a `LinkageType` value known to the backend.
#[must_use]
pub const fn linkagetype_name(value: u32) -> &'static str {
    match value {
        0 => "Export",
        1 => "Import",
        2 => "LinkOnceODR",
        3 => "WeakAMD",
        _ => "",
    }
}

/// `acos`.
pub const OCL_ACOS: u32 = 0;
/// `acosh`.
pub const OCL_ACOSH: u32 = 1;
/// `acospi`.
pub const OCL_ACOSPI: u32 = 2;
/// `asin`.
pub const OCL_ASIN: u32 = 3;
/// `asinh`.
pub const OCL_ASINH: u32 = 4;
/// `asinpi`.
pub const OCL_ASINPI: u32 = 5;
/// `atan`.
pub const OCL_ATAN: u32 = 6;
/// `atan2`.
pub const OCL_ATAN2: u32 = 7;
/// `atanh`.
pub const OCL_ATANH: u32 = 8;
/// `atanpi`.
pub const OCL_ATANPI: u32 = 9;
/// `atan2pi`.
pub const OCL_ATAN2PI: u32 = 10;
/// `cbrt`.
pub const OCL_CBRT: u32 = 11;
/// `ceil`.
pub const OCL_CEIL: u32 = 12;
/// `copysign`.
pub const OCL_COPYSIGN: u32 = 13;
/// `cos`.
pub const OCL_COS: u32 = 14;
/// `cosh`.
pub const OCL_COSH: u32 = 15;
/// `cospi`.
pub const OCL_COSPI: u32 = 16;
/// `erfc`.
pub const OCL_ERFC: u32 = 17;
/// `erf`.
pub const OCL_ERF: u32 = 18;
/// `exp`.
pub const OCL_EXP: u32 = 19;
/// `exp2`.
pub const OCL_EXP2: u32 = 20;
/// `exp10`.
pub const OCL_EXP10: u32 = 21;
/// `expm1`.
pub const OCL_EXPM1: u32 = 22;
/// `fabs`.
pub const OCL_FABS: u32 = 23;
/// `fdim`.
pub const OCL_FDIM: u32 = 24;
/// `floor`.
pub const OCL_FLOOR: u32 = 25;
/// `fma`.
pub const OCL_FMA: u32 = 26;
/// `fmax`.
pub const OCL_FMAX: u32 = 27;
/// `fmin`.
pub const OCL_FMIN: u32 = 28;
/// `fmod`.
pub const OCL_FMOD: u32 = 29;
/// `fract`.
pub const OCL_FRACT: u32 = 30;
/// `frexp`.
pub const OCL_FREXP: u32 = 31;
/// `hypot`.
pub const OCL_HYPOT: u32 = 32;
/// `ilogb`.
pub const OCL_ILOGB: u32 = 33;
/// `ldexp`.
pub const OCL_LDEXP: u32 = 34;
/// `lgamma`.
pub const OCL_LGAMMA: u32 = 35;
/// `lgamma_r`.
pub const OCL_LGAMMA_R: u32 = 36;
/// `log`.
pub const OCL_LOG: u32 = 37;
/// `log2`.
pub const OCL_LOG2: u32 = 38;
/// `log10`.
pub const OCL_LOG10: u32 = 39;
/// `log1p`.
pub const OCL_LOG1P: u32 = 40;
/// `logb`.
pub const OCL_LOGB: u32 = 41;
/// `mad`.
pub const OCL_MAD: u32 = 42;
/// `maxmag`.
pub const OCL_MAXMAG: u32 = 43;
/// `minmag`.
pub const OCL_MINMAG: u32 = 44;
/// `modf`.
pub const OCL_MODF: u32 = 45;
/// `nan`.
pub const OCL_NAN: u32 = 46;
/// `nextafter`.
pub const OCL_NEXTAFTER: u32 = 47;
/// `pow`.
pub const OCL_POW: u32 = 48;
/// `pown`.
pub const OCL_POWN: u32 = 49;
/// `powr`.
pub const OCL_POWR: u32 = 50;
/// `remainder`.
pub const OCL_REMAINDER: u32 = 51;
/// `remquo`.
pub const OCL_REMQUO: u32 = 52;
/// `rint`.
pub const OCL_RINT: u32 = 53;
/// `rootn`.
pub const OCL_ROOTN: u32 = 54;
/// `round`.
pub const OCL_ROUND: u32 = 55;
/// `rsqrt`.
pub const OCL_RSQRT: u32 = 56;
/// `sin`.
pub const OCL_SIN: u32 = 57;
/// `sincos`.
pub const OCL_SINCOS: u32 = 58;
/// `sinh`.
pub const OCL_SINH: u32 = 59;
/// `sinpi`.
pub const OCL_SINPI: u32 = 60;
/// `sqrt`.
pub const OCL_SQRT: u32 = 61;
/// `tan`.
pub const OCL_TAN: u32 = 62;
/// `tanh`.
pub const OCL_TANH: u32 = 63;
/// `tanpi`.
pub const OCL_TANPI: u32 = 64;
/// `tgamma`.
pub const OCL_TGAMMA: u32 = 65;
/// `trunc`.
pub const OCL_TRUNC: u32 = 66;
/// `half_cos`.
pub const OCL_HALF_COS: u32 = 67;
/// `half_divide`.
pub const OCL_HALF_DIVIDE: u32 = 68;
/// `half_exp`.
pub const OCL_HALF_EXP: u32 = 69;
/// `half_exp2`.
pub const OCL_HALF_EXP2: u32 = 70;
/// `half_exp10`.
pub const OCL_HALF_EXP10: u32 = 71;
/// `half_log`.
pub const OCL_HALF_LOG: u32 = 72;
/// `half_log2`.
pub const OCL_HALF_LOG2: u32 = 73;
/// `half_log10`.
pub const OCL_HALF_LOG10: u32 = 74;
/// `half_powr`.
pub const OCL_HALF_POWR: u32 = 75;
/// `half_recip`.
pub const OCL_HALF_RECIP: u32 = 76;
/// `half_rsqrt`.
pub const OCL_HALF_RSQRT: u32 = 77;
/// `half_sin`.
pub const OCL_HALF_SIN: u32 = 78;
/// `half_sqrt`.
pub const OCL_HALF_SQRT: u32 = 79;
/// `half_tan`.
pub const OCL_HALF_TAN: u32 = 80;
/// `native_cos`.
pub const OCL_NATIVE_COS: u32 = 81;
/// `native_divide`.
pub const OCL_NATIVE_DIVIDE: u32 = 82;
/// `native_exp`.
pub const OCL_NATIVE_EXP: u32 = 83;
/// `native_exp2`.
pub const OCL_NATIVE_EXP2: u32 = 84;
/// `native_exp10`.
pub const OCL_NATIVE_EXP10: u32 = 85;
/// `native_log`.
pub const OCL_NATIVE_LOG: u32 = 86;
/// `native_log2`.
pub const OCL_NATIVE_LOG2: u32 = 87;
/// `native_log10`.
pub const OCL_NATIVE_LOG10: u32 = 88;
/// `native_powr`.
pub const OCL_NATIVE_POWR: u32 = 89;
/// `native_recip`.
pub const OCL_NATIVE_RECIP: u32 = 90;
/// `native_rsqrt`.
pub const OCL_NATIVE_RSQRT: u32 = 91;
/// `native_sin`.
pub const OCL_NATIVE_SIN: u32 = 92;
/// `native_sqrt`.
pub const OCL_NATIVE_SQRT: u32 = 93;
/// `native_tan`.
pub const OCL_NATIVE_TAN: u32 = 94;
/// `fclamp`.
pub const OCL_FCLAMP: u32 = 95;
/// `degrees`.
pub const OCL_DEGREES: u32 = 96;
/// `fmax_common`.
pub const OCL_FMAX_COMMON: u32 = 97;
/// `fmin_common`.
pub const OCL_FMIN_COMMON: u32 = 98;
/// `mix`.
pub const OCL_MIX: u32 = 99;
/// `radians`.
pub const OCL_RADIANS: u32 = 100;
/// `step`.
pub const OCL_STEP: u32 = 101;
/// `smoothstep`.
pub const OCL_SMOOTHSTEP: u32 = 102;
/// `sign`.
pub const OCL_SIGN: u32 = 103;
/// `cross`.
pub const OCL_CROSS: u32 = 104;
/// `distance`.
pub const OCL_DISTANCE: u32 = 105;
/// `length`.
pub const OCL_LENGTH: u32 = 106;
/// `normalize`.
pub const OCL_NORMALIZE: u32 = 107;
/// `fast_distance`.
pub const OCL_FAST_DISTANCE: u32 = 108;
/// `fast_length`.
pub const OCL_FAST_LENGTH: u32 = 109;
/// `fast_normalize`.
pub const OCL_FAST_NORMALIZE: u32 = 110;
/// `s_abs`.
pub const OCL_S_ABS: u32 = 141;
/// `s_abs_diff`.
pub const OCL_S_ABS_DIFF: u32 = 142;
/// `s_add_sat`.
pub const OCL_S_ADD_SAT: u32 = 143;
/// `u_add_sat`.
pub const OCL_U_ADD_SAT: u32 = 144;
/// `s_hadd`.
pub const OCL_S_HADD: u32 = 145;
/// `u_hadd`.
pub const OCL_U_HADD: u32 = 146;
/// `s_rhadd`.
pub const OCL_S_RHADD: u32 = 147;
/// `u_rhadd`.
pub const OCL_U_RHADD: u32 = 148;
/// `s_clamp`.
pub const OCL_S_CLAMP: u32 = 149;
/// `u_clamp`.
pub const OCL_U_CLAMP: u32 = 150;
/// `clz`.
pub const OCL_CLZ: u32 = 151;
/// `ctz`.
pub const OCL_CTZ: u32 = 152;
/// `s_mad_hi`.
pub const OCL_S_MAD_HI: u32 = 153;
/// `u_mad_sat`.
pub const OCL_U_MAD_SAT: u32 = 154;
/// `s_mad_sat`.
pub const OCL_S_MAD_SAT: u32 = 155;
/// `s_max`.
pub const OCL_S_MAX: u32 = 156;
/// `u_max`.
pub const OCL_U_MAX: u32 = 157;
/// `s_min`.
pub const OCL_S_MIN: u32 = 158;
/// `u_min`.
pub const OCL_U_MIN: u32 = 159;
/// `s_mul_hi`.
pub const OCL_S_MUL_HI: u32 = 160;
/// `rotate`.
pub const OCL_ROTATE: u32 = 161;
/// `s_sub_sat`.
pub const OCL_S_SUB_SAT: u32 = 162;
/// `u_sub_sat`.
pub const OCL_U_SUB_SAT: u32 = 163;
/// `u_upsample`.
pub const OCL_U_UPSAMPLE: u32 = 164;
/// `s_upsample`.
pub const OCL_S_UPSAMPLE: u32 = 165;
/// `popcount`.
pub const OCL_POPCOUNT: u32 = 166;
/// `s_mad24`.
pub const OCL_S_MAD24: u32 = 167;
/// `u_mad24`.
pub const OCL_U_MAD24: u32 = 168;
/// `s_mul24`.
pub const OCL_S_MUL24: u32 = 169;
/// `u_mul24`.
pub const OCL_U_MUL24: u32 = 170;
/// `vloadn`.
pub const OCL_VLOADN: u32 = 171;
/// `vstoren`.
pub const OCL_VSTOREN: u32 = 172;
/// `vload_half`.
pub const OCL_VLOAD_HALF: u32 = 173;
/// `vload_halfn`.
pub const OCL_VLOAD_HALFN: u32 = 174;
/// `vstore_half`.
pub const OCL_VSTORE_HALF: u32 = 175;
/// `vstore_half_r`.
pub const OCL_VSTORE_HALF_R: u32 = 176;
/// `vstore_halfn`.
pub const OCL_VSTORE_HALFN: u32 = 177;
/// `vstore_halfn_r`.
pub const OCL_VSTORE_HALFN_R: u32 = 178;
/// `vloada_halfn`.
pub const OCL_VLOADA_HALFN: u32 = 179;
/// `vstorea_halfn`.
pub const OCL_VSTOREA_HALFN: u32 = 180;
/// `vstorea_halfn_r`.
pub const OCL_VSTOREA_HALFN_R: u32 = 181;
/// `shuffle`.
pub const OCL_SHUFFLE: u32 = 182;
/// `shuffle2`.
pub const OCL_SHUFFLE2: u32 = 183;
/// `printf`.
pub const OCL_PRINTF: u32 = 184;
/// `prefetch`.
pub const OCL_PREFETCH: u32 = 185;
/// `bitselect`.
pub const OCL_BITSELECT: u32 = 186;
/// `select`.
pub const OCL_SELECT: u32 = 187;
/// `u_abs`.
pub const OCL_U_ABS: u32 = 201;
/// `u_abs_diff`.
pub const OCL_U_ABS_DIFF: u32 = 202;
/// `u_mul_hi`.
pub const OCL_U_MUL_HI: u32 = 203;
/// `u_mad_hi`.
pub const OCL_U_MAD_HI: u32 = 204;

/// OpenCL.std extended instruction number for a textual name.
#[must_use]
pub fn ocl_opcode(name: &str) -> Option<u32> {
    let value = match name {
        "acos" => 0,
        "acosh" => 1,
        "acospi" => 2,
        "asin" => 3,
        "asinh" => 4,
        "asinpi" => 5,
        "atan" => 6,
        "atan2" => 7,
        "atan2pi" => 10,
        "atanh" => 8,
        "atanpi" => 9,
        "bitselect" => 186,
        "cbrt" => 11,
        "ceil" => 12,
        "clz" => 151,
        "copysign" => 13,
        "cos" => 14,
        "cosh" => 15,
        "cospi" => 16,
        "cross" => 104,
        "ctz" => 152,
        "degrees" => 96,
        "distance" => 105,
        "erf" => 18,
        "erfc" => 17,
        "exp" => 19,
        "exp10" => 21,
        "exp2" => 20,
        "expm1" => 22,
        "fabs" => 23,
        "fast_distance" => 108,
        "fast_length" => 109,
        "fast_normalize" => 110,
        "fclamp" => 95,
        "fdim" => 24,
        "floor" => 25,
        "fma" => 26,
        "fmax" => 27,
        "fmax_common" => 97,
        "fmin" => 28,
        "fmin_common" => 98,
        "fmod" => 29,
        "fract" => 30,
        "frexp" => 31,
        "half_cos" => 67,
        "half_divide" => 68,
        "half_exp" => 69,
        "half_exp10" => 71,
        "half_exp2" => 70,
        "half_log" => 72,
        "half_log10" => 74,
        "half_log2" => 73,
        "half_powr" => 75,
        "half_recip" => 76,
        "half_rsqrt" => 77,
        "half_sin" => 78,
        "half_sqrt" => 79,
        "half_tan" => 80,
        "hypot" => 32,
        "ilogb" => 33,
        "ldexp" => 34,
        "length" => 106,
        "lgamma" => 35,
        "lgamma_r" => 36,
        "log" => 37,
        "log10" => 39,
        "log1p" => 40,
        "log2" => 38,
        "logb" => 41,
        "mad" => 42,
        "maxmag" => 43,
        "minmag" => 44,
        "mix" => 99,
        "modf" => 45,
        "nan" => 46,
        "native_cos" => 81,
        "native_divide" => 82,
        "native_exp" => 83,
        "native_exp10" => 85,
        "native_exp2" => 84,
        "native_log" => 86,
        "native_log10" => 88,
        "native_log2" => 87,
        "native_powr" => 89,
        "native_recip" => 90,
        "native_rsqrt" => 91,
        "native_sin" => 92,
        "native_sqrt" => 93,
        "native_tan" => 94,
        "nextafter" => 47,
        "normalize" => 107,
        "popcount" => 166,
        "pow" => 48,
        "pown" => 49,
        "powr" => 50,
        "prefetch" => 185,
        "printf" => 184,
        "radians" => 100,
        "remainder" => 51,
        "remquo" => 52,
        "rint" => 53,
        "rootn" => 54,
        "rotate" => 161,
        "round" => 55,
        "rsqrt" => 56,
        "s_abs" => 141,
        "s_abs_diff" => 142,
        "s_add_sat" => 143,
        "s_clamp" => 149,
        "s_hadd" => 145,
        "s_mad24" => 167,
        "s_mad_hi" => 153,
        "s_mad_sat" => 155,
        "s_max" => 156,
        "s_min" => 158,
        "s_mul24" => 169,
        "s_mul_hi" => 160,
        "s_rhadd" => 147,
        "s_sub_sat" => 162,
        "s_upsample" => 165,
        "select" => 187,
        "shuffle" => 182,
        "shuffle2" => 183,
        "sign" => 103,
        "sin" => 57,
        "sincos" => 58,
        "sinh" => 59,
        "sinpi" => 60,
        "smoothstep" => 102,
        "sqrt" => 61,
        "step" => 101,
        "tan" => 62,
        "tanh" => 63,
        "tanpi" => 64,
        "tgamma" => 65,
        "trunc" => 66,
        "u_abs" => 201,
        "u_abs_diff" => 202,
        "u_add_sat" => 144,
        "u_clamp" => 150,
        "u_hadd" => 146,
        "u_mad24" => 168,
        "u_mad_hi" => 204,
        "u_mad_sat" => 154,
        "u_max" => 157,
        "u_min" => 159,
        "u_mul24" => 170,
        "u_mul_hi" => 203,
        "u_rhadd" => 148,
        "u_sub_sat" => 163,
        "u_upsample" => 164,
        "vload_half" => 173,
        "vload_halfn" => 174,
        "vloada_halfn" => 179,
        "vloadn" => 171,
        "vstore_half" => 175,
        "vstore_half_r" => 176,
        "vstore_halfn" => 177,
        "vstore_halfn_r" => 178,
        "vstorea_halfn" => 180,
        "vstorea_halfn_r" => 181,
        "vstoren" => 172,
        _ => return None,
    };
    Some(value)
}

/// Textual name of an OpenCL.std instruction number.
#[must_use]
pub const fn ocl_name(inst: u32) -> &'static str {
    match inst {
        0 => "acos",
        1 => "acosh",
        2 => "acospi",
        3 => "asin",
        4 => "asinh",
        5 => "asinpi",
        6 => "atan",
        7 => "atan2",
        10 => "atan2pi",
        8 => "atanh",
        9 => "atanpi",
        186 => "bitselect",
        11 => "cbrt",
        12 => "ceil",
        151 => "clz",
        13 => "copysign",
        14 => "cos",
        15 => "cosh",
        16 => "cospi",
        104 => "cross",
        152 => "ctz",
        96 => "degrees",
        105 => "distance",
        18 => "erf",
        17 => "erfc",
        19 => "exp",
        21 => "exp10",
        20 => "exp2",
        22 => "expm1",
        23 => "fabs",
        108 => "fast_distance",
        109 => "fast_length",
        110 => "fast_normalize",
        95 => "fclamp",
        24 => "fdim",
        25 => "floor",
        26 => "fma",
        27 => "fmax",
        97 => "fmax_common",
        28 => "fmin",
        98 => "fmin_common",
        29 => "fmod",
        30 => "fract",
        31 => "frexp",
        67 => "half_cos",
        68 => "half_divide",
        69 => "half_exp",
        71 => "half_exp10",
        70 => "half_exp2",
        72 => "half_log",
        74 => "half_log10",
        73 => "half_log2",
        75 => "half_powr",
        76 => "half_recip",
        77 => "half_rsqrt",
        78 => "half_sin",
        79 => "half_sqrt",
        80 => "half_tan",
        32 => "hypot",
        33 => "ilogb",
        34 => "ldexp",
        106 => "length",
        35 => "lgamma",
        36 => "lgamma_r",
        37 => "log",
        39 => "log10",
        40 => "log1p",
        38 => "log2",
        41 => "logb",
        42 => "mad",
        43 => "maxmag",
        44 => "minmag",
        99 => "mix",
        45 => "modf",
        46 => "nan",
        81 => "native_cos",
        82 => "native_divide",
        83 => "native_exp",
        85 => "native_exp10",
        84 => "native_exp2",
        86 => "native_log",
        88 => "native_log10",
        87 => "native_log2",
        89 => "native_powr",
        90 => "native_recip",
        91 => "native_rsqrt",
        92 => "native_sin",
        93 => "native_sqrt",
        94 => "native_tan",
        47 => "nextafter",
        107 => "normalize",
        166 => "popcount",
        48 => "pow",
        49 => "pown",
        50 => "powr",
        185 => "prefetch",
        184 => "printf",
        100 => "radians",
        51 => "remainder",
        52 => "remquo",
        53 => "rint",
        54 => "rootn",
        161 => "rotate",
        55 => "round",
        56 => "rsqrt",
        141 => "s_abs",
        142 => "s_abs_diff",
        143 => "s_add_sat",
        149 => "s_clamp",
        145 => "s_hadd",
        167 => "s_mad24",
        153 => "s_mad_hi",
        155 => "s_mad_sat",
        156 => "s_max",
        158 => "s_min",
        169 => "s_mul24",
        160 => "s_mul_hi",
        147 => "s_rhadd",
        162 => "s_sub_sat",
        165 => "s_upsample",
        187 => "select",
        182 => "shuffle",
        183 => "shuffle2",
        103 => "sign",
        57 => "sin",
        58 => "sincos",
        59 => "sinh",
        60 => "sinpi",
        102 => "smoothstep",
        61 => "sqrt",
        101 => "step",
        62 => "tan",
        63 => "tanh",
        64 => "tanpi",
        65 => "tgamma",
        66 => "trunc",
        201 => "u_abs",
        202 => "u_abs_diff",
        144 => "u_add_sat",
        150 => "u_clamp",
        146 => "u_hadd",
        168 => "u_mad24",
        204 => "u_mad_hi",
        154 => "u_mad_sat",
        157 => "u_max",
        159 => "u_min",
        170 => "u_mul24",
        203 => "u_mul_hi",
        148 => "u_rhadd",
        163 => "u_sub_sat",
        164 => "u_upsample",
        173 => "vload_half",
        174 => "vload_halfn",
        179 => "vloada_halfn",
        171 => "vloadn",
        175 => "vstore_half",
        176 => "vstore_half_r",
        177 => "vstore_halfn",
        178 => "vstore_halfn_r",
        180 => "vstorea_halfn",
        181 => "vstorea_halfn_r",
        172 => "vstoren",
        _ => "",
    }
}
