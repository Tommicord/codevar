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

//! Opcode, enumerant and extended-instruction numbers the interpreter
//! understands.
//!
//! Values follow the SPIR-V core grammar
//! (`spirv.core.grammar.json`, unified1) and the OpenCL.std extended
//! instruction set (`extinst.opencl.std.100.grammar.json`), both from
//! KhronosGroup/SPIRV-Headers. The driver keeps its own copy so the
//! bare-metal image does not depend on the compiler crates at run time;
//! the test suite cross-checks them against `codevar-ocl-ir`'s generated
//! tables.

#![allow(dead_code)]

// --- Module structure ---------------------------------------------------

/// `OpNop`.
pub const OP_NOP: u16 = 0;
/// `OpUndef`.
pub const OP_UNDEF: u16 = 1;
/// `OpSourceContinued`.
pub const OP_SOURCE_CONTINUED: u16 = 2;
/// `OpSource`.
pub const OP_SOURCE: u16 = 3;
/// `OpSourceExtension`.
pub const OP_SOURCE_EXTENSION: u16 = 4;
/// `OpName`.
pub const OP_NAME: u16 = 5;
/// `OpMemberName`.
pub const OP_MEMBER_NAME: u16 = 6;
/// `OpString`.
pub const OP_STRING: u16 = 7;
/// `OpLine`.
pub const OP_LINE: u16 = 8;
/// `OpExtension`.
pub const OP_EXTENSION: u16 = 10;
/// `OpExtInstImport`.
pub const OP_EXT_INST_IMPORT: u16 = 11;
/// `OpExtInst`.
pub const OP_EXT_INST: u16 = 12;
/// `OpMemoryModel`.
pub const OP_MEMORY_MODEL: u16 = 14;
/// `OpEntryPoint`.
pub const OP_ENTRY_POINT: u16 = 15;
/// `OpExecutionMode`.
pub const OP_EXECUTION_MODE: u16 = 16;
/// `OpCapability`.
pub const OP_CAPABILITY: u16 = 17;
/// `OpNoLine`.
pub const OP_NO_LINE: u16 = 317;

// --- Types --------------------------------------------------------------

/// `OpTypeVoid`.
pub const OP_TYPE_VOID: u16 = 19;
/// `OpTypeBool`.
pub const OP_TYPE_BOOL: u16 = 20;
/// `OpTypeInt`.
pub const OP_TYPE_INT: u16 = 21;
/// `OpTypeFloat`.
pub const OP_TYPE_FLOAT: u16 = 22;
/// `OpTypeVector`.
pub const OP_TYPE_VECTOR: u16 = 23;
/// `OpTypeMatrix`.
pub const OP_TYPE_MATRIX: u16 = 24;
/// `OpTypeImage`.
pub const OP_TYPE_IMAGE: u16 = 25;
/// `OpTypeSampler`.
pub const OP_TYPE_SAMPLER: u16 = 26;
/// `OpTypeSampledImage`.
pub const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
/// `OpTypeArray`.
pub const OP_TYPE_ARRAY: u16 = 28;
/// `OpTypeRuntimeArray`.
pub const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
/// `OpTypeStruct`.
pub const OP_TYPE_STRUCT: u16 = 30;
/// `OpTypeOpaque`.
pub const OP_TYPE_OPAQUE: u16 = 31;
/// `OpTypePointer`.
pub const OP_TYPE_POINTER: u16 = 32;
/// `OpTypeFunction`.
pub const OP_TYPE_FUNCTION: u16 = 33;
/// `OpTypeEvent`.
pub const OP_TYPE_EVENT: u16 = 34;
/// `OpTypeDeviceEvent`.
pub const OP_TYPE_DEVICE_EVENT: u16 = 35;
/// `OpTypeReserveId`.
pub const OP_TYPE_RESERVE_ID: u16 = 36;
/// `OpTypeQueue`.
pub const OP_TYPE_QUEUE: u16 = 37;
/// `OpTypePipe`.
pub const OP_TYPE_PIPE: u16 = 38;
/// `OpTypeForwardPointer`.
pub const OP_TYPE_FORWARD_POINTER: u16 = 39;

// --- Constants ----------------------------------------------------------

/// `OpConstantTrue`.
pub const OP_CONSTANT_TRUE: u16 = 41;
/// `OpConstantFalse`.
pub const OP_CONSTANT_FALSE: u16 = 42;
/// `OpConstant`.
pub const OP_CONSTANT: u16 = 43;
/// `OpConstantComposite`.
pub const OP_CONSTANT_COMPOSITE: u16 = 44;
/// `OpConstantNull`.
pub const OP_CONSTANT_NULL: u16 = 46;
/// `OpSpecConstantTrue`.
pub const OP_SPEC_CONSTANT_TRUE: u16 = 48;
/// `OpSpecConstantFalse`.
pub const OP_SPEC_CONSTANT_FALSE: u16 = 49;
/// `OpSpecConstant`.
pub const OP_SPEC_CONSTANT: u16 = 50;
/// `OpSpecConstantComposite`.
pub const OP_SPEC_CONSTANT_COMPOSITE: u16 = 51;

// --- Functions ----------------------------------------------------------

/// `OpFunction`.
pub const OP_FUNCTION: u16 = 54;
/// `OpFunctionParameter`.
pub const OP_FUNCTION_PARAMETER: u16 = 55;
/// `OpFunctionEnd`.
pub const OP_FUNCTION_END: u16 = 56;
/// `OpFunctionCall`.
pub const OP_FUNCTION_CALL: u16 = 57;
/// `OpVariable`.
pub const OP_VARIABLE: u16 = 59;

// --- Memory -------------------------------------------------------------

/// `OpLoad`.
pub const OP_LOAD: u16 = 61;
/// `OpStore`.
pub const OP_STORE: u16 = 62;
/// `OpCopyMemory`.
pub const OP_COPY_MEMORY: u16 = 63;
/// `OpCopyMemorySized`.
pub const OP_COPY_MEMORY_SIZED: u16 = 64;
/// `OpAccessChain`.
pub const OP_ACCESS_CHAIN: u16 = 65;
/// `OpInBoundsAccessChain`.
pub const OP_IN_BOUNDS_ACCESS_CHAIN: u16 = 66;
/// `OpPtrAccessChain`.
pub const OP_PTR_ACCESS_CHAIN: u16 = 67;
/// `OpArrayLength`.
pub const OP_ARRAY_LENGTH: u16 = 68;

// --- Decorations --------------------------------------------------------

/// `OpDecorate`.
pub const OP_DECORATE: u16 = 71;
/// `OpMemberDecorate`.
pub const OP_MEMBER_DECORATE: u16 = 72;

// --- Composites ---------------------------------------------------------

/// `OpVectorExtractDynamic`.
pub const OP_VECTOR_EXTRACT_DYNAMIC: u16 = 77;
/// `OpVectorInsertDynamic`.
pub const OP_VECTOR_INSERT_DYNAMIC: u16 = 78;
/// `OpVectorShuffle`.
pub const OP_VECTOR_SHUFFLE: u16 = 79;
/// `OpCompositeConstruct`.
pub const OP_COMPOSITE_CONSTRUCT: u16 = 80;
/// `OpCompositeExtract`.
pub const OP_COMPOSITE_EXTRACT: u16 = 81;
/// `OpCompositeInsert`.
pub const OP_COMPOSITE_INSERT: u16 = 82;
/// `OpCopyObject`.
pub const OP_COPY_OBJECT: u16 = 83;
/// `OpTranspose`.
pub const OP_TRANSPOSE: u16 = 84;

// --- Conversions --------------------------------------------------------

/// `OpConvertFToU`.
pub const OP_CONVERT_F_TO_U: u16 = 109;
/// `OpConvertFToS`.
pub const OP_CONVERT_F_TO_S: u16 = 110;
/// `OpConvertSToF`.
pub const OP_CONVERT_S_TO_F: u16 = 111;
/// `OpConvertUToF`.
pub const OP_CONVERT_U_TO_F: u16 = 112;
/// `OpUConvert`.
pub const OP_U_CONVERT: u16 = 113;
/// `OpSConvert`.
pub const OP_S_CONVERT: u16 = 114;
/// `OpFConvert`.
pub const OP_F_CONVERT: u16 = 115;
/// `OpBitcast`.
pub const OP_BITCAST: u16 = 124;

// --- Arithmetic ---------------------------------------------------------

/// `OpSNegate`.
pub const OP_S_NEGATE: u16 = 126;
/// `OpFNegate`.
pub const OP_F_NEGATE: u16 = 127;
/// `OpIAdd`.
pub const OP_I_ADD: u16 = 128;
/// `OpFAdd`.
pub const OP_F_ADD: u16 = 129;
/// `OpISub`.
pub const OP_I_SUB: u16 = 130;
/// `OpFSub`.
pub const OP_F_SUB: u16 = 131;
/// `OpIMul`.
pub const OP_I_MUL: u16 = 132;
/// `OpFMul`.
pub const OP_F_MUL: u16 = 133;
/// `OpUDiv`.
pub const OP_U_DIV: u16 = 134;
/// `OpSDiv`.
pub const OP_S_DIV: u16 = 135;
/// `OpFDiv`.
pub const OP_F_DIV: u16 = 136;
/// `OpUMod`.
pub const OP_U_MOD: u16 = 137;
/// `OpSRem`.
pub const OP_S_REM: u16 = 138;
/// `OpFMod`.
pub const OP_F_MOD: u16 = 139;
/// `OpFRem`.
pub const OP_F_REM: u16 = 140;

// --- Logic and comparison ----------------------------------------------

/// `OpLogicalOr`.
pub const OP_LOGICAL_OR: u16 = 166;
/// `OpLogicalAnd`.
pub const OP_LOGICAL_AND: u16 = 167;
/// `OpLogicalNot`.
pub const OP_LOGICAL_NOT: u16 = 168;
/// `OpSelect`.
pub const OP_SELECT: u16 = 169;
/// `OpIEqual`.
pub const OP_I_EQUAL: u16 = 170;
/// `OpINotEqual`.
pub const OP_I_NOT_EQUAL: u16 = 171;
/// `OpUGreaterThan`.
pub const OP_U_GREATER_THAN: u16 = 172;
/// `OpSGreaterThan`.
pub const OP_S_GREATER_THAN: u16 = 173;
/// `OpUGreaterThanEqual`.
pub const OP_U_GREATER_THAN_EQUAL: u16 = 174;
/// `OpSGreaterThanEqual`.
pub const OP_S_GREATER_THAN_EQUAL: u16 = 175;
/// `OpULessThan`.
pub const OP_U_LESS_THAN: u16 = 176;
/// `OpSLessThan`.
pub const OP_S_LESS_THAN: u16 = 177;
/// `OpULessThanEqual`.
pub const OP_U_LESS_THAN_EQUAL: u16 = 178;
/// `OpSLessThanEqual`.
pub const OP_S_LESS_THAN_EQUAL: u16 = 179;
/// `OpFOrdEqual`.
pub const OP_F_ORD_EQUAL: u16 = 180;
/// `OpFOrdNotEqual`.
pub const OP_F_ORD_NOT_EQUAL: u16 = 182;
/// `OpFOrdLessThan`.
pub const OP_F_ORD_LESS_THAN: u16 = 184;
/// `OpFOrdGreaterThan`.
pub const OP_F_ORD_GREATER_THAN: u16 = 186;
/// `OpFOrdLessThanEqual`.
pub const OP_F_ORD_LESS_THAN_EQUAL: u16 = 188;
/// `OpFOrdGreaterThanEqual`.
pub const OP_F_ORD_GREATER_THAN_EQUAL: u16 = 190;
/// `OpShiftRightLogical`.
pub const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
/// `OpShiftRightArithmetic`.
pub const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
/// `OpShiftLeftLogical`.
pub const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
/// `OpBitwiseOr`.
pub const OP_BITWISE_OR: u16 = 197;
/// `OpBitwiseXor`.
pub const OP_BITWISE_XOR: u16 = 198;
/// `OpBitwiseAnd`.
pub const OP_BITWISE_AND: u16 = 199;
/// `OpNot`.
pub const OP_NOT: u16 = 200;

// --- Synchronization ----------------------------------------------------

/// `OpControlBarrier`.
pub const OP_CONTROL_BARRIER: u16 = 224;
/// `OpMemoryBarrier`.
pub const OP_MEMORY_BARRIER: u16 = 225;

// --- Structure of the control flow --------------------------------------

/// `OpPhi`.
pub const OP_PHI: u16 = 245;
/// `OpLoopMerge`.
pub const OP_LOOP_MERGE: u16 = 246;
/// `OpSelectionMerge`.
pub const OP_SELECTION_MERGE: u16 = 247;
/// `OpLabel`.
pub const OP_LABEL: u16 = 248;
/// `OpBranch`.
pub const OP_BRANCH: u16 = 249;
/// `OpBranchConditional`.
pub const OP_BRANCH_CONDITIONAL: u16 = 250;
/// `OpSwitch`.
pub const OP_SWITCH: u16 = 251;
/// `OpKill`.
pub const OP_KILL: u16 = 252;
/// `OpReturn`.
pub const OP_RETURN: u16 = 253;
/// `OpReturnValue`.
pub const OP_RETURN_VALUE: u16 = 254;
/// `OpUnreachable`.
pub const OP_UNREACHABLE: u16 = 255;

// --- Group operations (sub-group collectives) ----------------------------

/// `OpGroupAsyncCopy`.
pub const OP_GROUP_ASYNC_COPY: u16 = 259;
/// `OpGroupWaitEvents`.
pub const OP_GROUP_WAIT_EVENTS: u16 = 260;
/// `OpGroupAll`.
pub const OP_GROUP_ALL: u16 = 261;
/// `OpGroupAny`.
pub const OP_GROUP_ANY: u16 = 262;
/// `OpGroupBroadcast`.
pub const OP_GROUP_BROADCAST: u16 = 263;
/// `OpGroupIAdd`.
pub const OP_GROUP_I_ADD: u16 = 264;
/// `OpGroupFAdd`.
pub const OP_GROUP_F_ADD: u16 = 265;
/// `OpGroupFMin`.
pub const OP_GROUP_F_MIN: u16 = 266;
/// `OpGroupUMin`.
pub const OP_GROUP_U_MIN: u16 = 267;
/// `OpGroupSMin`.
pub const OP_GROUP_S_MIN: u16 = 268;
/// `OpGroupFMax`.
pub const OP_GROUP_F_MAX: u16 = 269;
/// `OpGroupUMax`.
pub const OP_GROUP_U_MAX: u16 = 270;
/// `OpGroupSMax`.
pub const OP_GROUP_S_MAX: u16 = 271;

// --- Enumerant values ---------------------------------------------------

/// `AddressingModel::Logical`.
pub const ADDRESSING_MODEL_LOGICAL: u32 = 0;
/// `AddressingModel::Physical32`.
pub const ADDRESSING_MODEL_PHYSICAL32: u32 = 1;
/// `AddressingModel::Physical64`.
pub const ADDRESSING_MODEL_PHYSICAL64: u32 = 2;
/// `MemoryModel::Simple`.
pub const MEMORY_MODEL_SIMPLE: u32 = 0;
/// `MemoryModel::GLSL450`.
pub const MEMORY_MODEL_GLSL450: u32 = 1;
/// `MemoryModel::OpenCL`.
pub const MEMORY_MODEL_OPENCL: u32 = 2;
/// `ExecutionModel::Kernel`.
pub const EXECUTION_MODEL_KERNEL: u32 = 6;
/// `Capability::Kernel`.
pub const CAPABILITY_KERNEL: u32 = 6;

/// `StorageClass::UniformConstant` (images and samplers — rejected).
pub const STORAGE_UNIFORM_CONSTANT: u32 = 0;
/// `StorageClass::Input`.
pub const STORAGE_INPUT: u32 = 1;
/// `StorageClass::Output` (rejected).
pub const STORAGE_OUTPUT: u32 = 3;
/// `StorageClass::Workgroup` (OpenCL `local`).
pub const STORAGE_WORKGROUP: u32 = 4;
/// `StorageClass::CrossWorkgroup` (OpenCL `global`).
pub const STORAGE_CROSS_WORKGROUP: u32 = 5;
/// `StorageClass::Private` (OpenCL `private`, module scope rejected).
pub const STORAGE_PRIVATE: u32 = 6;
/// `StorageClass::Function` (OpenCL private, function scope).
pub const STORAGE_FUNCTION: u32 = 7;

/// `Decoration::ArrayStride`.
pub const DECORATION_ARRAY_STRIDE: u32 = 6;
/// `Decoration::BuiltIn`.
pub const DECORATION_BUILTIN: u32 = 11;
/// `Decoration::Offset`.
pub const DECORATION_OFFSET: u32 = 35;
/// `Decoration::LinkageAttributes`.
pub const DECORATION_LINKAGE_ATTRIBUTES: u32 = 41;

/// `LinkageType::Export`.
pub const LINKAGE_EXPORT: u32 = 0;
/// `LinkageType::Import`.
pub const LINKAGE_IMPORT: u32 = 1;

/// `Scope::CrossDevice`.
pub const SCOPE_CROSS_DEVICE: u32 = 0;
/// `Scope::Device`.
pub const SCOPE_DEVICE: u32 = 1;
/// `Scope::Workgroup`.
pub const SCOPE_WORKGROUP: u32 = 2;
/// `Scope::Subgroup`.
pub const SCOPE_SUBGROUP: u32 = 3;
/// `Scope::Invocation`.
pub const SCOPE_INVOCATION: u32 = 4;

/// `GroupOperation::Reduce`.
pub const GROUP_OP_REDUCE: u32 = 0;
/// `GroupOperation::InclusiveScan`.
pub const GROUP_OP_INCLUSIVE_SCAN: u32 = 1;
/// `GroupOperation::ExclusiveScan`.
pub const GROUP_OP_EXCLUSIVE_SCAN: u32 = 2;

/// `BuiltIn::NumWorkgroups`.
pub const BUILTIN_NUM_WORKGROUPS: u32 = 24;
/// `BuiltIn::WorkgroupSize`.
pub const BUILTIN_WORKGROUP_SIZE: u32 = 25;
/// `BuiltIn::WorkgroupId`.
pub const BUILTIN_WORKGROUP_ID: u32 = 26;
/// `BuiltIn::LocalInvocationId`.
pub const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
/// `BuiltIn::GlobalInvocationId`.
pub const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
/// `BuiltIn::LocalInvocationIndex`.
pub const BUILTIN_LOCAL_INVOCATION_INDEX: u32 = 29;
/// `BuiltIn::WorkDim`.
pub const BUILTIN_WORK_DIM: u32 = 30;
/// `BuiltIn::GlobalSize`.
pub const BUILTIN_GLOBAL_SIZE: u32 = 31;
/// `BuiltIn::EnqueuedWorkgroupSize`.
pub const BUILTIN_ENQUEUED_WORKGROUP_SIZE: u32 = 32;
/// `BuiltIn::GlobalOffset`.
pub const BUILTIN_GLOBAL_OFFSET: u32 = 33;
/// `BuiltIn::GlobalLinearId`.
pub const BUILTIN_GLOBAL_LINEAR_ID: u32 = 34;
/// `BuiltIn::SubgroupSize`.
pub const BUILTIN_SUBGROUP_SIZE: u32 = 36;
/// `BuiltIn::SubgroupMaxSize`.
pub const BUILTIN_SUBGROUP_MAX_SIZE: u32 = 37;
/// `BuiltIn::NumSubgroups`.
pub const BUILTIN_NUM_SUBGROUPS: u32 = 38;
/// `BuiltIn::NumEnqueuedSubgroups`.
pub const BUILTIN_NUM_ENQUEUED_SUBGROUPS: u32 = 39;
/// `BuiltIn::SubgroupId`.
pub const BUILTIN_SUBGROUP_ID: u32 = 40;

// --- OpenCL.std extended instructions -----------------------------------

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
/// `clamp` (float).
pub const OCL_F_CLAMP: u32 = 95;
/// `degrees`.
pub const OCL_DEGREES: u32 = 96;
/// `max` (float).
pub const OCL_F_MAX_COMMON: u32 = 97;
/// `min` (float).
pub const OCL_F_MIN_COMMON: u32 = 98;
/// `mix`.
pub const OCL_MIX: u32 = 99;
/// `radians`.
pub const OCL_RADIANS: u32 = 100;
/// `step`.
pub const OCL_STEP: u32 = 101;
/// `smoothstep`.
pub const OCL_SMOOTHSTEP: u32 = 102;
/// `sign` (float).
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
/// `abs` (signed integer).
pub const OCL_S_ABS: u32 = 141;
/// `abs_diff` (signed).
pub const OCL_S_ABS_DIFF: u32 = 142;
/// `add_sat` (signed).
pub const OCL_S_ADD_SAT: u32 = 143;
/// `add_sat` (unsigned).
pub const OCL_U_ADD_SAT: u32 = 144;
/// `hadd` (signed).
pub const OCL_S_HADD: u32 = 145;
/// `hadd` (unsigned).
pub const OCL_U_HADD: u32 = 146;
/// `rhadd` (signed).
pub const OCL_S_RHADD: u32 = 147;
/// `rhadd` (unsigned).
pub const OCL_U_RHADD: u32 = 148;
/// `clamp` (signed integer).
pub const OCL_S_CLAMP: u32 = 149;
/// `clamp` (unsigned integer).
pub const OCL_U_CLAMP: u32 = 150;
/// `clz`.
pub const OCL_CLZ: u32 = 151;
/// `ctz`.
pub const OCL_CTZ: u32 = 152;
/// `mad_hi` (signed).
pub const OCL_S_MAD_HI: u32 = 153;
/// `mad_sat` (unsigned).
pub const OCL_U_MAD_SAT: u32 = 154;
/// `mad_sat` (signed).
pub const OCL_S_MAD_SAT: u32 = 155;
/// `max` (signed integer).
pub const OCL_S_MAX: u32 = 156;
/// `max` (unsigned integer).
pub const OCL_U_MAX: u32 = 157;
/// `min` (signed integer).
pub const OCL_S_MIN: u32 = 158;
/// `min` (unsigned integer).
pub const OCL_U_MIN: u32 = 159;
/// `mul_hi` (signed).
pub const OCL_S_MUL_HI: u32 = 160;
/// `rotate`.
pub const OCL_ROTATE: u32 = 161;
/// `sub_sat` (signed).
pub const OCL_S_SUB_SAT: u32 = 162;
/// `sub_sat` (unsigned).
pub const OCL_U_SUB_SAT: u32 = 163;
/// `upsample` (unsigned).
pub const OCL_U_UPSAMPLE: u32 = 164;
/// `upsample` (signed).
pub const OCL_S_UPSAMPLE: u32 = 165;
/// `popcount`.
pub const OCL_POPCOUNT: u32 = 166;
/// `mad24` (signed).
pub const OCL_S_MAD24: u32 = 167;
/// `mad24` (unsigned).
pub const OCL_U_MAD24: u32 = 168;
/// `mul24` (signed).
pub const OCL_S_MUL24: u32 = 169;
/// `mul24` (unsigned).
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
/// `printf` (rejected at load).
pub const OCL_PRINTF: u32 = 184;
/// `prefetch` (no-op).
pub const OCL_PREFETCH: u32 = 185;
/// `bitselect`.
pub const OCL_BITSELECT: u32 = 186;
/// `select`.
pub const OCL_SELECT: u32 = 187;
/// `abs` (unsigned integer).
pub const OCL_U_ABS: u32 = 201;
/// `abs_diff` (unsigned).
pub const OCL_U_ABS_DIFF: u32 = 202;
/// `mul_hi` (unsigned).
pub const OCL_U_MUL_HI: u32 = 203;
/// `mad_hi` (unsigned).
pub const OCL_U_MAD_HI: u32 = 204;
