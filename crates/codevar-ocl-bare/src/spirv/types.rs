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

//! Type graph and memory layout of the values a kernel manipulates.
//!
//! Layout follows the natural C rules of the OpenCL C address spaces:
//! scalars occupy `width / 8` bytes, members sit at the next multiple of
//! their alignment, array strides round the element up to its alignment,
//! and a structure's size closes at its strictest member alignment. When
//! a third-party producer emitted explicit `Offset`/`ArrayStride`
//! decorations they win over the computed values — the workspace backend
//! emits none, so its binaries rely on exactly the rules below.

use alloc::vec::Vec;

use crate::error::{Error, Result};

/// One entry of the type table, indexed by `<id>`.
#[derive(Debug, Clone)]
pub(crate) enum Type {
    /// `OpTypeVoid`.
    Void,
    /// `OpTypeBool` (one byte in memory, `0`/`1` in registers).
    Bool,
    /// `OpTypeInt` — `width` bits, `signed` marks a signed integer.
    Int {
        /// Bit width (8, 16, 32 or 64).
        width: u16,
        /// Whether the integer is signed.
        signed: bool,
    },
    /// `OpTypeFloat` — `width` bits (32 or 64).
    Float {
        /// Bit width.
        width: u16,
    },
    /// `OpTypeVector`.
    Vector {
        /// Element type.
        elem: u32,
        /// Number of elements (`1..=16`).
        len: u16,
    },
    /// `OpTypeArray`.
    Array {
        /// Element type.
        elem: u32,
        /// Element count.
        len: u32,
    },
    /// `OpTypeStruct`.
    Struct {
        /// Field types in declaration order.
        fields: Vec<u32>,
    },
    /// `OpTypePointer`.
    Pointer {
        /// Storage class of the pointer.
        storage: u32,
        /// Pointed-to type.
        pointee: u32,
    },
    /// `OpTypeFunction`.
    Function {
        /// Return type.
        ret: u32,
        /// Parameter types.
        params: Vec<u32>,
    },
    /// A type the interpreter refuses (image, sampler, matrix, pipe,
    /// event, opaque, runtime array, …). Presence at load time is always
    /// an error.
    Unsupported,
}

/// Size and alignment of a type in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Size in bytes.
    pub size: u32,
    /// Alignment in bytes (a power of two).
    pub align: u32,
}

impl Layout {
    /// Rounds `value` up to the next multiple of `align`.
    #[must_use]
    #[inline]
    pub(crate) fn align_up(value: u32, align: u32) -> u32 {
        if align == 0 {
            return value;
        }
        value
            .div_ceil(align)
            .checked_mul(align)
            .unwrap_or(u32::MAX)
    }
}

/// The layout decorations collected from the annotation section.
#[derive(Debug, Default)]
pub(crate) struct LayoutDecorations {
    /// `(type id, member index, byte offset)` triples from `Offset`.
    pub member_offsets: Vec<(u32, u32, u32)>,
    /// `(type id, stride)` pairs from `ArrayStride`.
    pub array_strides: Vec<(u32, u32)>,
}

impl LayoutDecorations {
    /// The explicit offset of member `member` of `ty`, when decorated.
    #[must_use]
    pub(crate) fn member_offset(&self, ty: u32, member: u32) -> Option<u32> {
        self.member_offsets
            .iter()
            .find(|(t, m, _)| *t == ty && *m == member)
            .map(|(_, _, o)| *o)
    }

    /// The explicit stride of array `ty`, when decorated.
    #[must_use]
    pub(crate) fn array_stride(&self, ty: u32) -> Option<u32> {
        self.array_strides
            .iter()
            .find(|(t, _)| *t == ty)
            .map(|(_, s)| *s)
    }
}

/// Maximum nesting depth of the layout recursion — a malformed module
/// cannot turn a deep type chain into a stack overflow.
const MAX_LAYOUT_DEPTH: u32 = 64;

/// Computes the size and alignment of `ty`.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown id, an unsupported type kind,
/// a zero-sized structure or element, a non-power-of-two alignment from
/// a hostile decoration, or nesting deeper than [`MAX_LAYOUT_DEPTH`].
pub(crate) fn layout(
    types: &[Type],
    ty: u32,
    decorations: &LayoutDecorations,
) -> Result<Layout> {
    layout_at(types, ty, decorations, 0)
}

fn layout_at(
    types: &[Type],
    ty: u32,
    decorations: &LayoutDecorations,
    depth: u32,
) -> Result<Layout> {
    if depth >= MAX_LAYOUT_DEPTH {
        return Err(Error::InvalidBinary);
    }
    let entry = types.get(ty as usize).ok_or(Error::InvalidBinary)?;
    let result = match entry {
        Type::Void | Type::Function { .. } | Type::Unsupported => return Err(Error::InvalidBinary),
        Type::Bool => Layout { size: 1, align: 1 },
        Type::Int { width, .. } | Type::Float { width } => {
            let size = u32::from(*width) / 8;
            if size == 0 || !size.is_power_of_two() {
                return Err(Error::InvalidBinary);
            }
            Layout { size, align: size }
        }
        Type::Vector { elem, len } => {
            let inner = layout_at(types, *elem, decorations, depth + 1)?;
            let count = u32::from(*len);
            if count == 0 || count > 16 {
                return Err(Error::InvalidBinary);
            }
            // Tightly packed elements; the vector's alignment is the next
            // power-of-two multiple so `vec3` aligns like `vec4` (the C
            // rule for three-component vectors in every OpenCL ABI).
            Layout {
                size: inner.size.checked_mul(count).ok_or(Error::InvalidBinary)?,
                align: count
                    .next_power_of_two()
                    .checked_mul(inner.align)
                    .ok_or(Error::InvalidBinary)?,
            }
        }
        Type::Array { elem, len } => {
            let inner = layout_at(types, *elem, decorations, depth + 1)?;
            let stride = match decorations.array_stride(ty) {
                Some(stride) if stride >= inner.size => stride,
                Some(_) => return Err(Error::InvalidBinary),
                None => Layout::align_up(inner.size, inner.align),
            };
            Layout {
                size: stride
                    .checked_mul(*len)
                    .ok_or(Error::InvalidBinary)?,
                align: if inner.align == 0 {
                    1
                } else {
                    inner.align
                },
            }
        }
        Type::Struct { fields } => {
            let mut cursor = 0u32;
            let mut align = 1u32;
            for (index, field) in fields.iter().enumerate() {
                let inner = layout_at(types, *field, decorations, depth + 1)?;
                if inner.align == 0 || !inner.align.is_power_of_two() {
                    return Err(Error::InvalidBinary);
                }
                let offset = match decorations.member_offset(ty, index as u32) {
                    Some(offset) => offset,
                    None => Layout::align_up(cursor, inner.align),
                };
                cursor = offset
                    .checked_add(inner.size)
                    .ok_or(Error::InvalidBinary)?;
                align = align.max(inner.align);
            }
            Layout {
                size: Layout::align_up(cursor, align),
                align,
            }
        }
        Type::Pointer { .. } => Layout {
            size: core::mem::size_of::<usize>() as u32,
            align: core::mem::align_of::<usize>() as u32,
        },
    };
    Ok(result)
}

/// The byte size of `ty` in memory.
///
/// # Errors
///
/// [`Error::InvalidBinary`] as for [`layout`].
pub(crate) fn size_of(
    types: &[Type],
    ty: u32,
    decorations: &LayoutDecorations,
) -> Result<u32> {
    Ok(layout(types, ty, decorations)?.size)
}

/// The byte offset of each member of the structure `ty`, in declaration
/// order — the same walk [`layout`] performs for its own size, returned
/// per member so the interpreter can binary-search `(type, member)`
/// keys instead of recomputing layouts on every access chain.
///
/// # Errors
///
/// [`Error::InvalidBinary`] when `ty` is not a structure, a field has no
/// layout, a hostile decoration produces a non-power-of-two alignment or
/// an overflowing offset, or nesting exceeds [`MAX_LAYOUT_DEPTH`].
pub(crate) fn member_offsets(
    types: &[Type],
    ty: u32,
    decorations: &LayoutDecorations,
) -> Result<Vec<(u32, u32)>> {
    let fields = match types.get(ty as usize).ok_or(Error::InvalidBinary)? {
        Type::Struct { fields } => fields,
        _ => return Err(Error::InvalidBinary),
    };
    let mut offsets: Vec<(u32, u32)> = Vec::new();
    offsets
        .try_reserve_exact(fields.len())
        .map_err(|_| Error::OutOfMemory)?;
    let mut cursor = 0u32;
    for (index, field) in fields.iter().enumerate() {
        let inner = layout_at(types, *field, decorations, 1)?;
        if inner.align == 0 || !inner.align.is_power_of_two() {
            return Err(Error::InvalidBinary);
        }
        let offset = match decorations.member_offset(ty, index as u32) {
            Some(offset) => offset,
            None => Layout::align_up(cursor, inner.align),
        };
        cursor = offset
            .checked_add(inner.size)
            .ok_or(Error::InvalidBinary)?;
        offsets.push((index as u32, offset));
    }
    Ok(offsets)
}

/// Maximum number of scalar lanes a register-resident value may hold.
///
/// Vectors top out at 16 lanes; anything larger (a structure or array
/// carried in a register instead of memory) is rejected at load time so
/// the interpreter's scratch buffers stay fixed size.
pub(crate) const MAX_VALUE_LANES: u32 = 64;

/// The number of scalar lanes a value of `ty` occupies in a register.
///
/// A register holds one `u64` per scalar leaf — booleans, integers,
/// floats and pointers take one lane each regardless of width; vectors,
/// arrays and structures concatenate their members' lanes. This is the
/// interpreter's in-register representation and it is *not* the memory
/// layout: [`load_lanes`] and [`store_lanes`] convert at the boundary.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for an unknown id, a void/unsupported kind
/// with a value, nesting deeper than [`MAX_LAYOUT_DEPTH`], or a leaf
/// count above [`MAX_VALUE_LANES`].
pub(crate) fn lanes(
    types: &[Type],
    ty: u32,
    decorations: &LayoutDecorations,
) -> Result<u32> {
    let count = lanes_at(types, ty, decorations, 0)?;
    if count > MAX_VALUE_LANES {
        return Err(Error::InvalidBinary);
    }
    Ok(count)
}

/// Scalar-leaf count of `ty` without the `MAX_VALUE_LANES` cap.
///
/// Used only while parsing, for values that live in the uncapped constant
/// pool or as whole static objects (big composites, nulls, arena
/// initializers). Slot values used by the interpreter must go through
/// [`lanes`] instead, which enforces the 64-lane register invariant.
pub(crate) fn lanes_any(
    types: &[Type],
    ty: u32,
    decorations: &LayoutDecorations,
) -> Result<u32> {
    lanes_at(types, ty, decorations, 0)
}

fn lanes_at(
    types: &[Type],
    ty: u32,
    decorations: &LayoutDecorations,
    depth: u32,
) -> Result<u32> {
    if depth >= MAX_LAYOUT_DEPTH {
        return Err(Error::InvalidBinary);
    }
    let entry = types.get(ty as usize).ok_or(Error::InvalidBinary)?;
    let count = match entry {
        Type::Void | Type::Function { .. } | Type::Unsupported => return Err(Error::InvalidBinary),
        Type::Bool | Type::Int { .. } | Type::Float { .. } | Type::Pointer { .. } => 1,
        Type::Vector { elem, len } => {
            lanes_at(types, *elem, decorations, depth + 1)?
                .checked_mul(u32::from(*len))
                .ok_or(Error::InvalidBinary)?
        }
        Type::Array { elem, len } => {
            lanes_at(types, *elem, decorations, depth + 1)?
                .checked_mul(*len)
                .ok_or(Error::InvalidBinary)?
        }
        Type::Struct { fields } => {
            let mut total = 0u32;
            for field in fields {
                total = total
                    .checked_add(lanes_at(types, *field, decorations, depth + 1)?)
                    .ok_or(Error::InvalidBinary)?;
            }
            total
        }
    };
    Ok(count)
}

/// Reads a value of `ty` from `base` into `out`, one `u64` per lane.
///
/// # Safety
///
/// `base` must point at at least `size_of(ty)` readable bytes.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for layout failures or an `out` shorter than
/// the value's lane count.
pub(crate) unsafe fn load_lanes(
    types: &[Type],
    ty: u32,
    decorations: &LayoutDecorations,
    base: *const u8,
    out: &mut [u64],
) -> Result<u32> {
    if base.is_null() {
        return Err(Error::InvalidArgument);
    }
    // Validate the type (and the lane cap) before touching `out`.
    lanes(types, ty, decorations)?;
    load_lanes_at(types, ty, decorations, base, out, 0)
}

fn load_lanes_at(
    types: &[Type],
    ty: u32,
    decorations: &LayoutDecorations,
    base: *const u8,
    out: &mut [u64],
    depth: u32,
) -> Result<u32> {
    if depth >= MAX_LAYOUT_DEPTH {
        return Err(Error::InvalidBinary);
    }
    let entry = types.get(ty as usize).ok_or(Error::InvalidBinary)?;
    let mut used = 0u32;
    match entry {
        Type::Void | Type::Function { .. } | Type::Unsupported => return Err(Error::InvalidBinary),
        Type::Bool => {
            // SAFETY: caller guarantees `size_of(ty)` readable bytes (1).
            let byte = unsafe { *base };
            *out.first_mut().ok_or(Error::InvalidBinary)? = u64::from(byte != 0);
            used = 1;
        }
        Type::Int { width, .. } | Type::Float { width } => {
            // SAFETY: caller guarantees `width / 8` readable bytes.
            let lane = unsafe { read_scalar(base, *width)? };
            *out.first_mut().ok_or(Error::InvalidBinary)? = lane;
            used = 1;
        }
        Type::Pointer { .. } => {
            let mut bytes = [0u8; core::mem::size_of::<usize>()];
            // SAFETY: caller guarantees the pointer-sized region.
            unsafe {
                core::ptr::copy_nonoverlapping(base, bytes.as_mut_ptr(), bytes.len());
            }
            *out.first_mut().ok_or(Error::InvalidBinary)? = usize::from_ne_bytes(bytes) as u64;
            used = 1;
        }
        Type::Vector { elem, len } => {
            let inner_size = layout(types, *elem, decorations)?.size;
            for index in 0..u32::from(*len) {
                // SAFETY: elements are packed inside the vector's size.
                let at = unsafe { base.add((index * inner_size) as usize) };
                let slice = out
                    .get_mut(used as usize..)
                    .ok_or(Error::InvalidBinary)?;
                used += load_lanes_at(types, *elem, decorations, at, slice, depth + 1)?;
            }
        }
        Type::Array { elem, len } => {
            let inner = layout(types, *elem, decorations)?;
            let stride = match decorations.array_stride(ty) {
                Some(stride) if stride >= inner.size => stride,
                Some(_) => return Err(Error::InvalidBinary),
                None => Layout::align_up(inner.size, inner.align),
            };
            for index in 0..*len {
                // SAFETY: caller guarantees the whole array; stride lands
                // inside it.
                let at = unsafe { base.add((index * stride) as usize) };
                let slice = out
                    .get_mut(used as usize..)
                    .ok_or(Error::InvalidBinary)?;
                used += load_lanes_at(types, *elem, decorations, at, slice, depth + 1)?;
            }
        }
        Type::Struct { fields } => {
            let mut cursor = 0u32;
            for (index, field) in fields.iter().enumerate() {
                let inner = layout(types, *field, decorations)?;
                let offset = match decorations.member_offset(ty, index as u32) {
                    Some(offset) => offset,
                    None => Layout::align_up(cursor, inner.align),
                };
                cursor = offset + inner.size;
                // SAFETY: members live inside the structure's size.
                let at = unsafe { base.add(offset as usize) };
                let slice = out
                    .get_mut(used as usize..)
                    .ok_or(Error::InvalidBinary)?;
                used += load_lanes_at(types, *field, decorations, at, slice, depth + 1)?;
            }
        }
    }
    Ok(used)
}

/// Writes `count` lanes from `value` into `base` as a value of `ty`.
///
/// # Safety
///
/// `base` must point at at least `size_of(ty)` writable bytes, and
/// `value` must hold at least the value's lane count.
///
/// # Errors
///
/// [`Error::InvalidBinary`] for layout failures or a `value` shorter
/// than the value's lane count.
pub(crate) unsafe fn store_lanes(
    types: &[Type],
    ty: u32,
    decorations: &LayoutDecorations,
    base: *mut u8,
    value: &[u64],
) -> Result<()> {
    if base.is_null() {
        return Err(Error::InvalidArgument);
    }
    store_lanes_at(types, ty, decorations, base, value, 0)
}

fn store_lanes_at(
    types: &[Type],
    ty: u32,
    decorations: &LayoutDecorations,
    base: *mut u8,
    value: &[u64],
    depth: u32,
) -> Result<()> {
    if depth >= MAX_LAYOUT_DEPTH {
        return Err(Error::InvalidBinary);
    }
    let entry = types.get(ty as usize).ok_or(Error::InvalidBinary)?;
    let mut used = 0u32;
    match entry {
        Type::Void | Type::Function { .. } | Type::Unsupported => return Err(Error::InvalidBinary),
        Type::Bool => {
            let lane = *value.first().ok_or(Error::InvalidBinary)?;
            // SAFETY: caller guarantees one writable byte.
            unsafe { *base = u8::from(lane != 0) };
        }
        Type::Int { width, .. } | Type::Float { width } => {
            let lane = *value.first().ok_or(Error::InvalidBinary)?;
            // SAFETY: caller guarantees `width / 8` writable bytes.
            unsafe { write_scalar(base, *width, lane) };
        }
        Type::Pointer { .. } => {
            let bytes = (*value.first().ok_or(Error::InvalidBinary)? as usize).to_ne_bytes();
            // SAFETY: caller guarantees pointer-sized writable bytes.
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), base, bytes.len());
            }
        }
        Type::Vector { elem, len } => {
            let inner_size = layout(types, *elem, decorations)?.size;
            for index in 0..u32::from(*len) {
                // SAFETY: elements are packed inside the vector's size.
                let at = unsafe { base.add((index * inner_size) as usize) };
                let slice = value
                    .get(used as usize..)
                    .ok_or(Error::InvalidBinary)?;
                let inner_lanes = lanes(types, *elem, decorations)?;
                let slice = slice
                    .get(..inner_lanes as usize)
                    .ok_or(Error::InvalidBinary)?;
                store_lanes_at(types, *elem, decorations, at, slice, depth + 1)?;
                used += inner_lanes;
            }
        }
        Type::Array { elem, len } => {
            let inner = layout(types, *elem, decorations)?;
            let stride = match decorations.array_stride(ty) {
                Some(stride) if stride >= inner.size => stride,
                Some(_) => return Err(Error::InvalidBinary),
                None => Layout::align_up(inner.size, inner.align),
            };
            for index in 0..*len {
                // SAFETY: caller guarantees the whole array.
                let at = unsafe { base.add((index * stride) as usize) };
                let inner_lanes = lanes(types, *elem, decorations)?;
                let slice = value
                    .get(used as usize..used as usize + inner_lanes as usize)
                    .ok_or(Error::InvalidBinary)?;
                store_lanes_at(types, *elem, decorations, at, slice, depth + 1)?;
                used += inner_lanes;
            }
        }
        Type::Struct { fields } => {
            let mut cursor = 0u32;
            for (index, field) in fields.iter().enumerate() {
                let inner = layout(types, *field, decorations)?;
                let offset = match decorations.member_offset(ty, index as u32) {
                    Some(offset) => offset,
                    None => Layout::align_up(cursor, inner.align),
                };
                cursor = offset + inner.size;
                let inner_lanes = lanes(types, *field, decorations)?;
                let slice = value
                    .get(used as usize..used as usize + inner_lanes as usize)
                    .ok_or(Error::InvalidBinary)?;
                // SAFETY: members live inside the structure's size.
                let at = unsafe { base.add(offset as usize) };
                store_lanes_at(types, *field, decorations, at, slice, depth + 1)?;
                used += inner_lanes;
            }
        }
    }
    Ok(())
}

/// Reads a scalar of `width` bits from `base`, zero-extended to `u64`.
///
/// # Safety
///
/// `base` must be readable for `width / 8` bytes.
unsafe fn read_scalar(base: *const u8, width: u16) -> Result<u64> {
    match width {
        8 => {
            // SAFETY: one byte.
            Ok(u64::from(unsafe { *base }))
        }
        16 => {
            let mut bytes = [0u8; 2];
            // SAFETY: two bytes.
            unsafe {
                core::ptr::copy_nonoverlapping(base, bytes.as_mut_ptr(), 2);
            }
            Ok(u64::from(u16::from_ne_bytes(bytes)))
        }
        32 => {
            let mut bytes = [0u8; 4];
            // SAFETY: four bytes.
            unsafe {
                core::ptr::copy_nonoverlapping(base, bytes.as_mut_ptr(), 4);
            }
            Ok(u64::from(u32::from_ne_bytes(bytes)))
        }
        64 => {
            let mut bytes = [0u8; 8];
            // SAFETY: eight bytes.
            unsafe {
                core::ptr::copy_nonoverlapping(base, bytes.as_mut_ptr(), 8);
            }
            Ok(u64::from_ne_bytes(bytes))
        }
        _ => Err(Error::InvalidBinary),
    }
}

/// Writes the low `width / 8` bytes of `lane` to `base` in native order.
///
/// # Safety
///
/// `base` must be writable for `width / 8` bytes.
unsafe fn write_scalar(base: *mut u8, width: u16, lane: u64) {
    match width {
        8 => {
            // SAFETY: one byte.
            unsafe { *base = lane as u8 };
        }
        16 => {
            // SAFETY: two bytes.
            unsafe { core::ptr::copy_nonoverlapping((lane as u16).to_ne_bytes().as_ptr(), base, 2) };
        }
        32 => {
            // SAFETY: four bytes.
            unsafe { core::ptr::copy_nonoverlapping((lane as u32).to_ne_bytes().as_ptr(), base, 4) };
        }
        64 => {
            // SAFETY: eight bytes.
            unsafe { core::ptr::copy_nonoverlapping(lane.to_ne_bytes().as_ptr(), base, 8) };
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Vec<Type> {
        let mut types = vec![Type::Unsupported; 8];
        // 1: i32, 2: f32, 3: vec3<f32>, 4: i64,
        // 5: [4 x i32], 6: struct {i32, f32}, 7: ptr
        types[1] = Type::Int { width: 32, signed: true };
        types[2] = Type::Float { width: 32 };
        types[3] = Type::Vector { elem: 2, len: 3 };
        types[4] = Type::Int { width: 64, signed: false };
        types[5] = Type::Array { elem: 1, len: 4 };
        types[6] = Type::Struct { fields: vec![1, 2] };
        types[7] = Type::Pointer { storage: 5, pointee: 2 };
        types
    }

    #[test]
    fn scalars_follow_their_width() {
        let types = table();
        let none = LayoutDecorations::default();
        assert_eq!(
            layout(&types, 1, &none).expect("i32"),
            Layout { size: 4, align: 4 }
        );
        assert_eq!(
            layout(&types, 4, &none).expect("i64"),
            Layout { size: 8, align: 8 }
        );
        assert_eq!(
            layout(&types, 7, &none).expect("pointer"),
            Layout {
                size: core::mem::size_of::<usize>() as u32,
                align: core::mem::align_of::<usize>() as u32
            }
        );
    }

    #[test]
    fn vectors_are_tight_but_align_like_the_next_pow2() {
        let types = table();
        let none = LayoutDecorations::default();
        let vec3 = layout(&types, 3, &none).expect("vec3");
        assert_eq!(vec3.size, 12);
        assert_eq!(vec3.align, 16);
        let types2 = {
            let mut t = table();
            t[3] = Type::Vector { elem: 4, len: 3 };
            t
        };
        let long3 = layout(&types2, 3, &none).expect("long3");
        assert_eq!(long3.size, 24);
        assert_eq!(long3.align, 32);
    }

    #[test]
    fn arrays_round_the_stride_up_to_the_element_alignment() {
        let types = table();
        let none = LayoutDecorations::default();
        // 4 x i32 -> stride 4, size 16.
        assert_eq!(
            layout(&types, 5, &none).expect("array"),
            Layout { size: 16, align: 4 }
        );
        let vec_arrays = {
            let mut t = table();
            t[5] = Type::Array { elem: 3, len: 2 };
            t
        };
        // 2 x vec3: stride aligns the 12-byte element to 16.
        assert_eq!(
            layout(&vec_arrays, 5, &none).expect("array"),
            Layout { size: 32, align: 16 }
        );
        // An explicit stride replaces the rounded one when it still
        // fits the element: 2 x vec3 (12 bytes each) at stride 12.
        let decorated = LayoutDecorations {
            member_offsets: vec![],
            array_strides: vec![(5, 12)],
        };
        assert_eq!(
            layout(&vec_arrays, 5, &decorated).expect("stride 12"),
            Layout { size: 24, align: 16 }
        );
        // A stride smaller than the element would overlap: rejected.
        let overlapping = LayoutDecorations {
            member_offsets: vec![],
            array_strides: vec![(5, 8)],
        };
        assert_eq!(
            layout(&vec_arrays, 5, &overlapping),
            Err(Error::InvalidBinary)
        );
    }

    #[test]
    fn structs_use_natural_alignment_and_honor_offsets() {
        let types = table();
        let none = LayoutDecorations::default();
        // struct {i32, f32} -> 8/4.
        assert_eq!(
            layout(&types, 6, &none).expect("struct"),
            Layout { size: 8, align: 4 }
        );
        // struct {vec3, i32}: vec3 at 0 (size 12), i32 packs at 12,
        // and the tail rounds up to the 16-byte vector alignment.
        let mixed = {
            let mut t = table();
            t[6] = Type::Struct { fields: vec![3, 1] };
            t
        };
        assert_eq!(
            layout(&mixed, 6, &none).expect("struct"),
            Layout { size: 16, align: 16 }
        );
        // Explicit Offset wins: i32 at 12 (same natural position).
        let spaced = LayoutDecorations {
            member_offsets: vec![(6, 1, 12)],
            array_strides: vec![],
        };
        let laid = layout(&mixed, 6, &spaced).expect("decorated");
        assert_eq!(laid.size, 16); // align still 16 from vec3
    }

    #[test]
    fn unsupported_types_and_bad_ids_fail_closed() {
        let types = table();
        let none = LayoutDecorations::default();
        assert_eq!(layout(&types, 0, &none), Err(Error::InvalidBinary));
        assert_eq!(layout(&types, 99, &none), Err(Error::InvalidBinary));
        let mut image = types.clone();
        image[1] = Type::Unsupported;
        assert_eq!(layout(&image, 1, &none), Err(Error::InvalidBinary));
        // Too deep.
        let mut deep = vec![Type::Unsupported; 80];
        for i in 1..79 {
            deep[i] = Type::Struct { fields: vec![i as u32 + 1] };
        }
        deep[79] = Type::Int { width: 32, signed: true };
        assert_eq!(layout(&deep, 1, &none), Err(Error::InvalidBinary));
    }
}
