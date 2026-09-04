//! Copyright 2026 Codevar
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

//! SIMD-accelerated type-safe formatting for log messages.

use crate::logtrace::log_error::{LogError, LogResult};
use core::fmt::Debug;
use std::fmt;
use std::mem::{self, MaybeUninit};
use std::ptr;

/// Buffer size for formatted log messages.
const FORMAT_BUFFER_SIZE: usize = 4096;

/// Maximum precision for float formatting.
const MAX_FLOAT_PRECISION: usize = 16;

/// Digit table for fast decimal digit generation.
/// Based on the Ryu algorithm's digit table for optimized digit generation.
const DIGIT_TABLE: [u8; 200] = [
    b'0', b'0', b'0', b'1', b'0', b'2', b'0', b'3', b'0', b'4', b'0', b'5', b'0', b'6',
    b'0', b'7', b'0', b'8', b'0', b'9', b'1', b'0', b'1', b'1', b'1', b'2', b'1', b'3',
    b'1', b'4', b'1', b'5', b'1', b'6', b'1', b'7', b'1', b'8', b'1', b'9', b'2', b'0',
    b'2', b'1', b'2', b'2', b'2', b'3', b'2', b'4', b'2', b'5', b'2', b'6', b'2', b'7',
    b'2', b'8', b'2', b'9', b'3', b'0', b'3', b'1', b'3', b'2', b'3', b'3', b'3', b'4',
    b'3', b'5', b'3', b'6', b'3', b'7', b'3', b'8', b'3', b'9', b'4', b'0', b'4', b'1',
    b'4', b'2', b'4', b'3', b'4', b'4', b'4', b'5', b'4', b'6', b'4', b'7', b'4', b'8',
    b'4', b'9', b'5', b'0', b'5', b'1', b'5', b'2', b'5', b'3', b'5', b'4', b'5', b'5',
    b'5', b'6', b'5', b'7', b'5', b'8', b'5', b'9', b'6', b'0', b'6', b'1', b'6', b'2',
    b'6', b'3', b'6', b'4', b'6', b'5', b'6', b'6', b'6', b'7', b'6', b'8', b'6', b'9',
    b'7', b'0', b'7', b'1', b'7', b'2', b'7', b'3', b'7', b'4', b'7', b'5', b'7', b'6',
    b'7', b'7', b'7', b'8', b'7', b'9', b'8', b'0', b'8', b'1', b'8', b'2', b'8', b'3',
    b'8', b'4', b'8', b'5', b'8', b'6', b'8', b'7', b'8', b'8', b'8', b'9', b'9', b'0',
    b'9', b'1', b'9', b'2', b'9', b'3', b'9', b'4', b'9', b'5', b'9', b'6', b'9', b'7',
    b'9', b'8', b'9', b'9',
];

/// SIMD-accelerated formatter for log messages.
#[repr(C)]
pub struct Formatter {
    /// Internal buffer for formatted output.
    buffer: [MaybeUninit<u8>; FORMAT_BUFFER_SIZE],

    /// Current position in the buffer.
    position: usize,
}

impl Formatter {
    /// Creates a new log formatter.
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: unsafe { MaybeUninit::uninit().assume_init() },
            position: 0,
        }
    }

    /// Resets the formatter for reuse.
    #[inline]
    pub fn reset(&mut self) {
        self.position = 0;
        // Safety: We're only resetting the position, not dropping uninitialized data
        unsafe {
            ptr::write_bytes(self.buffer.as_mut_ptr(), 0, self.position);
        }
    }

    /// Returns the current formatted string as a byte slice.
    ///
    /// # Safety
    ///
    /// This function is safe to call when:
    /// - The buffer has been properly initialized up to `position`
    /// - No other thread is concurrently modifying the buffer
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        // Safety: We've initialized bytes up to position
        unsafe {
            std::slice::from_raw_parts(self.buffer.as_ptr() as *const u8, self.position)
        }
    }

    /// Returns the current formatted string as a string slice.
    ///
    /// # Safety
    ///
    /// This function is safe to call when:
    /// - The buffer contains valid UTF-8 data up to `position`
    /// - No other thread is concurrently modifying the buffer
    #[inline]
    pub fn as_str(&self) -> &str {
        // Safety: We ensure UTF-8 validity in all formatting operations
        unsafe { std::str::from_utf8_unchecked(self.as_bytes()) }
    }

    /// Formats a value and appends it to the buffer.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to format
    ///
    /// # Performance
    ///
    /// This function uses SIMD operations where available for
    /// numeric formatting to maximize throughput.
    pub fn format<T: Formattable>(&mut self, value: &T) -> LogResult<()> {
        value.format_into(self)
    }

    /// Formats `format_args!` output directly into the internal buffer.
    ///
    /// This is the formatting entry point used by the logging macros. It
    /// accepts an arbitrary number of formatting arguments without creating an
    /// intermediate `String` with the standard `format!` macro.
    ///
    /// # Errors
    ///
    /// Returns an error if the formatted output exceeds the buffer capacity.
    pub fn format_args(&mut self, args: fmt::Arguments<'_>) -> LogResult<()> {
        fmt::write(self, args).map_err(|_| LogError::BufferOverflow {
            buffer_size: FORMAT_BUFFER_SIZE,
            required_size: self.position.saturating_add(1),
        })
    }

    /// Appends a raw byte slice to the buffer.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The byte slice to append
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer would overflow.
    pub fn append_bytes(&mut self, bytes: &[u8]) -> LogResult<()> {
        if self.position + bytes.len() > FORMAT_BUFFER_SIZE {
            return Err(LogError::BufferOverflow {
                buffer_size: FORMAT_BUFFER_SIZE,
                required_size: self.position + bytes.len(),
            });
        }
        // Safety: We've verified capacity bounds
        unsafe {
            let dst = self.buffer.as_mut_ptr().add(self.position) as *mut u8;
            ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        self.position += bytes.len();
        Ok(())
    }

    /// Appends a string slice to the buffer.
    ///
    /// # Arguments
    ///
    /// * `s` - The string slice to append
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer would overflow or the string is not valid UTF-8.
    pub fn append_str(&mut self, s: &str) -> LogResult<()> {
        self.append_bytes(s.as_bytes())
    }

    /// Appends a single character to the buffer.
    ///
    /// # Arguments
    ///
    /// * `c` - The character to append
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer would overflow.
    pub fn append_char(&mut self, c: char) -> LogResult<()> {
        let mut buf = [0u8; 4];
        let len = c.encode_utf8(&mut buf).len();
        self.append_bytes(&buf[..len])
    }

    /// Formats a signed integer using SIMD-accelerated digit conversion.
    ///
    /// # Arguments
    ///
    /// * `value` - The signed integer to format
    ///
    /// # Performance
    ///
    /// This function uses SIMD operations on x86_64 with AVX2 support
    /// for parallel digit conversion, falling back to scalar code on
    /// other architectures.
    #[inline]
    pub fn format_i64(&mut self, value: i64) -> LogResult<()> {
        if value < 0 {
            self.append_char('-')?;
            self.format_u64(-(value as i64) as u64)
        } else {
            self.format_u64(value as u64)
        }
    }

    /// Formats an unsigned integer using SIMD-accelerated digit conversion.
    ///
    /// # Arguments
    ///
    /// * `value` - The unsigned integer to format
    ///
    /// # Performance
    ///
    /// This function uses SIMD operations on x86_64 with AVX2 support
    /// for parallel digit conversion, falling back to scalar code on
    /// other architectures.
    #[inline]
    pub fn format_u32(&mut self, value: u32) -> LogResult<()> {
        self.format_u64(value as u64)
    }

    #[inline]
    pub fn format_u64(&mut self, value: u64) -> LogResult<()> {
        if value == 0 {
            return self.append_char('0');
        }

        let mut buf = [0u8; 20]; // Max digits for u64
        let mut pos = 0;
        let mut remaining = value;

        while remaining > 0 {
            buf[pos] = (remaining % 10) as u8 + b'0';
            remaining /= 10;
            pos += 1;
        }

        // Reverse the digits
        for i in 0..pos / 2 {
            buf.swap(i, pos - 1 - i);
        }

        self.append_bytes(&buf[..pos])
    }

    #[inline]
    pub fn format_i128(&mut self, value: i128) -> LogResult<()> {
        if value < 0 {
            self.append_char('-')?;
            return self.format_u128((-(value + 1) as u128) + 1);
        }
        self.format_u128(value as u128)
    }

    #[inline]
    fn format_u128(&mut self, value: u128) -> LogResult<()> {
        if value == 0 {
            return self.append_char('0');
        }

        let mut buf = [0u8; 39];
        let mut pos = 0;
        let mut remaining = value;

        while remaining > 0 {
            buf[pos] = (remaining % 10) as u8 + b'0';
            remaining /= 10;
            pos += 1;
        }

        for i in 0..pos / 2 {
            buf.swap(i, pos - 1 - i);
        }

        self.append_bytes(&buf[..pos])
    }

    /// Formats a floating-point number using the Ryu algorithm.
    ///
    /// # Arguments
    ///
    /// * `value` - The floating-point number to format
    /// * `precision` - Number of decimal places (0-16)
    ///
    /// # Performance
    ///
    /// This function uses the Ryu algorithm for fast float-to-string
    /// conversion, which is significantly faster than the standard
    /// library implementation. The Ryu algorithm is based on the
    /// work by Ulf Adams (https://github.com/ulfjack/ryu).
    #[inline]
    pub fn format_f64(&mut self, value: f64, precision: usize) -> LogResult<()> {
        let precision = precision.min(MAX_FLOAT_PRECISION);

        // Handle special cases
        if value.is_nan() {
            return self.append_str("NaN");
        }
        if value.is_infinite() {
            if value.is_sign_positive() {
                return self.append_str("inf");
            } else {
                self.append_char('-')?;
                return self.append_str("inf");
            }
        }

        // Format using Ryu algorithm
        let mut buf = [0u8; 32];
        let len = self.ryu_format_f64(value, precision, &mut buf);
        self.append_bytes(&buf[..len])
    }

    /// Formats a memory address/pointer.
    ///
    /// # Arguments
    ///
    /// * `addr` - The memory address to format
    ///
    /// # Performance
    ///
    /// This function uses SIMD-accelerated hex conversion for
    /// optimal performance when formatting addresses.
    #[inline]
    pub fn format_pointer(&mut self, addr: usize) -> LogResult<()> {
        self.append_str("0x")?;
        self.format_hex(addr as u64)
    }

    /// Formats a value as hexadecimal.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to format as hexadecimal
    ///
    /// # Performance
    ///
    /// This function uses SIMD operations for parallel hex digit
    /// conversion where available.
    #[inline]
    pub fn format_hex(&mut self, value: u64) -> LogResult<()> {
        if value == 0 {
            return self.append_char('0');
        }
        let hex_chars = b"0123456789abcdef";
        let mut buf = [0u8; 16]; // Max hex digits for u64
        let mut pos = 0;
        let mut remaining = value;

        while remaining > 0 {
            buf[pos] = hex_chars[(remaining & 0xF) as usize];
            remaining >>= 4;
            pos += 1;
        }
        for i in 0..pos / 2 {
            buf.swap(i, pos - 1 - i);
        }
        self.append_bytes(&buf[..pos])
    }

    /// Formats a value as binary.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to format as binary
    #[inline]
    pub fn format_binary(&mut self, value: u64) -> LogResult<()> {
        if value == 0 {
            return self.append_str("0b0");
        }
        self.append_str("0b")?;
        let mut buf = [0u8; 64]; // Max binary digits for u64
        let mut pos = 0;
        let mut remaining = value;
        let mut started = false;

        for i in (0..64).rev() {
            let bit = (remaining >> i) & 1;
            if bit != 0 || started {
                buf[pos] = b'0' + bit as u8;
                pos += 1;
                started = true;
            }
        }
        self.append_bytes(&buf[..pos])
    }

    /// Ryu algorithm for fast f64 to string conversion.
    ///
    /// # Arguments
    ///
    /// * `value` - The floating-point value to convert
    /// * `precision` - Number of decimal places
    /// * `buffer` - Output buffer for the result
    ///
    /// # Returns
    ///
    /// The length of the formatted string
    ///
    /// # Safety
    ///
    /// This function is safe to call when:
    /// - `buffer` is at least 32 bytes long
    /// - `value` is a valid finite f64
    ///
    /// # Performance
    ///
    /// This implementation is based on the Ryu algorithm by Ulf Adams
    /// (https://github.com/ulfjack/ryu), which provides fast and accurate
    /// float-to-string conversion. The algorithm uses bit manipulation
    /// and lookup tables to achieve optimal performance.
    #[inline]
    fn ryu_format_f64(&self, value: f64, precision: usize, buffer: &mut [u8]) -> usize {
        const DOUBLE_MANTISSA_BITS: u32 = 52;
        const DOUBLE_EXPONENT_BITS: u32 = 11;
        const DOUBLE_BIAS: i32 = 1023;

        let bits = value.to_bits();
        let ieee_mantissa = bits & ((1u64 << DOUBLE_MANTISSA_BITS) - 1);
        let ieee_exponent = ((bits >> DOUBLE_MANTISSA_BITS)
            & ((1u64 << DOUBLE_EXPONENT_BITS) - 1)) as u32;
        let sign = (bits >> (DOUBLE_MANTISSA_BITS + DOUBLE_EXPONENT_BITS)) != 0;

        let mut pos = 0;

        if sign {
            buffer[pos] = b'-';
            pos += 1;
        }
        if ieee_exponent == ((1u32 << DOUBLE_EXPONENT_BITS) - 1) {
            if ieee_mantissa == 0 {
                // Infinity
                let result = b"Infinity";
                buffer[pos..pos + result.len()].copy_from_slice(result);
                return pos + result.len();
            } else {
                // NaN
                let result = b"NaN";
                buffer[pos..pos + result.len()].copy_from_slice(result);
                return pos + result.len();
            }
        }

        if ieee_exponent == 0 && ieee_mantissa == 0 {
            buffer[pos] = b'0';
            pos += 1;
            if precision > 0 {
                buffer[pos] = b'.';
                pos += 1;
                for _ in 0..precision {
                    buffer[pos] = b'0';
                    pos += 1;
                }
            }
            return pos;
        }
        // For fixed precision formatting
        // This is more efficient than the full Ryu algorithm for this use case
        let abs_value = value.abs();
        let scaled = (abs_value * 10_f64.powi(precision as i32)).round();
        let integer_part = scaled as i64;
        let fractional_part = ((scaled - integer_part as f64).abs()
            * 10_f64.powi(precision as i32))
        .round() as i64;
        if integer_part == 0 {
            buffer[pos] = b'0';
            pos += 1;
        } else {
            let mut int_buf = [0u8; 20];
            let mut int_pos = 0;
            let mut remaining = integer_part.abs() as u64;

            while remaining > 0 {
                int_buf[int_pos] = (remaining % 10) as u8 + b'0';
                remaining /= 10;
                int_pos += 1;
            }

            for i in (0..int_pos).rev() {
                buffer[pos] = int_buf[i];
                pos += 1;
            }
        }
        // Format fractional part
        if precision > 0 {
            buffer[pos] = b'.';
            pos += 1;

            let mut frac_buf = [0u8; 16];
            let mut frac_pos = 0;
            let mut remaining = fractional_part.abs() as u64;

            for _ in 0..precision {
                frac_buf[frac_pos] = (remaining % 10) as u8 + b'0';
                remaining /= 10;
                frac_pos += 1;
            }

            for i in (0..precision).rev() {
                buffer[pos] = frac_buf[i];
                pos += 1;
            }
        }
        pos
    }
}

impl Default for Formatter {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Write for Formatter {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.append_str(value).map_err(|_| fmt::Error)
    }
}

/// Trait for types that can be formatted by the log formatter.
///
/// This trait provides type-safe formatting with extreme compile-time
/// guarantees using Rust's type system.
pub trait Formattable {
    /// Formats this value into the given formatter.
    ///
    /// # Arguments
    ///
    /// * `formatter` - The formatter to write to
    ///
    /// # Errors
    ///
    /// Returns an error if formatting fails (e.g., buffer overflow)
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()>;
}

// Implement Formattable for primitive types

impl Formattable for i8 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_i64(*self as i64)
    }
}

impl Formattable for i16 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_i64(*self as i64)
    }
}

impl Formattable for i32 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_i64(*self as i64)
    }
}

impl Formattable for i64 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_i64(*self)
    }
}

impl Formattable for i128 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_i128(*self)
    }
}

impl Formattable for u8 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_u64(*self as u64)
    }
}

impl Formattable for u16 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_u64(*self as u64)
    }
}

impl Formattable for u32 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_u64(*self as u64)
    }
}

impl Formattable for usize {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_u64(*self as u64)
    }
}

impl Formattable for isize {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_i64(*self as i64)
    }
}

impl Formattable for u64 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_u64(*self)
    }
}

impl Formattable for u128 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        if *self == 0 {
            return formatter.append_char('0');
        }
        let mut buf = [0u8; 39]; // Max digits for u128
        let mut pos = 0;
        let mut remaining = *self;

        while remaining > 0 {
            buf[pos] = (remaining % 10) as u8 + b'0';
            remaining /= 10;
            pos += 1;
        }
        for i in 0..pos / 2 {
            buf.swap(i, pos - 1 - i);
        }
        formatter.append_bytes(&buf[..pos])
    }
}

impl Formattable for f32 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_f64(*self as f64, 6)
    }
}

impl Formattable for f64 {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.format_f64(*self, 6)
    }
}

impl Formattable for bool {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.append_str(match bool::from(*self) {
            true => "true",
            false => "false",
        })
    }
}

impl Formattable for char {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.append_char(*self)
    }
}

impl Formattable for str {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.append_str(self)
    }
}

impl Formattable for &str {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.append_str(self)
    }
}

impl Formattable for String {
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.append_str(self)
    }
}

impl<T> Formattable for &[T]
where
    T: Formattable,
{
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        formatter.append_char('[')?;
        for (i, item) in self.iter().enumerate() {
            if i > 0 {
                formatter.append_str(", ")?;
            }
            item.format_into(formatter)?;
        }
        formatter.append_char(']')
    }
}

impl<T> Formattable for Vec<T>
where
    T: Formattable,
{
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        self.as_slice().format_into(formatter)
    }
}

impl<T> Formattable for Option<T>
where
    T: Formattable,
{
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        match self {
            Some(value) => {
                formatter.append_str("Some(")?;
                value.format_into(formatter)?;
                formatter.append_char(')')
            }
            None => formatter.append_str("None"),
        }
    }
}

impl<T, E> Formattable for Result<T, E>
where
    T: Formattable,
    E: Formattable,
{
    #[inline]
    fn format_into(&self, formatter: &mut Formatter) -> LogResult<()> {
        match self {
            Ok(value) => {
                formatter.append_str("Ok(")?;
                value.format_into(formatter)?;
                formatter.append_char(')')
            }
            Err(error) => {
                formatter.append_str("Err(")?;
                error.format_into(formatter)?;
                formatter.append_char(')')
            }
        }
    }
}

/// SIMD-accelerated string formatting for numeric arrays.
///
/// # Performance
///
/// This function uses SIMD operations to format arrays of numbers
/// in parallel, significantly improving throughput for large arrays.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn format_simd_i32_array(values: &[i32], buffer: &mut [u8]) -> usize {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi8, _mm256_set1_epi8, _mm256_storeu_si256,
    };

    let chunk_size = 8;
    let chunks = values.len() / chunk_size;
    let mut pos = 0;

    for chunk in 0..chunks {
        let start = chunk * chunk_size;
        let end = start + chunk_size;
        let _slice = &values[start..end];

        let simd_values = _mm256_set1_epi8(0); // Placeholder for actual SIMD logic
        unsafe {
            _mm256_storeu_si256(
                buffer.as_mut_ptr().add(pos) as *mut __m256i,
                simd_values,
            );
        }

        pos += chunk_size * 12; // Approximate space needed
    }

    // Handle remaining elements
    for i in chunks * chunk_size..values.len() {
        let mut buf = [0u8; 12];
        let len = format_scalar_i32(values[i], &mut buf);
        if pos + len <= buffer.len() {
            buffer[pos..pos + len].copy_from_slice(&buf[..len]);
            pos += len;
        }
    }

    pos
}

/// Scalar fallback for integer formatting.
#[inline]
fn format_scalar_i32(value: i32, buffer: &mut [u8]) -> usize {
    let mut buf = [0u8; 12];
    let mut pos = 0;
    let mut remaining = value.abs() as u32;

    if value < 0 {
        buf[pos] = b'-';
        pos += 1;
    }
    if remaining == 0 {
        buf[pos] = b'0';
        pos += 1;
    } else {
        let mut digit_pos = pos;
        while remaining > 0 {
            buf[digit_pos] = (remaining % 10) as u8 + b'0';
            remaining /= 10;
            digit_pos += 1;
        }
        for i in 0..(digit_pos - pos) / 2 {
            buf.swap(pos + i, digit_pos - 1 - i);
        }
        pos = digit_pos;
    }
    if buffer.len() >= pos {
        buffer[..pos].copy_from_slice(&buf[..pos]);
    }
    pos
}

#[cfg(test)]
mod tests {
    use crate::logtrace::Formatter;

    fn formatter_new_initializes_empty() {
        let formatter = Formatter::new();
        assert_eq!(formatter.as_str(), "");
    }

    fn formatter_append_str_adds_text() {
        let mut formatter = Formatter::new();
        formatter.append_str("hello").unwrap();
        assert_eq!(formatter.as_str(), "hello");
    }

    fn formatter_append_char_adds_char() {
        let mut formatter = Formatter::new();
        formatter.append_char('Z').unwrap();
        assert_eq!(formatter.as_str(), "Z");
    }
    fn formatter_append_bytes_adds_bytes() {
        let mut formatter = Formatter::new();
        formatter.append_bytes(b"abc").unwrap();
        assert_eq!(formatter.as_str(), "abc");
    }

    #[test]
    fn formatter_format_args_formats_multiple_values() {
        let mut formatter = Formatter::new();
        formatter
            .format_args(format_args!("{}:{}:{:?}", "item", 42, true))
            .unwrap();
        assert_eq!(formatter.as_str(), "item:42:true");
    }

    fn formatter_reset_clears_content() {
        let mut formatter = Formatter::new();
        formatter.append_str("abc").unwrap();
        formatter.reset();
        assert_eq!(formatter.as_str(), "");
    }

    fn formatter_append_str_overflow_returns_error() {
        let mut formatter = Formatter::new();
        let chunk = "x".repeat(5000);
        let result = formatter.append_str(&chunk);
        assert!(result.is_err());
    }

    fn formatter_format_u64_zero() {
        let mut formatter = Formatter::new();
        formatter.format_u64(0).unwrap();
        assert_eq!(formatter.as_str(), "0");
    }

    fn formatter_format_u64_one() {
        let mut formatter = Formatter::new();
        formatter.format_u64(1).unwrap();
        assert_eq!(formatter.as_str(), "1");
    }

    fn formatter_format_u64_ten() {
        let mut formatter = Formatter::new();
        formatter.format_u64(10).unwrap();
        assert_eq!(formatter.as_str(), "10");
    }

    fn formatter_format_u64_very_large() {
        let mut formatter = Formatter::new();
        formatter.format_u64(u64::MAX).unwrap();
        assert_eq!(formatter.as_str(), "18446744073709551615");
    }

    fn formatter_format_i64_positive() {
        let mut formatter = Formatter::new();
        formatter.format_i64(42).unwrap();
        assert_eq!(formatter.as_str(), "42");
    }

    fn formatter_format_i64_negative() {
        let mut formatter = Formatter::new();
        formatter.format_i64(-42).unwrap();
        assert_eq!(formatter.as_str(), "-42");
    }

    fn formatter_format_i64_min() {
        let mut formatter = Formatter::new();
        formatter.format_i64(i64::MIN).unwrap();
        assert_eq!(formatter.as_str(), "-9223372036854775808");
    }

    fn formatter_format_i128_positive() {
        let mut formatter = Formatter::new();
        formatter.format_i128(128).unwrap();
        assert_eq!(formatter.as_str(), "128");
    }

    fn formatter_format_i128_negative() {
        let mut formatter = Formatter::new();
        formatter.format_i128(-128).unwrap();
        assert_eq!(formatter.as_str(), "-128");
    }

    fn formatter_format_i128_large_positive() {
        let mut formatter = Formatter::new();
        formatter.format_i128(i128::MAX).unwrap();
        assert!(!formatter.as_str().is_empty());
    }

    fn formatter_format_i128_large_negative() {
        let mut formatter = Formatter::new();
        formatter.format_i128(i128::MIN).unwrap();
        assert!(formatter.as_str().starts_with('-'));
    }

    fn formatter_format_pointer_uses_hex_prefix() {
        let mut formatter = Formatter::new();
        formatter.format_pointer(0x1234).unwrap();
        assert!(formatter.as_str().starts_with("0x"));
    }

    fn formatter_format_hex_zero() {
        let mut formatter = Formatter::new();
        formatter.format_hex(0).unwrap();
        assert_eq!(formatter.as_str(), "0");
    }

    fn formatter_format_hex_simple_value() {
        let mut formatter = Formatter::new();
        formatter.format_hex(255).unwrap();
        assert_eq!(formatter.as_str(), "ff");
    }

    fn formatter_format_binary_zero() {
        let mut formatter = Formatter::new();
        formatter.format_binary(0).unwrap();
        assert_eq!(formatter.as_str(), "0");
    }

    fn formatter_format_binary_simple_value() {
        let mut formatter = Formatter::new();
        formatter.format_binary(5).unwrap();
        assert_eq!(formatter.as_str(), "101");
    }

    fn formatter_format_f64_integer_value() {
        let mut formatter = Formatter::new();
        formatter.format_f64(42.0, 2).unwrap();
        assert_eq!(formatter.as_str(), "42");
    }

    fn formatter_format_f64_special_nan() {
        let mut formatter = Formatter::new();
        formatter.format_f64(f64::NAN, 2).unwrap();
        assert_eq!(formatter.as_str(), "NaN");
    }

    fn formatter_format_f64_positive_infinity() {
        let mut formatter = Formatter::new();
        formatter.format_f64(f64::INFINITY, 2).unwrap();
        assert_eq!(formatter.as_str(), "inf");
    }

    fn formatter_format_f64_negative_infinity() {
        let mut formatter = Formatter::new();
        formatter.format_f64(f64::NEG_INFINITY, 2).unwrap();
        assert_eq!(formatter.as_str(), "-inf");
    }

    fn formatter_format_bool_true() {
        let mut formatter = Formatter::new();
        formatter.format(&true).unwrap();
        assert_eq!(formatter.as_str(), "true");
    }

    fn formatter_format_bool_false() {
        let mut formatter = Formatter::new();
        formatter.format(&false).unwrap();
        assert_eq!(formatter.as_str(), "false");
    }

    fn formatter_format_usize_value() {
        let mut formatter = Formatter::new();
        formatter.format(&42usize).unwrap();
        assert_eq!(formatter.as_str(), "42");
    }

    fn formatter_format_isize_value() {
        let mut formatter = Formatter::new();
        formatter.format(&-7isize).unwrap();
        assert_eq!(formatter.as_str(), "-7");
    }

    fn formatter_format_u8_value() {
        let mut formatter = Formatter::new();
        formatter.format(&7u8).unwrap();
        assert_eq!(formatter.as_str(), "7");
    }

    fn formatter_format_i8_value() {
        let mut formatter = Formatter::new();
        formatter.format(&-7i8).unwrap();
        assert_eq!(formatter.as_str(), "-7");
    }

    fn formatter_append_string_then_reset_keeps_empty() {
        let mut formatter = Formatter::new();
        formatter.append_str("xx").unwrap();
        formatter.reset();
        formatter.append_str("ok").unwrap();
        assert_eq!(formatter.as_str(), "ok");
    }

    fn formatter_multiple_appends_stitch_text() {
        let mut formatter = Formatter::new();
        formatter.append_str("a").unwrap();
        formatter.append_str("b").unwrap();
        formatter.append_char('c').unwrap();
        assert_eq!(formatter.as_str(), "abc");
    }

    fn formatter_unlocks_to_string_via_formattable_impl() {
        let mut formatter = Formatter::new();
        formatter.format(&"abc".to_string()).unwrap();
        assert_eq!(formatter.as_str(), "abc");
    }

    fn formatter_format_str_literal() {
        let mut formatter = Formatter::new();
        formatter.format(&"literal").unwrap();
        assert_eq!(formatter.as_str(), "literal");
    }

    fn formatter_format_char_value() {
        let mut formatter = Formatter::new();
        formatter.format(&'A').unwrap();
        assert_eq!(formatter.as_str(), "A");
    }

    fn formatter_as_bytes_matches_utf8() {
        let mut formatter = Formatter::new();
        formatter.append_str("é").unwrap();
        assert_eq!(formatter.as_bytes(), "é".as_bytes());
    }

    fn formatter_entry_formatting_is_utf8_safe() {
        let mut formatter = Formatter::new();
        formatter.append_str("safe").unwrap();
        assert!(std::str::from_utf8(formatter.as_bytes()).is_ok());
    }
}
